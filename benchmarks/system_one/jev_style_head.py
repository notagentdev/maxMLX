"""A drop-in System One head backed by chaoliangUNSW's Jev-Style decision model.

The point of this adapter is a like-for-like comparison: the same benchmark
runners, the same scoring, the same option rendering — only the head changes.
Two deliberate asymmetries, both in the other model's favour:

*   The prompt is **theirs** verbatim (header first, options last, an "Answer:"
    seed). Our own head must not use that shape — the format instruction before
    the options costs an untrained model most of its answer mass, and the seed
    turns the next token into a digit — but this model was fine-tuned on it, so
    imposing our shape would measure a format mismatch rather than the model.
*   The readout is theirs too: the hidden state is projected onto the letter
    rows of the tied embedding matrix only. Verified bit-identical to slicing a
    full-vocabulary projection (delta 0.000e+00), so it costs no accuracy.

Their calibration temperature is already folded into the final RMSNorm weight,
so the softmax below is the calibrated distribution with no post-processing.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import mlx.core as mx
from mlx_lm import load

REPO = "chaoliangUNSW/Jev-Style-Qwen3.5-2B-Decision-MLX-bf16"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER = ("You are a decision function. Read the state, then answer the question "
          "by choosing exactly one option.\n\n")


@dataclass
class Answer:
    value: Any
    probability: float
    distribution: Dict[str, float]
    entropy: float
    expectation: float | None = None


@dataclass
class Decision:
    answers: Dict[str, Answer]
    prefill_tokens: int
    prefill_ms: float
    decide_ms: float

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    @property
    def values(self) -> Dict[str, Any]:
        return {k: a.value for k, a in self.answers.items()}


class JevStyleHead:
    """Same surface as `maxtoken.system_one.SystemOne`, different weights."""

    def __init__(self, model, tok):
        self.model, self.tok = model, tok
        self.inner = model.language_model.model
        self.label_ids = mx.array(
            [tok.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS]
        )

    @classmethod
    def load(cls, path: str = REPO, **_) -> "JevStyleHead":
        model, tok = load(path)
        model.eval()
        return cls(model, tok)

    def _spec_options(self, spec: Dict[str, Any]):
        """(descriptions, values) for the three question types we are asked."""
        kind = spec.get("type", "choice")
        if kind == "choice":
            values = list(spec["options"])
            return [str(v) for v in values], values
        if kind == "bool":
            labels = list(spec.get("labels", ("yes", "no")))
            return [str(l) for l in labels], [True, False]
        if kind == "score":
            lo, hi = int(spec["min"]), int(spec["max"])
            step = int(spec.get("step", 1))
            values = list(range(lo, hi + 1, step))
            return [str(v) for v in values], values
        raise ValueError(f"unknown question type {kind!r}")

    def _one(self, state: str, prompt: str, descriptions: List[str]):
        if not 2 <= len(descriptions) <= len(LETTERS):
            raise ValueError(f"{len(descriptions)} options outside 2..{len(LETTERS)}")
        lines = "\n".join(f"{LETTERS[i]}. {d}" for i, d in enumerate(descriptions))
        text = (f"{HEADER}[State]\n{state}\n\n[Question]\n{prompt}\n\n"
                f"[Options]\n{lines}\n\nAnswer:")
        ids = mx.array([self.tok.encode(text, add_special_tokens=False)])
        last = self.inner(ids)[0, -1]
        rows = self.inner.embed_tokens.weight[self.label_ids[: len(descriptions)]]
        probs = mx.softmax((rows @ last).astype(mx.float32))
        mx.eval(probs)
        return probs.tolist(), ids.shape[1]

    def decide(self, state: str, questions: Dict[str, Dict[str, Any]]) -> Decision:
        t0 = time.perf_counter()
        answers: Dict[str, Answer] = {}
        tokens = 0
        for name, spec in questions.items():
            descriptions, values = self._spec_options(spec)
            probs, ntok = self._one(state, spec.get("prompt", name), descriptions)
            tokens += ntok
            dist = {d: p for d, p in zip(descriptions, probs)}
            best = max(range(len(probs)), key=probs.__getitem__)
            entropy = -sum(p * (p and __import__("math").log(p)) for p in probs)
            expectation = None
            if spec.get("type") == "score":
                expectation = sum(v * p for v, p in zip(values, probs))
            answers[name] = Answer(values[best], probs[best], dist, entropy, expectation)
        dt = (time.perf_counter() - t0) * 1e3
        # No shared prefill: every question is its own pass, because their
        # prompt interleaves the question with the state rather than appending
        # it. That is a real cost of their format and is reported as such.
        return Decision(answers, tokens, 0.0, dt)
