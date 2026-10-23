"""Grammar-constrained decoding for ``response_format`` (JSON object / JSON
schema) via xgrammar.

Every decode step, the request's ``GrammarMatcher`` fills a bitmask of the
tokens its grammar allows next, and xgrammar's MLX kernel sets every other
logit to -inf before sampling, so the answer is valid JSON (matching the
schema) by construction instead of by prompting. The approach and the pinned
xgrammar version follow oMLX (``omlx/api/grammar.py``, Apache-2.0).

The processor has mlx-lm's logits-processor shape, ``(tokens, logits) ->
logits``, so both decode paths call it the same way: ``generate_step`` (one
request) and ``BatchGenerator`` (continuous batching), which pass the token
context of the row (the last prompt token, then every generated token).

Thinking models: with ``start_after`` set (the reasoning block's closing
marker), the processor stays out of the way while the model thinks and
constrains from the first token after the marker.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional

import numpy as np

from maxmlx.utils import init_logger

logger = init_logger(__name__)

# Longest whitespace run allowed between JSON tokens (newline + indentation).
MAX_WHITESPACE = 16


def available() -> bool:
    try:
        import xgrammar  # noqa: F401
    except Exception:  # noqa: BLE001 -- missing or broken install
        return False
    return True


class GrammarCompiler:
    """One xgrammar compiler per model (building the tokenizer info walks the
    whole vocabulary once); compiled grammars are cached by their spec."""

    def __init__(self, hf_tokenizer: Any, vocab_size: int, stop_token_ids):
        import xgrammar as xgr

        self.vocab_size = int(vocab_size)
        self._info = xgr.TokenizerInfo.from_huggingface(
            hf_tokenizer, vocab_size=self.vocab_size, stop_token_ids=sorted(int(t) for t in stop_token_ids)
        )
        self._compiler = xgr.GrammarCompiler(self._info)
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._tokenizer = hf_tokenizer

    def compile(self, grammar: Dict[str, Any]):
        key = json.dumps(grammar, sort_keys=True)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        kind = grammar.get("type")
        # Whitespace stays allowed (models indent their JSON) but bounded: with
        # unbounded whitespace a weak or confused model can emit spaces until
        # max_tokens runs out (reproduced with random logits: 300 tokens of
        # whitespace after the opening brace).
        if kind == "json_object":
            compiled = self._compiler.compile_json_schema(
                '{"type": "object"}', any_whitespace=True,
                max_whitespace_cnt=MAX_WHITESPACE, strict_mode=False,
            )
        elif kind == "json_schema":
            schema = grammar.get("schema")
            if not isinstance(schema, (dict, str)):
                raise ValueError("json_schema needs a 'schema' object")
            compiled = self._compiler.compile_json_schema(
                schema if isinstance(schema, str) else json.dumps(schema),
                any_whitespace=True, max_whitespace_cnt=MAX_WHITESPACE,
                strict_mode=bool(grammar.get("strict", True)),
            )
        else:
            raise ValueError(f"unsupported grammar type {kind!r}")
        with self._lock:
            self._cache[key] = compiled
        return compiled

    def token_id(self, text: str) -> Optional[int]:
        """The single token id of a marker like ``</think>``, else None."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return int(ids[0]) if len(ids) == 1 else None

    def processor(self, grammar: Dict[str, Any]) -> "GrammarLogitsProcessor":
        start_after = None
        marker = grammar.get("start_after")
        if marker:
            start_after = self.token_id(marker)
            if start_after is None:
                logger.warning(f"grammar: {marker!r} is not one token; constraining from the start")
        return GrammarLogitsProcessor(self.compile(grammar), self.vocab_size, start_after)


class GrammarLogitsProcessor:
    def __init__(self, compiled: Any, vocab_size: int, start_after: Optional[int] = None):
        import xgrammar as xgr
        from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

        self._matcher = xgr.GrammarMatcher(compiled)
        self._apply = apply_token_bitmask_mlx
        self._vocab_size = int(vocab_size)
        self._bitmask = np.full((1, (self._vocab_size + 31) // 32), -1, dtype=np.int32)
        self._start_after = start_after
        self.active = start_after is None
        self.terminated = False
        self._base: Optional[int] = None  # context length at the first call (prompt side)
        self._seen = 0  # generated tokens already fed to the matcher

    def _advance(self, tokens) -> None:
        n = int(tokens.shape[-1])
        if self._base is None:
            self._base = n
            return
        new = n - self._base - self._seen
        if new <= 0:
            return
        tail = np.array(tokens.reshape(-1)[n - new:]).tolist()
        self._seen += new
        for t in tail:
            if self.terminated:
                return
            if not self.active:
                if t == self._start_after:
                    self.active = True
                continue
            if not self._matcher.accept_token(int(t)):
                logger.warning(f"grammar: matcher rejected token {t}")
            if self._matcher.is_terminated():
                self.terminated = True

    def __call__(self, tokens, logits):
        import mlx.core as mx

        self._advance(tokens)
        if not self.active or self.terminated:
            return logits
        self._bitmask.fill(-1)
        self._matcher.fill_next_token_bitmask(self._bitmask)
        return self._apply(mx.array(self._bitmask), logits, self._vocab_size)


_BATCH_PATCHED = False


def patch_batch_generation() -> None:
    """mlx-lm's batch merge turns the processor list of a row that has none into
    ``None`` once another row in the batch has processors, and the decode step
    then iterates over it (TypeError). A grammar row next to free-text rows
    reaches exactly that, so the step treats ``None`` as no processors."""
    global _BATCH_PATCHED
    if _BATCH_PATCHED:
        return
    from mlx_lm.generate import GenerationBatch

    original = GenerationBatch._step

    def _step(self):
        procs = self.logits_processors
        if procs and any(procs) and any(p is None for p in procs):
            self.logits_processors = [p or [] for p in procs]
        return original(self)

    GenerationBatch._step = _step
    _BATCH_PATCHED = True
