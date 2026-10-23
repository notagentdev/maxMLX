"""Structured output: response_format becomes a grammar, and the grammar's
logits processor only lets valid JSON through.

The processor is driven by random logits, the worst case for a constraint: if
JSON matching the schema comes out of noise, it comes out of a model too.
Needs xgrammar and a Qwen tokenizer from the local Hugging Face cache (skipped
otherwise).
"""

import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("xgrammar")

from maxmlx.mlx_backend import grammar as G  # noqa: E402
from maxmlx.server.generation import GenSpec  # noqa: E402
from maxmlx.server.openai_api import _response_format_grammar, _with_thinking_phase  # noqa: E402
from maxmlx.core import SamplingParams  # noqa: E402

_SNAPSHOTS = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-MLX-4bit/snapshots/*/tokenizer.json"))
VOCAB, EOS = 248320, [248046, 248044]


@pytest.fixture(scope="module")
def compiler():
    if not _SNAPSHOTS:
        pytest.skip("no Qwen tokenizer in the local Hugging Face cache")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.dirname(_SNAPSHOTS[0]))
    return G.GrammarCompiler(tok, VOCAB, EOS), tok


def _decode_noise(proc, tok, seed=0, steps=300, prefix=()):
    rng = np.random.default_rng(seed)
    ctx = [0]
    proc(mx.array(ctx), mx.zeros((1, VOCAB)))
    for t in prefix:
        ctx.append(t)
        proc(mx.array(ctx), mx.zeros((1, VOCAB)))
    out = []
    for _ in range(steps):
        masked = proc(mx.array(ctx), mx.array(rng.normal(size=(1, VOCAB)).astype(np.float32)))
        t = int(mx.argmax(masked[0]).item())
        ctx.append(t)
        out.append(t)
        if t in EOS:
            return tok.decode(out[:-1])
    raise AssertionError(f"no end of sequence after {steps} tokens: {tok.decode(out)[:200]!r}")


SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}, "kind": {"enum": ["a", "b"]}},
    "required": ["answer", "kind"],
    "additionalProperties": False,
}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noise_under_a_schema_is_valid_json_matching_it(compiler, seed):
    comp, tok = compiler
    text = _decode_noise(comp.processor({"type": "json_schema", "schema": SCHEMA}), tok, seed)
    obj = json.loads(text)
    assert set(obj) == {"answer", "kind"} and isinstance(obj["answer"], int) and obj["kind"] in ("a", "b")


def test_json_object_gives_an_object(compiler):
    comp, tok = compiler
    assert isinstance(json.loads(_decode_noise(comp.processor({"type": "json_object"}), tok, seed=3)), dict)


def test_thinking_is_free_until_the_marker(compiler):
    comp, tok = compiler
    proc = comp.processor({"type": "json_schema", "schema": SCHEMA, "start_after": "</think>"})
    proc(mx.array([0]), mx.zeros((1, VOCAB)))
    logits = mx.array(np.random.default_rng(4).normal(size=(1, VOCAB)).astype(np.float32))
    assert mx.array_equal(proc(mx.array([0, 11]), logits), logits)  # thinking: untouched
    end = comp.token_id("</think>")
    text = _decode_noise(
        comp.processor({"type": "json_schema", "schema": SCHEMA, "start_after": "</think>"}),
        tok, seed=5, prefix=tok.encode("let me think", add_special_tokens=False) + [end],
    )
    assert json.loads(text)["kind"] in ("a", "b")


def test_response_format_mapping():
    assert _response_format_grammar(None) == (None, None)
    assert _response_format_grammar({"type": "text"}) == (None, None)
    assert _response_format_grammar({"type": "json_object"}) == ({"type": "json_object"}, None)
    g, err = _response_format_grammar({"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}})
    assert err is None and g == {"type": "json_schema", "schema": SCHEMA, "strict": True}
    assert _response_format_grammar({"type": "json_schema", "json_schema": {}})[1]
    assert _response_format_grammar({"type": "regex"})[1]


def test_a_thinking_model_is_constrained_after_its_reasoning():
    state = SimpleNamespace(config=SimpleNamespace(reasoning_parser="qwen3", thinking=None))
    spec = GenSpec(messages=[{"role": "user", "content": "x"}], sampling_params=SamplingParams())
    assert _with_thinking_phase({"type": "json_object"}, spec, state)["start_after"] == "</think>"
    spec_off = GenSpec(messages=[{"role": "user", "content": "x"}], sampling_params=SamplingParams(),
                       chat_template_kwargs={"enable_thinking": False})
    assert "start_after" not in _with_thinking_phase({"type": "json_object"}, spec_off, state)
