"""GPU time per component of one Qwen3.8-Flash-Next (Niwaki) decode token.

Every component's inputs are evaluated before it runs and its output after, so
each call is timed alone. The per-call sync adds ~0.1-0.3 ms, so the total is
above the real token time; the split between components is what this is for.

    ~/.venvs/qwen4-spike/bin/python profile_breakdown.py <model_dir> [tokens]
"""
import collections
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from bench_mlx_decode import Watchdog  # noqa: E402
from niwaki_stores import load_niwaki_with_stores  # noqa: E402

model_dir = os.path.abspath(sys.argv[1])
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 16

wd = Watchdog(min_free_gb=0.3, max_wired_gb=26.0)
wd.start()

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

model, processor = load_niwaki_with_stores(model_dir)
lm = model.language_model
tok = getattr(processor, "tokenizer", processor)
ids = tok.encode(tok.apply_chat_template(
    [{"role": "user", "content": "Explain in five sentences why the sky is blue."}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False))

times = collections.defaultdict(float)
calls = collections.Counter()
active = [False]


def arrays_in(obj):
    if isinstance(obj, mx.array):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from arrays_in(o)


def wrap(cls, label):
    orig = cls.__call__
    name = label if callable(label) else (lambda _self, _l=label: _l)

    def timed(self, *args, **kwargs):
        if not active[0]:
            return orig(self, *args, **kwargs)
        mx.eval(*arrays_in(args), *arrays_in(list(kwargs.values())))
        active[0] = False  # nested components count toward the outer one
        t = time.perf_counter()
        out = orig(self, *args, **kwargs)
        mx.eval(*arrays_in(out))
        key = name(self)
        times[key] += time.perf_counter() - t
        calls[key] += 1
        active[0] = True
        return out

    cls.__call__ = timed


layers = lm.model.layers
seen = {}
for layer in layers:
    parts = {
        "hyper-connection": layer.attn_hyper_connection,
        "gated-delta-net": getattr(layer, "linear_attn", None),
        "qsa-attention": getattr(layer, "self_attn", None),
        "ple": getattr(layer, "ple", None) if "ple" in layer else None,
    }
    for label, mod in parts.items():
        if mod is not None and type(mod) not in seen:
            seen[type(mod)] = label
def moe_label(mlp):
    sw = getattr(mlp, "switch_mlp", None)
    routed = sw is not None and ("gate_proj" in sw or "gate_up_proj" in sw or hasattr(sw, "gate_proj"))
    return "moe: routed + shared" if routed else "moe: shared only (Niwaki)"


seen.setdefault(type(layers[0].mlp), moe_label)
for cls, label in seen.items():
    wrap(cls, label)
head = getattr(lm, "lm_head", None)
if head is not None:
    wrap(type(head), "lm_head (and other " + type(head).__name__ + ")")
wrap(type(lm.model.hyper_connection_mixer), "hyper-connection")

shared_only = sum(1 for layer in layers if getattr(layer.mlp, "switch_mlp", None) is None)
print(f"layers {len(layers)}, MoE classes: {[type(l.mlp).__name__ for l in layers[:1]]}, "
      f"layers without routed experts: {shared_only}", flush=True)

cache = lm.make_cache()
out = lm(mx.array([ids]), cache=cache)
y = mx.argmax(getattr(out, "logits", out)[:, -1, :], axis=-1)
mx.eval(y)
for warm in range(4):
    out = lm(y[:, None], cache=cache)
    y = mx.argmax(getattr(out, "logits", out)[:, -1, :], axis=-1)
    mx.eval(y)

active[0] = True
t0 = time.perf_counter()
for _ in range(n_tok):
    out = lm(y[:, None], cache=cache)
    y = mx.argmax(getattr(out, "logits", out)[:, -1, :], axis=-1)
    mx.eval(y)
total = (time.perf_counter() - t0) / n_tok * 1e3
active[0] = False

timed_sum = sum(times.values()) / n_tok * 1e3
print(f"\nper token: {total:.1f} ms with per-component syncs ({timed_sum:.1f} ms inside components)")
for label, t in sorted(times.items(), key=lambda kv: -kv[1]):
    print(f"  {label:40s} {t / n_tok * 1e3:7.2f} ms  ({calls[label] // n_tok} calls/token, "
          f"{t / calls[label] * 1e3:.3f} ms each)")
print(f"wired peak {wd.peak_wired:.1f} GB")
