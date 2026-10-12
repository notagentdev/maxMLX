"""hc_kernel against the stock Qwen4ExpGatedResidual on real decode inputs:
max/relative error per output, then 97 chained calls timed both ways.

    ~/.venvs/qwen4-spike/bin/python test_hc_kernel.py <model_dir>
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from niwaki_stores import load_niwaki_with_stores  # noqa: E402

import mlx.core as mx  # noqa: E402

import hc_kernel  # noqa: E402

model, processor = load_niwaki_with_stores(os.path.abspath(sys.argv[1]))
lm = model.language_model
tok = getattr(processor, "tokenizer", processor)
layers = lm.model.layers
mods = [m for layer in layers for m in (layer.attn_hyper_connection, layer.mlp_hyper_connection)]
mods.append(lm.model.hyper_connection_mixer)
print("supported:", sum(hc_kernel.supported(m) for m in mods), "of", len(mods), flush=True)

# Real decode inputs: capture what each module sees on one decode token.
cls = type(mods[0])
orig = cls.__call__
seen = {}


def capture(self, h):
    if h.shape[:2] == (1, 1):
        seen.setdefault(id(self), mx.array(h))
    return orig(self, h)


cls.__call__ = capture
cache = lm.make_cache()
out = lm(mx.array([tok.encode("The quick brown fox jumps over the lazy dog because")]), cache=cache)
y = mx.argmax(out.logits[:, -1, :], axis=-1)
out = lm(y[:, None], cache=cache)
mx.eval(out.logits, *seen.values())
cls.__call__ = orig
print("captured decode inputs for", len(seen), "modules", flush=True)


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.max(mx.abs(a - b))), float(mx.linalg.norm(a - b) / mx.maximum(mx.linalg.norm(b), 1e-12))


def truth(m, h):
    """The module in fp32 throughout (dequantized weights): the yardstick both
    the stock bf16 path and the kernels are measured against."""
    x = h.astype(mx.float32).reshape(4, 2560)
    y = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + m.hc_norm.eps)
    normed = (y * (1.0 + m.hc_norm.weight.astype(mx.float32).reshape(4, 2560))).reshape(-1)

    def lin(l, v):
        w = mx.dequantize(l.weight, l.scales, l.biases, group_size=l.group_size, bits=l.bits).astype(mx.float32)
        return w @ v

    act = mx.sigmoid(lin(m.input_mix_weight_down, normed) / 4) * (lin(m.input_mix_weight_down, normed) / 4)
    mix = mx.sigmoid(lin(m.input_mix_weight_up, act)).reshape(4, 2560)
    mixed = mx.mean(mix * normed.reshape(4, 2560), axis=0)
    return mixed


errs = {"stock": 0.0, "kernel": 0.0}
for m in mods:
    h = seen[id(m)]
    t = truth(m, h)
    s_ = orig(m, h)
    k_ = hc_kernel.hc_fused(m, h)
    s_ = s_[0] if isinstance(s_, tuple) else s_
    k_ = k_[0] if isinstance(k_, tuple) else k_
    mx.eval(t, s_, k_)
    errs["stock"] = max(errs["stock"], rel(s_.reshape(-1), t)[1])
    errs["kernel"] = max(errs["kernel"], rel(k_.reshape(-1), t)[1])
print("worst relative error of `mixed` against an fp32 reference:", errs, flush=True)

worst = {"mixed": 0.0, "inject": 0.0}
for i, m in enumerate(mods):
    h = seen[id(m)]
    ref = orig(m, h)
    got = hc_kernel.hc_fused(m, h)
    if isinstance(ref, tuple):
        mx.eval(ref[0], ref[2], got[0], got[2])
        e_mix, e_inj = rel(got[0], ref[0]), rel(got[2], ref[2])
        worst["mixed"] = max(worst["mixed"], e_mix[1])
        worst["inject"] = max(worst["inject"], e_inj[1])
        if i in (0, 1, 40, 95):
            print(f"  module {i:2d}: mixed max abs {e_mix[0]:.2e} rel {e_mix[1]:.2e} | inject max abs {e_inj[0]:.2e} "
                  f"rel {e_inj[1]:.2e} | hyper passthrough identical {bool(mx.all(got[1] == ref[1]))}")
    else:
        mx.eval(ref, got)
        e = rel(got, ref)
        worst["mixed"] = max(worst["mixed"], e[1])
        print(f"  final mixer: max abs {e[0]:.2e} rel {e[1]:.2e}")
print("worst relative error:", worst, flush=True)


def chain(fn, reps=20):
    hs = [seen[id(m)] for m in mods]
    times = []
    for r in range(reps + 3):
        t = time.perf_counter()
        outs = []
        for m, h in zip(mods, hs):
            o = fn(m, h)
            outs.append(o[0] if isinstance(o, tuple) else o)
        mx.eval(*outs)
        if r >= 3:
            times.append(time.perf_counter() - t)
    times.sort()
    return times[len(times) // 2] * 1e3


t_stock = chain(orig)
t_kernel = chain(hc_kernel.hc_fused)
print(f"97 independent calls: stock {t_stock:.2f} ms, kernels {t_kernel:.2f} ms", flush=True)
