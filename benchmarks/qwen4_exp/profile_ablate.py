"""GPU time of each component of a Qwen3.8-Flash-Next (Niwaki) decode token, by
ablation: the token is timed with one component replaced by zeros, and the
difference to the full token is that component's cost. The text is garbage in
the ablated runs; only the timing matters. No per-component syncs, so unlike
profile_breakdown.py the numbers carry no sync overhead.

    ~/.venvs/qwen4-spike/bin/python profile_ablate.py <model_dir> [tokens]
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from bench_mlx_decode import Watchdog  # noqa: E402
from niwaki_stores import load_niwaki_with_stores  # noqa: E402

model_dir = os.path.abspath(sys.argv[1])
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 24

wd = Watchdog(min_free_gb=0.3, max_wired_gb=26.0)
wd.start()

import mlx.core as mx  # noqa: E402

model, processor = load_niwaki_with_stores(model_dir)
lm = model.language_model
tok = getattr(processor, "tokenizer", processor)
ids = tok.encode(tok.apply_chat_template(
    [{"role": "user", "content": "Explain in five sentences why the sky is blue."}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False))

layers = list(lm.model.layers)
layer_cls = type(layers[0])
skip = set()


def call(self, hidden_states, input_ids, mask, cache, position_ids):
    if "ple" in self and "ple" not in skip:
        hidden_states = hidden_states + self.ple(hidden_states, input_ids, cache, mask)
    if "hc" in skip:
        mixed, hyper, w = hidden_states[..., :2560], hidden_states, mx.ones((1, 1, 4))
    else:
        mixed, hyper, w = self.attn_hyper_connection(hidden_states)
    if "mixer" in skip:
        branch = mx.zeros_like(mixed)
    elif self.is_linear:
        branch = self.linear_attn(mixed, mask=mask, cache=cache)
    else:
        branch = self.self_attn(mixed, mask=mask, cache=cache, position_ids=position_ids)
    h = hyper + (branch[..., None, :] * w[..., None]).reshape(*hyper.shape)
    if "hc" in skip:
        mixed, hyper2, w2 = h[..., :2560], h, mx.ones((1, 1, 4))
    else:
        mixed, hyper2, w2 = self.mlp_hyper_connection(h)
    b2 = mx.zeros_like(mixed) if "moe" in skip else self.mlp(mixed)
    return hyper2 + (b2[..., None, :] * w2[..., None]).reshape(*hyper2.shape)


layer_cls.__call__ = call


def gpu_ms(n):
    cache = lm.make_cache()
    out = lm(mx.array([ids]), cache=cache)
    y = mx.argmax(getattr(out, "logits", out)[:, -1, :], axis=-1)
    mx.eval(y)
    total = 0.0
    for i in range(n + 4):
        out = lm(y[:, None], cache=cache)
        y = mx.argmax(getattr(out, "logits", out)[:, -1, :], axis=-1)
        t = time.perf_counter()
        mx.eval(y)
        if i >= 4:
            total += time.perf_counter() - t
    return total / n * 1e3


configs = [(), ("moe",), ("mixer",), ("hc",), ("ple",), ("moe", "mixer", "hc", "ple")]
base = None
for rep in range(2):
    print(f"-- pass {rep}", flush=True)
    for cfg in configs:
        skip.clear()
        skip.update(cfg)
        ms = gpu_ms(n_tok)
        if not cfg:
            base = ms
            print(f"  full token                 {ms:6.1f} ms GPU", flush=True)
        else:
            print(f"  without {'+'.join(cfg):18s} {ms:6.1f} ms GPU  -> its cost {base - ms:5.1f} ms", flush=True)
print(f"wired peak {wd.peak_wired:.1f} GB")
