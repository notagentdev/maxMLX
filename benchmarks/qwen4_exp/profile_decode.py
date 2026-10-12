"""Where a decode token of Qwen3.8-Flash-Next (Niwaki) goes: CPU graph build vs GPU.

Loads through niwaki_stores (PLE on disk, experts on the mapped store), prefills
a short prompt, then decodes greedily two ways:

  sync      eval every token before building the next: per token, "build" is
            the Python forward (graph construction plus the PLE host gather)
            and "gpu" is the eval that follows. Their sum is the token time
            of a loop that cannot overlap.
  pipelined mlx-lm's generate_step pattern: build token n+1 and async_eval it
            before reading token n. The PLE's host sync inside layer 2 limits
            how much of the build can overlap the GPU.

    ~/.venvs/qwen4-spike/bin/python profile_decode.py <model_dir> [tokens]
    MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER as for the spike.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from bench_mlx_decode import Watchdog, vm_gb  # noqa: E402
from niwaki_stores import load_niwaki_with_stores  # noqa: E402

model_dir = os.path.abspath(sys.argv[1])
n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 64

wd = Watchdog(min_free_gb=float(os.environ.get("MIN_FREE_GB", "0.3")),
              max_wired_gb=float(os.environ.get("MAX_WIRED_GB", "26")))
wd.start()

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

print(f"command buffers: ops={os.environ.get('MLX_MAX_OPS_PER_BUFFER', 'default')} "
      f"mb={os.environ.get('MLX_MAX_MB_PER_BUFFER', 'default')}", flush=True)
model, processor = load_niwaki_with_stores(model_dir)
lm = model.language_model
if os.environ.get("MAPS8") == "1" or os.environ.get("GDNFUSE") == "1":
    import niwaki_fusion

    if os.environ.get("MAPS8") == "1":
        print("maps:", niwaki_fusion.quantize_maps(model, bits=int(os.environ.get("MAPS_BITS", "8"))), flush=True)
    if os.environ.get("GDNFUSE") == "1":
        print("gdn projections fused:", niwaki_fusion.fuse_gdn_projections(model), flush=True)
if os.environ.get("HCK") == "1":
    import hc_kernel

    print("hyper-connection kernels:", hc_kernel.install(model), flush=True)
if os.environ.get("FUSION") == "1":
    import niwaki_fusion

    print("fusion:", niwaki_fusion.install(model), flush=True)
    if os.environ.get("GDNFULL") == "1":
        print("gdn layers compiled whole:", niwaki_fusion.compile_gdn_layers(model), flush=True)
tok = getattr(processor, "tokenizer", processor)
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain in five sentences why the sky is blue."}],
    add_generation_prompt=True, tokenize=False, enable_thinking=False)
ids = tok.encode(prompt)

ple_time = [0.0]
_ple_cls = type(next(layer.ple.ple_embedding.ngram_embedding
                     for layer in lm.model.layers if "ple" in layer))
_ple_call = _ple_cls.__call__


def _timed_ple(self, indices):
    t = time.perf_counter()
    r = _ple_call(self, indices)
    ple_time[0] += time.perf_counter() - t
    return r


_ple_cls.__call__ = _timed_ple


def logits_of(out):
    return getattr(out, "logits", out)


def run(mode: str):
    cache = lm.make_cache()
    t0 = time.perf_counter()
    logits = logits_of(lm(mx.array([ids]), cache=cache))
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    ttft = time.perf_counter() - t0
    build = gpu = 0.0
    cpu_in_eval = [0.0]
    ple_time[0] = 0.0
    out = [int(y.item())]
    t_dec = time.perf_counter()
    if mode == "sync":
        for _ in range(n_tok):
            a = time.perf_counter()
            logits = logits_of(lm(y[:, None], cache=cache))
            y = mx.argmax(logits[:, -1, :], axis=-1)
            b = time.perf_counter()
            pb = time.process_time()
            mx.eval(y)
            c = time.perf_counter()
            cpu_in_eval[0] += time.process_time() - pb
            build += b - a
            gpu += c - b
            out.append(int(y.item()))
    else:
        out = []
        for i in range(n_tok):
            logits = logits_of(lm(y[:, None], cache=cache))
            nxt = mx.argmax(logits[:, -1, :], axis=-1)
            mx.async_eval(nxt)
            out.append(int(y.item()))
            y = nxt
        mx.eval(y)
        out.append(int(y.item()))
    dt = time.perf_counter() - t_dec
    rate = n_tok / dt
    msg = (f"{mode:9s}: prefill {len(ids)} tok {ttft*1e3:.0f} ms; decode {n_tok} tok {dt:.2f}s = "
           f"{rate:.1f} tok/s ({dt/n_tok*1e3:.1f} ms/tok)")
    if mode == "sync":
        msg += (f"; per token build {build/n_tok*1e3:.1f} ms (PLE gather {ple_time[0]/n_tok*1e3:.1f}) "
                f"+ gpu {gpu/n_tok*1e3:.1f} ms (process CPU during it {cpu_in_eval[0]/n_tok*1e3:.1f} ms)")
    else:
        msg += f"; PLE gather {ple_time[0]/n_tok*1e3:.1f} ms/tok"
    print(msg, f"| wired peak {wd.peak_wired:.1f} GB", flush=True)
    import hashlib
    digest = hashlib.sha1(str(out).encode()).hexdigest()[:10]
    print(f"   tokens {len(out)} sha1 {digest}:", tok.decode(out)[:120].replace("\n", " "), flush=True)


for mode in ("sync", "pipelined", "sync", "pipelined"):
    run(mode)
print(f"wired {vm_gb()[0]:.1f} GB at the end", flush=True)
