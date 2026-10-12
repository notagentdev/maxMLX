"""Qwen3.8-Flash-Next (Niwaki) with and without its native MTP drafter.

Target through niwaki_stores (PLE on disk, mapped experts) plus the
niwaki_fusion levers that measured faster (8-bit healing maps, fused GDN
projections, per-layer compile, whole GDN layers compiled). The drafter is the
mlx-vlm ``qwen4_exp_mtp`` model built by build_mtp_drafter.py. Same prompt,
greedy (thinking off), mlx-vlm's own speculative loop.

    ~/.venvs/qwen4-spike/bin/python profile_mtp.py <target_dir> <drafter_dir> [tokens] [block]
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)
from bench_mlx_decode import Watchdog  # noqa: E402
from niwaki_stores import load_niwaki_with_stores  # noqa: E402

target_dir, drafter_dir = (os.path.abspath(a) for a in sys.argv[1:3])
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256
block = int(sys.argv[4]) if len(sys.argv) > 4 else 3

wd = Watchdog(min_free_gb=0.3, max_wired_gb=float(os.environ.get("MAX_WIRED_GB", "26")))
wd.start()

import mlx.core as mx  # noqa: E402
from mlx_vlm import stream_generate  # noqa: E402
from mlx_vlm.speculative.drafters import load_drafter  # noqa: E402

import niwaki_fusion  # noqa: E402

# Freed buffers stay in MLX's cache (and Metal keeps them wired) unless capped;
# the server caps it at 2 GiB, this tight run needs less.
mx.set_cache_limit(int(float(os.environ.get("MLX_CACHE_MB", "512")) * 2**20))

model, processor = load_niwaki_with_stores(target_dir)
if os.environ.get("FUSION", "1") == "1":
    print("maps:", niwaki_fusion.quantize_maps(model), flush=True)
    print("gdn projections fused:", niwaki_fusion.fuse_gdn_projections(model), flush=True)
    print("fusion:", niwaki_fusion.install(model), flush=True)
    print("gdn layers compiled whole:", niwaki_fusion.compile_gdn_layers(model), flush=True)
niwaki_fusion.patch_speculative_verifier()
drafter, kind = load_drafter(drafter_dir, kind="mtp")
mx.eval(drafter.parameters())
print(f"drafter: {kind}, active {mx.get_active_memory() / 2**30:.2f} GiB, wired peak so far {wd.peak_wired:.1f} GB",
      flush=True)

tok = getattr(processor, "tokenizer", processor)
prompts = {
    "english": "Explain in detail why the sky is blue, covering Rayleigh scattering and sunsets.",
    "code": "Write a Python function that parses an ISO-8601 date string without libraries, with tests.",
    "german": "Erkläre ausführlich, warum der Himmel blau ist und warum Sonnenuntergänge rot sind.",
}


def run(label, text, use_draft):
    prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    kwargs = {"max_tokens": max_tokens, "temperature": 0.0}
    if use_draft:
        kwargs.update(draft_model=drafter, draft_kind=kind, draft_block_size=block)
        drafter.accept_lens.clear()
        drafter.draft_lens.clear()
    t0 = time.perf_counter()
    first = None
    n = 0
    out = []
    for chunk in stream_generate(model, processor, prompt, **kwargs):
        if first is None:
            first = time.perf_counter()
        n += 1
        out.append(chunk.text)
    end = time.perf_counter()
    rate = (n - 1) / (end - first) if n > 1 else 0.0
    acc = ""
    if use_draft and drafter.accept_lens:
        rounds = len(drafter.accept_lens)
        acc = (f", {rounds} rounds, accepted {sum(drafter.accept_lens) / rounds:.2f} of "
               f"{sum(drafter.draft_lens) / max(1, len(drafter.draft_lens)):.2f} drafts/round")
    print(f"{label:8s} {'MTP' if use_draft else 'plain':5s}: {n} tok, TTFT {(first - t0) * 1e3:.0f} ms, "
          f"decode {rate:.1f} tok/s{acc} | wired peak {wd.peak_wired:.1f} GB", flush=True)
    return "".join(out)


for label, text in prompts.items():
    plain = run(label, text, False)
    spec = run(label, text, True)
    same = plain == spec
    print(f"   same text as plain: {same}" + ("" if same else f" (first diff at char "
          f"{next(i for i, (a, b) in enumerate(zip(plain, spec)) if a != b) if any(a != b for a, b in zip(plain, spec)) else min(len(plain), len(spec))})"),
          flush=True)
