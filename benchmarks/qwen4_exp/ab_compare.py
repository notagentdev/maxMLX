"""A/B: the same German prompts, greedy, through the stock Niwaki model and
through the optimized one (hc_kernel + niwaki_fusion), in separate processes.
Writes the token ids and text per prompt to a JSON file; --compare diffs two.

    ~/.venvs/qwen4-spike/bin/python ab_compare.py run <model_dir> <out.json>    (FAST=0|1, or LEVERS=hck,maps,gdnfuse,fusion,gdnfull)
    ~/.venvs/qwen4-spike/bin/python ab_compare.py compare <a.json> <b.json>
"""
import json
import os
import sys

PROMPTS = [
    "wer bist du und was ist deine aufgabe.",
    "was sind deine programmiersprachen?",
    "Wie viele geordnete Paare positiver ganzer Zahlen (x, y) erfüllen die Gleichung 1/x + 1/y = 1/12? "
    "Begründe deine Antwort Schritt für Schritt.",
]
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "160"))


def run(model_dir, out_path):
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_here, ".."))
    sys.path.insert(0, _here)
    import mlx.core as mx
    from mlx_vlm import stream_generate

    from niwaki_stores import load_niwaki_with_stores

    model, processor = load_niwaki_with_stores(os.path.abspath(model_dir), log=lambda *a: None)
    mx.set_cache_limit(512 * 2**20)
    levers = os.environ.get("LEVERS")
    if levers is None:
        levers = "hck,maps,gdnfuse,fusion,gdnfull" if os.environ.get("FAST", "1") == "1" else ""
    levers = [x for x in levers.split(",") if x]
    import hc_kernel
    import niwaki_fusion

    if "hck" in levers:
        hc_kernel.install(model)
    if "maps" in levers:
        niwaki_fusion.quantize_maps(model)
    if "gdnfuse" in levers:
        niwaki_fusion.fuse_gdn_projections(model)
    if "fusion" in levers:
        niwaki_fusion.install(model)
    if "gdnfull" in levers:
        niwaki_fusion.compile_gdn_layers(model)
    print("levers:", levers or "none (stock)", flush=True)
    tok = getattr(processor, "tokenizer", processor)
    results = []
    for think in (False, True):
        for p in PROMPTS:
            prompt = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=think)
            ids, text = [], []
            for ch in stream_generate(model, processor, prompt, max_tokens=MAX_TOKENS, temperature=0.0,
                                      prefill_step_size=64):
                ids.append(int(ch.token))
                text.append(ch.text)
            results.append({"think": think, "prompt": p, "ids": ids, "text": "".join(text)})
            print(f"[think={think}] {p[:40]!r}: {len(ids)} tok: {''.join(text)[:150]!r}", flush=True)
    json.dump({"levers": levers, "results": results}, open(out_path, "w"), ensure_ascii=False, indent=1)


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"A = {a['levers'] or 'stock'}   B = {b['levers'] or 'stock'}")
    for ra, rb in zip(a["results"], b["results"]):
        n = next((i for i, (x, y) in enumerate(zip(ra["ids"], rb["ids"])) if x != y), None)
        same = n is None and len(ra["ids"]) == len(rb["ids"])
        where = "identical" if same else f"first different token at {n if n is not None else min(len(ra['ids']), len(rb['ids']))}"
        print(f"\n[think={ra['think']}] {ra['prompt'][:50]!r}: {where} ({len(ra['ids'])} vs {len(rb['ids'])} tok)")
        if not same:
            print("  A:", ra["text"][:300].replace("\n", " "))
            print("  B:", rb["text"][:300].replace("\n", " "))
        else:
            print("  :", ra["text"][:300].replace("\n", " "))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        compare(sys.argv[2], sys.argv[3])
