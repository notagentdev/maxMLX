"""Score tickets720 with the Jev-Style decision model, in our result format.

    python benchmarks/system_one/run_tickets_jev.py <out.jsonl>
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from jev_style_head import REPO, JevStyleHead  # noqa: E402

SRC = HERE / "sources" / "tickets720.jsonl"


def main(out_path: str) -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    print(f"{len(rows)} rows; loading {REPO} ...", flush=True)
    head = JevStyleHead.load(REPO)

    started = time.perf_counter()
    correct = {}
    seen = {}
    with open(out_path, "w") as fh:
        for i, row in enumerate(rows, 1):
            options = row["options"]
            spec = {"type": "choice", "prompt": row["question"],
                    "options": [o["description"] for o in options]}
            answer = head.decide(row["state"], {"q": spec})["q"]
            probs = list(answer.distribution.values())
            fh.write(json.dumps({
                "id": row["id"],
                "option_ids": [o["id"] for o in options],
                "probabilities": probs,
            }) + "\n")
            fh.flush()
            fam = row["family"]
            seen[fam] = seen.get(fam, 0) + 1
            best = max(range(len(probs)), key=probs.__getitem__)
            correct[fam] = correct.get(fam, 0) + (best == row["label"])
            if i % 60 == 0:
                acc = {k: f"{correct[k]/seen[k]:.3f}" for k in sorted(seen)}
                el = time.perf_counter() - started
                print(f"  {i}/{len(rows)}  {acc}  {el/i*1000:.0f} ms/row", flush=True)

    el = time.perf_counter() - started
    print()
    for fam in sorted(seen):
        print(f"{fam:14s} {correct[fam]/seen[fam]:.4f}  ({correct[fam]}/{seen[fam]})")
    head_fams = [f for f in ("ticket_type", "queue") if f in seen]
    if head_fams:
        mean = sum(correct[f] / seen[f] for f in head_fams) / len(head_fams)
        print(f"\nheadline mean(type, queue) {mean:.4f}")
    print(f"{el:.1f} s total, {el/len(rows)*1000:.0f} ms/row; wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
