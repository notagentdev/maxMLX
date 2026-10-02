"""Score tickets720 with our System One head.

    python benchmarks/system_one/run_tickets.py <model> <out.jsonl>

The option descriptions, not the ids, are what the model sees — the ids are
labels for scoring only. Written incrementally so a long run is inspectable.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from run_jevbench import load_head  # noqa: E402

SRC = Path(__file__).parent / "sources" / "tickets720.jsonl"


def main(model_path: str, out_path: str) -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    print(f"{len(rows)} rows; loading {model_path} ...", flush=True)
    head = load_head(model_path)

    started = time.perf_counter()
    correct, seen = {}, {}
    with open(out_path, "w") as fh:
        for i, row in enumerate(rows, 1):
            options = row["options"]
            spec = {"type": "choice", "prompt": row["question"],
                    "options": [o["description"] for o in options]}
            answer = head.decide(row["state"], {"q": spec})["q"]
            probs = list(answer.distribution.values())
            fh.write(json.dumps({"id": row["id"],
                                 "option_ids": [o["id"] for o in options],
                                 "probabilities": probs}) + "\n")
            fh.flush()
            fam = row["family"]
            seen[fam] = seen.get(fam, 0) + 1
            best = max(range(len(probs)), key=probs.__getitem__)
            correct[fam] = correct.get(fam, 0) + (best == row["label"])
            if i % 60 == 0:
                acc = {k: f"{correct[k]/seen[k]:.3f}" for k in sorted(seen)}
                print(f"  {i}/{len(rows)}  {acc}  "
                      f"{(time.perf_counter()-started)/i*1000:.0f} ms/row", flush=True)

    el = time.perf_counter() - started
    print()
    for fam in sorted(seen):
        print(f"{fam:14s} {correct[fam]/seen[fam]:.4f}  ({correct[fam]}/{seen[fam]})")
    hf = [f for f in ("ticket_type", "queue") if f in seen]
    print(f"\nheadline mean(type, queue) {sum(correct[f]/seen[f] for f in hf)/len(hf):.4f}")
    print(f"{el:.1f} s total, {el/len(rows)*1000:.0f} ms/row; wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
