"""run_jevbench.py, but with the Jev-Style decision model as the head.

    python benchmarks/system_one/run_jevbench_jev.py <out.json>
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_jevbench  # noqa: E402
from jev_style_head import REPO, JevStyleHead  # noqa: E402

run_jevbench.load_head = lambda _path: JevStyleHead.load(REPO)

if __name__ == "__main__":
    run_jevbench.main(REPO, sys.argv[1])
