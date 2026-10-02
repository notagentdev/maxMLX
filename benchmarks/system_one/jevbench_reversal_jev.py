"""jevbench_reversal.py with the Jev-Style decision model as the head."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jevbench_reversal  # noqa: E402
from jev_style_head import REPO, JevStyleHead  # noqa: E402

jevbench_reversal.load_head = lambda _path: JevStyleHead.load(REPO)

if __name__ == "__main__":
    jevbench_reversal.main(REPO, sys.argv[1])
