"""The state directories, moved once from their pre-rename names.

maxMLX keeps its state in ``~/.maxmlx`` (settings, chats, prefix cache,
images, videos, daemon) and its packed expert stores in ``~/.cache/maxmlx``.
Before the rename they were ``~/.maxtoken`` and ``~/.cache/maxtoken`` --
tens of GB of prefix snapshots and stores that cost minutes to rebuild. The
first start after the rename moves each old directory (a rename, no copy)
and leaves a symlink in its place, so paths written into settings files or
scripts, and a server still running the old code, keep resolving.
"""

from __future__ import annotations

import os
import sys

LEGACY_DIRS = (
    ("~/.maxtoken", "~/.maxmlx"),
    ("~/.cache/maxtoken", "~/.cache/maxmlx"),
)


def migrate_legacy_dirs() -> None:
    for old, new in LEGACY_DIRS:
        old, new = os.path.expanduser(old), os.path.expanduser(new)
        if os.path.islink(old) or not os.path.isdir(old) or os.path.lexists(new):
            continue
        try:
            os.rename(old, new)
            os.symlink(new, old)
        except OSError as exc:  # never block a start over this
            print(f"maxmlx: could not move {old} to {new}: {exc}", file=sys.stderr)
