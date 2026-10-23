from __future__ import annotations

from maxmlx.daemon import main  # package dispatch: client verb → client, else → server

raise SystemExit(main(prog="python -m maxmlx.daemon"))
