"""Model architectures mlx-lm does not ship yet.

mlx-lm resolves a checkpoint's ``model_type`` to ``mlx_lm.models.<type>``.
``register_all()`` makes each module here importable under that name, so
``mlx_lm.load`` builds it like any built-in model. An mlx-lm release that
ships the architecture itself wins: its module is used and ours is skipped.
"""

import importlib
import importlib.util
import logging
import sys

logger = logging.getLogger(__name__)

# model_type -> module in this package
ARCHS = {"flint": "flint"}


def register_all() -> list:
    registered = []
    for model_type, module in ARCHS.items():
        target = f"mlx_lm.models.{model_type}"
        if target in sys.modules or importlib.util.find_spec(target) is not None:
            continue
        sys.modules[target] = importlib.import_module(f"{__name__}.{module}")
        registered.append(model_type)
    if registered:
        logger.info("registered model architectures with mlx-lm: %s", ", ".join(registered))
    return registered
