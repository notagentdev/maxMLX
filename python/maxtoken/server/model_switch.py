"""Serve a different local model from the console without leaving the port.

The engine already knows how to relaunch its workers with a new configuration
(``reload_backend``); a model switch is that relaunch with another checkpoint.
What this module adds is everything around it that is specific to the model:

- **Which models exist.** The downloads folder (one checkpoint per directory)
  and the Hugging Face hub cache (``models--org--name/snapshots/<rev>``). A
  directory is offered only when it holds a ``config.json``; one whose shards
  are missing (a cleanup that took the weights and left the index) is listed
  as incomplete instead of failing at load.
- **How each model is launched.** Flags such as the draft model differ per
  checkpoint (Ornith drafts with an external sidecar at k=2, the dense 27B with
  its own head at k=3). The model-specific part of the command line a model was
  last served with is remembered as its *launch profile* and reused when it is
  picked again; server flags (port, cache report, MoE backend, thinking) carry
  over from the running server. A model without a profile gets its own MTP
  head when it ships one: k=2 on a MoE trunk, k=3 on a dense one (the depths
  measured best on this machine).
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

# Flags that belong to the checkpoint rather than to the server. Each takes
# exactly one value (``--flag value`` or ``--flag=value``).
MODEL_FLAGS = (
    "--model-path", "--model", "--draft-model", "--draft-tokens",
    "--served-model-name", "--tool-call-parser", "--reasoning-parser",
)
PROFILE_FILE = "~/.maxtoken/launch-profiles.json"

# The names find_mtp_weights (mlx_backend/mtp_draft.py) checks first. Kept
# here because importing the backend package pulls MLX into the API process.
_MTP_HEADS = ("mtp.safetensors", "model-mtp-head.safetensors", os.path.join("mtp", "weights.safetensors"))


def split_argv(argv: Iterable[str]) -> tuple[list[str], list[str]]:
    """``(model_part, server_part)`` of a ``mt serve`` command line."""
    argv = list(argv)
    model: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.split("=", 1)[0] in MODEL_FLAGS:
            take = 1 if "=" in token else 2
            model.extend(argv[i:i + take])
            i += take
            continue
        rest.append(token)
        i += 1
    return model, rest


def identity(ref: str) -> str:
    """One name per checkpoint: the real path of a local directory, else the repo id."""
    expanded = os.path.expanduser(ref)
    return os.path.realpath(expanded) if os.path.isdir(expanded) else ref


def mtp_head(model_dir: str | Path) -> bool:
    return any((Path(model_dir) / name).is_file() for name in _MTP_HEADS)


# -- launch profiles ------------------------------------------------------------


def _profile_path() -> Path:
    return Path(os.environ.get("MAXTOKEN_LAUNCH_PROFILES", PROFILE_FILE)).expanduser()


def load_profiles() -> dict[str, list[str]]:
    try:
        data = json.loads(_profile_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: [str(t) for t in v] for k, v in data.items() if isinstance(v, list)}


def record_profile(model_ref: str, argv: Iterable[str]) -> None:
    """Remember the model-specific flags ``argv`` served ``model_ref`` with."""
    model_part, _ = split_argv(argv)
    if not model_part:
        return
    path = _profile_path()
    profiles = load_profiles()
    profiles[identity(model_ref)] = model_part
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(profiles, stream, indent=2)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def model_flags_for(entry: dict, profiles: dict[str, list[str]], saved: dict | None = None) -> list[str]:
    """The model-specific flags a switch to ``entry`` launches with."""
    remembered = profiles.get(identity(entry["id"]))
    if remembered:
        return list(remembered)
    flags = ["--model-path", entry["id"]]
    # A draft model saved in the console's settings for this checkpoint wins
    # (explicit flags would override it, so none are added).
    if entry.get("mtp_head") and not (saved or {}).get("draft_model"):
        flags += ["--draft-model", "mtp", "--draft-tokens", "2" if entry.get("moe") else "3"]
    return flags


def server_argv_with(launch_argv: Iterable[str], model_flags: Iterable[str]) -> list[str]:
    """The running server's own flags with another model's flags in place of its own."""
    _, server_part = split_argv(launch_argv)
    return server_part + list(model_flags)


# -- discovery ------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _mlx_lm_architectures() -> frozenset[str] | None:
    """The ``model_type`` values mlx-lm can load, read from its package files
    (``find_spec`` locates the package without importing it, and with it MLX).
    None when mlx-lm is not installed — then nothing is ruled out."""
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(next(iter(spec.submodule_search_locations)))
    names = {p.stem for p in (root / "models").glob("*.py") if not p.stem.startswith("_")}
    try:
        utils = (root / "utils.py").read_text(encoding="utf-8")
        block = re.search(r"MODEL_REMAPPING\s*=\s*\{(.*?)\}", utils, re.S)
        for alias, target in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', block.group(1) if block else ""):
            if target in names:
                names.add(alias)
    except OSError:
        pass
    return frozenset(names)


def hf_hub_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    return Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"


_SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.safetensors$")


def missing_shards(directory: Path) -> tuple[bool, list[str]]:
    """(has weights, missing shard names) for a checkpoint directory.

    The index is the authority, except when it is stale: some exports ship an index
    that names a sharding the folder does not contain (mlx-community/gemma-3-12b-it-4bit
    lists five ``-of-00005`` shards beside two ``-of-00002`` files). mlx-lm loads every
    ``*.safetensors`` regardless, so a complete ``-of-N`` set counts as complete."""
    present = {p.name for p in directory.glob("*.safetensors") if "mtp" not in p.name.lower()}
    try:
        shards = set(json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"].values())
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        shards = set()
    if shards:
        missing = sorted(shards - present)
        if not missing:
            return True, []
    else:
        missing = []
    groups: dict[tuple[str, str], set[int]] = {}
    for name in present:
        m = _SHARD.match(name)
        if m:
            groups.setdefault((m.group(1), m.group(3)), set()).add(int(m.group(2)))
    for (_, total), seen in groups.items():
        if seen == set(range(1, int(total) + 1)):
            return True, []
    if shards:
        return False, missing
    return bool(present), []


def _describe(directory: Path) -> dict | None:
    """Facts about one checkpoint directory, or None when it is not a model."""
    try:
        cfg = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    text = cfg.get("text_config") or {}
    get = lambda key: cfg.get(key, text.get(key))  # noqa: E731
    has_weights, missing = missing_shards(directory)
    size = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():  # follows the hub's blob symlinks; a dangling one is skipped
                size += path.stat().st_size
        except OSError:
            continue
    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    experts = get("num_experts") or get("n_routed_experts") or get("num_local_experts") or 0
    model_type = cfg.get("model_type")
    known = _mlx_lm_architectures()
    return {
        "model_type": model_type,
        # mlx-lm dispatches on the top-level type; Qwen3.8-Flash-Next
        # (qwen4_exp) and its MTP sidecars only run in the spike server.
        # ... or ships its own architecture file, which mlx-lm loads via "model_file"
        # (Kolibri-1's kolibri1.py).
        "supported": known is None or model_type in known
        or bool(cfg.get("model_file") and (directory / str(cfg["model_file"])).is_file()),
        "moe": bool(experts),
        "experts": int(experts or 0),
        "layers": get("num_hidden_layers"),
        "bits": quant.get("bits") if isinstance(quant, dict) else None,
        "vision": "vision_config" in cfg,
        "mtp_head": mtp_head(directory),
        "size_bytes": size,
        "complete": has_weights and not missing,
        "missing_shards": len(missing),
    }


def _hub_snapshot(repo_dir: Path) -> Path | None:
    snapshots = repo_dir / "snapshots"
    ref = repo_dir / "refs" / "main"
    try:
        if ref.is_file():
            pinned = snapshots / ref.read_text(encoding="utf-8").strip()
            if pinned.is_dir():
                return pinned
        candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    except OSError:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def local_models(models_dirs: Iterable[str | Path], current: str | None = None, hub: Path | None = None) -> list[dict[str, Any]]:
    """Every checkpoint in the downloads folder and the hub cache, by name within
    each source, with the one being served marked ``active``."""
    current_id = identity(current) if current else None
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(ref: str, directory: Path, source: str, name: str) -> None:
        real = os.path.realpath(directory)
        if real in seen:
            return
        facts = _describe(directory)
        if facts is None:
            return
        seen.add(real)
        active = current_id is not None and (identity(ref) == current_id or real == current_id)
        out.append({"id": ref, "name": name, "source": source, "path": str(directory), "active": active, **facts})

    for folder in models_dirs:
        root = Path(folder).expanduser()
        try:
            dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            dirs = []
        for d in dirs:
            add(str(d.resolve()), d, "downloads", d.name)
    hub = hub if hub is not None else hf_hub_dir()
    try:
        repos = sorted(p for p in hub.iterdir() if p.is_dir() and p.name.startswith("models--"))
    except OSError:
        repos = []
    for repo in repos:
        snapshot = _hub_snapshot(repo)
        if snapshot is None:
            continue
        repo_id = repo.name[len("models--"):].replace("--", "/", 1)
        add(repo_id, snapshot, "huggingface", repo_id)
    return out
