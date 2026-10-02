"""Download-tab configuration, kept apart from the model settings file.

``settings.json`` is validated strictly per served model; the downloader's models folder and
mirror endpoints belong to the machine, so they live in their own small file beside it.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_CONFIG_FILE = "~/.maxtoken/downloads.json"
DEFAULT_TASKS_FILE = "~/.maxtoken/download-tasks.json"
DEFAULT_MODELS_DIR = "~/.maxtoken/models"


@dataclass(frozen=True)
class DownloadConfig:
    models_dir: str = ""
    hf_endpoint: str = ""
    ms_endpoint: str = ""


def config_path() -> Path:
    return Path(os.environ.get("MAXTOKEN_DOWNLOADS_FILE", DEFAULT_CONFIG_FILE)).expanduser()


def tasks_path() -> Path:
    return Path(os.environ.get("MAXTOKEN_DOWNLOAD_TASKS_FILE", DEFAULT_TASKS_FILE)).expanduser()


def load_config() -> DownloadConfig:
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DownloadConfig()
    if not isinstance(data, dict):
        return DownloadConfig()
    return DownloadConfig(**{key: data[key].strip() for key in DownloadConfig.__dataclass_fields__
                             if isinstance(data.get(key), str)})


def validate_endpoint(value: str) -> str:
    """An empty value means the default hub; anything else must be an http(s) origin."""
    value = value.strip().rstrip("/")
    if not value:
        return ""
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("endpoint must be an http(s) URL such as https://hf-mirror.com")
    return value


def save_config(patch: dict) -> DownloadConfig:
    current = asdict(load_config())
    for key, value in patch.items():
        if key not in current:
            raise ValueError(f"unknown field: {key}")
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        current[key] = validate_endpoint(value) if key.endswith("_endpoint") else value.strip()
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(current, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return DownloadConfig(**current)


def resolve_models_dir(served_model_path: str | None = None) -> Path:
    """Where new models land: the saved folder, else beside the model being served (so a
    ``downloads/`` folder keeps collecting), else ``~/.maxtoken/models``."""
    configured = os.environ.get("MAXTOKEN_MODELS_DIR") or load_config().models_dir
    if configured:
        return Path(configured).expanduser()
    if served_model_path:
        served = Path(served_model_path).expanduser()
        if served.is_dir():
            return served.resolve().parent
    return Path(DEFAULT_MODELS_DIR).expanduser()
