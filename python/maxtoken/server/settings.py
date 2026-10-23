"""Validated, atomic browser settings, shared with CLI startup configuration."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_SETTINGS_FILE = "~/.maxtoken/settings.json"
SERVER_FIELDS = frozenset({"server_host", "server_port"})
NULLABLE_FIELDS = frozenset({"temperature", "top_k", "top_p", "max_output_tokens", "max_reasoning_tokens", "thinking", "max_seq_len_override", "moe_cache_rate", "draft_model", "served_model_name", "reasoning_parser"})
LIVE_FIELDS = frozenset({
    "temperature", "top_k", "top_p", "max_output_tokens", "max_reasoning_tokens",
    "thinking", "max_seq_len_override", "moe_cache_size",
})
TOOL_PARSERS = ("auto", "llama3", "qwen", "qwen25", "qwen3_coder", "mistral", "deepseekv32", "gemma4", "glm47", "minimax", "minimax_m3", "muse_glimmer", "gpt_oss", "gpt-oss")
REASONING_PARSERS = ("auto", "off", "deepseekv32", "gpt_oss", "qwen3", "glm", "flint", "minimax", "minimax_m3", "muse_glimmer", "gemma4")


class SettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_k: int | None = Field(default=None, ge=-1)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_reasoning_tokens: int | None = Field(default=None, ge=0)
    thinking: Literal["model", "on", "off"] | None = None
    max_seq_len_override: int | None = Field(default=None, ge=1024)
    moe_backend: Literal["auto", "fused", "offload", "cpu", "hybrid"] | None = None
    moe_cache_size: int | None = Field(default=None, ge=0)
    moe_cache_rate: float | None = Field(default=None, gt=0, le=1)
    moe_cache_auto: bool | None = None
    memory_ratio: float | None = Field(default=None, gt=0, le=1)
    cache_type: Literal["radix", "naive"] | None = None
    prefix_cache_dir: str | None = Field(default=None, min_length=1, max_length=4096)
    prefix_cache_disk_gb: float | None = Field(default=None, ge=0)
    enable_cache_report: bool | None = None
    draft_model: str | None = Field(default=None, min_length=1, max_length=4096)
    draft_tokens: int | None = Field(default=None, ge=1, le=7)
    max_running_req: int | None = Field(default=None, ge=1, le=256)
    max_extend_tokens: int | None = Field(default=None, ge=1)
    served_model_name: str | None = Field(default=None, min_length=1, max_length=256)
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    server_host: Literal["127.0.0.1", "::1", "localhost"] | None = None
    server_port: int | None = Field(default=None, ge=1, le=65535)

    @model_validator(mode="after")
    def validate_options(self):
        for key in self.model_fields_set - NULLABLE_FIELDS:
            if getattr(self, key) is None:
                raise ValueError(f"{key} cannot be null")
        if self.top_k == 0:
            raise ValueError("top_k must be -1 (disabled) or positive")
        if self.tool_call_parser is not None and self.tool_call_parser not in TOOL_PARSERS:
            raise ValueError("unsupported tool-call parser")
        if self.reasoning_parser is not None and self.reasoning_parser not in REASONING_PARSERS:
            raise ValueError("unsupported reasoning parser")
        for value in (self.prefix_cache_dir, self.draft_model, self.served_model_name):
            if value is not None and not value.strip():
                raise ValueError("paths and names must not be blank")
        count = int(bool(self.moe_cache_size)) + int(self.moe_cache_rate is not None) + int(bool(self.moe_cache_auto))
        if count > 1:
            raise ValueError("choose expert slots, residency fraction, or automatic budget, not multiple")
        if self.moe_backend == "fused" and count:
            raise ValueError("resident serving cannot also configure expert offload slots")
        return self


def model_key(model: str) -> str:
    expanded = os.path.expanduser(model)
    return os.path.abspath(expanded) if expanded.startswith(("/", ".")) else expanded


class SettingsStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.environ.get("MAXTOKEN_SETTINGS_FILE", DEFAULT_SETTINGS_FILE)).expanduser()

    def _read(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "server": {}, "models": {}}
        with self.path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported settings format; existing file was not changed")
        if not isinstance(data.get("server"), dict) or not isinstance(data.get("models"), dict):
            raise ValueError("invalid settings file; existing file was not changed")
        SettingsPatch.model_validate(data["server"])
        if set(data["server"]) - SERVER_FIELDS:
            raise ValueError("invalid global settings")
        for values in data["models"].values():
            SettingsPatch.model_validate(values)
            if set(values) & SERVER_FIELDS:
                raise ValueError("server settings must be global")
        return data

    def load(self, model: str) -> dict:
        data = self._read()
        return {**data["models"].get(model_key(model), {}), **data["server"]}

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def save(self, model: str, patch: dict) -> dict:
        SettingsPatch.model_validate(patch)
        if patch.get("moe_cache_size", 0):
            patch = {**patch, "moe_cache_rate": None, "moe_cache_auto": False}
        elif patch.get("moe_cache_rate") is not None:
            patch = {**patch, "moe_cache_size": 0, "moe_cache_auto": False}
        elif patch.get("moe_cache_auto"):
            patch = {**patch, "moe_cache_size": 0, "moe_cache_rate": None}
        with self._locked():
            data = self._read()
            current = {**data["models"].get(model_key(model), {}), **data["server"]}
            merged = {**current, **patch}
            SettingsPatch.model_validate(merged)
            for key, value in patch.items():
                target = data["server"] if key in SERVER_FIELDS else data["models"].setdefault(model_key(model), {})
                target[key] = value
            fd, temporary = tempfile.mkstemp(prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(data, stream, indent=2, allow_nan=False)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return {**data["models"].get(model_key(model), {}), **data["server"]}


def saved_startup_values(model: str, path: str | Path | None = None) -> dict:
    values = SettingsStore(path).load(model)
    if values.get("thinking") == "model":
        values["thinking"] = None
    return values
