"""Model switching from the console: which local checkpoints are offered, and which
command line each one is launched with."""

from __future__ import annotations

import json

import pytest

from maxtoken.server import model_switch as ms

LAUNCH = [
    "--model-path", "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit", "--moe-backend", "offload",
    "--enable-cache-report", "--port", "1234", "--thinking", "on",
    "--draft-model=downloads/ornith-1.5-35b-mtplx/mtp/weights.safetensors", "--draft-tokens", "2",
]


def test_split_keeps_server_flags_and_takes_model_flags_in_both_spellings():
    model, server = ms.split_argv(LAUNCH)
    assert server == ["--moe-backend", "offload", "--enable-cache-report", "--port", "1234", "--thinking", "on"]
    assert model == [
        "--model-path", "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit",
        "--draft-model=downloads/ornith-1.5-35b-mtplx/mtp/weights.safetensors", "--draft-tokens", "2",
    ]


@pytest.fixture
def profiles_file(tmp_path, monkeypatch):
    path = tmp_path / "profiles.json"
    monkeypatch.setenv("MAXTOKEN_LAUNCH_PROFILES", str(path))
    return path


def test_a_remembered_profile_wins_over_the_default(profiles_file):
    ms.record_profile("ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit", LAUNCH)
    entry = {"id": "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit", "mtp_head": False, "moe": True}
    flags = ms.model_flags_for(entry, ms.load_profiles())
    assert "--draft-model=downloads/ornith-1.5-35b-mtplx/mtp/weights.safetensors" in flags
    argv = ms.server_argv_with(["--model-path", "other", "--port", "1234", "--draft-model", "mtp"], flags)
    assert argv[:2] == ["--port", "1234"] and argv.count("--model-path") == 1


def test_default_drafts_with_the_models_own_head_at_the_measured_depth(profiles_file):
    moe = {"id": "/m/moe", "mtp_head": True, "moe": True}
    dense = {"id": "/m/dense", "mtp_head": True, "moe": False}
    plain = {"id": "/m/plain", "mtp_head": False, "moe": False}
    assert ms.model_flags_for(moe, {}) == ["--model-path", "/m/moe", "--draft-model", "mtp", "--draft-tokens", "2"]
    assert ms.model_flags_for(dense, {})[-1] == "3"
    assert ms.model_flags_for(plain, {}) == ["--model-path", "/m/plain"]
    # A draft model saved in the console settings is left to the settings store.
    assert ms.model_flags_for(dense, {}, {"draft_model": "x"}) == ["--model-path", "/m/dense"]


def _checkpoint(d, *, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
                present=None, cfg=None, mtp=False):
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg or {"model_type": "qwen3_5", "quantization": {"bits": 4}}))
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}))
    for s in shards if present is None else present:
        (d / s).write_bytes(b"x" * 10)
    if mtp:
        (d / "mtp.safetensors").write_bytes(b"x")
    return d


def test_discovery_marks_incomplete_checkpoints_and_the_served_one(tmp_path):
    downloads = tmp_path / "downloads"
    _checkpoint(downloads / "dense", mtp=True)
    _checkpoint(downloads / "gone", present=())
    _checkpoint(downloads / "moe", cfg={"model_type": "qwen3_5_moe", "text_config": {"num_experts": 256}})
    (downloads / "notes").mkdir()
    hub = tmp_path / "hub"
    snap = hub / "models--org--name" / "snapshots" / "abc"
    _checkpoint(snap)
    (hub / "models--org--name" / "refs").mkdir()
    (hub / "models--org--name" / "refs" / "main").write_text("abc")

    found = {m["name"]: m for m in ms.local_models([downloads], str(downloads / "dense"), hub=hub)}
    assert set(found) == {"dense", "gone", "moe", "org/name"}
    assert found["dense"]["active"] and found["dense"]["mtp_head"] and found["dense"]["complete"]
    assert not found["gone"]["complete"] and found["gone"]["missing_shards"] == 2
    assert found["moe"]["moe"] and not found["dense"]["moe"]
    assert found["org/name"]["id"] == "org/name" and found["org/name"]["source"] == "huggingface"
    assert found["dense"]["size_bytes"] >= 20


def test_a_stale_index_does_not_hide_a_complete_shard_set(tmp_path):
    d = tmp_path / "gemma"
    d.mkdir()
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)}}))
    (d / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert ms.missing_shards(d)[0] is False or ms.missing_shards(d)[1], "half of the real set is not complete"
    (d / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert ms.missing_shards(d) == (True, [])


def test_a_checkpoint_with_its_own_model_file_is_supported(tmp_path):
    d = _checkpoint(tmp_path / "kolibri", cfg={"model_type": "kolibri1", "model_file": "kolibri1.py"})
    assert ms.local_models([tmp_path])[0]["supported"] is False, "model_file named but missing"
    (d / "kolibri1.py").write_text("# architecture")
    assert ms.local_models([tmp_path])[0]["supported"] is True
