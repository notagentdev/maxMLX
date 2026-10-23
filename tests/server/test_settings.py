from __future__ import annotations


import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from maxmlx.server import api_server as api
from maxmlx.server.args import ServerArgs, parse_args
from maxmlx.server.settings import SettingsPatch, SettingsStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("MAXMLX_SETTINGS_FILE", str(path))
    return SettingsStore(path)


@pytest.fixture
def running(store, monkeypatch):
    config = ServerArgs(model_path="/models/test", max_seq_len_override=8192, context_length_ceiling=32768, settings_file=str(store.path))
    state = SimpleNamespace(config=config, maintenance_state="serving", settings_lock=None, generation_override={}, context_length_override=None, reasoning_budget_override=None, cache_pools={}, last_rebuild=None)
    monkeypatch.setattr(api, "_GLOBAL_STATE", state)
    monkeypatch.setattr(api, "_MODEL_SAMPLING", {"temperature": 0.7, "top_k": 20, "top_p": 0.95})
    return state, TestClient(api.app, client=("127.0.0.1", 50000))


def test_store_is_model_scoped_and_server_settings_are_global(store):
    store.save("/models/a", {"temperature": 0.8, "server_port": 2020})
    store.save("/models/b", {"temperature": 0.5})
    assert store.load("/models/a") == {"temperature": 0.8, "server_port": 2020}
    assert store.load("/models/b") == {"temperature": 0.5, "server_port": 2020}
    assert SettingsStore(store.path).load("/models/a")["temperature"] == 0.8


def test_disabling_a_saved_drafter_is_persistent(store):
    store.save("m", {"draft_model": "mtp"})
    store.save("m", {"draft_model": None})
    assert "draft_model" in store.load("m") and store.load("m")["draft_model"] is None


def test_slot_selection_clears_other_budget_modes(store):
    store.save("m", {"moe_backend": "offload", "moe_cache_rate": 0.25})
    store.save("m", {"moe_cache_size": 128})
    values = store.load("m")
    assert values["moe_cache_size"] == 128 and values["moe_cache_rate"] is None
    assert values["moe_cache_auto"] is False


def test_concurrent_saves_preserve_other_models(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: store.save(f"model-{i}", {"top_k": i + 1}), range(12)))
    assert len(json.loads(store.path.read_text())["models"]) == 12
    assert not list(store.path.parent.glob("*.tmp"))


def test_invalid_or_future_file_is_never_overwritten(store):
    store.path.write_text('{"version": 99}')
    with pytest.raises(ValueError):
        store.save("m", {"temperature": 0.8})
    assert store.path.read_text() == '{"version": 99}'


@pytest.mark.parametrize("patch", [{"top_k": 0}, {"top_p": 1.1}, {"temperature": float("nan")}, {"draft_tokens": 0}, {"server_port": 70000}, {"memory_ratio": 0}, {"cache_type": "foreign"}, {"max_running_req": None}, {"prefix_cache_dir": " "}, {"tool_call_parser": "foreign"}, {"moe_cache_rate": 0.2, "moe_cache_auto": True}, {"server_host": "0.0.0.0"}])
def test_invalid_values_are_rejected(patch):
    with pytest.raises(ValidationError):
        SettingsPatch.model_validate(patch)


def test_saved_values_load_at_startup_but_explicit_cli_wins(store, monkeypatch):
    monkeypatch.setattr("maxmlx.utils.cached_load_hf_config", lambda _: SimpleNamespace(to_dict=lambda: {"max_position_embeddings": 32768}))
    store.save("/models/test", {"temperature": 0.8, "top_p": 0.9, "server_port": 2020, "draft_model": "mtp", "max_seq_len_override": 8192})
    args, _ = parse_args(["--model", "/models/test", "--dtype", "bfloat16"])
    assert (args.temperature, args.top_p, args.server_port, args.draft_model) == (0.8, 0.9, 2020, "mtp")
    assert args.max_seq_len == 8192 and args.context_length_ceiling == 32768
    args, _ = parse_args(["--model", "/models/test", "--dtype", "bfloat16", "--temperature=0.4", "--port", "3030"])
    assert args.temperature == 0.4 and args.server_port == 3030


def test_abbreviated_cli_options_and_cache_mode_override_saved_values(store, monkeypatch):
    monkeypatch.setattr("maxmlx.utils.cached_load_hf_config", lambda _: SimpleNamespace(to_dict=lambda: {"max_position_embeddings": 32768}))
    store.save("/models/test", {"temperature": 0.8, "server_port": 2020, "moe_backend": "fused"})
    args, _ = parse_args(["--model", "/models/test", "--dtype", "bfloat16", "--temperat=0.4", "--por", "3030", "--moe-cache-rate", "0.4"])
    assert args.temperature == 0.4 and args.server_port == 3030
    assert args.moe_cache_rate == 0.4 and args.moe_backend == "auto"


def test_cleared_alias_and_reasoning_parser_are_not_permanently_pending(running, store):
    state, client = running
    state.config = replace(state.config, served_model_name="test", reasoning_parser=None)
    store.save(state.config.model_path, {"served_model_name": None, "reasoning_parser": None})
    response = client.get("/admin/settings")
    assert response.status_code == 200
    assert response.json()["pending_restart"] == []


def test_live_sampling_is_saved_and_used_by_all_adapters(running, store):
    state, client = running
    response = client.post("/admin/settings", json={"temperature": 0.9, "top_k": 40, "top_p": 0.8, "max_output_tokens": 8192, "thinking": "off"})
    assert response.status_code == 200, response.text
    assert api.effective_sampling() == {"temperature": 0.9, "top_k": 40, "top_p": 0.8, "max_tokens": 8192}
    assert store.load(state.config.model_path)["top_k"] == 40
    assert response.json()["settings"]["pending_restart"] == []


def test_one_save_persists_live_and_restart_settings_together(running, store):
    state, client = running
    state.config = replace(state.config, moe_backend="offload", moe_cache_size=128)
    patch = {"temperature": 0.9, "top_k": 30, "moe_backend": "offload", "moe_cache_size": 64, "moe_cache_rate": None, "moe_cache_auto": False, "max_running_req": 2}
    response = client.post("/admin/settings", json=patch)
    assert response.status_code == 200, response.text
    saved = store.load(state.config.model_path)
    assert all(saved[key] == value for key, value in patch.items())
    assert api.effective_sampling()["temperature"] == 0.9
    assert state.config.max_running_req == 4 and state.config.moe_cache_size == 128
    assert {"max_running_req", "moe_cache_size"}.issubset(response.json()["settings"]["pending_restart"])


def test_invalid_request_does_not_mutate_or_save(running, store):
    state, client = running
    response = client.post("/admin/settings", json={"temperature": 0.9, "thinking": "foreign"})
    assert response.status_code == 422
    assert state.generation_override == {} and not store.path.exists()
    response = client.post("/admin/settings", json={"max_seq_len_override": 65536})
    assert response.status_code == 422 and not store.path.exists()


def test_unavailable_engine_and_disk_failure_do_not_apply_live_settings(running, store, monkeypatch):
    state, client = running
    state.maintenance_state = "loading"
    assert client.post("/admin/settings", json={"temperature": 0.9}).status_code == 409
    assert not store.path.exists()
    state.maintenance_state = "serving"
    monkeypatch.setattr(SettingsStore, "save", lambda *args: (_ for _ in ()).throw(OSError("fixture disk failure")))
    response = client.post("/admin/settings", json={"temperature": 0.9})
    assert response.status_code == 500 and state.generation_override == {}


def test_restart_settings_are_saved_but_not_claimed_live(running, store):
    state, client = running
    response = client.post("/admin/settings", json={"draft_model": "mtp", "draft_tokens": 2, "max_running_req": 2})
    assert response.status_code == 200
    assert state.config.draft_model is None
    assert "draft_model" in response.json()["settings"]["pending_restart"]
    assert store.load(state.config.model_path)["draft_model"] == "mtp"


def test_settings_mutations_reject_remote_origins_and_dns_rebinding(running, store):
    _, client = running
    assert client.post("/admin/settings", json={"temperature": 0.9}, headers={"Origin": "https://foreign.example"}).status_code == 403
    assert client.post("/admin/settings", json={"temperature": 0.9}, headers={"Host": "foreign.example"}).status_code == 403
    remote = TestClient(api.app, client=("192.168.1.2", 50000))
    assert remote.post("/admin/settings", json={"temperature": 0.9}).status_code == 403
    assert not store.path.exists()


def test_reload_activates_saved_engine_config_without_rebinding_server(running, store, monkeypatch):
    state, client = running
    store.save(state.config.model_path, {"draft_model": "mtp", "draft_tokens": 2, "max_running_req": 2, "server_port": 2020})
    monkeypatch.setattr("maxmlx.utils.cached_load_hf_config", lambda _: SimpleNamespace(to_dict=lambda: {"max_position_embeddings": 32768}))
    captured = []
    async def reload(state, config=None):
        captured.append(config)
        return {"status": "ok"}, 200
    monkeypatch.setattr(api, "reload_backend", reload)
    response = client.post("/admin/reload", json={"apply_saved": True})
    assert response.status_code == 200, response.text
    config = captured[0]
    assert (config.draft_model, config.draft_tokens, config.max_running_req) == ("mtp", 2, 2)
    assert config.server_port == 1919
    assert config._unique_suffix == state.config._unique_suffix


def test_live_resize_failure_is_reported_as_saved_not_applied(running, store, monkeypatch):
    state, client = running
    async def failure(req):
        return {"status": "busy", "error": "request in progress"}
    monkeypatch.setattr(api, "cache_rebuild", failure)
    response = client.post("/admin/settings", json={"moe_cache_size": 128})
    assert response.status_code == 409
    assert response.json()["applied"] is False
    assert store.load(state.config.model_path)["moe_cache_size"] == 128
    assert state.config.moe_cache_size == 0


def test_automatic_slot_budget_uses_slot_cache_not_mapped_path(tmp_path, monkeypatch):
    from maxmlx.mlx_backend.worker import MlxScheduler
    scheduler = MlxScheduler.__new__(MlxScheduler)
    scheduler.model = SimpleNamespace(parameters=lambda: {})
    scheduler._mx = SimpleNamespace(eval=lambda *args: None, get_active_memory=lambda: 10)
    state = SimpleNamespace(glus=[object()], resize_total=lambda slots: slots, cache_bytes=lambda: 1024)
    called = []
    monkeypatch.setattr("maxmlx.mlx_backend.offload.attach_expert_offload", lambda *args: (called.append("slots") or state))
    monkeypatch.setattr("maxmlx.mlx_backend.ftw_mlx.attach_mapped_experts", lambda *args: called.append("mapped"))
    scheduler._resolve_slots_per_layer = lambda *args: 16
    scheduler._attach_offload(ServerArgs(model_path=str(tmp_path), moe_cache_auto=True))
    assert called == ["slots"]
