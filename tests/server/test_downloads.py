"""The console's Downloads tab: configuration, restart behaviour and the HTTP surface.

Nothing here touches the network: routes run against fakes or fail validation before any
transfer is queued.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from maxtoken.server import api_server as api
from maxtoken.server.downloader import config as dlconfig
from maxtoken.server.downloader.hf import DownloadStatus, DownloadTask, HFDownloader


@pytest.fixture(autouse=True)
def isolated_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MAXTOKEN_DOWNLOADS_FILE", str(tmp_path / "downloads.json"))
    monkeypatch.setenv("MAXTOKEN_DOWNLOAD_TASKS_FILE", str(tmp_path / "tasks.json"))
    monkeypatch.delenv("MAXTOKEN_MODELS_DIR", raising=False)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_GLOBAL_STATE", None)
    # A fresh service so downloaders built by one test never leak into the next.
    monkeypatch.setattr(api._DOWNLOADS, "_downloaders", {})
    return TestClient(api.app, client=("127.0.0.1", 50000))


class FakeDownloader:
    def __init__(self):
        self.model_dir = None
        self.calls = []
        self.tasks = {"t1": DownloadTask(task_id="t1", repo_id="o/m", status=DownloadStatus.FAILED)}

    def update_model_dir(self, path):
        self.model_dir = path

    def get_tasks(self):
        return [task.to_dict() for task in self.tasks.values()]

    async def start_download(self, repo_id, token=""):
        self.calls.append(("start", repo_id, token))
        if repo_id == "dup/dup":
            raise ValueError("already in progress")
        task = DownloadTask(task_id="t2", repo_id=repo_id, status=DownloadStatus.PENDING)
        self.tasks[task.task_id] = task
        return task

    async def cancel_download(self, task_id):
        self.calls.append(("cancel", task_id))
        return task_id == "t2"

    async def retry_download(self, task_id, token=""):
        self.calls.append(("retry", task_id, token))
        if task_id not in self.tasks:
            raise ValueError("Task not found")
        return DownloadTask(task_id="t3", repo_id=self.tasks[task_id].repo_id)

    def remove_task(self, task_id):
        return self.tasks.pop(task_id, None) is not None


def test_config_round_trip_and_endpoint_validation():
    assert dlconfig.load_config() == dlconfig.DownloadConfig()
    saved = dlconfig.save_config({"hf_endpoint": " https://hf-mirror.com/ ", "models_dir": "/data/models"})
    assert saved.hf_endpoint == "https://hf-mirror.com" and saved.models_dir == "/data/models"
    assert dlconfig.load_config() == saved
    # A later patch changes only what it names; an empty endpoint returns to the default hub.
    assert dlconfig.save_config({"hf_endpoint": ""}).models_dir == "/data/models"
    for bad in ("hf-mirror.com", "ftp://mirror", "https://"):
        with pytest.raises(ValueError):
            dlconfig.save_config({"hf_endpoint": bad})
    with pytest.raises(ValueError):
        dlconfig.save_config({"unknown": "x"})


def test_models_folder_prefers_setting_then_served_model_then_home(tmp_path, monkeypatch):
    served = tmp_path / "library" / "model-a"
    served.mkdir(parents=True)
    assert dlconfig.resolve_models_dir(str(served)) == served.parent.resolve()
    assert str(dlconfig.resolve_models_dir(None)).endswith("/.maxtoken/models")
    assert str(dlconfig.resolve_models_dir(str(tmp_path / "missing"))).endswith("/.maxtoken/models")
    dlconfig.save_config({"models_dir": str(tmp_path / "chosen")})
    assert dlconfig.resolve_models_dir(str(served)) == tmp_path / "chosen"
    monkeypatch.setenv("MAXTOKEN_MODELS_DIR", str(tmp_path / "forced"))
    assert dlconfig.resolve_models_dir(str(served)) == tmp_path / "forced"


def test_config_routes(client, tmp_path):
    doc = client.get("/admin/downloads/config").json()
    assert doc["configured"] == {"models_dir": "", "hf_endpoint": "", "ms_endpoint": ""}
    assert doc["system_memory_bytes"] > 0 and isinstance(doc["ms_available"], bool)
    ok = client.post("/admin/downloads/config", json={"hf_endpoint": "https://hf-mirror.com", "models_dir": str(tmp_path / "m")})
    assert ok.status_code == 200 and ok.json()["models_dir"] == str(tmp_path / "m")
    assert client.get("/admin/downloads/config").json()["configured"]["hf_endpoint"] == "https://hf-mirror.com"
    bad = client.post("/admin/downloads/config", json={"hf_endpoint": "not a url"})
    assert bad.status_code == 400 and "http" in bad.json()["detail"]
    assert json.loads((tmp_path / "downloads.json").read_text())["hf_endpoint"] == "https://hf-mirror.com"


def test_task_routes_drive_the_downloader(client, tmp_path):
    fake = FakeDownloader()
    api._DOWNLOADS._downloaders["hf"] = fake
    started = client.post("/admin/downloads/hf/start", json={"repo_id": "mlx-community/x", "token": "hf_t"})
    assert started.json()["task"]["repo_id"] == "mlx-community/x"
    assert fake.calls == [("start", "mlx-community/x", "hf_t")]
    assert fake.model_dir == str(dlconfig.resolve_models_dir(None))
    assert client.post("/admin/downloads/hf/start", json={"repo_id": "dup/dup"}).status_code == 400
    assert {t["task_id"] for t in client.get("/admin/downloads/hf/tasks").json()["tasks"]} == {"t1", "t2"}
    assert client.post("/admin/downloads/hf/cancel/t2").status_code == 200
    assert client.post("/admin/downloads/hf/cancel/nope").status_code == 404
    assert client.post("/admin/downloads/hf/retry/t1", json={"token": "x"}).json()["task"]["task_id"] == "t3"
    assert client.post("/admin/downloads/hf/retry/nope").status_code == 400
    assert client.delete("/admin/downloads/hf/task/t1").status_code == 200
    assert client.delete("/admin/downloads/hf/task/t1").status_code == 404


def test_unknown_source_and_bad_repo_id_never_reach_the_network(client):
    assert client.get("/admin/downloads/xx/tasks").status_code == 404
    for repo in ("no-slash", "a/b/c", ""):
        response = client.post("/admin/downloads/hf/start", json={"repo_id": repo})
        assert response.status_code == 400 and "Invalid repository ID" in response.json()["detail"]
    assert client.get("/admin/downloads/hf/search?q=").status_code == 400
    assert client.get("/admin/downloads/hf/model-info").status_code == 400


def test_a_restart_never_resumes_a_transfer_on_its_own(tmp_path):
    tasks_file = tmp_path / "tasks.json"
    rows = [DownloadTask(task_id=f"t{i}", repo_id=f"o/m{i}", status=status, token="secret").to_dict() | {"token": "secret"}
            for i, status in enumerate([DownloadStatus.DOWNLOADING, DownloadStatus.PENDING, DownloadStatus.FAILED, DownloadStatus.COMPLETED])]
    tasks_file.write_text(json.dumps(rows))

    async def restore():
        downloader = HFDownloader(str(tmp_path / "models"), tasks_file=tasks_file)
        await downloader.restore_tasks()
        return downloader

    downloader = asyncio.run(restore())
    by_id = {task["task_id"]: task for task in downloader.get_tasks()}
    assert set(by_id) == {"t0", "t1", "t2"}  # completed rows are dropped
    assert [by_id[k]["status"] for k in ("t0", "t1", "t2")] == ["cancelled", "cancelled", "failed"]
    assert "Retry to resume" in by_id["t0"]["error"] and by_id["t0"]["speed_bps"] == 0
    assert not downloader._active_tasks  # nothing was started
    # The queue file no longer promises a resume either.
    assert {row["status"] for row in json.loads(tasks_file.read_text())} == {"cancelled", "failed"}


def test_task_tokens_are_never_serialised_to_the_browser():
    task = DownloadTask(task_id="t", repo_id="o/m", token="hf_secret")
    assert "token" not in task.to_dict() and "hf_secret" not in json.dumps(task.to_dict())
