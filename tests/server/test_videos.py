"""Video jobs: sizes and frame grid, model discovery, the job lifecycle with a fake
worker, and the progress parser fed the pipeline's real output formats."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import threading
import time

import pytest

from maxtoken.server import videos as vd


def test_frames_follow_the_8k_plus_1_grid():
    assert vd.frames_for(4) == 97 and vd.frames_for(5) == 121 and vd.frames_for(1) == 25
    assert all((vd.frames_for(s) - 1) % 8 == 0 for s in (1, 2, 3, 7, 10))


def test_parse_video_size():
    assert vd.parse_video_size(None) == (768, 512)
    assert vd.parse_video_size("512x768") == (512, 768)
    for bad in ("700x500", "2048x512", "x"):
        with pytest.raises(ValueError):
            vd.parse_video_size(bad)


def test_finds_ltx_packs(tmp_path):
    good = tmp_path / "ltx-2.3-mlx-q4"
    good.mkdir()
    (good / "transformer-distilled-1.1.safetensors").write_bytes(b"x" * 5)
    (good / "vae_decoder.safetensors").write_bytes(b"x" * 5)
    (tmp_path / "ltx-partial").mkdir()
    (tmp_path / "ltx-partial" / "vae_decoder.safetensors").write_bytes(b"x")
    assert [m["name"] for m in vd.find_video_models([tmp_path])] == ["ltx-2.3-mlx-q4"]


class _Conn:
    def __init__(self):
        self.sent = []

    def send(self, message):
        self.sent.append(message)


def test_progress_stream_reads_phases_and_two_denoise_stages():
    conn = _Conn()
    stream = vd._ProgressStream(conn, open(os.devnull, "w"))
    stream.write("[Loading text encoder (Gemma)] ...\n")
    stream.write("[Loading text encoder (Gemma)] done in 4.2s\n")
    for done in range(9):
        stream.write(f"\rDenoising:  {done * 12}%|███       | {done}/8 [00:{done:02d}<00:10,  1.2s/it]")
    stream.write("\n[Upscaling] ...\n")
    for done in range(4):
        stream.write(f"\rDenoising (ancestral): {done * 33}%|██ | {done}/3 [00:01<00:02]")
    assert ("phase", "Loading text encoder (Gemma)") in conn.sent
    assert ("phase", "Upscaling") in conn.sent
    steps = [m for m in conn.sent if m[0] == "step"]
    assert steps[0] == ("step", 1, 0, 8) and ("step", 1, 8, 8) in steps
    assert steps[-1] == ("step", 2, 3, 3)


def _fake_spawn(log, *, fail=False, slow=False):
    def spawn(job):
        parent, child = mp.Pipe(duplex=False)
        stop = threading.Event()

        def run():
            log.append(job)
            child.send(("phase", "Loading the video model"))
            for stage, total in ((1, 8), (2, 3)):
                for step in range(total + 1):
                    if stop.is_set():
                        child.close()
                        return
                    child.send(("step", stage, step, total))
                    time.sleep(0.05 if slow else 0.002)
            if fail:
                child.send(("error", "MemoryError: boom"))
            else:
                os.makedirs(os.path.dirname(job["path"]), exist_ok=True)
                open(job["path"], "wb").write(b"mp4")
                child.send(("done", {"seconds": 0.1}))
            child.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        class _Proc:
            def is_alive(self):
                return thread.is_alive()

            def join(self, timeout=None):
                thread.join(timeout)

            def terminate(self):
                stop.set()

            def kill(self):
                stop.set()

        return parent, _Proc()

    return spawn


@pytest.fixture
def pack(tmp_path, monkeypatch):
    monkeypatch.setenv("MAXTOKEN_VIDEO_DIR", str(tmp_path / "out"))
    d = tmp_path / "models" / "ltx-2.3-mlx-q4"
    d.mkdir(parents=True)
    (d / "transformer-distilled-1.1.safetensors").write_bytes(b"x")
    (d / "vae_decoder.safetensors").write_bytes(b"x")
    g = tmp_path / "models" / "gemma-3-12b-it-4bit"
    g.mkdir()
    (g / "config.json").write_text("{}")
    (g / "model.safetensors.index.json").write_text('{"weight_map": {"a": "model-00001-of-00001.safetensors"}}')
    (g / "model-00001-of-00001.safetensors").write_bytes(b"x")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return tmp_path / "models"


async def _wait(service, job_id, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = service.get(job_id)
        if job["status"] in ("done", "error", "cancelled"):
            return job
        await asyncio.sleep(0.01)
    raise AssertionError(service.get(job_id))


def test_a_job_runs_to_a_served_file(pack):
    log = []
    service = vd.VideoService(lambda: [pack], spawn=_fake_spawn(log), memory=lambda: 40 * 2**30)

    async def run():
        job = service.submit("a fox", width=768, height=512, seconds=4, seed=9)
        assert job["status"] == "queued" and job["frames"] == 97 and "_model_path" not in job
        return await _wait(service, job["id"])

    job = asyncio.run(run())
    assert job["status"] == "done" and job["url"] == f"/admin/videos/files/{job['id']}.mp4"
    assert vd.video_path(job["id"]) is not None and vd.video_path("../x") is None
    assert log[0]["frames"] == 97 and log[0]["seed"] == 9 and log[0]["low_ram"] is False
    assert job["stage"] == 2 and job["step"] == 3


def test_little_free_memory_streams_the_weights(pack):
    log = []
    service = vd.VideoService(lambda: [pack], spawn=_fake_spawn(log), memory=lambda: 8 * 2**30)

    async def run():
        return await _wait(service, service.submit("x", width=512, height=512, seconds=2)["id"])

    assert asyncio.run(run())["low_ram"] is True and log[0]["low_ram"] is True


def test_jobs_queue_and_cancel(pack):
    log = []
    service = vd.VideoService(lambda: [pack], spawn=_fake_spawn(log, slow=True), memory=lambda: None)

    async def run():
        first = service.submit("one", width=512, height=512, seconds=2)
        second = service.submit("two", width=512, height=512, seconds=2)
        await asyncio.sleep(0.15)
        assert service.get(second["id"])["status"] == "queued"
        assert service.cancel(second["id"]) is True
        assert service.cancel(first["id"]) is True
        a = await _wait(service, first["id"])
        b = await _wait(service, second["id"])
        return a, b

    a, b = asyncio.run(run())
    assert a["status"] == "cancelled" and b["status"] == "cancelled"
    assert [j["prompt"] for j in log] == ["one"], "a cancelled queued job never starts"


def test_worker_errors_and_missing_models(pack, tmp_path):
    service = vd.VideoService(lambda: [pack], spawn=_fake_spawn([], fail=True), memory=lambda: None)

    async def run():
        return await _wait(service, service.submit("x", width=512, height=512, seconds=2)["id"])

    job = asyncio.run(run())
    assert job["status"] == "error" and "MemoryError" in job["error"]

    empty = vd.VideoService(lambda: [tmp_path / "nothing"], spawn=_fake_spawn([]))
    with pytest.raises(vd.VideoError) as err:
        empty.submit("x", width=512, height=512, seconds=2)  # refused before any job is queued
    assert err.value.status == 404


def test_a_local_gemma_wins_over_the_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert vd.find_text_encoder([tmp_path]) is None, "never point at an uncached hub id"
    snap = tmp_path / "hub" / "models--mlx-community--gemma-3-12b-it-4bit" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert vd.find_text_encoder([tmp_path]) == vd.GEMMA_REPO
    g = tmp_path / "gemma-3-12b-it-4bit"
    g.mkdir()
    (g / "config.json").write_text("{}")
    assert vd.find_text_encoder([tmp_path]) == vd.GEMMA_REPO, "no weights yet: keep the hub id"
    (g / "model.safetensors.index.json").write_text('{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}')
    (g / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert vd.find_text_encoder([tmp_path]) == vd.GEMMA_REPO, "half downloaded: keep the hub id"
    (g / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert vd.find_text_encoder([tmp_path]) == str(g.resolve())


def test_a_missing_text_encoder_fails_the_job_instead_of_downloading(pack):
    import shutil

    shutil.rmtree(pack / "gemma-3-12b-it-4bit")
    log = []
    service = vd.VideoService(lambda: [pack], spawn=_fake_spawn(log), memory=lambda: None)

    async def run():
        return await _wait(service, service.submit("x", width=512, height=512, seconds=2)["id"])

    job = asyncio.run(run())
    assert job["status"] == "error" and "not downloaded" in job["error"] and log == []
