"""Image generation service: size parsing, model discovery, and the job protocol with
the worker process -- driven by a fake worker that speaks the same pipe protocol, so no
image model is loaded."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import threading
import time

import pytest

from maxmlx.server import images as im


def test_parse_size():
    assert im.parse_size(None) == (1024, 1024)
    assert im.parse_size("768x1024") == (768, 1024)
    for bad in ("1024", "1000x1000", "4096x4096", "axb"):
        with pytest.raises(ValueError):
            im.parse_size(bad)


def test_finds_flux2_checkpoints_in_mflux_layout(tmp_path):
    good = tmp_path / "flux2-klein-4b-mflux-q4"
    for part in ("transformer", "text_encoder", "vae"):
        (good / part).mkdir(parents=True)
        (good / part / "0.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "flux2-partial" / "vae").mkdir(parents=True)
    (tmp_path / "Qwen3.6-35B").mkdir()
    found = im.find_image_models([tmp_path])
    assert [m["name"] for m in found] == ["flux2-klein-4b-mflux-q4"] and found[0]["size_bytes"] == 30


def _fake_spawn(log, *, fail_load=False, die=False):
    """A worker thread on one end of a real pipe, with a real mp.Event for cancel."""

    def spawn(model_path):
        parent, child = mp.Pipe()
        cancel = mp.Event()

        def run():
            log.append(("spawn", model_path))
            if fail_load:
                child.send(("error", "could not load"))
                return
            child.send(("ready", 0.01))
            while True:
                try:
                    job = child.recv()
                except (EOFError, OSError):
                    return
                if die:
                    child.close()
                    return
                child.send(("phase", "encoding"))
                for step in range(job["steps"] + 1):
                    if cancel.is_set():
                        cancel.clear()
                        child.send(("cancelled",))
                        break
                    child.send(("step", step, job["steps"]))
                    time.sleep(0.02)
                else:
                    child.send(("phase", "decoding"))
                    os.makedirs(os.path.dirname(job["path"]), exist_ok=True)
                    open(job["path"], "wb").write(b"\x89PNG fake")
                    log.append(("job", job["prompt"], job["seed"], job["width"], job["height"]))
                    if job.get("references"):
                        log.append(("refs", tuple(job["references"])))
                    child.send(("done", {"seconds": 0.1}))

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        class _Proc:
            def is_alive(self):
                return thread.is_alive()

            def join(self, timeout=None):
                thread.join(timeout)

            def terminate(self):
                pass

        return parent, _Proc(), cancel

    return spawn


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MAXMLX_IMAGE_DIR", str(tmp_path / "out"))
    d = tmp_path / "models" / "flux2-klein-4b-mflux-q4"
    for part in ("transformer", "text_encoder", "vae"):
        (d / part).mkdir(parents=True)
    return tmp_path / "models"


def test_generate_reports_progress_and_serves_the_file(model_dir):
    log = []
    service = im.ImageService(lambda: [model_dir], spawn=_fake_spawn(log), idle_s=0)
    seen = []

    async def run():
        task = asyncio.create_task(service.generate("a fox", width=768, height=512, steps=4, seed=7, request_id="r1"))
        while not task.done():
            if service.job:
                seen.append((service.job["phase"], service.job["step"]))
            await asyncio.sleep(0.01)
        return task.result()

    result = asyncio.run(run())
    assert result["seed"] == 7 and (result["width"], result["height"]) == (768, 512)
    assert result["url"] == f"/admin/images/files/{result['id']}.png"
    assert im.image_path(result["id"]) is not None and im.image_path("../etc/passwd") is None
    assert ("job", "a fox", 7, 768, 512) in log
    assert any(phase == "denoising" and step >= 2 for phase, step in seen), seen
    assert service.state == "ready" and service.job is None


def test_the_worker_is_reused_and_unloaded_when_idle(model_dir):
    log = []
    service = im.ImageService(lambda: [model_dir], spawn=_fake_spawn(log), idle_s=0.2)

    async def run():
        await service.generate("one", steps=1)
        await service.generate("two", steps=1)
        assert [e for e in log if e[0] == "spawn"].__len__() == 1
        await asyncio.sleep(0.4)
        return service.state

    assert asyncio.run(run()) == "unloaded"


def test_cancel_while_rendering_and_while_queued(model_dir):
    log = []
    service = im.ImageService(lambda: [model_dir], spawn=_fake_spawn(log), idle_s=0)

    async def run():
        first = asyncio.create_task(service.generate("slow", steps=40, request_id="a"))
        queued = asyncio.create_task(service.generate("never", steps=1, request_id="b"))
        await asyncio.sleep(0.15)
        assert service.cancel("b") is False  # still queued: dropped before it starts
        assert service.cancel("a") is True
        outcomes = await asyncio.gather(first, queued, return_exceptions=True)
        return outcomes

    first, queued = asyncio.run(run())
    assert isinstance(first, im.ImageError) and first.status == 499
    assert isinstance(queued, im.ImageError) and queued.status == 499
    assert not any(e[0] == "job" for e in log)


def test_load_failure_and_worker_death_are_reported(model_dir):
    service = im.ImageService(lambda: [model_dir], spawn=_fake_spawn([], fail_load=True), idle_s=0)
    with pytest.raises(im.ImageError, match="could not load"):
        asyncio.run(service.generate("x", steps=1))
    assert service.state == "unloaded" and service.last_error

    service = im.ImageService(lambda: [model_dir], spawn=_fake_spawn([], die=True), idle_s=0)
    with pytest.raises(im.ImageError, match="died"):
        asyncio.run(service.generate("x", steps=1))
    assert service.state == "unloaded"


def test_no_model_is_a_404(tmp_path):
    service = im.ImageService(lambda: [tmp_path], spawn=_fake_spawn([]), idle_s=0)
    with pytest.raises(im.ImageError) as err:
        asyncio.run(service.generate("x"))
    assert err.value.status == 404


def test_load_and_unload_on_request_and_report_the_countdown(model_dir):
    log = []
    service = im.ImageService(lambda: [model_dir], spawn=_fake_spawn(log), idle_s=60)

    async def run():
        st = await service.load()
        assert st["state"] == "ready" and st["loaded_at"] and st["load_seconds"] == 0.0
        assert 55 < st["unload_in_s"] <= 60, st["unload_in_s"]
        await service.load()  # already loaded: no second worker
        assert sum(1 for e in log if e[0] == "spawn") == 1
        st = await service.unload()
        assert st["state"] == "unloaded" and st["unload_in_s"] is None and st["memory_bytes"] is None

    asyncio.run(run())


def test_footprint_of_this_process_is_measured():
    assert im.process_footprint(os.getpid()) > 10 * 2**20
    assert im.process_footprint(None) is None


def _png(width, height, color=(200, 30, 30)):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_uploads_are_reencoded_bounded_and_addressable(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    monkeypatch.setenv("MAXMLX_IMAGE_DIR", str(tmp_path))
    small = im.save_upload(_png(300, 200))
    assert (small["width"], small["height"]) == (300, 200)
    assert small["url"] == f"/admin/images/files/{small['id']}.png"
    assert im.image_path(small["id"]).read_bytes()[:4] == b"\x89PNG"  # stored as PNG, not as sent
    big = im.save_upload(_png(4096, 1024))
    assert (big["width"], big["height"]) == (2048, 512)
    with pytest.raises(ValueError):
        im.save_upload(b"not an image at all")
    assert im.reference_path(small["url"]) == str(im.image_path(small["id"]))
    assert im.reference_path(small["id"]) == str(im.image_path(small["id"]))
    for bad in ("/etc/passwd", "0" * 24, "../" + small["id"] + "x"):
        with pytest.raises(ValueError):
            im.reference_path(bad)


def test_references_reach_the_worker(model_dir, tmp_path):
    pytest.importorskip("PIL")
    ref = im.save_upload(_png(64, 64))
    log = []
    service = im.ImageService(lambda: [model_dir], spawn=_fake_spawn(log), idle_s=0)
    result = asyncio.run(service.generate("make it blue", steps=1, references=[ref["id"]]))
    assert result["references"] == 1
    assert ("refs", (str(im.image_path(ref["id"])),)) in log
    with pytest.raises(im.ImageError) as err:
        asyncio.run(service.generate("x", steps=1, references=["deadbeef" * 3]))
    assert err.value.status == 400
    with pytest.raises(im.ImageError, match="at most 4"):
        asyncio.run(service.generate("x", steps=1, references=[ref["id"]] * 5))


def test_transparent_uploads_are_flattened_onto_white(tmp_path, monkeypatch):
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    monkeypatch.setenv("MAXMLX_IMAGE_DIR", str(tmp_path))
    sprite = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    sprite.paste((200, 20, 20, 255), (10, 10, 30, 30))
    buf = io.BytesIO()
    sprite.save(buf, format="PNG")
    stored = Image.open(im.image_path(im.save_upload(buf.getvalue())["id"]))
    assert stored.mode == "RGB"
    assert stored.getpixel((2, 2)) == (255, 255, 255) and stored.getpixel((20, 20)) == (200, 20, 20)
