"""Text-to-image for the console chat and ``/v1/images/generations``.

Images come from FLUX.2 Klein through mflux (MIT, https://github.com/filipstrand/mflux),
the MLX implementation of the model. They run in a worker process of their own, not in
the engine's: the language model keeps serving while an image renders, a crash takes
only the image worker down, and stopping the worker hands its ~10 GB back to the
system at once. The worker starts on the first request and stops after
``MAXTOKEN_IMAGE_IDLE_S`` seconds without one, because on a 32 GB machine an idle
image model would otherwise sit next to the language model for good.

Measured on the M1 Max (FLUX.2 Klein 4B, 4-bit, 1024x1024, 4 steps): ~8 s to load,
~27-30 s per image, 10.2 GB peak.

Generated files land in ``~/.maxtoken/images`` (``MAXTOKEN_IMAGE_DIR``) and are
served by id at ``/admin/images/files/<id>.png``; chats store that URL, not the pixels.
"""

from __future__ import annotations

import asyncio
import base64
import multiprocessing as mp
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from maxtoken.utils import init_logger

logger = init_logger(__name__)

IDLE_UNLOAD_S = float(os.environ.get("MAXTOKEN_IMAGE_IDLE_S", "600"))
MAX_REFERENCES = 4
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_REFERENCE_SIDE = 2048
DEFAULT_STEPS = 4
MAX_SIDE = 2048
_ID = re.compile(r"^[0-9a-f]{24}$")


def image_dir() -> Path:
    return Path(os.environ.get("MAXTOKEN_IMAGE_DIR", "~/.maxtoken/images")).expanduser()


def image_path(image_id: str) -> Path | None:
    """The file of a generated image, or None for an id that is not one of ours."""
    if not _ID.match(image_id):
        return None
    path = image_dir() / f"{image_id}.png"
    return path if path.is_file() else None


def parse_size(size: str | None) -> tuple[int, int]:
    """``"1024x768"`` -> (width, height); both a multiple of 16 within 256..2048."""
    if not size:
        return 1024, 1024
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", size)
    if not match:
        raise ValueError(f"size must look like 1024x1024, got {size!r}")
    width, height = int(match.group(1)), int(match.group(2))
    for side in (width, height):
        if not 256 <= side <= MAX_SIDE or side % 16:
            raise ValueError(f"each side must be a multiple of 16 between 256 and {MAX_SIDE}")
    return width, height


def process_footprint(pid: int | None) -> int | None:
    """A process's physical footprint in bytes, as Activity Monitor and ``footprint``
    count it. RSS leaves out the GPU (Metal) memory that holds the model weights."""
    if not pid:
        return None
    try:
        import ctypes

        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        buf = (ctypes.c_uint64 * 64)()
        # RUSAGE_INFO_V2: a 16-byte uuid, then uint64 fields; ri_phys_footprint is the 8th.
        if lib.proc_pid_rusage(int(pid), 2, ctypes.byref(buf)) != 0:
            return None
        return int(buf[2 + 7])
    except (OSError, AttributeError):
        return None


def save_upload(data: bytes) -> dict:
    """Store an uploaded reference image as a PNG under a fresh id.

    It is decoded and re-encoded, never stored as sent: what lands on disk is a
    plain RGB PNG (transparency flattened onto white) at most MAX_REFERENCE_SIDE on
    its long side, without the upload's metadata."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"image is larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    import io

    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            # Transparency becomes white: both pipelines read references as RGB, where a
            # transparent pixel (0, 0, 0, 0) would turn black -- a sprite on a black stage.
            if img.mode in ("RGBA", "LA", "P", "PA"):
                rgba = img.convert("RGBA")
                img = Image.new("RGB", rgba.size, (255, 255, 255))
                img.paste(rgba, mask=rgba.getchannel("A"))
            else:
                img = img.convert("RGB")
            img.thumbnail((MAX_REFERENCE_SIDE, MAX_REFERENCE_SIDE))
            image_id = secrets.token_hex(12)
            path = image_dir() / f"{image_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, format="PNG")
            width, height = img.size
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"not a readable image: {exc}") from exc
    return {"id": image_id, "url": f"/admin/images/files/{image_id}.png", "width": width, "height": height}


def reference_path(ref: str) -> str:
    """An image id or our file URL -> the file on disk; anything else is refused."""
    match = re.search(r"([0-9a-f]{24})(?:\.png)?$", ref or "")
    path = image_path(match.group(1)) if match else None
    if path is None:
        raise ValueError(f"unknown reference image {ref!r}")
    return str(path)


def find_image_models(folders: Iterable[str | Path]) -> list[dict]:
    """FLUX.2 checkpoints in mflux's layout (``transformer/``, ``text_encoder/``,
    ``vae/`` beside each other), smallest first."""
    found: dict[str, dict] = {}
    for folder in folders:
        root = Path(folder).expanduser()
        try:
            dirs = [p for p in root.iterdir() if p.is_dir()]
        except OSError:
            continue
        for d in dirs:
            name = d.name.lower()
            if not ("flux2" in name or "flux-2" in name or "flux.2" in name):
                continue
            if not all((d / part).is_dir() for part in ("transformer", "text_encoder", "vae")):
                continue
            size = sum(f.stat().st_size for f in d.rglob("*.safetensors") if ".cache" not in f.parts)
            found[str(d.resolve())] = {"id": str(d.resolve()), "name": d.name, "size_bytes": size}
    return sorted(found.values(), key=lambda m: (m["size_bytes"], m["name"]))


# -- the worker process ---------------------------------------------------------------


def _worker_main(conn: Any, model_path: str, cancel: Any) -> None:
    """Load one FLUX.2 model, then render jobs from ``conn`` until it closes.

    Messages out: ("ready", load_s) once, then per job ("phase", name),
    ("step", done, total), and finally ("done", meta), ("cancelled",) or ("error", text).
    """
    try:
        t0 = time.monotonic()
        import mlx.core as mx
        from mflux.models.common.config import ModelConfig
        from mflux.models.flux2.variants import Flux2Klein

        # Keep MLX's buffer cache small: between images the memory belongs to the
        # language model next door.
        mx.set_cache_limit(512 * 1024 * 1024)
        model = Flux2Klein(model_path=model_path, model_config=ModelConfig.from_name(model_path))
        # Reference-image generation is mflux's edit variant, which cannot run without a
        # reference (it concatenates their tokens unconditionally). It is the same network,
        # so it is built as a second view on the weights already loaded, not loaded twice.
        from mlx import nn
        from mflux.models.flux2.variants import Flux2KleinEdit

        edit = Flux2KleinEdit.__new__(Flux2KleinEdit)
        nn.Module.__init__(edit)
        for attr in ("prompt_cache", "model_config", "callbacks", "tiling_config", "tokenizers",
                     "vae", "transformer", "text_encoder", "bits", "lora_paths", "lora_scales"):
            setattr(edit, attr, getattr(model, attr))

        class _Progress:
            def call_before_loop(self, seed, prompt, latents, config, **_):
                conn.send(("step", 0, config.num_inference_steps))

            def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
                # Called after step t is built and before it is evaluated: t steps are done.
                if cancel.is_set():
                    raise _Cancelled()
                if t:
                    conn.send(("step", t, config.num_inference_steps))

            def call_after_loop(self, seed, prompt, latents, config):
                conn.send(("step", config.num_inference_steps, config.num_inference_steps))
                conn.send(("phase", "decoding"))

        class _Cancelled(Exception):
            pass

        model.callbacks.register(_Progress())
        conn.send(("ready", time.monotonic() - t0))
    except Exception as exc:  # noqa: BLE001 -- reported to the parent, which surfaces it
        conn.send(("error", f"could not load {os.path.basename(model_path)}: {type(exc).__name__}: {exc}"))
        return

    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        cancel.clear()
        started = time.monotonic()
        try:
            conn.send(("phase", "encoding"))
            common = dict(seed=job["seed"], prompt=job["prompt"], num_inference_steps=job["steps"],
                          width=job["width"], height=job["height"])
            if job.get("references"):
                image = edit.generate_image(image_paths=job["references"], **common)
            else:
                image = model.generate_image(**common)
            path = Path(job["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            image.image.save(path, format="PNG")
            conn.send(("done", {"seconds": round(time.monotonic() - started, 2)}))
        except _Cancelled:
            conn.send(("cancelled",))
        except Exception as exc:  # noqa: BLE001
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            mx.clear_cache()


def _spawn_worker(model_path: str) -> tuple[Any, Any, Any]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    cancel = ctx.Event()
    proc = ctx.Process(target=_worker_main, args=(child, model_path, cancel), name="maxtoken-images", daemon=True)
    proc.start()
    child.close()
    return parent, proc, cancel


class ImageError(Exception):
    def __init__(self, message: str, status: int = 500):
        super().__init__(message)
        self.status = status


class ImageService:
    """One image at a time, from a worker started on demand and stopped when idle."""

    def __init__(self, folders: Callable[[], Iterable[str | Path]],
                 spawn: Callable[[str], tuple[Any, Any, Any]] = _spawn_worker,
                 idle_s: float = IDLE_UNLOAD_S):
        self._folders = folders
        self._spawn = spawn
        self._idle_s = idle_s
        self._lock = asyncio.Lock()
        self._conn = self._proc = self._cancel = None
        self.model: str | None = None
        self.state = "unloaded"  # unloaded | loading | ready | generating
        self.job: dict | None = None  # {"request_id", "phase", "step", "total"}
        self.waiting = 0
        self.last_error: str | None = None
        self._idle_task: asyncio.Task | None = None
        self._abandoned: set[str] = set()
        self.loaded_at: float | None = None   # time.time() the worker became ready
        self.load_seconds: float | None = None
        self._unload_at: float | None = None  # time.time() the idle timer fires

    def models(self) -> list[dict]:
        return find_image_models(self._folders())

    def _pick(self, requested: str | None) -> dict:
        models = self.models()
        if not models:
            raise ImageError("no FLUX.2 image model found; download one (e.g. mflux-community/flux2-klein-4b-mflux-q4) into the models folder", 404)
        if requested:
            for m in models:
                if requested in (m["id"], m["name"]):
                    return m
            raise ImageError(f"{requested} is not a local image model", 404)
        preferred = os.environ.get("MAXTOKEN_IMAGE_MODEL")
        if preferred:
            for m in models:
                if preferred in (m["id"], m["name"]):
                    return m
        if self.model and any(m["id"] == self.model for m in models):
            return next(m for m in models if m["id"] == self.model)
        return models[0]

    def status(self) -> dict:
        pid = getattr(self._proc, "pid", None) if self._proc is not None else None
        now = time.time()
        return {
            "state": self.state, "model": os.path.basename(self.model) if self.model else None,
            "job": dict(self.job) if self.job else None, "waiting": self.waiting,
            "idle_unload_s": self._idle_s, "last_error": self.last_error,
            "pid": pid, "memory_bytes": process_footprint(pid) if self.state != "unloaded" else None,
            "loaded_at": self.loaded_at, "load_seconds": self.load_seconds,
            "unload_in_s": max(0.0, self._unload_at - now) if self._unload_at and self.state == "ready" else None,
            "models": [{"id": m["id"], "name": m["name"], "size_bytes": m["size_bytes"]} for m in self.models()],
        }

    async def _recv(self) -> tuple:
        conn = self._conn
        return await asyncio.to_thread(conn.recv)

    async def _ensure(self, model_id: str) -> None:
        if self._proc is not None and self.model == model_id and self._proc.is_alive():
            return
        await self._stop_worker()
        self.state, self.model = "loading", model_id
        if self.job:
            self.job["phase"] = "loading"
        self._conn, self._proc, self._cancel = self._spawn(model_id)
        try:
            message = await self._recv()
        except (EOFError, OSError):
            message = ("error", "the image worker exited while loading")
        if message[0] != "ready":
            await self._stop_worker()
            raise ImageError(message[1] if len(message) > 1 else "image worker failed to load")
        logger.info("Image model %s loaded in %.1f s", os.path.basename(model_id), message[1])
        self.state = "ready"
        self.loaded_at, self.load_seconds = time.time(), round(float(message[1]), 1)

    async def _stop_worker(self) -> None:
        conn, proc = self._conn, self._proc
        self._conn = self._proc = self._cancel = None
        self.state = "unloaded"
        self.loaded_at = self._unload_at = None
        if conn is not None:
            conn.close()
        if proc is not None:
            await asyncio.to_thread(proc.join, 5)
            if proc.is_alive():
                proc.terminate()
                await asyncio.to_thread(proc.join, 5)

    def _arm_idle(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
        self._unload_at = None
        if self._idle_s <= 0 or self._proc is None:
            return
        self._unload_at = time.time() + self._idle_s

        async def _later():
            await asyncio.sleep(self._idle_s)
            async with self._lock:
                if self.state == "ready":
                    logger.info("Image model idle for %.0f s; unloading it", self._idle_s)
                    await self._stop_worker()

        self._idle_task = asyncio.get_running_loop().create_task(_later())

    def cancel(self, request_id: str) -> bool:
        """Stop a request's image: before it starts if it is still queued, between
        denoising steps if it is rendering."""
        if self.job is not None and self.job.get("request_id") == request_id:
            if self._cancel is not None:
                self._cancel.set()
            return True
        if self.waiting:
            self._abandoned.add(request_id)
        return False

    async def generate(self, prompt: str, *, width: int = 1024, height: int = 1024,
                       steps: int = DEFAULT_STEPS, seed: int | None = None,
                       model: str | None = None, request_id: str | None = None,
                       references: list[str] | None = None) -> dict:
        """Render one image, from the prompt alone or guided by up to MAX_REFERENCES
        reference images (ids of uploaded or generated images). The job runs as its own task: a caller that goes away
        (closed tab, dropped connection) only flags it cancelled, so the worker's
        pipe is never abandoned mid-conversation."""
        if not prompt.strip():
            raise ImageError("prompt must not be empty", 400)
        if not 1 <= steps <= 50:
            raise ImageError("steps must be between 1 and 50", 400)
        if references and len(references) > MAX_REFERENCES:
            raise ImageError(f"at most {MAX_REFERENCES} reference images", 400)
        try:
            ref_paths = [reference_path(r) for r in references or []]
        except ValueError as exc:
            raise ImageError(str(exc), 400) from exc
        chosen = self._pick(model)
        seed = int(seed) if seed is not None else secrets.randbelow(2**31)
        job = {"prompt": prompt, "width": width, "height": height, "steps": steps, "seed": seed,
               "references": ref_paths}
        token = request_id or secrets.token_hex(8)
        task = asyncio.get_running_loop().create_task(self._run(chosen, job, token))
        # Nobody may be left to await it (see below); fetch its outcome so asyncio
        # does not log it as never retrieved.
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancel(token)
            raise

    async def _run(self, chosen: dict, job: dict, request_id: str) -> dict:
        self.waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self.waiting -= 1
        try:
            if request_id in self._abandoned:
                self._abandoned.discard(request_id)
                raise ImageError("cancelled", 499)
            self.job = {"request_id": request_id, "phase": "loading", "step": 0, "total": job["steps"]}
            await self._ensure(chosen["id"])
            image_id = secrets.token_hex(12)
            path = image_dir() / f"{image_id}.png"
            self.state = "generating"
            self._conn.send({**job, "path": str(path)})
            while True:
                try:
                    message = await self._recv()
                except (EOFError, OSError):
                    await self._stop_worker()
                    raise ImageError("the image worker died while rendering (out of memory?)")
                kind = message[0]
                if kind == "phase":
                    self.job["phase"] = message[1]
                elif kind == "step":
                    self.job.update(phase="denoising", step=message[1], total=message[2])
                elif kind == "done":
                    meta = message[1]
                    break
                elif kind == "cancelled":
                    raise ImageError("cancelled", 499)
                else:
                    raise ImageError(message[1] if len(message) > 1 else "image generation failed")
            self.last_error = None
            return {"id": image_id, "path": str(path), "url": f"/admin/images/files/{image_id}.png",
                    "seed": job["seed"], "width": job["width"], "height": job["height"],
                    "steps": job["steps"], "seconds": meta.get("seconds"), "model": chosen["name"],
                    "prompt": job["prompt"], "references": len(job["references"])}
        except ImageError as exc:
            if exc.status != 499:
                self.last_error = str(exc)
            raise
        finally:
            self.job = None
            if self._proc is not None:
                self.state = "ready"
            self._lock.release()
            self._arm_idle()

    async def load(self, model: str | None = None) -> dict:
        """Start the worker now (the console's Load button) instead of on the first image."""
        chosen = self._pick(model)
        async with self._lock:
            try:
                await self._ensure(chosen["id"])
            except ImageError as exc:
                self.last_error = str(exc)
                raise
            self._arm_idle()
        return self.status()

    async def unload(self) -> dict:
        """Stop the worker and give its memory back. A render in progress finishes first."""
        async with self._lock:
            if self._idle_task is not None:
                self._idle_task.cancel()
            await self._stop_worker()
        return self.status()

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
        await self._stop_worker()


def b64_png(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")
