"""Text- and image-to-video for the console chat, on LTX-2 via ltx-2-mlx.

ltx-2-mlx (MIT, https://github.com/dgrauet/ltx-2-mlx) is the MLX port of Lightricks'
LTX-2.x audio-video model (LTX-2 Community License: free below USD 10M annual
revenue). We run its distilled two-stage pipeline: half resolution, a 2x latent
upscale, then a short refine at full size, with a soundtrack generated alongside.

A render takes minutes, so it is a job, not a request:
- ``POST /admin/videos`` queues it and returns its id.
- ``GET /admin/videos/<id>`` reports phase and step until it is done or failed.
- The file is then served from ``/admin/videos/files/<id>.mp4``.

The chat stores the job id, so a reload resumes watching instead of losing it.

Every job gets a fresh worker process that exits when the job ends:
- The pipeline peaks around its transformer (11 GB at 4-bit) plus Gemma before that.
- A process that is gone is the only guarantee that all of it goes back to the
  language model next door.
- Loading costs seconds, rendering minutes.
- Cancelling is simply ending that process.

When less than MAXTOKEN_VIDEO_STREAM_BELOW_GB (default 20) of memory is available
at the start of a job, the transformer streams its blocks from disk
(``low_ram_streaming``). That is slower, but it fits beside a loaded language model.
"""

from __future__ import annotations

import asyncio
import io
import json
import multiprocessing as mp
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from maxtoken.utils import init_logger

logger = init_logger(__name__)

FRAME_RATE = 24
MAX_SECONDS = 10
STREAM_BELOW_GB = float(os.environ.get("MAXTOKEN_VIDEO_STREAM_BELOW_GB", "20"))
_ID = re.compile(r"^[0-9a-f]{24}$")


def video_dir() -> Path:
    return Path(os.environ.get("MAXTOKEN_VIDEO_DIR", "~/.maxtoken/videos")).expanduser()


def video_path(video_id: str) -> Path | None:
    if not _ID.match(video_id):
        return None
    path = video_dir() / f"{video_id}.mp4"
    return path if path.is_file() else None


def frames_for(seconds: float) -> int:
    """LTX frame counts live on the 8k+1 grid (the VAE compresses time 8x)."""
    k = max(1, round(seconds * FRAME_RATE / 8))
    return 8 * k + 1


def parse_video_size(size: str | None) -> tuple[int, int]:
    """``"768x512"`` -> (width, height); multiples of 64 within 256..1280."""
    if not size:
        return 768, 512
    match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", size)
    if not match:
        raise ValueError(f"size must look like 768x512, got {size!r}")
    width, height = int(match.group(1)), int(match.group(2))
    for side in (width, height):
        if not 256 <= side <= 1280 or side % 64:
            raise ValueError("each side must be a multiple of 64 between 256 and 1280")
    return width, height


def find_video_models(folders: Iterable[str | Path]) -> list[dict]:
    """LTX-2.x packs in ltx-2-mlx's layout: a distilled transformer beside a VAE decoder."""
    found: dict[str, dict] = {}
    for folder in folders:
        root = Path(folder).expanduser()
        try:
            dirs = [p for p in root.iterdir() if p.is_dir() and "ltx" in p.name.lower()]
        except OSError:
            continue
        for d in dirs:
            transformers = [p for p in d.glob("transformer-distilled*.safetensors")]
            decoders = list(d.glob("vae_decoder*.safetensors"))
            if not transformers or not decoders:
                continue
            size = sum(f.stat().st_size for f in d.glob("*.safetensors"))
            found[str(d.resolve())] = {"id": str(d.resolve()), "name": d.name, "size_bytes": size}
    return sorted(found.values(), key=lambda m: m["name"])


GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"


def find_text_encoder(folders: Iterable[str | Path]) -> str | None:
    """LTX-2.3 encodes prompts with Gemma 3 12B (LTX-2.5 packs ship their own): a complete
    local copy in a models folder, else the hub id when the hub cache already holds it,
    else None. Never a hub id that is not cached: the pipeline would then fetch 8 GB at
    full line speed in the middle of a video job."""
    for folder in folders:
        from .model_switch import missing_shards

        candidate = Path(folder).expanduser() / GEMMA_REPO.split("/")[1]
        if not (candidate / "config.json").is_file():
            continue
        has_weights, missing = missing_shards(candidate)
        if has_weights and not missing:
            return str(candidate.resolve())
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub")
    snapshots = Path(hub).expanduser() / ("models--" + GEMMA_REPO.replace("/", "--")) / "snapshots"
    if any((snap / "config.json").exists() for snap in snapshots.glob("*")):
        return GEMMA_REPO
    return None


def available_memory_bytes() -> int | None:
    """Free + inactive + purgeable + speculative pages: what a new process can get
    without the system compressing or swapping someone else."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    page = int(re.search(r"page size of (\d+)", out).group(1)) if "page size of" in out else 16384
    total = 0
    for key in ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            total += int(m.group(1))
    return total * page


# -- worker process ---------------------------------------------------------------------


class _ProgressStream(io.TextIOBase):
    """Stands in for stderr in the worker and turns the pipeline's own progress output
    (``[phase] ...`` lines and tqdm bars) into messages for the parent."""

    _BAR = re.compile(r"([A-Za-z][^:|\r\n]*):\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)")

    def __init__(self, conn: Any, original: Any):
        self._conn, self._original = conn, original
        self._stage, self._last = 0, None

    def write(self, text: str) -> int:
        try:
            self._original.write(text)
        except Exception:  # noqa: BLE001
            pass
        for part in re.split(r"[\r\n]", text):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"\[(.+?)\] \.\.\.$", part)
            if m:
                self._conn.send(("phase", m.group(1)))
                continue
            m = self._BAR.search(part)
            if m and "enois" in m.group(1):
                done, total = int(m.group(2)), int(m.group(3))
                if done == 0 and self._last != (0, total):
                    self._stage += 1
                self._last = (done, total)
                self._conn.send(("step", self._stage, done, total))
        return len(text)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:  # noqa: BLE001
            pass


def _worker_main(conn: Any, job: dict) -> None:
    started = time.monotonic()
    sys.stderr = _ProgressStream(conn, sys.__stderr__)
    try:
        from ltx_pipelines_mlx.distilled import DistilledPipeline

        conn.send(("phase", "Loading the video model"))
        pipe = DistilledPipeline(model_dir=job["model"], gemma_model_id=job["gemma"], low_memory=True,
                                 low_ram_streaming=job["low_ram"])
        pipe.verbose = True
        pipe.generate_audio = True
        Path(job["path"]).parent.mkdir(parents=True, exist_ok=True)
        kwargs = dict(prompt=job["prompt"], output_path=job["path"], height=job["height"], width=job["width"],
                      num_frames=job["frames"], frame_rate=FRAME_RATE, seed=job["seed"])
        if job.get("image"):
            kwargs["image"] = job["image"]
        pipe.generate_and_save(**kwargs)
        conn.send(("done", {"seconds": round(time.monotonic() - started, 1)}))
    except Exception as exc:  # noqa: BLE001 -- reported to the parent
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _spawn_worker(job: dict) -> tuple[Any, Any]:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker_main, args=(child, job), name="maxtoken-video", daemon=True)
    proc.start()
    child.close()
    return parent, proc


# -- jobs -------------------------------------------------------------------------------


class VideoError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class VideoService:
    """A queue of video jobs, run one at a time, each in a worker of its own."""

    def __init__(self, folders: Callable[[], Iterable[str | Path]],
                 spawn: Callable[[dict], tuple[Any, Any]] = _spawn_worker,
                 memory: Callable[[], int | None] = available_memory_bytes):
        self._folders = folders
        self._spawn = spawn
        self._memory = memory
        self.jobs: dict[str, dict] = {}
        self._order: list[str] = []
        self._queue: asyncio.Queue | None = None
        self._runner: asyncio.Task | None = None
        self._current: tuple[str, Any] | None = None  # (job id, process)

    def models(self) -> list[dict]:
        return find_video_models(self._folders())

    def status(self) -> dict:
        running = self._current[0] if self._current else None
        return {
            "models": self.models(),
            "running": self._public(self.jobs[running]) if running else None,
            "queued": sum(1 for j in self.jobs.values() if j["status"] == "queued"),
        }

    def get(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        return self._public(job) if job else None

    def submit(self, prompt: str, *, width: int, height: int, seconds: float,
               seed: int | None = None, image: str | None = None, model: str | None = None) -> dict:
        if not prompt.strip():
            raise VideoError("prompt must not be empty")
        if not 1 <= seconds <= MAX_SECONDS:
            raise VideoError(f"seconds must be between 1 and {MAX_SECONDS}")
        models = self.models()
        if not models:
            raise VideoError("no LTX-2 video model found; download one (e.g. dgrauet/ltx-2.3-mlx-q4) into the models folder", 404)
        chosen = next((m for m in models if model in (m["id"], m["name"])), None) if model else models[0]
        if chosen is None:
            raise VideoError(f"{model} is not a local video model", 404)
        job_id = secrets.token_hex(12)
        frames = frames_for(seconds)
        job = {
            "id": job_id, "status": "queued", "phase": "Waiting for the video model", "stage": 0,
            "step": 0, "total": 0, "prompt": prompt, "width": width, "height": height, "frames": frames,
            "seconds": round(frames / FRAME_RATE, 2), "seed": int(seed) if seed is not None else secrets.randbelow(2**31),
            "image": image, "model": chosen["name"], "created": time.time(), "started": None,
            "finished": None, "elapsed": None, "error": None, "url": None, "low_ram": None,
            "_model_path": chosen["id"],
        }
        self.jobs[job_id] = job
        self._order.append(job_id)
        self._trim()
        if self._queue is None:
            self._queue = asyncio.Queue()
        self._queue.put_nowait(job_id)
        if self._runner is None or self._runner.done():
            self._runner = asyncio.get_running_loop().create_task(self._run_queue())
        return self._public(job)

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job["status"] in ("done", "error", "cancelled"):
            return False
        if self._current and self._current[0] == job_id:
            self._current[1].terminate()
        job.update(status="cancelled", phase="Cancelled", finished=time.time())
        return True

    def _trim(self, keep: int = 200) -> None:
        while len(self._order) > keep:
            old = self._order.pop(0)
            if self.jobs.get(old, {}).get("status") in ("done", "error", "cancelled"):
                self.jobs.pop(old, None)

    @staticmethod
    def _public(job: dict) -> dict:
        return {k: v for k, v in job.items() if not k.startswith("_")}

    async def _run_queue(self) -> None:
        while self._queue is not None and not self._queue.empty():
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job["status"] != "queued":
                continue
            await self._run(job)

    async def _run(self, job: dict) -> None:
        available = self._memory()
        low_ram = available is not None and available < STREAM_BELOW_GB * 2**30
        path = video_dir() / f"{job['id']}.mp4"
        job.update(status="running", started=time.time(), low_ram=low_ram,
                   phase="Starting" + (" (streaming weights: little free memory)" if low_ram else ""))
        gemma = await asyncio.to_thread(find_text_encoder, list(self._folders()))
        if gemma is None:
            job.update(status="error", error=f"the text encoder {GEMMA_REPO} is not downloaded yet",
                       started=time.time(), finished=time.time(), elapsed=0.0)
            return
        conn, proc = self._spawn({"model": job["_model_path"], "gemma": gemma, "prompt": job["prompt"], "width": job["width"],
                                  "height": job["height"], "frames": job["frames"], "seed": job["seed"],
                                  "image": job["image"], "path": str(path), "low_ram": low_ram})
        self._current = (job["id"], proc)
        try:
            while True:
                try:
                    message = await asyncio.to_thread(conn.recv)
                except (EOFError, OSError):
                    if job["status"] != "cancelled":
                        job.update(status="error", error="the video worker exited unexpectedly (out of memory?)")
                    break
                kind = message[0]
                if kind == "phase":
                    job["phase"] = message[1]
                elif kind == "step":
                    job.update(stage=message[1], step=message[2], total=message[3],
                               phase=f"Stage {message[1]} · step {message[2]} of {message[3]}")
                elif kind == "done":
                    job.update(status="done", phase="Done", url=f"/admin/videos/files/{job['id']}.mp4")
                    break
                elif kind == "error":
                    job.update(status="error", error=message[1])
                    break
        finally:
            self._current = None
            conn.close()
            await asyncio.to_thread(proc.join, 10)
            if proc.is_alive():
                proc.kill()
            job["finished"] = time.time()
            job["elapsed"] = round(job["finished"] - job["started"], 1)
            logger.info("Video %s %s after %.0f s%s", job["id"], job["status"], job["elapsed"],
                        f": {job['error']}" if job.get("error") else "")

    async def shutdown(self) -> None:
        if self._current:
            self._current[1].terminate()
        if self._runner:
            self._runner.cancel()
