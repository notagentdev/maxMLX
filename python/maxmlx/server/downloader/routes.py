"""HTTP routes of the console's Downloads tab: ``/admin/downloads/...``.

One route set serves both sources (``hf`` = Hugging Face, ``ms`` = ModelScope); the downloader
objects are created on first use so a server that never opens the tab pays nothing.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import (
    DownloadConfig,
    load_config,
    resolve_models_dir,
    save_config,
    tasks_path,
)

logger = logging.getLogger(__name__)

SOURCES = ("hf", "ms")
TIMEOUT_DETAIL = "{name} API request timed out. The service may be temporarily unavailable."


class StartRequest(BaseModel):
    repo_id: str
    token: str = ""


class RetryRequest(BaseModel):
    token: str = ""


class ConfigPatch(BaseModel):
    models_dir: Optional[str] = None
    hf_endpoint: Optional[str] = None
    ms_endpoint: Optional[str] = None


def system_memory_bytes() -> int:
    """Physical memory; recommendations hide models that could never fit."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 16 * 1024**3


class DownloadService:
    """Owns the two downloaders and re-points them when the models folder changes."""

    def __init__(self, get_served_model_path: Callable[[], Optional[str]]):
        self._served = get_served_model_path
        self._downloaders: dict[str, Any] = {}
        self._restored = False
        self._lock = asyncio.Lock()

    def models_dir(self):
        return resolve_models_dir(self._served())

    def _build(self, source: str):
        if source == "hf":
            from .hf import HFDownloader

            return HFDownloader(str(self.models_dir()), tasks_file=tasks_path())
        if not ms_available():
            return None
        from .ms import MS_SDK_AVAILABLE, MSDownloader

        if not MS_SDK_AVAILABLE:
            return None
        return MSDownloader(str(self.models_dir()), tasks_file=tasks_path().with_name("download-tasks-ms.json"))

    async def get(self, source: str):
        if source not in SOURCES:
            raise HTTPException(status_code=404, detail=f"Unknown source: {source}")
        async with self._lock:
            if source not in self._downloaders:
                downloader = self._build(source)
                if downloader is not None:
                    await downloader.restore_tasks()
                self._downloaders[source] = downloader
            downloader = self._downloaders[source]
        if downloader is None:
            raise HTTPException(
                status_code=503,
                detail='ModelScope support is not installed. Run: pip install "maxmlx[modelscope]"',
            )
        downloader.update_model_dir(str(self.models_dir()))
        return downloader

    async def shutdown(self) -> None:
        for downloader in self._downloaders.values():
            if downloader is not None:
                await downloader.shutdown()


def _upstream_error(name: str, error: Exception) -> HTTPException:
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return HTTPException(status_code=504, detail=TIMEOUT_DETAIL.format(name=name))
    return HTTPException(status_code=502, detail=str(error))


def ms_available() -> bool:
    # find_spec: importing the SDK itself takes seconds, the config call must stay instant.
    import importlib.util

    return importlib.util.find_spec("modelscope") is not None


def register_download_routes(
    app: FastAPI, get_served_model_path: Callable[[], Optional[str]]
) -> DownloadService:
    service = DownloadService(get_served_model_path)
    names = {"hf": "HuggingFace", "ms": "ModelScope"}

    @app.get("/admin/downloads/config")
    async def get_config():
        config = load_config()
        return {
            "models_dir": str(service.models_dir()),
            "configured": {
                "models_dir": config.models_dir,
                "hf_endpoint": config.hf_endpoint,
                "ms_endpoint": config.ms_endpoint,
            },
            "env_models_dir": bool(os.environ.get("MAXMLX_MODELS_DIR")),
            "ms_available": ms_available(),
            "system_memory_bytes": system_memory_bytes(),
        }

    @app.post("/admin/downloads/config")
    async def set_config(patch: ConfigPatch):
        try:
            config: DownloadConfig = save_config(patch.model_dump(exclude_none=True))
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        # Endpoint resolution is cached per process; a changed mirror must be probed afresh.
        from . import hf

        hf._endpoint_resolution_cache.clear()
        return {"success": True, "models_dir": str(service.models_dir()), "configured": config.__dict__}

    @app.post("/admin/downloads/{source}/start")
    async def start(source: str, request: StartRequest):
        downloader = await service.get(source)
        try:
            task = await downloader.start_download(request.repo_id, request.token)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error))
        return {"success": True, "task": task.to_dict()}

    @app.get("/admin/downloads/{source}/tasks")
    async def tasks(source: str):
        return {"tasks": (await service.get(source)).get_tasks()}

    @app.post("/admin/downloads/{source}/cancel/{task_id}")
    async def cancel(source: str, task_id: str):
        if not await (await service.get(source)).cancel_download(task_id):
            raise HTTPException(status_code=404, detail="Task not found or not cancellable")
        return {"success": True}

    @app.post("/admin/downloads/{source}/retry/{task_id}")
    async def retry(source: str, task_id: str, request: RetryRequest = RetryRequest()):
        try:
            task = await (await service.get(source)).retry_download(task_id, request.token)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))
        return {"success": True, "task": task.to_dict()}

    @app.delete("/admin/downloads/{source}/task/{task_id}")
    async def remove(source: str, task_id: str):
        if not (await service.get(source)).remove_task(task_id):
            raise HTTPException(status_code=404, detail="Task not found or still active")
        return {"success": True}

    @app.get("/admin/downloads/{source}/recommended")
    async def recommended(source: str, mlx_only: bool = True):
        await service.get(source)
        if source == "hf":
            from .hf import HFDownloader as Hub
        else:
            from .ms import MSDownloader as Hub
        try:
            return await Hub.get_recommended_models(
                max_memory_bytes=system_memory_bytes(), result_limit=50, mlx_only=mlx_only
            )
        except Exception as error:
            raise _upstream_error(names[source], error)

    @app.get("/admin/downloads/{source}/search")
    async def search(
        source: str,
        q: str = "",
        sort: str = "trending",
        limit: int = 100,
        mlx_only: bool = True,
        min_params: Optional[int] = None,
        max_params: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        await service.get(source)
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
        try:
            if source == "hf":
                from .hf import HFDownloader

                return await HFDownloader.search_models(
                    query=q.strip(), sort=sort, limit=min(limit, 100), mlx_only=mlx_only,
                    min_params=min_params, max_params=max_params,
                    min_size=min_size, max_size=max_size,
                )
            from .ms import MSDownloader

            return await MSDownloader.search_models(
                query=q.strip(), sort=sort, limit=min(limit, 100), mlx_only=mlx_only
            )
        except Exception as error:
            raise _upstream_error(names[source], error)

    @app.get("/admin/downloads/{source}/model-info")
    async def model_info(source: str, repo_id: str = ""):
        await service.get(source)
        if not repo_id.strip():
            raise HTTPException(status_code=400, detail="Query parameter 'repo_id' is required")
        try:
            if source == "hf":
                from huggingface_hub.utils import RepositoryNotFoundError

                from .hf import HFDownloader

                try:
                    return await HFDownloader.get_model_info(repo_id=repo_id.strip())
                except RepositoryNotFoundError:
                    raise HTTPException(status_code=404, detail=f"Model '{repo_id.strip()}' not found")
            from .ms import MSDownloader

            try:
                return await MSDownloader.get_model_info(model_id=repo_id.strip())
            except Exception as error:
                if "NotExistError" in type(error).__name__ or "404" in str(error):
                    raise HTTPException(status_code=404, detail=f"Model '{repo_id.strip()}' not found")
                raise
        except HTTPException:
            raise
        except Exception as error:
            raise _upstream_error(names[source], error)

    return service
