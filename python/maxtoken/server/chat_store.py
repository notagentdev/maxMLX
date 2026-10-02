"""Persistent chats for the web console: one JSON file per conversation.

The console used to keep a conversation in the page and lose it on reload. Chats now live next
to the other MaxToken state (``~/.maxtoken/chats``) so they survive reloads, a changed server
port (which is a different browser origin) and a second browser. The page owns the document's
shape (messages, per-chat settings, the system prompt in use); this module validates what is
safe to store, writes atomically and answers list/search without re-parsing unchanged files.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

DEFAULT_CHATS_DIR = "~/.maxtoken/chats"
DEFAULT_PROFILES_FILE = "~/.maxtoken/chat-profiles.json"
CHAT_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
MAX_CHAT_BYTES = 32 * 1024 * 1024
MAX_PROFILES = 100
DEFAULT_PROFILE = {"name": "System Default", "content": ""}


def chats_dir() -> Path:
    return Path(os.environ.get("MAXTOKEN_CHATS_DIR", DEFAULT_CHATS_DIR)).expanduser()


def profiles_path() -> Path:
    return Path(os.environ.get("MAXTOKEN_CHAT_PROFILES_FILE", DEFAULT_PROFILES_FILE)).expanduser()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _searchable(doc: dict) -> str:
    parts = [str(doc.get("title") or "")]
    parts += [_text_of(message.get("content")) for message in doc.get("messages", []) if isinstance(message, dict)]
    return "\n".join(parts).lower()


def _summary(doc: dict) -> dict:
    return {
        "id": doc["id"],
        "title": doc.get("title") or "",
        "pinned": bool(doc.get("pinned")),
        "manualTitle": bool(doc.get("manualTitle")),
        "updatedAt": doc.get("updatedAt") or "",
        "model": doc.get("model"),
        "messageCount": len(doc.get("messages", [])),
    }


class ChatStore:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory).expanduser() if directory else chats_dir()
        self._lock = threading.Lock()
        # path -> ((mtime_ns, size), summary, lowercase text): unchanged files are never re-parsed.
        self._cache: dict[Path, tuple[tuple[int, int], dict, str]] = {}

    def _path(self, chat_id: str) -> Path:
        if not CHAT_ID.match(chat_id):
            raise ValueError("invalid chat id")
        return self.directory / f"{chat_id}.json"

    def _entry(self, path: Path):
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(path)
        if cached and cached[0] == stamp:
            return cached
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("id") != path.stem:
            raise ValueError("not a chat document")
        entry = (stamp, _summary(doc), _searchable(doc))
        self._cache[path] = entry
        return entry

    def list(self, query: str = "") -> list[dict]:
        query = query.strip().lower()
        found = []
        with self._lock:
            if not self.directory.is_dir():
                return []
            for path in self.directory.glob("*.json"):
                try:
                    _, summary, text = self._entry(path)
                except (OSError, ValueError):
                    continue  # a half-written or foreign file never breaks the list
                if not query or query in text:
                    found.append(summary)
        found.sort(key=lambda c: (not c["pinned"], _negate(c["updatedAt"])))
        return found

    def get(self, chat_id: str) -> dict | None:
        path = self._path(chat_id)
        with self._lock:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                return None

    def save(self, chat_id: str, doc: dict) -> dict:
        path = self._path(chat_id)
        if not isinstance(doc, dict) or doc.get("id") != chat_id:
            raise ValueError("document id does not match the URL")
        if not isinstance(doc.get("messages"), list) or not all(isinstance(m, dict) for m in doc["messages"]):
            raise ValueError("messages must be a list of objects")
        doc = {**doc, "updatedAt": doc.get("updatedAt") or _now()}
        text = json.dumps(doc, ensure_ascii=False, allow_nan=False)
        if len(text.encode("utf-8")) > MAX_CHAT_BYTES:
            raise ValueError("chat is too large to store")
        with self._lock:
            _atomic_write(path, text)
            self._cache.pop(path, None)
        return _summary(doc)

    def delete(self, chat_id: str) -> bool:
        path = self._path(chat_id)
        with self._lock:
            self._cache.pop(path, None)
            try:
                path.unlink()
            except FileNotFoundError:
                return False
        return True

    def delete_all(self) -> int:
        with self._lock:
            count = 0
            for path in list(self.directory.glob("*.json")) if self.directory.is_dir() else []:
                if CHAT_ID.match(path.stem):
                    path.unlink(missing_ok=True)
                    count += 1
            self._cache.clear()
        return count


def _negate(iso: str) -> float:
    """Sort key that puts the newest timestamp first; unparsable ones sort last."""
    try:
        return -datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def load_profiles(path: Path | None = None) -> list[dict]:
    path = path or profiles_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        profiles = data["profiles"]
        assert isinstance(profiles, list)
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        profiles = []
    clean = [{"name": p["name"], "content": p.get("content", "")} for p in profiles
             if isinstance(p, dict) and isinstance(p.get("name"), str) and isinstance(p.get("content", ""), str)]
    if not any(p["name"] == DEFAULT_PROFILE["name"] for p in clean):
        clean.insert(0, dict(DEFAULT_PROFILE))
    return clean


def save_profiles(profiles: Any, path: Path | None = None) -> list[dict]:
    if not isinstance(profiles, list) or len(profiles) > MAX_PROFILES:
        raise ValueError("profiles must be a list of at most %d entries" % MAX_PROFILES)
    seen, clean = set(), []
    for profile in profiles:
        if not isinstance(profile, dict) or not isinstance(profile.get("name"), str) or not isinstance(profile.get("content", ""), str):
            raise ValueError("each profile needs a name and text content")
        name = profile["name"].strip()
        if not name or len(name) > 80 or name in seen:
            raise ValueError(f"profile names must be unique, non-empty and short: {name!r}")
        seen.add(name)
        clean.append({"name": name, "content": profile.get("content", "")})
    if DEFAULT_PROFILE["name"] not in seen:
        clean.insert(0, dict(DEFAULT_PROFILE))
    _atomic_write(path or profiles_path(), json.dumps({"profiles": clean}, ensure_ascii=False, indent=2) + "\n")
    return clean


def register_chat_routes(app: FastAPI) -> None:
    def store() -> ChatStore:
        # Resolved per request so MAXTOKEN_CHATS_DIR changes (tests, relaunches) apply at once.
        current = chats_dir()
        if getattr(app.state, "chat_store", None) is None or app.state.chat_store.directory != current:
            app.state.chat_store = ChatStore(current)
        return app.state.chat_store

    def bad(error: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(error))

    @app.get("/admin/chats")
    async def list_chats(q: str = ""):
        return {"chats": store().list(q)}

    @app.delete("/admin/chats")
    async def delete_all_chats(all: bool = False):
        if not all:
            raise HTTPException(status_code=400, detail="pass all=true to delete every chat")
        return {"deleted": store().delete_all()}

    @app.get("/admin/chats/{chat_id}")
    async def get_chat(chat_id: str):
        try:
            doc = store().get(chat_id)
        except ValueError as error:
            raise bad(error)
        if doc is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        return doc

    @app.put("/admin/chats/{chat_id}")
    async def put_chat(chat_id: str, request: Request):
        try:
            return store().save(chat_id, await request.json())
        except ValueError as error:  # also json.JSONDecodeError
            raise bad(error)

    @app.delete("/admin/chats/{chat_id}")
    async def delete_chat(chat_id: str):
        try:
            removed = store().delete(chat_id)
        except ValueError as error:
            raise bad(error)
        if not removed:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"success": True}

    @app.get("/admin/chat-profiles")
    async def get_profiles():
        return {"profiles": load_profiles()}

    @app.put("/admin/chat-profiles")
    async def put_profiles(request: Request):
        try:
            body = await request.json()
            return {"profiles": save_profiles(body.get("profiles") if isinstance(body, dict) else None)}
        except ValueError as error:
            raise bad(error)
