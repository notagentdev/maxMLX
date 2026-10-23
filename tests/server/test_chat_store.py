"""Persistent console chats: storage, search, validation and the HTTP surface."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from maxmlx.server import api_server as api
from maxmlx.server import chat_store
from maxmlx.server.chat_store import ChatStore


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MAXMLX_CHATS_DIR", str(tmp_path / "chats"))
    monkeypatch.setenv("MAXMLX_CHAT_PROFILES_FILE", str(tmp_path / "profiles.json"))


@pytest.fixture
def client():
    return TestClient(api.app, client=("127.0.0.1", 50000))


def chat(chat_id="chat_1", text="hello", **extra):
    return {"id": chat_id, "title": "T " + chat_id, "messages": [{"id": "m1", "role": "user", "content": text}], **extra}


def test_chats_round_trip_and_survive_a_new_store(tmp_path):
    store = ChatStore(tmp_path / "c")
    summary = store.save("chat_1", chat(settings={"temperature": 0.3}, systemPrompt="be brief"))
    assert summary["messageCount"] == 1 and summary["updatedAt"]
    reopened = ChatStore(tmp_path / "c").get("chat_1")
    assert reopened["settings"] == {"temperature": 0.3} and reopened["systemPrompt"] == "be brief"
    assert ChatStore(tmp_path / "c").get("nope") is None


def test_list_orders_pinned_first_then_newest_and_searches_content(tmp_path):
    store = ChatStore(tmp_path)
    store.save("a", chat("a", "alpha pelican", updatedAt="2026-01-01T00:00:00+00:00"))
    store.save("b", chat("b", "beta", updatedAt="2026-03-01T00:00:00+00:00"))
    store.save("c", chat("c", "gamma", updatedAt="2026-02-01T00:00:00+00:00", pinned=True))
    assert [c["id"] for c in store.list()] == ["c", "b", "a"]
    assert [c["id"] for c in store.list("PELICAN")] == ["a"]
    assert [c["id"] for c in store.list("t c")] == ["c"]  # titles are searched too
    # An overwrite is visible at once: the cache is keyed on the file, not the id.
    store.save("a", chat("a", "now about otters", updatedAt="2026-01-01T00:00:00+00:00"))
    assert store.list("pelican") == [] and [c["id"] for c in store.list("otters")] == ["a"]


def test_content_arrays_are_searchable(tmp_path):
    store = ChatStore(tmp_path)
    store.save("a", {"id": "a", "messages": [{"role": "user", "content": [{"type": "text", "text": "needle"}]}]})
    assert [c["id"] for c in store.list("needle")] == ["a"]


def test_unreadable_files_do_not_break_the_list(tmp_path):
    store = ChatStore(tmp_path)
    store.save("a", chat("a"))
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "foreign.json").write_text(json.dumps({"id": "other", "messages": []}))
    assert [c["id"] for c in store.list()] == ["a"]


@pytest.mark.parametrize("chat_id", ["../evil", "a/b", "", "x" * 81, "dot.dot"])
def test_ids_that_could_escape_the_folder_are_rejected(tmp_path, chat_id):
    store = ChatStore(tmp_path)
    with pytest.raises(ValueError):
        store.save(chat_id, chat(chat_id))
    with pytest.raises(ValueError):
        store.get(chat_id)


def test_documents_are_validated(tmp_path, monkeypatch):
    store = ChatStore(tmp_path)
    with pytest.raises(ValueError):
        store.save("a", {"id": "other", "messages": []})
    with pytest.raises(ValueError):
        store.save("a", {"id": "a", "messages": "no"})
    with pytest.raises(ValueError):
        store.save("a", {"id": "a", "messages": ["x"]})
    monkeypatch.setattr(chat_store, "MAX_CHAT_BYTES", 100)
    with pytest.raises(ValueError, match="too large"):
        store.save("a", chat("a", "x" * 200))
    assert not list(tmp_path.glob("*"))  # nothing half-written is left behind


def test_delete_and_delete_all(tmp_path):
    store = ChatStore(tmp_path)
    for name in "abc":
        store.save(name, chat(name))
    assert store.delete("a") and not store.delete("a")
    assert store.delete_all() == 2 and store.list() == []


def test_http_surface(client, tmp_path):
    assert client.get("/admin/chats").json() == {"chats": []}
    assert client.put("/admin/chats/chat_1", json=chat(settings={"top_p": 0.5})).json()["messageCount"] == 1
    assert client.get("/admin/chats/chat_1").json()["settings"] == {"top_p": 0.5}
    assert [c["id"] for c in client.get("/admin/chats?q=hello").json()["chats"]] == ["chat_1"]
    assert client.get("/admin/chats?q=zzz").json()["chats"] == []
    assert client.get("/admin/chats/missing").status_code == 404
    assert client.put("/admin/chats/chat_2", json=chat("chat_1")).status_code == 400
    assert client.put("/admin/chats/chat_2", content="{oops", headers={"Content-Type": "application/json"}).status_code == 400
    assert client.get("/admin/chats/bad.id").status_code == 400
    assert client.delete("/admin/chats").status_code == 400  # deleting everything must be explicit
    assert client.delete("/admin/chats/chat_1").status_code == 200
    assert client.delete("/admin/chats/chat_1").status_code == 404
    client.put("/admin/chats/x", json=chat("x"))
    assert client.delete("/admin/chats?all=true").json() == {"deleted": 1}


def test_profiles_always_keep_the_default_and_validate(client, tmp_path):
    assert client.get("/admin/chat-profiles").json()["profiles"] == [{"name": "System Default", "content": ""}]
    saved = client.put("/admin/chat-profiles", json={"profiles": [{"name": "Coder", "content": "Write tests."}]}).json()["profiles"]
    assert [p["name"] for p in saved] == ["System Default", "Coder"]
    assert client.get("/admin/chat-profiles").json()["profiles"] == saved
    for bad in ([{"name": "a"}, {"name": "a"}], [{"name": " "}], [{"content": "x"}], "no"):
        assert client.put("/admin/chat-profiles", json={"profiles": bad}).status_code == 400
    assert client.put("/admin/chat-profiles", json=[]).status_code == 400
