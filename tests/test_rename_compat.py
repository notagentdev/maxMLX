"""The pre-rename names keep working: ``import maxtoken``, ``MAXTOKEN_*``
variables and the ``~/.maxtoken`` state directories."""

import os

import maxmlx
from maxmlx import paths


def test_maxtoken_imports_are_the_maxmlx_modules():
    import maxtoken
    import maxtoken.server.settings as old

    import maxmlx.server.settings as new

    assert maxtoken is maxmlx
    assert old is new


def test_legacy_env_configures_the_new_name(monkeypatch):
    monkeypatch.setenv("MAXTOKEN_RENAME_PROBE", "old")
    monkeypatch.setenv("FREETOKEN_RENAME_PROBE", "older")
    monkeypatch.delenv("MAXMLX_RENAME_PROBE", raising=False)
    maxmlx._adopt_legacy_env()
    assert os.environ["MAXMLX_RENAME_PROBE"] == "old"
    monkeypatch.setenv("MAXMLX_RENAME_PROBE", "new")
    maxmlx._adopt_legacy_env()
    assert os.environ["MAXMLX_RENAME_PROBE"] == "new"


def test_state_dirs_move_once_and_leave_a_symlink(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".maxtoken" / "chats").mkdir(parents=True)
    (tmp_path / ".maxtoken" / "settings.json").write_text("{}")
    paths.migrate_legacy_dirs()
    assert (tmp_path / ".maxmlx" / "settings.json").read_text() == "{}"
    assert (tmp_path / ".maxtoken").is_symlink()
    assert (tmp_path / ".maxtoken" / "chats").is_dir()
    assert not (tmp_path / ".cache" / "maxmlx").exists()  # nothing to move
    paths.migrate_legacy_dirs()  # idempotent
    assert (tmp_path / ".maxmlx" / "settings.json").exists()


def test_existing_new_dir_is_never_overwritten(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".maxtoken").mkdir()
    (tmp_path / ".maxmlx").mkdir()
    paths.migrate_legacy_dirs()
    assert (tmp_path / ".maxtoken").is_dir() and not (tmp_path / ".maxtoken").is_symlink()
