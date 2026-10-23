import json
from pathlib import Path

import pytest

from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json


def test_completion_publication_is_exclusive_and_keeps_existing_bytes(tmp_path):
    target = tmp_path / "complete.json"
    publish_json_exclusive(target, {"value": "中文", "complete": True})
    before = target.read_bytes()
    assert json.loads(before)["complete"]
    with pytest.raises(FileExistsError):
        publish_json_exclusive(target, {"different": True})
    assert target.read_bytes() == before
    assert not list(tmp_path.glob(".asset-manifest-*"))


def test_interrupt_before_publication_leaves_no_partial_marker(tmp_path, monkeypatch):
    target = tmp_path / "complete.json"

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("simulated interruption before atomic publication")

    with monkeypatch.context() as scope:
        scope.setattr(Path, "hardlink_to", interrupted)
        with pytest.raises(KeyboardInterrupt):
            publish_json_exclusive(target, {"complete": True})
    assert not target.exists()
    assert not list(tmp_path.glob(".asset-manifest-*"))
    publish_json_exclusive(target, {"complete": True})
    assert json.loads(target.read_text())["complete"]


def test_serialization_failure_cannot_create_a_completion_file(tmp_path):
    target = tmp_path / "complete.json"
    with pytest.raises(ValueError):
        publish_json_exclusive(target, {"invalid": float("nan")})
    assert not target.exists()
    assert not list(tmp_path.iterdir())


def test_owned_journal_updates_atomically_and_rejects_symlinks(tmp_path):
    target = tmp_path / "part.json"
    replace_owned_json(target, {"generation": 1})
    replace_owned_json(target, {"generation": 2})
    assert json.loads(target.read_text())["generation"] == 2
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        replace_owned_json(link, {"generation": 3})
    assert json.loads(target.read_text())["generation"] == 2
