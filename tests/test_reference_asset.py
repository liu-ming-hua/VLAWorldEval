import hashlib
import json

import pytest

from vlaworldeval.reference_asset import ensure_reference_asset


def arguments(tmp_path):
    return dict(root=tmp_path, size=7, sha256=hashlib.sha256(b"release").hexdigest(),
                source={"kind": "fixture", "version": "1"})


def test_reference_download_and_cached_check_are_not_upstream_verification(tmp_path):
    destination = tmp_path / "model.pt"
    report = ensure_reference_asset(destination, **arguments(tmp_path), fetch=lambda p: p.write_bytes(b"release"))
    assert report["ok"] and not report["cached"] and not report["upstream_checksum_verified"]
    assert ensure_reference_asset(destination, **arguments(tmp_path))["cached"]
    assert json.loads(destination.with_name("model.pt.vlaworldeval-reference.json").read_text())["reference_kind"] == "locally_verified_release"


def test_nonempty_wrong_reference_never_downloads_or_overwrites(tmp_path):
    destination = tmp_path / "model.pt"
    destination.write_bytes(b"partial")
    with pytest.raises(ValueError, match="preserving"):
        ensure_reference_asset(destination, **arguments(tmp_path),
                               fetch=lambda p: pytest.fail("Never overwrite an existing model"))
    assert destination.read_bytes() == b"partial"
    assert not destination.with_name("model.pt.vlaworldeval-reference.json").exists()


def test_failed_download_never_creates_completion_manifest(tmp_path):
    destination = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="not marking complete"):
        ensure_reference_asset(destination, **arguments(tmp_path), fetch=lambda p: p.write_bytes(b"bad"))
    assert not destination.with_name("model.pt.vlaworldeval-reference.json").exists()


def test_missing_reference_verify_only_does_not_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        ensure_reference_asset(tmp_path / "model.pt", **arguments(tmp_path))
