"""Publish complete JSON exclusively, without truncating an existing manifest."""
import json
import os
from pathlib import Path
import tempfile


def _write_json(path: Path, payload, *, exclusive, staging_directory=None):
    path = Path(path)
    # Serialization errors must happen before any completion file is created.
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    parent = path.parent if staging_directory is None else Path(staging_directory)
    with tempfile.TemporaryDirectory(prefix=".asset-manifest-", dir=parent) as temporary:
        stage = Path(temporary) / "manifest.json"
        with stage.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            # Same-filesystem publication never replaces an existing marker.
            path.hardlink_to(stage)
        else:
            if path.is_symlink():
                raise ValueError("Owned journal must not be a symlink")
            stage.replace(path)


def publish_json_exclusive(path: Path, payload, *, staging_directory: Path | None = None):
    _write_json(path, payload, exclusive=True, staging_directory=staging_directory)


def replace_owned_json(path: Path, payload):
    """Only update a mutable journal in an already claimed, locked cache.

    Do not use this for checkpoint or version completion manifests.
    """
    _write_json(path, payload, exclusive=False)
