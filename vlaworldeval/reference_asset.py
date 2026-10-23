"""Pinned local-reference integrity for releases without an upstream checksum.

The recorded reference must be labelled as local, never as an author signature
or published upstream checksum. Existing different assets are not overwritten.
"""
import json
from pathlib import Path
import re

from vlaworldeval.asset_lock import destination_locked
from vlaworldeval.assets import file_record
from vlaworldeval.atomic_json import publish_json_exclusive


@destination_locked("destination")
def ensure_reference_asset(destination: Path, *, root: Path, size: int, sha256: str,
                           source: dict, fetch=None) -> dict:
    destination = Path(destination).absolute()
    if type(size) is not int or size <= 0 or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("Reference asset requires a positive integer size and SHA256")
    marker = destination.with_name(destination.name + ".vlaworldeval-reference.json")
    if marker.is_symlink():
        raise ValueError("Reference manifest must not be a symlink")
    expected = {"size": size, "sha256": sha256}
    record = {"schema_version": 1, "source": source, "reference_kind": "locally_verified_release",
              "upstream_checksum_verified": False, "file": destination.name, **expected}
    if marker.exists() and json.loads(marker.read_text()) != record:
        raise ValueError("Existing reference identity differs; preserving it, choose a new destination")
    cached = destination.exists()
    if cached:
        if (not destination.is_file() or destination.stat().st_size != size
                or file_record(destination) != expected):
            raise ValueError(f"Existing reference asset is incomplete or differs: {destination}; preserving it")
    else:
        if fetch is None:
            raise FileNotFoundError(f"Reference asset missing: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fetch(destination)
        if (not destination.is_file() or destination.stat().st_size != size
                or file_record(destination) != expected):
            raise ValueError("Downloaded asset does not match the pinned local release reference; not marking complete")
    if not marker.exists():
        # Atomic exclusive publication: interruption cannot leave a truncated
        # completion marker that appears valid on the next run.
        publish_json_exclusive(marker, record)
    return {"ok": True, "cached": cached, "path": str(destination), "manifest": str(marker), **record}
