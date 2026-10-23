"""Authenticate a HF-mirrored OpenPI checkpoint against official GCS MD5s.

Only public metadata is fetched from GCS; model bytes stay on the HF mirror.
An HF repository name alone is not evidence that a community copy is official.
"""
import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

from vlaworldeval.assets import verify_snapshot
from vlaworldeval.atomic_json import publish_json_exclusive


def check_file(path, expected):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != int(expected["size"]):
        raise ValueError(f"Official checkpoint size mismatch: {path}")
    digest = hashlib.md5()  # GCS integrity checksum, not a security signature.
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = base64.b64encode(digest.digest()).decode("ascii")
    if value != expected["md5Hash"]:
        raise ValueError(f"Official checkpoint MD5 mismatch: {path}")
    return {"size": path.stat().st_size, "gcs_md5": value, "generation": expected["generation"]}


def verify_openpi_mirror(directory, *, model="pi0_fast_droid"):
    if model != "pi0_fast_droid":
        raise ValueError("This verifier admits only the official pi0_fast_droid tree")
    directory = Path(directory).resolve()
    mirror = verify_snapshot(directory)
    prefix = f"checkpoints/{model}/"
    opener = build_opener(ProxyHandler({}))
    objects, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?" + urlencode(params)
        with opener.open(url, timeout=30) as response:
            data = json.load(response)
        objects.extend(data.get("items", []))
        token = data.get("nextPageToken")
        if not token:
            break
    records = {}
    for item in objects:
        name = item["name"]
        if name.endswith("/"):
            continue
        relative = name.removeprefix(prefix)
        path = directory / relative
        if (not name.startswith(prefix) or Path(relative).is_absolute() or ".." in Path(relative).parts
                or not path.resolve().is_relative_to(directory) or relative in records):
            raise ValueError("Unsafe or duplicate official checkpoint filename")
        records[relative] = check_file(path, item)
    if not records or "assets/droid/norm_stats.json" not in records or "params/_METADATA" not in records:
        raise ValueError("Official checkpoint metadata is incomplete")
    result = {"schema_version": 1, "status": "official_gcs_bytes_verified",
        "model": model, "official_source": f"gs://openpi-assets/{prefix}",
        "weights_downloaded_from": mirror["repository"], "hf_revision": mirror["revision"],
        "files": records, "gcs_access": "public_metadata_only_no_weight_download"}
    marker = directory / ".vlaworldeval-official-source.json"
    if not marker.exists():
        publish_json_exclusive(marker, result)
    elif json.loads(marker.read_text()) != result:
        raise ValueError("Existing official-source marker disagrees with verified files")
    return result
