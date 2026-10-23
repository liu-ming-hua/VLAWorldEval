"""Verified, project-local DreamTrue release acquisition; no proxy or implicit model substitute."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vlaworldeval.asset_lock import asset_destination_lock  # noqa: E402
from vlaworldeval.assets import download_snapshot  # noqa: E402

DATASET_REVISION = "69cabc89029eb5996ba3209bc7071c2b87970252"
BASE_REVISION = "539c162b1387eac9dc4c20bd3f74671309e76a4c"
MODELSCOPE_BASE_REVISION = "2e854f5dfa7482ca3fef5b3af047672d05c0d395"
BASE_PATTERNS = ["diffusion_pytorch_model*", "google/umt5-xxl/*", "Wan2.1_VAE.pth",
                 "models_t5_umt5-xxl-enc-bf16.pth", "config.json", "LICENSE.txt"]
DOWNLOAD_RANGE_BYTES = 256 * 1024 * 1024
ARTIFACTS = (
    ("wmvideo/checkpoint-wmvideo.safetensors", 1476496816,
     "e7c34b3028404d5b24865d7fde05d18e97d82d4e30c4a5de1656481c7a33e919"),
    ("condition/condition-mini.tar.gz", 267999132,
     "c81133b49137b2e2b8d05880abf60275267cc5369963e4d866b8d28583760eb3"),
)


def direct_network():
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key)
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, destination, size, digest):
    with asset_destination_lock(ROOT, Path(destination)):
        for attempt in range(5):
            try:
                return _fetch(url, destination, size, digest)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if attempt == 4:
                    raise
                print(f"network retry {attempt + 1}/4: {Path(destination).name}: {exc}", flush=True)
                time.sleep(1)


def _fetch(url, destination, size, digest):
    destination = Path(destination)
    if destination.is_file():
        if destination.stat().st_size == size and sha256(destination) == digest:
            print(f"verified {destination}", flush=True)
            return destination
        raise ValueError(f"Existing asset failed verification; will not overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Reuse exact project-local common Wan assets only after full SHA256
    # verification. Never borrow a venv or accept merely matching filenames.
    if destination.name in {"models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"}:
        for candidate in (ROOT / "checkpoints").rglob(destination.name):
            if candidate == destination or not candidate.resolve().is_relative_to(ROOT):
                continue
            if candidate.is_file() and candidate.stat().st_size == size and sha256(candidate) == digest:
                with candidate.open("rb") as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                print(f"reused verified project-local asset {candidate} -> {destination}", flush=True)
                return destination
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.is_symlink():
        raise ValueError(f"Partial asset must not be a symlink: {temporary}")
    offset = temporary.stat().st_size if temporary.is_file() else 0
    if offset == size and sha256(temporary) == digest:
        temporary.rename(destination)
        return destination
    if offset >= size:
        raise ValueError(f"Invalid existing partial asset: {temporary}")
    while offset < size:
        # Bound each connection to 256 MiB. Multi-GB signed CDN connections
        # sometimes stall; every completed range remains resumable under lock.
        end = min(offset + DOWNLOAD_RANGE_BYTES, size) - 1
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
        with urllib.request.urlopen(request, timeout=120) as response:
            resumed = response.status == 206
            if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned an unexpected resume range")
            mode = "ab" if resumed and offset else "wb"
            with temporary.open(mode) as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        new_offset = temporary.stat().st_size
        if new_offset <= offset:
            raise ValueError("Asset server returned no forward progress")
        offset = new_offset
    if temporary.stat().st_size != size or sha256(temporary) != digest:
        raise ValueError(f"Downloaded asset failed size/SHA256 verification: {temporary}")
    temporary.rename(destination)
    print(f"downloaded and verified {destination}", flush=True)
    return destination


def extract_conditions(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for item in members:
            target = (destination / item.name).resolve()
            if not target.is_relative_to(destination) or not (item.isfile() or item.isdir()):
                raise ValueError(f"Unsafe archive member: {item.name}")
            if target.exists() and item.isfile():
                raise FileExistsError(f"Refusing to overwrite extracted condition: {target}")
        # All members validated above, including rejection of links and devices.
        # Manual extraction is compatible with Python 3.10 as well as 3.11.
        for item in members:
            target = destination / item.name
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(item) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def download_release(*, conditions_only=False):
    for name, size, digest in ARTIFACTS:
        if conditions_only and name.startswith("wmvideo/"):
            continue
        query = urllib.parse.urlencode({"Source": "SDK", "Revision": DATASET_REVISION, "FilePath": name})
        url = "https://modelscope.cn/api/v1/datasets/huoxingdawang/DreamTrue/repo?" + query
        base = ROOT / ("checkpoints/world_models/dreamtrue" if name.startswith("wmvideo/") else "datasets/dreamtrue")
        path = fetch(url, base / name, size, digest)
        if name.endswith(".tar.gz"):
            extracted = ROOT / "datasets/dreamtrue/release"
            with asset_destination_lock(ROOT, extracted):
                marker = extracted / ".condition-mini.sha256"
                if marker.is_file() and marker.read_text().strip() == digest:
                    continue
                extract_conditions(path, extracted)
                marker.write_text(digest + "\n")


def download_base(local_base=None, *, source="hf"):
    target = ROOT / "checkpoints/world_models/dreamtrue/Wan-AI/Wan2.1-VACE-14B"
    if local_base:
        # Shared downloader requires files to resolve inside its destination.
        # Reuse a complete, verified project-local snapshot in place (adapter's
        # base_model_path), rather than creating symlinks it would reject.
        local = Path(local_base).resolve()
        if not local.is_relative_to(ROOT):
            raise ValueError("Local base must be inside the project")
        target = local
    if source == "modelscope":
        # The SDK validates branch/tag names and rejects a valid Git commit;
        # the official file API supports immutable commits directly.
        listing_url = "https://modelscope.cn/api/v1/models/Wan-AI/Wan2.1-VACE-14B/repo/files?" + urllib.parse.urlencode(
            {"Revision": MODELSCOPE_BASE_REVISION, "Recursive": "true"})
        with urllib.request.urlopen(listing_url, timeout=60) as stream:
            listing = json.load(stream)
        entries = [x for x in listing["Data"]["Files"] if x["Type"] == "blob" and
                   any(fnmatch.fnmatchcase(x["Path"], pattern) for pattern in BASE_PATTERNS)]
        if len(entries) != 16:
            raise ValueError("Incomplete pinned ModelScope base listing")
        # One directory lock covers parallel file workers and metadata.
        # The shared HF downloader subsequently verifies the exact HF revision,
        # including all LFS/Git blob checksums, before publishing its manifest.
        with asset_destination_lock(ROOT, target):
            def acquire(entry):
                name = entry["Path"]
                if Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Unsafe ModelScope path")
                url = "https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B/resolve/" + MODELSCOPE_BASE_REVISION + "/" + name
                return fetch(url, target / name, entry["Size"], entry["Sha256"])
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(acquire, entries))
    return download_snapshot("Wan-AI/Wan2.1-VACE-14B", target, root=ROOT, revision=BASE_REVISION,
                             patterns=BASE_PATTERNS)


def download_raw_droid(index=0):
    """Acquire the exact condition manifest episode from DROID's official GCS bucket.

    GCS generation and MD5 are verified, SHA256 recorded; no other episode or
    converted camera layout is substituted. SVO recordings are not needed.
    """
    manifest = ROOT / "datasets/dreamtrue/release/condition/data-config/droid.json"
    entry = json.loads(manifest.read_text())[index]
    version, lab, status, timestamp = entry["data_key"].split("/")
    import datetime
    date = datetime.datetime.strptime(timestamp.replace("_", " "), "%a %b %d %H:%M:%S %Y").strftime("%Y-%m-%d")
    relative = f"{version}/{lab}/{status}/{date}/{timestamp}"
    prefix = "robotics/droid_raw/" + relative + "/"
    listing_url = "https://storage.googleapis.com/storage/v1/b/gresearch/o?" + urllib.parse.urlencode(
        {"prefix": prefix, "maxResults": 100})
    with urllib.request.urlopen(listing_url, timeout=30) as response:
        listing = json.load(response)
    selected = [x for x in listing.get("items", []) if x["name"].endswith((".json", ".h5", ".mp4"))]
    if not selected or "nextPageToken" in listing:
        raise ValueError("Incomplete official raw DROID object listing")
    destination = ROOT / "datasets/dreamtrue/raw/droid" / relative
    records = []
    with asset_destination_lock(ROOT, destination):
        for item in selected:
            name = item["name"][len(prefix):]
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe GCS object path")
            url = "https://storage.googleapis.com/download/storage/v1/b/gresearch/o/" + urllib.parse.quote(
                item["name"], safe="") + "?" + urllib.parse.urlencode({"alt": "media", "generation": item["generation"]})
            if target.is_file():
                content = target.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=120) as response:
                    content = response.read()
            if len(content) != int(item["size"]) or base64.b64encode(hashlib.md5(content).digest()).decode() != item["md5Hash"]:
                raise ValueError(f"GCS size/MD5 verification failed: {name}")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(content)
            records.append({"path": str(target.relative_to(ROOT)), "object": item["name"],
                "generation": item["generation"], "md5": item["md5Hash"], "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest()})
            print(f"verified official DROID {name}", flush=True)
        (destination / "dreamtrue-raw-provenance.json").write_text(json.dumps(
            {"source": "gs://gresearch/robotics/droid_raw", "data_key": entry["data_key"], "files": records}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-only", action="store_true")
    parser.add_argument("--base", action="store_true", help="Also acquire exact VACE-14B base weights")
    parser.add_argument("--base-source", choices=["hf", "modelscope"], default="hf")
    parser.add_argument("--raw-droid", action="store_true", help="Download one exact official condition-matched raw episode")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--local-base", type=Path, help="Reuse project-local base only after upstream hash verification")
    args = parser.parse_args()
    if args.local_base and not args.local_base.resolve().is_relative_to(ROOT):
        parser.error("--local-base must be inside the project")
    direct_network()
    download_release(conditions_only=args.conditions_only)
    if args.base:
        download_base(args.local_base, source=args.base_source)
    if args.raw_droid:
        download_raw_droid(args.index)


if __name__ == "__main__":
    main()
