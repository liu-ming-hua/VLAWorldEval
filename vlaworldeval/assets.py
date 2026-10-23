"""Evidence-based local snapshot verification; downloads are explicit opt-in."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from vlaworldeval.asset_lock import destination_locked
from vlaworldeval.atomic_json import publish_json_exclusive

MANIFEST = ".vlaworldeval-download.json"


def retryable_transport_error(error):
    """Hub cache-miss wrappers may hide a transport failure in their cause."""
    types = (ConnectionError, TimeoutError)
    try:
        from requests.exceptions import ConnectionError as RequestConnectionError, Timeout, ChunkedEncodingError
        types += (RequestConnectionError, Timeout, ChunkedEncodingError)
    except ImportError:
        pass
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        status = getattr(getattr(error, "response", None), "status_code", None)
        if status is not None:
            return status in {408, 429, 500, 502, 503, 504}
        if isinstance(error, types):
            return True
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return False


def transfer_file(operation, *, filename, attempts=5):
    """Resume the same immutable file on transient errors, not the whole cohort."""
    if type(attempts) is not int or not 1 <= attempts <= 10:
        raise ValueError("File transfer attempts must be an integer in[1,10]")
    for index in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if index+1 == attempts or not retryable_transport_error(exc):
                raise
            logging.getLogger(__name__).warning("Resuming %s after %s (attempt%d/%d)",
                filename, type(exc).__name__, index+2, attempts)
            time.sleep(min(10*(index+1), 30))


def file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"size": path.stat().st_size, "sha256": digest.hexdigest()}


def remote_file_record(path: Path, remote) -> dict:
    record = file_record(path)
    if remote.size != record["size"]:
        raise ValueError(f"Remote file size mismatch: {remote.rfilename}")
    if remote.lfs is not None:
        checksum = remote.lfs.get("sha256") if isinstance(remote.lfs, dict) else remote.lfs.sha256
        if checksum != record["sha256"]:
            raise ValueError(f"Remote LFS checksum mismatch: {remote.rfilename}")
        record["remote_checksum"] = {"algorithm": "sha256", "value": checksum}
    else:
        git_hash = hashlib.sha1(f"blob {record['size']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                git_hash.update(block)
        if git_hash.hexdigest() != remote.blob_id:
            raise ValueError(f"Remote Git blob checksum mismatch: {remote.rfilename}")
        record["remote_checksum"] = {"algorithm": "git_blob_sha1", "value": remote.blob_id}
    return record


def verify_snapshot(directory: Path, *, names=None) -> dict:
    directory = Path(directory).resolve()
    marker = directory / MANIFEST
    if marker.is_symlink():
        raise ValueError("Snapshot manifest must not be a symlink")
    report = json.loads(marker.read_text())
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1 or not isinstance(report.get("files"), dict) or not report["files"]):
        raise ValueError("Invalid or empty asset download manifest")
    if not isinstance(report.get("repository"), str) or not report["repository"]:
        raise ValueError("Asset manifest requires a repository identity")
    if not re.fullmatch(r"[0-9a-f]{40}", str(report.get("revision", ""))):
        raise ValueError("Snapshot manifest must record an immutable HF commit")
    selected = None
    if names is not None:
        if not isinstance(names, (list, tuple, set)) or not names or any(not isinstance(n, str) for n in names):
            raise ValueError("Selected snapshot verification needs nonempty explicit filenames")
        selected = set(names)
        if len(selected) != len(names) or not selected <= report["files"].keys():
            raise ValueError("Duplicate or undeclared snapshot verification filenames")
    for name, expected in report["files"].items():
        if (not isinstance(name, str) or not name or not isinstance(expected, dict)
                or type(expected.get("size")) is not int or expected["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(expected.get("sha256", "")))):
            raise ValueError("Invalid asset file hash/size record")
        path = directory / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or not path.resolve().is_relative_to(directory)):
            raise ValueError(f"Snapshot file leaves asset directory: {name}")
        if selected is not None and name not in selected:
            continue
        actual = file_record(path) if path.is_file() else None
        if actual is None or any(actual[key] != expected[key] for key in ("size", "sha256")):
            raise ValueError(f"Asset missing or corrupted: {name}")
    return {"ok": True, "directory": str(directory), "repository": report["repository"],
            "revision": report["revision"], "files_verified": len(report["files"]) if selected is None else len(selected),
            **({} if selected is None else {"verification_scope": "selected_files_only", "snapshot_files_declared": len(report["files"])})}


@destination_locked("directory")
def download_snapshot(repo: str, directory: Path, *, root: Path, revision="main",
                      patterns=None, repo_type="model") -> dict:
    """Resolve one immutable revision, retain resume metadata, and hash files.

    A verified manifest freezes a previous default-main download. An explicit
    different revision requires a different destination, never a silent update.
    Missing/corrupt files fail verification; remove the manifest explicitly to
    request repair (the resumable snapshot cache and model files are retained).
    """
    directory = Path(directory).absolute()
    root = Path(root).resolve()
    if not directory.resolve().is_relative_to(root) or directory.is_symlink():
        raise ValueError("Asset destination must be inside the project, not a symlink")
    if repo_type not in {"model", "dataset"}:
        raise ValueError("repo_type must be model or dataset")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Use an explicit namespace/repository Hugging Face identifier")
    if not isinstance(revision, str) or not revision or revision.startswith("-"):
        raise ValueError("Invalid HF revision")
    patterns = list(patterns or [])
    if any(not isinstance(p, str) or not p for p in patterns):
        raise ValueError("allow patterns must be nonempty strings")
    if (directory / MANIFEST).is_symlink():
        raise ValueError("Snapshot manifest must not be a symlink")
    if (directory / MANIFEST).is_file():
        previous = json.loads((directory / MANIFEST).read_text())
        if (previous.get("repository") != repo or previous.get("repo_type") != repo_type
                or previous.get("allow_patterns") != patterns
                or revision not in {previous.get("requested_revision"), previous.get("revision")}):
            raise ValueError("Asset identity/revision changed; choose a new destination")
        return verify_snapshot(directory)
    from vlaworldeval.workspace import direct_network_environment
    environment = direct_network_environment(root)
    # Configure before importing huggingface_hub (it captures constants at import).
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key, None)
    for key in ("HF_ENDPOINT", "HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE",
                "XDG_CACHE_HOME", "UV_CACHE_DIR", "UV_INDEX_URL", "HF_HUB_DISABLE_XET", "HF_HUB_DOWNLOAD_TIMEOUT"):
        os.environ[key] = environment[key]
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ValueError("Download requires huggingface_hub in the download-tool environment; "
                         "offline asset verification does not require it") from exc
    api = HfApi(endpoint=environment["HF_ENDPOINT"])
    info = api.repo_info(repo_id=repo, repo_type=repo_type, revision=revision, files_metadata=True)
    commit = info.sha
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit)):
        raise ValueError("Mirror returned an invalid immutable commit")
    remote_files = {item.rfilename: item for item in info.siblings}
    names = list(remote_files)
    if patterns:
        # Python3.10 fnmatch caches only256 compiled patterns. Thousands of
        # exact episode paths otherwise repeatedly compile regexes for every
        # repository entry. Exact names are set lookups; compile glob rules once.
        exact = {p for p in patterns if not any(c in p for c in "*?[")}
        globs = [re.compile(fnmatch.translate(p)) for p in patterns if p not in exact]
        names = [name for name in names if name in exact or any(p.match(name) for p in globs)]
    if not names:
        raise ValueError("No repository files matched the requested snapshot")
    if any(Path(name).is_absolute() or ".." in Path(name).parts or name == MANIFEST for name in names):
        raise ValueError("Repository returned unsafe or reserved snapshot filenames")
    directory.mkdir(parents=True, exist_ok=True)
    # A legacy directory has no trusted revision manifest. Verify every
    # existing selected file BEFORE snapshot_download may replace it. Preserve
    # different/corrupt/user-modified weights; a new version needs a new folder.
    for name in names:
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError(f"Legacy snapshot path leaves its directory: {name}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Legacy asset is not a regular file: {name}")
            try:
                remote_file_record(path, remote_files[name])
            except ValueError as exc:
                raise ValueError(f"Existing asset differs from pinned upstream: {name}; "
                                 "preserving it, choose a new destination") from exc
    # Passing thousands of exact names makes Hub's fnmatch filter quadratic
    # against large repositories. Original glob patterns select the identical
    # immutable payload; every resolved name is still verified below.
    if len(patterns) > 512 or len(names) > 64:
        # Large exact episode manifests exceed Hub's regex filter cache. Names
        # were already resolved against this immutable commit above; avoid a
        # second quadratic filtering pass, without widening the downloaded set.
        # Large multipart archives also need per-file range retries rather
        # than redoing the whole corpus when one2GB response is interrupted.
        from concurrent.futures import ThreadPoolExecutor
        from huggingface_hub import hf_hub_download
        def fetch(name):
            return transfer_file(lambda: hf_hub_download(repo_id=repo, repo_type=repo_type, revision=commit,
                filename=name, local_dir=str(directory), cache_dir=environment["HF_HUB_CACHE"],
                endpoint=environment["HF_ENDPOINT"]), filename=name)
        with ThreadPoolExecutor(max_workers=2) as executor:
            for _ in executor.map(fetch, names):
                pass
    else:
        snapshot_download(repo_id=repo, repo_type=repo_type, revision=commit,
                          local_dir=str(directory), allow_patterns=patterns or None, max_workers=1,
                          cache_dir=environment["HF_HUB_CACHE"], endpoint=environment["HF_ENDPOINT"])
    records = {}
    for name in names:
        path = directory / name
        if (Path(name).is_absolute() or ".." in Path(name).parts
                or not path.resolve().is_relative_to(directory) or not path.is_file()):
            raise ValueError(f"Snapshot incomplete or unsafe: {name}")
        records[name] = remote_file_record(path, remote_files[name])
    report = {"schema_version": 1, "repository": repo, "repo_type": repo_type,
              "requested_revision": revision, "revision": commit,
              "endpoint": environment["HF_ENDPOINT"], "allow_patterns": patterns, "files": records}
    # Create the final marker only after all required repository files are read.
    publish_json_exclusive(directory / MANIFEST, report)
    return verify_snapshot(directory)


def download_main(argv=None):
    parser = argparse.ArgumentParser(description="Proxy-free, revision-recorded HF snapshot download")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    args = parser.parse_args(argv)
    try:
        report = download_snapshot(args.repo, args.destination, root=args.root,
                                   revision=args.revision, patterns=args.allow,
                                   repo_type=args.repo_type)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    download_main()
