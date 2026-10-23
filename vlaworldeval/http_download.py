"""Bounded, resumable, proxy-free Range downloads for pinned official archives."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import shutil
from threading import Event
import time
from urllib.request import build_opener, ProxyHandler, Request
from vlaworldeval.asset_lock import destination_locked
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json
from vlaworldeval.assets import file_record


def md5(path):
    digest = hashlib.md5(usedforsecurity=False)  # Published transport checksum, not authentication.
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


@destination_locked("output")
def download_archive(url, output: Path, *, root: Path, size: int, expected_md5: str,
                     workers=4, chunk_size=1024**2):
    output = output.absolute()
    if (output.is_symlink() or not output.resolve().is_relative_to(root.resolve())
            or output.resolve() == root.resolve() or not url.startswith("https://")):
        raise ValueError("Use HTTPS and a project-local archive destination")
    if (type(size) is not int or not 0 < size <= 20 * 1024**3
            or type(workers) is not int or not 1 <= workers <= 8
            or type(chunk_size) is not int or not 1 <= chunk_size <= 64 * 1024**2
            or (size + chunk_size - 1) // chunk_size > 65536):
        raise ValueError("Require size 1..20 GiB, integer workers 1..8, integer chunk_size 1..64 MiB, and at most 65536 ranges")
    try:
        if not isinstance(expected_md5, str) or len(base64.b64decode(expected_md5, validate=True)) != 16:
            raise ValueError("Expected a base64-encoded 16-byte upstream MD5")
    except (ValueError, TypeError) as exc:
        raise ValueError("Expected a base64-encoded 16-byte upstream MD5") from exc
    if output.exists():
        if output.stat().st_size != size or md5(output) != expected_md5:
            raise ValueError("Existing archive is incomplete or differs; preserve it and choose a new output")
        return {"ok": True, "output": str(output), "cached": True}
    parts = output.with_name(output.name + ".parts")
    if parts.is_symlink():
        raise ValueError("Download part directory must not be a symlink")
    parts.mkdir(parents=True, exist_ok=True)
    identity = {"url": url, "size": size, "upstream_md5": expected_md5, "chunk_size": chunk_size}
    marker = parts / "identity.json"
    if marker.is_symlink():
        raise ValueError("Download identity manifest must not be a symlink")
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Part-directory identity differs; preserving it")
    elif any(parts.iterdir()):
        raise ValueError("Nonempty unowned part directory; preserving it")
    else:
        # Stage outside .parts: a crash before initial identity publication
        # must not leave an apparently nonempty/unowned resume directory.
        publish_json_exclusive(marker, identity, staging_directory=parts.parent)
    count = (size + chunk_size - 1) // chunk_size
    cancelled = Event()

    def fetch(index):
        start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        path = parts / f"{index:06d}.part"
        journal = parts / f"{index:06d}.part.json"
        if path.is_symlink() or journal.is_symlink():
            raise ValueError("Download part must not be a symlink")
        if path.is_file() and path.stat().st_size == end - start + 1 and journal.is_file():
            try:
                saved = json.loads(journal.read_text())
            except ValueError:
                saved = None
            actual = {"range": [start, end], **file_record(path)}
            if saved == actual:
                return path
        # Size alone does not establish a reusable range. Repair only pieces
        # inside this identity-claimed temporary cache, never final weights.
        for attempt in range(3):
            try:
                if cancelled.is_set():
                    raise OSError("Download cancelled after another range failed")
                request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
                with build_opener(ProxyHandler({})).open(request, timeout=120) as source:
                    if source.status != 206 or source.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Server did not honor the exact pinned byte range")
                    # Only overwrite our owned incomplete transport part, never
                    # a completed model or an unrelated user directory.
                    with path.open("wb") as out:
                        remaining = end - start + 1
                        while remaining:
                            if cancelled.is_set():
                                raise OSError("Download cancelled after another range failed")
                            block = source.read(min(256 * 1024, remaining))
                            if not block:
                                raise OSError("Truncated download range")
                            out.write(block)
                            remaining -= len(block)
                replace_owned_json(journal, {"range": [start, end], **file_record(path)})
                return path
            except OSError:
                if attempt == 2 or cancelled.is_set():
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        indices = iter(range(count))
        pending = {pool.submit(fetch, index) for index in range(min(count, workers))}
        # Submit only a bounded window; map() on Python 3.11 would eagerly
        # queue every range, continuing a large transfer after the first error.
        for _ in range(min(count, workers)):
            next(indices)
        completed = 0
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                # Check all finished requests before scheduling any more.
                for future in done:
                    future.result()
                completed += len(done)
                if completed % 32 == 0 or completed == count:
                    print(f"Completed ranges: {completed}/{count}", flush=True)
                for _ in done:
                    index = next(indices, None)
                    if index is not None:
                        pending.add(pool.submit(fetch, index))
        finally:
            cancelled.set()
            for future in pending:
                future.cancel()
    stage = parts / "assembled.tmp"
    if stage.is_symlink():
        raise ValueError("Assembly stage must not be a symlink")
    with stage.open("wb") as out:
        for index in range(count):
            with (parts / f"{index:06d}.part").open("rb") as source:
                shutil.copyfileobj(source, out)
    if stage.stat().st_size != size or md5(stage) != expected_md5:
        raise ValueError("Completed archive differs from pinned upstream checksum; parts retained for diagnosis")
    # Hard-link creation is exclusive, atomic, and on the same filesystem.
    # Keep resumable parts/provenance; do not race-overwrite an existing output.
    output.hardlink_to(stage)
    return {"ok": True, "output": str(output), "size": size, "upstream_md5": expected_md5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    try:
        report = download_archive(args.url, args.output, root=args.root, size=args.size,
                                  expected_md5=args.md5, workers=args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
