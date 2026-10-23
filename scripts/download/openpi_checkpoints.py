"""Download official OpenPI GCS checkpoint trees into project-local storage."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.parse import quote

import gcsfs
import requests


MODELS = {
    "pi0-droid": "openpi-assets/checkpoints/pi0_droid",
    "pi05-droid": "openpi-assets/checkpoints/pi05_droid",
}

SEGMENT_THRESHOLD = 256 * 1024 * 1024
SEGMENT_SIZE = 64 * 1024 * 1024


def md5_base64(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - GCS exposes MD5 for integrity, not security.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def file_matches(path: Path, metadata: dict) -> bool:
    if not path.is_file() or path.stat().st_size != int(metadata.get("size", -1)):
        return False
    expected_md5 = metadata.get("md5Hash")
    return not expected_md5 or md5_base64(path) == expected_md5


def download_range(url: str, destination: Path, start: int, end: int) -> None:
    """Download one exact byte range, preserving complete segments for resume."""
    expected_size = end - start + 1
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    if destination.exists() and destination.stat().st_size > expected_size:
        destination.unlink()
    for attempt in range(10):
        offset = destination.stat().st_size if destination.exists() else 0
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start + offset}-{end}"},
                stream=True,
                timeout=(30, 20),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise IOError(
                        f"Server ignored Range {start + offset}-{end}: "
                        f"HTTP {response.status_code}"
                    )
                with destination.open("ab") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if destination.stat().st_size == expected_size:
                return
        except (OSError, requests.RequestException):
            if attempt == 9:
                raise
            time.sleep(min(2**attempt, 30))
    raise IOError(f"Incomplete byte range {start}-{end}: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--segment-workers", type=int, default=8)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    models = args.models or sorted(MODELS)
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}; choose from {', '.join(sorted(MODELS))}")
    for model in models:
        remote = MODELS[model]
        output = args.output_root / model
        output.mkdir(parents=True, exist_ok=True)
        # A marker is only valid after this invocation completes all byte and
        # checksum checks.  Do not leave a stale success marker during repair.
        (output / ".complete.json").unlink(missing_ok=True)
        print(f"Downloading gs://{remote} -> {output}", flush=True)
        filesystem = gcsfs.GCSFileSystem(token="anon")
        files = {
            path: metadata
            for path, metadata in filesystem.find(remote, detail=True).items()
            # GCS may expose the checkpoint prefix itself as a zero-byte
            # object.  It is a directory marker, not a checkpoint file.
            if metadata.get("type") == "file" and path.rstrip("/") != remote.rstrip("/")
        }

        def fetch(item: tuple[str, dict]) -> str:
            path, metadata = item
            relative = Path(path).relative_to(remote)
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected_size = int(metadata.get("size", -1))
            if file_matches(destination, metadata):
                return f"reuse {relative}"
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists() and temporary.stat().st_size >= expected_size:
                # A prior interrupted downloader may have appended a response
                # after a server ignored Range.  Such a file cannot be resumed.
                temporary.unlink()
            url = f"https://storage.googleapis.com/{quote(path, safe='/')}"
            if expected_size >= SEGMENT_THRESHOLD:
                if temporary.exists():
                    temporary.unlink()
                segments = []
                for index, start in enumerate(range(0, expected_size, SEGMENT_SIZE)):
                    end = min(start + SEGMENT_SIZE, expected_size) - 1
                    segment = destination.with_name(
                        f"{destination.name}.part.{index:05d}"
                    )
                    segments.append((segment, start, end))
                with ThreadPoolExecutor(max_workers=args.segment_workers) as segment_pool:
                    range_futures = [
                        segment_pool.submit(download_range, url, segment, start, end)
                        for segment, start, end in segments
                    ]
                    for range_future in as_completed(range_futures):
                        range_future.result()
                with temporary.open("wb") as output_handle:
                    for segment, _, _ in segments:
                        with segment.open("rb") as input_handle:
                            shutil.copyfileobj(input_handle, output_handle)
                for segment, _, _ in segments:
                    segment.unlink()
                if temporary.stat().st_size != expected_size:
                    raise IOError(
                        f"Size mismatch for {path}: "
                        f"{temporary.stat().st_size} != {expected_size}"
                    )
                expected_md5 = metadata.get("md5Hash")
                if expected_md5 and md5_base64(temporary) != expected_md5:
                    temporary.unlink()
                    raise IOError(f"MD5 mismatch for {path}")
                temporary.replace(destination)
                return f"done  {relative}"
            for attempt in range(10):
                offset = temporary.stat().st_size if temporary.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                try:
                    with requests.get(
                        url, headers=headers, stream=True, timeout=(30, 300)
                    ) as response:
                        response.raise_for_status()
                        # A server that ignores Range returns 200; restart the
                        # temporary file rather than appending duplicate bytes.
                        append = offset > 0 and response.status_code == 206
                        with temporary.open("ab" if append else "wb") as handle:
                            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                                if chunk:
                                    handle.write(chunk)
                    if temporary.stat().st_size == expected_size:
                        break
                except (OSError, requests.RequestException):
                    if attempt == 9:
                        raise
                    time.sleep(min(2**attempt, 30))
            if temporary.stat().st_size != expected_size:
                raise IOError(
                    f"Size mismatch for {path}: {temporary.stat().st_size} != {expected_size}"
                )
            expected_md5 = metadata.get("md5Hash")
            if expected_md5 and md5_base64(temporary) != expected_md5:
                temporary.unlink()
                raise IOError(f"MD5 mismatch for {path}")
            temporary.replace(destination)
            return f"done  {relative}"

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch, item) for item in files.items()]
            for future in as_completed(futures):
                print(future.result(), flush=True)

        recorded_files: dict[str, dict[str, int | str | None]] = {}
        for path, metadata in files.items():
            relative = Path(path).relative_to(remote)
            destination = output / relative
            expected_size = int(metadata.get("size", -1))
            if not file_matches(destination, metadata):
                raise IOError(f"Final verification failed for {destination}")
            recorded_files[str(relative)] = {
                "size": expected_size,
                "md5": metadata.get("md5Hash"),
            }
        marker = {
            "source": f"gs://{remote}",
            "file_count": len(recorded_files),
            "total_bytes": sum(item["size"] for item in recorded_files.values()),
            "files": dict(sorted(recorded_files.items())),
        }
        temporary_marker = output / ".complete.json.part"
        temporary_marker.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        temporary_marker.replace(output / ".complete.json")
        print(
            f"Complete: {model} ({marker['file_count']} files, "
            f"{marker['total_bytes'] / 1024**3:.2f} GiB)",
            flush=True,
        )


if __name__ == "__main__":
    main()
