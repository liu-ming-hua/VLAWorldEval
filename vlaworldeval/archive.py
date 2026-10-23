"""Verified, non-overwriting extraction of official non-HF model archives."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
import zipfile

from vlaworldeval.assets import file_record
from vlaworldeval.asset_lock import destination_locked
from vlaworldeval.atomic_json import publish_json_exclusive


@destination_locked("destination")
def extract_archive(archive: Path, destination: Path, *, root: Path,
                    expected_md5: str, strip_prefix: str = "") -> dict:
    root = root.resolve()
    destination = destination.absolute()
    if destination.is_symlink() or not destination.resolve().is_relative_to(root) or destination.resolve() == root:
        raise ValueError("Archive destination must be a project-local asset directory")
    checksum = hashlib.md5(usedforsecurity=False)  # Transport checksum, not a security signature.
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    if base64.b64encode(checksum.digest()).decode() != expected_md5:
        raise ValueError("Archive does not match the pinned upstream GCS MD5")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".archive-extract-", dir=destination.parent))
    records = {}
    seen = set()
    total = 0

    def target(name, size):
        nonlocal total
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe archive path: {name}")
        if strip_prefix:
            if not name.startswith(strip_prefix + "/"):
                raise ValueError(f"Archive member outside expected prefix: {name}")
            name = name[len(strip_prefix) + 1:]
        if not name or name in seen or name == ".vlaworldeval-archive.json":
            raise ValueError(f"Empty, duplicate or reserved archive member: {name}")
        seen.add(name)
        total += size
        if size < 0 or total > 20 * 1024**3:
            raise ValueError("Archive exceeds the 20 GiB uncompressed asset limit")
        result = stage / name
        if not result.resolve().is_relative_to(stage):
            raise ValueError("Archive escapes extraction directory")
        result.parent.mkdir(parents=True, exist_ok=True)
        return name, result

    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as source:
                for member in source.infolist():
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                        raise ValueError("Archive links and special files are forbidden")
                    if member.is_dir():
                        continue
                    name, output = target(member.filename, member.file_size)
                    with source.open(member) as stream, output.open("xb") as out:
                        shutil.copyfileobj(stream, out)
                    records[name] = file_record(output)
        else:
            with tarfile.open(archive) as source:
                for member in source:
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError("Archive links and special files are forbidden")
                    name, output = target(member.name, member.size)
                    with source.extractfile(member) as stream, output.open("xb") as out:
                        shutil.copyfileobj(stream, out)
                    records[name] = file_record(output)
        if "saved_model.pb" not in records or not any(n.startswith("variables/") for n in records):
            raise ValueError("Archive does not contain a complete TensorFlow SavedModel")
        report = {"schema_version": 1, "archive_sha256": file_record(archive)["sha256"],
                  "upstream_gcs_md5": expected_md5, "files": records}
        if destination.exists():
            if not destination.is_dir():
                raise ValueError("Destination is not a directory")
            # Adopt legacy files only after byte-for-byte verification against
            # this pinned archive. Never overwrite or remove user assets.
            for name, expected in records.items():
                current = destination / name
                if (not current.resolve().is_relative_to(destination.resolve()) or not current.is_file()
                        or file_record(current) != expected):
                    raise ValueError(f"Existing asset differs or is incomplete: {name}; choose a new destination")
            marker = destination / ".vlaworldeval-archive.json"
            if marker.is_symlink():
                raise ValueError("Archive manifest must not be a symlink")
            if marker.exists():
                if json.loads(marker.read_text()) != report:
                    raise ValueError("Existing archive provenance differs; preserving it")
            else:
                publish_json_exclusive(marker, report)
        else:
            publish_json_exclusive(stage / ".vlaworldeval-archive.json", report)
            stage.rename(destination)
        return {"ok": True, "directory": str(destination), "files_verified": len(records), **report}
    finally:
        if stage.exists():
            shutil.rmtree(stage)  # Only our newly-created private extraction stage.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--strip-prefix", default="")
    args = parser.parse_args()
    try:
        report = extract_archive(args.archive, args.destination, root=args.root,
                                 expected_md5=args.md5, strip_prefix=args.strip_prefix)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
