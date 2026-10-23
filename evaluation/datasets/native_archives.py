"""Bounded extraction of a frozen native split, never archive-wide unpacking."""
from __future__ import annotations

import io
import json
from pathlib import Path
import re
import shutil
import tarfile

from vlaworldeval.assets import MANIFEST, file_record, verify_snapshot
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json
from evaluation.reporting.task_scores import _digest

IRASIM_TRAIN_PARTS = 100


class ShardStream(io.RawIOBase):
    """Read preverified local parts in order without joining another200GB file."""

    def __init__(self, parts):
        super().__init__()
        self.parts, self.current = iter(parts), None

    def readable(self):
        return True

    def read(self, size=-1):
        if self.closed or size < 0:
            raise ValueError("Use bounded reads on an open shard stream")
        result = bytearray()
        while len(result) < size:
            if self.current is None:
                path = next(self.parts, None)
                if path is None:
                    break
                self.current = Path(path).open("rb")
            block = self.current.read(size-len(result))
            if block:
                result.extend(block)
            else:
                self.current.close()
                self.current = None
        return bytes(result)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def extract_irasim_test_assets(root, transport, annotations, output):
    """Select every published test video/latent named in the frozen annotations.

    Files retain their exact upstream bytes. No VAE decoding, synthetic images,
    missing-case substitution or automatic removal of incomplete extraction.
    """
    from evaluation.planning.checkpoint import IRASIM_TEST_COUNTS
    root, transport, annotations = map(lambda p: Path(p).resolve(), (root, transport, annotations))
    output = Path(output).absolute()
    if output.is_symlink() or output.exists() or not output.resolve().is_relative_to(root) or output.resolve() == root:
        raise ValueError("Choose a new project-local split asset directory")
    verified = verify_snapshot(transport)
    source_manifest = json.loads((transport/MANIFEST).read_text())
    if verified["repository"] != "fangqi/IRASim" or verified["revision"] != "dfcbf85c27b2c5d041dbf5df3df272111e758b48":
        raise ValueError("Native test extraction requires the pinned official archive snapshot")
    names = sorted(n for n in source_manifest["files"] if re.fullmatch(r"languagetable_train_data\.tar\.gz\.sfpart_[a-z]{2}", n))
    expected = [f"languagetable_train_data.tar.gz.sfpart_{chr(97+i//26)}{chr(97+i%26)}" for i in range(IRASIM_TRAIN_PARTS)]
    if names != expected:
        raise ValueError("Complete consecutive training archive parts required; no gaps or reordered concatenation")
    selection_path = annotations/"coverage.json"
    selection = json.loads(selection_path.read_text())
    episodes, clips = IRASIM_TEST_COUNTS
    if (selection.get("test_episodes") != episodes or selection.get("official_short_clips") != clips
            or len(selection.get("files", {})) != episodes or selection.get("revision") != verified["revision"]
            or selection["source_shard"]["sha256"] != source_manifest["files"][names[0]]["sha256"]):
        raise ValueError("Frozen complete test annotations do not match the archive identity")
    required = set()
    for stem, record in selection["files"].items():
        paths = (f"videos/test/{stem}/rgb.mp4", f"latent_videos/test/{stem}/0.pt")
        if (not re.fullmatch(r"[0-9]{6}", stem) or record.get("video_path") != paths[0]
                or record.get("latent_video_path") != paths[1]
                or record.get("path") != f"annotation/test/{stem}.json"
                or _digest(annotations/record["path"]) != record["sha256"]):
            raise ValueError("Test annotation checksum or native path differs")
        required.update(paths)
    records, total = {}, 0
    prefix = "opensource_robotdata/languagetable/"
    with asset_destination_lock(root, output):
        output.mkdir(parents=True)
        progress = {"schema_version": 1, "status": "extracting", "expected_payload_files": len(required),
                    "extracted_payload_files": 0, "selection_sha256": _digest(selection_path),
                    "model_inference_performed": False}
        journal = output/"progress.json"
        publish_json_exclusive(journal, progress)
        try:
            for record in selection["files"].values():
                path = output/record["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(annotations/record["path"], path)
                records[record["path"]] = file_record(path)
            with ShardStream(transport/name for name in names) as stream, tarfile.open(fileobj=stream, mode="r|gz") as archive:
                for member in archive:
                    relative = member.name[len(prefix):] if member.name.startswith(prefix) else None
                    if relative not in required:
                        continue
                    if not member.isfile() or relative in records or not 0 < member.size <= 1024**3:
                        raise ValueError("Invalid/duplicate/oversized selected archive member")
                    total += member.size
                    if total > 512*1024**3:
                        raise ValueError("Selected native test payload exceeds512GiB bound")
                    path = output/relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name+".partial")
                    with archive.extractfile(member) as source, temporary.open("xb") as destination:
                        shutil.copyfileobj(source, destination, length=8*1024**2)
                    if temporary.stat().st_size != member.size:
                        raise ValueError("Truncated test asset; preserve partial extraction for audit")
                    temporary.rename(path)
                    records[relative] = file_record(path)
                    progress.update(extracted_payload_files=progress["extracted_payload_files"]+1, last_member=relative)
                    if progress["extracted_payload_files"] % 128 == 0:
                        replace_owned_json(journal, progress)
                    if required <= records.keys():
                        break
            if not required <= records.keys():
                raise ValueError(f"Missing{len(required-records.keys())} frozen test assets; no case replacement")
            report = {"schema_version": 1, "status": "extracted_not_inferred", "source_snapshot": verified,
                "archive_manifest_sha256": _digest(transport/MANIFEST), "annotation_manifest_sha256": _digest(selection_path),
                "annotation_root": str(annotations), "test_episodes": episodes, "official_short_clips": clips,
                "files": records, "payload_bytes": total, "unselected_members_extracted": False,
                "media_and_tensor_validation": "pending native case preparation", "model_inference_performed": False,
                "full_test_set_completed": False}
            publish_json_exclusive(output/"coverage.json", report)
            progress["status"] = "completed"
        except BaseException as exc:
            progress.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            replace_owned_json(journal, progress)
    return {"test_episodes": episodes, "payload_files": len(required), "output": str(output), "model_inference_performed": False}
