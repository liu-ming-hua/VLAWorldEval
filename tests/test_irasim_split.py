"""Verified archive prefixes cannot be relabelled as complete RGB benchmarks."""
import io
import json
import tarfile

import pytest

from evaluation.planning import checkpoint
from vlaworldeval.assets import MANIFEST


def transport(tmp_path, monkeypatch, *, problem=None):
    from vlaworldeval import paths
    monkeypatch.setattr(paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(checkpoint, "IRASIM_TEST_COUNTS", (2, 2))
    monkeypatch.setattr(checkpoint, "verify_snapshot", lambda _: {
        "repository": "fangqi/IRASim", "revision": "dfcbf85c27b2c5d041dbf5df3df272111e758b48"})
    directory = tmp_path/"transport"
    directory.mkdir()
    name = "languagetable_train_data.tar.gz.sfpart_aa"
    (directory/MANIFEST).write_text(json.dumps({"files": {name: {"fixture": True}}}))
    with tarfile.open(directory/name, "w:gz") as archive:
        for index in range(1 if problem == "missing" else 2):
            stem = f"{index:06d}"
            annotation = {"episode_id": stem, "video_path": f"videos/test/{stem}/rgb.mp4",
                "latent_video_path": f"latent_videos/test/{stem}/0.pt", "actions": [[0., .01]]*17}
            if problem == "path":
                annotation["video_path"] = "../outside.mp4"
            payload = json.dumps(annotation).encode()
            member = tarfile.TarInfo(f"opensource_robotdata/languagetable/annotation/test/{stem}.json")
            member.size = len(payload)
            if problem == "link":
                member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
            archive.addfile(member, io.BytesIO(payload))
        boundary = tarfile.TarInfo("opensource_robotdata/languagetable/annotation/train")
        boundary.type = tarfile.DIRTYPE
        archive.addfile(boundary)
    return directory


def test_complete_native_selection_does_not_claim_rgb_or_inference(tmp_path, monkeypatch):
    source = transport(tmp_path, monkeypatch)
    output = tmp_path/"prepared"
    report = checkpoint.extract_irasim_test_annotations(source, output)
    assert report == {"test_episodes": 2, "official_short_clips": 2,
                      "full_archive_downloaded": False, "model_inference_performed": False}
    saved = json.loads((output/"coverage.json").read_text())
    assert len(saved["clips"]) == 2 and not saved["rgb_and_latents_acquired"]
    assert saved["clips"][0]["source_frame_indices"] == list(range(16))
    with pytest.raises(ValueError, match="new project-local"):
        checkpoint.extract_irasim_test_annotations(source, output)


@pytest.mark.parametrize("problem", ["missing", "path", "link"])
def test_invalid_or_incomplete_annotation_group_is_not_published(tmp_path, monkeypatch, problem):
    source = transport(tmp_path, monkeypatch, problem=problem)
    output = tmp_path/"prepared"
    with pytest.raises(ValueError):
        checkpoint.extract_irasim_test_annotations(source, output)
    assert not (output/"coverage.json").exists()
