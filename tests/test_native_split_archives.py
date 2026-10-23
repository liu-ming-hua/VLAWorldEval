"""Multipart extraction is complete for the selected split, not arbitrary tar paths."""
import io
import json
import tarfile

import pytest

from evaluation.datasets import native_archives as module
from evaluation.planning import checkpoint
from vlaworldeval.assets import MANIFEST, file_record


def fixture(tmp_path, monkeypatch, *, missing=False, link=False):
    monkeypatch.setattr(module, "IRASIM_TRAIN_PARTS", 2)
    monkeypatch.setattr(checkpoint, "IRASIM_TEST_COUNTS", (2, 2))
    root, annotations = tmp_path/"transport", tmp_path/"annotations"
    root.mkdir()
    annotations.mkdir()
    buffer = io.BytesIO()
    files = {}
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for episode in ("000000", "000001"):
            ann = annotations/f"annotation/test/{episode}.json"
            ann.parent.mkdir(parents=True, exist_ok=True)
            ann.write_text(json.dumps({"fixture": "not inference data"}))
            names = (f"videos/test/{episode}/rgb.mp4", f"latent_videos/test/{episode}/0.pt")
            files[episode] = {"path": str(ann.relative_to(annotations)), **file_record(ann),
                              "video_path": names[0], "latent_video_path": names[1]}
            for name in names:
                if missing and name == names[1] and episode == "000001":
                    continue
                member = tarfile.TarInfo("opensource_robotdata/languagetable/"+name)
                payload = b"synthetic extraction fixture"
                member.size = len(payload)
                if link:
                    member.type, member.linkname = tarfile.SYMTYPE, "/etc/passwd"
                archive.addfile(member, io.BytesIO(payload))
        member = tarfile.TarInfo("unselected/train/large_file")
        member.size = 5
        archive.addfile(member, io.BytesIO(b"train"))
    payload = buffer.getvalue()
    names = [f"languagetable_train_data.tar.gz.sfpart_a{suffix}" for suffix in ("a", "b")]
    boundary = len(payload)//2
    for name, data in zip(names, (payload[:boundary], payload[boundary:])):
        (root/name).write_bytes(data)
    source_files = {name: file_record(root/name) for name in names}
    (root/MANIFEST).write_text(json.dumps({"files": source_files}))
    (annotations/"coverage.json").write_text(json.dumps({"test_episodes": 2, "official_short_clips": 2,
        "revision": "dfcbf85c27b2c5d041dbf5df3df272111e758b48", "files": files,
        "source_shard": source_files[names[0]]}))
    monkeypatch.setattr(module, "verify_snapshot", lambda _: {"repository": "fangqi/IRASim",
        "revision": "dfcbf85c27b2c5d041dbf5df3df272111e758b48", "files_verified": 2})
    return root, annotations


def test_split_extraction_preserves_every_required_asset_and_skips_train(tmp_path, monkeypatch):
    source, annotations = fixture(tmp_path, monkeypatch)
    output = tmp_path/"output"
    result = module.extract_irasim_test_assets(tmp_path, source, annotations, output)
    assert result["payload_files"] == 4 and not result["model_inference_performed"]
    report = json.loads((output/"coverage.json").read_text())
    assert len(report["files"]) == 6
    assert not (output/"unselected").exists() and not report["full_test_set_completed"]
    with pytest.raises(ValueError, match="new project-local"):
        module.extract_irasim_test_assets(tmp_path, source, annotations, output)


@pytest.mark.parametrize("problem", ["missing", "link"])
def test_missing_or_link_payload_never_publishes_completion(tmp_path, monkeypatch, problem):
    source, annotations = fixture(tmp_path, monkeypatch, **{problem: True})
    output = tmp_path/"output"
    with pytest.raises(ValueError):
        module.extract_irasim_test_assets(tmp_path, source, annotations, output)
    assert not (output/"coverage.json").exists()
    assert json.loads((output/"progress.json").read_text())["status"] == "failed"


def test_shard_stream_has_bounded_reads_and_correct_eof(tmp_path):
    for index, value in enumerate((b"abc", b"def")):
        (tmp_path/str(index)).write_bytes(value)
    with module.ShardStream(tmp_path/str(i) for i in range(2)) as stream:
        with pytest.raises(ValueError):
            stream.read()
        assert stream.read(4) == b"abcd"
        assert stream.read(4) == b"ef"
        assert stream.read(1) == b""
