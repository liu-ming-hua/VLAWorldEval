"""Dense review exports cannot create inference or turn previews into approval."""

import json
import subprocess
import sys
from pathlib import Path

import mediapy as media
import numpy as np
import pytest

from evaluation.reporting.cosmos_temporal_review import (
    export_cosmos_temporal_review,
    read_native_prefix,
)
from evaluation.reporting.task_scores import _digest

pytestmark = pytest.mark.process


@pytest.mark.parametrize(
    "flag,value",
    [("--output", "unused"), ("--case", "eef_ep000351"), ("--resource-overrides", "unused.json")],
)
def test_review_cli_rejects_silently_ignored_options(flag, value):
    script = Path(__file__).resolve().parents[2] / "scripts/checks/cosmos_horizon.py"
    result = subprocess.run(
        [sys.executable, str(script), "audit-review", "--directory", "unused", flag, value],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "is only valid for" in result.stderr


def test_native_decoder_never_resizes_or_pads(tmp_path):
    path = tmp_path / "input.mp4"
    media.write_video(path, np.full((8, 24, 16, 3), 64, np.uint8), fps=15)
    frames = read_native_prefix(path, 5)
    assert frames.shape == (5, 24, 16, 3)
    assert np.max(np.abs(frames.astype(int) - 64)) < 4
    with pytest.raises(ValueError, match="Incomplete"):
        read_native_prefix(path, 9)
    for count in (True, 0, -1):
        with pytest.raises(ValueError, match="positive integer"):
            read_native_prefix(path, count)
    media.write_video(tmp_path / "wrong_fps.mp4", frames, fps=10)
    with pytest.raises(ValueError, match="15FPS"):
        read_native_prefix(tmp_path / "wrong_fps.mp4", 5)


@pytest.fixture
def review(tmp_path, monkeypatch):
    import evaluation.reporting.cosmos_temporal_review as module

    parent = tmp_path / "parent"
    parent.mkdir()
    inputs = {}
    for name in (
        "real.wrist",
        "real.exterior_1",
        "real.exterior_2",
        "seed0_historical",
        "seed1",
        "seed2",
    ):
        path = parent / (name + ".mp4")
        path.write_bytes(b"fixture input, not model inference")
        inputs[name] = {"path": str(path), "sha256": _digest(path)}
    (parent / "comparisons.json").write_text(
        json.dumps({"cases": [{"case": "eef_ep000351", "model": "eef", "inputs": inputs}]})
    )
    for filename in ("quality_review_working.json", "execution_audit.json"):
        (parent / filename).write_text("{}")
    summary = {"status": "completed_execution_and_assistant_screening_not_quality_approval"}
    for field, filename in (
        ("review_sha256", "quality_review_working.json"),
        ("execution_audit_sha256", "execution_audit.json"),
        ("comparisons_sha256", "comparisons.json"),
    ):
        summary[field] = _digest(parent / filename)
    (parent / "summary.json").write_text(json.dumps(summary))

    def decode(path, count):
        height = 360 if Path(path).name.startswith("real.") else 540
        return np.broadcast_to(np.array([64, 96, 128], np.uint8), (count, height, 640, 3))

    monkeypatch.setattr(module, "read_native_prefix", decode)
    return tmp_path, parent


def test_review_contains_every_frame_and_preserves_parent_without_approval(review):
    root, parent = review
    before = {p.name: _digest(p) for p in parent.iterdir()}
    output = root / "review"
    result = export_cosmos_temporal_review(root, parent, output, cases=("eef_ep000351",))
    assert result["new_world_model_calls"] == result["vla_calls"] == 0
    assert result["quality_admitted"] is result["human_review_performed"] is False
    row = result["cases"][0]
    assert row["frames"] == 65 and row["native_eef_chunks"] == 4
    assert [t for page in row["sheets"] for t in page["source_ticks"]] == list(range(65))
    assert all(_digest(output / row["case"] / name) == sha for name, sha in row["files"].items())
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames,pix_fmt",
                "-of",
                "json",
                str(output / row["case"] / "comparison.mp4"),
            ],
            text=True,
        )
    )["streams"][0]
    assert (
        stream["width"],
        stream["height"],
        stream["nb_read_frames"],
        stream["avg_frame_rate"],
        stream["pix_fmt"],
    ) == (2560, 568, "65", "15/1", "yuv420p")
    assert {p.name: _digest(p) for p in parent.iterdir()} == before
    assert "requestVideoFrameCallback" in (output / "index.html").read_text()
    with pytest.raises(FileExistsError):
        export_cosmos_temporal_review(root, parent, output, cases=("eef_ep000351",))
    from evaluation.reporting.cosmos_temporal_review import audit_cosmos_temporal_review

    assert audit_cosmos_temporal_review(root, output)["quality_admitted"] is False
    notes = {
        "reviewer": "assistant",
        "human_review_performed": False,
        "full_native_frame_review_performed": False,
        "cases": [
            {
                "case": row["case"],
                "inspected_sheets": [p["file"] for p in row["sheets"]],
                "native_ticks": [32, 64],
                "quality_admitted": False,
                "task_success": None,
            }
        ],
    }
    (output / "observations.json").write_text(json.dumps(notes))
    assert audit_cosmos_temporal_review(root, output)["observations_sha256"] == _digest(
        output / "observations.json"
    )
    notes["cases"][0]["quality_admitted"] = True
    (output / "observations.json").write_text(json.dumps(notes))
    with pytest.raises(ValueError, match="implicit quality approval"):
        audit_cosmos_temporal_review(root, output)
    (output / "observations.json").unlink()
    (output / row["case"] / row["sheets"][0]["file"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Review artifact changed"):
        audit_cosmos_temporal_review(root, output)


@pytest.mark.parametrize("fault", ["parent", "input", "case", "ticks", "escape", "duplicate"])
def test_review_fails_closed_before_creating_results(review, fault):
    root, parent = review
    cases, ticks = ("eef_ep000351",), 64
    if fault == "parent":
        (parent / "execution_audit.json").write_text('{"changed":true}')
    elif fault == "input":
        (parent / "seed1.mp4").write_bytes(b"changed")
    elif fault == "case":
        cases = ("eef_ep999999",)
    elif fault == "ticks":
        ticks = True
    elif fault == "escape":
        cases = ("../escape",)
    else:
        cases = ("eef_ep000351", "eef_ep000351")
    with pytest.raises(ValueError):
        export_cosmos_temporal_review(
            root, parent, root / "review", cases=cases, source_ticks=ticks
        )
    assert not (root / "review").exists()
