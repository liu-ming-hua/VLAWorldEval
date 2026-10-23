"""Synthetic review artifacts validate rendering, not any robot checkpoint."""
import json

import mediapy as media
import numpy as np
import pytest

from evaluation.execution.transport import write_packet
from evaluation.reporting.image_feedback import compare_image_feedback, pixel_diagnostics, temporal_protocol_warning
from evaluation.reporting.positive_controls import VIEWS
from evaluation.reporting.task_scores import _digest


def test_legacy_ctrl_window_is_flagged_not_certified():
    assert "advances12" in temporal_protocol_warning({"model": "ctrl", "source_stride": 15})
    assert temporal_protocol_warning({"model": "ctrl", "source_stride": 12}) is None
    assert temporal_protocol_warning({"model": "cosmos", "source_stride": 16}) is None


def test_pixel_metrics_exclude_initial_and_border_not_task_success():
    real = np.zeros((5, 20, 16, 3), np.uint8)
    ar = real + 20
    anchored = real + 5
    anchored[0] = 255
    anchored[:, -12:] = 255
    result = pixel_diagnostics(real, ar, anchored, frames_per_chunk=2)
    assert result["autoregressive_mean"] == 20 and result["anchored_mean"] == 5
    assert result["by_frame_offset"]["anchored"] == [5, 5]
    assert result["task_success_evaluated"] is False
    with pytest.raises(ValueError):
        pixel_diagnostics(real, ar, anchored, frames_per_chunk=3)


def test_three_column_diagnostic_audits_and_renders(tmp_path):
    base = tmp_path / "study"
    base.mkdir()
    config = {"video": {"fps": 15, "frames_per_chunk": 2}, "output_views": ["droid."+v for v in VIEWS]}
    config_path, packet_path, anchors_path = (base / name for name in ["config.json", "input.npz", "anchors.npz"])
    config_path.write_text(json.dumps(config))
    actions = np.ones((3, 2, 7), np.float32)
    write_packet(packet_path, {"actions": actions})
    write_packet(anchors_path, {"fixture": True})
    reference = {"episode": 1, "views": {}}
    for name in VIEWS:
        p = base / f"real_{name}.mp4"
        media.write_video(p, np.zeros((7, 20, 16, 3), np.uint8), fps=15)
        reference["views"]["droid."+name] = {"file": str(p.relative_to(tmp_path)), "sha256": _digest(p)}
    old = tmp_path / "old"
    old.mkdir()
    (old / "study.json").write_text(json.dumps({"references": [reference]}))
    modes = ["autoregressive", "ground_truth_visual_context"]
    case = {"id": "fixture", "episode": 1, "chunks": 3, "source_stride": 2}
    for key, path in [("config", config_path), ("input", packet_path), ("anchors", anchors_path)]:
        case[key], case[key+"_sha256"] = str(path), _digest(path)
    for mode in modes:
        out = base / "fixture" / mode
        out.mkdir(parents=True)
        np.save(out / "actions.npy", actions)
        files = {"actions.npy": _digest(out / "actions.npy")}
        for name in VIEWS:
            p = out / f"droid.{name}.mp4"
            media.write_video(p, np.full((7, 20, 16, 3), 10, np.uint8), fps=15)
            files[p.name] = _digest(p)
        report = {"status": "completed", "cleanup": {"returncode": 0, "signals": [], "errors": []},
            "vla_calls": 0, "world_model_calls": 3, "feedback_mode": mode,
            "input_sha256": _digest(packet_path), "config_sha256": _digest(config_path), "files": files,
            "image_anchors_sha256": _digest(anchors_path), "anchor_events": [],
            "chunk_prediction_sha256": [{"fixture": "synthetic-not-model-evidence"}]*3}
        if mode != "autoregressive":
            report["anchor_events"] = [{"chunk": i, "source_frame": 15+i*2,
                "details": {"step_index": i, "state_changed": False}} for i in [1, 2]]
        (out / "summary.json").write_text(json.dumps(report))
    (base / "study.json").write_text(json.dumps({"cases": [case], "plan": {"source_study": "old", "modes": modes, "start_frame": 15}}))
    result = compare_image_feedback(tmp_path, base, "review")
    row = result["cases"][0]
    assert row["comparison_frames"] == 7 and row["initial_chunk_exact_match"] is True
    assert row["columns"] == ["real", "autoregressive", "ground_truth_visual_context"]
    assert row["task_success_evaluated"] is False and row["quality_admitted"] is None
    assert (tmp_path / "review/index.html").is_file()
    with pytest.raises(FileExistsError):
        compare_image_feedback(tmp_path, base, "review")
    # Artifact hashes alone must not allow a changed action stream.
    out = base / "fixture" / modes[1]
    np.save(out / "actions.npy", actions * 2)
    report = json.loads((out / "summary.json").read_text())
    report["files"]["actions.npy"] = _digest(out / "actions.npy")
    (out / "summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="actions differ"):
        compare_image_feedback(tmp_path, base, "bad_review")
