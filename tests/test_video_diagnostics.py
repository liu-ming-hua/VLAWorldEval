import json

import numpy as np
import pytest

from cli.main import main
from evaluation.reporting.video_diagnostics import DiagnosticThresholds, diagnose_artifact, diagnose_frames, diagnose_video


def test_static_stream_and_single_frame_are_not_success_scores():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    result = diagnose_frames((frame for _ in range(100)), fps=10)
    assert result["frames"] == 100
    assert result["near_static_transition_fraction"] == 1
    assert result["longest_near_static_seconds"] == 9.9
    assert "near_static_video" in result["warnings"]
    one = diagnose_frames([frame], fps=29.97)
    assert one["near_static_transition_fraction"] is None and not one["warnings"]
    json.dumps(one, allow_nan=False)


def test_motion_jump_noise_and_no_uint8_wraparound():
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    white = np.full_like(black, 255)
    result = diagnose_frames([black, white], fps=20)
    assert result["mean_transition_mae"] == 255
    assert result["large_jump_transitions"] == 1
    noise = np.random.default_rng(0).integers(0, 256, black.shape, dtype=np.uint8)
    assert "high_frequency_growth_review_required" in diagnose_frames([black, noise], fps=20)["warnings"]


@pytest.mark.parametrize("kwargs", [{"fps": 0}, {"fps": float("nan")}, {"fps": True}])
def test_invalid_fps(kwargs):
    with pytest.raises(ValueError):
        diagnose_frames([], **kwargs)


def test_invalid_frames_and_thresholds():
    with pytest.raises(ValueError):
        diagnose_frames([], fps=20)
    with pytest.raises(ValueError):
        diagnose_frames([np.zeros((8, 8, 3))], fps=20)
    with pytest.raises(ValueError):
        DiagnosticThresholds(static_fraction=2)
    with pytest.raises(ValueError):
        DiagnosticThresholds(jump_mae=float("inf"))


def test_real_decode_and_read_only_cli(tmp_path, capsys):
    import cv2
    video = tmp_path / "rollout.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (16, 16))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()
    (tmp_path / "metadata.json").write_text(json.dumps({"format_version": 2, "video_file": video.name}))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert main(["inspect", "--artifact", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["views"]["primary"]["frames"] == 5
    assert not report["task_success_evaluated"]
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        diagnose_video(tmp_path / "absent.mp4")


def test_path_escape_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"format_version": 2, "view_files": {"front": "../bad.mp4"}}))
    with pytest.raises(ValueError, match="leaves"):
        diagnose_artifact(tmp_path)
