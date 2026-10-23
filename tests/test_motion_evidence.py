"""Observation scoring cannot float free from its original video and reference."""
import json

import pytest

from evaluation.reporting.motion_detector import score_annotations
from evaluation.reporting.task_scores import _digest


def evidence(tmp_path):
    config = tmp_path/"config.json"
    config.write_text(json.dumps({"video": {"fps": 10}}))
    video, reference = tmp_path/"camera.mp4", tmp_path/"reference.npz"
    video.write_bytes(b"fixture video, not decoded in annotation mode")
    reference.write_bytes(b"fixture original reference")
    summary = tmp_path/"summary.json"
    summary.write_text(json.dumps({"status": "completed", "cleanup": {"returncode": 0, "errors": [], "signals": []},
        "config_sha256": _digest(config), "files": {"camera.mp4": _digest(video)},
        "reference_evaluation": {"reference_sha256": _digest(reference), "metrics": {"camera": {"frames_scored": 2}}}}))
    observations = {"schema_version": 1, "summary_sha256": _digest(summary),
        "reference_sha256": _digest(reference), "evaluator": {"identity": "synthetic unit-test observations, not human review"},
        "views": {"camera": {"gripper": {"prediction": [[.1, .1], [.2, .1], [.3, .1]],
            "reference": [[.1, .1], [.2, .1], [.3, .1]],
            "prediction_valid": [True]*3, "reference_valid": [True]*3}}}}
    annotation = tmp_path/"tracks.json"
    annotation.write_text(json.dumps(observations))
    return summary, reference, annotation


def test_scoring_bound_observations_requires_no_model_libraries(tmp_path):
    summary, reference, annotation = evidence(tmp_path)
    r = score_annotations(summary, reference, annotation, tmp_path/"motion.json")
    assert r["tracks"]["camera"]["gripper"]["ade"] == 0
    assert r["event_status"] == "not_annotated"
    assert not r["physical_action_tracking_verified"] and not r["official_ewmbench_score"]


@pytest.mark.parametrize("target", ["summary", "reference", "video", "config"])
def test_changed_artifact_cannot_reuse_tracks(tmp_path, target):
    summary, reference, annotation = evidence(tmp_path)
    path = {"summary": summary, "reference": reference,
            "video": tmp_path/"camera.mp4", "config": tmp_path/"config.json"}[target]
    path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError):
        score_annotations(summary, reference, annotation, tmp_path/"motion.json")
    assert not (tmp_path/"motion.json").exists()
