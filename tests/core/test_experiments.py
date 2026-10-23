import json
from pathlib import Path

import pytest

from evaluation.planning.experiments import build_experiment_plan

CONFIG = Path("configs/experiments/droid_framework_v1.json")


def test_experiment_plan_expands_seeds_and_marks_not_executed():
    plan = build_experiment_plan(CONFIG)
    assert len(plan["jobs"]) == 18
    assert plan["execution_ready"] is False
    assert plan["gpu_overlap"] == [1]
    long = next(job for job in plan["jobs"] if job["id"] == "loop_integrated_long")
    assert long["action_count"] == 320
    assert long["duration_seconds"] == pytest.approx(320 / 15)
    assert all(job["status"] == "planned_not_executed" for job in plan["jobs"])


@pytest.mark.parametrize(
    "change,match",
    [
        ("fps", "FPS"),
        ("tail", "drop_incomplete"),
        ("oracle", "state baseline"),
        ("rotation", "WM-native"),
        ("output", "leaves the project"),
        ("duplicate", "duplicate"),
        ("environment", "does not match"),
    ],
)
def test_invalid_experiment_plan_fails_before_gpu_loading(tmp_path, change, match):
    value = json.loads(CONFIG.read_text())
    if change == "fps":
        value["dataset"]["fps"] = 5
    elif change == "tail":
        value["dataset"]["tail_policy"] = "pad_actions"
    elif change == "oracle":
        value["experiments"][4]["state_provider"] = "dataset_ground_truth"
    elif change == "rotation":
        value["experiments"][0]["action_source"] = "all_channels_zero"
    elif change == "output":
        value["output_root"] = "../outside_project"
    elif change == "environment":
        value["runtime"]["vla_environment"] = "vla.octo"
    else:
        value["experiments"][1]["id"] = value["experiments"][0]["id"]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=match):
        build_experiment_plan(path)
