import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from evaluation.execution.droid_study import choose_layout, policy_gate, run_study
from evaluation.planning.configuration import ConfigurationError
from evaluation.planning.droid_study import load_study_spec, prepare_study, study_suite
from evaluation.reporting.task_scores import _digest

pytestmark = pytest.mark.process


def template():
    return {"dataset": {}, "runtime": {}, "suite_id": "example"}


def spec_fixture():
    return json.loads(
        (
            Path(__file__).resolve().parents[2] / "configs/studies/droid_positive_controls_v1.json"
        ).read_text()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("study_id", "../../escape"),
        ("study_id", ""),
        ("schema_version", True),
        ("episodes", [0, 1, 2, 3, True]),
        ("episodes", [0, 1, 2, 3, 3]),
        ("episodes", {}),
        ("episodes", [0]),
        ("comparison_source_ticks", -240),
        ("comparison_source_ticks", 0),
        ("comparison_source_ticks", 241),
        ("start_frame", -1),
        ("start_frame", False),
        ("fps", "15"),
        ("seed", True),
        ("policies", ["unknown"]),
        ("policies", []),
        ("world_models", ["ctrl", "ctrl"]),
        ("allowed_gpus", [0, 1, 2]),
        ("allowed_gpus", [0, 1, 2, True]),
        ("memory_gib", {"ctrl": 24, "cosmos": 16, "vla": float("nan")}),
        ("memory_gib", {"ctrl": 24, "cosmos": 16, "vla": 0}),
        ("state_provider", "oracle"),
        ("quality_gate", "auto_admit"),
        ("dataset", ""),
    ],
)
def test_invalid_study_fields_rejected_before_output(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr("evaluation.planning.droid_study.project_root", lambda: tmp_path)
    spec = spec_fixture()
    spec[field] = value
    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec))
    with pytest.raises(ConfigurationError) as exc:
        prepare_study(config, tmp_path / "output")
    assert exc.value.details["issues"][0]["field"] == field or exc.value.details["issues"][0][
        "field"
    ].startswith(field + ".")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("mutation", ["missing", "unknown", "object", "json"])
def test_study_schema_errors_are_structured(tmp_path, mutation):
    spec = spec_fixture()
    if mutation == "missing":
        del spec["policies"]
    elif mutation == "unknown":
        spec["polcies"] = spec.pop("policies")
    elif mutation == "object":
        spec = []
    path = tmp_path / "config.json"
    path.write_text("{" if mutation == "json" else json.dumps(spec))
    with pytest.raises(ConfigurationError):
        load_study_spec(path)


def test_current_frozen_study_spec_is_valid(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(spec_fixture()))
    assert load_study_spec(path) == spec_fixture()


@pytest.mark.parametrize("failure", ["short_window", "missing_video"])
def test_metadata_preflight_leaves_no_partial_study(tmp_path, monkeypatch, failure):
    from evaluation.datasets.droid import CAMERAS

    monkeypatch.setattr("evaluation.planning.droid_study.project_root", lambda: tmp_path)
    spec = spec_fixture()
    meta = tmp_path / spec["dataset"] / "meta/episodes/chunk-000"
    meta.mkdir(parents=True)
    (meta / "file-000.parquet").touch()
    rows = []
    for episode in spec["episodes"]:
        row = {
            "episode_index": episode,
            "length": 10 if failure == "short_window" else 300,
            "tasks": ["fixture"],
        }
        for key in CAMERAS.values():
            row.update(
                {
                    f"videos/{key}/chunk_index": 0,
                    f"videos/{key}/file_index": 0,
                    f"videos/{key}/from_timestamp": 0.0,
                }
            )
        rows.append(row)
    pq = ModuleType("pyarrow.parquet")
    pq.read_table = lambda *a, **k: SimpleNamespace(to_pylist=lambda: rows)
    arrow = ModuleType("pyarrow")
    arrow.parquet = pq
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(spec))
    with pytest.raises(ConfigurationError if failure == "short_window" else FileNotFoundError):
        prepare_study(config, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_equal_control_ticks_not_equal_chunks():
    common = dict(study_id="fixed", episode=0, length=471, start=15, ticks=240, policy="pi0")
    a = study_suite(template(), horizon=15, wm="ctrl", **common)
    b = study_suite(template(), horizon=16, wm="cosmos", **common)
    assert a["experiments"][-1]["chunks"] == 16
    assert b["experiments"][-1]["chunks"] == 15
    assert a["metadata"]["comparison_end_frame"] == b["metadata"]["comparison_end_frame"] == 255
    assert a["experiments"][1]["chunks"] == 30
    assert b["experiments"][1]["chunks"] == 28


@pytest.mark.parametrize("length,ticks", [(150, 240), (256, 239), (255, 240)])
def test_matched_time_rejects_short_or_partial_chunks(length, ticks):
    with pytest.raises(ValueError):
        study_suite(
            template(),
            study_id="fixed",
            episode=0,
            length=length,
            start=15,
            ticks=ticks,
            horizon=16,
            policy="pi0",
            wm="cosmos",
        )


def test_resources_do_not_use_out_of_pool_free_gpus_or_drop_reserves():
    values = {0: 4000, 1: 13000, 2: 13000, 3: 13000, 7: 80000}
    assert (
        choose_layout(
            values, pool=[0, 1, 2, 3], wm_gpus=4, wm_budget=16, vla_budget=20, needs_vla=False
        )
        is None
    )
    values.update({0: 60000, 1: 30000, 2: 30000, 3: 30000})
    result = choose_layout(
        values, pool=[0, 1, 2, 3], wm_gpus=4, wm_budget=16, vla_budget=20, needs_vla=True
    )
    assert result["vla_gpus"] == [0]
    assert set(result["world_model_gpus"]) == {0, 1, 2, 3}


def test_policy_requires_review_bound_to_executed_positive_control(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.execution.droid_study.project_root", lambda: tmp_path)
    entry = {"episode": 0, "world_model": "cosmos"}
    assert policy_gate(entry, tmp_path) == "pending_review"
    evidence = tmp_path / "report.json"
    evidence.write_text(
        json.dumps(
            {
                "ok": True,
                "entry": {**entry, "control_owner": True},
                "execution_performed": True,
                "execution": {
                    "jobs": [
                        {"id": k, "status": "completed", "world_model_calls": 1, "vla_calls": 0}
                        for k in ("real_action_smoke", "real_action_full", "real_action_matched")
                    ]
                },
            }
        )
    )
    review = {
        **entry,
        "decision": "admit",
        "reviewer": "fixture-not-scientific-evidence",
        "reason": "fixture",
        "evidence_report": "report.json",
        "evidence_sha256": _digest(evidence),
        "reference_task_visible": True,
        "action_following": True,
        "visual_consistency": True,
        "long_rollout_plausible": True,
    }
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({"schema_version": 1, "records": [review]}))
    assert policy_gate(entry, tmp_path) == "admit"
    original = json.loads(evidence.read_text())
    for field, value in (("status", "failed"), ("world_model_calls", 0), ("vla_calls", 1)):
        modified = json.loads(json.dumps(original))
        modified["execution"]["jobs"][0][field] = value
        evidence.write_text(json.dumps(modified))
        review["evidence_sha256"] = _digest(evidence)
        path.write_text(json.dumps({"schema_version": 1, "records": [review]}))
        with pytest.raises(ValueError, match="completed WM-only"):
            policy_gate(entry, tmp_path)
    evidence.write_text(json.dumps(original))
    review["evidence_sha256"] = _digest(evidence)
    review["action_following"] = False
    path.write_text(json.dumps({"schema_version": 1, "records": [review]}))
    with pytest.raises(ValueError, match="plausibility"):
        policy_gate(entry, tmp_path)
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        policy_gate(entry, tmp_path)


@pytest.mark.parametrize("query_failure", [False, True])
def test_queue_waiting_does_not_launch_and_protects_frozen_inputs(
    tmp_path, monkeypatch, query_failure
):
    monkeypatch.setattr("evaluation.execution.droid_study.project_root", lambda: tmp_path)

    def query(*args, **kwargs):
        if query_failure:
            raise subprocess.TimeoutExpired("nvidia-smi", 10)
        return "0, 1000\n1, 1000\n2, 1000\n3, 1000\n"

    monkeypatch.setattr("evaluation.execution.droid_study.subprocess.check_output", query)
    monkeypatch.setattr(
        "evaluation.execution.droid_study.evaluate_suite",
        lambda *a, **k: pytest.fail("must not launch"),
    )
    directory = tmp_path / "study"
    directory.mkdir()
    suite = tmp_path / "suite.json"
    suite.write_text("{}")
    entry = {
        "id": "case",
        "control_owner": True,
        "episode": 0,
        "world_model": "cosmos",
        "config": "suite.json",
        "sha256": _digest(suite),
    }
    (directory / "study.json").write_text(
        json.dumps(
            {
                "spec": {"allowed_gpus": [0, 1, 2, 3], "memory_gib": {"cosmos": 16, "vla": 20}},
                "entries": [entry],
            }
        )
    )
    report = run_study(directory)
    assert report["status"] == "waiting_for_gpu"
    assert report["entries"]["case"]["status"] == "waiting_for_gpu"
    assert ("resource_error" in report["entries"]["case"]) == query_failure
    assert not (directory / "runs").exists()
    assert run_study(directory)["status"] == "waiting_for_gpu"
    pipeline = run_study(directory, phase="all")
    assert pipeline["status"] == "waiting_for_gpu"
    assert "policies" not in pipeline
    suite.write_text("modified")
    with pytest.raises(ValueError, match="Frozen"):
        run_study(directory)
