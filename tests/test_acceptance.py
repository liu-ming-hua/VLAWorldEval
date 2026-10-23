"""Acceptance opt-in, recorded stages, failure propagation and artifact audits."""
import json
from pathlib import Path

import pytest
import numpy as np

import evaluation.validation.acceptance as acceptance
from evaluation.reporting.artifact_checks import audit_attempt, audit_job
from evaluation.planning.configuration import ConfigurationError


@pytest.mark.parametrize("level,execute,pairs", [("smoke", False, ["one"]), ("long", False, ["one"]),
                                                ("smoke", True, []), ("long", True, []), ("local", True, [])])
def test_inference_requires_explicit_flags_before_any_report(tmp_path, monkeypatch, level, execute, pairs):
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    with pytest.raises(ValueError):
        acceptance.run_acceptance(tmp_path / "absent.json", level=level, execute=execute, pairs=pairs)
    assert not (tmp_path / "results").exists()


def test_real_matrix_is_weight_free_and_excludes_ablations():
    matrix = acceptance.load_acceptance_matrix(Path("configs/acceptance/framework_v1.json"))
    assert len(matrix["entries"]) == 5
    assert all(len(entry["smoke"]) == 2 and len(entry["long"]) == 1 for entry in matrix["entries"])


@pytest.mark.parametrize("cleanup,expected", [
    ({"returncode": 0, "errors": [], "signals": []}, True),
    ({"returncode": 0, "errors": [], "signals": ["SIGTERM"]}, False),
    ({"returncode": 0, "errors": []}, False),
])
def test_attempt_audit_rejects_forced_or_unproven_cleanup(monkeypatch, cleanup, expected):
    monkeypatch.setattr("evaluation.reporting.artifact_checks.audit_job", lambda *a: {"ok": True})
    report = audit_attempt({"status": "completed", "jobs": [{"fixture": True}],
                            "worker_cleanup": [{"status": "closed", "cleanup": cleanup}]})
    assert report["cleanup_ok"] is expected
    assert report["ok"] is expected


def test_invalid_matrix_rejected(tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": [{"id": "broken"}]}))
    with pytest.raises(ConfigurationError, match="Required fields"):
        acceptance.load_acceptance_matrix(path)


def setup_matrix(tmp_path, monkeypatch):
    path = tmp_path / "matrix.json"
    path.write_text("{}")
    data = {"schema_version": 1, "entries": [
        {"id": name, "suite": name + ".json", "smoke": ["replay", "loop"], "long": ["long"], "seeds": [0]}
        for name in ("one", "two")]}
    monkeypatch.setattr(acceptance, "ROOT", tmp_path)
    monkeypatch.setattr(acceptance, "load_acceptance_matrix", lambda *args: data)
    return path


def test_default_local_has_no_model_execution_and_unique_reports(tmp_path, monkeypatch):
    path = setup_matrix(tmp_path, monkeypatch)
    monkeypatch.setattr(acceptance, "diagnose_suite", lambda *args, **kwargs: {"ok": True, "model_loading_tested": False})
    def forbidden(*args, **kwargs):
        raise AssertionError("No model execution in local acceptance")
    monkeypatch.setattr("evaluation.execution.executor.execute_suite", forbidden)
    first, second = acceptance.run_acceptance(path), acceptance.run_acceptance(path)
    assert first["report"] != second["report"]
    assert first["ok"] and not first["inference_requested"]
    assert all(entry["status"] == "local_passed" and not entry["inference_completed"] for entry in first["entries"])
    assert Path(first["report"]).is_file()


def test_preflight_failures_do_not_pass_or_start_models(tmp_path, monkeypatch):
    path = setup_matrix(tmp_path, monkeypatch)
    monkeypatch.setattr(acceptance, "diagnose_suite", lambda *args, **kwargs: {"ok": False})
    report = acceptance.run_acceptance(path, level="preflight")
    assert not report["ok"] and len(report["entries"]) == 2
    assert all(entry["status"] == "failed_preflight" for entry in report["entries"])


def test_execution_failure_keeps_exact_attempt_and_stops_batch(tmp_path, monkeypatch):
    path = setup_matrix(tmp_path, monkeypatch)
    monkeypatch.setattr(acceptance, "diagnose_suite", lambda *args, **kwargs: {"ok": True})
    def failed(*args, on_attempt, **kwargs):
        on_attempt(tmp_path / "exact_attempt")
        raise RuntimeError("injected execution failure")
    monkeypatch.setattr("evaluation.execution.executor.execute_suite", failed)
    report = acceptance.run_acceptance(path, level="smoke", execute=True, pairs=["one", "two"])
    assert report["status"] == "failed"
    assert report["entries"][0]["attempt"] == str(tmp_path / "exact_attempt")
    assert report["not_run_after_failure"] == ["two"]


def test_successful_execution_with_bad_artifacts_is_not_accepted(tmp_path, monkeypatch):
    path = setup_matrix(tmp_path, monkeypatch)
    monkeypatch.setattr(acceptance, "diagnose_suite", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr("evaluation.execution.executor.execute_suite", lambda *args, **kwargs: {"status": "completed"})
    monkeypatch.setattr("evaluation.reporting.artifact_checks.audit_attempt", lambda *args, **kwargs: {"ok": False, "failures": ["bad action"]})
    report = acceptance.run_acceptance(path, level="smoke", execute=True, pairs=["one"])
    assert not report["ok"] and report["entries"][0]["status"] == "failed_artifacts"


def test_interrupt_persists_report_then_propagates(tmp_path, monkeypatch):
    path = setup_matrix(tmp_path, monkeypatch)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("injected interruption")
    monkeypatch.setattr(acceptance, "diagnose_suite", interrupted)
    with pytest.raises(KeyboardInterrupt):
        acceptance.run_acceptance(path)
    saved = json.loads(next((tmp_path / "results/acceptance").glob("*/report.json")).read_text())
    assert saved["status"] == "interrupted" and not saved["ok"]


def test_missing_artifacts_are_a_failed_audit(tmp_path):
    result = audit_job({"output": str(tmp_path)})
    assert not result["ok"] and result["failures"]
    result = audit_attempt({"status": "completed", "jobs": [], "worker_cleanup": []})
    assert not result["ok"] and not result["quality_evaluated"]


@pytest.mark.parametrize("damage", ["valid", "nonfinite", "views", "frames", "handoff"])
def test_artifact_audit_rejects_corruption(tmp_path, monkeypatch, damage):
    action = {"dimension": 2, "horizon": 1, "rate_hz": 20}
    pair = {"vla": {"action": action}, "world_model": {"action": action, "output_views": ["primary"]},
            "view_bridge": {"name": "identity", "config": {}}}
    job = {"chunks": 2, "mode": "policy_loop", "state_provider": "none"}
    metadata = {"format_version": 2, "num_rollout_steps": 2, "num_video_frames": 3,
                "video_fps": 20, "trial": {"partial": False}, "view_files": {"primary": "views/primary.mp4"}}
    if damage == "views":
        metadata["view_files"] = {}
    for filename, value in (("pair_manifest.json", pair), ("job.json", job), ("metadata.json", metadata)):
        (tmp_path / filename).write_text(json.dumps(value))
    values = np.ones((2, 1, 2), dtype=np.float32)
    if damage == "nonfinite":
        values[0, 0, 0] = np.nan
    arrays = {}
    for prefix in ("native", "bridged"):
        arrays[f"{prefix}_actions"] = values
        arrays[f"{prefix}_action_lengths"] = np.ones(2, dtype=np.int32)
        arrays[f"{prefix}_action_mask"] = np.ones((2, 1), dtype=bool)
    np.savez(tmp_path / "actions.npz", **arrays)
    np.savez(tmp_path / "states.npz", state_values=np.empty((3, 0)), state_mask=np.zeros((3, 0), dtype=bool))
    journal = tmp_path / "partial"
    journal.mkdir()
    for index in range(2):
        (journal / f"chunk_{index:05d}.json").write_text(json.dumps({
            "output_terminal_view_hashes": {"primary": "generated"},
            "input_view_hashes": {"primary": "wrong" if damage == "handoff" else "generated"},
            "observation_input_source": "previous_generated_terminal_views"}))
    monkeypatch.setattr("evaluation.reporting.artifact_checks.subprocess.check_output", lambda *args, **kwargs:
                        json.dumps({"streams": [{"nb_frames": "2" if damage == "frames" else "3", "avg_frame_rate": "20/1"}]}))
    result = audit_job({"output": str(tmp_path), "status": "completed", "completed_chunks": 2,
                        "world_model_calls": 2, "vla_calls": 2})
    assert result["ok"] == (damage == "valid")
