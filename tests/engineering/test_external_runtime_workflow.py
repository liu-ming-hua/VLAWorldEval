import json

import pytest

from evaluation.execution.workflow import evaluate_suite
from vlaworldeval.worldgym_runtime import runtime_directory, verify_runtime


def test_missing_or_forged_runtime_is_not_admitted(tmp_path):
    with pytest.raises(ValueError, match="worldgym_runtime.sh"):
        verify_runtime(tmp_path)
    directory = runtime_directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / "runtime.json").write_text(
        json.dumps({"schema_version": 1, "revision": "fake", "files": {}})
    )
    with pytest.raises(ValueError, match="manifest"):
        verify_runtime(tmp_path)
    (directory / "runtime.json").write_text("[]")
    with pytest.raises(ValueError, match="manifest"):
        verify_runtime(tmp_path)


def test_workflow_preflight_failure_never_starts_executor(monkeypatch):
    monkeypatch.setattr(
        "evaluation.execution.workflow.diagnose_suite", lambda *a, **k: {"ok": False}
    )
    report = evaluate_suite("bad.json", execute=True)
    assert report["status"] == "preflight_failed" and not report["execution_performed"]


def test_workflow_check_only_admits_data_without_gpu_or_executor(monkeypatch):
    seen = {}

    def diagnose(*args, **kwargs):
        seen.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("evaluation.execution.workflow.diagnose_suite", diagnose)
    report = evaluate_suite("suite.json", experiments=["loop_smoke"])
    assert report["ok"] and report["status"] == "checked_not_executed"
    assert seen["check_data"] and not seen["check_gpu"]
    assert not report["execution_performed"]


def test_workflow_relative_config_uses_asset_root_not_current_directory(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr("evaluation.execution.workflow.project_root", lambda: tmp_path)

    def diagnose(config, **kwargs):
        seen.append(config)
        return {"ok": True}

    monkeypatch.setattr("evaluation.execution.workflow.diagnose_suite", diagnose)
    evaluate_suite("configs/example.json")
    assert seen == [tmp_path / "configs/example.json"]


def test_workflow_execution_failure_propagates(monkeypatch):
    monkeypatch.setattr(
        "evaluation.execution.workflow.diagnose_suite", lambda *a, **k: {"ok": True}
    )

    def fail(*a, **k):
        raise RuntimeError("model failed")

    monkeypatch.setattr("evaluation.execution.executor.execute_suite", fail)
    with pytest.raises(RuntimeError, match="model failed"):
        evaluate_suite("suite.json", execute=True)


@pytest.mark.parametrize("audit_ok", [True, False])
def test_workflow_completed_result_and_failed_audit_are_distinct(tmp_path, monkeypatch, audit_ok):
    monkeypatch.setattr(
        "evaluation.execution.workflow.diagnose_suite", lambda *a, **k: {"ok": True}
    )
    (tmp_path / "rollout.mp4").touch()
    summary = {"status": "completed", "jobs": [{"id": "loop", "output": str(tmp_path)}]}
    monkeypatch.setattr("evaluation.execution.executor.execute_suite", lambda *a, **k: summary)
    monkeypatch.setattr(
        "evaluation.reporting.artifact_checks.audit_job", lambda record: {"ok": audit_ok}
    )
    monkeypatch.setattr(
        "evaluation.reporting.artifact_checks.audit_attempt", lambda record: {"ok": audit_ok}
    )
    report = evaluate_suite("suite.json", execute=True)
    assert report["execution_performed"] and report["ok"] == audit_ok
    assert report["artifacts"][0]["videos"] == [str(tmp_path / "rollout.mp4")]
    assert report["status"] == ("completed" if audit_ok else "artifact_audit_failed")
