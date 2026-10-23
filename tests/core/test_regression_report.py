import json

import pytest

from evaluation.reporting.regression_report import summarize_regressions


def test_missing_preflight_or_failed_reports_cannot_pass(tmp_path):
    report = tmp_path / "preflight.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "checked_not_executed",
                "execution_performed": False,
                "ok": True,
            }
        )
    )
    value = summarize_regressions(tmp_path, [report, tmp_path / "missing.json"])
    assert not value["ok"] and len(value["errors"]) == 2
    assert not value["task_success_evaluated"]


def test_duplicate_or_outside_reports_are_rejected(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_regressions(tmp_path, [report, report])
    with pytest.raises(ValueError, match="leaves"):
        summarize_regressions(tmp_path, [tmp_path.parent / "outside.json"])
    with pytest.raises(ValueError, match="At least"):
        summarize_regressions(tmp_path, [])


@pytest.mark.parametrize("audit_ok", [True, False])
def test_report_reaudits_artifacts_and_decodes_video_not_cached_flags(
    tmp_path, monkeypatch, audit_ok
):
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "metadata.json").write_text(
        json.dumps(
            {"num_video_frames": 401, "video_fps": 20, "view_files": {"primary": "rollout.mp4"}}
        )
    )
    (directory / "job.json").write_text(json.dumps({"state_provider": "none"}))
    (directory / "dataset.json").write_text(json.dumps({"imagined_beyond_sample": True}))
    job = {
        "id": "long",
        "output": str(directory),
        "completed_chunks": 400,
        "vla_calls": 400,
        "world_model_calls": 400,
    }
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "execution_performed": True,
                "status": "completed",
                "ok": True,
                "execution": {"jobs": [job]},
                "preflight": {"selection": [{"id": "long"}]},
            }
        )
    )
    seen = []

    def audit(summary, *, expected_jobs):
        seen.append(expected_jobs)
        return {"ok": audit_ok}

    monkeypatch.setattr("evaluation.reporting.regression_report.audit_attempt", audit)
    monkeypatch.setattr(
        "evaluation.reporting.regression_report.diagnose_artifact", lambda p: {"decoded": str(p)}
    )
    value = summarize_regressions(tmp_path, [report])
    assert seen and value["ok"] == audit_ok
    row = value["entries"][0]["jobs"][0]
    assert row["evaluation_mode"] is None  # Do not invent a mode for older minimal reports.
    assert row["duration_seconds"] == 20.05 and row["imagined_beyond_sample"]
    assert row["visual_diagnostics"] == {"decoded": str(directory)}


def test_invalid_artifact_fps_returns_failure_without_dividing_by_zero(tmp_path, monkeypatch):
    directory = tmp_path / "job"
    directory.mkdir()
    (directory / "metadata.json").write_text(
        json.dumps({"num_video_frames": 1, "video_fps": 0, "view_files": {}})
    )
    for name in ("job", "dataset"):
        (directory / f"{name}.json").write_text("{}")
    path = tmp_path / "report.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "execution_performed": True,
                "execution": {"jobs": [{"output": str(directory)}]},
                "preflight": {"selection": [{"id": "long"}]},
            }
        )
    )
    monkeypatch.setattr(
        "evaluation.reporting.regression_report.audit_attempt", lambda *a, **k: {"ok": False}
    )
    value = summarize_regressions(tmp_path, [path])
    assert not value["ok"] and "Invalid artifact FPS" in value["errors"][0]["error"]
