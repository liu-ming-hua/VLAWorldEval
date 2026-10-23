"""Synthetic evidence fixtures exercise versioning, never model quality."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from evaluation.reporting import checkpoint_metrics as scorer
from evaluation.reporting.native_reference import summarize_native_references
from evaluation.reporting.task_scores import _digest


@pytest.fixture
def version(tmp_path, monkeypatch):
    prepared, run = tmp_path/"prepared", tmp_path/"runs/case0"
    prepared.mkdir()
    run.mkdir(parents=True)
    config = {"adapter": "FixtureAdapter", "video": {"fps": 10}}
    for path in (prepared/"config.json", run/"config.json"):
        path.write_text(json.dumps(config))
    for name in ("input.npz", "reference.npz"):
        (prepared/name).write_bytes(b"synthetic bound packet")
    (run/"camera.mp4").write_bytes(b"synthetic codec placeholder; evaluator mocked")
    report = {"status": "completed", "cleanup": {"returncode": 0, "signals": [], "errors": []},
              "input_sha256": _digest(prepared/"input.npz"), "config_sha256": _digest(prepared/"config.json"),
              "files": {"camera.mp4": _digest(run/"camera.mp4")},
              "reference_evaluation": {"reference_sha256": _digest(prepared/"reference.npz"),
                  "reference_source": {"dataset_domain": "fixture", "split": "test", "episode": "0", "condition_kind": "recorded_command"},
                  "metrics": {"camera": {"frames_scored": 8, "mse_01": .1, "mean_frame_psnr_db": 20,
                      "psnr_db_from_mean_mse": 10, "ssim_rgb_gaussian11_sigma1p5_population": .8,
                      "relative_mse_improvement_over_frozen": .2}}}}
    summary = run/"summary.json"
    summary.write_text(json.dumps(report))
    legacy = {"summary_sha256": _digest(summary), "reference_sha256": _digest(prepared/"reference.npz"),
              "views": {"camera": {"lpips_vgg": .99}}}
    (run/"perceptual.json").write_text(json.dumps(legacy))
    (run/"motion").mkdir()
    (run/"motion/motion.json").write_text('{"invalid_legacy_motion":true}')
    plan = tmp_path/"plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "scope": "fixture only", "unavailable": {},
        "cases": [{"id": "case0", "prepared": "prepared"}]}))
    monkeypatch.setattr(scorer, "get_environment", lambda _: SimpleNamespace(exists=True, python=sys.executable))
    calls = []

    def evaluate(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        Path(command[command.index("--output")+1]).write_text(json.dumps({
            "status": "completed", "summary_sha256": _digest(summary),
            "reference_sha256": _digest(prepared/"reference.npz"),
            "implementation_sha256": scorer.metric_source_fingerprints("lpips"),
            "views": {"camera": {"lpips_vgg": .17}}}))

    monkeypatch.setattr(scorer.subprocess, "run", evaluate)
    journal = scorer.score_checkpoint_study(tmp_path, plan, tmp_path/"runs", tmp_path/"metrics")
    assert journal["status"] == "completed"
    return tmp_path, plan, journal, calls


def test_report_uses_explicit_version_and_never_adjacent_legacy_files(version):
    root, plan, journal, calls = version
    legacy = root/"runs/case0/perceptual.json"
    old_hash = _digest(legacy)
    report = summarize_native_references(root, "runs", "report", plan_path=plan, metrics_directory="metrics")
    assert report["all_planned_replays_completed"] and report["perceptual_scored_cases"] == ["case0"]
    assert report["groups"][0]["lpips_vgg_decoded"] == .17
    assert not report["motion_rows"]
    assert report["scoring_journal"]["sha256"] == _digest(root/"metrics/progress.json")
    assert _digest(legacy) == old_hash and len(calls) == 1
    assert journal["vla_calls"] == journal["world_model_calls"] == 0


@pytest.mark.parametrize("fault", ["artifact", "unfinished", "plan", "summary", "implementation", "inputs", "outside_version", "missing_kind"])
def test_version_journal_cannot_hide_changed_or_cross_case_evidence(version, fault):
    root, plan, journal, _ = version
    path = root/"metrics/progress.json"
    metric = root/"metrics/cases/case0/perceptual.json"
    if fault == "artifact":
        metric.write_text('{}')
    elif fault == "unfinished":
        journal["status"] = "waiting_for_replays"
    elif fault == "plan":
        journal["source_plan_sha256"] = "wrong"
    elif fault == "summary":
        journal["jobs"]["case0"]["summary_sha256"] = "wrong"
    elif fault == "implementation":
        journal["metric_implementation_sha256"]["lpips"] = {"other.py": "a"*64}
    elif fault == "inputs":
        journal["case_input_sha256"]["case0"]["reference_sha256"] = "wrong"
    elif fault == "outside_version":
        legacy = root/"runs/case0/perceptual.json"
        journal["jobs"]["case0"]["metrics"]["lpips"] = {"path": str(legacy), "sha256": _digest(legacy)}
    else:
        journal["jobs"]["case0"]["metrics"] = {}
    path.write_text(json.dumps(journal))
    with pytest.raises(ValueError):
        summarize_native_references(root, "runs", "report", plan_path=plan, metrics_directory="metrics")
    assert not (root/"report").exists()


def test_explicit_historical_version_does_not_require_todays_metric_hash(version, monkeypatch):
    root, plan, _, _ = version
    monkeypatch.setattr(scorer, "metric_source_fingerprints", lambda _: {"different.py": "b"*64})
    report = summarize_native_references(root, "runs", "report", plan_path=plan, metrics_directory="metrics")
    assert report["groups"][0]["lpips_vgg_decoded"] == .17


def test_metric_store_requires_an_explicit_plan(version):
    root, _, _, _ = version
    with pytest.raises(ValueError, match="exact --plan"):
        summarize_native_references(root, "runs", "report", metrics_directory="metrics")


@pytest.mark.parametrize("field", ["status", "summary_sha256", "reference_sha256", "implementation_sha256"])
def test_successful_subprocess_with_bad_evidence_is_still_a_failed_metric(version, monkeypatch, field):
    root, plan, _, _ = version
    good = json.loads((root/"metrics/cases/case0/perceptual.json").read_text())

    def invalid(command, **kwargs):
        value = {**good, field: "wrong"}
        Path(command[command.index("--output")+1]).write_text(json.dumps(value))

    monkeypatch.setattr(scorer.subprocess, "run", invalid)
    result = scorer.score_checkpoint_study(root, plan, root/"runs", root/"bad_metrics")
    assert result["status"] == "failed" and result["jobs"]["case0"]["status"] == "failed"
    assert (root/"metrics/cases/case0/perceptual.json").read_text() == json.dumps(good)
