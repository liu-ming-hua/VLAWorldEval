"""Full-plan metric admission cannot hide failures or select lucky duplicates."""
import json
import sys
from types import SimpleNamespace

import pytest

from evaluation.reporting.checkpoint_metrics import select_replays, scoring_failure_journal, score_checkpoint_study, metric_source_fingerprints
from evaluation.reporting.task_scores import _digest


def evidence(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for name in ("input.npz", "reference.npz", "config.json"):
        (prepared / name).write_bytes(name.encode())
    report = {
        "status": "completed",
        "cleanup": {"returncode": 0, "signals": [], "errors": []},
        "input_sha256": _digest(prepared / "input.npz"),
        "config_sha256": _digest(prepared / "config.json"),
        "reference_evaluation": {"reference_sha256": _digest(prepared / "reference.npz")},
    }
    path = tmp_path / "runs" / "first" / "case0" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    plan = {"schema_version": 1, "cases": [{"id": "case0", "prepared": "prepared"}]}
    return plan, report, path


def test_selects_clean_exact_case_and_preserves_failed_attempt(tmp_path):
    plan, report, path = evidence(tmp_path)
    failed = tmp_path / "runs" / "old" / "case0" / "summary.json"
    failed.parent.mkdir(parents=True)
    failed.write_text(json.dumps({**report, "status": "failed", "error": "SIGTERM"}))
    selected, excluded = select_replays(tmp_path, tmp_path / "runs", plan)
    assert selected["case0"][0] == path
    assert len(excluded) == 1 and excluded[0]["error"] == "SIGTERM"


def test_duplicate_clean_replays_are_not_cherry_picked(tmp_path):
    plan, report, _ = evidence(tmp_path)
    path = tmp_path / "runs" / "second" / "case0" / "summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Multiple clean replays"):
        select_replays(tmp_path, tmp_path / "runs", plan)


def test_migrated_replay_root_is_admitted_without_widening_to_unrelated_preflights(tmp_path):
    plan, _, path = evidence(tmp_path)
    migrated = tmp_path/"migrated"
    migrated.mkdir()
    moved = migrated/"case0"
    path.parent.rename(moved)
    selected, _ = select_replays(tmp_path, tmp_path/"runs", plan, additional_directories=[migrated])
    assert selected["case0"][0] == moved/"summary.json"


def test_duplicate_clean_cases_across_roots_are_rejected(tmp_path):
    plan, report, _ = evidence(tmp_path)
    migrated = tmp_path/"migrated/case0"
    migrated.mkdir(parents=True)
    (migrated/"summary.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Multiple clean replays"):
        select_replays(tmp_path, tmp_path/"runs", plan, additional_directories=[migrated.parent])


@pytest.mark.parametrize("additional", ["runs", "runs/first", "."])
def test_overlapping_replay_roots_cannot_double_count(tmp_path, additional):
    plan, _, _ = evidence(tmp_path)
    with pytest.raises(ValueError, match="nonoverlapping"):
        select_replays(tmp_path, tmp_path/"runs", plan, additional_directories=[tmp_path/additional])


@pytest.mark.parametrize("name", ["input.npz", "config.json", "reference.npz"])
def test_mutated_preparation_rejected(tmp_path, name):
    plan, _, _ = evidence(tmp_path)
    (tmp_path / "prepared" / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="admitted input/reference"):
        select_replays(tmp_path, tmp_path / "runs", plan)


@pytest.mark.parametrize("case_id", ["../escape", "bad/name", ""])
def test_unsafe_case_ids_rejected(tmp_path, case_id):
    plan, _, _ = evidence(tmp_path)
    plan["cases"][0]["id"] = case_id
    with pytest.raises(ValueError, match="safe case IDs"):
        select_replays(tmp_path, tmp_path / "runs", plan)


def test_duplicate_plan_ids_rejected(tmp_path):
    plan, _, _ = evidence(tmp_path)
    plan["cases"] *= 2
    with pytest.raises(ValueError, match="Duplicate evaluation case"):
        select_replays(tmp_path, tmp_path / "runs", plan)


def test_unclean_worker_is_excluded_not_scored(tmp_path):
    plan, report, path = evidence(tmp_path)
    report["cleanup"]["signals"] = ["SIGTERM"]
    path.write_text(json.dumps(report))
    selected, excluded = select_replays(tmp_path, tmp_path / "runs", plan)
    assert not selected and len(excluded) == 1


@pytest.mark.parametrize("exception,status", [(ValueError("source changed"), "failed"), (KeyboardInterrupt(), "interrupted")])
def test_errors_are_persisted_not_left_running(tmp_path, exception, status):
    path = tmp_path/"progress.json"
    journal = {"status": "running", "jobs": {"done": {"status": "completed"}}}
    with pytest.raises(type(exception)), scoring_failure_journal(path, journal):
        raise exception
    saved = json.loads(path.read_text())
    assert saved["status"] == status
    assert saved["jobs"] == journal["jobs"]
    assert saved["error"]["type"] == type(exception).__name__


@pytest.mark.parametrize("plan", [[], {}, {"schema_version": 1, "cases": ["not an object"]}])
def test_malformed_plan_has_clear_admission_error(tmp_path, plan):
    with pytest.raises(ValueError, match="Nonempty frozen plan"):
        select_replays(tmp_path, tmp_path, plan)


def test_gpu_metrics_assignment_is_explicit_and_journalled(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from evaluation.reporting import checkpoint_metrics as module
    from evaluation.execution import admission
    plan, _, _ = evidence(tmp_path)
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(module, 'get_environment', lambda _: SimpleNamespace(exists=True, python=sys.executable))
    leases = []
    monkeypatch.setattr(admission, 'native_gpu_lease', lambda root, gpus: leases.append(gpus) or nullcontext())
    monkeypatch.setattr(admission, 'resource_admission', lambda *a: None)
    def evaluate(command, **kwargs):
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == '2'
        assert command[command.index('--device')+1] == 'cuda'
        assert command[command.index('--frame-batch-size')+1] == '8'
        destination = module.Path(command[command.index('--output')+1])
        summary = module.Path(command[command.index('--summary')+1])
        reference = module.Path(command[command.index('--reference')+1])
        destination.write_text(json.dumps({'status': 'completed', 'summary_sha256': _digest(summary),
            'reference_sha256': _digest(reference), 'implementation_sha256': metric_source_fingerprints('lpips')}))
    monkeypatch.setattr(module.subprocess, 'run', evaluate)
    result = module.score_checkpoint_study(tmp_path, plan_path, tmp_path/'runs', tmp_path/'gpu_metrics',
        workers=1, metric_gpus=[2], frame_batch_size=8)
    assert result['status'] == 'completed' and result['metric_gpus'] == [2]
    assert result['frame_batch_size'] == 8 and leases == [[2]]


@pytest.mark.parametrize('gpus,workers,batch', [([4], 1, 1), ([0, 0], 2, 1), ([0], 2, 1), ([0], 1, 0)])
def test_invalid_metric_gpu_or_batch_configuration_rejected(tmp_path, gpus, workers, batch):
    with pytest.raises(ValueError, match='metric GPUs'):
        score_checkpoint_study(tmp_path, tmp_path/'plan.json', tmp_path/'runs', tmp_path/'output',
            workers=workers, metric_gpus=gpus, frame_batch_size=batch)


def test_copied_resource_evidence_does_not_require_old_machine_paths(tmp_path):
    test_only_bound_memory_override_can_differ_from_full_plan(tmp_path, None)
    study_dir = tmp_path/'queue'
    summary = study_dir/'runs/case0/summary.json'
    saved = summary.parent/'config.json'
    saved.write_bytes((study_dir/'effective.json').read_bytes())
    study_path, progress_path = study_dir/'study.json', study_dir/'progress.json'
    study, progress = json.loads(study_path.read_text()), json.loads(progress_path.read_text())
    study['jobs'][0]['config'] = '/old-unavailable-server/prepared/config.json'
    study_path.write_text(json.dumps(study))
    progress['study_sha256'] = _digest(study_path)
    progress['jobs']['case0']['effective_config'] = '/old-unavailable-server/queue/effective.json'
    progress_path.write_text(json.dumps(progress))
    plan = {'schema_version': 1, 'cases': [{'id': 'case0', 'prepared': 'prepared'}]}
    selected, _ = select_replays(tmp_path, study_dir, plan)
    assert selected['case0'][0] == summary
    # Byte changes still fail even though path relocation is permitted.
    saved.write_text(saved.read_text()+' ')
    with pytest.raises(ValueError, match='admitted input/reference'):
        select_replays(tmp_path, study_dir, plan)


@pytest.mark.parametrize("timeout", [3600, 86400])
def test_score_versions_are_independent_and_preserve_legacy_sidecars(tmp_path, monkeypatch, timeout):
    from evaluation.reporting import checkpoint_metrics as module
    plan, _, _ = evidence(tmp_path)
    plan_path = tmp_path/"plan.json"
    plan_path.write_text(json.dumps(plan))
    calls = []
    legacy = tmp_path/"runs/first/case0/perceptual.json"
    legacy.write_text('{"legacy_unversioned":true}')
    legacy_hash = _digest(legacy)
    monkeypatch.setattr(module, "get_environment", lambda _: SimpleNamespace(exists=True, python=sys.executable))

    def evaluate(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["timeout"] == timeout
        summary = module.Path(command[command.index("--summary")+1])
        reference = module.Path(command[command.index("--reference")+1])
        destination = module.Path(command[command.index("--output")+1])
        destination.write_text(json.dumps({"status": "completed", "summary_sha256": _digest(summary),
            "reference_sha256": _digest(reference), "implementation_sha256": metric_source_fingerprints("lpips")}))

    monkeypatch.setattr(module.subprocess, "run", evaluate)
    for version in (1, 2):
        r = score_checkpoint_study(tmp_path, plan_path, tmp_path/"runs", tmp_path/f"metrics{version}", provider="none", timeout_seconds=timeout)
        assert r["status"] == "completed" and r["expected_cases"] == 1
        assert r["motion_evaluation"] == "not_requested_no_calibrated_provider"
        assert r["metric_timeout_seconds"] == timeout and r["decoder_timeout_seconds"] == 3600
        assert any(key.endswith("evaluation/datasets/video.py") for key in r["evaluator_source_sha256"])
        record = r["jobs"]["case0"]["metrics"]["lpips"]
        assert module.Path(record["path"]) == tmp_path/f"metrics{version}/cases/case0/perceptual.json"
        assert record["sha256"] == _digest(module.Path(record["path"]))
    assert len(calls) == 2 and _digest(legacy) == legacy_hash
    sidecar = tmp_path/"metrics1/cases/case0/perceptual.json"
    value = json.loads(sidecar.read_text())
    value.pop("implementation_sha256")
    sidecar.write_text(json.dumps(value))
    r = score_checkpoint_study(tmp_path, plan_path, tmp_path/"runs", tmp_path/"metrics3", provider="none", timeout_seconds=timeout)
    assert r["status"] == "completed" and len(calls) == 3
    assert _digest(legacy) == legacy_hash


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), 604801])
def test_invalid_metric_deadline_rejected_before_output(tmp_path, timeout):
    plan, _, _ = evidence(tmp_path)
    source = tmp_path/"plan.json"
    source.write_text(json.dumps(plan))
    output = tmp_path/"metrics"
    with pytest.raises(ValueError, match="Metric timeout"):
        score_checkpoint_study(tmp_path, source, tmp_path/"runs", output, timeout_seconds=timeout)
    assert not output.exists()


@pytest.mark.parametrize("change", [None, "scientific", "study"])
def test_only_bound_memory_override_can_differ_from_full_plan(tmp_path, change):
    plan, report, old = evidence(tmp_path)
    config = tmp_path/"prepared/config.json"
    original = {"memory_gib": 22, "gpus": [0], "seed": 0}
    config.write_text(json.dumps(original))
    source_hash = _digest(config)
    study_dir = tmp_path/"queue"
    study_dir.mkdir()
    effective = study_dir/"effective.json"
    effective.write_text(json.dumps({**original, "memory_gib": 21.5, **({"seed": 1} if change == "scientific" else {})}))
    study_path = study_dir/"study.json"
    study_path.write_text(json.dumps({"jobs": [{"id": "case0", "config": str(config),
        "config_sha256": source_hash, "input_sha256": report["input_sha256"],
        "reference_sha256": report["reference_evaluation"]["reference_sha256"]}]}))
    progress = {"study_sha256": _digest(study_path), "jobs": {"case0": {
        "effective_config": str(effective), "effective_config_sha256": _digest(effective),
        "source_config_sha256": source_hash}}}
    (study_dir/"progress.json").write_text(json.dumps(progress))
    if change == "study":
        study_path.write_text(study_path.read_text()+" ")
    old.unlink()
    path = study_dir/"runs/case0/summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**report, "config_sha256": _digest(effective)}))
    if change:
        with pytest.raises(ValueError, match="admitted input/reference"):
            select_replays(tmp_path, study_dir, plan)
    else:
        selected, _ = select_replays(tmp_path, study_dir, plan)
        assert selected["case0"][0] == path
