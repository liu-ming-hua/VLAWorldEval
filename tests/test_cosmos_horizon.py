"""Horizon prefixes are source-time aligned; a resumed queue never overwrites."""
import json
from pathlib import Path

import pytest
import numpy as np

from evaluation.planning.cosmos_horizon import native_chunks
from evaluation.reporting.task_scores import _digest


@pytest.mark.parametrize("ticks,expected", [(32, (2, 1)), (64, (4, 2)), (128, (8, 4))])
def test_native_models_share_source_seconds_not_chunk_counts(ticks, expected):
    assert (native_chunks(ticks, 16), native_chunks(ticks, 32)) == expected
    assert expected[0] * 16 / 15 == expected[1] * 32 / 15


@pytest.mark.parametrize("ticks,stride", [(31, 16), (0, 16), (-1, 16), (True, 16), (32, 0), (32, False)])
def test_incomplete_or_invalid_windows_never_pad(ticks, stride):
    with pytest.raises(ValueError, match="complete native chunks"):
        native_chunks(ticks, stride)


@pytest.fixture
def queue(tmp_path, monkeypatch):
    import evaluation.execution.native_study as module
    root = tmp_path
    directory = root / "results/study"
    directory.mkdir(parents=True)
    config, packet = directory / "config.json", directory / "input.npz"
    config.write_text('{"gpus":[0]}')
    packet.write_bytes(b"fixture")
    env = directory / "environments.json"
    env.write_text('{}')
    job = {"id": "test", "chunks": 2, "config": str(config), "config_sha256": _digest(config),
        "input": str(packet), "input_sha256": _digest(packet)}
    study = {"plan": {"allowed_gpus": [0, 1, 2, 3]}, "environments": str(env),
        "environments_sha256": _digest(env), "jobs": [job]}
    (directory / "study.json").write_text(json.dumps(study))
    calls = []
    def run(config, packet, output, *, admit_only=False):
        calls.append(admit_only)
        if not admit_only:
            output.mkdir(parents=True, exist_ok=False)
            (output / "summary.json").write_text('{"status":"completed"}')
        return {"status": "completed", "world_model_calls": 2}
    monkeypatch.setattr(module, "run_native_replay", run)
    return module, root, directory, calls


def test_queue_resumes_cleanly_and_restores_environment(queue, monkeypatch):
    import os
    module, root, directory, calls = queue
    monkeypatch.setenv("VLAWORLDEVAL_ENVIRONMENTS", "previous-override")
    assert module.run_native_study(root, directory)["world_model_calls"] == 2
    assert calls == [True, False]
    assert os.environ["VLAWORLDEVAL_ENVIRONMENTS"] == "previous-override"
    assert module.run_native_study(root, directory)["status"] == "completed"
    assert calls == [True, False]
    (directory / "input.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frozen job"):
        module.run_native_study(root, directory)


@pytest.mark.parametrize("fault", ["running", "missing", "report", "calls", "study"])
def test_completed_audit_never_promotes_partial_or_mutated_queue(queue, fault):
    from evaluation.reporting.cosmos_horizon import _completed_progress
    module, root, directory, calls = queue
    module.run_native_study(root, directory)
    study = json.loads((directory / "study.json").read_text())
    _completed_progress(directory, study)
    path = directory / "progress.json"
    progress = json.loads(path.read_text())
    if fault == "running":
        progress["status"] = "running"
    elif fault == "missing":
        progress["jobs"].clear()
    elif fault == "report":
        (directory / "runs/test/summary.json").write_text('{}')
    elif fault == "calls":
        progress["jobs"]["test"]["world_model_calls"] = 999
    else:
        progress["study_sha256"] = "changed"
    path.write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="Audit requires|progress/report mismatch"):
        _completed_progress(directory, study)


def test_queue_resource_wait_does_not_create_attempt_or_mark_model_failed(queue, monkeypatch):
    from evaluation.execution.admission import ResourceAdmissionError
    module, root, directory, calls = queue
    def busy(*args, **kwargs):
        raise ResourceAdmissionError("GPU busy")
    monkeypatch.setattr(module, "run_native_replay", busy)
    result = module.run_native_study(root, directory, watch=False)
    assert result["status"] == "waiting_for_gpu"
    assert not (directory / "runs").exists()
    assert json.loads((directory / "progress.json").read_text())["jobs"]["test"]["status"] == "waiting_for_gpu"


def test_exclusive_queue_waits_without_loading_or_changing_scientific_inputs(queue):
    from evaluation.execution.admission import native_gpu_lease
    module, root, directory, calls = queue
    frozen = (directory / "study.json").read_bytes()
    with native_gpu_lease(root, [0]):
        result = module.run_native_study(root, directory, watch=False, exclusive_gpu=True)
        assert result["status"] == "waiting_for_gpu_lease"
        assert not calls and not (directory / "runs").exists()
        assert not (directory / "progress.json").exists()
        assert json.loads((directory/"resource_wait.json").read_text())["model_loaded"] is False
    assert module.run_native_study(root, directory, exclusive_gpu=True)["status"] == "completed"
    assert (directory / "study.json").read_bytes() == frozen
    assert calls == [True, False]


def test_exclusive_queue_does_not_swallow_attempt_resource_error(queue, monkeypatch):
    from evaluation.execution.admission import ResourceAdmissionError, native_gpu_lease
    module, root, directory, calls = queue
    def partial(config, packet, output, *, admit_only=False):
        if not admit_only:
            output.mkdir(parents=True)
            raise ResourceAdmissionError("actual worker failure")
    monkeypatch.setattr(module, "run_native_replay", partial)
    with pytest.raises(ResourceAdmissionError, match="actual worker failure"):
        module.run_native_study(root, directory, exclusive_gpu=True)
    assert json.loads((directory/"progress.json").read_text())["status"] == "failed"
    with native_gpu_lease(root, [0]):
        pass


def test_allocator_override_preserves_frozen_scientific_config_and_is_auditable(queue):
    module, root, directory, calls = queue
    path = directory / "config.json"
    path.write_text('{"gpus":[0],"environment":"fixture","memory_gib":32,"seed":1}')
    study = json.loads((directory / "study.json").read_text())
    study["jobs"][0]["config_sha256"] = _digest(path)
    (directory / "study.json").write_text(json.dumps(study))
    assert module.run_native_study(root, directory, resource_overrides={"fixture":20})["status"] == "completed"
    actual, sha = module.resolved_native_config(root, directory, study["jobs"][0])
    assert actual != path and sha == _digest(actual)
    assert json.loads(path.read_text())["memory_gib"] == 32
    assert json.loads(actual.read_text()) == {"gpus":[0],"environment":"fixture","memory_gib":20,"seed":1}
    changed = json.loads(actual.read_text())
    changed["seed"] = 999
    actual.write_text(json.dumps(changed))
    progress = json.loads((directory / "progress.json").read_text())
    progress["jobs"]["test"]["effective_config_sha256"] = _digest(actual)
    (directory / "progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="Only memory_gib"):
        module.resolved_native_config(root, directory, study["jobs"][0])


def test_explicit_device_override_preserves_original_plan_and_is_auditable(queue):
    module, root, directory, calls = queue
    path = directory / "config.json"
    path.write_text('{"gpus":[0],"environment":"fixture","memory_gib":12,"seed":1}')
    study = json.loads((directory / "study.json").read_text())
    study["jobs"][0]["config_sha256"] = _digest(path)
    (directory / "study.json").write_text(json.dumps(study))
    frozen = (directory / "study.json").read_bytes()
    assert module.run_native_study(root, directory, gpu_override=7, exclusive_gpu=True)["status"] == "completed"
    actual, _ = module.resolved_native_config(root, directory, study["jobs"][0])
    assert json.loads(path.read_text())["gpus"] == [0]
    assert json.loads(actual.read_text())["gpus"] == [7]
    assert (directory / "study.json").read_bytes() == frozen
    value = json.loads(actual.read_text())
    value["gpus"] = [6]
    actual.write_text(json.dumps(value))
    progress = json.loads((directory/"progress.json").read_text())
    progress["jobs"]["test"]["effective_config_sha256"] = _digest(actual)
    (directory/"progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="Invalid explicit single-GPU"):
        module.resolved_native_config(root, directory, study["jobs"][0])


def test_waiting_allocator_change_retains_old_derivation_and_has_no_failed_attempt(queue, monkeypatch):
    from evaluation.execution.admission import ResourceAdmissionError
    module, root, directory, calls = queue
    path = directory/"config.json"
    path.write_text('{"gpus":[0],"environment":"fixture","memory_gib":32,"seed":1}')
    study = json.loads((directory/"study.json").read_text())
    study["jobs"][0]["config_sha256"] = _digest(path)
    (directory/"study.json").write_text(json.dumps(study))
    original = module.run_native_replay
    def busy(*args, **kwargs):
        raise ResourceAdmissionError("shared GPU capacity")
    monkeypatch.setattr(module, "run_native_replay", busy)
    assert module.run_native_study(root, directory, watch=False,
        resource_overrides={"fixture":12})["status"] == "waiting_for_gpu"
    before, _ = module.resolved_native_config(root, directory, study["jobs"][0])
    old_bytes = before.read_bytes()
    assert not (directory/"runs").exists()
    monkeypatch.setattr(module, "run_native_replay", original)
    assert module.run_native_study(root, directory,
        resource_overrides={"fixture":8})["status"] == "completed"
    after, _ = module.resolved_native_config(root, directory, study["jobs"][0])
    assert after != before and before.read_bytes() == old_bytes
    assert json.loads(after.read_text())["memory_gib"] == 8
    assert json.loads(path.read_text())["memory_gib"] == 32


@pytest.mark.parametrize("gpu", [True, -1, 1.5, "7"])
def test_invalid_gpu_override_never_launches(queue, gpu):
    module, root, directory, calls = queue
    with pytest.raises(ValueError, match="GPU override"):
        module.run_native_study(root, directory, gpu_override=gpu)
    assert not calls


def test_device_override_cannot_collapse_distributed_topology(queue):
    module, root, directory, calls = queue
    path = directory/"config.json"
    path.write_text('{"gpus":[0,1]}')
    study = json.loads((directory/"study.json").read_text())
    study["jobs"][0]["config_sha256"] = _digest(path)
    (directory/"study.json").write_text(json.dumps(study))
    with pytest.raises(ValueError, match="distributed topology"):
        module.run_native_study(root, directory, gpu_override=7)
    assert not calls


def test_rank_map_preserves_world_size_and_frozen_science(queue):
    module, root, directory, calls = queue
    path = directory/"config.json"
    path.write_text('{"gpus":[0,1,2,3],"environment":"fixture","memory_gib":16,"seed":1}')
    study_path = directory/"study.json"
    study = json.loads(study_path.read_text())
    study["jobs"][0]["config_sha256"] = _digest(path)
    study_path.write_text(json.dumps(study))
    frozen = study_path.read_bytes()
    placement = [0, 3, 4, 7]
    assert module.run_native_study(root, directory, exclusive_gpu=True, gpu_map=placement)["status"] == "completed"
    actual, _ = module.resolved_native_config(root, directory, study["jobs"][0])
    assert json.loads(actual.read_text()) == {"gpus": placement,"environment":"fixture","memory_gib":16,"seed":1}
    assert json.loads(path.read_text())["gpus"] == [0, 1, 2, 3]
    assert study_path.read_bytes() == frozen
    derived = json.loads(actual.read_text())
    derived["gpus"] = list(reversed(placement))
    actual.write_text(json.dumps(derived))
    p = directory/"progress.json"
    progress = json.loads(p.read_text())
    progress["jobs"]["test"]["effective_config_sha256"] = _digest(actual)
    p.write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="same-topology GPU rank map"):
        module.resolved_native_config(root, directory, study["jobs"][0])


@pytest.mark.parametrize("placement", [[], [0, 0], [True], [-1], [1.5], "0,3,4,7"])
def test_invalid_rank_map_never_launches(queue, placement):
    module, root, directory, calls = queue
    with pytest.raises(ValueError, match="GPU placement"):
        module.run_native_study(root, directory, gpu_map=placement)
    assert not calls


def test_rank_map_cannot_change_world_size_or_mix_single_device_option(queue):
    module, root, directory, calls = queue
    with pytest.raises(ValueError, match="distributed topology"):
        module.run_native_study(root, directory, gpu_map=[0, 3])
    with pytest.raises(ValueError, match="GPU placement"):
        module.run_native_study(root, directory, gpu_override=7, gpu_map=[7])
    assert not calls


@pytest.mark.parametrize("watch", [False, True])
def test_resource_loss_after_preflight_waits_without_creating_failed_attempt(queue, monkeypatch, watch):
    from evaluation.execution.admission import ResourceAdmissionError
    module, root, directory, calls = queue
    original = module.run_native_replay
    actual_calls = []

    def race(*args, **kwargs):
        if not kwargs.get("admit_only"):
            actual_calls.append(True)
            if len(actual_calls) == 1:
                raise ResourceAdmissionError("GPU changed after preflight")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_native_replay", race)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = module.run_native_study(root, directory, watch=watch)
    assert result["status"] == ("completed" if watch else "waiting_for_gpu")
    progress = json.loads((directory/"progress.json").read_text())
    assert progress["jobs"]["test"]["status"] == result["status"]
    if not watch:
        assert not (directory/"runs").exists()
        assert module.run_native_study(root, directory)["status"] == "completed"


def test_resource_error_after_attempt_creation_is_not_silently_retried(queue, monkeypatch):
    from evaluation.execution.admission import ResourceAdmissionError
    module, root, directory, calls = queue

    def partial(config, packet, output, *, admit_only=False):
        if not admit_only:
            output.mkdir(parents=True)
            (output/"partial.txt").write_text("preserve")
            raise ResourceAdmissionError("Worker attempt already exists")

    monkeypatch.setattr(module, "run_native_replay", partial)
    with pytest.raises(ResourceAdmissionError, match="already exists"):
        module.run_native_study(root, directory)
    assert json.loads((directory/"progress.json").read_text())["status"] == "failed"
    assert (directory/"runs/test/partial.txt").read_text() == "preserve"


@pytest.mark.parametrize("budget", [True, -1, float('nan')])
def test_invalid_allocator_override_rejected_before_running(queue, budget):
    module, root, directory, calls = queue
    with pytest.raises(ValueError, match="Resource overrides"):
        module.run_native_study(root, directory, resource_overrides={None:budget})
    assert not calls


def test_queue_failure_is_preserved_and_not_silently_retried(queue, monkeypatch):
    module, root, directory, calls = queue
    def failure(*args, **kwargs):
        if kwargs.get("admit_only"):
            return {}
        raise RuntimeError("fixture model failure")
    monkeypatch.setattr(module, "run_native_replay", failure)
    with pytest.raises(RuntimeError, match="model failure"):
        module.run_native_study(root, directory)
    assert json.loads((directory / "progress.json").read_text())["status"] == "failed"
    with pytest.raises(ValueError, match="Interrupted/failed"):
        module.run_native_study(root, directory)


@pytest.mark.parametrize("fault", ["id_escape", "duplicate", "gpu_bool", "config_escape"])
def test_queue_rejects_unsafe_frozen_job_declarations_before_execution(queue, fault):
    module, root, directory, calls = queue
    path = directory / "study.json"
    study = json.loads(path.read_text())
    if fault == "id_escape":
        study["jobs"][0]["id"] = "../../outside"
    elif fault == "duplicate":
        study["jobs"].append(dict(study["jobs"][0]))
    elif fault == "gpu_bool":
        study["plan"]["allowed_gpus"] = [True]
    else:
        study["jobs"][0]["config"] = str(root.parent / "outside.json")
    path.write_text(json.dumps(study))
    with pytest.raises(ValueError):
        module.run_native_study(root, directory)
    assert not calls and not (directory / "runs").exists()


def test_plan_covers_fixed_cases_seeds_and_valid_source_windows():
    plan = json.loads(Path('configs/studies/cosmos_horizon_v1.json').read_text())
    assert plan["episodes"] == [0, 351, 365, 795, 903]
    assert plan["new_seeds"] == [1, 2]
    assert sum(native_chunks(128, stride) for stride in (16, 32)) * 5 * 2 == 120


def test_prefix_metrics_ignore_condition_and_border_and_do_not_claim_success():
    from evaluation.reporting.cosmos_horizon import horizon_scores
    real = np.zeros((129, 16, 16, 3), np.uint8)
    predicted = real.copy()
    predicted[0] = 255
    predicted[:, -12:] = 255
    predicted[33:65, :-12] = 10
    result = horizon_scores(real, predicted, [32, 64, 128])
    assert [v["mean"] for v in result["prefixes"]] == [0, 5, 2.5]
    assert [v["mean"] for v in result["incremental_32_tick_windows"]] == [0, 10, 0, 0]
    assert result["prefixes_are_independent_replications"] is False
    assert result["task_success_evaluated"] is False
    with pytest.raises(ValueError, match="no padding"):
        horizon_scores(real, predicted, [256])


def test_recorded_motion_is_not_a_contact_or_visibility_label():
    from evaluation.reporting.cosmos_horizon import reference_motion
    poses = np.zeros((5, 6))
    poses[:, 0] = np.arange(5) * .01
    result = reference_motion(poses, [0, 0, 0, 1, 1], 1, [2, 3])
    assert result[0]["eef_translation_path_m"] == pytest.approx(.02)
    assert result[0]["gripper_threshold_crossings_source_frames"] == [3]
    assert result[1]["measured_gripper_range"] == 1
    with pytest.raises(ValueError, match="complete finite"):
        reference_motion(poses, [0] * 5, 1, [4])


def _screening_fixture():
    return {"case": "eef_ep000000", "decisions": {"32": "uncertain", "64": "uncertain", "128": "reject"},
        "notes": {str(h): "Fixture observation, not a task score" for h in (32, 64, 128)},
        "explicit_issue_seeds": {"32": [], "64": [], "128": [2]}, "quality_admitted": False, "task_success": None,
        "evidence": [{"source_ticks": h} for h in (32, 64, 128)]}


@pytest.mark.parametrize("fault", ["approval", "missing", "no_seed", "bool_seed", "duplicate", "bad_tick"])
def test_screening_cannot_silently_admit_or_omit_counterexample_evidence(fault):
    from evaluation.reporting.cosmos_horizon import _validate_screening_row
    row = _screening_fixture()
    _validate_screening_row(row, [32, 64, 128])
    if fault == "approval":
        row["quality_admitted"] = True
    elif fault == "missing":
        row["evidence"].pop()
    elif fault == "no_seed":
        row["explicit_issue_seeds"]["128"] = []
    elif fault == "bool_seed":
        row["explicit_issue_seeds"]["128"] = [True]
    elif fault == "duplicate":
        row["explicit_issue_seeds"]["128"] = [2, 2]
    else:
        row["evidence"][0]["source_ticks"] = [32]
    with pytest.raises(ValueError, match="Screening"):
        _validate_screening_row(row, [32, 64, 128])


def test_four_column_renderer_is_source_aligned_and_hash_checked(tmp_path, monkeypatch):
    import mediapy as media
    from evaluation.reporting.cosmos_horizon import render_horizon_study
    from evaluation.reporting.positive_controls import VIEWS
    root = tmp_path
    old = root / "results/old"
    old.mkdir(parents=True)
    directory = root / "results/new"
    directory.mkdir()
    reference_views = {}
    for name in VIEWS:
        path = old / (name + ".mp4")
        media.write_video(path, np.full((129, 16, 16, 3), 20, np.uint8), fps=15)
        reference_views["droid." + name] = {"file": str(path.relative_to(root)), "sha256": _digest(path)}
    (old / "study.json").write_text(json.dumps({"references": [{"episode": 0, "views": reference_views}]}))
    baseline = old / "generated.mp4"
    media.write_video(baseline, np.full((129, 24, 16, 3), 20, np.uint8), fps=15)
    jobs = []
    for seed in (1, 2):
        folder = directory / "runs" / f"seed{seed}"
        folder.mkdir(parents=True)
        media.write_video(folder / "droid.cosmos_composite.mp4", np.full((129, 24, 16, 3), 20 + seed, np.uint8), fps=15)
        jobs.append({"id": f"seed{seed}", "seed": seed, "episode": 0, "model": "eef"})
    study = {"plan": {"reference_study": "results/old", "horizon_source_ticks": [32, 64, 128], "new_seeds": [1, 2]},
        "jobs": jobs, "baselines": [{"model": "eef", "episode": 0, "video": str(baseline), "source_stride": 16}]}
    (directory / "study.json").write_text(json.dumps(study))
    first = render_horizon_study(root, directory)
    row = first["cases"][0]
    assert row["frames"] == 129 and row["source_control_seconds"] == 128 / 15
    assert row["task_success_evaluated"] is False and row["quality_admitted"] is None
    assert "same_input_seed1_seed2_pixel_distance" in row
    assert "hold_initial_baseline" in row
    assert len(row["metrics"]["seed1"]["wrist"]["prefixes"]) == 3
    from evaluation.reporting.positive_controls import _read_tiles
    from evaluation.reporting.cosmos_horizon import _probe_prefix_video
    assert _probe_prefix_video(baseline, (16, 24))["nb_read_frames"] == "129"
    with pytest.raises(ValueError, match="Actual video differs"):
        _probe_prefix_video(baseline, (24, 16))
    full = _read_tiles(baseline, 15, composite=True)
    short = _read_tiles(baseline, 15, composite=True, max_frames=7)
    assert np.array_equal(full[:7], short)
    for limit in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="max_frames"):
            _read_tiles(baseline, 15, max_frames=limit)
    second = render_horizon_study(root, directory)
    assert second["cases"][0]["output_sha256"] == row["output_sha256"]
    import evaluation.reporting.cosmos_horizon as module
    audit = {"jobs": [{}, {}], "new_world_model_calls": 12}
    (directory / "execution_audit.json").write_text(json.dumps(audit))
    monkeypatch.setattr(module, "audit_horizon_study", lambda *args: audit)
    screen = _screening_fixture()
    (directory / "review").mkdir()
    for item in screen["evidence"]:
        path = directory / "review" / (str(item["source_ticks"]) + ".png")
        path.write_bytes(b"fixture evidence")
        item.update(path=str(path), sha256=_digest(path))
    (directory / "quality_review_working.json").write_text(json.dumps({"reviewer":"assistant",
        "human_review_performed":False, "cases":[screen]}))
    summary = module.summarize_horizon_study(root, directory)
    assert summary["quality_admitted_cases"] == 0 and summary["task_success_evaluated"] is False
    assert summary["cases"][0]["first_observed_issue_tick"] == 128
    assert summary["cases"][0]["true_failure_onset_measured"] is False
    assert (directory / "REPORT_CN.md").is_file()
    path.write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="Screening evidence changed"):
        module.summarize_horizon_study(root, directory)
    (directory / "comparisons/eef_ep000000/contact_sheet.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="output changed"):
        render_horizon_study(root, directory)
