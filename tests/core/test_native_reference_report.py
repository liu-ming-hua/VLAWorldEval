"""Native-domain reporting must preserve independent episodes and evidence."""

import json

import pytest

from evaluation.reporting.native_reference import summarize_native_references
from evaluation.reporting.task_scores import _digest


def prepared_case(root, name):
    prepared = root / "prepared" / name
    prepared.mkdir(parents=True)
    for filename in ("input.npz", "reference.npz"):
        (prepared / filename).write_bytes(b"synthetic admission fixture")
    (prepared / "config.json").write_text(
        json.dumps({"adapter": "FixtureAdapter", "video": {"fps": 10}})
    )
    return prepared


def case(root, name, domain="domain_a", episode="0", value=0.1):
    prepared = prepared_case(root, name)
    directory = root / "runs" / name
    directory.mkdir(parents=True)
    config = directory / "config.json"
    config.write_text(json.dumps({"adapter": "FixtureAdapter", "video": {"fps": 10}}))
    video = directory / "prediction.mp4"
    video.write_bytes(b"fixture evidence, no decoding required")
    report = {
        "status": "completed",
        "cleanup": {"returncode": 0, "errors": [], "signals": []},
        "config_sha256": _digest(config),
        "files": {video.name: _digest(video)},
        "input_sha256": _digest(prepared / "input.npz"),
        "reference_evaluation": {
            "reference_sha256": _digest(prepared / "reference.npz"),
            "reference_source": {
                "dataset_domain": domain,
                "split": "validation",
                "episode": episode,
                "condition_kind": "recorded_command",
            },
            "metrics": {
                "camera": {
                    "frames_scored": 8,
                    "mse_01": value,
                    "mean_frame_psnr_db": 20,
                    "psnr_db_from_mean_mse": 19,
                    "ssim_rgb_gaussian11_sigma1p5_population": 0.8,
                    "relative_mse_improvement_over_frozen": 0.2,
                }
            },
        },
    }
    (directory / "summary.json").write_text(json.dumps(report))
    return directory


def test_macro_episode_groups_and_missing_cases_are_not_reliability_claims(tmp_path):
    case(tmp_path, "a", value=0.1)
    case(tmp_path, "a_window2", value=0.3)
    case(tmp_path, "b", episode="1", value=0.5)
    case(tmp_path, "c", domain="domain_b", value=0.7)
    case(tmp_path, "unplanned", value=1000)
    prepared_case(tmp_path, "missing")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {"id": name, "prepared": f"prepared/{name}"}
                    for name in ("a", "a_window2", "b", "c", "missing")
                ],
                "unavailable": {"physical_truth": "not recorded"},
            }
        )
    )
    result = summarize_native_references(tmp_path, "runs", "report", plan_path="plan.json")
    assert len(result["groups"]) == 2
    first = result["groups"][0]
    assert first["episodes"] == 2 and first["cases"] == 3
    assert first["mse_01"] == pytest.approx(0.35)
    assert first["lpips_vgg_decoded"] is None
    assert not first["paper_test_set_reproduced"]
    assert not result["cross_domain_ranking_valid"]
    assert not result["physical_action_tracking_verified"]
    assert result["pending_or_failed_cases"] == ["missing"]
    assert not result["all_planned_replays_completed"]
    assert "unplanned" not in result["completed_cases"]
    assert any("unplanned" in item["path"] for item in result["excluded"])
    assert result["source_plan_sha256"] == _digest(plan)
    assert result["budget_selection"] is None
    assert not result["full_test_set_completed"]
    with pytest.raises(FileExistsError):
        summarize_native_references(tmp_path, "runs", "report")


def test_complete_budget_report_is_not_full_test_population(tmp_path):
    from evaluation.planning.native_subset import freeze_reference_subset

    case(tmp_path, "a")
    case(tmp_path, "b", episode="1")
    population = tmp_path / "population.json"
    population.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "released task validation",
                "cases": [{"id": n, "prepared": f"prepared/{n}"} for n in ("a", "b")],
                "unavailable": {"paper_test": "membership unknown"},
            }
        )
    )
    freeze_reference_subset(tmp_path, population, tmp_path / "budget", limit=1)
    report = summarize_native_references(tmp_path, "runs", "report", plan_path="budget/plan.json")
    assert report["all_planned_replays_completed"]
    assert len(report["completed_cases"]) == 1
    assert report["budget_selection"]["population_cases"] == 2
    assert report["budget_selection"]["selected_fraction_of_source_cases"] == 0.5
    assert "1/2" in report["evaluation_scope"]
    assert not report["full_test_set_completed"]


def test_report_merges_exact_old_and_migrated_roots(tmp_path):
    case(tmp_path, "a")
    moved = case(tmp_path, "b", episode="1")
    destination = tmp_path / "migrated"
    destination.mkdir()
    moved.rename(destination / "b")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "unavailable": {},
                "cases": [{"id": name, "prepared": f"prepared/{name}"} for name in ("a", "b")],
            }
        )
    )
    report = summarize_native_references(
        tmp_path, "runs", "report", plan_path=plan, additional_directories=[destination]
    )
    assert report["completed_cases"] == ["a", "b"]
    assert report["all_planned_replays_completed"]
    assert len(report["replay_directories"]) == 2


@pytest.mark.parametrize("target", ["config", "video", "perceptual"])
def test_changed_evidence_is_rejected_without_publishing_report(tmp_path, target):
    directory = case(tmp_path, "a")
    if target == "config":
        (directory / "config.json").write_text('{"adapter":"Changed"}')
    elif target == "video":
        (directory / "prediction.mp4").write_bytes(b"changed")
    else:
        (directory / "perceptual.json").write_text(
            json.dumps({"summary_sha256": "wrong", "reference_sha256": "fixture"})
        )
    with pytest.raises(ValueError):
        summarize_native_references(tmp_path, "runs", "report")
    assert not (tmp_path / "report").exists()


def test_interrupted_worker_does_not_count_as_completed(tmp_path):
    directory = case(tmp_path, "a")
    summary = directory / "summary.json"
    report = json.loads(summary.read_text())
    report["cleanup"]["signals"] = ["SIGTERM"]
    summary.write_text(json.dumps(report))
    result = summarize_native_references(tmp_path, "runs", "report")
    assert not result["rows"] and not result["groups"]
    assert len(result["excluded"]) == 1


def test_short_and_long_windows_are_not_pooled(tmp_path):
    case(tmp_path, "short")
    directory = case(tmp_path, "long")
    path = directory / "summary.json"
    value = json.loads(path.read_text())
    value["reference_evaluation"]["metrics"]["camera"]["frames_scored"] = 192
    path.write_text(json.dumps(value))
    result = summarize_native_references(tmp_path, "runs", "report")
    assert len(result["groups"]) == 2
    assert {g["future_frames"] for g in result["groups"]} == {8, 192}
    assert all(g["episode_uncertainty"]["mse_01"]["episodes"] == 1 for g in result["groups"])


@pytest.mark.parametrize("filename", ["input.npz", "reference.npz", "config.json"])
def test_planned_report_checks_actual_prepared_hashes_not_only_ids(tmp_path, filename):
    case(tmp_path, "a")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"id": "a", "prepared": "prepared/a"}],
                "unavailable": {},
            }
        )
    )
    (tmp_path / "prepared/a" / filename).write_bytes(b"changed after model evaluation")
    with pytest.raises(ValueError, match="admitted input/reference"):
        summarize_native_references(tmp_path, "runs", "report", plan_path=plan)
    assert not (tmp_path / "report").exists()


def test_planned_report_rejects_duplicate_clean_attempts(tmp_path):
    import shutil

    directory = case(tmp_path, "a")
    shutil.copytree(directory, tmp_path / "runs/retry/a")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"id": "a", "prepared": "prepared/a"}],
                "unavailable": {},
            }
        )
    )
    with pytest.raises(ValueError, match="Multiple clean replays"):
        summarize_native_references(tmp_path, "runs", "report", plan_path=plan)


def test_plan_requires_prepared_evidence_even_when_no_model_result_exists(tmp_path):
    (tmp_path / "runs").mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"schema_version": 1, "cases": [{"id": "pending"}], "unavailable": {}})
    )
    with pytest.raises(ValueError, match="frozen plan"):
        summarize_native_references(tmp_path, "runs", "report", plan_path=plan)
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [{"id": "pending", "prepared": "absent"}],
                "unavailable": {},
            }
        )
    )
    with pytest.raises(FileNotFoundError):
        summarize_native_references(tmp_path, "runs", "report", plan_path=plan)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("different", ["checkpoint", "steps", "seed_and_resources"])
def test_same_adapter_different_science_is_not_pooled_but_seeds_are(tmp_path, different):
    for index, name in enumerate(("a", "b")):
        directory = case(tmp_path, name)
        config = {
            "adapter": "FixtureAdapter",
            "video": {"fps": 10},
            "gpus": [index],
            "memory_gib": 16 + index,
            "seed": index,
            "kwargs": {
                "model_path": "checkpoint-a",
                "seed": index,
                "runtime_dir": f"scratch{index}",
                "steps": 30,
            },
        }
        if index and different == "checkpoint":
            config["kwargs"]["model_path"] = "checkpoint-b"
        if index and different == "steps":
            config["kwargs"]["steps"] = 50
        path = directory / "config.json"
        path.write_text(json.dumps(config))
        summary = directory / "summary.json"
        report = json.loads(summary.read_text())
        report["config_sha256"] = _digest(path)
        summary.write_text(json.dumps(report))
    result = summarize_native_references(tmp_path, "runs", "report")
    assert len(result["groups"]) == (1 if different == "seed_and_resources" else 2)
    if different == "seed_and_resources":
        assert result["groups"][0]["seeds"] == [0, 1]


def test_perceptual_implementation_versions_are_not_silently_mixed(tmp_path):
    for name in ("a", "b"):
        directory = case(tmp_path, name)
        summary = directory / "summary.json"
        report = json.loads(summary.read_text())
        (directory / "perceptual.json").write_text(
            json.dumps(
                {
                    "summary_sha256": _digest(summary),
                    "reference_sha256": report["reference_evaluation"]["reference_sha256"],
                    "implementation_sha256": {"fixture_metric.py": name * 64},
                    "views": {"camera": {"lpips_vgg": 0.1}},
                }
            )
        )
    result = summarize_native_references(tmp_path, "runs", "report")
    assert len(result["groups"]) == 2 and len(result["perceptual_implementations"]) == 2
    assert result["perceptual_scored_cases"] == ["a", "b"]
