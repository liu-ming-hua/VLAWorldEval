"""Event selection is recorded-only, fixed and fail-closed before inference."""

import json

import numpy as np
import pytest

from evaluation.planning.cosmos_events import event_window, freeze_event_study
from evaluation.reporting.task_scores import _digest

pytestmark = pytest.mark.process


def test_event_window_uses_first_sensor_crossing_not_generated_quality():
    values = np.zeros(200)
    values[5:10] = 1  # Ignore pre-frame15 crossings.
    values[80:100] = 1
    values[140:] = 1
    assert event_window(values) == {"event": 80, "start": 48, "end": 112}
    values = np.zeros(100)
    values[16:] = 0.6
    assert event_window(values) == {"event": 16, "start": 15, "end": 79}


@pytest.mark.parametrize("values", [[], [0] * 100, [float("nan")], [-0.1], [1.1], [[0], [1]]])
def test_event_window_rejects_invalid_or_absent_events(values):
    with pytest.raises(ValueError):
        event_window(values)


def test_event_window_never_pads_tail():
    with pytest.raises(ValueError, match="tail padding"):
        event_window([0] * 80 + [1] * 10)


@pytest.fixture
def preparation(tmp_path):
    root = tmp_path
    directory = root / "study"
    directory.mkdir()

    def asset(name):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        return str(path), _digest(path)

    plan = {"parent": "parent"}
    _, plan_sha = asset("configs/studies/cosmos_event_windows_v1.json")
    _, parent_sha = asset("parent/study.json")
    env, env_sha = asset("environments.json")
    cases, notes = [], []
    for episode in (0, 351, 365, 795, 903):
        path, sha = asset(f"ep{episode}.jpg")
        data, data_sha = asset(f"ep{episode}.parquet")
        ref, ref_sha = asset(f"ep{episode}.mp4")
        cases.append(
            {
                "episode": episode,
                "event_sheet": path,
                "event_sheet_sha256": sha,
                "data_path": data,
                "data_sha256": data_sha,
                "references": {"composite": {"path": ref, "sha256": ref_sha}},
            }
        )
        notes.append(
            {
                "episode": episode,
                "relevance": "uncertain",
                "notes": "Sensor event is not verified contact",
                "contact_ground_truth": False,
                "task_success": None,
            }
        )
    job = {}
    for name in ("input", "config", "template_config"):
        job[name], job[name + "_sha256"] = asset(name)
    prepared = {
        "plan": plan,
        "plan_sha256": plan_sha,
        "parent_study_sha256": parent_sha,
        "environments": env,
        "environments_sha256": env_sha,
        "cases": cases,
        "jobs": [job],
    }
    (directory / "preparation.json").write_text(json.dumps(prepared))
    (directory / "source_review.json").write_text(
        json.dumps({"reviewer": "assistant", "human_review_performed": False, "cases": notes})
    )
    from evaluation.planning.cosmos_events import INFERENCE_SOURCES

    for name in INFERENCE_SOURCES:
        asset(name)
    return root, directory, prepared


def test_freeze_requires_review_before_publish_and_never_overwrites(preparation):
    root, directory, _ = preparation
    assert not (directory / "study.json").exists()
    result = freeze_event_study(root, directory)
    assert result["status"] == "frozen_before_inference"
    assert len(result["cases"]) == 5
    with pytest.raises(FileExistsError):
        freeze_event_study(root, directory)


@pytest.mark.parametrize(
    "fault", ["reference", "environment", "input", "parent", "review", "drop", "human"]
)
def test_freeze_rejects_mutations_or_implicit_success(preparation, fault):
    root, directory, prepared = preparation
    if fault == "reference":
        path = prepared["cases"][0]["references"]["composite"]["path"]
    elif fault == "environment":
        path = prepared["environments"]
    elif fault == "input":
        path = prepared["jobs"][0]["input"]
    elif fault == "parent":
        path = root / "parent/study.json"
    else:
        path = None
        review = json.loads((directory / "source_review.json").read_text())
        if fault == "drop":
            review["cases"].pop()
        elif fault == "human":
            review["human_review_performed"] = True
        else:
            review["cases"][0]["task_success"] = True
        (directory / "source_review.json").write_text(json.dumps(review))
    if path:
        from pathlib import Path

        Path(path).write_text("mutated")
    with pytest.raises(ValueError):
        freeze_event_study(root, directory)
    assert not (directory / "study.json").exists()


@pytest.mark.parametrize("status", ["running", "waiting_for_gpu", "failed", "interrupted"])
def test_finalizer_never_approves_or_retries_an_incomplete_queue(tmp_path, monkeypatch, status):
    import evaluation.reporting.cosmos_events as module

    (tmp_path / "progress.json").write_text(json.dumps({"status": status}))
    monkeypatch.setattr(module, "audit_event_study", lambda *_: pytest.fail("No incomplete audit"))
    monkeypatch.setattr(
        module, "render_event_study", lambda *_: pytest.fail("No incomplete comparison")
    )
    with pytest.raises((ValueError, TimeoutError)):
        module.finalize_event_study(tmp_path, tmp_path)
    assert not (tmp_path / "finalization.json").exists()


def test_finalizer_is_bounded_and_does_not_grant_scientific_approval(tmp_path, monkeypatch):
    import evaluation.reporting.cosmos_events as module

    for file in ("study.json", "execution_audit.json", "comparisons.json", "index.html"):
        (tmp_path / file).write_text("fixture")
    (tmp_path / "progress.json").write_text('{"status":"completed"}')
    monkeypatch.setattr(module, "audit_event_study", lambda *_: {"world_model_calls": 60})
    monkeypatch.setattr(module, "render_event_study", lambda *_: {})
    for kwargs in ({"timeout": True}, {"timeout": 0}, {"poll_seconds": 61}, {"wait": 1}):
        with pytest.raises(ValueError):
            module.finalize_event_study(tmp_path, tmp_path, **kwargs)
    result = module.finalize_event_study(tmp_path, tmp_path)
    assert result["world_model_calls"] == 60 and result["vla_calls"] == 0
    assert result["quality_admitted"] is False and result["task_success"] is None
    assert result["status"] == "artifacts_complete_pending_generated_quality_review"
    with pytest.raises(FileExistsError):
        module.finalize_event_study(tmp_path, tmp_path)


def test_finalizer_wait_timeout_does_not_launch_inference(tmp_path, monkeypatch):
    import evaluation.reporting.cosmos_events as module

    times = iter((0, 0, 2, 2))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="incomplete"):
        module.finalize_event_study(tmp_path, tmp_path, wait=True, timeout=1)


def test_event_cli_rejects_ignored_wait_flag():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts/checks/cosmos_events.py"
    result = subprocess.run(
        [sys.executable, str(script), "render", "--directory", "unused", "--wait"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "only valid for finalize" in result.stderr
