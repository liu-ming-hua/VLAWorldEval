import json

import pytest

from evaluation.planning.native_batches import FILES, freeze_ready_batch, validate_ready_batch
from evaluation.reporting.task_scores import _digest


def fixture(tmp_path):
    cases, completed = [], {}
    for i in range(5):
        d = tmp_path / "datasets" / f"case{i}"
        d.mkdir(parents=True)
        for name in FILES:
            (d / name).write_bytes(f"{i}:{name}".encode())
        identifier = f"case{i}"
        cases.append({"id": identifier, "prepared": str(d.relative_to(tmp_path))})
        completed[identifier] = {"chunks": 2, "sha256": {name: _digest(d / name) for name in FILES}}
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"schema_version": 1, "scope": "frozen", "cases": cases, "unavailable": {}})
    )
    progress = tmp_path / "progress.json"
    value = {
        "frozen": {"plan_sha256": _digest(plan)},
        "completed": dict(list(completed.items())[:3]),
    }
    progress.write_text(json.dumps(value))
    return plan, progress, value, completed


def test_incremental_batches_cover_original_population_exactly_once(tmp_path):
    plan, progress, value, completed = fixture(tmp_path)
    out = tmp_path / "batches"
    first = freeze_ready_batch(tmp_path, plan, progress, out, limit=1)
    second = freeze_ready_batch(tmp_path, plan, progress, out, limit=2)
    assert first["batch_cases"] == 1 and second["batch_cases"] == 2
    waiting = freeze_ready_batch(tmp_path, plan, progress, out)
    assert not waiting["ready"] and not waiting["all_scheduled"]
    value["completed"] = completed
    progress.write_text(json.dumps(value))
    last = freeze_ready_batch(tmp_path, plan, progress, out)
    assert last["batch_cases"] == 2 and last["all_scheduled"]
    ids = []
    for manifest in sorted(out.glob("batch*/prepared_manifest.json")):
        ids.extend(json.loads(manifest.read_text())["cases"])
        validate_ready_batch(tmp_path, manifest.parent / "plan.json", manifest)
    assert len(ids) == len(set(ids)) == 5
    assert freeze_ready_batch(tmp_path, plan, progress, out)["all_scheduled"]


def test_changed_prepared_payload_cannot_be_admitted_after_transfer(tmp_path):
    plan, progress, _, _ = fixture(tmp_path)
    batch = freeze_ready_batch(tmp_path, plan, progress, tmp_path / "batches", limit=1)
    (tmp_path / "datasets/case0/input.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        validate_ready_batch(tmp_path, batch["plan"], batch["manifest"])


def test_changed_parent_plan_cannot_silently_change_remaining_cohort(tmp_path):
    plan, progress, _, _ = fixture(tmp_path)
    freeze_ready_batch(tmp_path, plan, progress, tmp_path / "batches", limit=1)
    plan.write_text(plan.read_text() + "\n")
    with pytest.raises(ValueError, match="bound"):
        freeze_ready_batch(tmp_path, plan, progress, tmp_path / "batches")
