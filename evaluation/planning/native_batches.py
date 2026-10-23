"""Append-only ready batches of a frozen cohort; never replace pending cases."""
import hashlib
import json
from pathlib import Path

from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.atomic_json import publish_json_exclusive

FILES = ("input.npz", "config.json", "reference.npz")


def freeze_ready_batch(root, plan_path, progress_path, output, *, limit=32):
    root = Path(root).resolve()
    plan_path, progress_path, output = [project_path(root, str(p)) for p in
                                        (plan_path, progress_path, output)]
    if type(limit) is not int or limit < 1:
        raise ValueError("Positive batch limit required")
    cohort = json.loads(plan_path.read_text())
    digest = _digest(plan_path)
    payload = progress_path.read_bytes()
    progress = json.loads(payload)
    if progress["frozen"]["plan_sha256"] != digest:
        raise ValueError("Preparation is not bound to this frozen cohort")
    population = {c["id"]: c for c in cohort["cases"]}
    if not population or len(population) != len(cohort["cases"]):
        raise ValueError("Nonempty unique cohort required")
    with asset_destination_lock(root, output):
        output.mkdir(parents=True, exist_ok=True)
        used = set()
        previous = sorted(output.glob("batch*/prepared_manifest.json"))
        for path in previous:
            record = json.loads(path.read_text())
            identifiers = set(record["cases"])
            batch_cases = json.loads((path.parent/"plan.json").read_text())["cases"]
            if (record["cohort_plan_sha256"] != digest or used & identifiers
                    or not identifiers.issubset(population)
                    or identifiers != {c["id"] for c in batch_cases}
                    or any(population.get(c["id"]) != c for c in batch_cases)
                    or _digest(path.parent/"plan.json") != record["plan_sha256"]):
                raise ValueError("Prior batch changed, overlaps, or belongs to another cohort")
            used.update(identifiers)
        ready = [c for c in cohort["cases"]
                 if c["id"] not in used and c["id"] in progress["completed"]][:limit]
        if not ready:
            return {"status": "completed", "ready": False, "all_scheduled": len(used) == len(population),
                    "scheduled_cases": len(used), "expected_cases": len(population)}
        records = {}
        for case in ready:
            record = progress["completed"][case["id"]]
            directory = project_path(root, case["prepared"])
            if set(record["sha256"]) != set(FILES) or any(
                    _digest(directory/name) != record["sha256"][name] for name in FILES):
                raise ValueError("Ready case differs from checksum-bound preparation")
            records[case["id"]] = record
        batch = output/f"batch{len(previous):04d}"
        batch.mkdir(exist_ok=False)
        publish_json_exclusive(batch/"plan.json", {**cohort, "cases": ready,
            "scope": f"Disjoint ready batch: {len(ready)} of {len(population)} frozen cases; "
                     "remaining cohort stays pending, no horizon changes"})
        publish_json_exclusive(batch/"prepared_manifest.json", {
            "schema_version": 1, "plan_sha256": _digest(batch/"plan.json"),
            "cohort_plan": str(plan_path.relative_to(root)), "cohort_plan_sha256": digest,
            "preparation_snapshot_sha256": hashlib.sha256(payload).hexdigest(), "cases": records})
        return {"status": "completed", "ready": True, "batch": batch.name,
                "plan": str((batch/"plan.json").relative_to(root)),
                "manifest": str((batch/"prepared_manifest.json").relative_to(root)),
                "batch_cases": len(ready), "scheduled_cases": len(used)+len(ready),
                "expected_cases": len(population), "all_scheduled": len(used)+len(ready) == len(population)}


def validate_ready_batch(root, plan_path, manifest_path):
    root = Path(root).resolve()
    plan_path, manifest_path = [project_path(root, str(p)) for p in (plan_path, manifest_path)]
    evidence = json.loads(manifest_path.read_text())
    cohort_path = project_path(root, evidence["cohort_plan"])
    if (_digest(plan_path) != evidence["plan_sha256"]
            or _digest(cohort_path) != evidence["cohort_plan_sha256"]):
        raise ValueError("Batch/cohort manifest changed during transfer")
    cohort = {c["id"]: c for c in json.loads(cohort_path.read_text())["cases"]}
    cases = json.loads(plan_path.read_text())["cases"]
    if (not cases or len({c["id"] for c in cases}) != len(cases)
            or set(evidence["cases"]) != {c["id"] for c in cases}):
        raise ValueError("Batch manifest membership mismatch")
    for case in cases:
        if cohort.get(case["id"]) != case:
            raise ValueError("Batch is not an exact subset of the frozen cohort")
        directory = project_path(root, case["prepared"])
        for name in FILES:
            if _digest(directory/name) != evidence["cases"][case["id"]]["sha256"][name]:
                raise ValueError("Cross-host input/reference/config checksum mismatch")
