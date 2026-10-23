"""Freeze a budget-limited native cohort without looking at model outcomes."""
import hashlib
import json
from pathlib import Path
import re

from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive


def select_cases(cases, limit=400, seed=42):
    if type(limit) is not int or limit < 1 or type(seed) is not int:
        raise ValueError("Positive integer limit and integer sampling seed required")
    identifiers = [case.get("id") for case in cases]
    if (not identifiers or any(not isinstance(i, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", i) for i in identifiers)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError("A nonempty population of unique safe case IDs is required")
    # Hash ranking is stable across Python versions and input ordering. Never
    # inspect completed counts, video quality, metrics or action magnitude.
    return sorted(cases, key=lambda c: (
        hashlib.sha256(f"{seed}:{c['id']}".encode()).hexdigest(), c["id"]))[:limit]


def freeze_reference_subset(root, plan_path, output, *, limit=400, seed=42):
    root = Path(root).resolve()
    plan_path, output = [project_path(root, str(p)) for p in (plan_path, output)]
    source_digest = _digest(plan_path)
    plan = json.loads(plan_path.read_text())
    if (set(plan) != {"schema_version", "scope", "cases", "unavailable"}
            or type(plan["schema_version"]) is not int or plan["schema_version"] != 1
            or not isinstance(plan["cases"], list) or not isinstance(plan["unavailable"], dict)):
        raise ValueError("A native reference population plan is required")
    for case in plan["cases"]:
        if set(case) != {"id", "prepared"}:
            raise ValueError("Native cases require id/prepared only")
        project_path(root, case["prepared"])
    cases = select_cases(plan["cases"], limit, seed)
    audit = {"schema_version": 1, "source_plan": str(plan_path.relative_to(root)),
             "source_plan_sha256": source_digest, "population_cases": len(plan["cases"]),
             "requested_cases": limit, "selected_cases": len(cases), "sampling_seed": seed,
             "sampling_method": "ascending SHA256 of UTF-8 seed:case_id; independent of outcomes",
             "selected_ids": [c["id"] for c in cases],
             "shortfall": max(0, limit-len(cases)), "full_test_set_claim": False,
             "full_source_population_selected": len(cases) == len(plan["cases"]),
             "trajectory_lengths_changed": False, "model_inference_performed": False}
    if _digest(plan_path) != source_digest:
        raise ValueError("Population changed during selection")
    output.mkdir(parents=True, exist_ok=False)
    publish_json_exclusive(output/"selection.json", audit)
    publish_json_exclusive(output/"plan.json", {
        **plan, "scope": f"Budget-limited native evaluation: {len(cases)}/{len(plan['cases'])} cases; "
        f"target {limit}, sampling seed {seed}; not a full-test-set claim; native horizons unchanged",
        "cases": cases})
    return {"status": "completed", **audit, "plan": str(output/"plan.json")}


def audit_reference_subset(root, plan_path, plan):
    """Reproduce an adjacent budget selection; never infer paper-test coverage."""
    root = Path(root).resolve()
    plan_path = project_path(root, str(plan_path))
    selection_path = plan_path.parent/"selection.json"
    if not selection_path.exists():
        return None
    selection_digest = _digest(selection_path)
    audit = json.loads(selection_path.read_text())
    source_path = project_path(root, audit["source_plan"])
    if _digest(source_path) != audit["source_plan_sha256"]:
        raise ValueError("Budget source population changed")
    source = json.loads(source_path.read_text())
    selected = select_cases(source["cases"], audit["requested_cases"], audit["sampling_seed"])
    expected = {
        "schema_version": 1, "population_cases": len(source["cases"]),
        "selected_cases": len(selected), "selected_ids": [c["id"] for c in selected],
        "sampling_method": "ascending SHA256 of UTF-8 seed:case_id; independent of outcomes",
        "shortfall": max(0, audit["requested_cases"]-len(selected)),
        "full_test_set_claim": False, "full_source_population_selected": len(selected) == len(source["cases"]),
        "trajectory_lengths_changed": False, "model_inference_performed": False,
    }
    expected_scope = (f"Budget-limited native evaluation: {len(selected)}/{len(source['cases'])} cases; "
                      f"target {audit['requested_cases']}, sampling seed {audit['sampling_seed']}; "
                      "not a full-test-set claim; native horizons unchanged")
    if (any(type(audit.get(k)) is not type(v) or audit.get(k) != v for k, v in expected.items())
            or plan != {**source, "scope": expected_scope, "cases": selected}):
        raise ValueError("Budget selection does not reproduce the frozen population and cohort")
    if _digest(selection_path) != selection_digest or _digest(source_path) != audit["source_plan_sha256"]:
        raise ValueError("Budget selection evidence changed during auditing")
    return {**audit, "selection_sha256": selection_digest,
            "selected_fraction_of_source_cases": len(selected)/len(source["cases"]),
            "source_membership_verified": True,
            "source_is_complete_official_test_set_verified": False}


def prepare_cosmos_subset(root, plan_path, dataset, selection_path, config_path,
                          cache_progress, output):
    """Materialize just the frozen subset, reusing checksum-bound native packets.

    No VLA/WM is loaded. Native all-chunk conversion remains in the same
    prepare_native_case implementation used for the original full holdout.
    """
    from evaluation.datasets.cosmos_reference import SnapshotChecks, prepare_native_case
    from evaluation.planning.cosmos import validate_selection
    from vlaworldeval.asset_lock import asset_destination_lock
    from vlaworldeval.atomic_json import replace_owned_json
    root = Path(root).resolve()
    plan_path, dataset, selection_path, config_path, cache_progress, output = [
        project_path(root, str(p)) for p in
        (plan_path, dataset, selection_path, config_path, cache_progress, output)]
    plan = json.loads(plan_path.read_text())
    selection = json.loads(selection_path.read_text())
    validate_selection(selection, dataset)
    rows = {r["id"]+"_seed0": r for r in selection["episodes"] if r["complete16_action_chunks"]}
    ids = [c["id"] for c in plan["cases"]]
    if len(ids) != 400 or len(set(ids)) != 400 or not set(ids).issubset(rows):
        raise ValueError("Expected400 unique native-eligible holdout episodes")
    cached = json.loads(cache_progress.read_text())
    if (cached["frozen"]["selection_sha256"] != _digest(selection_path)
            or cached["frozen"]["config_sha256"] != _digest(config_path)):
        raise ValueError("Cached native preparation uses another selection/config")
    sources = dict(cached["frozen"]["preparation_code_sha256"])
    for name, sha in sources.items():
        if _digest(root/name) != sha:
            raise ValueError("Native preparation implementation differs from frozen cache")
    sources[str(Path(__file__).resolve().relative_to(root))] = _digest(Path(__file__))
    checks = SnapshotChecks(dataset)
    if checks.manifest_sha256 != cached["frozen"]["snapshot_manifest_sha256"]:
        raise ValueError("Source dataset snapshot differs from cached preparation")
    frozen = {"plan_sha256": _digest(plan_path), "selection_sha256": _digest(selection_path),
              "config_sha256": _digest(config_path), "cache_progress_sha256": _digest(cache_progress),
              "snapshot_manifest_sha256": checks.manifest_sha256, "preparation_code_sha256": sources}
    with asset_destination_lock(root, output):
        output.mkdir(parents=True, exist_ok=True)
        journal = output/"progress.json"
        if journal.exists():
            progress = json.loads(journal.read_text())
            if progress["frozen"] != frozen:
                raise ValueError("Subset preparation inputs changed; use a new version")
        else:
            progress = {"schema_version": 1, "frozen": frozen, "status": "preparing", "completed": {}}
            publish_json_exclusive(journal, progress)
        try:
            for case in plan["cases"]:
                identifier = case["id"]
                directory = project_path(root, case["prepared"])
                row = rows[identifier]
                record = progress["completed"].get(identifier)
                is_cached = directory == cache_progress.parent/"cases"/identifier
                if record is None and is_cached:
                    record = cached["completed"][identifier]
                if record is None:
                    result = prepare_native_case(root, dataset/row["subset"], config_path, directory,
                        episode=row["episode"], chunks="all", _checks=checks,
                        _cohort={"subset": row["subset"], "selection_sha256": frozen["selection_sha256"],
                                 "membership": "reconstructed20k_loader_holdout_not_scene_disjoint"})
                    if result["source_length"] != row["length"]:
                        raise ValueError("Prepared trajectory length differs from selection")
                    record = {"chunks": result["chunks"], "sha256": {
                        name: _digest(directory/name) for name in ("input.npz", "config.json", "reference.npz")}}
                if (record["chunks"] != row["complete16_action_chunks"]
                        or any(_digest(directory/name) != sha for name, sha in record["sha256"].items())):
                    raise ValueError("Subset trajectory changed or native horizon was truncated")
                progress["completed"][identifier] = record
                progress["status"] = "preparing"
                progress.pop("error", None)
                replace_owned_json(journal, progress)
                print(f"prepared {len(progress['completed'])}/400 {identifier}", flush=True)
            manifest = {"schema_version": 1, "plan_sha256": frozen["plan_sha256"],
                        "cases": progress["completed"], "expected_cases": 400,
                        "world_model_calls": sum(r["chunks"] for r in progress["completed"].values()),
                        "full_test_set_claim": False, "trajectory_lengths_changed": False}
            path = output/"prepared_manifest.json"
            if not path.exists():
                publish_json_exclusive(path, manifest)
            elif json.loads(path.read_text()) != manifest:
                raise ValueError("Published subset manifest changed")
            progress["status"] = "prepared_not_evaluated"
            replace_owned_json(journal, progress)
            return {"status": "prepared_not_evaluated", "manifest": str(path), "cases": 400,
                    "world_model_calls": manifest["world_model_calls"]}
        except BaseException as exc:
            progress.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error={"type": type(exc).__name__, "message": str(exc)})
            replace_owned_json(journal, progress)
            raise
