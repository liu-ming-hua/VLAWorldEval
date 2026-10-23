"""Opt-in integration acceptance reusing diagnostics and the sole suite executor."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

from evaluation.execution.admission import select_jobs
from evaluation.planning.configuration import invalid, load_suite
from evaluation.planning.diagnostics import diagnose_suite
from evaluation.planning.experiments import ROOT, build_experiment_plan
from evaluation.planning.registry import load_runtime_registry, project_path
from evaluation.manifests import load_pair_manifest
from evaluation.execution.reliability import exception_details


def load_acceptance_matrix(path, *, root=ROOT):
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        invalid(path, "$", str(exc))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("entries"), list) or not data["entries"]:
        invalid(path, "$", "Expected schema_version=1 and nonempty entries")
    seen = set()
    for index, entry in enumerate(data["entries"]):
        field = f"entries[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"id", "suite", "smoke", "long", "seeds"}:
            invalid(path, field, "Required fields: id, suite, smoke, long, seeds (no extra fields)")
        identifier = entry["id"]
        if not isinstance(identifier, str) or not identifier or not all(c.isalnum() or c in "_-" for c in identifier) or identifier in seen:
            invalid(path, field + ".id", "Invalid/duplicate acceptance entry id")
        seen.add(identifier)
        if not isinstance(entry["seeds"], list) or not entry["seeds"] or any(type(seed) is not int or seed < 0 for seed in entry["seeds"]):
            invalid(path, field + ".seeds", "Expected nonnegative integer seeds")
        for level in ("smoke", "long"):
            values = entry[level]
            if not isinstance(values, list) or not values or any(not isinstance(item, str) for item in values) or len(values) != len(set(values)):
                invalid(path, field + "." + level, "Expected unique experiment IDs")
        try:
            suite = project_path(root, entry["suite"])
            plan = build_experiment_plan(suite, root=root)
            config = load_suite(suite)
            pair = load_pair_manifest(project_path(root, config["pair_manifest"]))
            profile = load_runtime_registry(root).pair_profile(pair)
            from evaluation.modes import POLICY_MODES
            for level in ("smoke", "long"):
                jobs = select_jobs(plan, entry[level], entry["seeds"])
                if level == "smoke" and (any(job["chunks"] > 3 for job in jobs)
                                          or {"policy_loop" if job["mode"] in POLICY_MODES else job["mode"] for job in jobs} != {"wm_replay", "policy_loop"}):
                    raise ValueError("Smoke requires replay and policy-loop jobs of at most 3 chunks")
                if level == "long" and any(job["mode"] not in POLICY_MODES for job in jobs):
                    raise ValueError("Long acceptance selects policy-loop jobs only")
                if any((job["mode"] == "wm_replay"
                        and profile["replay_transforms"].get(job["action_source"]) != "identity")
                       or job["state_provider"] in {"hold_initial", "dataset_ground_truth", "state_completion_prediction"}
                       for job in jobs):
                    raise ValueError("Engineering acceptance must not select action/state ablations")
        except (OSError, ValueError) as exc:
            invalid(path, field, str(exc))
    return data


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def run_acceptance(matrix_path, *, pairs=(), level="local", execute=False):
    if level not in {"local", "preflight", "smoke", "long"}:
        raise ValueError("Unknown acceptance level")
    inference = level in {"smoke", "long"}
    if inference and (not execute or not pairs):
        raise ValueError("Inference acceptance requires --execute and at least one explicit --pair")
    if execute and not inference:
        raise ValueError("--execute is only valid for smoke/long")
    matrix_path = Path(matrix_path)
    matrix_path = matrix_path if matrix_path.is_absolute() else ROOT / matrix_path
    matrix = load_acceptance_matrix(matrix_path)
    unknown = set(pairs) - {entry["id"] for entry in matrix["entries"]}
    if unknown:
        raise ValueError(f"Unknown acceptance pairs: {sorted(unknown)}")
    entries = [entry for entry in matrix["entries"] if not pairs or entry["id"] in pairs]
    directory = ROOT / "results/acceptance" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "report.json"
    report = {"schema_version": 1, "level": level, "status": "running", "report": str(path),
              "matrix": str(matrix_path),
              "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(), "selected_entries": entries,
              "inference_requested": inference, "quality_evaluated": False, "entries": []}
    _save(path, report)
    _save(directory / "matrix.json", matrix)
    try:
        for definition in entries:
            entry = {"id": definition["id"], "status": "running", "inference_completed": False}
            report["entries"].append(entry)
            _save(path, report)
            try:
                jobs = definition["long"] if level == "long" else definition["smoke"]
                diagnostic = diagnose_suite(ROOT / definition["suite"], experiments=jobs, seeds=definition["seeds"],
                                             check_data=level != "local", check_gpu=level != "local")
                entry["diagnostics"] = diagnostic
                if not diagnostic["ok"]:
                    entry["status"] = "failed_preflight"
                elif not inference:
                    entry["status"] = "local_passed" if level == "local" else "preflight_passed"
                else:
                    from evaluation.execution.executor import execute_suite
                    from evaluation.reporting.artifact_checks import audit_attempt

                    def started(attempt):
                        entry["attempt"] = str(attempt)
                        _save(path, report)

                    with redirect_stdout(sys.stderr):
                        summary = execute_suite(ROOT / definition["suite"], experiments=jobs,
                                                seeds=definition["seeds"], on_attempt=started)
                    entry["audit"] = audit_attempt(summary, expected_jobs=diagnostic.get("selection"))
                    entry["inference_completed"] = True
                    entry["status"] = "inference_passed" if entry["audit"]["ok"] else "failed_artifacts"
            except BaseException as exc:
                entry.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed_execution",
                             error_info=exception_details(exc))
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                _save(path, report)
            if inference and entry["status"] != "inference_passed":
                report["not_run_after_failure"] = [item["id"] for item in entries[len(report["entries"]):]]
                break
        report["ok"] = all(entry["status"] in {"local_passed", "preflight_passed", "inference_passed"} for entry in report["entries"])
        report["status"] = "passed" if report["ok"] else "failed"
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
                      ok=False, error_info=exception_details(exc))
        raise
    finally:
        _save(path, report)
    return report
