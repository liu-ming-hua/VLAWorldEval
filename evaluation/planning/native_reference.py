"""Freeze prepared reference-bound cases for the existing native study queue."""
import json
from contextlib import ExitStack
from pathlib import Path
import re

from adapters.environment import load_environment_registry
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive


def prepare_reference_study(root, plan_path, output):
    root = Path(root).resolve()
    plan_path = project_path(root, str(plan_path))
    output = project_path(root, str(output))
    plan = json.loads(plan_path.read_text())
    if (set(plan) != {"schema_version", "scope", "cases", "unavailable"}
            or type(plan["schema_version"]) is not int or plan["schema_version"] != 1
            or not isinstance(plan["scope"], str) or not isinstance(plan["unavailable"], dict)
            or not isinstance(plan["cases"], list) or not plan["cases"]):
        raise ValueError("Native reference plan requires cases and explicit unavailable conditions")
    jobs, identifiers = [], set()
    for case in plan["cases"]:
        if (not isinstance(case, dict) or set(case) != {"id", "prepared"}
                or not isinstance(case["id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", case["id"])
                or case["id"] in identifiers):
            raise ValueError("Native cases require safe unique IDs and a prepared directory")
        identifiers.add(case["id"])
        directory = project_path(root, case["prepared"])
        job = {"id": case["id"]}
        for name, file in (("config", "config.json"), ("input", "input.npz"), ("reference", "reference.npz")):
            path = project_path(root, str(directory / file))
            job[name], job[name+"_sha256"] = str(path), _digest(path)
        from evaluation.execution.transport import read_packet
        job["chunks"] = len(read_packet(Path(job["input"]))["actions"])
        jobs.append(job)
    specs = load_environment_registry()
    required = {json.loads(Path(job["config"]).read_text())["environment"] for job in jobs}
    if any(key not in specs or not specs[key].exists for key in required):
        raise ValueError("Every queued case requires an installed registered environment")
    output.mkdir(parents=True, exist_ok=False)
    environment = output / "environments.json"
    paths = {key: str(specs[key].path.resolve()) for key in sorted(required)}
    publish_json_exclusive(environment, {"schema_version": 1, "paths": paths})
    result = {"schema_version": 1, "plan": {"allowed_gpus": [0, 1, 2, 3], "scope": plan["scope"]},
              "source_plan": str(plan_path), "source_plan_sha256": _digest(plan_path), "jobs": jobs,
              "environments": str(environment), "environments_sha256": _digest(environment),
              "unavailable": plan["unavailable"], "vla_calls": 0, "model_inference_performed": False}
    publish_json_exclusive(output / "study.json", result)
    return result


def prepare_reference_retry(root, directory, output):
    """Retry missing cases without overwriting evidence or shrinking original coverage."""
    from vlaworldeval.asset_lock import asset_destination_lock
    from evaluation.execution.reliability import clean_worker_exit
    root = Path(root).resolve()
    directory, output = project_path(root, str(directory)), project_path(root, str(output))
    if output.exists():
        raise FileExistsError(output)
    with asset_destination_lock(root, directory):
        study_path = directory/"study.json"
        study = json.loads(study_path.read_text())
        progress = json.loads((directory/"progress.json").read_text())
        if progress["study_sha256"] != _digest(study_path) or progress["status"] not in {"failed", "completed"}:
            raise ValueError("Retry requires a stopped, unchanged source study")
        pending = []
        for job in study["jobs"]:
            for key in ("input", "config", *(["reference"] if "reference" in job else [])):
                if _digest(project_path(root, job[key])) != job[key+"_sha256"]:
                    raise ValueError("Retry must preserve exact admitted inputs/config/reference")
            entry = progress["jobs"].get(job["id"], {})
            if entry.get("status") == "completed":
                report = directory/"runs"/job["id"]/"summary.json"
                if (_digest(report) != entry["report_sha256"]
                        or not clean_worker_exit(json.loads(report.read_text()).get("cleanup"))):
                    raise ValueError("Completed source result changed or did not exit cleanly")
            else:
                pending.append(job)
        if not pending:
            raise ValueError("No missing cases to retry")
        result = {**study, "jobs": pending, "retry_of": str(study_path),
                  "retry_source_sha256": _digest(study_path),
                  "original_expected_cases": len(study["jobs"]), "skipped_completed_cases": len(study["jobs"])-len(pending)}
        output.mkdir(parents=True)
        publish_json_exclusive(output/"study.json", result)
    return result


def prepare_reference_remainder(root, plan_path, directories, output, gpus):
    """Freeze only missing full-plan cases, admitting cross-host/retry evidence.

    Unlike repartition, source studies may be overlapping historical attempts.
    The shared selector rejects duplicate successful cases and validates their
    input/config/reference bindings. Nothing is run or scored here.
    """
    from evaluation.reporting.checkpoint_metrics import select_replays
    from evaluation.execution.transport import read_packet
    root = Path(root).resolve()
    plan_path, output = [project_path(root, str(p)) for p in (plan_path, output)]
    if output.exists():
        raise FileExistsError(output)
    if (not isinstance(gpus, list) or not gpus or len(set(gpus)) != len(gpus)
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError("Unique nonnegative target GPUs required")
    if not directories:
        raise ValueError("Explicit replay source directories required")
    plan_hash = _digest(plan_path)
    plan = json.loads(plan_path.read_text())
    selected, excluded = select_replays(root, directories[0], plan,
                                       additional_directories=directories[1:])
    reused = {}
    for identifier, (path, _) in selected.items():
        report = json.loads(path.read_text())
        for name, sha in report["files"].items():
            target = (path.parent/name).resolve()
            if not target.is_relative_to(path.parent.resolve()) or _digest(target) != sha:
                raise ValueError("Previously completed replay artifact changed")
        reused[identifier] = {"summary": str(path.relative_to(root)), "sha256": _digest(path)}
    pending = [case for case in plan["cases"] if case["id"] not in selected]
    shards, costs = {gpu: [] for gpu in gpus}, dict.fromkeys(gpus, 0)
    inputs = {}
    for case in pending:
        prepared = project_path(root, case["prepared"])
        hashes = {name: _digest(prepared/name) for name in ("config.json", "input.npz", "reference.npz")}
        chunks = len(read_packet(prepared/"input.npz")["actions"])
        inputs[case["id"]] = {"sha256": hashes, "chunks": chunks}
    for case in sorted(pending, key=lambda c: (-inputs[c["id"]]["chunks"], c["id"])):
        gpu = min(gpus, key=lambda g: costs[g])
        shards[gpu].append(case)
        costs[gpu] += inputs[case["id"]]["chunks"] + 1
    if _digest(plan_path) != plan_hash:
        raise ValueError("Full plan changed during remainder audit")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "source_plan": str(plan_path.relative_to(root)),
              "source_plan_sha256": plan_hash, "planned_cases": len(plan["cases"]),
              "reused_cases": reused, "pending_cases": len(pending),
              "pending_inputs": inputs, "excluded_attempts": excluded,
              "inference_performed": False, "full_metrics_completed": False, "shards": []}
    for gpu, cases in shards.items():
        if not cases:
            continue
        path = output/f"gpu{gpu}_plan.json"
        publish_json_exclusive(path, {**plan, "cases": cases,
            "scope": "Exact remaining cases from the full frozen plan; not a new sampled cohort"})
        result["shards"].append({"gpu": gpu, "plan": str(path.relative_to(root)),
                                 "cases": len(cases), "estimated_calls_plus_loads": costs[gpu]})
    publish_json_exclusive(output/"remainder.json", result)
    return result


def prepare_reference_repartition(root, directories, output, gpus):
    """Rebalance stopped shards without rewriting attempts or scientific inputs.

    Device placement is an explicit run-time override. Original config hashes,
    source GPU declarations and completed evidence remain frozen.
    """
    from vlaworldeval.asset_lock import asset_destination_lock
    from evaluation.execution.reliability import clean_worker_exit
    root = Path(root).resolve()
    output = project_path(root, str(output))
    sources = [project_path(root, str(path)) for path in directories]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Distinct stopped source studies are required")
    if (not isinstance(gpus, list) or not gpus or len(set(gpus)) != len(gpus)
            or any(type(gpu) is not int or gpu < 0 for gpu in gpus)):
        raise ValueError("Unique nonnegative target GPUs are required")
    if output.exists():
        raise FileExistsError(output)
    pending, seen, audits, source_allowed = [], set(), [], set()
    environment = None
    with ExitStack() as locks:
        for directory in sorted(sources):
            locks.enter_context(asset_destination_lock(root, directory))
        for directory in sources:
            study_path, progress_path = directory/"study.json", directory/"progress.json"
            study, progress = json.loads(study_path.read_text()), json.loads(progress_path.read_text())
            if (progress.get("status") not in {"failed", "completed"}
                    or progress.get("study_sha256") != _digest(study_path)):
                raise ValueError("Repartition requires stopped unchanged source studies")
            env_path = project_path(root, study["environments"])
            if _digest(env_path) != study["environments_sha256"]:
                raise ValueError("Frozen environment override changed")
            current_environment = json.loads(env_path.read_text())
            if environment is not None and current_environment != environment:
                raise ValueError("Repartition requires the same installed environments")
            environment = current_environment
            source_allowed.update(study["plan"]["allowed_gpus"])
            done = 0
            for job in study["jobs"]:
                if job["id"] in seen:
                    raise ValueError("Overlapping source studies would duplicate evaluation cases")
                seen.add(job["id"])
                for key in ("input", "config", *(["reference"] if "reference" in job else [])):
                    if _digest(project_path(root, job[key])) != job[key+"_sha256"]:
                        raise ValueError("Repartition must preserve exact frozen input/config/reference")
                if len(json.loads(Path(job["config"]).read_text()).get("gpus", [])) != 1:
                    raise ValueError("Single-GPU repartition cannot change distributed topology")
                entry = progress["jobs"].get(job["id"], {})
                if entry.get("status") == "completed":
                    path = directory/"runs"/job["id"]/"summary.json"
                    if (_digest(path) != entry["report_sha256"]
                            or not clean_worker_exit(json.loads(path.read_text()).get("cleanup"))):
                        raise ValueError("Completed source report changed or cleanup failed")
                    done += 1
                else:
                    pending.append(job)
            audits.append({"study": str(study_path), "study_sha256": _digest(study_path),
                "progress": str(progress_path), "progress_sha256": _digest(progress_path),
                "declared_cases": len(study["jobs"]), "skipped_completed_cases": done})
        if not pending:
            raise ValueError("No pending cases to repartition")
        # A fixed per-episode load cost matters alongside denoising calls.
        shards = {gpu: [] for gpu in gpus}
        costs = dict.fromkeys(gpus, 0)
        for job in sorted(pending, key=lambda job: -job["chunks"]):
            gpu = min(gpus, key=lambda gpu: costs[gpu])
            shards[gpu].append(job)
            costs[gpu] += job["chunks"]+1
        output.mkdir(parents=True)
        env_path = output/"environments.json"
        publish_json_exclusive(env_path, environment)
        result = {"schema_version": 1, "source_studies": audits, "target_gpus": gpus,
            "source_declared_cases": len(seen), "skipped_completed_cases": len(seen)-len(pending),
            "pending_cases": len(pending), "pending_model_calls": sum(job["chunks"] for job in pending),
            "scientific_inputs_changed": False, "shards": []}
        for gpu, jobs in shards.items():
            if not jobs:
                continue
            directory = output/f"gpu{gpu}"
            directory.mkdir()
            study = {"schema_version": 1, "plan": {"allowed_gpus": sorted(source_allowed),
                "scope": "Remaining exact native cases from stopped studies; explicit runtime GPU relocation"},
                "jobs": sorted(jobs, key=lambda job: job["chunks"]),
                "environments": str(env_path), "environments_sha256": _digest(env_path),
                "source_studies": audits, "runtime_gpu_override_required": gpu,
                "vla_calls": 0, "model_inference_performed": False}
            publish_json_exclusive(directory/"study.json", study)
            result["shards"].append({"gpu": gpu, "directory": str(directory), "cases": len(jobs),
                "model_calls": sum(job["chunks"] for job in jobs), "study_sha256": _digest(directory/"study.json")})
        publish_json_exclusive(output/"repartition.json", result)
    return result
