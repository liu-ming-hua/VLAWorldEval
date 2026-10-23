"""Serial, recoverable queue for frozen WM-native studies, no private WM loop."""
import json
import hashlib
import math
import os
from pathlib import Path
import re
import time

from evaluation.execution.admission import ResourceAdmissionError, native_gpu_lease
from evaluation.execution.native_replay import run_native_replay
from evaluation.execution.reliability import run_signal_handlers
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.atomic_json import replace_owned_json, publish_json_exclusive


def _valid_gpu_map(value):
    return (isinstance(value, list) and bool(value)
            and all(type(gpu) is int and gpu >= 0 for gpu in value)
            and len(value) == len(set(value)))


def run_native_study(root, directory, *, watch=True, poll_seconds=30, resource_overrides=None,
                     exclusive_gpu=False, gpu_override=None, gpu_map=None, reuse_worker=False,
                     worker_session=None):
    root = Path(root).resolve()
    directory = project_path(root, str(directory))
    if type(reuse_worker) is not bool or (reuse_worker and worker_session is not None):
        raise ValueError("Use either owned resident worker or an explicit stream session")
    if reuse_worker:
        from evaluation.execution.worker_session import NativeWorkerSession
        with NativeWorkerSession(directory/"resident_session") as session:
            run_native_study(root, directory, watch=watch, poll_seconds=poll_seconds,
                resource_overrides=resource_overrides, exclusive_gpu=exclusive_gpu,
                gpu_override=gpu_override, gpu_map=gpu_map, worker_session=session)
        return json.loads((directory/"progress.json").read_text())
    study = json.loads((directory / "study.json").read_text())
    if type(poll_seconds) is not int or not 1 <= poll_seconds <= 60:
        raise ValueError("poll_seconds must be an integer in[1,60]")
    if type(exclusive_gpu) is not bool:
        raise ValueError("exclusive_gpu must be boolean")
    if gpu_override is not None and (type(gpu_override) is not int or gpu_override < 0):
        raise ValueError("Explicit GPU override must be a nonnegative integer")
    if gpu_map is not None and (gpu_override is not None or not _valid_gpu_map(gpu_map)):
        raise ValueError("Use one explicit GPU placement: --gpu or a unique nonempty --gpus rank map")
    required_gpu = study.get("runtime_gpu_override_required")
    if required_gpu is not None and (type(required_gpu) is not int or gpu_override != required_gpu):
        raise ValueError("Repartitioned study requires its explicit assigned --gpu")
    allowed = study["plan"]["allowed_gpus"]
    if (not isinstance(allowed, list) or any(type(g) is not int or g not in {0, 1, 2, 3} for g in allowed)
            or len(set(allowed)) != len(allowed)):
        raise ValueError("Study may only use GPU0–3")
    jobs = study["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Native study needs nonempty jobs")
    identifiers = []
    for job in jobs:
        name = job.get("id")
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name)
                or type(job.get("chunks")) is not int or job["chunks"] < 1):
            raise ValueError("Native jobs require safe unique IDs and positive chunk counts")
        identifiers.append(name)
        for key in ("input", "config", *(["reference"] if "reference" in job else [])):
            project_path(root, job[key])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Native jobs require safe unique IDs and positive chunk counts")
    overrides = {} if resource_overrides is None else resource_overrides
    environments = {json.loads(Path(job["config"]).read_text()).get("environment") for job in jobs}
    if (not isinstance(overrides, dict) or set(overrides) - environments
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in overrides.values())):
        raise ValueError("Resource overrides must map declared environments to finite positive GiB budgets")
    override = project_path(root, study["environments"])
    if _digest(override) != study["environments_sha256"]:
        raise ValueError("Frozen environment override changed")
    old_environment = os.environ.get("VLAWORLDEVAL_ENVIRONMENTS")
    os.environ["VLAWORLDEVAL_ENVIRONMENTS"] = study["environments"]
    try:
        gpus = sorted({gpu for job in jobs for gpu in json.loads(Path(job["config"]).read_text()).get("gpus", [])})
        if not set(gpus) <= set(allowed):
            raise ValueError("Job GPU assignment exceeds the frozen study")
        if gpu_override is not None:
            if any(len(json.loads(Path(job["config"]).read_text()).get("gpus", [])) != 1 for job in jobs):
                raise ValueError("Explicit single-GPU override cannot change distributed topology")
            gpus = [gpu_override]
        if gpu_map is not None:
            if any(len(json.loads(Path(job["config"]).read_text()).get("gpus", [])) != len(gpu_map) for job in jobs):
                raise ValueError("Explicit GPU rank map cannot change distributed topology")
            gpus = list(gpu_map)
        while True:
            # Acquire before starting any worker. Waiting queues consume no GPU
            # memory; independent studies never share RNG or latent histories.
            acquired = False
            try:
                with native_gpu_lease(root, gpus if exclusive_gpu else []):
                    acquired = True
                    if exclusive_gpu:
                        replace_owned_json(directory / "resource_wait.json", {
                            "status": "lease_acquired", "gpus": gpus, "exclusive_gpu": True})
                    try:
                        return _run_locked(root, directory, study, watch, poll_seconds, overrides,
                                           gpu_override, gpu_map, worker_session)
                    finally:
                        if exclusive_gpu:
                            replace_owned_json(directory / "resource_wait.json", {
                                "status": "lease_released", "gpus": gpus, "exclusive_gpu": True})
            except ResourceAdmissionError as exc:
                if acquired:
                    raise  # Never mask errors from a real inference attempt.
                replace_owned_json(directory / "resource_wait.json", {
                    "status": "waiting_for_gpu_lease", "gpus": gpus, "reason": str(exc),
                    "exclusive_gpu": True, "model_loaded": False})
                if not watch:
                    return {"status": "waiting_for_gpu_lease", "gpus": gpus}
                time.sleep(poll_seconds)
    finally:
        if old_environment is None:
            os.environ.pop("VLAWORLDEVAL_ENVIRONMENTS", None)
        else:
            os.environ["VLAWORLDEVAL_ENVIRONMENTS"] = old_environment


def _run_locked(root, directory, study, watch, poll_seconds, overrides, gpu_override=None, gpu_map=None,
                worker_session=None):
    progress_path = directory / "progress.json"
    with asset_destination_lock(root, directory):
        if worker_session is not None:
            worker_session.track_study(directory)
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "status": "running", "study_sha256": _digest(directory / "study.json"), "jobs": {}}
        if progress["study_sha256"] != _digest(directory / "study.json"):
            raise ValueError("Frozen study changed")
        if set(progress["jobs"]) - {job["id"] for job in study["jobs"]}:
            raise ValueError("Progress contains undeclared jobs")
        for job in study["jobs"]:
            if worker_session is not None and worker_session.stop_requested:
                break
            for key in ("input", "config", *(["reference"] if "reference" in job else [])):
                if _digest(Path(job[key])) != job[key + "_sha256"]:
                    raise ValueError("Frozen job input/config changed")
            previous = progress["jobs"].get(job["id"], {})
            if previous.get("status") == "generated_pending_worker_close":
                path = directory/"runs"/job["id"]/"summary.json"
                if (worker_session is None or path not in worker_session.reports
                        or _digest(path) != previous["report_sha256"]):
                    raise ValueError("Uncertified resident attempt requires an explicit new study")
                continue
            if previous.get("status") == "completed":
                if _digest(directory / "runs" / job["id"] / "summary.json") != previous["report_sha256"]:
                    raise ValueError("Completed report changed")
                continue
            if previous.get("status") in {"running", "failed"}:
                raise ValueError("Interrupted/failed attempts require explicit new study, not silent overwrite")
            config = json.loads(Path(job["config"]).read_text())
            if not set(config["gpus"]) <= set(study["plan"]["allowed_gpus"]):
                raise ValueError("Job GPU assignment exceeds the frozen study")
            effective = Path(job["config"])
            evaluation_options = {"reference_path": Path(job["reference"])} if "reference" in job else {}
            inference_options = {"worker_session": worker_session} if worker_session is not None else {}
            provenance = {}
            changed = []
            if config.get("environment") in overrides:
                config["memory_gib"] = overrides[config["environment"]]
                changed.append("memory_gib")
            if gpu_override is not None:
                config["gpus"] = [gpu_override]
                changed.append("gpus")
            if gpu_map is not None:
                config["gpus"] = list(gpu_map)
                changed.append("gpus")
            if changed:
                # A stopped queue may change its allocator budget while waiting
                # before an attempt exists. Keep old derivations immutable;
                # completed jobs retain their original per-case provenance.
                identity = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
                effective = directory / "resource_configs" / (job["id"] + "." + identity + ".json")
                effective.parent.mkdir(exist_ok=True)
                if effective.exists():
                    if json.loads(effective.read_text()) != config:
                        raise ValueError("Derived resource config already exists with different limits")
                else:
                    publish_json_exclusive(effective, config)
                provenance = {"effective_config": str(effective), "effective_config_sha256": _digest(effective),
                    "source_config_sha256": job["config_sha256"], "only_override": ",".join(changed)}
                if gpu_override is not None:
                    provenance["explicit_gpu_override"] = gpu_override
                if gpu_map is not None:
                    provenance["explicit_gpu_map"] = list(gpu_map)
            while True:
                try:
                    run_native_replay(effective, Path(job["input"]), directory / "runs" / job["id"],
                                      admit_only=True, **evaluation_options)
                    break
                except ResourceAdmissionError as exc:
                    progress.update(status="waiting_for_gpu")
                    progress["jobs"][job["id"]] = {"status": "waiting_for_gpu", "reason": str(exc), **provenance}
                    replace_owned_json(progress_path, progress)
                    if not watch:
                        return {"status": "waiting_for_gpu", "job": job["id"]}
                    time.sleep(poll_seconds)
            progress["status"] = "running"
            progress["jobs"][job["id"]] = {"status": "running", "chunks": job["chunks"], **provenance}
            replace_owned_json(progress_path, progress)
            try:
                while True:
                    try:
                        progress["status"] = "running"
                        progress["jobs"][job["id"]].update(status="running")
                        replace_owned_json(progress_path, progress)
                        with run_signal_handlers():
                            result = run_native_replay(effective, Path(job["input"]), directory / "runs" / job["id"],
                                                       **evaluation_options, **inference_options)
                        break
                    except ResourceAdmissionError as exc:
                        # Another user may allocate between preflight and actual
                        # admission. Only wait if no attempt has been created;
                        # never hide worker failures or overwrite partial output.
                        if (directory / "runs" / job["id"]).exists():
                            raise
                        progress["status"] = "waiting_for_gpu"
                        progress["jobs"][job["id"]].update(status="waiting_for_gpu", reason=str(exc))
                        replace_owned_json(progress_path, progress)
                        if not watch:
                            return {"status": "waiting_for_gpu", "job": job["id"]}
                        time.sleep(poll_seconds)
                expected_status = "completed" if worker_session is None else "generated_pending_worker_close"
                if result["status"] != expected_status:
                    raise ValueError("Native replay failed")
                progress["jobs"][job["id"]].update(status=result["status"], world_model_calls=result["world_model_calls"],
                    report_sha256=_digest(directory / "runs" / job["id"] / "summary.json"))
            except BaseException as exc:
                progress["jobs"][job["id"]].update(status="failed", error=str(exc))
                progress["status"] = "failed"
                replace_owned_json(progress_path, progress)
                raise
            replace_owned_json(progress_path, progress)
        progress["status"] = "completed" if worker_session is None else "generated_pending_worker_close"
        replace_owned_json(progress_path, progress)
    return {"status": progress["status"], "jobs": len(progress["jobs"]),
        "world_model_calls": sum(j["world_model_calls"] for j in progress["jobs"].values()), "vla_calls": 0}


def resolved_native_config(root, directory, job):
    """Resolve explicit memory/device provenance; reject scientific changes."""
    directory = Path(directory)
    progress = json.loads((directory / "progress.json").read_text())
    entry = progress["jobs"].get(job["id"], {})
    if "effective_config" not in entry:
        return Path(job["config"]), job["config_sha256"]
    path = project_path(root, entry["effective_config"])
    if (_digest(path) != entry["effective_config_sha256"] or entry["source_config_sha256"] != job["config_sha256"]
            or _digest(Path(job["config"])) != job["config_sha256"]):
        raise ValueError("Resource derivation provenance changed")
    old, actual = json.loads(Path(job["config"]).read_text()), json.loads(path.read_text())
    validate_resource_override(old, actual, entry)
    return path, entry["effective_config_sha256"]


def validate_resource_override(old, actual, entry):
    """Content-only validation also works on hash-bound copied study evidence."""
    allowed_fields = {"memory_gib"}
    if "explicit_gpu_override" in entry and "explicit_gpu_map" in entry:
        raise ValueError("Ambiguous GPU placement provenance")
    if "explicit_gpu_override" in entry:
        gpu = entry["explicit_gpu_override"]
        if (type(gpu) is not int or gpu < 0 or len(old.get("gpus", [])) != 1
                or actual.get("gpus") != [gpu]):
            raise ValueError("Invalid explicit single-GPU resource derivation")
        allowed_fields.add("gpus")
    if "explicit_gpu_map" in entry:
        placement = entry["explicit_gpu_map"]
        if (not _valid_gpu_map(placement) or len(old.get("gpus", [])) != len(placement)
                or actual.get("gpus") != placement):
            raise ValueError("Invalid explicit same-topology GPU rank map")
        allowed_fields.add("gpus")
    budget = actual.get("memory_gib")
    if (type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0
            or {k: v for k, v in old.items() if k not in allowed_fields}
            != {k: v for k, v in actual.items() if k not in allowed_fields}):
        raise ValueError("Only memory_gib and explicitly journalled same-topology GPU placement may differ in a resource-only config")
