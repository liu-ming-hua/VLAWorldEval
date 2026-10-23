"""Postprocess a complete frozen replay plan through registered evaluator runtimes.

No policy/world-model calls. Failed attempts remain visible and retries must
produce a separate clean replay with the identical admitted input/reference.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from adapters.environment import get_environment
from evaluation.execution.reliability import clean_worker_exit
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.cache import runtime_cache_environment


@contextmanager
def scoring_failure_journal(progress, journal):
    """Source drift/signals must not leave a falsely running metrics journal."""
    try:
        yield
    except BaseException as exc:
        journal.update(status="failed" if isinstance(exc, Exception) else "interrupted",
                       error={"type": type(exc).__name__, "message": str(exc)})
        replace_owned_json(progress, journal)
        raise


def metric_source_fingerprints(kind):
    code_root = Path(__file__).resolve().parents[2]
    files = ["evaluation/datasets/native_reference.py", "evaluation/datasets/video.py", "evaluation/reporting/metric_store.py"]
    files += (["evaluation/reporting/perceptual.py"] if kind == "lpips" else
              [f"evaluation/reporting/{name}.py" for name in ("motion", "motion_detector", "pusht_observer", "native_reference")])
    return {name: _digest(code_root/name) for name in files}


def admitted_resource_config(root, path, report, hashes, name, *, original_config_path=None):
    """Accept only journalled memory/device derivations, never scientific changes."""
    if report.get("config_sha256") == hashes["config_sha256"]:
        return True
    if (path.parents[2]/"batch_admission.json").is_file():
        return admitted_batch_config(root, path, report, hashes, name)
    from evaluation.execution.native_study import resolved_native_config
    try:
        directory = project_path(root, str(path.parents[2]))
        study_path = directory/"study.json"
        study = json.loads(study_path.read_text())
        progress = json.loads((directory/"progress.json").read_text())
        if progress["study_sha256"] != _digest(study_path):
            return False
        jobs = [job for job in study["jobs"] if job["id"] == name]
        if len(jobs) != 1 or any(jobs[0].get(key) != value for key, value in hashes.items()):
            return False
        if original_config_path is not None and (path.parent/"config.json").is_file():
            # Historical absolute machine paths need not exist here. Validate
            # the exact copied study/progress, original admitted config, and
            # saved effective bytes without rewriting any historical evidence.
            from evaluation.execution.native_study import validate_resource_override
            entry = progress["jobs"][name]
            saved = path.parent/"config.json"
            if (entry["source_config_sha256"] != hashes["config_sha256"]
                    or _digest(original_config_path) != hashes["config_sha256"]
                    or _digest(saved) != entry["effective_config_sha256"]
                    or report["config_sha256"] != entry["effective_config_sha256"]):
                return False
            validate_resource_override(json.loads(original_config_path.read_text()), json.loads(saved.read_text()), entry)
            return True
        _, effective_hash = resolved_native_config(root, directory, jobs[0])
        return report.get("config_sha256") == effective_hash
    except (OSError, ValueError, KeyError, TypeError):
        return False


def admitted_batch_config(root, path, report, hashes, name):
    """Allow explicit batching/resource deltas, never a scientific protocol delta."""
    try:
        manifest = json.loads((path.parents[2]/"batch_admission.json").read_text())
        if (manifest["kind"] != "native_cross_trajectory_batch"
                or _digest(project_path(root, manifest["source_plan"])) != manifest["source_plan_sha256"]):
            return False
        entry = manifest["cases"][name]
        original_path = project_path(root, entry["original_config"])
        effective_path = project_path(root, entry["effective_config"])
        if (_digest(original_path) != hashes["config_sha256"]
                or entry["original_sha256"] != hashes["config_sha256"]
                or _digest(effective_path) != entry["effective_sha256"]
                or report["config_sha256"] != entry["effective_sha256"]
                or _digest(path.parent/"config.json") != entry["effective_sha256"]):
            return False
        original, effective = [json.loads(p.read_text()) for p in (original_path, effective_path)]
        batch = effective["kwargs"].pop("inference_batch_size")
        if type(batch) is not int or batch < 1 or batch != manifest["batch_size"]:
            return False
        for key in ("gpus", "memory_gib"):
            effective[key] = original[key]
        return original == effective and original["adapter"] == "Cosmos3DroidFDNativeAdapter"
    except (OSError, KeyError, ValueError, TypeError):
        return False


def replay_summary_paths(root, directory, additional_directories=()):
    """Explicit disjoint roots combine migrated runs without scanning preflights."""
    directories = [project_path(root, str(p)) for p in (directory, *additional_directories)]
    for index, path in enumerate(directories):
        if any(path == other or path.is_relative_to(other) or other.is_relative_to(path)
               for other in directories[:index]):
            raise ValueError("Replay roots must be distinct and nonoverlapping")
    return sorted(path for directory in directories for path in directory.rglob("summary.json"))


def select_replays(root, directory, plan, *, expected=None, additional_directories=()):
    if (not isinstance(plan, dict) or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or not isinstance(plan.get("cases"), list) or not plan["cases"]
            or any(not isinstance(c, dict) or not isinstance(c.get("id"), str)
                   or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", c["id"])
                   or not isinstance(c.get("prepared"), str) or not c["prepared"] for c in plan["cases"])):
        raise ValueError("Nonempty frozen plan with safe case IDs required")
    wanted = {case["id"]: case for case in plan["cases"]}
    if len(wanted) != len(plan["cases"]):
        raise ValueError("Duplicate evaluation case IDs")
    selected, excluded = {}, []
    for path in replay_summary_paths(root, directory, additional_directories):
        if not path.resolve().is_relative_to(Path(root).resolve()):
            raise ValueError("Replay summary leaves the project")
        name = path.parent.name
        if name not in wanted:
            continue
        report = json.loads(path.read_text())
        if report.get("status") != "completed" or not clean_worker_exit(report.get("cleanup")):
            excluded.append({"case": name, "summary": str(path), "error": report.get("error")})
            continue
        prepared = project_path(root, wanted[name]["prepared"])
        hashes = {key+"_sha256": _digest(prepared/file) for key, file in
                  (("input", "input.npz"), ("reference", "reference.npz"), ("config", "config.json"))}
        if expected is not None and hashes != expected[name]:
            raise ValueError("Prepared input/reference/config changed during scoring")
        if (report.get("input_sha256") != hashes["input_sha256"]
                or report.get("reference_evaluation", {}).get("reference_sha256") != hashes["reference_sha256"]
                or not admitted_resource_config(root, path, report, hashes, name,
                                                original_config_path=prepared/"config.json")):
            raise ValueError("Replay differs from the full plan's admitted input/reference")
        if name in selected:
            raise ValueError("Multiple clean replays for the same frozen case; select explicitly, do not cherry-pick")
        selected[name] = (path, prepared/"reference.npz")
    return selected, excluded


def score_checkpoint_study(root, plan_path, directory, output, *, watch=False, workers=2, provider="none",
                           timeout_seconds=3600, additional_directories=(), metric_gpus=(), frame_batch_size=1):
    root = Path(root).resolve()
    plan_path, directory, output = [project_path(root, str(p)) for p in (plan_path, directory, output)]
    additional_directories = tuple(project_path(root, str(p)) for p in additional_directories)
    if type(workers) is not int or not 1 <= workers <= 4 or provider not in {"ewmbench", "pusht_color", "none"}:
        raise ValueError("Use1..4 evaluator workers and an explicit observation provider")
    if (not isinstance(metric_gpus, (list, tuple)) or any(type(g) is not int or g not in {0, 1, 2, 3} for g in metric_gpus)
            or len(metric_gpus) != len(set(metric_gpus)) or (metric_gpus and workers > len(metric_gpus))
            or type(frame_batch_size) is not int or not 1 <= frame_batch_size <= 32):
        raise ValueError("Use unique metric GPUs0..3, at most one worker per GPU, and frame batch1..32")
    from queue import Queue
    from contextlib import nullcontext
    from evaluation.execution.admission import native_gpu_lease, resource_admission
    gpu_queue = Queue()
    for gpu in metric_gpus:
        gpu_queue.put(gpu)
    if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
            or not 1 <= timeout_seconds <= 604800):
        raise ValueError("Metric timeout must be finite,1..604800 seconds")
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads(plan_path.read_text())
    select_replays(root, directory, plan, additional_directories=additional_directories)
    expected = {case["id"]: {key+"_sha256": _digest(project_path(root, case["prepared"])/file)
                for key, file in (("input", "input.npz"), ("reference", "reference.npz"), ("config", "config.json"))}
                for case in plan["cases"]}
    runtime = get_environment("evaluators.video")
    if not runtime.exists:
        raise ValueError("Install the declared evaluator runtime before scoring")
    output.mkdir(parents=True)
    (output/"logs").mkdir()
    environment = {**os.environ, **runtime_cache_environment(root), "CUDA_VISIBLE_DEVICES": "",
                   "HF_HUB_OFFLINE": "1", "YOLO_CONFIG_DIR": str(root/".cache/ultralytics")}
    code_root = Path(__file__).resolve().parents[2]
    sources = [code_root/"evaluation/reporting"/f"{name}.py" for name in
               ("checkpoint_metrics", "motion", "motion_detector", "pusht_observer", "perceptual", "native_reference")]
    sources += [code_root/"evaluation/datasets"/f"{name}.py" for name in ("native_reference", "video")]
    sources.append(code_root/"evaluation/reporting/metric_store.py")
    fingerprints = {str(p): _digest(p) for p in sources}
    journal = {"schema_version": 1, "status": "running", "source_plan_sha256": _digest(plan_path),
        "replay_directories": [str(directory), *map(str, additional_directories)],
        "expected_cases": len(plan["cases"]), "jobs": {}, "excluded_attempts": [],
        "provider": provider, "evaluator_source_sha256": fingerprints, "case_input_sha256": expected,
        "metric_timeout_seconds": timeout_seconds, "decoder_timeout_seconds": 3600,
        "sidecar_layout": "version_local",
        "metric_gpus": list(metric_gpus), "frame_batch_size": frame_batch_size,
        "metric_implementation_sha256": {kind: metric_source_fingerprints(kind)
            for kind in (("lpips",) if provider == "none" else ("lpips", "motion"))},
        "motion_evaluation": "not_requested_no_calibrated_provider" if provider == "none" else "provider_declared_not_physical_validation",
        "vla_calls": 0, "world_model_calls": 0}
    progress = output/"progress.json"
    publish_json_exclusive(progress, journal)

    def score(name, pair):
        from evaluation.reporting.metric_store import admit_metric
        summary, reference = pair
        case_output = output/"cases"/name
        case_output.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for kind in (("lpips",) if provider == "none" else ("lpips", "motion")):
            destination = case_output/("perceptual.json" if kind == "lpips" else "motion/motion.json")
            command = [str(runtime.python), "-m", "evaluation.reporting."+("perceptual" if kind == "lpips" else "motion_detector"),
                       "--summary", str(summary), "--reference", str(reference), "--output",
                       str(destination if kind == "lpips" else destination.parent)]
            if kind == "motion":
                command += ["--provider", provider]
            gpu = gpu_queue.get() if metric_gpus and kind == "lpips" else None
            job_environment = dict(environment)
            if gpu is not None:
                job_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                command += ["--device", "cuda"]
            if kind == "lpips" and frame_batch_size != 1:
                command += ["--frame-batch-size", str(frame_batch_size)]
            try:
                with (output/"logs"/f"{name}_{kind}.log").open("x") as log:
                    subprocess.run(command, cwd=root, env=job_environment, check=True, timeout=timeout_seconds,
                                   stdout=log, stderr=subprocess.STDOUT)
            finally:
                if gpu is not None:
                    gpu_queue.put(gpu)
            artifacts[kind] = admit_metric(destination, summary_sha256=_digest(summary),
                reference_sha256=_digest(reference), implementation=journal["metric_implementation_sha256"][kind])
        return {"status": "completed", "summary": str(summary), "summary_sha256": _digest(summary), "metrics": artifacts}

    with scoring_failure_journal(progress, journal), asset_destination_lock(root, output), \
            (native_gpu_lease(root, list(metric_gpus)) if metric_gpus else nullcontext()), \
            ThreadPoolExecutor(max_workers=workers) as executor:
        if metric_gpus:
            resource_admission({"world_model_gpus": list(metric_gpus), "max_process_memory_gib_per_gpu": 8}, False)
        while True:
            if _digest(plan_path) != journal["source_plan_sha256"] or any(_digest(Path(p)) != h for p, h in fingerprints.items()):
                raise ValueError("Scoring source/plan changed during execution; preserve journal and start a new version")
            replays, journal["excluded_attempts"] = select_replays(root, directory, plan, expected=expected,
                additional_directories=additional_directories)
            ready = {name: pair for name, pair in replays.items() if name not in journal["jobs"]}
            futures = {executor.submit(score, name, pair): name for name, pair in ready.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    journal["jobs"][name] = future.result()
                except Exception as exc:
                    journal["jobs"][name] = {"status": "failed", "error": str(exc), "retry_requires_new_scoring_directory": True}
                replace_owned_json(progress, journal)
            completed = sum(j["status"] == "completed" for j in journal["jobs"].values())
            failed = any(j["status"] == "failed" for j in journal["jobs"].values())
            journal["pending_cases"] = sorted(set(case["id"] for case in plan["cases"])-set(journal["jobs"]))
            journal["status"] = "failed" if failed else "completed" if completed == len(plan["cases"]) else "waiting_for_replays"
            replace_owned_json(progress, journal)
            if failed or journal["status"] == "completed" or not watch:
                return journal
            time.sleep(15)
