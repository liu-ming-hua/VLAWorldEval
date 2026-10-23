"""Read-only preflight reports; never imports/starts a model or downloads assets."""
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone
import hashlib

from adapters.environment import load_environment_registry
from evaluation.execution.admission import admit_episodes, resource_admission, select_jobs
from evaluation.planning.configuration import ConfigurationError, load_suite
from evaluation.planning.experiments import ROOT, build_experiment_plan
from evaluation.manifests import load_pair_manifest
from evaluation.planning.registry import load_runtime_registry


def diagnose_suite(path, *, experiments=(), seeds=(), check_data=False, check_gpu=False):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    checks = []
    report = {"config": str(path.resolve()), "checks": checks, "model_loading_tested": False,
              "observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "execution_ready": False, "requested_checks": {"data": check_data, "gpu": check_gpu},
              "notes": ["Passing diagnostics is not model inference or task success.",
                        "Package compatibility is not tested; environment checks only inspect executable paths.",
                        "GPU admission is a snapshot, not a reservation; run rechecks resources."]}

    def add(name, status, detail, **extra):
        checks.append({"name": name, "status": status, "detail": detail, **extra})

    def finish():
        report["ok"] = not any(check["status"] == "failed" for check in checks)
        report["status"] = "checks_passed" if report["ok"] else "failed"
        report["coverage"] = {name: next((item["status"] for item in checks if item["name"] == name), "skipped")
                              for name in ("configuration", "data", "gpu")}
        return report

    try:
        plan = build_experiment_plan(path)
        config = load_suite(path)
        report["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        jobs = select_jobs(plan, experiments, seeds)
    except (ConfigurationError, ValueError, OSError) as exc:
        add("configuration", "failed", str(exc), error=getattr(exc, "details", None),
            hint="Correct the reported file/field or requested experiment/seed before running.")
        add("data", "skipped", "A valid selected suite is required")
        add("gpu", "skipped", "A valid selected suite is required")
        return finish()
    add("configuration", "passed", f"{plan['pair_id']}: {len(jobs)} selected jobs")
    report["selection"] = [{key: job[key] for key in ("id", "episode_index", "seed", "chunks")} for job in jobs]
    report["runtime_profile"] = plan["runtime_profile"]
    report["worker_options"] = plan["worker_options"]
    pair = load_pair_manifest(ROOT / config["pair_manifest"])
    if pair.world_model.adapter == "WorldGymAdapter":
        from vlaworldeval.worldgym_runtime import verify_runtime
        try:
            evidence = verify_runtime(ROOT)
            add("runtime.worldgym", "passed", f"Pinned runtime verified: {evidence['revision']}")
        except (ValueError, OSError) as exc:
            add("runtime.worldgym", "failed", str(exc),
                hint="Run bash scripts/setup/worldgym_runtime.sh; existing mismatched code is never overwritten.")
    registry = load_runtime_registry(ROOT)
    locations = registry.asset_paths(ROOT, pair, config["runtime"], config["sampling"])
    locations.update(dataset_root=plan["dataset_root"], checkpoint=str(ROOT / config["runtime"]["world_model_checkpoint"]))
    report["asset_paths"] = locations
    for name, present in plan["local_assets_present"].items():
        location = locations.get(name)
        if location is None:
            matching = sorted((key for key in locations if name.startswith(key + ".")), key=len, reverse=True)
            location = locations[matching[0]] if matching else None
        detail = "Present" if present else "Missing or incomplete"
        add(f"asset.{name}", "passed" if present else "failed", f"{detail}: {location}" if location else detail,
            hint=None if present else "Check suite paths, environment/registry.toml and runtime_registry.json assets; doctor does not install/download.")
    environments = load_environment_registry()
    for role in ("vla", "world_model"):
        python = environments[config["runtime"][f"{role}_environment"]].python
        executable = python.is_file() and os.access(python, os.X_OK)
        add(f"environment.{role}", "passed" if executable else "failed", str(python),
            hint="Executable path only; no Python/model/package probe was launched.")
    for tool in ("ffmpeg", "ffprobe"):
        executable = shutil.which(tool)
        add(f"tool.{tool}", "passed" if executable else "failed", executable or "Not found on PATH")
    if check_data:
        try:
            episodes = admit_episodes(registry, registry.pair_profile(pair), Path(plan["dataset_root"]),
                                       config["dataset"], jobs, pair.world_model.action.horizon)
            report["datasets"] = [packet.provenance for packet in episodes.values()]
            add("data", "passed", f"{len(episodes)} distinct episode/chunk/mode windows admitted")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            add("data", "failed", str(exc), hint="Check video/action coverage, FPS, instruction and complete native chunks.")
    else:
        add("data", "skipped", "Use --data to admit selected episode windows")
    if check_gpu:
        try:
            from evaluation.modes import POLICY_MODES
            report["resources"] = resource_admission(config["runtime"], any(job["mode"] in POLICY_MODES for job in jobs))
            add("gpu", "passed", "Live free-memory budgets admitted; no worker launched")
        except RuntimeError as exc:
            add("gpu", "failed", str(exc), hint="Use available GPU indices/budgets; do not terminate unrelated jobs.")
    else:
        add("gpu", "skipped", "Use --gpu to query live NVIDIA free-memory budgets")
    return finish()


def format_diagnostics(report):
    lines = [f"VLAWorldEval doctor: {report['status']}", f"Config: {report['config']}"]
    for check in report["checks"]:
        lines.append(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        if check["status"] == "failed" and check.get("hint"):
            lines.append(f"  Hint: {check['hint']}")
    lines.append("No model loaded; no installation/download performed. Passing is not inference acceptance.")
    return "\n".join(lines)
