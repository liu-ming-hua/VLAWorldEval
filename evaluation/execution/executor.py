"""Unified suite executor: admission, persistent workers, journals and artifacts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from adapters.environment import get_environment
from evaluation.reporting.artifacts import save_rollout_artifacts
from evaluation.evaluator import PolicyWorldModelEvaluator, WorldModelReplayEvaluator
from evaluation.planning.experiments import ROOT, build_experiment_plan
from evaluation.manifests import load_pair_manifest, pair_for_job
from evaluation.modes import POLICY_MODES
from evaluation.execution.client import ModelWorker, RemoteVLA, RemoteWorldModel
from evaluation.execution.transport import write_packet
from evaluation.factories import resolve_workers
from evaluation.planning.registry import load_runtime_registry, project_path
from evaluation.execution.reliability import WorkerError, clean_worker_exit, exception_details, resolve_worker_options
from evaluation.planning.configuration import load_suite
from evaluation.execution.admission import admit_episodes, resource_admission, select_jobs


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def fingerprint(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git_checkout_evidence(source):
    """Provenance must not require an unrelated original checkout to exist.

    Git-less exports and vendored runtimes still have framework file hashes.
    Missing Git identity is recorded explicitly, never invented or fatal.
    """
    record = {"path": str(source), "head": None, "dirty": None}
    if not source.is_dir() or not (source / ".git").exists():
        return {**record, "status": "not_a_git_checkout"}
    try:
        record["head"] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        record["dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL, timeout=10).strip())
    except (OSError, subprocess.SubprocessError):
        return {**record, "status": "git_probe_failed"}
    return {**record, "status": "inspected"}


def build_replay_actions(chunks, source, seed, *, transform=None):
    if transform is None:
        transform = {"dataset_pose_derived": "identity", "official_worldgym_actions": "identity",
                     "shuffle_valid_dataset_actions": "shuffle_rows",
                     "zero_translation_identity_rotation_preserve_gripper": "zero_xyz_identity_rot6d"}.get(source)
    values = np.concatenate(chunks).copy()
    if transform == "shuffle_rows":
        values = values[np.random.default_rng(seed).permutation(len(values))]
    elif transform == "zero_xyz_identity_rot6d":
        if values.shape[1] != 10:
            raise ValueError("zero_xyz_identity_rot6d requires 10-D actions")
        values[:, :3] = 0
        values[:, 3:9] = [1, 0, 0, 0, 1, 0]
    elif transform != "identity":
        raise ValueError(source)
    return [part.copy() for part in np.split(values, len(chunks))]


def execute_suite(config_path, *, experiments=(), seeds=(), admit_only=False, on_attempt=None):
    config_path = Path(config_path).resolve()
    config = load_suite(config_path)
    plan = build_experiment_plan(config_path)
    jobs = select_jobs(plan, experiments, seeds)
    if not all(plan["local_assets_present"].values()):
        raise FileNotFoundError(f"Missing suite assets: {plan['local_assets_present']}")
    base_manifest = load_pair_manifest(ROOT / config["pair_manifest"])
    if base_manifest.world_model.adapter == "WorldGymAdapter":
        from vlaworldeval.worldgym_runtime import verify_runtime
        verify_runtime(ROOT)
    if not plan["executor_available"]:
        raise ValueError("No runtime factory for this pair")
    registry = load_runtime_registry(ROOT)
    profile = registry.pair_profile(base_manifest)
    # Admit every selected data window before any GPU worker is started.
    episodes = admit_episodes(registry, profile, Path(plan["dataset_root"]), config["dataset"],
                              jobs, base_manifest.world_model.action.horizon)
    needs_vla = any(job["mode"] in POLICY_MODES for job in jobs)
    resources = resource_admission(plan["runtime"], needs_vla)
    if admit_only:
        return {"status": "admitted_not_executed", "jobs": len(jobs), "resources": resources,
                "datasets": [value.provenance for value in episodes.values()]}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    suite_dir = ROOT / config["output_root"] / f"attempt_{attempt}_{digest}"
    suite_dir.mkdir(parents=True, exist_ok=False)
    write_json(suite_dir / "resolved_suite.json", config)
    write_json(suite_dir / "selection.json", {"experiments": list(experiments), "seeds": list(seeds),
                                              "jobs": jobs})
    write_json(suite_dir / "admission.json", resources)
    runtime, sampling = plan["runtime"], plan["sampling"]
    checkpoint = ROOT / runtime["world_model_checkpoint"]
    vla_spec, wm_spec = resolve_workers(ROOT, base_manifest, runtime, sampling, suite_dir)
    write_json(suite_dir / "resolved_workers.json", {"vla": vars(vla_spec), "world_model": vars(wm_spec),
                                                   "runtime": runtime, "runtime_profile": profile["id"],
                                                   "worker_options": plan["worker_options"]})
    provenance = {"suite_config": fingerprint(config_path), "pair": fingerprint(ROOT / config["pair_manifest"]),
                  "runtime_registry": fingerprint(ROOT / "configs/runtime_registry.json"),
                  "runtime_profile": profile["id"]}
    provenance.update(wm_checkpoint={"path": str(checkpoint), "size": checkpoint.stat().st_size,
                                    "mtime_ns": checkpoint.stat().st_mtime_ns},
                      sources={}, worker_specs={"vla": vars(vla_spec), "world_model": vars(wm_spec)})
    declarations = profile.get("provenance", {})
    for name, path in declarations.get("files", {}).items():
        provenance[name] = fingerprint(project_path(ROOT, path))
    for name, path in declarations.get("wm_files", {}).items():
        provenance[name] = fingerprint(project_path(ROOT, str(checkpoint / path)))
    for name in declarations.get("action_bridge_files", []):
        provenance[name] = fingerprint(project_path(ROOT, base_manifest.action_bridge.config[name]))
    for name in declarations.get("git_sources", []):
        source = ROOT / "third_party" / name
        evidence = git_checkout_evidence(source)
        provenance["sources"][name] = evidence["head"]
        provenance.setdefault("source_checkouts", {})[name] = evidence
    # Hash imported code rather than a potentially older asset checkout.
    code_root = Path(__file__).resolve().parents[2]
    source_files = [str(path.relative_to(code_root)) for package in
                    ("adapters", "cli", "evaluation", "vlaworldeval")
                    for path in (code_root / package).rglob("*.py")]
    source_files += [vla_spec.module.replace(".", "/") + ".py", wm_spec.module.replace(".", "/") + ".py"]
    source_files += declarations.get("source_files", [])
    provenance["framework_code_root"] = str(code_root)
    provenance["framework_sources"] = [fingerprint(code_root / path) for path in sorted(set(source_files))]
    write_json(suite_dir / "provenance.json", provenance)
    workers, summary = [], {"status": "running", "jobs": [], "attempt": str(suite_dir)}
    write_json(suite_dir / "summary.json", summary)
    wm_worker = vla_worker = None
    try:
        if on_attempt:
            on_attempt(suite_dir)
        vla_spec, wm_spec = resolve_workers(ROOT, base_manifest, runtime, sampling, suite_dir)
        if needs_vla:
            vla_worker = ModelWorker(get_environment(runtime["vla_environment"]).python, suite_dir / "workers/vla",
                                    module=vla_spec.module, adapter=vla_spec.adapter,
                                    kwargs=vla_spec.kwargs, gpus=runtime["vla_gpus"],
                                    memory_gib=resources["vla_process_gib"], options=resolve_worker_options(runtime, "vla"))
            workers.append(vla_worker)
        wm_worker = ModelWorker(get_environment(runtime["world_model_environment"]).python, suite_dir / "workers/world_model",
                                module=wm_spec.module, adapter=wm_spec.adapter,
                                kwargs=wm_spec.kwargs,
                                gpus=runtime["world_model_gpus"], memory_gib=resources["world_model_process_gib"],
                                options=resolve_worker_options(runtime, "world_model"))
        workers.append(wm_worker)
        for job in jobs:
            out = suite_dir / job["id"] / f"episode_{job['episode_index']:06d}" / f"seed_{job['seed']:03d}"
            out.mkdir(parents=True, exist_ok=False)
            write_json(out / "job.json", job)
            dataset = episodes[(job["episode_index"], job["chunks"], job["mode"])]
            write_json(out / "dataset.json", dataset.provenance)
            manifest = pair_for_job(base_manifest, job)
            write_json(out / "pair_manifest.json", manifest.to_dict())
            record = {"id": job["id"], "episode": job["episode_index"], "seed": job["seed"],
                      "output": str(out), "status": "running", "completed_chunks": 0}
            summary["jobs"].append(record)
            started = time.perf_counter()
            trace = None
            journal = out / "partial"
            journal.mkdir()

            def checkpoint_step(current):
                nonlocal trace
                trace = current
                index = len(current.bridged_actions) - 1
                state = current.observations[-1].robot_state
                write_packet(journal / f"chunk_{index:05d}.npz", {
                    "views": current.world_outputs[-1].views,
                    "action": current.bridged_actions[-1].values,
                    "native_action": current.native_actions[-1].values if current.native_actions else None,
                    "state": state.values if state else None, "state_source": state.source if state else None,
                    "timing": current.timings[-1],
                })
                input_views = current.observations[-2].views
                write_json(journal / f"chunk_{index:05d}.json", {
                    "observation_input_source": "initial_dataset" if index == 0 else "previous_generated_terminal_views",
                    "vla_input_source": ("initial_dataset" if index == 0 else "previous_generated_terminal_views")
                                        if job["mode"] in POLICY_MODES else None,
                    "evaluation_mode": job["mode"],
                    "state_update_metadata": state.metadata if state is not None and job["mode"] == "pilworld" else None,
                    "input_view_hashes": {key: hashlib.sha256(value[-1:].tobytes()).hexdigest() for key, value in input_views.items()},
                    "input_history_hashes": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in input_views.items()},
                    "output_history_hashes": {key: hashlib.sha256(value.tobytes()).hexdigest()
                                              for key, value in current.world_outputs[-1].views.items()},
                    "output_observation_history_hashes": {
                        key: hashlib.sha256(value.tobytes()).hexdigest()
                        for key, value in current.observations[-1].views.items()},
                    "output_terminal_view_hashes": {
                        key: hashlib.sha256(value[-1].tobytes()).hexdigest()
                        for key, value in current.world_outputs[-1].views.items()
                    },
                    "wm_memory": wm_worker.memory, "vla_memory": vla_worker.memory if vla_worker else None,
                    "wm_action_source": job["action_source"],
                    "action_bridge_metadata": current.bridged_actions[-1].metadata,
                    "reference_frame_range": [config['dataset']['start_frame'] + index * dataset.provenance.get("source_frames_per_chunk", manifest.world_model.action.horizon),
                                           config['dataset']['start_frame'] + (index + 1) * dataset.provenance.get("source_frames_per_chunk", manifest.world_model.action.horizon)],
                })
                record["completed_chunks"] = index + 1
                record["status"] = "running"
                write_json(out / "status.json", record)
                write_json(suite_dir / "summary.json", summary)
                print(f"{job['id']} seed={job['seed']}: chunk {index + 1}/{job['chunks']}", file=sys.stderr, flush=True)

            try:
                wm_worker.call("seed", seed=job["seed"])
                wm = RemoteWorldModel(wm_worker)
                if job["mode"] in POLICY_MODES:
                    vla_worker.call("seed", seed=job["seed"])
                    evaluator = PolicyWorldModelEvaluator(RemoteVLA(vla_worker), wm, pair_manifest=manifest)
                    trace = evaluator.rollout_detailed(dataset.trial, job["chunks"], on_step=checkpoint_step)
                else:
                    actions = build_replay_actions(dataset.replay_actions, job["action_source"], job["seed"],
                                                   transform=profile["replay_transforms"][job["action_source"]])
                    trace = WorldModelReplayEvaluator(wm, manifest).rollout_detailed(
                        dataset.trial, actions, on_step=checkpoint_step, action_source=job["action_source"])
                save_trace(out, trace, job, dataset.provenance, partial=False)
                record.update(status="completed", seconds=time.perf_counter() - started,
                              vla_calls=len(trace.native_actions), world_model_calls=len(trace.bridged_actions))
                # Standard artifacts now contain these arrays; retain lightweight step audit JSON only.
                for path in journal.glob("chunk_*.npz"):
                    path.unlink()
                # The decoder has consumed these redundant per-chunk clips.
                for path in (suite_dir / "workers/cosmos_runtime").rglob("*.mp4"):
                    path.unlink()
            except BaseException as exc:
                record.update(status="failed", error=traceback.format_exc(), error_info=exception_details(exc),
                              seconds=time.perf_counter() - started)
                if trace is not None and trace.bridged_actions:
                    try:
                        save_trace(out, trace, job, dataset.provenance, partial=True)
                    except Exception:
                        record["artifact_error"] = traceback.format_exc()
                raise
            finally:
                write_json(out / "status.json", record)
                write_json(suite_dir / "summary.json", summary)
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="failed", error=traceback.format_exc(), error_info=exception_details(exc))
        raise
    finally:
        for worker in reversed(workers):
            worker.close()
        summary["worker_loads"] = [worker.ready for worker in workers]
        # Include startup-failed workers, which never entered the loaded list.
        summary["worker_cleanup"] = []
        for path in sorted((suite_dir / "workers").glob("*/lifecycle.json")):
            try:
                lifecycle = json.loads(path.read_text())
                summary["worker_cleanup"].append({"directory": str(path.parent), **lifecycle})
            except (OSError, ValueError) as exc:
                summary["worker_cleanup"].append({"directory": str(path.parent), "read_error": str(exc)})
        cleanup_failed = not summary["worker_cleanup"] or any(
            item.get("read_error") or item.get("status") != "closed"
            or not clean_worker_exit(item.get("cleanup")) for item in summary["worker_cleanup"])
        cleanup_error = None
        if cleanup_failed and summary["status"] == "completed":
            cleanup_error = WorkerError("cleanup_failed", "Workers did not cleanly shut down; inspect worker_cleanup",
                                        phase="cleanup", directory=suite_dir)
            summary.update(status="failed", error_info=exception_details(cleanup_error), error=str(cleanup_error))
        write_json(suite_dir / "summary.json", summary)
        print(f"Suite artifacts: {suite_dir}", file=sys.stderr, flush=True)
        if cleanup_error:
            raise cleanup_error
    return summary


def save_trace(out, trace, job, dataset, *, partial):
    save_rollout_artifacts(out, trace.primary_video(), [a.values for a in trace.bridged_actions],
                          {"instruction": dataset["instruction"], "experiment": job["id"], "mode": job["mode"],
                           "state_provider_applicability": "used" if job["mode"] in POLICY_MODES else "not_used_in_wm_replay",
                           "seed": job["seed"], "partial": partial}, fps=(trace.manifest.world_model.video.fps
                               if trace.manifest.world_model.video else dataset["fps"]),
                          native_action_chunks=[a.values for a in trace.native_actions],
                          states=[o.robot_state for o in trace.observations], view_videos=trace.view_videos(),
                          pair_manifest=trace.manifest.to_dict(), timings=trace.timings,
                          action_bridge_metadata=[a.metadata for a in trace.bridged_actions])
    from evaluation.reporting.video_diagnostics import diagnose_artifact
    write_json(out / "video_diagnostics.json", diagnose_artifact(out))
