"""Weight-free experiment planning; validation is not an execution claim."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
from typing import Any

from adapters.environment import load_environment_registry
from evaluation.manifests import load_pair_manifest, pair_for_job
from evaluation.modes import POLICY_MODES
from evaluation.planning.registry import load_runtime_registry
from evaluation.execution.reliability import resolve_worker_options
from evaluation.planning.configuration import ConfigurationError, invalid, load_suite

from vlaworldeval.paths import project_root

ROOT = project_root()


def _project_path(value: str, root: Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Experiment path leaves the project: {value}")
    return path


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _indices(values: Any, name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty integer list")
    if any(type(x) is not int or x < 0 for x in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique nonnegative integers")
    return values


def _build_experiment_plan(path: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    """Expand a suite and reject semantic/resource mismatches before GPU use.

    This deliberately does not launch workers. Dataset episode lengths and file
    coverage are live executor admission checks, not guessed here.
    """
    path = Path(path)
    path = path if path.is_absolute() else root / path
    config = load_suite(path)

    def checked(field, function, *args):
        try:
            return function(*args)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            invalid(path, field, str(exc))

    if config.get("schema_version") != 1:
        invalid(path, "schema_version", "Unsupported experiment schema_version")
    suite_id = config["suite_id"]
    if not suite_id or Path(suite_id).name != suite_id or suite_id in {".", ".."}:
        invalid(path, "suite_id", "Invalid suite_id")
    pair_path = checked("pair_manifest", _project_path, config["pair_manifest"], root)
    pair = checked("pair_manifest", load_pair_manifest, pair_path)
    definitions = checked("runtime_registry", load_runtime_registry, root)
    profile = checked("pair_manifest", definitions.pair_profile, pair)
    dataset = config["dataset"]
    if dataset.get("loader", profile["dataset"]) != profile["dataset"]:
        invalid(path, "dataset.loader", "Dataset loader does not match registered runtime profile")
    dataset_root = checked("dataset.root", _project_path, dataset["root"], root)
    if dataset["fps"] != profile.get("dataset_source_fps", pair.world_model.action.rate_hz):
        invalid(path, "dataset.fps", "Dataset FPS and world-model action rate do not match")
    if type(dataset["start_frame"]) is not int or dataset["start_frame"] < 0:
        invalid(path, "dataset.start_frame", "start_frame must be a nonnegative integer")
    if dataset["tail_policy"] != "drop_incomplete_chunk":
        invalid(path, "dataset.tail_policy", "Acceptance requires drop_incomplete_chunk, not action padding")
    runtime = config["runtime"]
    for name in ("worker_options", "vla_worker_options", "world_model_worker_options"):
        if name in runtime:
            checked(f"runtime.{name}", resolve_worker_options, {"worker_options": runtime[name]}, "vla")
    worker_options = {role: asdict(checked(f"runtime.{role}_worker_options", resolve_worker_options, runtime, role))
                      for role in ("vla", "world_model")}
    registry = load_environment_registry()
    for key in ("vla_environment", "world_model_environment"):
        if runtime[key] not in registry:
            invalid(path, f"runtime.{key}", f"Unregistered {key}: {runtime[key]}")
    from evaluation.planning.catalog import load_model_catalog
    catalog = load_model_catalog(root)
    for group, adapter, key in (
        ("vlas", pair.vla.adapter, "vla_environment"),
        ("world_models", pair.world_model.adapter, "world_model_environment"),
    ):
        expected = {entry["environment"] for entry in catalog[group].values()
                    if entry["adapter"] == adapter}
        if runtime[key] not in expected:
            invalid(path, f"runtime.{key}", f"{key} does not match pair adapter {adapter}")
        if runtime[key] != definitions.models[group][adapter]["environment"]:
            invalid(path, f"runtime.{key}", f"{key} does not match registered runtime environment for {adapter}")
    vla_gpus = checked("runtime.vla_gpus", _indices, runtime["vla_gpus"], "vla_gpus")
    wm_gpus = checked("runtime.world_model_gpus", _indices, runtime["world_model_gpus"], "world_model_gpus")
    for group, adapter, gpus in (("vlas", pair.vla.adapter, vla_gpus),
                                 ("world_models", pair.world_model.adapter, wm_gpus)):
        if not definitions.models[group][adapter]["distributed"] and len(gpus) != 1:
            invalid(path, f"runtime.{'vla' if group == 'vlas' else 'world_model'}_gpus",
                    f"{adapter} factory is single-GPU, not a distributed adapter")
    if runtime["execution"] != "persistent_isolated_workers":
        invalid(path, "runtime.execution", "This suite requires isolated persistent workers")
    if runtime["scheduling"] != "sequential_vla_world_model":
        invalid(path, "runtime.scheduling", "Only sequential scheduling is declared")
    memory = runtime["max_process_memory_gib_per_gpu"]
    if (not isinstance(memory, (int, float)) or isinstance(memory, bool)
            or not math.isfinite(memory) or memory <= 0):
        invalid(path, "runtime.max_process_memory_gib_per_gpu", "GPU process memory budget must be positive")
    checkpoint = checked("runtime.world_model_checkpoint", _project_path, runtime["world_model_checkpoint"], root)
    if "vla_memory_gib" in runtime:
        budget = runtime["vla_memory_gib"]
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            invalid(path, "runtime.vla_memory_gib", "vla_memory_gib must be finite and positive")
    sampling = config["sampling"]
    checked("sampling.num_inference_steps", _positive_int, sampling["num_inference_steps"], "num_inference_steps")
    for name, required in profile["sampling_constraints"].items():
        if type(sampling[name]) is not type(required) or sampling[name] != required:
            invalid(path, f"sampling.{name}", f"Runtime profile {profile['id']} requires {name}={required!r}")
    if type(sampling["diffusion_cache"]) is not bool:
        invalid(path, "sampling.diffusion_cache", "diffusion_cache must be boolean")
    for key in ("guidance", "shift"):
        if (not isinstance(sampling[key], (int, float)) or isinstance(sampling[key], bool)
                or not math.isfinite(sampling[key]) or sampling[key] <= 0):
            invalid(path, f"sampling.{key}", f"{key} must be positive")
    output_root = checked("output_root", _project_path, config["output_root"], root)
    jobs, ids = [], set()
    checked("dataset.episodes", _indices, dataset["episodes"], "episodes")
    checked("seeds", _indices, config["seeds"], "seeds")
    for index, experiment in enumerate(config["experiments"]):
        location = f"experiments[{index}]"
        identifier = experiment["id"]
        if identifier in ids or not identifier or not all(c.isalnum() or c in "_-" for c in identifier):
            invalid(path, f"{location}.id", f"Invalid or duplicate experiment id: {identifier}")
        ids.add(identifier)
        chunks = checked(f"{location}.chunks", _positive_int, experiment["chunks"], "chunks")
        mode, action, provider = (experiment[k] for k in ("mode", "action_source", "state_provider"))
        if mode == "wm_replay":
            if action not in profile["replay_actions"] or provider != "none":
                invalid(path, f"{location}.action_source/state_provider", "WM replay requires valid WM-native actions and no recurrent VLA state")
        elif mode in POLICY_MODES:
            if action != "vla" or provider not in profile["state_providers"]:
                invalid(path, f"{location}.action_source/state_provider", "Policy loop requires VLA actions and a declared supported state baseline")
            checked(f"{location}.mode/state_provider", pair_for_job, pair, experiment)
            if mode == "pilworld":
                if mode not in profile.get("evaluation_modes", ["policy_loop"]):
                    invalid(path, f"{location}.mode", "pilworld is not admitted for this runtime profile")
                from adapters.actions.base import build_action_bridge
                bridge = checked(f"{location}.action_bridge", build_action_bridge, pair.action_bridge)
                if not callable(getattr(bridge, "pilworld_state", None)):
                    invalid(path, f"{location}.action_bridge", "pilworld requires a terminal absolute target mapper")
        else:
            invalid(path, f"{location}.mode", f"Unsupported experiment mode: {mode}")
        episodes = checked(f"{location}.episodes", _indices, experiment.get("episodes", dataset["episodes"]), "episodes")
        seeds = checked(f"{location}.seeds", _indices, experiment.get("seeds", config["seeds"]), "seeds")
        for episode in episodes:
            for seed in seeds:
                jobs.append({
                    **experiment, "episode_index": episode, "seed": seed,
                    "action_count": chunks * pair.world_model.action.horizon,
                    "duration_seconds": chunks * pair.world_model.action.advance_steps / pair.world_model.action.rate_hz,
                    "generated_future_frames": chunks * (pair.world_model.video.advance_frames
                        if pair.world_model.video else pair.world_model.action.horizon),
                    "video_fps": pair.world_model.video.fps if pair.world_model.video else dataset["fps"],
                    "output": str(output_root / identifier / f"episode_{episode:06d}" / f"seed_{seed:03d}"),
                    "status": "planned_not_executed",
                })
    if not jobs:
        invalid(path, "experiments", "Experiment suite is empty")
    return {
        "suite_id": suite_id, "pair_id": pair.pair_id,
        "runtime_profile": profile["id"], "dataset_loader": profile["dataset"],
        "dataset_root": str(dataset_root), "dataset_start_frame": dataset["start_frame"],
        "runtime": runtime, "sampling": sampling, "jobs": jobs,
        "worker_options": worker_options,
        "local_assets_present": {
            "dataset_root": dataset_root.is_dir(), "checkpoint": checkpoint.exists(),
            "vla_python": registry[runtime["vla_environment"]].python.is_file()
                          and os.access(registry[runtime["vla_environment"]].python, os.X_OK),
            "world_model_python": registry[runtime["world_model_environment"]].python.is_file()
                                  and os.access(registry[runtime["world_model_environment"]].python, os.X_OK),
            **definitions.asset_status(root, pair, runtime, sampling),
        },
        "execution_ready": False,
        "executor_available": True,
        "pending_admission_checks": [
            "Per-episode video/data coverage, full chunks, synchronization and instruction",
            "Live GPU resource admission (resident budgets add on overlapping devices)",
            "Calibration split isolation must be audited before publication experiments",
        ],
        "gpu_overlap": sorted(set(vla_gpus) & set(wm_gpus)),
        "resource_note": "Sequential calls do not release resident model weights; overlapping GPU budgets add up.",
    }


def build_experiment_plan(path: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    path = Path(path)
    path = path if path.is_absolute() else root / path
    try:
        return _build_experiment_plan(path, root=root)
    except ConfigurationError:
        raise
    except (ValueError, KeyError, TypeError, OSError) as exc:
        invalid(path, "dependencies", str(exc))


__all__ = ["build_experiment_plan"]
