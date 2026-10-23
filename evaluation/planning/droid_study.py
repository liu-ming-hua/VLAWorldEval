"""Freeze a DROID experiment set before seeing generated outcomes."""
from copy import deepcopy
import json
import math
import re
import subprocess

from evaluation.datasets.droid import CAMERAS
from evaluation.manifests import load_pair_manifest
from evaluation.planning.configuration import invalid, load_suite
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive
from vlaworldeval.paths import project_root


def load_study_spec(path):
    """Validate public input before optional dependencies or output creation."""
    try:
        spec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        invalid(path, "$", str(exc))
    fields = {"schema_version", "study_id", "dataset", "episodes", "start_frame",
              "comparison_source_ticks", "fps", "seed", "policies", "world_models",
              "allowed_gpus", "memory_gib", "state_provider", "quality_gate", "train_overlap"}
    if not isinstance(spec, dict):
        invalid(path, "$", "Expected an object")
    for key in sorted(fields - spec.keys()):
        invalid(path, key, "Required field is missing")
    for key in sorted(spec.keys() - fields):
        invalid(path, key, "Unknown field")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        invalid(path, "schema_version", "Expected 1")
    if not isinstance(spec["study_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", spec["study_id"]):
        invalid(path, "study_id", "Use 1–128 letters, digits, underscores or hyphens; begin with a letter or digit")
    if not isinstance(spec["dataset"], str) or not spec["dataset"].strip():
        invalid(path, "dataset", "Expected a nonempty project-local path")
    for key, minimum in (("start_frame", 0), ("comparison_source_ticks", 1)):
        if type(spec[key]) is not int or spec[key] < minimum:
            invalid(path, key, f"Expected an integer >= {minimum}")
    if spec["comparison_source_ticks"] % 240:
        invalid(path, "comparison_source_ticks", "Require complete 240-tick blocks")
    for key, expected in (("fps", 15), ("seed", 0)):
        if type(spec[key]) is not int or spec[key] != expected:
            invalid(path, key, f"This protocol requires integer {expected}")
    for key, allowed in (("policies", {"pi0", "pi05", "pi0_fast_droid", "groot_n17_droid"}),
                         ("world_models", {"ctrl", "cosmos"}), ("allowed_gpus", {0, 1, 2, 3})):
        values = spec[key]
        expected = int if key == "allowed_gpus" else str
        if (not isinstance(values, list) or not values or any(type(v) is not expected or v not in allowed for v in values)
                or len(set(values)) != len(values)):
            invalid(path, key, f"Require a nonempty unique list from {sorted(allowed)}")
    if "cosmos" in spec["world_models"] and set(spec["allowed_gpus"]) != {0, 1, 2, 3}:
        invalid(path, "allowed_gpus", "Cosmos requires all four GPUs0–3; this queue cannot admit it with fewer")
    episodes = spec["episodes"]
    if (not isinstance(episodes, list) or not 5 <= len(episodes) <= 10
            or any(type(e) is not int or e < 0 for e in episodes) or len(set(episodes)) != len(episodes)):
        invalid(path, "episodes", "Require 5–10 unique nonnegative integer episodes")
    memory = spec["memory_gib"]
    if not isinstance(memory, dict) or set(memory) != {"ctrl", "cosmos", "vla"}:
        invalid(path, "memory_gib", "Require ctrl, cosmos and vla budgets")
    for key, value in memory.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            invalid(path, f"memory_gib.{key}", "Require a finite positive number")
    for key, expected in (("state_provider", "integrated_action_target"),
                          ("quality_gate", "evidence_bound_manual_review_required"), ("train_overlap", "unknown")):
        if spec[key] != expected:
            invalid(path, key, f"This protocol requires {expected}")
    return spec


def study_suite(template, *, study_id, episode, length, start, ticks, horizon, policy, wm):
    if any(type(v) is not int or v < 1 for v in (length, ticks, horizon)):
        raise ValueError("Lengths and horizons must be positive integers")
    if type(start) is not int or start < 0 or type(episode) is not int or episode < 0:
        raise ValueError("Invalid episode/start")
    if ticks % horizon or start + ticks >= length:
        raise ValueError("Matched horizon needs complete native chunks and a terminal observation")
    suite = deepcopy(template)
    suite["suite_id"] = f"{study_id}_{policy}_{wm}_ep{episode:06d}"
    suite["dataset"].update(episodes=[episode], start_frame=start)
    suite["seeds"] = [0]
    suite["output_root"] = f"results/experiments/{suite['suite_id']}"
    suite["experiments"] = [
        {"id": "real_action_smoke", "mode": "wm_replay", "action_source": "dataset_pose_derived",
         "state_provider": "none", "chunks": 1},
        {"id": "real_action_full", "mode": "wm_replay", "action_source": "dataset_pose_derived",
         "state_provider": "none", "chunks": (length-start-1)//horizon},
        {"id": "real_action_matched", "mode": "wm_replay", "action_source": "dataset_pose_derived",
         "state_provider": "none", "chunks": ticks//horizon},
        {"id": "policy_matched", "mode": "policy_loop", "action_source": "vla",
         "state_provider": "integrated_action_target", "chunks": ticks//horizon},
    ]
    suite["metadata"] = {"purpose": "droid_positive_control_gated_comparison",
        "task_success": None, "train_overlap": "unknown", "selection_uses_generated_outcomes": False,
        "comparison_source_ticks": ticks, "source_horizon": horizon,
        "comparison_end_frame": start+ticks, "full_end_frame": start+((length-start-1)//horizon)*horizon,
        "discarded_full_tail_ticks": (length-start-1)%horizon,
        "recurrent_state_source": "action_target_proxy_not_measured_or_wm_predicted"}
    return suite


def prepare_study(config_path, output):
    root = project_root()
    config_path, output = project_path(root, str(config_path)), project_path(root, str(output))
    if output.exists():
        raise FileExistsError(output)
    spec = load_study_spec(config_path)
    dataset = project_path(root, spec["dataset"])
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("DROID preparation requires Parquet support; run CORE_DROID=1 bash scripts/setup/core.sh") from exc
    cases = {}
    for path in sorted((dataset / "meta/episodes").glob("chunk-*/file-*.parquet")):
        rows = pq.read_table(path, filters=[("episode_index", "in", spec["episodes"])]).to_pylist()
        for row in rows:
            if row["episode_index"] in cases:
                raise ValueError("Duplicate episode metadata")
            cases[row["episode_index"]] = row
    if set(cases) != set(spec["episodes"]):
        raise ValueError("Missing local episode metadata")
    # Reject bad windows/missing views before leaving a partially frozen directory.
    for episode, row in cases.items():
        if type(row.get("length")) is not int or row["length"] <= spec["start_frame"] + spec["comparison_source_ticks"]:
            invalid(config_path, f"episodes[{episode}]", "Insufficient frames for the matched window and terminal observation")
        if not isinstance(row.get("tasks"), list) or not row["tasks"] or not isinstance(row["tasks"][0], str):
            invalid(config_path, f"episodes[{episode}]", "Episode requires a task instruction")
        for key in CAMERAS.values():
            prefix = f"videos/{key}"
            for field in ("chunk_index", "file_index", "from_timestamp"):
                value = row.get(f"{prefix}/{field}")
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    invalid(config_path, f"episodes[{episode}].{prefix}/{field}", "Missing or invalid video metadata")
                if field != "from_timestamp" and type(value) is not int:
                    invalid(config_path, f"episodes[{episode}].{prefix}/{field}", "Expected integer video index")
            video = dataset / f"{prefix}/chunk-{row[prefix+'/chunk_index']:03d}/file-{row[prefix+'/file_index']:03d}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
    entries, references = [], []
    output.mkdir(parents=True)
    for episode in spec["episodes"]:
        row = cases[episode]
        views = {}
        for name, key in CAMERAS.items():
            prefix = f"videos/{key}"
            video = dataset / f"{prefix}/chunk-{row[prefix+'/chunk_index']:03d}/file-{row[prefix+'/file_index']:03d}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            target = output / "references" / f"ep{episode:06d}_{name}.mp4"
            target.parent.mkdir(exist_ok=True)
            # Transcoding uses accurate seek; references are never model outputs.
            subprocess.run(["ffmpeg", "-v", "error", "-ss",
                str(row[prefix+"/from_timestamp"]+spec["start_frame"]/15), "-i", str(video), "-frames:v",
                str(row["length"]-spec["start_frame"]), "-c:v", "libx264", "-threads", "2",
                "-crf", "20", str(target)], check=True)
            views[name] = {"file": str(target.relative_to(root)), "sha256": _digest(target),
                "source_video": str(video.relative_to(root)), "source_offset_seconds": row[prefix+"/from_timestamp"]}
        references.append({"episode": episode, "length": row["length"], "instruction": row["tasks"][0].split(" | ")[0],
                           "views": views, "reference_task_success": None, "review": "pending"})
        for wm in spec["world_models"]:
            for policy in spec["policies"]:
                # Resource/sampling templates only; native pair identity is explicit.
                template = load_suite(root / f"configs/experiments/pi0_fast_droid_{wm}_v1.json")
                pair_name = {"pi0": "pi0", "pi05": "pi05", "pi0_fast_droid": "pi0_fast_droid",
                             "groot_n17_droid": "groot_n17_droid"}[policy]
                suffix = "ctrl_world" if wm == "ctrl" else "cosmos3_droid"
                template["pair_manifest"] = f"configs/pairs/{pair_name}__{suffix}.json"
                manifest = load_pair_manifest(root / template["pair_manifest"])
                horizon = 12 if wm == "ctrl" else 16
                suite = study_suite(template, study_id=spec["study_id"], episode=episode,
                    length=row["length"], start=spec["start_frame"], ticks=spec["comparison_source_ticks"],
                    horizon=horizon, policy=policy, wm=wm)
                if manifest.world_model.action.horizon != (5 if wm == "ctrl" else 16):
                    raise ValueError("Unexpected pair temporal protocol")
                suite["runtime"].update(vla_environment=
                    ("vla.groot_n17" if policy == "groot_n17_droid" else "vla.openpi"),
                    vla_gpus=[0], world_model_gpus=[0] if wm == "ctrl" else [0,1,2,3],
                    vla_memory_gib=spec["memory_gib"]["vla"],
                    max_process_memory_gib_per_gpu=spec["memory_gib"][wm])
                path = output / "suites" / f"{suite['suite_id']}.json"
                path.parent.mkdir(exist_ok=True)
                publish_json_exclusive(path, suite)
                entries.append({"id": suite["suite_id"], "episode": episode, "policy": policy, "world_model": wm,
                    "config": str(path.relative_to(root)), "sha256": _digest(path),
                    "control_owner": policy == spec["policies"][0]})
    result = {"schema_version": 1, "spec": spec, "spec_sha256": _digest(config_path),
              "status": "frozen_not_executed", "references": references, "entries": entries}
    publish_json_exclusive(output / "study.json", result)
    return {"output": str(output), "cases": len(cases), "policy_suites": len(entries),
            "positive_control_suites": sum(e["control_owner"] for e in entries), "models_executed": False}
