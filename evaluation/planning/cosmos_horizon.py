"""Freeze comparable SOURCE-TIME windows for two native Cosmos protocols."""
import json
from pathlib import Path

import numpy as np

from adapters.observations.droid import compose_droid_cosmos_view
from evaluation.datasets.droid import load_droid_episode
from evaluation.execution.transport import read_packet, write_packet
from evaluation.reporting.task_scores import _digest
from vlaworldeval.paths import fresh_output_path


def native_chunks(source_ticks, stride):
    if type(stride) is not int or stride <= 0 or type(source_ticks) is not int or source_ticks <= 0 or source_ticks % stride:
        raise ValueError("Source horizon must consist of complete native chunks; no padding")
    return source_ticks // stride


def prepare_cosmos_horizon(root, output):
    root = Path(root).resolve()
    output = fresh_output_path(root, output)
    plan_path = root / "configs/studies/cosmos_horizon_v1.json"
    plan = json.loads(plan_path.read_text())
    expected = {"episodes": [0, 351, 365, 795, 903], "new_seeds": [1, 2],
        "horizon_source_ticks": [32, 64, 128], "models": ["eef", "joint"],
        "allowed_gpus": [0, 1, 2, 3], "fps": 15, "start_frame": 15}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("This version requires its declared fixed cases, seeds, source-time windows and GPUs")
    maximum = max(plan["horizon_source_ticks"])
    old = json.loads((root / plan["reference_study"] / "summary.json").read_text())
    env = root / plan["environment"]
    if not (env / "bin/python").is_file():
        raise ValueError("Historical verified environment is required; no automatic installation")
    prepared, baselines = [], []
    for episode in plan["episodes"]:
        # EEF loader validates real source length and all independent cameras.
        data = load_droid_episode(root / "datasets/cosmos3_droid/success", episode,
            plan["start_frame"], native_chunks(maximum, 16))
        views = data.trial["observation_bundle"]["views"]
        image = compose_droid_cosmos_view(views["droid.exterior_1"], views["droid.exterior_2"], views["droid.wrist"])[0]
        detail = next(d for d in old["details"] if d["episode"] == episode and d["world_model"] == "cosmos")
        full = Path(next(j["output"] for j in detail["jobs"] if j["id"] == "real_action_full"))
        worker_path = full.parents[2] / "workers/world_model"
        worker = json.loads((worker_path / "worker.json").read_text())
        ready = read_packet(worker_path / "ready.npz")
        pair = json.loads((full / "pair_manifest.json").read_text())["world_model"]
        kwargs = dict(worker["kwargs"])
        kwargs["runtime_dir"] = {"suite_path": "runtime"}
        for name, value in list(kwargs.items()):
            if isinstance(value, str) and value.startswith(str(root) + "/"):
                kwargs[name] = {"project_path": str(Path(value).relative_to(root))}
        eef_config = {"schema_version": 1, "environment": "world_models.cosmos3_nano_droid_fd",
            "module": worker["module"], "adapter": worker["adapter"], "kwargs": kwargs,
            "gpus": plan["allowed_gpus"], "memory_gib": worker["memory_gib"], "seed": 0,
            "action": pair["action"], "video": pair["video"], "output_views": pair["output_views"],
            "reset_operation": "reset", "instruction_conditioning": False,
            "worker_options": {"startup_timeout_seconds": 900, "request_timeout_seconds": 900, "shutdown_grace_seconds": 30}}
        with np.load(full / "actions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["bridged_actions"][:native_chunks(maximum, 16)], np.stack(data.replay_actions)):
                raise ValueError("Historical EEF actions differ from admitted source")
        eef_packet = {"protocol": eef_config["action"], "actions": np.stack(data.replay_actions),
            "reset_args": {"value": image}, "initial_views": {"droid.cosmos_composite": image},
            "source": {**data.provenance, "conversion": "Official raw EEF conversion; source actions equal frozen historical controls"}}
        joint_root = root / plan["joint_study"] / f"ep{episode:06d}"
        joint_report = json.loads((joint_root / "full/summary.json").read_text())
        joint_packet = read_packet(joint_root / "full_input.npz")
        joint_config = json.loads((joint_root / "full/config.json").read_text())
        joint_config["kwargs"]["runtime_dir"] = {"suite_path": "runtime"}
        if not np.array_equal(joint_packet["initial_views"]["droid.cosmos_composite"], image):
            raise ValueError("Native models must start from exactly the same composite")
        joint_packet["actions"] = joint_packet["actions"][:native_chunks(maximum, 32)].copy()
        joint_packet["source"] = {**joint_packet["source"], "source_frames_per_chunk": 32,
            "end_frame": plan["start_frame"] + maximum,
            "replay_action_source": "recorded action.joint_position + action.gripper_position",
            "conversion": "Recorded absolute joint commands;32 targets per chunk; initial measured q; recurrent last-command proxy"}
        import pyarrow.parquet as pq
        rows = pq.read_table(data.provenance["data_path"], filters=[("episode_index", "=", episode)],
            columns=["action.joint_position", "action.gripper_position"]).to_pylist()
        expected = np.asarray([list(row["action.joint_position"]) + [float(row["action.gripper_position"])]
            for row in rows[plan["start_frame"]:plan["start_frame"] + maximum]], np.float32).reshape(-1, 32, 8)
        if not np.array_equal(expected, joint_packet["actions"]):
            raise ValueError("Historical joint commands differ from original parquet")
        for model, config, packet, baseline, packages in (
                ("eef", eef_config, eef_packet, full, ready["packages"]),
                ("joint", joint_config, joint_packet, joint_root / "full", joint_report["runtime"]["packages"])):
            baseline_meta = baseline / ("metadata.json" if model == "eef" else "summary.json")
            baseline_video = baseline / ("views/droid.cosmos_composite.mp4" if model == "eef" else "droid.cosmos_composite.mp4")
            if not baseline_video.is_file():
                raise FileNotFoundError(baseline_video)
            baselines.append({"model": model, "episode": episode, "directory": str(baseline),
                "video": str(baseline_video), "video_sha256": _digest(baseline_video),
                "report": str(baseline_meta), "report_sha256": _digest(baseline_meta),
                "packages": packages, "source_stride": 16 if model == "eef" else 32,
                "legacy_source_labels_not_edited": model == "joint"})
            for seed in plan["new_seeds"]:
                prepared.append(({"id": f"{model}_ep{episode:06d}_seed{seed}", "model": model,
                    "episode": episode, "seed": seed, "chunks": len(packet["actions"])},
                    {**config, "seed": seed}, packet))
    (output / "inputs").mkdir(parents=True)
    (output / "configs").mkdir()
    overrides = {"schema_version": 1, "paths": {key: str(env) for key in
        ("world_models.cosmos3_nano_droid_fd", "world_models.cosmos3_droid_joint_fd")}}
    override_path = output / "environments.json"
    override_path.write_text(json.dumps(overrides, indent=2))
    study = {"schema_version": 1, "plan": plan, "plan_sha256": _digest(plan_path), "baselines": baselines,
        "environments": str(override_path), "environments_sha256": _digest(override_path), "jobs": []}
    # One shared action/image packet per model/episode; seeds only change config.
    for job, config, packet in prepared:
        key = f"{job['model']}_ep{job['episode']:06d}"
        path = output / "inputs" / (key + ".npz")
        if not path.exists():
            write_packet(path, packet)
        config_path = output / "configs" / (job["id"] + ".json")
        config_path.write_text(json.dumps(config, indent=2))
        study["jobs"].append({**job, "input": str(path), "input_sha256": _digest(path),
            "config": str(config_path), "config_sha256": _digest(config_path)})
    study["planned_new_world_model_calls"] = sum(j["chunks"] for j in study["jobs"])
    (output / "study.json").write_text(json.dumps(study, indent=2))
    return study
