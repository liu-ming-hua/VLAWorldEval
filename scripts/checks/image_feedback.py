#!/usr/bin/env python3
"""Prepare/run paired visual-feedback controls using the shared replay executor."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def decode_boundaries(source, chunks, stride):
    """Compatibility import; common decoding lives in the dataset layer."""
    from evaluation.datasets.video import decode_boundaries as decode
    return decode(source, chunks, stride)


def prepare(output):
    import numpy as np
    from PIL import Image
    from evaluation.datasets.droid import load_droid_episode
    from evaluation.execution.transport import read_packet, write_packet
    from evaluation.reporting.task_scores import _digest
    from adapters.observations.droid import compose_droid_cosmos_view
    from evaluation.planning.registry import project_path
    output = project_path(ROOT, str(output))
    if output.exists():
        raise FileExistsError(output)
    plan_path = ROOT / "configs/studies/droid_image_feedback_v1.json"
    plan = json.loads(plan_path.read_text())
    old = json.loads((ROOT / plan["source_study"] / "summary.json").read_text())
    prepared = []
    # Validate all source actions and boundaries before emitting the study.
    for case in plan["cases"]:
        if case["model"] == "cosmos_joint_fd":
            packet = read_packet(ROOT / plan["joint_source"])
            config = json.loads((ROOT / "configs/replay/cosmos_joint_droid_native_v1.json").read_text())
            config["kwargs"]["runtime_dir"] = {"suite_path": "runtime"}
        else:
            detail = next(d for d in old["details"] if (d["episode"], d["world_model"]) == (case["episode"], case["model"]))
            job = next(j for j in detail["jobs"] if j["id"] == "real_action_full")
            artifact = Path(job["output"])
            worker = json.loads((artifact.parents[2] / "workers/world_model/worker.json").read_text())
            pair = json.loads((artifact / "pair_manifest.json").read_text())["world_model"]
            data = load_droid_episode(ROOT / plan["dataset"], case["episode"], plan["start_frame"], case["chunks"],
                protocol="ctrl_world" if case["model"] == "ctrl" else "cosmos")
            actions = np.stack(data.replay_actions)
            with np.load(artifact / "actions.npz", allow_pickle=False) as saved:
                if not np.array_equal(actions, saved["bridged_actions"]):
                    raise ValueError("Prepared actions differ from the frozen earlier controls")
            views = data.trial["observation_bundle"]["views"]
            kwargs = dict(worker["kwargs"])
            kwargs.pop("prompt", None)
            if "runtime_dir" in kwargs:
                kwargs["runtime_dir"] = {"suite_path": "runtime"}
            for name, value in list(kwargs.items()):
                if isinstance(value, str) and value.startswith(str(ROOT) + "/"):
                    kwargs[name] = {"project_path": str(Path(value).relative_to(ROOT))}
            config = {"schema_version": 1, "environment": "world_models." + ("ctrl_world" if case["model"] == "ctrl" else "cosmos3_nano_droid_fd"),
                "module": worker["module"], "adapter": worker["adapter"], "kwargs": kwargs,
                "gpus": [0] if case["model"] == "ctrl" else plan["allowed_gpus"],
                "memory_gib": worker["memory_gib"], "seed": plan["seed"], "action": pair["action"],
                "video": pair["video"], "output_views": pair["output_views"],
                "reset_operation": "reset_official_replay" if case["model"] == "ctrl" else "reset",
                "instruction_conditioning": case["model"] == "ctrl",
                "worker_options": {"startup_timeout_seconds": 900, "request_timeout_seconds": 900, "shutdown_grace_seconds": 30}}
            if case["model"] == "ctrl":
                initial = {name: np.asarray(Image.fromarray(views[name]).resize((320, 192))) for name in config["output_views"]}
                reset = {"image": list(initial.values()), "first_action": actions[0, 0]}
            else:
                composite = compose_droid_cosmos_view(views["droid.exterior_1"], views["droid.exterior_2"], views["droid.wrist"])[0]
                initial, reset = {"droid.cosmos_composite": composite}, {"value": composite}
            packet = {"protocol": config["action"], "actions": actions, "reset_args": reset,
                "initial_views": initial, "source": {**data.provenance, "conversion": "Frozen official DROID conversion; action values equal earlier saved controls"}}
        if len(packet["actions"]) != case["chunks"]:
            raise ValueError("Frozen chunk count mismatch")
        boundaries = decode_boundaries(packet["source"], case["chunks"], case["source_stride"])
        if case["model"] == "ctrl":
            images = np.stack([np.stack([np.asarray(Image.fromarray(boundaries[name][i]).resize((320, 192)))
                for name in config["output_views"]]) for i in range(case["chunks"])])
        else:
            images = compose_droid_cosmos_view(boundaries["droid.exterior_1"], boundaries["droid.exterior_2"], boundaries["droid.wrist"])
        anchors = {"images": images, "source_frame_indices": plan["start_frame"] + np.arange(case["chunks"]) * case["source_stride"],
            "source": {"conversion": "Real RGB from each camera at exact chunk start; no future robot states loaded", "videos": packet["source"]["videos"]},
            "intervention": "ground_truth_visual_context"}
        from evaluation.execution.native_replay import admit_image_anchors
        prepared.append((case, config, packet, anchors))
    (output / "inputs").mkdir(parents=True)
    (output / "configs").mkdir()
    study = {"schema_version": 1, "plan": plan, "plan_sha256": _digest(plan_path), "cases": []}
    for case, config, packet, anchors in prepared:
        packet_path = output / "inputs" / (case["id"] + ".npz")
        anchor_path = output / "inputs" / (case["id"] + "_anchors.npz")
        config_path = output / "configs" / (case["id"] + ".json")
        write_packet(packet_path, packet)
        write_packet(anchor_path, anchors)
        admit_image_anchors(anchor_path, packet, config)
        config_path.write_text(json.dumps(config, indent=2))
        study["cases"].append({**case, "input": str(packet_path), "anchors": str(anchor_path), "config": str(config_path),
            "input_sha256": _digest(packet_path), "anchors_sha256": _digest(anchor_path), "config_sha256": _digest(config_path)})
    (output / "study.json").write_text(json.dumps(study, indent=2))
    return {"status": "prepared", "cases": len(prepared), "planned_wm_calls": 88, "model_inference_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "run", "compare"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=["autoregressive", "ground_truth_visual_context"])
    args = parser.parse_args()
    from evaluation.planning.registry import project_path
    directory = project_path(ROOT, str(args.directory))
    if args.operation == "prepare":
        report = prepare(directory)
    elif args.operation == "compare":
        from evaluation.reporting.image_feedback import compare_image_feedback
        result = compare_image_feedback(ROOT, directory, args.output or directory / "comparisons",
            cases=(args.case,) if args.case else ())
        report = {"status": result["status"], "cases": len(result["cases"]), "model_calls": 0}
    else:
        from evaluation.execution.native_replay import run_native_replay
        from evaluation.reporting.task_scores import _digest
        from evaluation.execution.reliability import run_signal_handlers
        study = json.loads((directory / "study.json").read_text())
        case = next(c for c in study["cases"] if c["id"] == args.case)
        if args.mode is None:
            parser.error("run requires --mode")
        for key in ("input", "anchors", "config"):
            if _digest(Path(case[key])) != case[key + "_sha256"]:
                raise ValueError("Frozen study input/config changed")
        with run_signal_handlers():
            report = run_native_replay(Path(case["config"]), Path(case["input"]), directory / case["id"] / args.mode,
                image_anchors=Path(case["anchors"]) if args.mode == "ground_truth_visual_context" else None)
        report = {k: report[k] for k in ("status", "world_model_calls", "vla_calls", "feedback_mode")}
    print(json.dumps(report), flush=True)
    return 0 if report["status"] in {"prepared", "completed", "comparisons_built_not_quality_approved"} else 1


if __name__ == "__main__":
    sys.exit(main())
