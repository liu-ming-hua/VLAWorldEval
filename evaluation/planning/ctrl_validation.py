"""Freeze corrected Ctrl replay inputs; never reuse legacy 15-tick packets."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.datasets.droid import load_droid_episode
from evaluation.execution.native_replay import admit_image_anchors
from evaluation.execution.transport import write_packet
from evaluation.reporting.task_scores import _digest
from vlaworldeval.paths import fresh_output_path


def prepare_ctrl_validation(root, output):
    from evaluation.datasets.video import decode_boundaries
    root = Path(root).resolve()
    output = fresh_output_path(root, output)
    plan_path = root / "configs/studies/ctrl_official_timing_v2.json"
    plan = json.loads(plan_path.read_text())
    references = json.loads((root / plan["source_study"] / "study.json").read_text())["references"]
    prepared = []
    for case in plan["cases"]:
        reference = next(r for r in references if r["episode"] == case["episode"])
        chunks = case.get("chunks", (reference["length"] - plan["start_frame"] - 1) // 12)
        data = load_droid_episode(root / plan["dataset"], case["episode"], plan["start_frame"], chunks, protocol="ctrl_world")
        config = json.loads((root / "configs/replay/ctrl_world_native_v2.json").read_text())
        config.update(gpus=case["gpus"], memory_gib=24, seed=plan["seed"])
        names = config["output_views"]
        initial = {name: np.asarray(Image.fromarray(data.trial["observation_bundle"]["views"][f"droid.{name}"]).resize((320, 192))) for name in names}
        actions = np.stack(data.replay_actions)
        packet = {"protocol": config["action"], "actions": actions,
            "reset_args": {"image": list(initial.values()), "first_action": actions[0, 0]},
            "initial_views": initial, "source": {**data.provenance,
                "conversion": "Ctrl official inclusive observed EEF+measured gripper, sample stride3, advance12, no tail padding"}}
        modes = ["autoregressive", "ground_truth_visual_context"] if case["id"] == "ctrl_ep795" else ["autoregressive"]
        anchors = None
        if len(modes) == 2:
            real = decode_boundaries(data.provenance, chunks, 12)
            images = np.stack([np.stack([np.asarray(Image.fromarray(real[f"droid.{name}"][i]).resize((320, 192))) for name in names]) for i in range(chunks)])
            anchors = {"images": images, "source_frame_indices": plan["start_frame"] + np.arange(chunks) * 12,
                "source": {"conversion": "real RGB only at inclusive-window starts", "videos": data.provenance["videos"]},
                "intervention": "ground_truth_visual_context"}
        prepared.append(({**case, "model": "ctrl", "chunks": chunks, "source_stride": 12, "modes": modes}, config, packet, anchors))
    (output / "inputs").mkdir(parents=True)
    (output / "configs").mkdir()
    study = {"schema_version": 1, "plan": {**plan, "modes": ["autoregressive", "ground_truth_visual_context"]},
        "plan_sha256": _digest(plan_path), "cases": [], "task_success_evaluated": False}
    for case, config, packet, anchors in prepared:
        for field, folder, suffix, value in (("input", "inputs", ".npz", packet), ("config", "configs", ".json", config), ("anchors", "inputs", "_anchors.npz", anchors)):
            if value is None:
                continue
            path = output / folder / (case["id"] + suffix)
            if field == "config":
                path.write_text(json.dumps(value, indent=2))
            else:
                write_packet(path, value)
            case[field] = str(path)
            case[field + "_sha256"] = _digest(path)
        if anchors is not None:
            admit_image_anchors(Path(case["anchors"]), packet, config)
        study["cases"].append(case)
    study["planned_world_model_calls"] = sum(c["chunks"] * len(c["modes"]) for c in study["cases"])
    (output / "study.json").write_text(json.dumps(study, indent=2))
    return study
