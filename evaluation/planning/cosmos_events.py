"""Freeze real-action event windows only after a separate recorded-video review."""
import json
from pathlib import Path

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw

from adapters.observations.droid import compose_droid_cosmos_view
from evaluation.datasets.droid import load_droid_episode
from evaluation.datasets.video import decode_source_views
from evaluation.execution.transport import write_packet
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive
from vlaworldeval.paths import fresh_output_path

INFERENCE_SOURCES = ["adapters/world_models/cosmos3_nano_droid_fd.py", "adapters/world_models/cosmos3_droid_joint_fd.py",
    "adapters/backends/cosmos.py", "adapters/backends/cosmos_actions.py", "adapters/actions/droid.py",
    "adapters/observations/droid.py", "evaluation/execution/native_replay.py", "evaluation/execution/worker.py",
    "evaluation/contracts.py", "evaluation/manifests.py"]


def event_window(gripper):
    values = np.asarray(gripper, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Require finite normalized measured gripper trajectory")
    events = np.flatnonzero((values[1:] > .5) & (values[:-1] <= .5)) + 1
    events = events[events >= 15]
    if not len(events):
        raise ValueError("No measured upward crossing after frame15; do not silently replace episode")
    event = int(events[0])
    start = max(15, event - 32)
    if start + 64 >= len(values):
        raise ValueError("Incomplete event window; no tail padding")
    return {"event": event, "start": start, "end": start + 64}


def prepare_event_study(root, output):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    output = fresh_output_path(root, output)
    plan_path = root / "configs/studies/cosmos_event_windows_v1.json"
    plan = json.loads(plan_path.read_text())
    if (plan["source_ticks"] != 64 or plan["fps"] != 15 or plan["new_seeds"] != [1, 2] or plan["allowed_gpus"] != [0, 1, 2, 3]
            or plan["models"] != ["eef", "joint"] or [c["episode"] for c in plan["cases"]] != [0, 351, 365, 795, 903]):
        raise ValueError("Event study v1 requires its fixed five cases, models, seeds, horizon and GPU scope")
    parent = root / plan["parent"]
    old = json.loads((parent / "study.json").read_text())
    prepared, cases = [], []
    output.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "configs", "references"):
        (output / name).mkdir()
    for case in plan["cases"]:
        episode, start = case["episode"], case["start"]
        data = load_droid_episode(root / plan["dataset"], episode, start, 4)
        table = pq.read_table(data.provenance["data_path"], filters=[("episode_index", "=", episode)])
        rows = sorted(table.to_pylist(), key=lambda r: r["frame_index"])
        if [r["frame_index"] for r in rows] != list(range(len(rows))):
            raise ValueError("Event rows must be contiguous episode-local frames")
        measured = np.asarray([r["observation.state.gripper_position"] for r in rows], np.float32)
        if event_window(measured) != {k: case[k] for k in ("event", "start", "end")}:
            raise ValueError("Declared window differs from original measured sensor rule")
        real = decode_source_views(data.provenance, 65)
        composite = compose_droid_cosmos_view(real["droid.exterior_1"], real["droid.exterior_2"], real["droid.wrist"])
        views = data.trial["observation_bundle"]["views"]
        initial = compose_droid_cosmos_view(views["droid.exterior_1"], views["droid.exterior_2"], views["droid.wrist"])[0]
        if not np.array_equal(composite[0], initial):
            raise ValueError("Reference first frame differs from model reset")
        folder = output / "references" / f"ep{episode:06d}"
        folder.mkdir()
        references = {}
        for name, frames in real.items():
            file = folder / (name.split(".")[-1] + ".mp4")
            media.write_video(file, frames, fps=15, crf=18, ffmpeg_args=("-threads", "2", "-movflags", "+faststart"))
            references["real." + name.split(".")[-1]] = {"path": str(file), "sha256": _digest(file)}
        for name, frames in (("composite", composite), ("hold", np.repeat(initial[None], 65, axis=0))):
            file = folder / (name + ".mp4")
            media.write_video(file, frames, fps=15, crf=18, ffmpeg_args=("-threads", "2", "-movflags", "+faststart"))
            references[name] = {"path": str(file), "sha256": _digest(file)}
        sheet = Image.new("RGB", (1920, 568), (20, 20, 20))
        draw = ImageDraw.Draw(sheet)
        for col, tick in enumerate((16, 32, 48)):
            draw.text((col * 640 + 5, 5), f"ep{episode} source={start+tick} relative={tick/15:.3f}s | REAL ONLY", fill="white")
            sheet.paste(Image.fromarray(composite[tick]), (col * 640, 28))
        sheet.save(folder / "event_sheet.jpg", quality=95)
        cases.append({**case, "instruction": data.provenance["instruction"], "references": references,
            "event_sheet": str(folder / "event_sheet.jpg"), "event_sheet_sha256": _digest(folder / "event_sheet.jpg"),
            "data_path": data.provenance["data_path"], "data_sha256": _digest(Path(data.provenance["data_path"])),
            "measured_gripper_at_ticks_0_16_32_48_64": measured[start + np.array([0, 16, 32, 48, 64])].tolist()})
        for model, stride in (("eef", 16), ("joint", 32)):
            template = next(j for j in old["jobs"] if j["model"] == model and j["episode"] == episode and j["seed"] == 1)
            config_path = Path(template["config"])
            if _digest(config_path) != template["config_sha256"]:
                raise ValueError("Parent native config changed")
            config = json.loads(config_path.read_text())
            source = {**data.provenance, "source_frames_per_chunk": stride, "sensor_event_frame": case["event"],
                "event_is_contact_label": False, "conversion": "Official measured EEF transitions+commanded gripper" if model == "eef" else "Recorded absolute joint+gripper commands; measured initial q then last-command proxy"}
            if model == "eef":
                actions, reset = np.stack(data.replay_actions), initial
            else:
                values = np.asarray([list(r["action.joint_position"]) + [r["action.gripper_position"]] for r in rows[start:start+64]], np.float32)
                actions = values.reshape(2, 32, 8)
                reset = {"observation/image": initial, "observation/joint_position": np.asarray(rows[start]["observation.state.joint_positions"], np.float32),
                    "observation/gripper_position": measured[start:start+1], "prompt": data.provenance["instruction"]}
            packet = {"protocol": config["action"], "actions": actions, "reset_args": {"value": reset},
                "initial_views": {"droid.cosmos_composite": initial}, "source": source}
            path = output / "inputs" / f"{model}_ep{episode:06d}.npz"
            write_packet(path, packet)
            for seed in plan["new_seeds"]:
                key = f"{model}_ep{episode:06d}_seed{seed}"
                target = output / "configs" / (key + ".json")
                target.write_text(json.dumps({**config, "seed": seed}, indent=2))
                prepared.append({"id": key, "model": model, "episode": episode, "seed": seed, "chunks": 64 // stride,
                    "input": str(path), "input_sha256": _digest(path), "config": str(target), "config_sha256": _digest(target),
                    "template_config": str(config_path), "template_config_sha256": _digest(config_path)})
    env_path = output / "environments.json"
    env_path.write_bytes((parent / "environments.json").read_bytes())
    report = {"schema_version": 1, "plan": plan, "plan_sha256": _digest(plan_path), "cases": cases, "jobs": prepared,
        "parent_study_sha256": _digest(parent / "study.json"), "environments": str(env_path), "environments_sha256": _digest(env_path),
        "planned_new_world_model_calls": sum(j["chunks"] for j in prepared), "status": "awaiting_recorded_source_review"}
    (output / "preparation.json").write_text(json.dumps(report, indent=2))
    return report


def freeze_event_study(root, directory):
    root, directory = Path(root).resolve(), project_path(root, str(directory))
    prepared = json.loads((directory / "preparation.json").read_text())
    review_path = directory / "source_review.json"
    review = json.loads(review_path.read_text())
    if (_digest(root / "configs/studies/cosmos_event_windows_v1.json") != prepared["plan_sha256"]
            or _digest(root / prepared["plan"]["parent"] / "study.json") != prepared["parent_study_sha256"]
            or _digest(Path(prepared["environments"])) != prepared["environments_sha256"]):
        raise ValueError("Plan, parent or frozen environments changed")
    if (review.get("reviewer") != "assistant" or review.get("human_review_performed") is not False
            or len(review["cases"]) != 5 or {r["episode"] for r in review["cases"]} != {c["episode"] for c in prepared["cases"]}):
        raise ValueError("Require all five recorded-event reviews, without silently dropping uncertain cases")
    for case in prepared["cases"]:
        if any(_digest(Path(value["path"])) != value["sha256"] for value in case["references"].values()):
            raise ValueError("Recorded reference or hold diagnostic changed")
        note = next(r for r in review["cases"] if r["episode"] == case["episode"])
        if (note.get("relevance") not in {"relevant_candidate", "uncertain"} or not note.get("notes")
                or note.get("contact_ground_truth") is not False or note.get("task_success") is not None
                or _digest(Path(case["event_sheet"])) != case["event_sheet_sha256"]
                or _digest(Path(case["data_path"])) != case["data_sha256"]):
            raise ValueError("Source review or recorded source changed")
    for job in prepared["jobs"]:
        for name in ("input", "config", "template_config"):
            if _digest(Path(job[name])) != job[name + "_sha256"]:
                raise ValueError("Prepared native inputs/config changed")
    result = {**prepared, "status": "frozen_before_inference", "preparation_sha256": _digest(directory / "preparation.json"),
        "source_review_sha256": _digest(review_path), "inference_source_sha256": {name: _digest(root / name) for name in INFERENCE_SOURCES}}
    publish_json_exclusive(directory / "study.json", result)
    return result
