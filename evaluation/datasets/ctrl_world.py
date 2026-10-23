"""Complete published97-window Ctrl validation, not the paper's256-clip set."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from evaluation.execution.transport import read_packet, write_packet
from evaluation.reporting.task_scores import _digest
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.assets import file_record
from vlaworldeval.atomic_json import publish_json_exclusive

REVISION = "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d"
VIEWS = ("exterior_1", "exterior_2", "wrist")


def verified_asset(repo, relative):
    path = repo/relative
    if not path.resolve().is_relative_to(repo.resolve()):
        raise ValueError("Native Ctrl assets must stay inside the pinned checkout")
    expected = subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"{REVISION}:{relative}"], text=True).strip()
    actual = subprocess.check_output(["git", "-C", str(repo), "hash-object", "--", str(path)], text=True).strip()
    if expected != actual:
        raise ValueError(f"Released Ctrl source asset differs: {relative}")
    return file_record(path)


def released_selection(repo):
    repo = Path(repo).resolve()
    name = "dataset_meta_info/droid_subset/val_sample.json"
    record = verified_asset(repo, name)
    samples = json.loads((repo/name).read_text())
    if (len(samples) != 97 or any(set(s) != {"episode_id", "frame_ids"} or type(s["episode_id"]) is not int
            or not isinstance(s["frame_ids"], list) or len(s["frame_ids"]) != 1
            or type(s["frame_ids"][0]) is not int or s["frame_ids"][0] < 0 for s in samples)
            or len({(s["episode_id"], s["frame_ids"][0]) for s in samples}) != 97):
        raise ValueError("Complete pinned97-window validation list required; no replacement")
    return samples, {name: record}


def load_released_episode(repo, episode):
    """Native get_traj_info RGB decoding; measured coarse states checked against raw."""
    from decord import VideoReader, cpu
    base = "dataset_example/droid_subset"
    ann_name = f"{base}/annotation/val/{episode}.json"
    records = {ann_name: verified_asset(repo, ann_name)}
    annotation = json.loads((repo/ann_name).read_text())
    states = np.asarray(annotation["states"], dtype=np.float64)
    raw = np.concatenate([np.asarray(annotation["observation.state.cartesian_position"], dtype=np.float64),
                          np.asarray(annotation["observation.state.gripper_position"], dtype=np.float64)[:, None]], axis=1)
    if (annotation.get("episode_id") != episode or states.shape != (annotation["video_length"], 7)
            or not np.isfinite(states).all() or not np.array_equal(states, raw[::3])):
        raise ValueError("Released coarse states must exactly match15Hz measured EEF/gripper stride3")
    views = {}
    for camera, name in enumerate(VIEWS):
        relative = f"videos/val/{episode}/{camera}.mp4"
        if annotation["videos"][camera]["video_path"] != relative:
            raise ValueError("Native released camera order/path differs")
        asset = f"{base}/{relative}"
        records[asset] = verified_asset(repo, asset)
        reader = VideoReader(str(repo/asset), ctx=cpu(0), num_threads=2)
        if len(reader) != len(states) or abs(reader.get_avg_fps()-5) > .001:
            raise ValueError("Released RGB must cover all coarse states at5FPS")
        value = reader.get_batch(range(len(reader))).asnumpy()
        if value.dtype != np.uint8 or value.shape != (len(states), 192, 320, 3):
            raise ValueError("Expected native192x320 RGB, not a resized substitute")
        views[name] = value
    instruction = annotation["texts"][0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Released Ctrl replay requires its original instruction")
    return states, views, instruction, records


def inclusive_action_windows(states):
    """Exactly the upstream min(frame_id,last_id) rule, including shared anchors."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 7 or len(states) < 2 or not np.isfinite(states).all():
        raise ValueError("A complete native Ctrl trajectory needs at least two finite7D states")
    count = (len(states)-2)//4+1
    indices = np.minimum(np.arange(count)[:, None]*4+np.arange(5), len(states)-1)
    return states[indices].astype(np.float32), indices*3, (len(states)-2)%4+1


def prepare_released_validation(root, repo, config_path, output, *, mode="short", gpu=None):
    from evaluation.datasets.native_reference import save_reference
    from evaluation.planning.registry import project_path
    from evaluation.reporting.replay_metrics import validate_reference
    root = Path(root).resolve()
    repo, config_path, output = [project_path(root, str(p)) for p in (repo, config_path, output)]
    if output.exists():
        raise FileExistsError(output)
    if mode not in {"short", "long"} or (gpu is not None and (type(gpu) is not int or gpu not in range(4))):
        raise ValueError("Choose short/long replay and a GPU in0..3")
    samples, records = released_selection(repo)
    config = json.loads(config_path.read_text())
    if (config.get("adapter") != "CtrlWorldAdapter" or config.get("reset_operation") != "reset_official_replay"
            or config["action"]["horizon"] != 5 or config["action"]["dimension"] != 7
            or config["video"] != {"fps": 5, "frames_per_chunk": 5, "temporal_layout": "inclusive"}
            or config["output_views"] != list(VIEWS) or config.get("seed") != 0
            or config["kwargs"].get("num_inference_steps") != 50 or not config["instruction_conditioning"]):
        raise ValueError("Use the native5x7 inclusive5FPS, seed0,50-step text-conditioned Ctrl replay config")
    stat_names = ("dataset_meta_info/droid/stat.json", "dataset_meta_info/droid_subset/stat.json")
    for name in stat_names:
        records[name] = verified_asset(repo, name)
    if records[stat_names[0]]["sha256"] != records[stat_names[1]]["sha256"]:
        raise ValueError("Model/default and released subset normalization statistics differ")
    data = {episode: load_released_episode(repo, episode) for episode in sorted({s["episode_id"] for s in samples})}
    for states, _, _, assets in data.values():
        records.update(assets)
    declared = {(s["episode_id"], s["frame_ids"][0]) for s in samples}
    if declared != {(episode, start) for episode, (states, *_) in data.items() for start in range(len(states)-4)}:
        raise ValueError("Released validation list no longer covers every complete5-frame stride1 window")
    if mode == "long":
        samples = [{"episode_id": episode, "frame_ids": [0]} for episode in sorted(data)]
    if gpu is not None:
        config = {**config, "gpus": [gpu]}
    with asset_destination_lock(root, output):
        output.mkdir(parents=True)
        cases = []
        for index, sample in enumerate(samples):
            episode, start = sample["episode_id"], sample["frame_ids"][0]
            states, views, instruction, _ = data[episode]
            indices = np.arange(len(states), dtype=np.int64) if mode == "long" else np.arange(start, start+5, dtype=np.int64)
            raw_indices = indices*3
            runtime = config
            if mode == "long":
                actions, condition_indices, valid = inclusive_action_windows(states)
                runtime = {**config, "video": {**config["video"], "final_chunk_valid_frames": valid}}
            else:
                actions, condition_indices = states[indices].astype(np.float32)[None], raw_indices[None]
            first = {name: frames[start].copy() for name, frames in views.items()}
            source = {"source_commit": REVISION, "episode_index": episode, "source_dataset": "released_droid_subset_validation",
                "instruction": instruction, "start_frame": int(raw_indices[0]), "end_frame": int(raw_indices[-1]),
                "fps": 15, "encoded_video_fps": 5, "source_frames_per_chunk": 12, "temporal_layout": "inclusive",
                "action_source_frame_indices": condition_indices.tolist(), "released_validation_index": index,
                "conversion": "Pinned released validation episodes/windows; Decord native RGB, measured coarse states exactly equal raw stride3; official repeated-initial replay memory, no training augmentation"}
            if mode == "long":
                source.update(last_recorded_source_frame=int(raw_indices[-1]), tail_policy="repeat_last_recorded_state",
                    conversion=source["conversion"]+"; full AR with native min(frame,last) tail conditions, padding predictions cropped",
                    padded_model_future_samples=int(len(actions)*4-(len(states)-1)), padded_predictions_scored=False)
            packet = {"protocol": config["action"], "actions": actions,
                "reset_args": {"image": [first[name] for name in VIEWS], "first_action": actions[0, 0]},
                "initial_views": first, "source": source}
            identifier = (f"ctrl_released_val{index:03d}_ep{episode}_start{start:04d}_seed0" if mode == "short"
                          else f"ctrl_released_long_ep{episode}_seed0")
            target = output/"cases"/identifier
            target.mkdir(parents=True)
            write_packet(target/"input.npz", packet)
            reference_views = {name: value[indices].copy() for name, value in views.items()}
            save_reference(target/"input.npz", target/"reference.npz", reference_views,
                {name: raw_indices for name in VIEWS}, domain="droid",
                split="official_released_subset_val97" if mode == "short" else "released_subset_validation_full_episodes",
                episode=episode, conversion=source["conversion"], condition_kind="future_measured_kinematics",
                frame_axis="original15hz_state_tick")
            write_packet(target/"reference.npz", read_packet(target/"reference.npz"), compressed=True)
            publish_json_exclusive(target/"config.json", runtime)
            validate_reference(read_packet(target/"reference.npz"), runtime, packet,
                {name: frame[None] for name, frame in first.items()}, input_sha256=_digest(target/"input.npz"))
            cases.append({"id": identifier, "prepared": str(target.relative_to(root))})
        publish_json_exclusive(output/"plan.json", {"schema_version": 1,
            "scope": ("All97 published droid_subset val windows,2 episodes, native replay skip1 without random train augmentation; NOT paper256 clips" if mode == "short"
                      else "Both complete episodes named in published droid_subset val list; every native5FPS state, no incomplete-tail drop; NOT full paper test"),
            "cases": cases, "unavailable": {"full_paper_test": "Exact256 paper clip identities not published/verified",
                **({"autoregressive_full_episodes": "Window-validation plan does not substitute for long autoregressive tests"} if mode == "short" else {})}})
        publish_json_exclusive(output/"coverage.json", {"schema_version": 1, "source_commit": REVISION,
            "source_files": records, "source_episodes": sorted(data), "published_short_windows": 97,
            "expected_cases": 97 if mode == "short" else len(data), "mode": mode,
            "prepared_cases": len(cases), "seed": 0, "model_inference_performed": False,
            "random_train_augmentation_used": False, "full_paper_test_completed": False})
    return {"plan": str(output/"plan.json"), "windows": len(cases), "episodes": len(data), "model_inference_performed": False}
