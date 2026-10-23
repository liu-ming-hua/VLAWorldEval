"""Admit local LeRobot-v3 DROID episodes with independent camera offsets."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from adapters.actions.droid import poses_to_cosmos_droid_action
from evaluation.contracts import RobotState
from evaluation.datasets.episode import EpisodeInput as DroidEpisode

CAMERAS = {
    "droid.exterior_1": "observation.image.exterior_image_1_left",
    "droid.exterior_2": "observation.image.exterior_image_2_left",
    "droid.wrist": "observation.image.wrist_image_left",
}


def validate_rows(frame_indices, timestamps, poses, gripper, *, start, chunks, horizon=16, fps=15):
    required = start + chunks * horizon + 1
    if start < 0 or chunks < 1 or len(poses) < required:
        raise ValueError(f"Incomplete action window: need {required} pose rows, have {len(poses)}; no tail padding")
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError("Episode frame indices must be contiguous and start at zero")
    if len(timestamps) != len(poses) or not np.allclose(np.diff(timestamps), 1 / fps, atol=1e-4):
        raise ValueError("DROID timestamps do not match the declared FPS")
    if poses.shape != (len(frame_indices), 6) or gripper.shape != (len(frame_indices),):
        raise ValueError("DROID pose/gripper dimensions do not match the frame timeline")
    if not np.isfinite(poses).all() or not np.isfinite(gripper).all():
        raise ValueError("Dataset contains nonfinite pose/gripper values")
    if np.any((gripper < 0) | (gripper > 1)):
        raise ValueError("DROID gripper commands must be in [0,1]")


def episode_instruction(tasks, *, required=True):
    """Missing language is legal only when the caller explicitly does not need it."""
    if type(required) is not bool:
        raise ValueError("Instruction requirement must be an explicit boolean")
    instruction = str(tasks[0]).split(" | ")[0].strip() if tasks else ""
    if required and not instruction:
        raise ValueError("Episode task instruction is missing")
    return instruction


def load_droid_episode(root: Path, episode_index: int, start: int, chunks: int, *, fps=15, protocol="cosmos", state_format="eef", require_instruction=True) -> DroidEpisode:
    if protocol not in {"cosmos", "ctrl_world"}:
        raise ValueError("Unknown DROID model protocol")
    if state_format not in {"eef", "openpi", "groot_n17"}:
        raise ValueError("Unknown DROID policy state format")
    horizon = 16 if protocol == "cosmos" else 12  # Ctrl: inclusive 5 samples at 0,3,6,9,12.
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ValueError("DROID parquet loading requires PyArrow; run CORE_DROID=1 bash scripts/setup/core.sh or install the data-droid extra in the core environment") from exc

    root = root.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != fps:
        raise ValueError("Dataset info FPS mismatch")
    row, meta_path = None, None
    for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        candidates = pq.read_table(path, filters=[("episode_index", "=", episode_index)]).to_pylist()
        if candidates:
            if len(candidates) != 1:
                raise ValueError("Duplicate episode metadata")
            row, meta_path = candidates[0], path
            break
    if row is None:
        raise ValueError(f"Episode {episode_index} metadata is not local")
    data_path = root / f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
    columns = ["frame_index", "timestamp", "observation.state.cartesian_position",
               "observation.state.gripper_position", "action.gripper_position"]
    if state_format != "eef":
        columns.append("observation.state.joint_positions")
    data = pq.read_table(data_path, columns=columns, filters=[("episode_index", "=", episode_index)])
    indices = np.asarray(data["frame_index"].to_numpy())
    timestamps = np.asarray(data["timestamp"].to_numpy())
    poses = np.asarray(data["observation.state.cartesian_position"].to_pylist(), dtype=np.float32)
    grip = np.asarray(data["action.gripper_position"].to_pylist(), dtype=np.float32).reshape(-1)
    measured_grip = np.asarray(data["observation.state.gripper_position"].to_pylist(), dtype=np.float32).reshape(-1)
    if len(indices) != row["length"]:
        raise ValueError("Local parquet episode length differs from metadata")
    validate_rows(indices, timestamps, poses, grip, start=start, chunks=chunks, fps=fps, horizon=horizon)
    if measured_grip.shape != grip.shape or not np.isfinite(measured_grip).all() or np.any((measured_grip < 0) | (measured_grip > 1)):
        raise ValueError("Measured gripper state must be finite, normalized and synchronized")
    instruction = episode_instruction(row["tasks"], required=require_instruction)
    views, videos = {}, {}
    for name, key in CAMERAS.items():
        prefix = f"videos/{key}"
        path = root / f"videos/{key}/chunk-{row[prefix + '/chunk_index']:03d}/file-{row[prefix + '/file_index']:03d}.mp4"
        offset = float(row[prefix + "/from_timestamp"])
        if not path.is_file():
            raise FileNotFoundError(path)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(path),
        ], text=True))
        stream = probe["streams"][0]
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        if not denominator or abs(numerator / denominator - fps) > 0.01:
            raise ValueError(f"Video FPS mismatch: {path}")
        end_time = offset + (start + chunks * horizon) / fps
        if float(probe["format"]["duration"]) + 1 / fps < end_time:
            raise ValueError(f"Video does not cover all requested actions: {path}")
        # Output-side seek decodes accurately rather than selecting a keyframe.
        history_start = start - 15 if state_format == "groot_n17" else start
        history_count = 16 if state_format == "groot_n17" else 1
        if history_start < 0:
            raise ValueError("GR00T N1.7 requires real 15-frame prehistory, not repeated initial frames")
        pixels = subprocess.check_output([
            "ffmpeg", "-v", "error", "-i", str(path), "-ss", str(offset + history_start / fps),
            "-frames:v", str(history_count), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ])
        history = np.frombuffer(pixels, dtype=np.uint8).reshape(history_count, stream["height"], stream["width"], 3).copy()
        views[name] = history if history_count > 1 else history[0]
        videos[name] = {"path": str(path), "offset_seconds": offset, "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns, "initial_frame_time": offset + start / fps}
    state = RobotState(values={
        "state.eef_position": poses[start, :3], "state.eef_rotation": poses[start, 3:],
        "state.gripper_position": measured_grip[start:start + 1],
    }, schema="droid_eef_rpy_gripper_v1", source="dataset_ground_truth")
    kinematic_evidence = None
    if state_format != "eef":
        from adapters.actions.openpi_droid import PandaKinematics
        from adapters.actions.droid import _euler_xyz_matrix
        joints = np.asarray(data["observation.state.joint_positions"].to_pylist(), dtype=np.float64)
        if joints.shape != (len(poses), 7) or not np.isfinite(joints).all():
            raise ValueError("OpenPI requires synchronized finite joint states [T,7]")
        geometry = PandaKinematics("configs/kinematics/droid_panda_link8_v1.json")
        transform = geometry.transforms(joints)
        position_error = np.linalg.norm(transform[:, :3, 3] - poses[:, :3], axis=1)
        relative = transform[:, :3, :3].transpose(0, 2, 1) @ _euler_xyz_matrix(poses[:, 3:])
        angle_error = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1))
        if position_error.max() > 0.002 or angle_error.max() > 0.01:
            raise ValueError("Dataset EEF does not match Panda link8 FK; robot/tool/frame must be specified")
        kinematic_evidence = {"rows_checked": len(joints), "max_position_error_m": float(position_error.max()),
                              "max_rotation_error_rad": float(angle_error.max()), "ee_link": "panda_link8"}
        state = RobotState(values={
            "observation/joint_position": joints[start].astype(np.float32),
            "observation/gripper_position": measured_grip[start:start + 1],
            "state.eef_position": poses[start, :3], "state.eef_rotation": poses[start, 3:],
        }, schema="droid_openpi_joint_gripper_eef_v1", source="dataset_ground_truth")
        if state_format == "groot_n17":
            from adapters.actions.groot_n17_droid import n17_eef_state
            state = RobotState(values={"eef_9d": n17_eef_state(poses[start:start + 1])[0],
                "joint_position": joints[start].astype(np.float32),
                "gripper_position": measured_grip[start:start + 1]},
                schema="groot_n17_droid_joint_eef9d_v1", source="dataset_ground_truth")
    if protocol == "cosmos":
        actions = [poses_to_cosmos_droid_action(poses[i:i + 17], grip[i:i + 16])
                   for i in range(start, start + chunks * 16, 16)]
    else:
        # Official Ctrl replay conditions on measured Cartesian state/gripper,
        # not the differently encoded commanded action.cartesian_position.
        actions = [np.concatenate([poses[i:i + 13:3], measured_grip[i:i + 13:3, None]], axis=1)
                   for i in range(start, start + chunks * 12, 12)]
    return DroidEpisode(
        trial={"observation_bundle": {"views": views, "robot_state": state},
               "instruction": instruction, "initial_state": state},
        replay_actions=actions,
        provenance={"episode_index": episode_index, "start_frame": start, "end_frame": start + chunks * horizon,
                    "fps": fps, "source_frames_per_chunk": horizon,
                    "temporal_layout": "inclusive" if protocol == "ctrl_world" else "future",
                    "action_source_frame_indices": [[i + j * 3 for j in range(5)]
                        for i in range(start, start + chunks * horizon, horizon)] if protocol == "ctrl_world" else None,
                    "instruction": instruction, "length": len(indices), "tail_policy": "drop_incomplete_chunk",
                    "kinematics_validation": kinematic_evidence,
                    "metadata_path": str(meta_path), "data_path": str(data_path), "data_size": data_path.stat().st_size,
                    "data_mtime_ns": data_path.stat().st_mtime_ns, "videos": videos,
                    "replay_action_source": ("measured EEF pose transitions + commanded gripper" if protocol == "cosmos"
                                             else "measured absolute EEF pose + measured gripper; source stride=3")},
    )


def load_openpi_droid_episode(root, episode_index, start, chunks, *, fps=15):
    """Same Cosmos replay protocol, with admitted OpenPI joint/gripper inputs."""
    return load_droid_episode(root, episode_index, start, chunks, fps=fps, state_format="openpi")


def load_groot_n17_droid_episode(root, episode_index, start, chunks, *, fps=15):
    return load_droid_episode(root, episode_index, start, chunks, fps=fps, state_format="groot_n17")


def _ctrl_policy_episode(root, episode_index, start, chunks, fps, state_format):
    packet = load_droid_episode(root, episode_index, start, chunks, fps=fps,
                                protocol="ctrl_world", state_format=state_format)
    state = packet.trial["initial_state"]
    if state_format == "groot_n17":
        from adapters.actions.openpi_droid import PandaKinematics
        geometry = PandaKinematics("configs/kinematics/droid_panda_link8_v1.json")
        initial_pose = np.concatenate([geometry.poses(state.values["joint_position"][None])[0],
                                        state.values["gripper_position"]])
        # Six 5Hz history slots represent [-15,-12,-9,-6,-3,0] source ticks.
        # N1.7 receives slots [-5,0], preserving its one-second [-15,0] gap.
        packet.trial["observation_bundle"]["views"] = {
            name: history[::3] for name, history in packet.trial["observation_bundle"]["views"].items()}
    else:
        initial_pose = np.concatenate([state.values["state.eef_position"],
            state.values["state.eef_rotation"], state.values.get("state.gripper_position",
            state.values.get("observation/gripper_position"))])
    packet.trial["episode_bundle"] = {
        "views": [packet.trial["observation_bundle"]["views"][name][-1]
                  if state_format == "groot_n17" else packet.trial["observation_bundle"]["views"][name]
                  for name in CAMERAS],
        "first_action": initial_pose, "instruction": packet.trial["instruction"]}
    return packet


def load_openpi_ctrl_episode(root, episode_index, start, chunks, *, fps=15):
    return _ctrl_policy_episode(root, episode_index, start, chunks, fps, "openpi")


def load_groot_n15_ctrl_episode(root, episode_index, start, chunks, *, fps=15):
    return _ctrl_policy_episode(root, episode_index, start, chunks, fps, "eef")


def load_groot_n17_ctrl_episode(root, episode_index, start, chunks, *, fps=15):
    return _ctrl_policy_episode(root, episode_index, start, chunks, fps, "groot_n17")
