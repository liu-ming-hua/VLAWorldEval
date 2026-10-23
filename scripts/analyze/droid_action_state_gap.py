#!/usr/bin/env python3
"""Measure how DROID command actions differ from next measured robot state.

This is the M1 gate for StateBridge. It compares common proxy-state choices on
episode-disjoint transitions and writes a compact JSON report. No video or
model checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/cosmos3_droid"
DEFAULT_OUTPUT = ROOT / "results/analysis/droid_action_state_gap_v1/summary.json"
COLUMNS = (
    "episode_index",
    "frame_index",
    "observation.state.cartesian_position",
    "observation.state.gripper_position",
    "action.cartesian_position",
    "action.cartesian_velocity",
    "action.gripper_position",
)


def _rows(column: Any) -> np.ndarray:
    return np.asarray(column.to_pylist(), dtype=np.float32)


def _angle_wrap(value: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(value), np.cos(value))


def _rotation_matrix_xyz(euler: np.ndarray) -> np.ndarray:
    """Vectorized intrinsic XYZ rotation matching the DROID Euler convention."""
    euler = np.asarray(euler, dtype=np.float64)
    x, y, z = np.moveaxis(euler, -1, 0)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    result = np.empty((*euler.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0, 0] = cy * cz
    result[..., 0, 1] = -cy * sz
    result[..., 0, 2] = sy
    result[..., 1, 0] = cx * sz + sx * sy * cz
    result[..., 1, 1] = cx * cz - sx * sy * sz
    result[..., 1, 2] = -sx * cy
    result[..., 2, 0] = sx * sz - cx * sy * cz
    result[..., 2, 1] = sx * cz + cx * sy * sz
    result[..., 2, 2] = cx * cy
    return result


def _rotation_error_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted_matrix = _rotation_matrix_xyz(predicted)
    target_matrix = _rotation_matrix_xyz(target)
    relative = np.einsum("...ji,...jk->...ik", predicted_matrix, target_matrix)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def _quantiles(value: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(value)),
        "p50": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "p99": float(np.quantile(value, 0.99)),
        "max": float(np.max(value)),
    }


def load_transitions(
    dataset_root: Path,
    split: str,
    *,
    max_files: int,
    max_transitions: int,
) -> dict[str, np.ndarray]:
    paths = sorted((dataset_root / split / "data").glob("chunk-*/*.parquet"))[:max_files]
    if not paths:
        raise FileNotFoundError(f"No DROID parquet files found under {dataset_root / split / 'data'}")
    buffers: dict[str, list[np.ndarray]] = {
        "episode": [],
        "current_pose": [],
        "next_pose": [],
        "velocity": [],
        "command_pose": [],
        "current_gripper": [],
        "next_gripper": [],
        "command_gripper": [],
    }
    count = 0
    for path in paths:
        table = pq.read_table(path, columns=list(COLUMNS))
        episode = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        pose = _rows(table["observation.state.cartesian_position"])
        gripper = _rows(table["observation.state.gripper_position"]).reshape(-1)
        command_pose = _rows(table["action.cartesian_position"])
        velocity = _rows(table["action.cartesian_velocity"])
        command_gripper = _rows(table["action.gripper_position"]).reshape(-1)
        valid = (episode[1:] == episode[:-1]) & (frame[1:] == frame[:-1] + 1)
        indices = np.flatnonzero(valid)
        remaining = max_transitions - count
        if remaining <= 0:
            break
        indices = indices[:remaining]
        buffers["episode"].append(episode[indices])
        buffers["current_pose"].append(pose[indices])
        buffers["next_pose"].append(pose[indices + 1])
        buffers["velocity"].append(velocity[indices])
        buffers["command_pose"].append(command_pose[indices])
        buffers["current_gripper"].append(gripper[indices])
        buffers["next_gripper"].append(gripper[indices + 1])
        buffers["command_gripper"].append(command_gripper[indices])
        count += len(indices)
    return {key: np.concatenate(value) for key, value in buffers.items()}


def fit_velocity_models(data: dict[str, np.ndarray], train: np.ndarray) -> dict[str, np.ndarray]:
    velocity = data["velocity"][train].astype(np.float64)
    delta = data["next_pose"][train].astype(np.float64) - data["current_pose"][train]
    delta[:, 3:] = _angle_wrap(delta[:, 3:])
    diagonal = np.sum(velocity * delta, axis=0) / np.maximum(
        np.sum(velocity * velocity, axis=0), 1e-12
    )
    design = np.concatenate([velocity, np.ones((len(velocity), 1))], axis=1)
    ridge = np.eye(design.shape[1]) * 1e-6
    ridge[-1, -1] = 0.0
    full = np.linalg.solve(design.T @ design + ridge, design.T @ delta)
    return {"diagonal_scale": diagonal, "full_affine": full}


def proxy_metrics(
    data: dict[str, np.ndarray],
    models: dict[str, np.ndarray],
    fps: float,
    horizon: int,
) -> dict[str, Any]:
    if horizon < 1 or horizon > len(data["episode"]):
        raise ValueError(f"Invalid evaluation horizon {horizon}")
    starts = np.arange(len(data["episode"]) - horizon + 1)
    ends = starts + horizon - 1
    contiguous = data["episode"][starts] == data["episode"][ends]
    starts, ends = starts[contiguous], ends[contiguous]
    held_out = data["episode"][starts] % 5 == 0
    starts, ends = starts[held_out], ends[held_out]
    if len(starts) == 0:
        raise RuntimeError(f"No held-out windows found for horizon {horizon}")

    current = data["current_pose"][starts]
    target = data["next_pose"][ends]
    command = data["command_pose"][ends]
    velocity_sum = np.zeros((len(starts), 6), dtype=np.float64)
    for offset in range(horizon):
        velocity_sum += data["velocity"][starts + offset]
    full = models["full_affine"]
    fitted_full_delta = velocity_sum @ full[:-1] + horizon * full[-1]
    predictions = {
        "hold_current_state": current,
        "last_absolute_action_target": command,
        "velocity_divided_by_fps": current + velocity_sum / fps,
        "fitted_diagonal_velocity": current + velocity_sum * models["diagonal_scale"],
        "fitted_full_velocity": current + fitted_full_delta,
    }
    result = {}
    target_gripper = data["next_gripper"][ends]
    current_gripper = data["current_gripper"][starts]
    command_gripper = data["command_gripper"][ends]
    for name, predicted in predictions.items():
        position_cm = np.linalg.norm(predicted[:, :3] - target[:, :3], axis=1) * 100.0
        rotation_deg = _rotation_error_deg(predicted[:, 3:], target[:, 3:])
        predicted_gripper = command_gripper if name == "last_absolute_action_target" else current_gripper
        gripper_error = np.abs(predicted_gripper - target_gripper)
        result[name] = {
            "windows": int(len(starts)),
            "position_error_cm": _quantiles(position_cm),
            "rotation_geodesic_error_deg": _quantiles(rotation_deg),
            "gripper_absolute_error": _quantiles(gripper_error),
            "gripper_binary_accuracy": float(
                np.mean((predicted_gripper >= 0.5) == (target_gripper >= 0.5))
            ),
        }
    return result


def episode_residuals(data: dict[str, np.ndarray], mask: np.ndarray, limit: int = 100) -> list[dict]:
    episode = data["episode"][mask]
    target = data["next_pose"][mask]
    command = data["command_pose"][mask]
    position_cm = np.linalg.norm(command[:, :3] - target[:, :3], axis=1) * 100.0
    rotation_deg = _rotation_error_deg(command[:, 3:], target[:, 3:])
    rows = []
    for episode_id in np.unique(episode):
        selected = episode == episode_id
        pos = float(np.mean(position_cm[selected]))
        rot = float(np.mean(rotation_deg[selected]))
        rows.append(
            {
                "episode_index": int(episode_id),
                "transitions": int(np.sum(selected)),
                "mean_position_error_cm": pos,
                "mean_rotation_error_deg": rot,
                "ranking_score": pos + rot / 5.0,
            }
        )
    return sorted(rows, key=lambda item: item["ranking_score"], reverse=True)[:limit]


def analyze_split(data: dict[str, np.ndarray], fps: float, horizons: list[int]) -> dict[str, Any]:
    held_out = data["episode"] % 5 == 0
    train = ~held_out
    if not np.any(train) or not np.any(held_out):
        raise RuntimeError("Episode-hash split produced an empty train or held-out set")
    models = fit_velocity_models(data, train)
    delta = data["next_pose"] - data["current_pose"]
    delta[:, 3:] = _angle_wrap(delta[:, 3:])
    correlations = [
        float(np.corrcoef(data["velocity"][:, index], delta[:, index])[0, 1])
        for index in range(6)
    ]
    horizon_metrics = {
        str(horizon): proxy_metrics(data, models, fps, horizon) for horizon in horizons
    }
    return {
        "transitions": int(len(data["episode"])),
        "episodes": int(len(np.unique(data["episode"]))),
        "train_transitions": int(np.sum(train)),
        "held_out_transitions": int(np.sum(held_out)),
        "split_rule": "episode_index % 5 == 0 is held out",
        "velocity_to_measured_delta_correlation": correlations,
        "fitted_diagonal_velocity_scale": models["diagonal_scale"].tolist(),
        "fitted_full_velocity_affine": models["full_affine"].tolist(),
        "held_out_proxy_metrics": horizon_metrics[str(horizons[0])],
        "held_out_proxy_metrics_by_horizon": horizon_metrics,
        "highest_last_action_residual_episodes": episode_residuals(data, held_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", nargs="+", default=["success", "failure"])
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--max-transitions", type=int, default=500_000)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8, 16])
    args = parser.parse_args()
    if (
        args.max_files < 1
        or args.max_transitions < 1
        or args.fps <= 0
        or not args.horizons
        or min(args.horizons) < 1
    ):
        raise ValueError("max-files, max-transitions, fps, and horizons must be positive")

    report = {
        "schema_version": 1,
        "analysis": "DROID command action versus next measured state",
        "dataset_root": str(args.dataset_root.resolve()),
        "fps": args.fps,
        "max_files_per_split": args.max_files,
        "max_transitions_per_split": args.max_transitions,
        "horizons": args.horizons,
        "rotation_metric": "geodesic angle from intrinsic XYZ Euler matrices",
        "limitations": [
            "The current split is episode-disjoint but not yet scene-disjoint.",
            "This diagnostic fits linear velocity baselines; it is not the StateBridge model.",
        ],
        "splits": {},
    }
    for split in args.splits:
        data = load_transitions(
            args.dataset_root,
            split,
            max_files=args.max_files,
            max_transitions=args.max_transitions,
        )
        report["splits"][split] = analyze_split(data, args.fps, args.horizons)
        print(
            f"{split}: {report['splits'][split]['transitions']} transitions, "
            f"{report['splits'][split]['episodes']} episodes"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
