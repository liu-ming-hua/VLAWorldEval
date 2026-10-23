"""DROID policy-command to Cosmos forward-dynamics action bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from adapters.actions.base import ActionBridge
from evaluation.contracts import ActionChunk, ObservationBundle, RobotState
from evaluation.manifests import ActionBridgeSpec


from vlaworldeval.paths import project_root

PROJECT_ROOT = project_root()
OPENCV_DROID_ROT = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _euler_xyz_matrix(euler: np.ndarray) -> np.ndarray:
    """Extrinsic xyz matrix, matching Cosmos' official ``euler_xyz`` helper."""
    euler = np.asarray(euler, dtype=np.float64)
    x, y, z = np.moveaxis(euler, -1, 0)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    result = np.empty((*euler.shape[:-1], 3, 3), dtype=np.float64)
    result[..., 0, 0] = cz * cy
    result[..., 0, 1] = cz * sy * sx - sz * cx
    result[..., 0, 2] = cz * sy * cx + sz * sx
    result[..., 1, 0] = sz * cy
    result[..., 1, 1] = sz * sy * sx + cz * cx
    result[..., 1, 2] = sz * sy * cx - cz * sx
    result[..., 2, 0] = -sy
    result[..., 2, 1] = cy * sx
    result[..., 2, 2] = cy * cx
    return result


def _state_component(state: RobotState, name: str, width: int) -> np.ndarray:
    if name not in state.values:
        raise ValueError(f"DROID state schema is missing component {name!r}")
    value = np.asarray(state.values[name], dtype=np.float64)
    if value.size < width or value.shape[-1] != width:
        raise ValueError(f"DROID state component {name!r} must end in width {width}, got {value.shape}")
    return value.reshape(-1, width)[-1]


class DroidVelocityDynamics:
    """Versioned affine baseline from command velocity to measured pose delta."""

    def __init__(self, calibration_path: str | Path):
        self.path = _project_path(calibration_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"DROID dynamics calibration not found: {self.path}")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported DROID dynamics schema in {self.path}")
        self.name = str(payload["name"])
        self.coefficients = np.asarray(payload["model"]["coefficients"], dtype=np.float64)
        if self.coefficients.shape != (7, 6):
            raise ValueError(
                f"DROID affine coefficients must be [7,6], got {self.coefficients.shape}"
            )

    def pose_trajectory(self, state: RobotState, action: ActionChunk) -> tuple[np.ndarray, np.ndarray]:
        if action.dimension != 7:
            raise ValueError(f"DROID velocity dynamics expects 7-D actions, got {action.dimension}")
        initial = np.concatenate(
            [
                _state_component(state, "state.eef_position", 3),
                _state_component(state, "state.eef_rotation", 3),
            ]
        )
        velocity = action.values[:, :6].astype(np.float64)
        design = np.concatenate([velocity, np.ones((len(velocity), 1))], axis=1)
        deltas = design @ self.coefficients
        poses = np.concatenate([initial[None], initial[None] + np.cumsum(deltas, axis=0)])
        poses[:, 3:] = np.arctan2(np.sin(poses[:, 3:]), np.cos(poses[:, 3:]))
        gripper = np.clip(action.values[:, 6].astype(np.float64), 0.0, 1.0)
        return poses, gripper


def poses_to_cosmos_droid_action(poses: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Convert predicted DROID absolute poses into official raw Cosmos actions."""
    poses = np.asarray(poses, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != 6 or len(poses) != len(gripper) + 1:
        raise ValueError(
            f"Expected poses [H+1,6] and gripper [H], got {poses.shape}/{gripper.shape}"
        )
    rotation = _euler_xyz_matrix(poses[:, 3:]) @ OPENCV_DROID_ROT
    relative_rotation = np.einsum("tji,tjk->tik", rotation[:-1], rotation[1:])
    relative_translation = np.einsum(
        "tji,tj->ti", rotation[:-1], poses[1:, :3] - poses[:-1, :3]
    )
    rot6d = relative_rotation[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
    return np.concatenate(
        [relative_translation, rot6d, (1.0 - gripper)[:, None]], axis=1
    ).astype(np.float32)


class GrootN15CosmosDroidActionBridge(ActionBridge):
    """Map GR00T DROID velocity commands to Cosmos' measured-pose action domain."""

    def __init__(self, spec: ActionBridgeSpec):
        super().__init__(spec)
        calibration = spec.config.get("dynamics_calibration")
        if not calibration:
            raise ValueError(
                "groot_n15_to_cosmos3_droid_v1 requires dynamics_calibration"
            )
        self.dynamics = DroidVelocityDynamics(calibration)
        self.action_normalization = str(spec.config.get("action_normalization", "none"))
        self.stats_path: Path | None = None
        self.q01: np.ndarray | None = None
        self.q99: np.ndarray | None = None
        if self.action_normalization == "quantile_rot":
            stats_path = spec.config.get("normalizer_stats")
            if not stats_path:
                raise ValueError("quantile_rot requires normalizer_stats")
            self.stats_path = _project_path(stats_path).resolve()
            stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
            self.q01 = np.asarray(stats["q01"], dtype=np.float32)
            self.q99 = np.asarray(stats["q99"], dtype=np.float32)
            if self.q01.shape != (10,) or self.q99.shape != (10,):
                raise ValueError("Cosmos DROID normalization stats must be 10-D")
        elif self.action_normalization != "none":
            raise ValueError(
                "action_normalization must be 'none' for the released DROID FD model "
                "or 'quantile_rot' for a separately declared legacy experiment"
            )

    def _cosmos_action(
        self, action: ActionChunk, observation: ObservationBundle
    ) -> np.ndarray:
        if observation.robot_state is None:
            raise ValueError("GR00T/Cosmos action bridge requires current DROID EEF state")
        poses, gripper = self.dynamics.pose_trajectory(observation.robot_state, action)
        raw = poses_to_cosmos_droid_action(poses, gripper)
        if self.action_normalization == "none":
            return raw
        assert self.q01 is not None and self.q99 is not None
        scale = np.maximum((self.q99 - self.q01) * 0.5, 1e-8)
        offset = (self.q99 + self.q01) * 0.5
        return np.clip((raw - offset) / scale, -1.0, 1.0).astype(np.float32)

    def convert(
        self,
        action: ActionChunk,
        *,
        observation: ObservationBundle,
        step_index: int,
    ) -> ActionChunk:
        self.spec.source.wrap(action.values, source=action.source)
        values = self._cosmos_action(action, observation)
        target = self.spec.target
        return ActionChunk(
            values=values,
            representation=target.representation,
            frame=target.frame,
            semantics=target.semantics,
            source=self.spec.name,
            rate_hz=target.rate_hz,
            component_names=target.components,
            normalization=target.normalization,
            metadata={
                "step_index": step_index,
                "dynamics_calibration": self.dynamics.name,
                "action_normalization": self.action_normalization,
                "normalizer_stats": str(self.stats_path) if self.stats_path else None,
            },
        )

    def integrate_state(self, previous: RobotState, action: ActionChunk, context: Any) -> dict:
        """Return the same terminal proxy state used to construct Cosmos actions."""
        poses, gripper = self.dynamics.pose_trajectory(previous, action)
        return {
            "state.eef_position": poses[-1, :3].astype(np.float32),
            "state.eef_rotation": poses[-1, 3:].astype(np.float32),
            "state.gripper_position": np.asarray([gripper[-1]], dtype=np.float32),
        }


__all__ = [
    "DroidVelocityDynamics",
    "GrootN15CosmosDroidActionBridge",
    "poses_to_cosmos_droid_action",
]
