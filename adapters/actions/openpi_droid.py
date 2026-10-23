"""OpenPI DROID controller targets -> Cosmos actions, not physical simulation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adapters.actions.base import ActionBridge
from adapters.actions.droid import _euler_xyz_matrix, _state_component, poses_to_cosmos_droid_action
from vlaworldeval.paths import project_root


class PandaKinematics:
    """NumPy serial-chain FK for the declared DROID/Polymetis link8 frame."""

    def __init__(self, path):
        path = Path(path)
        self.path = path if path.is_absolute() else project_root() / path
        definition = json.loads(self.path.read_text())
        if definition.get("schema_version") != 1 or definition.get("ee_link") != "panda_link8":
            raise ValueError("Expected versioned Panda link8 kinematics")
        self.origins = np.asarray(definition["joint_origins_xyz"], dtype=np.float64)
        self.rolls = np.asarray(definition["joint_origin_roll"], dtype=np.float64)
        self.tip = np.asarray(definition["tip_translation"], dtype=np.float64)
        self.lower = np.asarray(definition["joint_limits_lower"], dtype=np.float64)
        self.upper = np.asarray(definition["joint_limits_upper"], dtype=np.float64)
        if (self.origins.shape != (7, 3) or self.rolls.shape != (7,)
                or self.tip.shape != (3,) or self.lower.shape != (7,) or self.upper.shape != (7,)
                or not all(np.isfinite(x).all() for x in
                           (self.origins, self.rolls, self.tip, self.lower, self.upper))
                or np.any(self.lower >= self.upper)):
            raise ValueError("Invalid Panda kinematic geometry or limits")

    def transforms(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.ndim != 2 or joints.shape[1] != 7 or not np.isfinite(joints).all():
            raise ValueError("Panda joints must be finite [T,7]")
        transforms = np.repeat(np.eye(4)[None], len(joints), axis=0)
        for i, (origin, roll) in enumerate(zip(self.origins, self.rolls)):
            fixed = np.eye(4)
            fixed[:3, :3] = _euler_xyz_matrix(np.array([roll, 0, 0]))
            fixed[:3, 3] = origin
            turn = np.repeat(np.eye(4)[None], len(joints), axis=0)
            turn[:, :3, :3] = _euler_xyz_matrix(
                np.column_stack([np.zeros((len(joints), 2)), joints[:, i]]))
            transforms = transforms @ fixed @ turn
        transforms[:, :3, 3] += transforms[:, :3, :3] @ self.tip
        return transforms

    def poses(self, joints):
        transforms = self.transforms(joints)
        rotations = transforms[:, :3, :3]
        # Extrinsic xyz, matching the existing DROID -> Cosmos conversion.
        pitch = np.arctan2(-rotations[:, 2, 0], np.hypot(rotations[:, 0, 0], rotations[:, 1, 0]))
        roll = np.arctan2(rotations[:, 2, 1], rotations[:, 2, 2])
        yaw = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
        singular = np.abs(np.cos(pitch)) < 1e-8
        roll[singular] = np.arctan2(-rotations[singular, 1, 2], rotations[singular, 1, 1])
        yaw[singular] = 0
        return np.column_stack([transforms[:, :3, 3], roll, pitch, yaw])


class OpenPICosmosDroidActionBridge(ActionBridge):
    """Declared ideal controller following, with explicit zero-velocity tail.

    DROID normalized velocity is scaled by 0.2 rad/control tick (NOT rad/s).
    OpenPI's deployment clips each command and binarizes the closed-gripper
    value. Joint limits are saturated explicitly; no contact/servo lag is modeled.
    """

    def __init__(self, spec):
        super().__init__(spec)
        if (spec.source.dimension != 8 or spec.source.horizon not in {10, 15}
                or spec.source.semantics != "normalized_joint_velocity_with_absolute_gripper"
                or spec.source.rate_hz != 15 or spec.target.rate_hz != 15
                or spec.target.horizon != 16 or spec.target.dimension != 10
                or spec.config.get("tail_policy") != "zero_velocity_hold_gripper"):
            raise ValueError("OpenPI bridge requires native 10/15x8 at 15Hz -> 16x10 with explicit idle tail")
        self.kinematics = PandaKinematics(spec.config["kinematics"])
        if spec.config.get("joint_delta_per_control_tick") != 0.2:
            raise ValueError("This DROID controller version requires 0.2 rad/control tick")

    def trajectory(self, previous, action):
        self.spec.source.wrap(action.values, source=action.source)
        q0 = _state_component(previous, "observation/joint_position", 7)
        if np.any(q0 < self.kinematics.lower - 1e-5) or np.any(q0 > self.kinematics.upper + 1e-5):
            raise ValueError("Initial joint state outside Panda limits")
        values = np.clip(action.values.astype(np.float64), -1, 1)
        commands = np.zeros((16, 8))
        commands[:action.horizon, :7] = values[:, :7]
        commands[:action.horizon, 7] = (values[:, 7] > 0.5).astype(float)
        commands[action.horizon:, 7] = commands[action.horizon - 1, 7]
        joints = [q0]
        saturations = 0
        for command in commands:
            target = joints[-1] + 0.2 * command[:7]
            bounded = np.clip(target, self.kinematics.lower, self.kinematics.upper)
            saturations += int(np.count_nonzero(target != bounded))
            joints.append(bounded)
        joints = np.asarray(joints)
        return joints, commands[:, 7], saturations

    def convert(self, action, *, observation, step_index):
        if observation.robot_state is None:
            raise ValueError("OpenPI/Cosmos requires current joint state")
        joints, closed_gripper, saturations = self.trajectory(observation.robot_state, action)
        poses = self.kinematics.poses(joints)
        # OpenPI/DROID uses 1=closed. The selected Cosmos dataset branch
        # flips DROID gripper to 1=open, exactly once in the shared converter.
        values = poses_to_cosmos_droid_action(poses, closed_gripper)
        converted = self.spec.target.wrap(values, source=self.spec.name)
        from dataclasses import replace
        return replace(converted, metadata={
            "step_index": step_index, "predicted_action_steps": action.horizon,
            "idle_tail_steps": 16 - action.horizon,
            "tail_policy": "zero_velocity_hold_gripper",
            "joint_delta_per_control_tick": 0.2,
            "joint_limit_saturations": saturations,
            "state_assumption": "ideal_controller_target_not_measured_state",
            "gripper_convention": "openpi_1_closed_to_cosmos_1_open",
        })

    def integrate_state(self, previous, action, context):
        joints, gripper, _ = self.trajectory(previous, action)
        pose = self.kinematics.poses(joints[-1:])[0]
        return {
            "observation/joint_position": joints[-1].astype(np.float32),
            "observation/gripper_position": np.array([gripper[-1]], dtype=np.float32),
            "state.eef_position": pose[:3].astype(np.float32),
            "state.eef_rotation": pose[3:].astype(np.float32),
        }
