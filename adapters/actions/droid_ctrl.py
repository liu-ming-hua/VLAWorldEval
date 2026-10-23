"""DROID policies -> absolute EEF waypoints at Ctrl-World's 5Hz action rate."""
from dataclasses import replace

import numpy as np

from adapters.actions.base import ActionBridge
from adapters.actions.droid import DroidVelocityDynamics
from adapters.actions.openpi_droid import OpenPICosmosDroidActionBridge


class DroidCtrlActionBridge(ActionBridge):
    """Execute 12 ticks; condition on current state plus targets at 3,6,9,12."""

    def __init__(self, spec):
        super().__init__(spec)
        if spec.source.rate_hz != 15 or spec.target.rate_hz != 5 or spec.target.horizon != 5 or spec.target.dimension != 7 or spec.target.temporal_layout != "inclusive":
            raise ValueError("DROID/Ctrl requires 15Hz policy actions -> 5x7 at 5Hz")
        self.joint_proxy = None
        self.dynamics = None
        self.n17_proxy = None
        if spec.source.dimension == 8:
            cosmos_target = replace(spec.target, dimension=10, horizon=16, rate_hz=15, components=(), temporal_layout="future")
            self.joint_proxy = OpenPICosmosDroidActionBridge(replace(spec, target=cosmos_target))
        elif spec.source.dimension == 7 and spec.source.horizon == 16:
            self.dynamics = DroidVelocityDynamics(spec.config["dynamics_calibration"])
        elif spec.source.dimension == 17 and spec.source.horizon == 40:
            from adapters.actions.groot_n17_droid import GrootN17CosmosDroidActionBridge
            cosmos_target = replace(spec.target, dimension=10, horizon=16, rate_hz=15, components=(), temporal_layout="future")
            self.n17_proxy = GrootN17CosmosDroidActionBridge(replace(spec, target=cosmos_target,
                config={**spec.config, "execute_prefix": 16}))
        else:
            raise ValueError("Unsupported DROID policy action protocol")

    def trajectory(self, state, action):
        self.spec.source.wrap(action.values, source=action.source)
        if self.joint_proxy is not None:
            joints, grip, saturated = self.joint_proxy.trajectory(state, action)
            poses = self.joint_proxy.kinematics.poses(joints)
            return poses[:16], grip[:15], saturated
        if self.n17_proxy is not None:
            joints, grip, saturated = self.n17_proxy.trajectory(state, action)
            return self.n17_proxy.kinematics.poses(joints[:16]), grip[:15], saturated
        poses, grip = self.dynamics.pose_trajectory(state, action)
        return poses[:16], grip[:15], 0

    def convert(self, action, *, observation, step_index):
        if observation.robot_state is None:
            raise ValueError("DROID/Ctrl bridge requires current robot state")
        poses, grip, saturated = self.trajectory(observation.robot_state, action)
        state = observation.robot_state.values
        initial_grip = state.get("observation/gripper_position", state.get("gripper_position", state.get("state.gripper_position")))
        values = np.column_stack([poses[[0, 3, 6, 9, 12]],
            np.concatenate([np.asarray(initial_grip).reshape(-1)[-1:], grip[[2, 5, 8, 11]]])])
        return replace(self.spec.target.wrap(values, source=self.spec.name), metadata={
            "step_index": step_index, "source_controller_ticks": 12,
            "waypoint_ticks": [0, 3, 6, 9, 12],
            "idle_tail_steps": max(0, 12 - action.horizon),
            "discarded_replanning_tail_steps": max(0, action.horizon - 12),
            "joint_limit_saturations": saturated,
            "state_assumption": "ideal_controller_target_not_measured_state"})

    def integrate_state(self, previous, action, context):
        if self.joint_proxy is not None:
            joints, grip, _ = self.joint_proxy.trajectory(previous, action)
            pose = self.joint_proxy.kinematics.poses(joints[12:13])[0]
            return {"observation/joint_position": joints[12].astype(np.float32),
                "observation/gripper_position": np.array([grip[11]], dtype=np.float32),
                "state.eef_position": pose[:3].astype(np.float32),
                "state.eef_rotation": pose[3:].astype(np.float32)}
        if self.n17_proxy is not None:
            from adapters.actions.groot_n17_droid import n17_eef_state
            joints, grip, _ = self.n17_proxy.trajectory(previous, action)
            return {"joint_position": joints[12].astype(np.float32),
                "gripper_position": np.array([grip[11]], dtype=np.float32),
                "eef_9d": n17_eef_state(self.n17_proxy.kinematics.poses(joints[12:13]))[0]}
        poses, grip, _ = self.trajectory(previous, action)
        return {"state.eef_position": poses[12, :3].astype(np.float32),
                "state.eef_rotation": poses[12, 3:].astype(np.float32),
                "state.gripper_position": np.array([grip[11]], dtype=np.float32)}

    def pilworld_state(self, previous, action, *, executed_steps, state_schema):
        from evaluation.contracts import RobotState
        if self.n17_proxy is None or executed_steps != 12 or action.semantics != "absolute_target":
            raise ValueError("Strict pilworld/Ctrl endpoint mapper supports only N1.7 absolute targets")
        return RobotState(values=self.integrate_state(previous, action, None),
            schema=state_schema, source="pilworld_action_target", metadata={
                "native_action_index": 11, "selected_action_head": "decoded_absolute_joint_position",
                "target_constraints": "Panda joint limits and gripper [0,1]",
                "eef_state_mapping": "Panda link8 FK, official N1.7 EEF9D encoding"})
