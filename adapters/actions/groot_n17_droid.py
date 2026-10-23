"""GR00T N1.7 decoded absolute joint targets -> DROID Cosmos actions."""
import numpy as np

from adapters.actions.base import ActionBridge
from adapters.actions.droid import _euler_xyz_matrix, poses_to_cosmos_droid_action
from adapters.actions.openpi_droid import PandaKinematics


N17_ROTATION_CORRECTION = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)


def n17_eef_state(poses):
    """Exact upstream DROID downloader encoding: intrinsic XYZ, first 2 rows."""
    poses = np.asarray(poses, dtype=np.float64)
    rotation = _euler_xyz_matrix(-poses[..., 3:]).swapaxes(-1, -2) @ N17_ROTATION_CORRECTION
    return np.concatenate([poses[..., :3], rotation[..., :2, :].reshape(*poses.shape[:-1], 6)], axis=-1).astype(np.float32)


class GrootN17CosmosDroidActionBridge(ActionBridge):
    """Execute the first 16 decoded joint targets, not the redundant EEF head.

    The official processor already restores relative training actions to
    absolute targets. Remaining 24 predictions are discarded on replanning.
    """

    def __init__(self, spec):
        super().__init__(spec)
        if (spec.source.horizon != 40 or spec.source.dimension != 17
                or spec.source.semantics != "absolute_target"
                or spec.target.horizon != 16 or spec.target.dimension != 10
                or spec.config.get("execute_prefix") != 16):
            raise ValueError("N1.7/Cosmos requires decoded [40,17] targets and explicit 16-step prefix")
        self.kinematics = PandaKinematics(spec.config["kinematics"])

    def trajectory(self, previous, action):
        self.spec.source.wrap(action.values, source=action.source)
        q0 = np.asarray(previous.values["joint_position"]).reshape(-1, 7)[-1]
        targets = action.values[:16, 10:17]
        bounded = np.clip(targets, self.kinematics.lower, self.kinematics.upper)
        joints = np.concatenate([q0[None], bounded])
        gripper = np.clip(action.values[:16, 9], 0, 1)
        return joints, gripper, int(np.count_nonzero(targets != bounded))

    def convert(self, action, *, observation, step_index):
        from dataclasses import replace
        if observation.robot_state is None:
            raise ValueError("GR00T N1.7 bridge requires current joint state")
        joints, gripper, saturated = self.trajectory(observation.robot_state, action)
        values = poses_to_cosmos_droid_action(self.kinematics.poses(joints), gripper)
        return replace(self.spec.target.wrap(values, source=self.spec.name), metadata={
            "executed_action_steps": 16, "discarded_replanning_tail_steps": 24,
            "selected_action_head": "decoded_absolute_joint_position",
            "unused_redundant_head": "decoded_absolute_eef_9d",
            "joint_limit_saturations": saturated, "step_index": step_index,
            "state_assumption": "ideal_joint_target_not_measured_state"})

    def integrate_state(self, previous, action, context):
        joints, gripper, _ = self.trajectory(previous, action)
        return {"joint_position": joints[-1].astype(np.float32),
                "gripper_position": np.array([gripper[-1]], dtype=np.float32),
                "eef_9d": n17_eef_state(self.kinematics.poses(joints[-1:]))[0]}

    def pilworld_state(self, previous, action, *, executed_steps, state_schema):
        from evaluation.contracts import RobotState
        if executed_steps != 16 or action.semantics != "absolute_target":
            raise ValueError("N1.7/Cosmos endpoint mapper requires 16 absolute target steps")
        return RobotState(values=self.integrate_state(previous, action, None),
            schema=state_schema, source="pilworld_action_target", metadata={
                "native_action_index": 15, "selected_action_head": "decoded_absolute_joint_position",
                "target_constraints": "Panda joint limits and gripper [0,1]",
                "eef_state_mapping": "Panda link8 FK, official N1.7 EEF9D encoding"})
