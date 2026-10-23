"""CPU checks of the unmodified official FK using actual observed joints."""
from pathlib import Path
import json

import numpy as np
import pytest

from evaluation.datasets.iws import official_fk_actions

pytest.importorskip("sapien", reason="Official FK dependencies belong to the IWS environment")
ROOT = Path(__file__).resolve().parents[1]


def test_real_observed_joints_match_independent_urdf_fk_and_world_transform():
    import h5py
    from yixuan_utilities.kinematics_helper import KinHelper
    from interactive_world_sim.utils.aloha_conts import (
        MASTER_GRIPPER_JOINT_UNNORMALIZE_FN, PUPPET_GRIPPER_JOINT_NORMALIZE_FN,
    )

    with h5py.File(ROOT / "datasets/iws/pusht/val/episode_0.hdf5") as f:
        joints = f["obs/joint_pos"][:21]
        bases = f["obs/world_t_robot_base"][:21]
    actual, audit = official_fk_actions(joints, bases)
    kin = KinHelper("trossen_vx300s")
    expected = []
    names = [joint.name for joint in kin.sapien_robot.get_active_joints()]
    eef = next(link for link in kin.urdf_robot.links if link.name == "vx300s/ee_arm_link")
    for joint, base in zip(joints, bases):
        xy = []
        for arm in range(2):
            q = joint[arm * 7:(arm + 1) * 7].copy()
            q[-1] = MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(PUPPET_GRIPPER_JOINT_NORMALIZE_FN(q[-1]))
            q = np.concatenate([q, q[-1:]])
            # Independent URDF graph FK, not KinHelper or the action commands.
            robot_t_eef = kin.urdf_robot.link_fk(cfg=dict(zip(names, q)))[eef].copy()
            robot_t_eef[0, 3] = np.clip(robot_t_eef[0, 3], 0.25, 1.0)
            robot_t_eef[1, 3] = np.clip(robot_t_eef[1, 3], -0.25, 0.25)
            xy.extend((base[arm] @ robot_t_eef)[:2, 3])
        expected.append(xy)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0)
    assert audit["robot"] == "trossen_vx300s"
    # Perturb real observed joints and world bases: results must respond.
    moved_joints = joints[:1].copy()
    moved_joints[0, 0] += 0.1
    changed, _ = official_fk_actions(moved_joints, bases[:1])
    assert not np.allclose(changed, actual[:1], atol=1e-5)
    moved_base = bases[:1].copy()
    moved_base[0, 0, 0, 3] += 0.123
    translated, _ = official_fk_actions(joints[:1], moved_base)
    np.testing.assert_allclose(translated - actual[:1], [[0.123, 0, 0, 0]], atol=1e-6)
    print(json.dumps({"observed_frames": len(joints), "joint_field": "obs/joint_pos",
                      "transform_field": "obs/world_t_robot_base", "stored_commands_used": False,
                      "independent_engine": "urdfpy.URDF.link_fk",
                      "max_abs_world_xy_error_metres": float(np.max(np.abs(actual - expected))),
                      "joint_perturbation_changed_output": bool(not np.allclose(changed, actual[:1], atol=1e-5)),
                      "world_translation_error_metres": float(np.max(np.abs(translated - actual[:1] - [[0.123, 0, 0, 0]])))}, indent=2))


@pytest.mark.parametrize("joints,bases", [
    (np.zeros((1, 7)), np.tile(np.eye(4), (1, 2, 1, 1))),
    (np.zeros((1, 14)), np.zeros((1, 2, 4, 4))),
    (np.full((1, 14), np.nan), np.tile(np.eye(4), (1, 2, 1, 1))),
])
def test_invalid_observed_fk_inputs_fail(joints, bases):
    with pytest.raises(ValueError):
        official_fk_actions(joints, bases)
