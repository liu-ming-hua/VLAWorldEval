from dataclasses import replace

import numpy as np
import pytest

from adapters.actions.base import build_action_bridge
from adapters.observations.droid import split_ctrl_decoded_video
from evaluation.contracts import ObservationBundle, RobotState
from evaluation.manifests import load_pair_manifest
from evaluation.planning.experiments import build_experiment_plan


def test_ctrl_nchw_decode_preserves_color_and_view_order():
    rgb = np.zeros((2, 12, 6, 3), dtype=np.uint8)
    rgb[:, :4, :, 0] = 11
    rgb[:, 4:8, :, 1] = 22
    rgb[:, 8:, :, 2] = 33
    decoded = split_ctrl_decoded_video(rgb.transpose(0, 3, 1, 2))
    assert decoded.shape == (2, 3, 4, 6, 3)
    assert np.all(decoded[:, 0, :, :, 0] == 11)
    assert np.all(decoded[:, 1, :, :, 1] == 22)
    assert np.all(decoded[:, 2, :, :, 2] == 33)
    assert np.all(decoded[:, 0, :, :, 1:] == 0)


@pytest.mark.parametrize("name", ["pi0", "pi05", "groot_n15"])
def test_ctrl_rate_mapping_and_terminal_state(name):
    pair = load_pair_manifest(f"configs/pairs/{name}__ctrl_world.json")
    plan = build_experiment_plan(f"configs/experiments/{name}_ctrl_world_droid_v1.json")
    assert plan["jobs"][1]["duration_seconds"] == 16  # 20 windows advance 12 source ticks each.
    assert plan["jobs"][1]["generated_future_frames"] == 80
    bridge = build_action_bridge(pair.action_bridge)
    q = np.array([0, -0.3, 0, -1.7, 0, 1.6, 0.4], dtype=np.float32)
    state = RobotState(
        values={
            "observation/joint_position": q,
            "observation/gripper_position": np.zeros(1),
            "state.eef_position": np.array([0.5, 0, 0.3]),
            "state.eef_rotation": np.zeros(3),
            "state.gripper_position": np.zeros(1),
        },
        schema=pair.vla.state_schema,
        source="dataset_ground_truth",
    )
    observation = ObservationBundle(
        views={"droid.wrist": np.zeros((8, 8, 3), np.uint8)}, instruction="move", robot_state=state
    )
    values = np.zeros((pair.vla.action.horizon, pair.vla.action.dimension), np.float32)
    action = pair.vla.action.wrap(values, source="test")
    output = bridge.convert(action, observation=observation, step_index=0)
    next_state = bridge.integrate_state(state, action, None)
    assert output.values.shape == (5, 7)
    assert np.allclose(next_state["state.eef_position"], output.values[-1, :3])
    assert np.allclose(
        next_state["state.gripper_position"]
        if name == "groot_n15"
        else next_state["observation/gripper_position"],
        output.values[-1, -1:],
    )


def test_n17_decoded_targets_are_not_integrated_again_and_history_is_exact():
    from adapters.actions.groot_n17_droid import n17_eef_state

    pair = load_pair_manifest("configs/pairs/groot_n17__cosmos3_droid.json")
    bridge = build_action_bridge(pair.action_bridge)
    q = np.array([0, -0.3, 0, -1.7, 0, 1.6, 0.4], dtype=np.float32)
    state = RobotState(
        values={
            "joint_position": q,
            "gripper_position": np.zeros(1),
            "eef_9d": n17_eef_state(bridge.kinematics.poses(q[None]))[0],
        },
        schema=pair.vla.state_schema,
        source="dataset_ground_truth",
    )
    history = np.stack([np.full((8, 8, 3), i, np.uint8) for i in range(16)])
    observation = ObservationBundle(
        views={n: history for n in ("droid.exterior_1", "droid.exterior_2", "droid.wrist")},
        instruction="move",
        robot_state=state,
    )
    selected = pair.vla.adapter_input(observation)
    assert selected["wrist_image_left"].shape[0] == 2
    assert np.all(selected["wrist_image_left"][0] == 0)
    assert np.all(selected["wrist_image_left"][1] == 15)
    with pytest.raises(ValueError, match="lacks required history"):
        pair.vla.adapter_input(
            replace(observation, views={k: v[-1:] for k, v in observation.views.items()})
        )
    values = np.zeros((40, 17), np.float32)
    values[:, 10:] = q
    values[16:, 10:] = q + 0.2  # Discarded, must not affect next state.
    action = pair.vla.action.wrap(values, source="test")
    converted = bridge.convert(action, observation=observation, step_index=0)
    assert np.allclose(converted.values[:, :3], 0)
    assert np.allclose(bridge.integrate_state(state, action, None)["joint_position"], q)
    assert converted.metadata["discarded_replanning_tail_steps"] == 24


def test_n17_ctrl_history_clock_and_executed_terminal_target():
    from adapters.actions.groot_n17_droid import n17_eef_state

    pair = load_pair_manifest("configs/pairs/groot_n17__ctrl_world.json")
    bridge = build_action_bridge(pair.action_bridge)
    q = np.array([0, -0.3, 0, -1.7, 0, 1.6, 0.4], dtype=np.float32)
    state = RobotState(
        values={
            "joint_position": q,
            "gripper_position": np.zeros(1),
            "eef_9d": n17_eef_state(bridge.n17_proxy.kinematics.poses(q[None]))[0],
        },
        schema=pair.vla.state_schema,
        source="dataset_ground_truth",
    )
    values = np.zeros((40, 17), np.float32)
    values[:, 10:] = q
    values[12:, 10:] = q + 0.2
    action = pair.vla.action.wrap(values, source="test")
    history = np.stack([np.full((8, 8, 3), i, np.uint8) for i in range(6)])
    observation = ObservationBundle(
        views={n: history for n in ("droid.exterior_1", "droid.exterior_2", "droid.wrist")},
        instruction="move",
        robot_state=state,
    )
    selected = pair.vla.adapter_input(observation)["wrist_image_left"]
    assert np.all(selected[0] == 0) and np.all(selected[1] == 5)
    output = bridge.convert(action, observation=observation, step_index=0)
    assert output.metadata["discarded_replanning_tail_steps"] == 28
    assert np.allclose(bridge.integrate_state(state, action, None)["joint_position"], q)
