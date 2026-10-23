"""OpenPI joint protocol, idle tail, FK and recurrent proxy contracts."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from adapters.actions.base import build_action_bridge
from adapters.actions.droid import _euler_xyz_matrix, poses_to_cosmos_droid_action
from evaluation.contracts import ObservationBundle, RobotState
from evaluation.manifests import load_pair_manifest
from evaluation.planning.experiments import build_experiment_plan

Q = np.array([0, -0.3, 0, -1.7, 0, 1.6, 0.4], dtype=np.float32)


def setup(name="pi0"):
    pair = load_pair_manifest(f"configs/pairs/{name}__cosmos3_droid.json")
    bridge = build_action_bridge(pair.action_bridge)
    pose = bridge.kinematics.poses(Q[None])[0]
    state = RobotState(values={
        "observation/joint_position": Q,
        "observation/gripper_position": np.array([0.0]),
        "state.eef_position": pose[:3], "state.eef_rotation": pose[3:],
    }, schema=pair.vla.state_schema, source="dataset_ground_truth")
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    observation = ObservationBundle(views={name: image for name in
        ("droid.exterior_1", "droid.exterior_2", "droid.wrist")},
        instruction="move", robot_state=state)
    return pair, bridge, observation


@pytest.mark.parametrize("name,horizon", [("pi0", 10), ("pi05", 15)])
def test_idle_tail_does_not_repeat_velocity(name, horizon):
    pair, bridge, observation = setup(name)
    values = np.zeros((horizon, 8), dtype=np.float32)
    values[:, 0] = 0.01
    values[:, 7] = 1
    action = pair.vla.action.wrap(values, source="test")
    joints, grip, _ = bridge.trajectory(observation.robot_state, action)
    assert joints[horizon, 0] == pytest.approx(Q[0] + horizon * 0.002)
    assert np.array_equal(joints[horizon:], np.repeat(joints[horizon:horizon + 1], 17 - horizon, axis=0))
    converted = bridge.convert(action, observation=observation, step_index=0)
    assert converted.values.shape == (16, 10)
    assert converted.metadata["idle_tail_steps"] == 16 - horizon
    assert np.allclose(converted.values[horizon:, :3], 0)
    assert np.allclose(converted.values[horizon:, 3:9], [1, 0, 0, 0, 1, 0])
    assert np.all(converted.values[:, 9] == 0)  # OpenPI 1=closed -> Cosmos flipped.
    state = bridge.integrate_state(observation.robot_state, action, None)
    assert np.allclose(state["observation/joint_position"], joints[-1])
    assert state["observation/gripper_position"].item() == 1
    plan = build_experiment_plan(f"configs/experiments/{name}_cosmos_droid_v1.json")
    assert plan["dataset_loader"] == "droid_openpi"
    assert plan["jobs"][0]["action_count"] == 32


def test_controller_clip_binarize_and_joint_limit_saturation():
    pair, bridge, observation = setup()
    values = np.zeros((10, 8), dtype=np.float32)
    values[:, 0] = 100
    values[:, 7] = 0.5
    action = pair.vla.action.wrap(values, source="test")
    joints, grip, _ = bridge.trajectory(observation.robot_state, action)
    assert joints[1, 0] == pytest.approx(0.2)  # NOT command / 15.
    assert np.all(grip == 0)
    altered = RobotState(values={**observation.robot_state.values,
        "observation/joint_position": np.array([2.89, *Q[1:]])},
        schema=pair.vla.state_schema, source="dataset_ground_truth")
    joints, _, count = bridge.trajectory(altered, action)
    assert joints[-1, 0] == pytest.approx(2.8973)
    assert count > 0


def test_fk_round_trip_and_shared_cosmos_conversion():
    pair, bridge, observation = setup()
    poses = bridge.kinematics.poses(np.stack([Q, Q + 0.001]))
    transforms = bridge.kinematics.transforms(np.stack([Q, Q + 0.001]))
    assert np.allclose(_euler_xyz_matrix(poses[:, 3:]), transforms[:, :3, :3], atol=1e-8)
    assert np.allclose(poses_to_cosmos_droid_action(poses, [0])[:, -1], 1)
    with pytest.raises(ValueError, match="finite"):
        bridge.kinematics.transforms(np.full((1, 7), np.nan))
    with pytest.raises(ValueError, match="idle tail"):
        build_action_bridge(replace(pair.action_bridge, config={**pair.action_bridge.config, "tail_policy": "repeat_last"}))


def test_official_episode_fk_admission_when_assets_exist():
    root = Path("datasets/cosmos3_droid/success")
    if not root.is_dir():
        pytest.skip("Official DROID episode assets not installed")
    pytest.importorskip("pyarrow")
    from evaluation.datasets.droid import load_openpi_droid_episode
    packet = load_openpi_droid_episode(root, 0, 30, 2)
    evidence = packet.provenance["kinematics_validation"]
    assert evidence["rows_checked"] > 32
    assert evidence["max_position_error_m"] < 0.002
    state = packet.trial["initial_state"]
    assert state.source == "dataset_ground_truth"
    assert state.values["observation/joint_position"].shape == (7,)


@pytest.mark.parametrize("name", ["pi0", "pi05"])
def test_public_evaluator_state_and_generated_view_handoff(name):
    from adapters.base import VLAAdapter, WorldModelAdapter
    from evaluation.contracts import WorldModelOutput
    from evaluation.evaluator import PolicyWorldModelEvaluator

    pair, _, initial = setup(name)

    class Policy(VLAAdapter):
        adapter_identity = pair.vla.adapter

        def __init__(self):
            self.states, self.images = [], []

        def reset(self, initial_state=None):
            pass

        def predict(self, image, instruction, state=None):
            self.states.append(state["observation/joint_position"].copy())
            self.images.append(image["observation/wrist_image_left"].copy())
            values = np.zeros((pair.vla.action.horizon, 8), dtype=np.float32)
            values[:, 0] = 0.01
            return values

    class World(WorldModelAdapter):
        adapter_identity = pair.world_model.adapter

        def reset(self, image):
            self.shape = image.shape

        def step(self, action):
            raise NotImplementedError

        def step_output(self, action, **kwargs):
            return WorldModelOutput(views={"droid.cosmos_composite":
                np.full((16, *self.shape), 42, dtype=np.uint8)})

    policy = Policy()
    trace = PolicyWorldModelEvaluator(policy, World(), pair_manifest=pair).rollout_detailed(
        {"observation_bundle": initial, "instruction": "move"}, 3)
    assert len(trace.native_actions) == len(trace.bridged_actions) == 3
    assert trace.primary_video().shape[0] == 49
    assert np.all(policy.images[1] == 42)  # Generated, not dataset future views.
    assert policy.states[1][0] > policy.states[0][0]
    assert trace.observations[0].robot_state.source == "dataset_ground_truth"
    assert all(o.robot_state.source == "integrated_action_target" for o in trace.observations[1:])
    assert all(a.metadata["idle_tail_steps"] == 16 - pair.vla.action.horizon for a in trace.bridged_actions)
