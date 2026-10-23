"""PiL-World-style feedback is explicit, absolute and terminal-time aligned."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from adapters.actions.base import build_action_bridge
from adapters.actions.groot_n17_droid import n17_eef_state
from adapters.base import VLAAdapter, WorldModelAdapter
from evaluation.contracts import ObservationBundle, RobotState, WorldModelOutput
from evaluation.evaluator import PolicyWorldModelEvaluator
from evaluation.manifests import load_pair_manifest, pair_for_job
from evaluation.modes import pilworld_alignment
from evaluation.planning.experiments import build_experiment_plan
from evaluation.state_providers import PilWorldActionTargetProvider, StateContext


Q = np.array([0, -.3, 0, -1.7, 0, 1.6, .4], dtype=np.float32)


def setup(wm="cosmos"):
    pair = load_pair_manifest(f"configs/pairs/groot_n17__{'cosmos3_droid' if wm == 'cosmos' else 'ctrl_world'}.json")
    pair = pair_for_job(pair, {"mode": "pilworld", "state_provider": "pilworld_action_target"})
    bridge = build_action_bridge(pair.action_bridge)
    geometry = bridge.kinematics if wm == "cosmos" else bridge.n17_proxy.kinematics
    state = RobotState(values={"joint_position": Q, "gripper_position": np.zeros(1),
        "eef_9d": n17_eef_state(geometry.poses(Q[None]))[0]},
        schema=pair.vla.state_schema, source="dataset_ground_truth")
    depth = 1 - min(pair.vla.history_offsets)
    views = {name: np.zeros((depth, 16, 16, 3), np.uint8) for name in
        ("droid.exterior_1", "droid.exterior_2", "droid.wrist")}
    obs = ObservationBundle(views=views, instruction="move", robot_state=state)
    values = np.zeros((40,17), np.float32)
    values[:, 10:] = Q
    steps = 16 if wm == "cosmos" else 12
    values[steps - 1, 10] = .01
    values[steps:, 10] = .4  # Predicted tail is NOT used to update state.
    action = pair.vla.action.wrap(values, source="test")
    converted = bridge.convert(action, observation=obs, step_index=0)
    context = StateContext(0, obs, WorldModelOutput(views={"front": np.zeros((16,16,3), np.uint8)}),
                           action, converted, None, {})
    return pair, bridge, obs, context


@pytest.mark.parametrize("wm,stride,index", [("cosmos",1,15),("ctrl",3,11)])
def test_endpoint_is_last_aligned_target_not_native_chunk_tail(wm, stride, index):
    pair, bridge, _, context = setup(wm)
    provider = PilWorldActionTargetProvider(pair.state_provider, bridge.pilworld_state)
    state = provider.provide(context)
    assert state.source == "pilworld_action_target"
    assert state.values["joint_position"][0] == pytest.approx(.01)
    assert state.metadata["native_action_index"] == index
    assert state.metadata["action_stride"] == stride
    assert state.metadata["wm_predicted_state"] is False
    assert state.metadata["evaluation_mode"] == "pilworld"
    # Existing N1.7 target proxy has the same numeric state; this mode makes
    # the assumption/endpoint auditable, not a claimed new prediction model.
    existing = bridge.integrate_state(context.previous_observation.robot_state, context.native_action, None)
    for key in state.values:
        assert np.array_equal(state.values[key], existing[key])
    plan = build_experiment_plan(f"configs/experiments/pilworld_groot_n17_{wm}_v1.json")
    assert [job["mode"] for job in plan["jobs"]] == ["pilworld", "pilworld"]


@pytest.mark.parametrize("bad", ["velocity", "noninteger_stride", "missing_rate", "too_short"])
def test_invalid_pilworld_action_protocol_rejected(bad):
    _, _, _, context = setup()
    native, wm = context.native_action, context.bridged_action
    if bad == "velocity":
        native = replace(native, semantics="delta")
    elif bad == "noninteger_stride":
        wm = replace(wm, rate_hz=7)
    elif bad == "missing_rate":
        native = replace(native, rate_hz=None)
    else:
        native = replace(native, values=native.values[:5])
    with pytest.raises(ValueError):
        pilworld_alignment(native, wm)


def test_endpoint_mapper_cannot_mislabel_state_or_use_future_target():
    pair, bridge, _, context = setup()
    with pytest.raises(ValueError, match="explicit"):
        PilWorldActionTargetProvider(pair.state_provider)
    def wrong_endpoint(previous, action, **kwargs):
        state = bridge.pilworld_state(previous, action, **kwargs)
        return replace(state, metadata={"native_action_index": 39})
    with pytest.raises(ValueError, match="not aligned"):
        PilWorldActionTargetProvider(pair.state_provider, wrong_endpoint).provide(context)
    def wrong_source(previous, action, **kwargs):
        return replace(bridge.pilworld_state(previous, action, **kwargs), source="world_model_prediction")
    with pytest.raises(ValueError, match="schema/source"):
        PilWorldActionTargetProvider(pair.state_provider, wrong_source).provide(context)


@pytest.mark.parametrize("mode,provider", [("pilworld","integrated_action_target"),
                                           ("policy_loop","pilworld_action_target")])
def test_mode_and_state_feedback_cannot_be_mixed(tmp_path, mode, provider):
    value = json.loads(Path("configs/experiments/pilworld_groot_n17_cosmos_v1.json").read_text())
    value["experiments"][0].update(mode=mode, state_provider=provider)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="vice versa"):
        build_experiment_plan(path)


def test_twenty_chunk_pilworld_loop_preserves_generated_history_and_sources():
    pair, _, obs, _ = setup()
    class Policy(VLAAdapter):
        adapter_identity = "GrootN17Adapter"
        def __init__(self):
            self.inputs = []
        def reset(self, initial_state=None):
            pass
        def predict(self, image, instruction, state=None):
            self.inputs.append(image["wrist_image_left"].copy())
            values = np.zeros((40,17), np.float32)
            values[:, 10:] = state["joint_position"]
            return values
    class World(WorldModelAdapter):
        adapter_identity = "Cosmos3NanoDroidFDAdapter"
        def reset(self, image):
            self.shape, self.count = image.shape, 0
        def step(self, action):
            raise NotImplementedError
        def step_output(self, action, **kwargs):
            self.count += 1
            frames = np.stack([np.full(self.shape, (16*(self.count-1)+i+1)%255, np.uint8) for i in range(16)])
            return WorldModelOutput(views={"droid.cosmos_composite": frames})
    policy = Policy()
    trace = PolicyWorldModelEvaluator(policy, World(), pair_manifest=pair).rollout_detailed(
        {"observation_bundle": obs, "instruction": "move"}, 20)
    assert len(trace.native_actions) == len(trace.bridged_actions) == 20
    assert trace.primary_video().shape[0] == 321
    assert np.all(policy.inputs[1][0] == 1) and np.all(policy.inputs[1][1] == 16)
    assert trace.observations[0].robot_state.source == "dataset_ground_truth"
    assert all(x.robot_state.source == "pilworld_action_target" for x in trace.observations[1:])
