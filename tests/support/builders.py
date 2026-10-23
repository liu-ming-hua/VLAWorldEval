import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adapters.actions.base import build_action_bridge
from adapters.actions.groot_n17_droid import n17_eef_state
from evaluation.contracts import ObservationBundle, RobotState, WorldModelOutput
from evaluation.execution.transport import write_packet
from evaluation.manifests import load_pair_manifest, pair_for_job
from evaluation.state_providers import StateContext

Q = np.array([0, -0.3, 0, -1.7, 0, 1.6, 0.4], dtype=np.float32)


def action_spec():
    return {
        "dimension": 2,
        "horizon": 2,
        "representation": "joint",
        "frame": "robot",
        "semantics": "absolute_target",
        "rate_hz": 10,
        "normalization": "none",
        "components": ["joint_0", "joint_1"],
    }


def manifest_dict():
    action = action_spec()
    return {
        "schema_version": 1,
        "pair_id": "fake_vla__fake_wm",
        "seed": 7,
        "vla": {
            "adapter": "ProtocolVLA",
            "view_mapping": {"camera.front": "front", "camera.wrist": "wrist"},
            "view_mode": "mapping",
            "squeeze_single_frame": True,
            "state_schema": "fake_joint_v1",
            "action": action,
        },
        "action_bridge": {
            "name": "identity",
            "source": action,
            "target": action,
        },
        "world_model": {
            "adapter": "ProtocolWorldModel",
            "view_mapping": {"front": "front", "wrist": "wrist"},
            "view_mode": "mapping",
            "squeeze_single_frame": True,
            "output_views": ["front", "wrist"],
            "action": action,
        },
        "state_provider": {
            "type": "dataset_ground_truth",
            "state_schema": "fake_joint_v1",
        },
    }


def replay_fixture(tmp_path, monkeypatch, *, negative=False):
    import evaluation.execution.native_replay as module
    from evaluation.execution.client import ModelWorker

    monkeypatch.setattr(
        module,
        "get_environment",
        lambda key: SimpleNamespace(scope="world_model", exists=True, python=Path(sys.executable)),
    )

    # Production admission allows only trusted WM package modules. Test the
    # runner with a real persistent fixture worker, not a mock stepping loop.
    def launch(*args, **kwargs):
        kwargs.update(module="tests.support.runtime", adapter="FixtureWM")
        return ModelWorker(*args, **kwargs)

    monkeypatch.setattr(module, "ModelWorker", launch)
    action = {
        "dimension": 2,
        "horizon": 2,
        "representation": "fixture",
        "frame": "fixture",
        "semantics": "fixture",
        "rate_hz": 10,
    }
    config = {
        "schema_version": 1,
        "environment": "fixture",
        "module": "adapters.world_models.fixture",
        "adapter": "FixtureWM",
        "kwargs": {},
        "gpus": [],
        "memory_gib": 1,
        "seed": 0,
        "action": action,
        "video": {"fps": 10, "frames_per_chunk": 2},
        "output_views": ["front", "wrist"],
        "reset_operation": "reset",
        "worker_options": {},
        "instruction_conditioning": False,
    }
    config_path, packet_path = tmp_path / "config.json", tmp_path / "input.npz"
    config_path.write_text(json.dumps(config))
    actions = np.ones((20, 2, 2), dtype=np.float32)
    if negative:
        actions[3] = -1
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    packet = {
        "protocol": action,
        "actions": actions,
        "reset_args": {"value": frame},
        "initial_views": {"front": frame, "wrist": frame},
        "source": {"conversion": "synthetic fixture, not robot evidence"},
    }
    write_packet(packet_path, packet)
    return module, config_path, packet_path, packet


def setup(wm="cosmos"):
    pair = load_pair_manifest(
        f"configs/pairs/groot_n17__{'cosmos3_droid' if wm == 'cosmos' else 'ctrl_world'}.json"
    )
    pair = pair_for_job(pair, {"mode": "pilworld", "state_provider": "pilworld_action_target"})
    bridge = build_action_bridge(pair.action_bridge)
    geometry = bridge.kinematics if wm == "cosmos" else bridge.n17_proxy.kinematics
    state = RobotState(
        values={
            "joint_position": Q,
            "gripper_position": np.zeros(1),
            "eef_9d": n17_eef_state(geometry.poses(Q[None]))[0],
        },
        schema=pair.vla.state_schema,
        source="dataset_ground_truth",
    )
    depth = 1 - min(pair.vla.history_offsets)
    views = {
        name: np.zeros((depth, 16, 16, 3), np.uint8)
        for name in ("droid.exterior_1", "droid.exterior_2", "droid.wrist")
    }
    obs = ObservationBundle(views=views, instruction="move", robot_state=state)
    values = np.zeros((40, 17), np.float32)
    values[:, 10:] = Q
    steps = 16 if wm == "cosmos" else 12
    values[steps - 1, 10] = 0.01
    values[steps:, 10] = 0.4  # Predicted tail is NOT used to update state.
    action = pair.vla.action.wrap(values, source="test")
    converted = bridge.convert(action, observation=obs, step_index=0)
    context = StateContext(
        0,
        obs,
        WorldModelOutput(views={"front": np.zeros((16, 16, 3), np.uint8)}),
        action,
        converted,
        None,
        {},
    )
    return pair, bridge, obs, context
