import json

import numpy as np
import pytest

from adapters.actions.base import IdentityActionBridge, build_action_bridge
from adapters.actions.droid import poses_to_cosmos_droid_action
from adapters.base import VLAAdapter, WorldModelAdapter
from adapters.observations.droid import compose_droid_cosmos_view, split_droid_cosmos_view
from adapters.world_models.cosmos3_nano_droid_fd import Cosmos3NanoDroidFDAdapter
from evaluation.reporting.artifacts import pack_states, save_rollout_artifacts
from evaluation.contracts import ActionChunk, ObservationBundle, RobotState
from evaluation.evaluator import PolicyWorldModelEvaluator
from evaluation.manifests import PairManifest, load_pair_manifest
from evaluation.state_providers import DatasetGroundTruthProvider


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


class ProtocolVLA(VLAAdapter):
    def reset(self, initial_state=None):
        self.initial_state = initial_state
        self.calls = 0

    def predict(self, image, instruction, state=None):
        assert set(image) == {"camera.front", "camera.wrist"}
        assert image["camera.front"].shape == (16, 16, 3)
        assert instruction == "move"
        assert set(state) == {"joint"}
        self.calls += 1
        return np.full((2, 2), self.calls, dtype=np.float32)


class ProtocolWorldModel(WorldModelAdapter):
    def reset(self, image):
        assert set(image) == {"front", "wrist"}
        self.calls = 0
        self.last_video = None

    def step(self, action):
        self.calls += 1
        assert np.asarray(action).shape == (2, 2)
        front = np.full((2, 16, 16, 3), self.calls, dtype=np.uint8)
        wrist = np.full((2, 16, 16, 3), self.calls + 10, dtype=np.uint8)
        self.last_video = np.stack([front, wrist], axis=1)
        return front[-1]


def test_contracts_normalize_and_validate_shapes():
    state = RobotState.from_value(
        {"joint": [1.0, 2.0]}, schema="joint_v1", source="dataset_ground_truth"
    )
    observation = ObservationBundle(
        views={"front": np.ones((8, 8, 3), dtype=np.float32)},
        instruction="move",
        robot_state=state,
    )
    action = ActionChunk.from_value(
        [1.0, 2.0],
        representation="joint",
        frame="robot",
        semantics="absolute_target",
        source="test",
    )

    assert observation.views["front"].shape == (1, 8, 8, 3)
    assert observation.views["front"].dtype == np.uint8
    assert action.values.shape == (1, 2)
    with pytest.raises(ValueError, match="RGB"):
        ObservationBundle(views={"bad": np.zeros((8, 8))}, instruction="move")


def test_droid_cosmos_view_bridge_roundtrip_layout():
    exterior_1 = np.full((2, 12, 16, 3), 10, dtype=np.uint8)
    exterior_2 = np.full((2, 12, 16, 3), 20, dtype=np.uint8)
    wrist = np.full((2, 12, 16, 3), 30, dtype=np.uint8)

    composite = compose_droid_cosmos_view(exterior_1, exterior_2, wrist)
    split = split_droid_cosmos_view(composite)

    assert composite.shape == (2, 18, 16, 3)
    assert split["wrist"].shape == wrist.shape
    assert split["exterior_1"].shape == exterior_1.shape
    assert np.all(split["wrist"] == 30)
    assert np.all(split["exterior_1"] == 10)
    assert np.all(split["exterior_2"] == 20)


def test_droid_action_bridge_uses_versioned_dynamics_and_official_layout():
    manifest = load_pair_manifest("configs/pairs/groot_n15__cosmos3_droid.json")
    bridge = build_action_bridge(manifest.action_bridge)
    state = RobotState(
        values={
            "state.eef_position": np.asarray([0.5, 0.0, 0.3]),
            "state.eef_rotation": np.asarray([0.1, -0.2, 0.3]),
            "state.gripper_position": np.asarray([0.0]),
        },
        schema="droid_eef_rpy_gripper_v1",
        source="dataset_ground_truth",
    )
    observation = ObservationBundle(
        views={"front": np.zeros((16, 16, 3), dtype=np.uint8)},
        instruction="move",
        robot_state=state,
    )
    native = manifest.vla.action.wrap(np.zeros((16, 7), dtype=np.float32), source="test")
    converted = bridge.convert(native, observation=observation, step_index=0)
    integrated = bridge.integrate_state(state, native, None)

    assert converted.values.shape == (16, 10)
    assert np.isfinite(converted.values).all()
    assert converted.normalization == "none_raw_metric"
    assert converted.metadata["action_normalization"] == "none"
    assert converted.metadata["normalizer_stats"] is None
    assert set(integrated) == {
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    }

    repeated_pose = np.repeat(np.asarray([[0.5, 0.0, 0.3, 0.1, -0.2, 0.3]]), 3, axis=0)
    raw = poses_to_cosmos_droid_action(repeated_pose, np.asarray([0.0, 1.0]))
    np.testing.assert_allclose(raw[:, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        raw[:, 3:9], np.tile([1, 0, 0, 0, 1, 0], (2, 1)), atol=1e-6
    )
    np.testing.assert_allclose(raw[:, 9], [1.0, 0.0])


def test_cosmos_droid_fd_rejects_non_official_action_shape_without_loading_model():
    action = Cosmos3NanoDroidFDAdapter.validate_action(np.zeros((16, 10), dtype=np.float32))
    assert action.shape == (16, 10)
    with pytest.raises(ValueError, match="exactly raw action"):
        Cosmos3NanoDroidFDAdapter.validate_action(np.zeros((15, 10), dtype=np.float32))


def test_cosmos_droid_fd_checkpoint_static_contract(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": {"config": {
        "action_gen": True, "max_action_dim": 64,
        "tokenizer": {"encode_exact_durations": [17]},
    }}}))
    weights = {f"layer_{i}": f"shard_{i}.safetensors" for i in range(7)}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weights}))
    for name in weights.values():
        (tmp_path / name).touch()
    contract = Cosmos3NanoDroidFDAdapter.validate_checkpoint_contract(
        tmp_path
    )
    assert contract == {
        "action_gen": True,
        "max_action_dim": 64,
        "encode_exact_durations": [17],
        "num_shards": 7,
        "action_shape": [16, 10],
    }


def test_pair_policy_resize_is_explicit_and_does_not_mutate_observation():
    manifest = load_pair_manifest("configs/pairs/groot_n15__cosmos3_droid.json")
    views = {name: np.zeros((360, 640, 3), dtype=np.uint8)
             for name in manifest.vla.required_bundle_views}
    observation = ObservationBundle(views=views, instruction="move")
    inputs = manifest.vla.adapter_input(observation)
    assert all(image.shape == (256, 256, 3) for image in inputs.values())
    assert all(image.shape == (1, 360, 640, 3) for image in observation.views.values())


def test_droid_bridge_removes_official_bottom_right_padding():
    from adapters.observations.base import build_view_bridge
    from evaluation.contracts import WorldModelOutput

    manifest = load_pair_manifest("configs/pairs/groot_n15__cosmos3_droid.json")
    bridge = build_view_bridge(manifest.view_bridge)
    observation = ObservationBundle(views={
        "droid.exterior_1": np.full((12, 16, 3), 10, dtype=np.uint8),
        "droid.exterior_2": np.full((12, 16, 3), 20, dtype=np.uint8),
        "droid.wrist": np.full((12, 16, 3), 30, dtype=np.uint8),
    }, instruction="move")
    prepared = bridge.prepare_observation(observation)
    composite = prepared.views["droid.cosmos_composite"]
    padded = np.pad(composite, ((0, 0), (0, 4), (0, 8), (0, 0)), mode="reflect")
    output = bridge.process_world_output(WorldModelOutput(views={"droid.cosmos_composite": padded}))
    np.testing.assert_array_equal(output.views["droid.cosmos_composite"], composite)
    np.testing.assert_array_equal(output.views["droid.wrist"], prepared.views["droid.wrist"])


def test_flat_delta_provider_integrates_all_executed_actions():
    from evaluation.manifests import StateProviderSpec
    from evaluation.contracts import WorldModelOutput
    from evaluation.state_providers import IntegratedActionTargetProvider, StateContext

    action = ActionChunk.from_value([[1, 2], [3, 4]], representation="joint",
                                    frame="robot", semantics="delta", source="test")
    state = RobotState.from_value([10, 20], schema="joint_v1", source="dataset_ground_truth")
    context = StateContext(0, ObservationBundle(
        views={"front": np.zeros((2, 2, 3), dtype=np.uint8)}, instruction="move", robot_state=state,
    ), WorldModelOutput(views={"front": np.zeros((2, 2, 3), dtype=np.uint8)}),
        action, action, None, {})
    provider = IntegratedActionTargetProvider(StateProviderSpec("integrated_action_target", "joint_v1"))
    np.testing.assert_allclose(provider.provide(context).values["state"], [14, 26])


def test_droid_bridge_preserves_camera_boundary_after_latent_crop():
    from adapters.observations.base import build_view_bridge
    from evaluation.contracts import WorldModelOutput

    manifest = load_pair_manifest("configs/pairs/groot_n15__cosmos3_droid.json")
    bridge = build_view_bridge(manifest.view_bridge)
    original = ObservationBundle(views={
        "droid.exterior_1": np.full((36, 64, 3), 10, dtype=np.uint8),
        "droid.exterior_2": np.full((36, 64, 3), 20, dtype=np.uint8),
        "droid.wrist": np.full((36, 64, 3), 30, dtype=np.uint8),
    }, instruction="move")
    prepared = bridge.prepare_observation(original)
    latent_cropped = prepared.views["droid.cosmos_composite"][:, :48]
    output = bridge.process_world_output(WorldModelOutput(views={"droid.cosmos_composite": latent_cropped}))
    np.testing.assert_array_equal(output.views["droid.wrist"], prepared.views["droid.wrist"])
    assert output.views["droid.cosmos_composite"].shape == (1, 54, 64, 3)


def test_manifest_roundtrip_and_rejects_hidden_action_mismatch(tmp_path):
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(manifest_dict()), encoding="utf-8")
    manifest = load_pair_manifest(path)

    assert manifest.pair_id == "fake_vla__fake_wm"
    assert manifest.state_provider.type == "dataset_ground_truth"
    bad = manifest_dict()
    bad["action_bridge"]["target"] = {
        **action_spec(),
        "dimension": 3,
        "components": ["joint_0", "joint_1", "joint_2"],
    }
    with pytest.raises(ValueError, match="World-model action spec"):
        PairManifest.from_dict(bad)


def test_protocol_rollout_records_views_actions_and_state_sources():
    manifest = PairManifest.from_dict(manifest_dict())
    bridge = IdentityActionBridge(manifest.action_bridge)
    state_provider = DatasetGroundTruthProvider(manifest.state_provider)
    evaluator = PolicyWorldModelEvaluator(
        ProtocolVLA(),
        ProtocolWorldModel(),
        pair_manifest=manifest,
        action_bridge=bridge,
        state_provider=state_provider,
    )
    initial_state = RobotState.from_value(
        {"joint": [0.0, 0.0]},
        schema="fake_joint_v1",
        source="dataset_ground_truth",
    )
    trial = {
        "instruction": "move",
        "observation_bundle": ObservationBundle(
            views={
                "front": np.zeros((16, 16, 3), dtype=np.uint8),
                "wrist": np.zeros((16, 16, 3), dtype=np.uint8),
            },
            instruction="move",
            robot_state=initial_state,
        ),
        "state_sequence": {"joint": np.asarray([[0, 0], [1, 2], [3, 4]])},
    }

    trace = evaluator.rollout_detailed(trial, rollout_length=2)

    assert len(trace.native_actions) == len(trace.bridged_actions) == 2
    assert trace.observations[-1].robot_state.source == "dataset_ground_truth"
    assert trace.observations[-1].robot_state.values["joint"].tolist() == [3.0, 4.0]
    assert set(trace.view_videos()) == {"front", "wrist"}
    assert trace.primary_video().shape == (5, 16, 16, 3)


def test_artifact_v2_saves_native_bridged_actions_and_states(tmp_path):
    states = [
        RobotState.from_value(
            {"joint": [0, 1]}, schema="joint_v1", source="dataset_ground_truth"
        ),
        RobotState.from_value(
            {"joint": [1, 2]}, schema="joint_v1", source="last_action_target"
        ),
    ]
    video = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    paths = save_rollout_artifacts(
        tmp_path,
        video,
        [np.ones((2, 2), dtype=np.float32)],
        {"instruction": "move"},
        native_action_chunks=[np.zeros((2, 2), dtype=np.float32)],
        states=states,
        view_videos={"front": video},
        pair_manifest={"pair_id": "test"},
        timings=[{"policy_seconds": 0.1, "world_model_seconds": 0.2}],
    )

    actions = np.load(paths["action_path"])
    packed = np.load(paths["state_path"])
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert actions["native_actions"].shape == (1, 2, 2)
    assert actions["bridged_actions"].shape == (1, 2, 2)
    assert packed["state_sources"].tolist() == [
        "dataset_ground_truth",
        "last_action_target",
    ]
    assert metadata["format_version"] == 2
    assert metadata["pair_manifest"]["pair_id"] == "test"
    assert (tmp_path / "views/front.mp4").is_file()


def test_pack_states_keeps_uncertainty_aligned():
    state = RobotState(
        values={"joint": np.asarray([1.0, 2.0])},
        uncertainty={"joint": np.asarray([0.1, 0.2])},
        schema="joint_v1",
        source="state_completion_prediction",
    )
    packed = pack_states([None, state])
    assert np.isnan(packed["state_uncertainty"][0]).all()
    np.testing.assert_allclose(packed["state_uncertainty"][1], [0.1, 0.2])
