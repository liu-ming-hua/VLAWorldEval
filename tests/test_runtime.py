from pathlib import Path
import sys

import numpy as np
import pytest

from evaluation.datasets.droid import validate_rows
from evaluation.evaluator import PolicyWorldModelEvaluator, WorldModelReplayEvaluator
from evaluation.execution.executor import build_replay_actions
from evaluation.manifests import PairManifest
from evaluation.execution.client import ModelWorker, RemoteVLA, RemoteWorldModel
from evaluation.execution.transport import read_packet, write_packet
from tests.test_protocol_contracts import manifest_dict


def test_packet_roundtrip_and_rejects_pickle(tmp_path):
    path = tmp_path / "packet.npz"
    value = {"views": {"front": np.ones((1, 4, 4, 3), dtype=np.uint8)}, "state": None}
    write_packet(path, value)
    np.testing.assert_array_equal(read_packet(path)["views"]["front"], value["views"]["front"])
    with pytest.raises(ValueError, match="Object"):
        write_packet(path, np.asarray([object()], dtype=object))


def worker(tmp_path, name):
    return ModelWorker(Path(sys.executable), tmp_path / name, module="tests.runtime_fixtures",
                       adapter=name, kwargs={}, gpus=[], memory_gib=0, timeout=15)


def test_adapter_cleanup_failure_is_not_marked_closed(tmp_path):
    import json
    instance = worker(tmp_path, "BrokenCloseFixtureVLA")
    cleanup = instance.close()
    assert cleanup["returncode"] != 0
    lifecycle = json.loads((instance.directory / "lifecycle.json").read_text())
    assert lifecycle["status"] == "failed"
    assert lifecycle["error"]["code"] == "shutdown_failed"
    assert json.loads((instance.directory / "shutdown.json").read_text())["status"] == "failed"


def test_persistent_workers_drive_twenty_real_ipc_rounds(tmp_path):
    vla, wm = worker(tmp_path, "FixtureVLA"), worker(tmp_path, "FixtureWM")
    pids = (vla.ready["pid"], wm.ready["pid"])
    try:
        manifest = manifest_dict()
        manifest["vla"]["adapter"] = "FixtureVLA"
        manifest["world_model"]["adapter"] = "FixtureWM"
        manifest["state_provider"]["type"] = "hold_initial"
        observations = []
        trace = PolicyWorldModelEvaluator(RemoteVLA(vla), RemoteWorldModel(wm),
                                          pair_manifest=PairManifest.from_dict(manifest)).rollout_detailed({
            "instruction": "move", "observation_bundle": {
                "views": {"front": np.zeros((16, 16, 3), dtype=np.uint8),
                          "wrist": np.zeros((16, 16, 3), dtype=np.uint8)},
                "robot_state": {"joint": [0, 0]},
            },
        }, 20, on_step=lambda current: observations.append(len(current.bridged_actions)))
        assert observations == list(range(1, 21))
        assert trace.primary_video().shape == (41, 16, 16, 3)
        assert trace.native_actions[1].values[0, 0] == 3  # generated first frame=1; second query adds 2
        assert trace.final_state.source == "hold_initial"
        assert (vla.ready["pid"], wm.ready["pid"]) == pids
        assert vla.process.poll() is None and wm.process.poll() is None
        assert not list(vla.directory.glob("request_*.npz"))
        assert not list(wm.directory.glob("response_*.npz"))
    finally:
        vla.close()
        wm.close()
    assert vla.process.poll() is not None and wm.process.poll() is not None


def test_worker_failure_is_visible_and_process_is_reaped(tmp_path):
    wm = worker(tmp_path, "FixtureWM")
    try:
        wm.call("reset", value=np.zeros((16, 16, 3), dtype=np.uint8))
        with pytest.raises(RuntimeError, match="deliberate fixture failure"):
            wm.call("step", action=np.full((2, 2), -1), output_views=["front", "wrist"], state_schema=None)
    finally:
        wm.close()
    assert wm.process.poll() is not None


def test_action_ablations_preserve_valid_rotation_and_gripper():
    values = np.arange(32 * 10, dtype=np.float32).reshape(32, 10)
    values[:, 3:9] = [1, 0, 0, 0, 1, 0]
    chunks = list(np.split(values, 2))
    shuffled = np.concatenate(build_replay_actions(chunks, "shuffle_valid_dataset_actions", 7))
    assert sorted(map(tuple, shuffled)) == sorted(map(tuple, values))
    quiet = np.concatenate(build_replay_actions(chunks, "zero_translation_identity_rotation_preserve_gripper", 0))
    np.testing.assert_array_equal(quiet[:, :3], 0)
    np.testing.assert_array_equal(quiet[:, 3:9], values[:, 3:9])
    np.testing.assert_array_equal(quiet[:, 9], values[:, 9])


def test_failed_replay_retains_last_completed_trace_and_request(tmp_path):
    wm = worker(tmp_path, "FixtureWM")
    completed = []
    try:
        data = manifest_dict()
        data["world_model"]["adapter"] = "FixtureWM"
        trial = {"instruction": "move", "observation_bundle": {
            "views": {name: np.zeros((16, 16, 3), dtype=np.uint8) for name in ("front", "wrist")},
            "robot_state": {"joint": [0, 0]},
        }}
        with pytest.raises(RuntimeError, match="deliberate fixture failure"):
            WorldModelReplayEvaluator(RemoteWorldModel(wm), PairManifest.from_dict(data)).rollout_detailed(
                trial, [np.ones((2, 2)), -np.ones((2, 2))], on_step=completed.append)
        assert len(completed) == 1
        assert len(completed[0].bridged_actions) == 1
        assert completed[0].primary_video().shape[0] == 3
        assert len(list(wm.directory.glob("request_*.npz"))) == 1
    finally:
        wm.close()


def test_data_admission_rejects_tail_and_time_mismatch():
    indices = np.arange(34)
    timestamps = indices / 15
    pose, grip = np.zeros((34, 6)), np.zeros(34)
    validate_rows(indices, timestamps, pose, grip, start=0, chunks=2)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_rows(indices, timestamps, pose, grip, start=2, chunks=2)
    with pytest.raises(ValueError, match="timestamps"):
        validate_rows(indices, timestamps * 2, pose, grip, start=0, chunks=2)


def test_wm_replay_never_queries_a_policy_or_invents_state(tmp_path):
    wm = worker(tmp_path, "FixtureWM")
    try:
        data = manifest_dict()
        data["world_model"]["adapter"] = "FixtureWM"
        manifest = PairManifest.from_dict(data)
        trial = {"instruction": "move", "observation_bundle": {
            "views": {name: np.zeros((16, 16, 3), dtype=np.uint8) for name in ("front", "wrist")},
            "robot_state": {"joint": [0, 0]},
        }}
        trace = WorldModelReplayEvaluator(RemoteWorldModel(wm), manifest).rollout_detailed(
            trial, [np.ones((2, 2))] * 3)
        assert not trace.native_actions
        assert trace.final_state is None
        assert len(trace.bridged_actions) == 3
    finally:
        wm.close()
