from pathlib import Path

import numpy as np
import pytest

from evaluation.planning.experiments import build_experiment_plan
from evaluation.factories import resolve_workers
from evaluation.manifests import load_pair_manifest
from evaluation.evaluator import PolicyWorldModelEvaluator
from evaluation.execution.client import ModelWorker, RemoteVLA, RemoteWorldModel
from evaluation.datasets.worldgym import load_worldgym_episode
from tests.test_protocol_contracts import manifest_dict
from evaluation.manifests import PairManifest
import sys
import json


@pytest.mark.parametrize("name,horizon", [("openvla", 1), ("spatialvla", 4), ("octo", 4), ("rt1_x", 1)])
def test_worldgym_plans_preserve_native_chunk_and_equal_long_duration(name, horizon):
    plan = build_experiment_plan(f"configs/experiments/{name}_worldgym_v1.json")
    assert plan["executor_available"]
    job = next(job for job in plan["jobs"] if job["id"] == "loop_long")
    assert job["action_count"] == 80
    assert job["chunks"] == 80 // horizon
    assert job["duration_seconds"] == 4
    pair = load_pair_manifest(f"configs/pairs/{name}__worldgym.json")
    vla, wm = resolve_workers(Path.cwd(), pair, plan["runtime"], plan["sampling"], Path("unused"))
    assert vla.adapter == pair.vla.adapter and wm.adapter == "WorldGymAdapter"


def test_official_worldgym_data_rejects_incomplete_replay_but_allows_imagination():
    root = Path("datasets/worldgym/official/bridge/test")
    if not root.is_dir():
        pytest.skip("Official fixture not present")
    sample = load_worldgym_episode(root, 2, 0, 3, 4, 20, instruction="move", replay=True)
    assert len(sample.replay_actions) == 3
    assert sample.replay_actions[0].shape == (4, 10)
    with pytest.raises(ValueError, match="Incomplete"):
        load_worldgym_episode(root, 2, 0, 20, 4, 20, instruction="move", replay=True)
    imagined = load_worldgym_episode(root, 2, 0, 20, 4, 20, instruction="move", replay=False)
    assert imagined.provenance["imagined_beyond_sample"]
    assert imagined.trial["observation_bundle"].get("robot_state") is None


def test_image_only_policy_keeps_state_absent_in_persistent_loop(tmp_path):
    workers = [ModelWorker(Path(sys.executable), tmp_path / name, module="tests.runtime_fixtures",
                           adapter=name, kwargs={}, gpus=[], memory_gib=0, timeout=15)
               for name in ("FixtureVLA", "FixtureWM")]
    try:
        data = manifest_dict()
        data["vla"]["adapter"] = "FixtureVLA"
        data["world_model"]["adapter"] = "FixtureWM"
        data["vla"].pop("state_schema", None)
        data["state_provider"] = {"type": "none", "state_schema": "not_required"}
        trace = PolicyWorldModelEvaluator(RemoteVLA(workers[0]), RemoteWorldModel(workers[1]),
                                          pair_manifest=PairManifest.from_dict(data)).rollout_detailed({
            "instruction": "move", "observation_bundle": {"views": {
                name: np.zeros((16, 16, 3), np.uint8) for name in ("front", "wrist")}}}, 3)
        assert all(observation.robot_state is None for observation in trace.observations)
    finally:
        for worker in reversed(workers):
            worker.close()


def test_none_provider_is_rejected_for_state_dependent_policy():
    data = manifest_dict()
    data["state_provider"] = {"type": "none", "state_schema": "not_required"}
    with pytest.raises(ValueError, match="state-dependent"):
        PairManifest.from_dict(data)


def test_worldgym_rejects_distributed_launch_before_model_load(tmp_path):
    config = json.loads(Path("configs/experiments/openvla_worldgym_v1.json").read_text())
    config["runtime"]["world_model_gpus"] = [1, 2]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="single-GPU"):
        build_experiment_plan(path)
