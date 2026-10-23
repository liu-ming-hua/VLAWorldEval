import json
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.evaluator import PolicyWorldModelEvaluator
from evaluation.execution.client import ModelWorker, RemoteVLA, RemoteWorldModel
from evaluation.factories import resolve_workers
from evaluation.manifests import PairManifest, load_pair_manifest
from evaluation.planning.experiments import build_experiment_plan
from tests.support.builders import manifest_dict

pytestmark = pytest.mark.process


@pytest.mark.parametrize(
    "name,horizon", [("openvla", 1), ("spatialvla", 4), ("octo", 4), ("rt1_x", 1)]
)
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


def test_image_only_policy_keeps_state_absent_in_persistent_loop(tmp_path):
    workers = [
        ModelWorker(
            Path(sys.executable),
            tmp_path / name,
            module="tests.support.runtime",
            adapter=name,
            kwargs={},
            gpus=[],
            memory_gib=0,
            timeout=15,
        )
        for name in ("FixtureVLA", "FixtureWM")
    ]
    try:
        data = manifest_dict()
        data["vla"]["adapter"] = "FixtureVLA"
        data["world_model"]["adapter"] = "FixtureWM"
        data["vla"].pop("state_schema", None)
        data["state_provider"] = {"type": "none", "state_schema": "not_required"}
        trace = PolicyWorldModelEvaluator(
            RemoteVLA(workers[0]),
            RemoteWorldModel(workers[1]),
            pair_manifest=PairManifest.from_dict(data),
        ).rollout_detailed(
            {
                "instruction": "move",
                "observation_bundle": {
                    "views": {name: np.zeros((16, 16, 3), np.uint8) for name in ("front", "wrist")}
                },
            },
            3,
        )
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
