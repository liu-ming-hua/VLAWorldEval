#!/usr/bin/env python3
"""Real OpenPI inference + bridge/proxy check; NO world-model video generation."""
import argparse
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    from adapters.actions.base import build_action_bridge
    from adapters.environment import load_environment_registry
    from evaluation.contracts import ObservationBundle, RobotState
    from evaluation.datasets.droid import load_openpi_droid_episode
    from evaluation.execution.admission import resource_admission
    from evaluation.execution.client import ModelWorker, RemoteVLA
    from evaluation.manifests import load_pair_manifest
    from evaluation.planning.registry import project_path
    from vlaworldeval.atomic_json import publish_json_exclusive
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi0", "pi05"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = project_path(ROOT, args.output)
    pair = load_pair_manifest(ROOT / f"configs/pairs/{args.model}__cosmos3_droid.json")
    packet = load_openpi_droid_episode(ROOT / "datasets/cosmos3_droid/success", 0, 30, 2)
    # Resource admission is for one policy only; never start a Cosmos worker.
    resource_admission({"world_model_gpus": [], "vla_gpus": [args.gpu],
                        "vla_memory_gib": 20, "max_process_memory_gib_per_gpu": 20}, True)
    output.mkdir(parents=True, exist_ok=False)
    bridge = build_action_bridge(pair.action_bridge)
    observation = ObservationBundle(**packet.trial["observation_bundle"],
                                   instruction=packet.trial["instruction"])
    report = {"status": "running", "world_model_inference_performed": False,
              "images": "held_initial_measured_views_for_protocol_check_only",
              "pair_id": pair.pair_id, "dataset": packet.provenance,
              "initial_state_source": "dataset_ground_truth", "steps": []}
    worker = None
    try:
        worker = ModelWorker(load_environment_registry()["vla.openpi"].python, output / "worker",
                             module="adapters.vla.openpi", adapter=pair.vla.adapter,
                             kwargs={"checkpoint_path": str(ROOT / f"checkpoints/vla/openpi/{args.model}-droid")},
                             gpus=[args.gpu], memory_gib=20)
        policy = RemoteVLA(worker)
        policy.reset(observation.robot_state.as_policy_input())
        for index in range(2):
            native = pair.vla.action.wrap(policy.predict_observation(observation, pair.vla), source=pair.vla.adapter)
            converted = bridge.convert(native, observation=observation, step_index=index)
            state = RobotState(values=bridge.integrate_state(observation.robot_state, native, None),
                               schema=pair.vla.state_schema, source="integrated_action_target")
            np.savez_compressed(output / f"chunk_{index:03d}.npz", native_actions=native.values,
                                wm_actions=converted.values,
                                **{key: value for key, value in state.values.items()})
            report["steps"].append({"native_shape": list(native.values.shape),
                                    "wm_shape": list(converted.values.shape),
                                    "input_state_source": observation.robot_state.source,
                                    "next_state_source": state.source,
                                    "bridge": converted.metadata, "worker_memory": worker.memory})
            observation = replace(observation, robot_state=state, step_index=index + 1)
        report["status"] = "passed_policy_and_bridge_only"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["cleanup"] = worker.close() if worker is not None else None
        publish_json_exclusive(output / "report.json", report)
    print(f"{report['status']}: {output}")


if __name__ == "__main__":
    main()
