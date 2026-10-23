"""Run with an installed wheel interpreter, away from the source checkout.

Exercises the real isolated IPC workers as well as the in-process CPU demo.
No Torch, learned weights, GPUs, cloud APIs or future dataset frames.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from evaluation.execution.client import ModelWorker, RemoteVLA, RemoteWorldModel
from evaluation.evaluator import PolicyWorldModelEvaluator
from vlaworldeval.demo import DemoWorldModel, demo_manifest, run_demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for package in ("torch", "jax", "tensorflow", "transformers"):
        if importlib.util.find_spec(package) is not None:
            raise ValueError(f"Acceptance requires a lightweight installed interpreter, found {package}")
    args.output.mkdir(parents=True, exist_ok=False)
    demo = run_demo(args.output / "demo", chunks=30)
    workers = []
    report = {"status": "running", "installed_python": sys.executable, "demo": demo}
    try:
        for adapter in ("DemoPolicy", "DemoWorldModel"):
            workers.append(ModelWorker(Path(sys.executable), args.output / adapter, module="vlaworldeval.demo",
                                       adapter=adapter, kwargs={}, gpus=[], memory_gib=1))
        initial = DemoWorldModel.render(np.array([12, 12]))
        trial = {"instruction": "move", "observation_bundle": {
            "views": {"front": initial, "wrist": initial[:, ::-1].copy()}, "instruction": "move"}}
        trace = PolicyWorldModelEvaluator(RemoteVLA(workers[0]), RemoteWorldModel(workers[1]),
                                         pair_manifest=demo_manifest()).rollout_detailed(trial, 20)
        assert trace.primary_video().shape == (61, 64, 64, 3)
        assert len(trace.native_actions) == 20
        report.update(status="passed", ipc_chunks=20, ipc_frames=61)
    except BaseException:
        report["status"] = "failed"
        raise
    finally:
        report["cleanup"] = [worker.close() for worker in workers]
        if any(worker.failure for worker in workers):
            report["status"] = "failed"
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise ValueError("Installed workers failed clean shutdown")


if __name__ == "__main__":
    main()
