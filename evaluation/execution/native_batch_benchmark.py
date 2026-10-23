"""Four-GPU native Cosmos throughput qualification, separate from evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from evaluation.execution.admission import native_gpu_lease
from evaluation.execution.client import ModelWorker
from evaluation.execution.native_replay import admit_native_replay
from evaluation.execution.reliability import clean_worker_exit
from evaluation.execution.transport import read_packet
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json
from vlaworldeval.paths import project_root


def benchmark(plan_path, output, sizes=(1, 2, 4, 8)):
    root = project_root()
    plan = json.loads(Path(plan_path).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = plan["cases"][:max(sizes)]
    packets = [read_packet(root/c["prepared"]/"input.npz") for c in cases]
    config, spec, _, _, _, kwargs, options = admit_native_replay(
        root/cases[0]["prepared"]/"config.json", root/cases[0]["prepared"]/"input.npz",
        suite_dir=output/"runtime")
    kwargs.update(inference_batch_size=max(sizes), runtime_dir=str(output/"runtime"))
    records = {"status": "running", "sizes": [], "config": config,
               "memory_budget_gib": 72, "gpus": [0, 1, 2, 3]}
    publish_json_exclusive(output/"progress.json", records)
    worker = None
    baselines = []
    try:
        with native_gpu_lease(root, [0, 1, 2, 3]):
            worker = ModelWorker(spec.python, output/"worker", module=config["module"],
                adapter=config["adapter"], kwargs=kwargs, gpus=[0, 1, 2, 3],
                memory_gib=72, options=options)
            for i, packet in enumerate(packets):
                item = {"image": packet["reset_args"]["value"], "action": packet["actions"][0],
                    "seed": config["seed"], "runtime_dir": str(output/f"baseline_i{i}")}
                result = worker.call("native_batch_step", items=[item])
                baselines.append(next(iter(result[0]["views"].values())).copy())
            for size in sizes:
                entry = {"batch_size": size, "seconds": [], "results": []}
                for repeat in range(3):
                    items = [{"image": p["reset_args"]["value"], "action": p["actions"][0],
                        "seed": config["seed"], "runtime_dir": str(output/f"b{size}_r{repeat}_i{i}")}
                        for i, p in enumerate(packets[:size])]
                    start = time.perf_counter()
                    result = worker.call("native_batch_step", items=items)
                    entry["seconds"].append(time.perf_counter()-start)
                    first = next(iter(result[0]["views"].values()))
                    errors = [float(np.abs(next(iter(r["views"].values())).astype(float)-baseline).mean())
                              for r, baseline in zip(result, baselines)]
                    entry["results"].append({"first_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
                        "sample_mae_vs_batch1": errors, "max_mae_vs_batch1": max(errors),
                        "finite": all(np.isfinite(v).all() for r in result for v in r["views"].values())})
                    entry["memory"] = worker.memory
                    replace_owned_json(output/"progress.json", {**records, "current": entry})
                entry["steady_chunks_per_second"] = size / float(np.median(entry["seconds"][1:]))
                entry["qualified"] = all(r["finite"] and r["max_mae_vs_batch1"] < 1.0 for r in entry["results"])
                records["sizes"].append(entry)
                replace_owned_json(output/"progress.json", records)
            cleanup = worker.close()
            if not clean_worker_exit(cleanup):
                raise RuntimeError("Benchmark worker did not close cleanly")
        qualified = [r for r in records["sizes"] if r["qualified"]]
        records.update(status="completed", cleanup=cleanup,
            best_batch_size=max(qualified, key=lambda r: r["steady_chunks_per_second"])["batch_size"])
    except BaseException as exc:
        records.update(status="failed", error=str(exc))
        qualified = [r for r in records["sizes"] if r["qualified"]]
        if not isinstance(exc, Exception) or not qualified:
            raise
        records.update(status="qualified_with_probe_failure",
            best_batch_size=max(qualified, key=lambda r: r["steady_chunks_per_second"])["batch_size"],
            rejected_probe=entry["batch_size"])
    finally:
        if worker is not None:
            worker.close()
        replace_owned_json(output/"progress.json", records)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    benchmark(args.plan, args.directory, tuple(args.sizes))
