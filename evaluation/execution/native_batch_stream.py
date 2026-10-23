"""Run a frozen cohort with dynamic cross-trajectory batching and shared scoring."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path

from evaluation.execution.admission import native_gpu_lease
from evaluation.execution.batch_session import BatchedNativeSession
from evaluation.execution.native_replay import run_native_replay
from evaluation.planning.native_reference import prepare_reference_study
from evaluation.reporting.checkpoint_metrics import select_replays
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json
from vlaworldeval.paths import project_root


def run(plan_path, output, previous, *, batch_size, memory_gib=72):
    root = project_root()
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    plan = json.loads(plan_path.read_text())
    selected, _ = select_replays(root, previous[0], plan, additional_directories=previous[1:])
    for path, _ in selected.values():
        report = json.loads(path.read_text())
        if any(_digest(path.parent/name) != sha for name, sha in report["files"].items()):
            raise ValueError("Previous completed artifacts changed")
    output.mkdir(parents=True, exist_ok=False)
    pending = {**plan, "cases": [c for c in plan["cases"] if c["id"] not in selected]}
    publish_json_exclusive(output/"plan.json", pending)
    publish_json_exclusive(output/"reuse.json", {i: {"summary": str(p), "sha256": _digest(p)}
                                                for i, (p, _) in selected.items()})
    progress = {"status": "preparing", "expected_cases": len(plan["cases"]),
        "previous_completed": len(selected), "generated_in_session": 0,
        "batch_size": batch_size, "memory_gib_per_gpu": memory_gib, "gpus": [0, 1, 2, 3],
        "cohort_sha256": _digest(plan_path), "failures": {}}
    journal = output/"progress.json"
    publish_json_exclusive(journal, progress)
    if not pending["cases"]:
        progress.update(status="completed", certified_completed=len(selected))
        replace_owned_json(journal, progress)
        return
    prepare_reference_study(root, output/"plan.json", output/"study")
    study = json.loads((output/"study/study.json").read_text())
    configs = output/"configs"
    configs.mkdir()
    admission = {"schema_version": 1, "kind": "native_cross_trajectory_batch",
        "source_plan": str(plan_path), "source_plan_sha256": _digest(plan_path),
        "batch_size": batch_size, "cases": {}}
    for job in study["jobs"]:
        original = json.loads(Path(job["config"]).read_text())
        if original["adapter"] != "Cosmos3DroidFDNativeAdapter":
            raise ValueError("Batch qualification currently covers native Cosmos only")
        original.update(gpus=[0, 1, 2, 3], memory_gib=memory_gib)
        original["kwargs"]["inference_batch_size"] = batch_size
        destination = configs/f'{job["id"]}.json'
        publish_json_exclusive(destination, original)
        job["batch_config"] = destination
        admission["cases"][job["id"]] = {"original_config": job["config"],
            "original_sha256": _digest(Path(job["config"])), "effective_config": str(destination),
            "effective_sha256": _digest(destination)}
    publish_json_exclusive(output/"batch_admission.json", admission)

    def generated(path):
        if json.loads(path.read_text())["status"] == "generated_pending_worker_close":
            progress["generated_in_session"] += 1
            progress["last_generated_case"] = path.parent.name
            progress["model_loads"] = 1
            replace_owned_json(journal, progress)

    try:
        with native_gpu_lease(root, [0, 1, 2, 3]):
            with BatchedNativeSession(output/"resident_session", batch_size=batch_size,
                                      on_generated=generated) as session:
                progress["status"] = "running"
                replace_owned_json(journal, progress)
                # Extra slots let CPU scoring/video writing overlap GPU generation.
                # run_native_replay remains the sole inference/scoring/report loop.
                with ThreadPoolExecutor(max_workers=batch_size+2) as pool:
                    jobs = iter(study["jobs"])
                    futures = {}

                    def submit_next():
                        if (output/"stop_after_cases").exists() or session.batch_error is not None:
                            return
                        job = next(jobs, None)
                        if job is not None:
                            future = pool.submit(run_native_replay, job["batch_config"], Path(job["input"]),
                                output/"runs"/job["id"], reference_path=Path(job["reference"]),
                                worker_session=session)
                            futures[future] = job

                    for _ in range(batch_size+2):
                        submit_next()
                    while futures:
                        done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                        for future in done:
                            job = futures.pop(future)
                            try:
                                future.result()
                            except BaseException as exc:
                                with session.lock:
                                    progress["failures"][job["id"]] = str(exc)
                                    replace_owned_json(journal, progress)
                            submit_next()
            if not session.clean or progress["failures"]:
                raise RuntimeError("Batch cohort contains failed jobs or unclean worker closure")
        total = len(selected)+progress["generated_in_session"]
        progress.update(status="completed" if total == len(plan["cases"]) else "stopped_partial",
                        certified_completed=total)
    except BaseException as exc:
        progress.update(status="failed", error=str(exc))
        raise
    finally:
        replace_owned_json(journal, progress)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--previous", type=Path, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--benchmark", type=Path)
    args = parser.parse_args()
    if args.benchmark:
        evidence = json.loads(args.benchmark.read_text())
        if evidence.get("status") not in {"completed", "qualified_with_probe_failure"}:
            raise ValueError("A clean completed benchmark is required")
        size = evidence["best_batch_size"]
    else:
        size = args.batch_size
    if type(size) is not int or size < 1:
        parser.error("Specify --batch-size or a qualified --benchmark")
    run(args.plan, args.directory, args.previous, batch_size=size)
