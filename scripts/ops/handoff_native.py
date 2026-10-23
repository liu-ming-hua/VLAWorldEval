#!/usr/bin/env python3
"""Drain explicit native controllers and freeze a disjoint cross-host remainder.

This orchestrates the framework's native-study entry point, not model inference.
Source studies, scientific configs, packets, references and failures stay intact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from evaluation.execution.reliability import clean_worker_exit
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def drain(root, controllers, record):
    tasks = []
    for pid, directory in controllers:
        directory = project_path(root, directory)
        if not alive(pid):
            continue
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        argv = [arg.decode() for arg in argv if arg]
        if ("native-study" not in argv or "run" not in argv or "--directory" not in argv
                or Path(f"/proc/{pid}/cwd").resolve() != root
                or project_path(root, argv[argv.index("--directory")+1]) != directory
                or os.getpgid(pid) != pid):
            raise ValueError(f"Refusing unidentified/non-isolated controller {pid}")
        progress = json.loads((directory / "progress.json").read_text())
        current = [name for name, job in progress["jobs"].items() if job["status"] == "running"]
        tasks.append({"pid": pid, "directory": str(directory), "preserve_current": current})
    publish_json_exclusive(record, {"controllers": tasks, "scope": "wait current cases, then stop task process groups"})
    waiting = tasks.copy()
    while waiting:
        for task in waiting.copy():
            pid, directory = task["pid"], Path(task["directory"])
            if not alive(pid):
                waiting.remove(task)
                continue
            progress = json.loads((directory / "progress.json").read_text())
            states = [progress["jobs"].get(name, {}).get("status") for name in task["preserve_current"]]
            if all(state in {"completed", "failed"} for state in states):
                # The original case finished; a just-started subsequent attempt
                # may be interrupted and is retained as such, never counted done.
                assert os.getpgid(pid) == pid
                os.killpg(pid, signal.SIGTERM)
                print("drained", pid, states, flush=True)
                waiting.remove(task)
        if waiting:
            time.sleep(2)
    while any(alive(task["pid"]) for task in tasks):
        time.sleep(2)


def export_remainder(root, studies, output, gpus):
    if not gpus or len(gpus) != len(set(gpus)) or any(type(g) is not int or g < 0 for g in gpus):
        raise ValueError("Unique nonnegative target GPUs required")
    if output.exists():
        raise FileExistsError(output)
    jobs, completed, sources = {}, set(), []
    for value in studies:
        directory = project_path(root, value)
        study_path, progress_path = directory / "study.json", directory / "progress.json"
        study, progress = json.loads(study_path.read_text()), json.loads(progress_path.read_text())
        if progress["study_sha256"] != _digest(study_path):
            raise ValueError("Source study changed")
        sources.append({"study": str(study_path.relative_to(root)), "study_sha256": _digest(study_path),
                        "progress_sha256": _digest(progress_path), "recorded_status": progress["status"]})
        for job in study["jobs"]:
            if job["id"] in jobs and jobs[job["id"]] != job:
                raise ValueError("Overlapping jobs disagree")
            jobs[job["id"]] = job
            entry = progress["jobs"].get(job["id"], {})
            if entry.get("status") == "completed":
                summary = directory / "runs" / job["id"] / "summary.json"
                if (_digest(summary) != entry["report_sha256"]
                        or not clean_worker_exit(json.loads(summary.read_text()).get("cleanup"))):
                    raise ValueError("Source completed evidence invalid")
                if job["id"] in completed:
                    raise ValueError("Duplicate successful source case")
                completed.add(job["id"])
    pending = [job for name, job in jobs.items() if name not in completed]
    shards, costs = {g: [] for g in gpus}, dict.fromkeys(gpus, 0)
    for job in sorted(pending, key=lambda j: (-j["chunks"], j["id"])):
        for key in ("input", "config", "reference"):
            if _digest(project_path(root, job[key])) != job[key+"_sha256"]:
                raise ValueError("Pending scientific input changed")
        gpu = min(gpus, key=lambda g: costs[g])
        prepared = str(Path(job["input"]).parent.relative_to(root))
        shards[gpu].append({"id": job["id"], "prepared": prepared})
        costs[gpu] += job["chunks"] + 1
    output.mkdir(parents=True)
    for gpu, cases in shards.items():
        publish_json_exclusive(output / f"gpu{gpu}_plan.json", {
            "schema_version": 1, "scope": "Exact disjoint remaining source cases; not the full cohort alone",
            "cases": cases, "unavailable": {}})
    audit = {"schema_version": 1, "source_root": str(root), "source_studies": sources, "source_union": len(jobs),
             "source_completed": len(completed), "remote_pending": len(pending),
             "completed_ids": sorted(completed), "pending": pending,
             "gpus": gpus, "shard_counts": {g: len(v) for g, v in shards.items()},
             "scientific_input_bytes_changed": False, "same_case_scheduled_on_both_hosts": False}
    publish_json_exclusive(output / "handoff.json", audit)
    print(json.dumps({k: audit[k] for k in ("source_union", "source_completed", "remote_pending", "shard_counts")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--controller", action="append", default=[], help="PID=project-relative-study")
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = project_path(root, args.output)
    controllers = [(int(value.split("=", 1)[0]), value.split("=", 1)[1]) for value in args.controller]
    drain_record = output.parent / (output.name + "_drain.json")
    drain_record.parent.mkdir(parents=True, exist_ok=True)
    drain(root, controllers, drain_record)
    export_remainder(root, args.study, output, args.gpus)


if __name__ == "__main__":
    main()
