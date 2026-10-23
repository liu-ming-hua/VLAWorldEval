#!/usr/bin/env python3
"""Schedule migrated native cases exclusively through the common framework."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from evaluation.planning.native_reference import prepare_reference_study
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest


def launch(root, mode, handoff=None, origin_root=None, shard=None, *, native_plan=None,
           manifest=None, campaign_directory=None, after_irasim=False):
    base = (project_path(root, str(campaign_directory)) if campaign_directory else
            root / "results/checkpoint_capability_v1/remote_server_v1")
    base.mkdir(parents=True, exist_ok=True)
    if after_irasim:
        if mode != "cosmos" or native_plan is None or campaign_directory is None:
            raise ValueError("Budget dependency gate requires explicit Cosmos plan and campaign directory")
        wait_irasim(root, base)
    if mode == "irasim":
        evidence = json.loads((handoff / "handoff.json").read_text())
        origin = Path(evidence.get("source_root") or origin_root or "")
        if not origin.is_absolute():
            raise ValueError("Explicit original project root required for a legacy handoff")
        if not evidence["gpus"] or not set(evidence["gpus"]).issubset({0, 1, 2, 3}):
            raise ValueError("This remote campaign is restricted to physical GPUs 0–3")
        shards = [(f"gpu{g}", handoff / f"gpu{g}_plan.json", [g]) for g in evidence["gpus"]]
        if shard is not None:
            shards = [item for item in shards if item[0] == shard]
            if not shards:
                raise ValueError("Unknown handoff shard")
        selected_cases = [case for _, plan, _ in shards
                          for case in json.loads(plan.read_text())["cases"]]
        selected = {case["id"] for case in selected_cases}
        pending = {job["id"]: job for job in evidence["pending"]}
        if len(selected) != len(selected_cases) or not selected.issubset(pending):
            raise ValueError("Shard contains a case outside frozen pending ownership")
        for case in selected_cases:
            expected = str(Path(pending[case["id"]]["input"]).parent.relative_to(origin))
            if case["prepared"] != expected:
                raise ValueError("Shard prepared path differs from frozen source ownership")
        for identifier in selected:
            job = pending[identifier]
            for key in ("input", "config", "reference"):
                local = project_path(root, str(Path(job[key]).relative_to(origin)))
                if _digest(local) != job[key+"_sha256"]:
                    raise ValueError("Cross-host input/reference/config hash mismatch")
    else:
        if shard is not None:
            raise ValueError("Cosmos uses the full cohort on one four-GPU group, not per-rank shards")
        if native_plan is not None:
            plan_path = project_path(root, str(native_plan))
            if manifest is None:
                raise ValueError("Migrated subset requires an explicit source prepared manifest")
            evidence = json.loads(project_path(root, str(manifest)).read_text())
            plan = json.loads(plan_path.read_text())
            if (evidence["plan_sha256"] != _digest(plan_path)
                    or len(plan["cases"]) != 400
                    or set(evidence["cases"]) != {c["id"] for c in plan["cases"]}):
                raise ValueError("Prepared subset differs from frozen400-case selection")
            for case in plan["cases"]:
                for name in ("input.npz", "config.json", "reference.npz"):
                    path = project_path(root, case["prepared"]) / name
                    if _digest(path) != evidence["cases"][case["id"]]["sha256"][name]:
                        raise ValueError("Cross-host subset input/reference/config hash mismatch")
            shards = [("quad0", plan_path, [0, 1, 2, 3])]
        else:
            shards = cosmos_full_shards(root)
    for name, plan, gpus in shards:
        directory = base / (mode + "_runs") / name
        if not directory.exists():
            prepare_reference_study(root, plan, directory)
        session = f"vwe-remote-{mode}-{name}" if campaign_directory is None else f"vwe-budget400-{mode}-{name}"
        if subprocess.run(["tmux", "has-session", "-t", "=" + session], capture_output=True).returncode == 0:
            raise ValueError(f"Existing live session: {session}; do not duplicate")
        command = ["bash", "scripts/ops/remote.sh", "env", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
                   "environment/core/bin/python", "-m", "cli", "native-study", "run", "--directory",
                   str(directory.relative_to(root)), "--watch", "--exclusive-gpu"]
        command += ["--gpu", str(gpus[0]), "--memory-gib", "12"] if mode == "irasim" else [
            "--gpus", *map(str, gpus)]
        # Cosmos cooperatively waits for the corresponding IRASim rank leases.
        shell = shlex.join(command) + " > " + shlex.quote(str(base / f"{mode}_{name}.log")) + " 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(root), shell], check=True)
        print("started", session, "gpus", gpus, flush=True)


def cosmos_full_shards(root):
    """Legacy full-cohort protocol; budget campaigns must pass their plan."""
    corpus = root / "datasets/cosmos3_droid_prepared/full_holdout_v2"
    plan = json.loads((corpus / "plan.json").read_text())
    progress = json.loads((corpus / "progress.json").read_text())
    if len(plan["cases"]) != 2145 or len(progress["completed"]) != 2145:
        raise ValueError("Full2145-case Cosmos native cohort required; no partial substitute")
    for case in sorted(plan["cases"], key=lambda c: (-progress["completed"][c["id"]]["chunks"], c["id"])):
        directory = project_path(root, case["prepared"])
        for name, digest in progress["completed"][case["id"]]["sha256"].items():
            if _digest(directory / name) != digest:
                raise ValueError("Cosmos prepared bytes differ from frozen source manifest")
    return [("quad0", corpus / "plan.json", [0, 1, 2, 3])]


def wait_irasim(root, base):
    """Finish all four smaller-model queues before scheduling the large model."""
    while True:
        ready = True
        for gpu in range(4):
            directory = base/"irasim_runs"/f"gpu{gpu}"
            study = directory/"study.json"
            progress = json.loads((directory/"progress.json").read_text())
            if progress["study_sha256"] != _digest(study):
                raise ValueError("Dependency study changed")
            if progress["status"] not in {"running", "completed", "waiting_for_gpu_lease"}:
                raise ValueError("IRASim dependency stopped/failed; inspect instead of silently starting Cosmos")
            ready &= progress["status"] == "completed"
        if ready:
            return
        print("waiting for all four budget IRASim queues; no Cosmos model loaded", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["irasim", "cosmos"], required=True)
    parser.add_argument("--handoff", type=Path)
    parser.add_argument("--origin-root", type=Path)
    parser.add_argument("--shard", choices=["gpu0", "gpu1", "gpu2", "gpu3"])
    parser.add_argument("--plan", type=Path, help="explicit400-case Cosmos budget plan")
    parser.add_argument("--manifest", type=Path, help="source checksum manifest for the prepared subset")
    parser.add_argument("--campaign-directory", type=Path, help="new output namespace, never reuse cancelled full studies")
    parser.add_argument("--after-irasim", action="store_true", help="wait for all four budget IRASim queues to complete")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    launch(root, args.mode, project_path(root, str(args.handoff)) if args.handoff else None,
           args.origin_root, args.shard, native_plan=args.plan, manifest=args.manifest,
           campaign_directory=args.campaign_directory, after_irasim=args.after_irasim)
