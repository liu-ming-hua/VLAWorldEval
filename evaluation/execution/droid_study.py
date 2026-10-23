"""Serial, resumable DROID queues with live resource admission and review gates."""
from itertools import combinations
import json
import subprocess
import sys
import time

from evaluation.execution.executor import write_json
from evaluation.execution.reliability import exception_details
from evaluation.execution.workflow import evaluate_suite
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.paths import project_root


def choose_layout(free_mib, *, pool, wm_gpus, wm_budget, vla_budget, needs_vla):
    """Same CUDA/NCCL headroom as admission; no process eviction or overcommit."""
    for group in combinations(sorted(pool, key=lambda g: -free_mib.get(g, 0)), wm_gpus):
        for vla in (sorted(pool, key=lambda g: -free_mib.get(g, 0)) if needs_vla else [None]):
            budgets = {g: wm_budget+3+0.5 for g in group}
            if vla is not None:
                budgets[vla] = budgets.get(vla, 0.5) + vla_budget+1
            if all(free_mib.get(g, 0) >= budget*1024 for g, budget in budgets.items()):
                return {"world_model_gpus": list(group), "vla_gpus": [vla] if needs_vla else [group[0]]}
    return None


def policy_gate(entry, directory):
    """Pixel motion is not a success/physics gate. Require explicit bound review."""
    root = project_root()
    reviews = directory / "reviews.json"
    if not reviews.is_file():
        return "pending_review"
    records = json.loads(reviews.read_text())
    if records.get("schema_version") != 1 or not isinstance(records.get("records"), list):
        raise ValueError("Reviews require schema_version=1 and records")
    matched = [r for r in records["records"] if r.get("episode") == entry["episode"]
               and r.get("world_model") == entry["world_model"]]
    if not matched:
        return "pending_review"
    if len(matched) != 1:
        raise ValueError("Duplicate WM/case review")
    review = matched[0]
    if review.get("decision") not in {"admit", "reject", "uncertain"} or not review.get("reviewer") or not review.get("reason"):
        raise ValueError("A review needs decision/reviewer/reason")
    evidence = project_path(root, review["evidence_report"])
    if _digest(evidence) != review["evidence_sha256"]:
        raise ValueError("Review evidence hash mismatch")
    report = json.loads(evidence.read_text())
    if report.get("entry", {}).get("episode") != entry["episode"] or report["entry"]["world_model"] != entry["world_model"]:
        raise ValueError("Review must bind the same positive control")
    if review["decision"] == "admit" and not report.get("ok"):
        raise ValueError("Cannot admit an unsuccessful engineering control")
    if review["decision"] == "admit" and (report.get("execution_performed") is not True or
            not report["entry"].get("control_owner") or
            {j["id"] for j in report.get("execution", {}).get("jobs", [])} !=
            {"real_action_smoke", "real_action_full", "real_action_matched"}):
        raise ValueError("Admission requires executed smoke/full/matched positive controls")
    if review["decision"] == "admit":
        jobs = report["execution"]["jobs"]
        if len(jobs) != 3 or any(j.get("status") != "completed" or
                type(j.get("world_model_calls")) is not int or j["world_model_calls"] < 1 or
                type(j.get("vla_calls")) is not int or j["vla_calls"] != 0 for j in jobs):
            raise ValueError("Admission requires three completed WM-only control jobs with actual model calls")
    if review["decision"] == "admit" and not all(review.get(k) is True for k in
            ("reference_task_visible", "action_following", "visual_consistency", "long_rollout_plausible")):
        raise ValueError("Admit requires visible reference and all WM plausibility checks")
    return review["decision"]


def run_study(directory, *, phase="controls", watch=False, poll_seconds=30):
    if phase == "all":
        controls = run_study(directory, phase="controls", watch=watch, poll_seconds=poll_seconds)
        if controls["status"] in {"waiting_for_gpu", "running"}:
            return {"status": controls["status"], "controls": controls, "task_success_verified": False}
        policies = run_study(directory, phase="policies", watch=watch, poll_seconds=poll_seconds)
        status = policies["status"]
        if status == "finished" and controls["status"] == "finished_with_failures":
            status = "finished_with_failures"
        return {"status": status, "controls": controls, "policies": policies, "task_success_verified": False}
    root = project_root()
    directory = project_path(root, str(directory))
    study_path = directory / "study.json"
    study = json.loads(study_path.read_text())
    if phase not in {"controls", "policies"} or type(poll_seconds) is not int or not 1 <= poll_seconds <= 60:
        raise ValueError("Invalid queue phase or polling period")
    spec = study["spec"]
    if set(spec["allowed_gpus"]) - {0,1,2,3}:
        raise ValueError("Study may only use GPU0–3")
    entries = [e for e in study["entries"] if phase == "policies" or e["control_owner"]]
    progress_path = directory / f"{phase}_progress.json"
    with asset_destination_lock(root, directory):
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "schema_version": 1, "phase": phase, "study_sha256": _digest(study_path),
            "status": "running", "task_success_verified": False,
            "entries": {e["id"]: {"status": "pending"} for e in entries}}
        if progress["study_sha256"] != _digest(study_path):
            raise ValueError("Frozen study changed; create a new study version")
        # An interrupted entry is visible and requires explicit new execution;
        # never silently rerun it and overwrite its evidence.
        for value in progress["entries"].values():
            if value["status"] == "running":
                value["status"] = "interrupted"
        while True:
            waiting_gpu = waiting_review = False
            for entry in entries:
                state = progress["entries"][entry["id"]]
                if state["status"] not in {"pending", "waiting_for_gpu", "waiting_for_review"}:
                    continue
                config_path = project_path(root, entry["config"])
                if _digest(config_path) != entry["sha256"]:
                    raise ValueError("Frozen suite changed")
                if phase == "policies":
                    gate = policy_gate(entry, directory)
                    state["quality_gate"] = gate
                    if gate == "pending_review":
                        state["status"] = "waiting_for_review"
                        waiting_review = True
                        continue
                    if gate != "admit":
                        state.update(status="excluded_by_quality_gate", exclusion_reason=gate)
                        continue
                try:
                    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free",
                        "--format=csv,noheader,nounits"], text=True, timeout=10)
                    free = {int(r.split(",")[0]): int(r.split(",")[1]) for r in output.splitlines()}
                except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                    state.update(status="waiting_for_gpu", resource_error=exception_details(exc))
                    waiting_gpu = True
                    continue
                state.pop("resource_error", None)
                layout = choose_layout(free, pool=spec["allowed_gpus"],
                    wm_gpus=1 if entry["world_model"] == "ctrl" else 4,
                    wm_budget=spec["memory_gib"][entry["world_model"]],
                    vla_budget=spec["memory_gib"]["vla"], needs_vla=phase == "policies")
                if layout is None:
                    state.update(status="waiting_for_gpu", observed_free_mib=free)
                    waiting_gpu = True
                    continue
                config = json.loads(config_path.read_text())
                config["runtime"].update(layout)
                attempts = state.setdefault("attempts", [])
                run_dir = directory / "runs" / phase / entry["id"] / f"dispatch_{len(attempts)+1:04d}"
                run_dir.mkdir(parents=True, exist_ok=False)
                dispatch = run_dir / "suite.json"
                publish_json_exclusive(dispatch, config)
                state.update(status="running", dispatch=str(dispatch.relative_to(root)))
                progress["status"] = "running"
                write_json(progress_path, progress)
                report_path = run_dir / "report.json"
                attempt = []
                try:
                    report = evaluate_suite(dispatch, experiments=["policy_matched"] if phase == "policies"
                        else ["real_action_smoke", "real_action_full", "real_action_matched"], execute=True,
                        on_attempt=lambda path: attempt.append(str(path)))
                    report["entry"] = entry
                    gpu_race = report["status"] == "preflight_failed" and any(
                        c["name"] == "gpu" and c["status"] == "failed" for c in report["preflight"]["checks"])
                    state.update(status="completed" if report["ok"] else "waiting_for_gpu" if gpu_race else "failed")
                    waiting_gpu |= gpu_race
                except Exception as exc:
                    report = {"schema_version": 1, "ok": False, "status": "failed", "entry": entry,
                              "attempt": attempt, "error": exception_details(exc), "task_success_verified": False}
                    state.update(status="failed")
                publish_json_exclusive(report_path, report)
                state["report"] = str(report_path.relative_to(root))
                state["report_sha256"] = _digest(report_path)
                attempts.append({"report": state["report"], "sha256": state["report_sha256"], "status": state["status"]})
                write_json(progress_path, progress)
                print(json.dumps({"phase": phase, "entry": entry["id"], "status": state["status"]}), file=sys.stderr, flush=True)
            progress["status"] = ("waiting_for_review" if waiting_review else
                "waiting_for_gpu" if waiting_gpu else "finished_with_failures" if any(
                e["status"] in {"failed", "interrupted"} for e in progress["entries"].values()) else "finished")
            write_json(progress_path, progress)
            if not watch or not (waiting_gpu or waiting_review):
                return progress
            time.sleep(poll_seconds)
