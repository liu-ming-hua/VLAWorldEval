"""Event-window execution audit and synchronized native-layout comparisons."""
import html
import json
from pathlib import Path
import time

import numpy as np

from evaluation.execution.reliability import clean_worker_exit
from evaluation.execution.transport import read_packet
from evaluation.planning.cosmos_events import INFERENCE_SOURCES
from evaluation.reporting.cosmos_horizon import _completed_progress, _probe_prefix_video
from evaluation.reporting.cosmos_temporal_review import _render_case
from evaluation.reporting.task_scores import _digest
from vlaworldeval.atomic_json import publish_json_exclusive


def finalize_event_study(root, directory, *, wait=False, timeout=7200, poll_seconds=30):
    """CPU-only bounded completion observer; never starts/retries inference."""
    if (type(wait) is not bool or type(timeout) is not int or not 1 <= timeout <= 14400
            or type(poll_seconds) is not int or not 1 <= poll_seconds <= 60):
        raise ValueError("Require a bounded completion observer")
    directory = Path(directory).resolve()
    if (directory / "finalization.json").exists():
        raise FileExistsError("Finalization already recorded; inspect its hashes, do not overwrite")
    started = time.monotonic()
    while True:
        path = directory / "progress.json"
        progress = json.loads(path.read_text()) if path.exists() else {}
        status = progress.get("status")
        if status == "completed":
            break
        if status not in {None, "running", "waiting_for_gpu"}:
            raise ValueError("Queue failed/interrupted; preserve attempts and inspect before finalization")
        if not wait or time.monotonic() - started >= timeout:
            raise TimeoutError("Queue incomplete; no audit or comparison has been approved")
        time.sleep(min(poll_seconds, max(0, timeout - (time.monotonic() - started))))
    audit = audit_event_study(root, directory)
    render_event_study(root, directory)
    result = {"status": "artifacts_complete_pending_generated_quality_review", "world_model_calls": audit["world_model_calls"],
        "vla_calls": 0, "quality_admitted": False, "task_success": None,
        "study_sha256": _digest(directory / "study.json"), "execution_audit_sha256": _digest(directory / "execution_audit.json"),
        "comparisons_sha256": _digest(directory / "comparisons.json"), "index_sha256": _digest(directory / "index.html")}
    publish_json_exclusive(directory / "finalization.json", result)
    return result


def audit_event_study(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text())
    _completed_progress(directory, study)
    expected = {(model, episode, seed) for model in ("eef", "joint")
        for episode in (0, 351, 365, 795, 903) for seed in (1, 2)}
    if (len(study["jobs"]) != 20 or {(j["model"], j["episode"], j["seed"]) for j in study["jobs"]} != expected
            or any(j["chunks"] != (4 if j["model"] == "eef" else 2) for j in study["jobs"])):
        raise ValueError("Incomplete fixed event matrix")
    for path, sha in ((directory / "preparation.json", study["preparation_sha256"]),
            (directory / "source_review.json", study["source_review_sha256"]),
            (Path(study["environments"]), study["environments_sha256"]),
            (root / study["plan"]["parent"] / "study.json", study["parent_study_sha256"]),
            (root / "configs/studies/cosmos_event_windows_v1.json", study["plan_sha256"])):
        if _digest(path) != sha:
            raise ValueError("Frozen study provenance changed")
    hashes = {name: _digest(root / name) for name in INFERENCE_SOURCES}
    if hashes != study["inference_source_sha256"]:
        raise ValueError("Inference-critical source changed")
    parent = json.loads((root / study["plan"]["parent"] / "study.json").read_text())
    rows = []
    for case in study["cases"]:
        for path, sha in [(case["data_path"], case["data_sha256"]), (case["event_sheet"], case["event_sheet_sha256"])] + [
                (r["path"], r["sha256"]) for r in case["references"].values()]:
            if _digest(Path(path)) != sha:
                raise ValueError("Recorded data or reference changed")
    for job in study["jobs"]:
        for field in ("input", "config", "template_config"):
            if _digest(Path(job[field])) != job[field + "_sha256"]:
                raise ValueError("Frozen native input/config changed")
        folder = directory / "runs" / job["id"]
        report = json.loads((folder / "summary.json").read_text())
        if (report["status"] != "completed" or not clean_worker_exit(report["cleanup"])
                or report["world_model_calls"] != job["chunks"] or report["vla_calls"] != 0
                or report["frames_per_view"] != 65 or report["anchor_events"]
                or report["feedback_mode"] != "autoregressive"
                or report["config_sha256"] != job["config_sha256"] or report["input_sha256"] != job["input_sha256"]
                or any(report["framework_code_sha256"].get(name) != sha for name, sha in hashes.items())):
            raise ValueError("Replay protocol/source mismatch: " + job["id"])
        for name, sha in report["files"].items():
            if _digest(folder / name) != sha:
                raise ValueError("Executed output changed")
        if not np.array_equal(read_packet(Path(job["input"]))["actions"], np.load(folder / "actions.npy", allow_pickle=False)):
            raise ValueError("Executed actions differ from frozen packet")
        baseline = next(b for b in parent["baselines"] if b["model"] == job["model"] and b["episode"] == job["episode"])
        if any(report["runtime"]["packages"].get(k) != baseline["packages"].get(k)
                for k in ("torch", "diffusers", "transformers", "numpy")):
            raise ValueError("Verified model environment changed")
        worker = json.loads((folder / "worker/worker.json").read_text())
        if (worker["gpus"] != [0, 1, 2, 3] or worker["memory_gib"] != 16
                or any(k.lower().endswith("_proxy") for k in worker["environment"])
                or any(worker["environment"].get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))):
            raise ValueError("GPU, offline or proxy constraint violated")
        rows.append({"id": job["id"], "world_model_calls": report["world_model_calls"],
            "report_sha256": _digest(folder / "summary.json"),
            "actual_video_probe": _probe_prefix_video(folder / "droid.cosmos_composite.mp4", (640, 540), frames=65)})
    result = {"status": "passed_execution_audit_not_quality_approval", "jobs": rows,
        "world_model_calls": sum(r["world_model_calls"] for r in rows), "vla_calls": 0,
        "quality_admitted": False, "task_success": None, "study_sha256": _digest(directory / "study.json")}
    (directory / "execution_audit.json").write_text(json.dumps(result, indent=2))
    return result


def render_event_study(root, directory):
    directory = Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text())
    _completed_progress(directory, study)
    output = directory / "comparisons"
    output.mkdir(exist_ok=True)
    rows, cards = [], []
    for case in study["cases"]:
        for model in study["plan"]["models"]:
            key = f"{model}_ep{case['episode']:06d}"
            folder = output / key
            inputs = {k: v for k, v in case["references"].items() if k != "composite"}
            for seed in (1, 2):
                path = directory / "runs" / f"{key}_seed{seed}" / "droid.cosmos_composite.mp4"
                inputs[f"seed{seed}"] = {"path": str(path), "sha256": _digest(path)}
            for value in inputs.values():
                if _digest(Path(value["path"])) != value["sha256"]:
                    raise ValueError("Render input changed")
            if folder.exists():
                row = json.loads((folder / "comparison.json").read_text())
                if row["inputs"] != inputs or any(_digest(folder / n) != sha for n, sha in row["files"].items()):
                    raise ValueError("Previously rendered comparison changed")
            else:
                row = _render_case(folder, key, inputs, 64, generated_keys=("hold", "seed1", "seed2"),
                    labels=("REAL", "HOLD INITIAL (not seed0)", "SEED1", "SEED2"))
                row.pop("native_eef_chunks")
                row.update(model=model, native_chunks=4 if model == "eef" else 2, source_start=case["start"], source_end=case["end"])
                (folder / "comparison.json").write_text(json.dumps(row, indent=2))
            _probe_prefix_video(folder / "comparison.mp4", (2560, 568), frames=65)
            rows.append(row)
            title = html.escape(f"{key}: source frames {case['start']}–{case['end']} | {case['instruction']}")
            sheets = " ".join(f'<a href="comparisons/{key}/{s["file"]}">{s["source_ticks"][0]}–{s["source_ticks"][-1]}</a>' for s in row["sheets"])
            cards.append(f'<section><h2>{title}</h2><video controls preload="metadata" src="comparisons/{key}/comparison.mp4"></video><p>{sheets}</p>'
                f'<a href="comparisons/{key}/native_t032.png">Native event tick32</a> · <a href="comparisons/{key}/native_t064.png">Native end</a></section>')
    result = {"status": "rendered_not_quality_approval", "cases": rows, "new_world_model_calls": 0,
        "quality_admitted": False, "task_success": None, "study_sha256": _digest(directory / "study.json")}
    (directory / "comparisons.json").write_text(json.dumps(result, indent=2))
    (directory / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Cosmos event controls</title>'
        '<style>body{background:#191b20;color:#eee;font:16px system-ui;margin:24px}video{width:100%}a{color:#acf}section{margin-bottom:40px}</style>'
        '<h1>REAL / HOLD INITIAL / SEED1 / SEED2</h1><p>64 control ticks at15Hz =4.267s;65 displayed frames. Each column: wrist above two exteriors. '
        'Hold is a CPU display diagnostic, NOT seed0 or a model intervention. Sensor threshold is NOT contact/task-success ground truth. '
        'Community checkpoints; training overlap unknown. Execution success is NOT quality approval.</p>' + ''.join(cards))
    return result
