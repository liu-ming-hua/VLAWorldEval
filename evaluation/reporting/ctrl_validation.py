"""Audit corrected Ctrl evidence and render one continuous comparison per case."""
import html
import json
from pathlib import Path
import subprocess

import numpy as np

from evaluation.execution.reliability import clean_worker_exit
from evaluation.execution.transport import read_packet
from evaluation.reporting.positive_controls import render_droid_comparison, VIEWS
from evaluation.reporting.task_scores import _digest


def case_run_directory(directory, case, mode):
    """Explicit successful retry pointer; never infer newest attempt by mtime."""
    base = directory / case
    pointer = base / ("selected_" + mode + ".json")
    if not pointer.exists():
        return base / mode
    selection = json.loads(pointer.read_text())
    path = base / selection["relative_path"]
    if not path.resolve().is_relative_to(base.resolve()) or path.is_symlink():
        raise ValueError("Retry pointer leaves the case")
    if _digest(path / "summary.json") != selection["report_sha256"]:
        raise ValueError("Selected retry report changed")
    return path


def audit_ctrl_validation(root, directory):
    import pyarrow.parquet as pq
    root, directory = Path(root).resolve(), Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text())
    jobs = []
    for case in study["cases"]:
        packet = read_packet(Path(case["input"]))
        for key in ("input", "config", "anchors"):
            if key in case and _digest(Path(case[key])) != case[key + "_sha256"]:
                raise ValueError("Frozen study asset changed")
        src = packet["source"]
        table = pq.read_table(src["data_path"], filters=[("episode_index", "=", case["episode"])],
            columns=["observation.state.cartesian_position", "observation.state.gripper_position"])
        poses = np.asarray(table["observation.state.cartesian_position"].to_pylist(), np.float32)
        grip = np.asarray(table["observation.state.gripper_position"].to_pylist(), np.float32).reshape(-1)
        indices = study["plan"]["start_frame"] + np.arange(case["chunks"])[:, None] * 12 + np.arange(5) * 3
        expected = np.concatenate([poses[indices], grip[indices, None]], axis=-1)
        if not np.array_equal(expected, packet["actions"]) or not np.array_equal(indices, src["action_source_frame_indices"]):
            raise ValueError("Ctrl actions/source indices differ from original measured parquet")
        for mode in case["modes"]:
            folder = case_run_directory(directory, case["id"], mode)
            report = json.loads((folder / "summary.json").read_text())
            if (report["status"] != "completed" or not clean_worker_exit(report["cleanup"])
                    or report["world_model_calls"] != case["chunks"] or report["vla_calls"] != 0
                    or report["frames_per_view"] != 4 * case["chunks"] + 1
                    or report["temporal_layout"] != "inclusive" or report["video_fps"] != 5
                    or report["config_sha256"] != case["config_sha256"] or report["input_sha256"] != case["input_sha256"]):
                raise ValueError(f"Incomplete or protocol-inconsistent replay: {case['id']}/{mode} ({report.get('status')})")
            if not np.array_equal(np.load(folder / "actions.npy", allow_pickle=False), expected):
                raise ValueError("Saved executed actions differ from source")
            for name, sha in report["files"].items():
                if _digest(folder / name) != sha:
                    raise ValueError("Replay artifact hash changed")
                if name.endswith(".mp4"):
                    stream = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames,avg_frame_rate", "-of", "json", str(folder / name)], text=True))["streams"][0]
                    if int(stream["nb_read_frames"]) != 4 * case["chunks"] + 1 or stream["avg_frame_rate"] != "5/1":
                        raise ValueError("Actual video timeline differs from declared inclusive stitching")
            worker = json.loads((folder / "worker/worker.json").read_text())
            if (not set(worker["gpus"]) <= {0, 1, 2, 3}
                    or any(k.lower().endswith("_proxy") for k in worker["environment"])):
                raise ValueError("GPU/proxy restrictions violated")
            if mode == "ground_truth_visual_context":
                if len(report["anchor_events"]) != case["chunks"] - 1 or report["image_anchors_sha256"] != case["anchors_sha256"]:
                    raise ValueError("Wrong visual anchor evidence")
                for i, event in enumerate(report["anchor_events"], 1):
                    if event["source_frame"] != study["plan"]["start_frame"] + i * 12 or event["details"]["state_changed"] or event["details"]["step_index"] != i:
                        raise ValueError("Visual intervention changed time/state")
            critical = ["adapters/world_models/ctrl_world.py", "evaluation/datasets/droid.py", "evaluation/execution/native_replay.py"]
            stable = {name: report["framework_code_sha256"][name] == _digest(root / name) for name in critical}
            # Smoke predates a strict-length guard/helper extraction. Retain it
            # as historical smoke, but require all long jobs to match current code.
            if case["id"] != "ctrl_ep795_smoke" and not all(stable.values()):
                raise ValueError("Critical inference source changed since long replay launch")
            failed = directory / case["id"] / mode / "summary.json"
            prior = []
            if failed != folder / "summary.json" and failed.exists():
                old = json.loads(failed.read_text())
                prior.append({"report": str(failed.relative_to(root)), "sha256": _digest(failed),
                    "status": old["status"], "completed_wm_calls": old["world_model_calls"], "error": old.get("error"),
                    "interrupted_in_flight_call_count_unknown": True})
            jobs.append({"case": case["id"], "mode": mode, "world_model_calls": case["chunks"], "prior_attempts": prior,
                "frames": report["frames_per_view"], "source_end_frame": int(indices[-1, -1]),
                "source_control_seconds": case["chunks"] * .8, "dropped_tail_ticks": len(poses) - 1 - int(indices[-1, -1]),
                "critical_source_matches_current": stable, "report": str((folder / "summary.json").relative_to(root)),
                "report_sha256": _digest(folder / "summary.json"), "task_success": None})
    pair = directory / "ctrl_ep795"
    ar = json.loads((pair / "autoregressive/summary.json").read_text())
    anchored = json.loads((pair / "ground_truth_visual_context/summary.json").read_text())
    if ar["chunk_prediction_sha256"][0] != anchored["chunk_prediction_sha256"][0]:
        raise ValueError("Paired first chunks differ before intervention")
    reference = json.loads((directory / "official_reference/report.json").read_text())
    if reference["status"] != "passed" or not clean_worker_exit(reference["cleanup"]):
        raise ValueError("Official forward_wm parity not passed")
    smoke_path = directory / "current_source_smoke/summary.json"
    smoke = json.loads(smoke_path.read_text())
    if (smoke["status"] != "completed" or not clean_worker_exit(smoke["cleanup"])
            or smoke["world_model_calls"] != 1 or smoke["vla_calls"] != 0
            or smoke["chunk_prediction_sha256"][0] != ar["chunk_prediction_sha256"][0]
            or smoke["framework_code_sha256"]["adapters/world_models/ctrl_world.py"] != _digest(root / "adapters/world_models/ctrl_world.py")):
        raise ValueError("Current-source smoke failed or differs from the full replay first window")
    result = {"status": "passed_engineering_not_quality_approval", "jobs": jobs,
        "world_model_calls": sum(j["world_model_calls"] for j in jobs) + reference["world_model_calls"] + 1,
        "vla_calls": 0, "paired_first_chunk_exact": True, "official_reference": reference,
        "current_source_smoke": {"report": str(smoke_path.relative_to(root)), "sha256": _digest(smoke_path), "matches_full_first_chunk": True},
        "completed_calls_in_interrupted_attempts": sum(p["completed_wm_calls"] for j in jobs for p in j["prior_attempts"]),
        "quality_admitted": None, "task_success_evaluated": False, "human_review_performed": False}
    (directory / "execution_audit.json").write_text(json.dumps(result, indent=2))
    return result


def render_ctrl_validation(root, directory, *, cases=()):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text())
    references = json.loads((root / study["plan"]["source_study"] / "study.json").read_text())["references"]
    rows = []
    for case in study["cases"]:
        if case["id"].endswith("_smoke") or case["id"] == "ctrl_ep795":
            continue
        if cases and case["id"] not in cases:
            continue
        reference = next(r for r in references if r["episode"] == case["episode"])
        generated = case_run_directory(directory, case["id"], "autoregressive")
        output = directory / "comparisons" / case["id"]
        if output.exists():
            result = json.loads((output / "comparison.json").read_text())
            for item in result["inputs"].values():
                if _digest(Path(item["path"])) != item["sha256"]:
                    raise ValueError("Previously rendered comparison input changed")
            for name, sha in result["output_sha256"].items():
                if _digest(output / name) != sha:
                    raise ValueError("Previously rendered comparison output changed")
        else:
            result = render_droid_comparison(root,
                references={name: reference["views"][f"droid.{name}"]["file"] for name in VIEWS},
                generated={name: str(generated / (name + ".mp4")) for name in VIEWS},
                output=output, label=case["id"] + " inclusive-v2", source_ticks_per_frame=3, native_fps=5)
            result["output_sha256"] = {name: _digest(output / name) for name in result["files"]}
            (output / "comparison.json").write_text(json.dumps(result, indent=2))
        rows.append({"case": case["id"], "comparison": str((output / "comparison.mp4").relative_to(directory)),
            "report": result})
    paired = directory / "paired_comparison/ctrl_ep795/comparison.mp4"
    if not paired.is_file():
        raise ValueError("Render paired795 comparison first")
    cards = '<section><h2>episode795: REAL / AR / REAL-IMAGE-ANCHORED</h2><video controls preload="none" src="paired_comparison/ctrl_ep795/comparison.mp4"></video></section>'
    cards += ''.join(f'<section><h2>{html.escape(row["case"])}</h2><video controls preload="none" src="{row["comparison"]}"></video></section>' for row in rows)
    (directory / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Ctrl corrected protocol</title><style>body{font-family:system-ui;background:#181a1e;color:#eee;margin:24px}video{max-width:100%;width:768px}a{color:#acf}</style><h1>Ctrl-World inclusive timing + per-camera VAE</h1><p>Recorded measured actions. No VLA; no task-success approval. Reanchored795 is diagnostic, not autonomous closed loop.</p>'+cards)
    return {"status": "rendered_not_quality_approved", "cases": len(rows) + 1, "partial_selection": bool(cases)}
