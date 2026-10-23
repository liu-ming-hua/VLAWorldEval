"""Source-aligned multi-seed diagnostics. No pixel-to-task-success shortcut."""
import html
import json
from pathlib import Path
import subprocess

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw

from adapters.observations.droid import split_droid_cosmos_view
from evaluation.execution.reliability import clean_worker_exit
from evaluation.execution.transport import read_packet
from evaluation.reporting.positive_controls import _read_tiles, TILE, VIEWS
from evaluation.reporting.task_scores import _digest


def reference_motion(poses, gripper, start, horizons):
    """Measured motion, not automatically inferred grasp/contact annotations."""
    poses, gripper = np.asarray(poses), np.asarray(gripper).reshape(-1)
    if (poses.ndim != 2 or poses.shape[1] != 6 or len(poses) != len(gripper)
            or not np.isfinite(poses).all() or not np.isfinite(gripper).all()
            or type(start) is not int or start < 0 or not horizons
            or any(type(h) is not int or h <= 0 or start + h >= len(poses) for h in horizons)):
        raise ValueError("Reference motion requires complete finite pose/gripper windows")
    rows = []
    for h in horizons:
        p, g = poses[start:start + h + 1], gripper[start:start + h + 1]
        rows.append({"source_ticks": h, "seconds": h / 15,
            "eef_translation_path_m": float(np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1).sum()),
            "eef_start_to_end_m": float(np.linalg.norm(p[-1, :3] - p[0, :3])),
            "measured_gripper_range": float(np.ptp(g)),
            "gripper_threshold_crossings_source_frames": (start + np.flatnonzero((g[1:] > .5) != (g[:-1] > .5)) + 1).tolist()})
    return rows


def write_reference_motion(directory):
    import pyarrow.parquet as pq
    directory = Path(directory)
    study = json.loads((directory / "study.json").read_text())
    rows = []
    for episode in study["plan"]["episodes"]:
        job = next(j for j in study["jobs"] if j["model"] == "eef" and j["episode"] == episode)
        packet = read_packet(Path(job["input"]))
        source = packet["source"]
        table = pq.read_table(source["data_path"], filters=[("episode_index", "=", episode)],
            columns=["observation.state.cartesian_position", "observation.state.gripper_position"])
        values = reference_motion(table["observation.state.cartesian_position"].to_pylist(),
            table["observation.state.gripper_position"].to_pylist(), study["plan"]["start_frame"], study["plan"]["horizon_source_ticks"])
        rows.append({"episode": episode, "instruction": source["instruction"], "reference_motion": values})
    result = {"status": "recorded_motion_diagnostics_not_contact_labels", "cases": rows,
        "note": "Path length/gripper crossings do not certify visibility, contact, grasp or task completion"}
    (directory / "reference_motion.json").write_text(json.dumps(result, indent=2))
    return result


def horizon_scores(real, prediction, horizons):
    """Dependent prefix diagnostics, not independent short-rollout trials."""
    if real.shape != prediction.shape or real.ndim != 4 or real.shape[-1] != 3 or real.shape[1] <= 12:
        raise ValueError("Require aligned RGB clips and enough rows for border exclusion")
    if not horizons or any(type(h) is not int or not 0 < h < len(real) for h in horizons):
        raise ValueError("Horizon exceeds available real/predicted future; no padding")
    error = np.abs(real[1:, :-12].astype(np.float32) - prediction[1:, :-12].astype(np.float32)).mean(axis=(1, 2, 3))
    return {"metric": "display-grid RGB L1,0..255; initial real frame and bottom12 display rows excluded",
        "prefixes": [{"source_ticks": h, "seconds": h / 15, "mean": float(error[:h].mean()),
            "terminal": float(error[h - 1])} for h in horizons],
        "incremental_32_tick_windows": [{"start_tick": start + 1, "end_tick": min(start + 32, len(error)),
            "mean": float(error[start:start + 32].mean())} for start in range(0, len(error), 32)],
        "prefixes_are_independent_replications": False, "task_success_evaluated": False}


def _camera_tiles(path, max_frames=None):
    composite = _read_tiles(path, 15, composite=True, max_frames=max_frames)
    split = split_droid_cosmos_view(composite)
    return {name: np.stack([np.asarray(Image.fromarray(frame).resize(TILE)) for frame in split[name]]) for name in VIEWS}


def add_hold_diagnostic(row, real, horizons):
    """Auxiliary no-motion baseline, never an automatic quality gate."""
    hold = {name: horizon_scores(frames, np.repeat(frames[:1], len(frames), axis=0), horizons) for name, frames in real.items()}
    row["hold_initial_baseline"] = hold
    row["model_minus_hold_prefix_l1"] = {seed: {name: [float(a["mean"] - b["mean"])
        for a, b in zip(views[name]["prefixes"], hold[name]["prefixes"])] for name in VIEWS}
        for seed, views in row["metrics"].items()}
    row["hold_baseline_scope"] = "Auxiliary fixed initial-image diagnostic; negative pixel delta is not action-following, physical validity or task success"


def _completed_progress(directory, study):
    progress = json.loads((directory / "progress.json").read_text())
    if (progress.get("status") != "completed"
            or progress.get("study_sha256") != _digest(directory / "study.json")
            or set(progress.get("jobs", {})) != {job["id"] for job in study["jobs"]}):
        raise ValueError("Audit requires the complete, unchanged frozen queue")
    for job in study["jobs"]:
        entry = progress["jobs"][job["id"]]
        if (entry.get("status") != "completed" or entry.get("world_model_calls") != job["chunks"]
                or entry.get("report_sha256") != _digest(directory / "runs" / job["id"] / "summary.json")):
            raise ValueError("Completed progress/report mismatch: " + job["id"])


def _probe_prefix_video(path, geometry, *, frames=129):
    stream = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames,avg_frame_rate,codec_name,pix_fmt,width,height",
        "-of", "json", str(path)], text=True, timeout=30))["streams"][0]
    if (int(stream["nb_read_frames"]) != frames or stream["avg_frame_rate"] != "15/1"
            or (stream["width"], stream["height"]) != geometry
            or stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"):
        raise ValueError(f"Actual video differs from the {frames}-frame/15FPS browser-compatible prefix: " + str(path))
    return stream


def audit_horizon_study(root, directory):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text())
    _completed_progress(directory, study)
    if _digest(Path(study["environments"])) != study["environments_sha256"]:
        raise ValueError("Frozen environment override changed")
    results = []
    critical = ["adapters/world_models/cosmos3_nano_droid_fd.py", "adapters/world_models/cosmos3_droid_joint_fd.py",
        "adapters/backends/cosmos.py", "adapters/backends/cosmos_actions.py", "adapters/actions/droid.py",
        "adapters/observations/droid.py", "evaluation/execution/native_replay.py", "evaluation/execution/worker.py",
        "evaluation/contracts.py", "evaluation/manifests.py"]
    source_hashes = {name: _digest(root / name) for name in critical}
    for baseline in study["baselines"]:
        for path, sha in ((baseline["video"], baseline["video_sha256"]), (baseline["report"], baseline["report_sha256"])):
            if _digest(Path(path)) != sha:
                raise ValueError("Historical baseline changed")
        jobs = [j for j in study["jobs"] if (j["model"], j["episode"]) == (baseline["model"], baseline["episode"])]
        for job in jobs:
            for key in ("input", "config"):
                if _digest(Path(job[key])) != job[key + "_sha256"]:
                    raise ValueError("Frozen study inputs changed")
            folder = directory / "runs" / job["id"]
            report = json.loads((folder / "summary.json").read_text())
            from evaluation.execution.native_study import resolved_native_config
            actual_config_path, actual_config_sha = resolved_native_config(root, directory, job)
            if (report["status"] != "completed" or not clean_worker_exit(report["cleanup"])
                    or report["world_model_calls"] != job["chunks"] or report["vla_calls"] != 0
                    or report["frames_per_view"] != 129 or report["anchor_events"]
                    or report["feedback_mode"] != "autoregressive"
                    or report["config_sha256"] != actual_config_sha or report["input_sha256"] != job["input_sha256"]):
                raise ValueError("Incomplete or protocol-inconsistent native replay: " + job["id"])
            for filename, sha in report["files"].items():
                if _digest(folder / filename) != sha:
                    raise ValueError("Replay output changed")
            probe = _probe_prefix_video(folder / "droid.cosmos_composite.mp4", (640, 540))
            packet = read_packet(Path(job["input"]))
            if not np.array_equal(packet["actions"], np.load(folder / "actions.npy", allow_pickle=False)):
                raise ValueError("Executed actions differ from frozen packet")
            if any(report["framework_code_sha256"].get(name) != sha for name, sha in source_hashes.items()):
                raise ValueError("Inference-critical source changed across new seed jobs")
            for key in ("torch", "diffusers", "transformers", "numpy"):
                if baseline["packages"].get(key) != report["runtime"]["packages"].get(key):
                    raise ValueError("Historical/new inference environment differs for " + key)
            worker = json.loads((folder / "worker/worker.json").read_text())
            actual_config = json.loads(actual_config_path.read_text())
            if (set(worker["gpus"]) != {0, 1, 2, 3}
                    or worker["gpus"] != actual_config["gpus"] or worker["memory_gib"] != actual_config["memory_gib"]
                    or worker["environment"].get("HF_HUB_OFFLINE") != "1"
                    or worker["environment"].get("TRANSFORMERS_OFFLINE") != "1"
                    or any(key.lower().endswith("_proxy") for key in worker["environment"])):
                raise ValueError("GPU/allocator/offline/proxy constraint violated")
            results.append({"id": job["id"], "world_model_calls": report["world_model_calls"],
                "report_sha256": _digest(folder / "summary.json"), "packages_match_historical": True,
                "actual_config": str(actual_config_path), "actual_config_sha256": actual_config_sha,
                "source_config_sha256": job["config_sha256"], "actual_video_probe": probe})
    result = {"status": "passed_artifact_and_environment_audit_not_quality_approval", "jobs": results,
        "new_world_model_calls": sum(j["world_model_calls"] for j in results), "vla_calls": 0,
        "historical_seed0_baselines": len(study["baselines"]), "task_success_evaluated": False,
        "new_seed_inference_critical_sources": source_hashes,
        "source_version_limit": "Historical seed0 source is retained, not re-executed; initial/action/environment alignment is audited, not a bit-exact source-reproduction claim"}
    (directory / "execution_audit.json").write_text(json.dumps(result, indent=2))
    return result


def render_horizon_study(root, directory, *, selection=()):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    study = json.loads((directory / "study.json").read_text())
    references = json.loads((root / study["plan"]["reference_study"] / "study.json").read_text())["references"]
    horizons = study["plan"]["horizon_source_ticks"]
    maximum = max(horizons)
    rows = []
    for baseline in study["baselines"]:
        key = f"{baseline['model']}_ep{baseline['episode']:06d}"
        if selection and key not in selection:
            continue
        output = directory / "comparisons" / key
        if output.exists():
            row = json.loads((output / "comparison.json").read_text())
            for item in row["inputs"].values():
                if _digest(Path(item["path"])) != item["sha256"]:
                    raise ValueError("Previously rendered input changed")
            for name, sha in row["output_sha256"].items():
                if _digest(output / name) != sha:
                    raise ValueError("Previously rendered output changed")
            if "same_input_seed1_seed2_pixel_distance" not in row:
                # Upgrade a derived diagnostic only after checking its frozen
                # media. Never rerun inference or replace an experiment video.
                first = _camera_tiles(Path(row["inputs"]["seed1"]["path"]), maximum + 1)
                second = _camera_tiles(Path(row["inputs"]["seed2"]["path"]), maximum + 1)
                row["same_input_seed1_seed2_pixel_distance"] = {name: horizon_scores(
                    first[name][:maximum + 1], second[name][:maximum + 1], horizons) for name in VIEWS}
                row["derived_metric_upgrade"] = "seed spread added; source and comparison videos unchanged"
                (output / "comparison.json").write_text(json.dumps(row, indent=2))
            if "hold_initial_baseline" not in row:
                real = {name: _read_tiles(Path(row["inputs"]["real." + name]["path"]), 15, letterbox=False, max_frames=maximum + 1) for name in VIEWS}
                add_hold_diagnostic(row, real, horizons)
                (output / "comparison.json").write_text(json.dumps(row, indent=2))
            rows.append(row)
            continue
        reference = next(r for r in references if r["episode"] == baseline["episode"])
        real = {name: _read_tiles(root / reference["views"][f"droid.{name}"]["file"], 15, letterbox=False, max_frames=maximum + 1) for name in VIEWS}
        paths = {"seed0_historical": Path(baseline["video"])}
        for seed in study["plan"]["new_seeds"]:
            job = next(j for j in study["jobs"] if j["model"] == baseline["model"] and j["episode"] == baseline["episode"] and j["seed"] == seed)
            paths[f"seed{seed}"] = directory / "runs" / job["id"] / "droid.cosmos_composite.mp4"
        generated = {name: _camera_tiles(path, maximum + 1) for name, path in paths.items()}
        if any(len(v) < maximum + 1 for v in real.values()) or any(len(v) < maximum + 1 for views in generated.values() for v in views.values()):
            raise ValueError("Incomplete video; no temporal padding permitted")
        real = {name: video[:maximum + 1] for name, video in real.items()}
        generated = {seed: {name: frames[:maximum + 1] for name, frames in views.items()} for seed, views in generated.items()}
        metrics = {seed: {name: horizon_scores(real[name], views[name], horizons) for name in VIEWS} for seed, views in generated.items()}
        spread = {name: horizon_scores(generated["seed1"][name], generated["seed2"][name], horizons) for name in VIEWS}
        inputs = {"real." + name: {"path": str(root / reference["views"][f"droid.{name}"]["file"]),
            "sha256": reference["views"][f"droid.{name}"]["sha256"]} for name in VIEWS}
        inputs.update({seed: {"path": str(path), "sha256": _digest(path)} for seed, path in paths.items()})
        if any(_digest(Path(item["path"])) != item["sha256"] for item in inputs.values()):
            raise ValueError("Reference checksum mismatch")
        output.mkdir(parents=True, exist_ok=False)
        chosen = {0, 16, 32, 64, 96, 128}
        samples = []
        with media.VideoWriter(output / "comparison.mp4", (526, 1024), fps=15, crf=20, encoded_format="yuv420p",
                ffmpeg_args=("-threads", "2", "-movflags", "+faststart")) as writer:
            for tick in range(maximum + 1):
                canvas = Image.new("RGB", (1024, 526), (20, 20, 20))
                draw = ImageDraw.Draw(canvas)
                draw.text((5, 5), f"{key} source t={tick/15:.2f}s | real / historical0 / seed1 / seed2 | NO VLA", fill="white")
                for row, name in enumerate(VIEWS):
                    top = 28 + row * 166
                    frames = [real[name][tick]] + [views[name][tick] for views in generated.values()]
                    for col, (label, frame) in enumerate(zip(("REAL", "SEED0 historical", "SEED1", "SEED2"), frames)):
                        draw.text((col * 256 + 5, top + 3), label + " " + name, fill="white")
                        canvas.paste(Image.fromarray(frame), (col * 256, top + 22))
                writer.add_image(np.asarray(canvas))
                if tick in chosen:
                    samples.append(canvas)
        sheet = Image.new("RGB", (1024 * len(samples), 526))
        for i, sample in enumerate(samples):
            sheet.paste(sample, (1024 * i, 0))
        sheet.save(output / "contact_sheet.jpg", quality=95)
        row = {"case": key, "model": baseline["model"], "episode": baseline["episode"],
            "frames": maximum + 1, "source_control_seconds": maximum / 15, "metrics": metrics,
            "same_input_seed1_seed2_pixel_distance": spread,
            "inputs": inputs, "native_action_stride": baseline["source_stride"],
            "prefixes_are_independent_replications": False, "quality_admitted": None,
            "task_success_evaluated": False, "human_review_performed": False,
            "historical_full_video": baseline["video"],
            "output_sha256": {name: _digest(output / name) for name in ("comparison.mp4", "contact_sheet.jpg")}}
        add_hold_diagnostic(row, real, horizons)
        (output / "comparison.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
    cards = ''.join(f'<section><h2>{html.escape(row["case"])}</h2><video controls preload="none" src="comparisons/{row["case"]}/comparison.mp4"></video><p><a href="comparisons/{row["case"]}/contact_sheet.jpg">Six timepoints</a> · <a href="comparisons/{row["case"]}/comparison.json">Provenance / diagnostics</a></p></section>' for row in rows)
    (directory / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Cosmos source-time horizons</title><style>body{font-family:system-ui;background:#181a1e;color:#eee;margin:24px}a{color:#acf}video{width:1024px;max-width:100%}</style><h1>Real / historical seed0 / seed1 / seed2</h1><p>Source-aligned prefixes; no VLA, no state intervention, no automatic task/quality approval. Native horizons and action domains differ: do not rank checkpoints by pixel errors.</p>'+cards)
    result = {"status": "rendered_not_quality_approved", "cases": rows, "partial_selection": bool(selection), "new_model_calls": 0}
    (directory / "comparisons.json").write_text(json.dumps(result, indent=2))
    return result


def summarize_horizon_study(root, directory):
    """Bind conservative assistant screening to completed, hash-checked media.

    The reviewer supplies observations; this never converts pixels to approval.
    Three checked snapshots cannot locate the true onset of a failure.
    """
    from evaluation.planning.registry import project_path
    root, directory = Path(root).resolve(), Path(directory).resolve()
    audit = audit_horizon_study(root, directory)
    study = json.loads((directory / "study.json").read_text())
    expected = {f"{b['model']}_ep{b['episode']:06d}" for b in study["baselines"]}
    comparison = json.loads((directory / "comparisons.json").read_text())
    if comparison["partial_selection"] or {c["case"] for c in comparison["cases"]} != expected:
        raise ValueError("Final summary requires the full comparison gallery")
    review_path = directory / "quality_review_working.json"
    review = json.loads(review_path.read_text())
    if (review.get("reviewer") != "assistant" or review.get("human_review_performed") is not False
            or len(review["cases"]) != len(expected) or {r["case"] for r in review["cases"]} != expected):
        raise ValueError("Require a complete explicitly non-human assistant screening")
    horizons = study["plan"]["horizon_source_ticks"]
    rows = []
    for row in review["cases"]:
        _validate_screening_row(row, horizons)
        for item in row["evidence"] + row.get("additional_evidence", []):
            path = project_path(root, item["path"])
            if not path.is_relative_to(directory / "review") or _digest(path) != item["sha256"]:
                raise ValueError("Screening evidence changed or is outside its review directory")
        case_dir = directory / "comparisons" / row["case"]
        stored = json.loads((case_dir / "comparison.json").read_text())
        if stored != next(c for c in comparison["cases"] if c["case"] == row["case"]):
            raise ValueError("Comparison index differs from its case")
        for name, sha in stored["output_sha256"].items():
            if _digest(case_dir / name) != sha:
                raise ValueError("Reviewed comparison media changed")
        for item in stored["inputs"].values():
            if _digest(Path(item["path"])) != item["sha256"]:
                raise ValueError("Compared source media changed")
        failures = [h for h in horizons if row["decisions"][str(h)] == "reject"]
        rows.append({**row, "first_observed_issue_tick": min(failures) if failures else None,
            "true_failure_onset_measured": False, "comparison_sha256": _digest(case_dir / "comparison.json"),
            "actual_video_probe": _probe_prefix_video(case_dir / "comparison.mp4", (1024, 526))})
    rows.sort(key=lambda r: (int(r["case"].split("ep")[-1]), r["case"]))
    counts = {str(h): {decision: sum(r["decisions"][str(h)] == decision for r in rows)
        for decision in ("uncertain", "reject")} for h in horizons}
    result = {"status": "completed_execution_and_assistant_screening_not_quality_approval",
        "new_jobs": len(audit["jobs"]), "new_world_model_calls": audit["new_world_model_calls"],
        "vla_calls": 0, "historical_seed0_baselines": len(study["baselines"]),
        "execution_audit_sha256": _digest(directory / "execution_audit.json"),
        "review_sha256": _digest(review_path), "comparisons_sha256": _digest(directory / "comparisons.json"),
        "training_overlap": study["plan"].get("train_overlap", "unknown"),
        "counts_at_checked_timepoints": counts, "cases": rows, "quality_admitted_cases": 0,
        "human_review_performed": False, "task_success_evaluated": False,
        "seed_schedule": "base_seed + chunk_index; overlapping offset schedules, no IID claim or confidence intervals",
        "limits": ["Dependent prefixes, not independent horizon trials", "Historical seed0 not re-executed",
            "Snapshot screen cannot certify preceding continuous intervals", "EEF raw-crop / joint restored feedback differ",
            "Joint recurrent q is last-command proxy", "No task ranking, real execution or state ablation"]}
    probe = directory / "feedback_preprocessing_diagnostic.json"
    probe_note = ""
    if probe.exists():
        diagnostic = json.loads(probe.read_text())
        for name, sha in diagnostic["upstream_source_sha256"].items():
            if _digest(project_path(root, name)) != sha:
                raise ValueError("Auxiliary preprocessing source changed")
        result["auxiliary_preprocessing_sha256"] = _digest(probe)
        probe_note = "官方[CPU预处理检查](feedback_preprocessing_diagnostic.json)没有发现保留内容缩放；它不是模型推理的因果干预。"
    (directory / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    columns = " | ".join(f"{h/15:.2f}秒" for h in horizons)
    table = "\n".join("| " + row["case"] + " | " + " | ".join(row["decisions"][str(h)] for h in horizons) + " |" for row in rows)
    report = f"""# Cosmos固定DROID预测时域：阶段结果

[同步视频](index.html) · [结构化结果](summary.json) · [执行审计](execution_audit.json) · [筛查证据](quality_review_working.json)

已完成{len(audit['jobs'])}项新增回放、{audit['new_world_model_calls']}次WM调用、0次VLA调用，
复用{len(study['baselines'])}条历史seed0长AR。五条固定DROID数据、两种社区checkpoint、
新seed1/2，GPU0～3。新增视频各129帧、15FPS，控制跨度128tick＝8.53秒。

## 结果含义

| 模型与episode | {columns} |
| --- | {" | ".join("---" for _ in horizons)} |
{table}

这里的reject是抽样时间点已有明确反例，uncertain是证据不足，不是通过。
这是助手对截图的保守筛查，不是完整逐帧视频审查或人类标注。
目前质量准入为0，任务成功均未判定；不能据此评判VLA好坏或给模型排名。
每个时间点的说明、明确异常seed和截图哈希见结构化结果。

## 保留的边界

- 32/64/128tick是同一条回放的相关前缀，不是独立长度消融；首次观察到异常不等于真实失效起点。
- 历史seed0不是当前源码重执行，新seed1/2共享推理关键源码；不声称三seed逐位源码复现。
- 每段seed为base_seed+chunk_index，偏移序列有重叠；不假定独立统计重复，不报告置信区间。
- EEF来自实测EEF轨迹转换，joint来自关节命令，二者动作域不同；像素诊断不能跨模型排名。
- EEF后续反馈528×640原生裁边图像，joint反馈540×640恢复画布；未做配对干预，不能把漂移全归因于checkpoint。
- joint后续关节state是最后命令目标代理，不是实测state或WM预测state；本阶段没有state干预。
- 大部分完整任务事件晚于8.53秒；低像素误差、流畅视频或静态场景都不足以证明动作跟随及任务成功。
- 这五条数据与社区checkpoint训练样本的重叠未知，不声称独立held-out泛化评测。

## 下一步

先连续复核候选短窗口的机器人与物体关系；必要时配对检查边界处理对反馈的影响。
{probe_note}
不能将裁边差异直接称作退化根因。
只有真实动作正对照获得可核验的质量证据后，才进入同窗口VLA实验。
不启动StateBridge训练、state消融或新增checkpoint搜索来掩盖当前限制。
"""
    (directory / "REPORT_CN.md").write_text(report)
    return result


def _validate_screening_row(row, horizons):
    keys = {str(h) for h in horizons}
    evidence_ticks = [item.get("source_ticks") for item in row.get("evidence", [])]
    if (set(row.get("decisions", {})) != keys or set(row.get("notes", {})) != keys
            or set(row.get("explicit_issue_seeds", {})) != keys
            or row.get("quality_admitted") is not False or row.get("task_success") is not None
            or len(evidence_ticks) != len(horizons) or any(type(t) is not int for t in evidence_ticks)
            or set(evidence_ticks) != set(horizons)):
        raise ValueError("Screening requires all checked timepoints, evidence and no implicit approval")
    for h in keys:
        seeds = row["explicit_issue_seeds"][h]
        if (row["decisions"][h] not in {"reject", "uncertain"}
                or not isinstance(row["notes"][h], str) or not row["notes"][h].strip()
                or not isinstance(seeds, list)
                or any(type(seed) is not int or seed not in {0, 1, 2} for seed in seeds)
                or len(set(seeds)) != len(seeds)
                or (row["decisions"][h] == "reject" and not seeds)):
            raise ValueError("Screening needs conservative decisions, notes and explicit counterexample seeds")
