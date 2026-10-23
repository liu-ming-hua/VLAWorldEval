"""Three-column paired diagnostics. Pixel errors are not robot task scores."""
import html
import json
from pathlib import Path

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw

from evaluation.execution.reliability import clean_worker_exit
from evaluation.reporting.positive_controls import VIEWS, TILE, _read_tiles
from evaluation.planning.registry import project_path
from vlaworldeval.paths import fresh_output_path


def temporal_protocol_warning(case):
    if case.get("model") == "ctrl" and case["source_stride"] == 15:
        return ("Frozen Ctrl advances15 source ticks; upstream inclusive5-frame window advances12. "
                "This confounds pure visual-accumulation attribution; corrected protocol not tested here.")
    return None


def pixel_diagnostics(real, autoregressive, anchored, *, frames_per_chunk, temporal_layout="future"):
    """Native prediction sample times only; no held-display duplicates or labels."""
    if (not (real.shape == autoregressive.shape == anchored.shape) or real.ndim != 4
            or real.shape[-1] != 3 or real.shape[1] <= 12):
        raise ValueError("Pixel comparisons need aligned RGB video shapes")
    if frames_per_chunk < 1 or (len(real) - 1) % frames_per_chunk:
        raise ValueError("Require complete chunks after the single conditioning frame")
    # Ignore restored bottom border; same mask in both arms and the real view.
    def errors(value):
        first = 0 if temporal_layout == "inclusive" else 1
        return np.abs(real[first:, :-12].astype(np.float32) - value[first:, :-12].astype(np.float32)).mean(axis=(1, 2, 3))
    ar, gt = errors(autoregressive), errors(anchored)
    if temporal_layout == "inclusive":
        # Saved boundary belongs to the NEXT prediction, exactly as upstream.
        # Do not group it as the previous window's last future frame.
        count = (len(real) - 1) // frames_per_chunk
        groups = [(ar[i * frames_per_chunk:(i + 1) * frames_per_chunk if i < count - 1 else None],
                   gt[i * frames_per_chunk:(i + 1) * frames_per_chunk if i < count - 1 else None]) for i in range(count)]
        return {"intensity_scale": "0..255", "metric": "display-grid RGB L1; bottom 12 display rows excluded",
            "temporal_layout": "inclusive", "initial_predicted_frame_included": True,
            "autoregressive_mean": float(ar.mean()), "anchored_mean": float(gt.mean()),
            "anchored_minus_ar": float(gt.mean() - ar.mean()),
            "per_chunk": [{"chunk": i, "samples": len(a), "ar_mean": float(a.mean()), "anchored_mean": float(g.mean()),
                "ar_last_saved": float(a[-1]), "anchored_last_saved": float(g[-1])} for i, (a, g) in enumerate(groups)],
            "window_frame_offsets": list(range(frames_per_chunk)),
            "by_frame_offset": {"autoregressive": [float(ar[j:-1:frames_per_chunk].mean()) for j in range(frames_per_chunk)],
                "anchored": [float(gt[j:-1:frames_per_chunk].mean()) for j in range(frames_per_chunk)]},
            "final_terminal_only": {"autoregressive": float(ar[-1]), "anchored": float(gt[-1])},
            "task_success_evaluated": False, "physical_validity_verified": False}
    return {"intensity_scale": "0..255", "metric": "display-grid RGB L1; bottom 12 display rows excluded",
        "autoregressive_mean": float(ar.mean()), "anchored_mean": float(gt.mean()),
        "anchored_minus_ar": float(gt.mean() - ar.mean()),
        "per_chunk": [{"chunk": i, "ar_mean": float(a.mean()), "anchored_mean": float(g.mean()),
            "ar_terminal": float(a[-1]), "anchored_terminal": float(g[-1])}
            for i, (a, g) in enumerate(zip(ar.reshape(-1, frames_per_chunk), gt.reshape(-1, frames_per_chunk)))],
        "by_frame_offset": {"autoregressive": ar.reshape(-1, frames_per_chunk).mean(axis=0).tolist(),
                            "anchored": gt.reshape(-1, frames_per_chunk).mean(axis=0).tolist()},
        "future_frame_offsets": list(range(1, frames_per_chunk + 1)),
        "task_success_evaluated": False, "physical_validity_verified": False}


def _views(folder, config):
    if config["output_views"] == ["droid.cosmos_composite"]:
        from adapters.observations.droid import split_droid_cosmos_view
        frames = _read_tiles(folder / "droid.cosmos_composite.mp4", config["video"]["fps"], composite=True)
        split = split_droid_cosmos_view(frames)
        return {name: np.stack([np.asarray(Image.fromarray(frame).resize(TILE)) for frame in split[name]]) for name in VIEWS}
    return {name: _read_tiles(folder / ((f"droid.{name}" if f"droid.{name}" in config["output_views"] else name) + ".mp4"), config["video"]["fps"], letterbox=False) for name in VIEWS}


def compare_image_feedback(root, directory, output, *, cases=()):
    """Audit both arms, then render REAL / AR / real-image-anchored videos."""
    from evaluation.reporting.task_scores import _digest
    from evaluation.execution.transport import read_packet
    root = Path(root).resolve()
    directory = project_path(root, str(directory))
    output = fresh_output_path(root, output)
    study = json.loads((directory / "study.json").read_text())
    selected = [c for c in study["cases"] if not cases or c["id"] in cases]
    if not selected or set(cases) - {c["id"] for c in selected}:
        raise ValueError("Unknown or empty case selection")
    references = json.loads((root / study["plan"]["source_study"] / "study.json").read_text())["references"]
    prepared = []
    for case in selected:
        for field in ("input", "anchors", "config"):
            if _digest(project_path(root, case[field])) != case[field + "_sha256"]:
                raise ValueError("Frozen diagnostic inputs changed")
        config = json.loads(Path(case["config"]).read_text())
        expected_actions = read_packet(Path(case["input"]))["actions"]
        summaries = {}
        for mode in study["plan"]["modes"]:
            folder = directory / case["id"] / mode
            report = json.loads((folder / "summary.json").read_text())
            if (report["status"] != "completed" or not clean_worker_exit(report["cleanup"])
                    or report["vla_calls"] != 0 or report["world_model_calls"] != case["chunks"]):
                raise ValueError("Both arms must have complete, clean executions")
            if report["input_sha256"] != case["input_sha256"] or report["config_sha256"] != case["config_sha256"] or report["feedback_mode"] != mode:
                raise ValueError("Paired execution input/config/mode mismatch")
            for filename, expected in report["files"].items():
                if _digest(folder / filename) != expected:
                    raise ValueError("Replay artifact checksum mismatch")
            if not np.array_equal(np.load(folder / "actions.npy", allow_pickle=False), expected_actions):
                raise ValueError("Executed actions differ from the frozen shared packet")
            summaries[mode] = report
        ar, anchored = (summaries[m] for m in study["plan"]["modes"])
        events = anchored["anchor_events"]
        if (ar["anchor_events"] or len(events) != case["chunks"] - 1
                or anchored["image_anchors_sha256"] != case["anchors_sha256"]):
            raise ValueError("Visual interventions differ from the declared protocol")
        for i, event in enumerate(events, 1):
            if event["chunk"] != i or event["source_frame"] != study["plan"]["start_frame"] + i * case["source_stride"] or event["details"]["state_changed"] is not False or event["details"]["step_index"] != i:
                raise ValueError("Visual reanchor changed time/state or used wrong image time")
        reference = next(r for r in references if r["episode"] == case["episode"])
        for item in reference["views"].values():
            if _digest(project_path(root, item["file"])) != item["sha256"]:
                raise ValueError("Real reference checksum mismatch")
        prepared.append((case, config, summaries, reference))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case, config, summaries, reference in prepared:
        folder = output / case["id"]
        folder.mkdir()
        real = {name: _read_tiles(root / reference["views"]["droid." + name]["file"], 15, letterbox=False) for name in VIEWS}
        arms = {mode: _views(directory / case["id"] / mode, config) for mode in study["plan"]["modes"]}
        from evaluation.manifests import VideoSpec
        video_spec = VideoSpec.from_dict(config["video"])
        native_frames = video_spec.frame_count(case["chunks"])
        stride = case["source_stride"] // video_spec.advance_frames
        if stride * video_spec.advance_frames != case["source_stride"]:
            raise ValueError("Display needs integer source tick spacing")
        ticks = np.arange((native_frames - 1) * stride + 1)
        indices = ticks // stride
        if any(len(v) < len(ticks) for v in real.values()) or any(len(v) != native_frames for arm in arms.values() for v in arm.values()):
            raise ValueError("No padding/trimming of incomplete diagnostic videos")
        scores = {name: pixel_diagnostics(real[name][np.arange(native_frames)*stride],
            arms["autoregressive"][name], arms["ground_truth_visual_context"][name],
            frames_per_chunk=video_spec.advance_frames, temporal_layout=video_spec.temporal_layout) for name in VIEWS}
        shape = (526, 768)
        chosen = set(np.linspace(0, len(ticks)-1, 5, dtype=int).tolist())
        samples = []
        with media.VideoWriter(folder / "comparison.mp4", shape, fps=15, crf=20, encoded_format="yuv420p",
                ffmpeg_args=("-threads", "2", "-movflags", "+faststart")) as writer:
            for tick, index in zip(ticks, indices):
                canvas = Image.new("RGB", (768, 526), (20, 20, 20))
                draw = ImageDraw.Draw(canvas)
                if video_spec.temporal_layout == "inclusive":
                    part = f"prediction_chunk={min(tick//case['source_stride'], case['chunks']-1)}"
                else:
                    part = "condition" if tick == 0 else f"prediction_chunk={(tick-1)//case['source_stride']}"
                draw.text((5, 5), f"{case['id']} t={tick/15:.2f}s {part} | RGB anchors only; NOT closed loop", fill="white")
                for row, name in enumerate(VIEWS):
                    top = 28 + row * 166
                    images = (real[name][tick], arms["autoregressive"][name][index], arms["ground_truth_visual_context"][name][index])
                    for col, (label, frame) in enumerate(zip(("REAL", "AR", "REAL-IMAGE-ANCHORED"), images)):
                        draw.text((col*256+5, top+3), f"{label} {name}", fill="white")
                        canvas.paste(Image.fromarray(frame), (col*256, top+22))
                writer.add_image(np.asarray(canvas))
                if int(tick) in chosen:
                    samples.append(canvas.copy())
        sheet = Image.new("RGB", (768*len(samples), 526))
        for i, sample in enumerate(samples):
            sheet.paste(sample, (768*i, 0))
        sheet.save(folder / "contact_sheet.jpg", quality=95)
        first_equal = summaries["autoregressive"]["chunk_prediction_sha256"][0] == summaries["ground_truth_visual_context"]["chunk_prediction_sha256"][0]
        row = {"case": case["id"], "initial_chunk_exact_match": first_equal,
            "temporal_protocol_warning": temporal_protocol_warning(case),
            "official_temporal_protocol_aligned": False if temporal_protocol_warning(case) else None,
            "pairing_warning": None if first_equal else "First chunks differ before intervention; do not attribute later differences solely to visual feedback",
            "source_ticks_per_generated_frame": stride,
            "source_control_seconds": int(ticks[-1])/15, "comparison_frames": len(ticks),
            "video_sha256": _digest(folder / "comparison.mp4"), "pixel_diagnostics": scores,
            "columns": ["real", "autoregressive", "ground_truth_visual_context"],
            "display_geometry": "each camera resized to common grid; no camera calibration or model input change",
            "boundary_note": "No intermediate real image inserted in generated videos; anchored chunks may jump at boundaries",
            "saved_initial_frame_source": "generated_window_start" if video_spec.temporal_layout == "inclusive" else "recorded_initial_image",
            "source_reports": {mode: {"path": str(directory/case["id"]/mode/"summary.json"),
                "sha256": _digest(directory/case["id"]/mode/"summary.json")} for mode in study["plan"]["modes"]},
            "task_success_evaluated": False, "quality_admitted": None, "human_review_performed": False}
        (folder / "comparison.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
    cards = ''.join(f'<section><h2>{html.escape(r["case"])}</h2><video controls preload="none" width="768" src="{r["case"]}/comparison.mp4"></video><p><a href="{r["case"]}/contact_sheet.jpg">Frames</a> · <a href="{r["case"]}/comparison.json">Diagnostics</a></p></section>' for r in rows)
    warnings = ''.join('<p>'+html.escape(r["temporal_protocol_warning"])+'</p>' for r in rows if r["temporal_protocol_warning"])
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Image feedback diagnostics</title><style>body{font-family:system-ui;background:#17191c;color:#eee;margin:24px}a{color:#9cf}video{max-width:100%}</style><h1>REAL / AR / real-image-anchored</h1><p>Diagnostic only. Recorded images condition the third column at chunk boundaries. No VLA, no task-success approval.</p>'+warnings+cards)
    result = {"status": "comparisons_built_not_quality_approved", "cases": rows,
        "model_calls_during_comparison": 0, "pixel_metrics_are_not_task_scores": True}
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    return result
