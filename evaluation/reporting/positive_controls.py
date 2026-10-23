"""Read-only inputs -> synchronized REAL/generated DROID review videos, not scores."""
import hashlib
import html
import json
from pathlib import Path
import subprocess

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw

from evaluation.planning.registry import project_path
from vlaworldeval.paths import fresh_output_path

VIEWS = ("exterior_1", "exterior_2", "wrist")
TILE = (256, 144)


def write_comparison_gallery(output, entries):
    """Portable local HTML; no server, remote assets, scripts or quality labels."""
    cards = []
    for entry in entries:
        label = html.escape(entry["label"])
        directory = html.escape(entry.get("directory", entry["label"]), quote=True)
        cards.append(f'<section><h2>{label}</h2><video controls preload="none" width="512" '
            f'src="{directory}/comparison.mp4"></video><p><a href="{directory}/contact_sheet.jpg">'
            f'Contact sheet</a> · <a href="{directory}/comparison.json">Provenance</a></p></section>')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>DROID positive controls</title>'
        '<style>body{font-family:system-ui;margin:24px;background:#17191c;color:#eee}a{color:#9cf}'
        'main{display:flex;flex-wrap:wrap;gap:24px}section{max-width:540px}h2{font-size:16px}'
        'video{max-width:100%;background:black}</style><h1>REAL vs generated DROID controls</h1>'
        '<p>Left: recorded real video. Right: generated video. Three synchronized camera rows.</p>'
        '<p>Display follows source control time, not native output playback FPS. Generated frames may be '
        'held for display only. No VLA, no reference feedback, no automatic quality approval.</p><main>'
        + ''.join(cards) + '</main></html>')
    with (Path(output) / "index.html").open("x") as stream:
        stream.write(page)


def aligned_indices(generated_frames, source_ticks_per_frame):
    """Hold generated display frames; never interpolate motion or model inputs."""
    if type(generated_frames) is not int or generated_frames < 2:
        raise ValueError("Need the initial frame and at least one generated frame")
    if type(source_ticks_per_frame) is not int or source_ticks_per_frame < 1:
        raise ValueError("Require a declared positive integer source-tick stride")
    ticks = np.arange((generated_frames - 1) * source_ticks_per_frame + 1)
    return ticks, ticks // source_ticks_per_frame


def _read_tiles(path, fps, *, composite=False, letterbox=True, max_frames=None):
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError("max_frames must be a positive integer or None")
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate", "-of", "json", str(path)], text=True))["streams"][0]
    top, bottom = map(int, info["avg_frame_rate"].split("/"))
    if abs(top / bottom - fps) > 1e-6:
        raise ValueError(f"Unexpected native FPS for {path}: {top}/{bottom}, expected {fps}")
    width, height = (512, 432) if composite else TILE
    resize = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    if not letterbox:
        resize = f"scale={width}:{height}"
    if composite:
        # A native 540x640 RoboLab image has exact 2/3 wrist +1/3 exteriors.
        # Preserve that geometry at 432x512; split before thumbnail letterboxing.
        resize = "scale=512:432"
    limit = [] if max_frames is None else ["-frames:v", str(max_frames)]
    data = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path),
        "-vf", resize, "-vsync", "0", *limit, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(data, np.uint8).reshape(-1, height, width, 3)


def render_droid_comparison(root, *, references, generated, output, label,
                            source_ticks_per_frame, native_fps, source_fps=15, composite=None):
    """No GT feedback. This is post-hoc display, retaining native source videos."""
    root = Path(root).resolve()
    output = fresh_output_path(root, output)
    if set(references) != set(VIEWS) or (composite is None and set(generated) != set(VIEWS)):
        raise ValueError("Need three distinct, named DROID views")
    if composite is not None and generated:
        raise ValueError("Select generated views or a native composite, not both")
    paths = {"real." + key: project_path(root, str(value)) for key, value in references.items()}
    if len(set(paths.values())) != len(VIEWS):
        raise ValueError("Real camera files must be distinct")
    real = {key: _read_tiles(paths["real." + key], source_fps) for key in VIEWS}
    if composite is not None:
        from adapters.observations.droid import split_droid_cosmos_view
        paths["generated.composite"] = project_path(root, str(composite))
        frames = _read_tiles(paths["generated.composite"], native_fps, composite=True)
        split = split_droid_cosmos_view(frames)
        imagined = {key: np.stack([np.asarray(Image.fromarray(frame).resize(TILE)) for frame in split[key]])
                    for key in VIEWS}
    else:
        paths.update({"generated." + key: project_path(root, str(value)) for key, value in generated.items()})
        if len({paths["generated." + key] for key in VIEWS}) != len(VIEWS):
            raise ValueError("Generated camera files must be distinct, or declare a composite explicitly")
        imagined = {key: _read_tiles(paths["generated." + key], native_fps) for key in VIEWS}
    counts = {len(value) for value in imagined.values()}
    if len(counts) != 1:
        raise ValueError("Generated camera frame counts differ")
    ticks, indices = aligned_indices(counts.pop(), source_ticks_per_frame)
    if any(len(value) < len(ticks) for value in real.values()):
        raise ValueError("Real reference is too short; no reference padding or frame repetition")
    shape = (28 + 3 * (22 + TILE[1]), 2 * TILE[0])
    samples, chosen = [], set(np.linspace(0, len(ticks) - 1, 5, dtype=int).tolist())
    output.mkdir(parents=True, exist_ok=False)
    with media.VideoWriter(output / "comparison.mp4", shape, fps=source_fps,
            crf=20, encoded_format="yuv420p", ffmpeg_args=("-threads", "2", "-movflags", "+faststart")) as writer:
        for tick, index in zip(ticks, indices):
            canvas = Image.new("RGB", (shape[1], shape[0]), (20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 5), f"REAL | GENERATED: {label} | t={tick/source_fps:.2f}s", fill="white")
            for row, key in enumerate(VIEWS):
                top = 28 + row * (22 + TILE[1])
                draw.text((5, top + 3), f"REAL {key}", fill="white")
                draw.text((TILE[0] + 5, top + 3), f"GENERATED {key}", fill="white")
                canvas.paste(Image.fromarray(real[key][tick]), (0, top + 22))
                canvas.paste(Image.fromarray(imagined[key][index]), (TILE[0], top + 22))
            writer.add_image(np.asarray(canvas))
            if int(tick) in chosen:
                samples.append(canvas.copy())
    sheet = Image.new("RGB", (shape[1] * len(samples), shape[0]))
    for index, canvas in enumerate(samples):
        sheet.paste(canvas, (index * shape[1], 0))
    sheet.save(output / "contact_sheet.jpg", quality=90)
    report = {"schema_version": 1, "label": label, "model_inference_performed": False,
        "quality_admitted": None, "human_review_performed": False,
        "source_fps": source_fps, "native_generated_fps": native_fps,
        "source_ticks_per_generated_frame": source_ticks_per_frame,
        "time_origin": "first frame of the real reference clips, the conditioning time",
        "source_control_seconds": int(ticks[-1]) / source_fps, "comparison_frames": len(ticks),
        "display_only_resampling": "hold generated frame to source tick; never changes model inputs or native files",
        "view_order": list(VIEWS), "columns": ["real", "generated"],
        "inputs": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   for key, path in paths.items()}, "files": ["comparison.mp4", "contact_sheet.jpg"]}
    (output / "comparison.json").write_text(json.dumps(report, indent=2))
    return report


def compare_frozen_study(root, directory, output):
    """Audit and render existing single/full EEF controls; never launch models."""
    from evaluation.reporting.artifact_checks import audit_attempt
    root = Path(root).resolve()
    directory = project_path(root, str(directory))
    output = fresh_output_path(root, output)
    study = json.loads((directory / "study.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    refs = {r["episode"]: r for r in study["references"]}
    prepared = []
    for detail in summary["details"]:
        report_path = project_path(root, detail["report"])
        report = json.loads(report_path.read_text())
        if report.get("execution_performed") is not True or not audit_attempt(report["execution"])["ok"]:
            raise ValueError(f"Positive-control execution did not pass re-audit: {report_path}")
        ref = refs[detail["episode"]]
        references = {key: ref["views"]["droid." + key]["file"] for key in VIEWS}
        for key in VIEWS:
            expected = ref["views"]["droid." + key]["sha256"]
            if hashlib.sha256(project_path(root, references[key]).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen reference checksum mismatch")
        for job in report["execution"]["jobs"]:
            if job["id"] not in {"real_action_smoke", "real_action_full"}:
                continue
            if job["vla_calls"] != 0:
                raise ValueError("Positive controls cannot consume VLA-generated actions")
            folder = project_path(root, job["output"])
            metadata = json.loads((folder / "metadata.json").read_text())
            pair = json.loads((folder / "pair_manifest.json").read_text())
            wm = pair["world_model"]
            from evaluation.manifests import ActionSpec, VideoSpec
            action_spec, video_spec = ActionSpec.from_dict(wm["action"]), VideoSpec.from_dict(wm["video"])
            stride = study["spec"]["fps"] * action_spec.advance_steps / (action_spec.rate_hz * video_spec.advance_frames)
            if stride != int(stride):
                raise ValueError("This review renderer requires integer source-tick spacing")
            prepared.append({"references": references, "generated": {key: folder / metadata["view_files"]["droid." + key] for key in VIEWS},
                "name": f"ep{detail['episode']:06d}_{detail['world_model']}_{job['id']}",
                "source_ticks_per_frame": int(stride), "native_fps": metadata["video_fps"]})
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for item in prepared:
        name = item.pop("name")
        rows.append(render_droid_comparison(root, **item, output=output / name, label=name))
    result = {"schema_version": 1, "status": "comparisons_built_not_human_reviewed", "model_calls": 0,
              "source_study": str(directory), "comparisons": rows}
    (output / "index.json").write_text(json.dumps(result, indent=2))
    write_comparison_gallery(output, rows)
    return {"status": result["status"], "comparisons": len(rows), "output": str(output)}
