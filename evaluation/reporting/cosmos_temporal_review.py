"""Read-only temporal review of existing Cosmos controls; never a quality gate."""
import html
import json
from pathlib import Path
import re
import subprocess

import mediapy as media
import numpy as np
from PIL import Image, ImageDraw

from adapters.observations.droid import compose_droid_cosmos_view
from evaluation.planning.registry import project_path
from evaluation.reporting.task_scores import _digest
from vlaworldeval.paths import fresh_output_path

LABELS = ("REAL", "HISTORICAL SEED0", "SEED1", "SEED2")


def read_native_prefix(path, count):
    """Decode the original pixels, not a resized comparison thumbnail."""
    if type(count) is not int or count < 1:
        raise ValueError("Frame count must be a positive integer")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(path)],
        text=True, timeout=30))["streams"][0]
    if probe["avg_frame_rate"] != "15/1":
        raise ValueError("Temporal review requires source-aligned 15FPS inputs")
    data = subprocess.check_output(["ffmpeg", "-v", "error", "-threads", "2", "-i", str(path),
        "-frames:v", str(count), "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], timeout=60)
    frames = np.frombuffer(data, np.uint8).reshape(-1, probe["height"], probe["width"], 3)
    if len(frames) != count:
        raise ValueError("Incomplete temporal prefix; no padding or repeated frames")
    return frames


def _native_canvas(clips, tick, case, labels=LABELS):
    canvas = Image.new("RGB", (2560, 568), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for col, (label, frames) in enumerate(zip(labels, clips)):
        draw.text((col * 640 + 5, 5), f"{case} | {label} | tick={tick} t={tick/15:.3f}s", fill="white")
        canvas.paste(Image.fromarray(frames[tick]), (col * 640, 28))
    return canvas


def _render_case(output, key, inputs, ticks, *, generated_keys=("seed0_historical", "seed1", "seed2"), labels=LABELS):
    if len(generated_keys) != 3 or len(labels) != 4:
        raise ValueError("Temporal layout requires REAL and three explicit comparison roles")
    real = {name: read_native_prefix(inputs["real." + name]["path"], ticks + 1)
        for name in ("exterior_1", "exterior_2", "wrist")}
    clips = [compose_droid_cosmos_view(real["exterior_1"], real["exterior_2"], real["wrist"])]
    clips.extend(read_native_prefix(inputs[name]["path"], ticks + 1)
        for name in generated_keys)
    if any(clip.shape != (ticks + 1, 540, 640, 3) for clip in clips):
        raise ValueError("Require native DROID composites [T,540,640,3]")
    output.mkdir()
    sheets = []
    with media.VideoWriter(output / "comparison.mp4", (568, 2560), fps=15, crf=18,
            encoded_format="yuv420p", ffmpeg_args=("-threads", "2", "-movflags", "+faststart")) as writer:
        for tick in range(ticks + 1):
            canvas = _native_canvas(clips, tick, key, labels)
            writer.add_image(np.asarray(canvas))
            if tick in (32, ticks):
                canvas.save(output / f"native_t{tick:03d}.png")
        for first in range(0, ticks + 1, 8):
            indices = list(range(first, min(first + 8, ticks + 1)))
            # All frames are present, but these sheets are explicitly thumbnails.
            sheet = Image.new("RGB", (640, 153 * len(indices)), (20, 20, 20))
            draw = ImageDraw.Draw(sheet)
            for row, tick in enumerate(indices):
                draw.text((5, row * 153 + 2), f"{key} tick={tick} | " + " / ".join(labels), fill="white")
                for col, clip in enumerate(clips):
                    sheet.paste(Image.fromarray(clip[tick]).resize((160, 135)), (col * 160, row * 153 + 18))
            path = output / f"sheet_{first:03d}_{indices[-1]:03d}.jpg"
            sheet.save(path, quality=95)
            sheets.append({"file": path.name, "source_ticks": indices})
    return {"case": key, "frames": ticks + 1, "source_control_seconds": ticks / 15,
        "native_eef_chunks": ticks // 16, "inputs": inputs, "sheets": sheets, "column_roles": list(labels),
        "layout": "Each column: wrist640x360 above two exterior320x180; REAL exteriors resized by the official composition bridge",
        "sheet_resolution": "160x135/composite thumbnail; absence of fine defects cannot be certified",
        "quality_admitted": False, "task_success": None,
        "files": {p.name: _digest(p) for p in sorted(output.iterdir()) if p.is_file()}}


def export_cosmos_temporal_review(root, parent, output, *, cases=("eef_ep000351", "eef_ep000365", "eef_ep000903"), source_ticks=64):
    root = Path(root).resolve()
    parent = project_path(root, str(parent))
    output = fresh_output_path(root, output)
    if type(source_ticks) is not int or not 32 <= source_ticks <= 128 or source_ticks % 16:
        raise ValueError("Require complete EEF windows from32 to128 ticks")
    if (not cases or any(not isinstance(key, str) or not re.fullmatch(r"eef_ep\d{6}", key) for key in cases)
            or len(set(cases)) != len(cases)):
        raise ValueError("Require nonempty unique review cases")
    summary = json.loads((parent / "summary.json").read_text())
    if summary.get("status") != "completed_execution_and_assistant_screening_not_quality_approval":
        raise ValueError("Require the completed parent horizon study")
    for field, name in (("review_sha256", "quality_review_working.json"), ("comparisons_sha256", "comparisons.json"),
            ("execution_audit_sha256", "execution_audit.json")):
        if summary[field] != _digest(parent / name):
            raise ValueError("Parent evidence changed")
    rows = {row["case"]: row for row in json.loads((parent / "comparisons.json").read_text())["cases"]}
    if any(key not in rows or rows[key]["model"] != "eef" for key in cases):
        raise ValueError("Review cases must be existing EEF controls")
    for key in cases:
        for item in rows[key]["inputs"].values():
            if _digest(project_path(root, item["path"])) != item["sha256"]:
                raise ValueError("Parent input video changed")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "building", "schema_version": 1, "parent": str(parent),
        "parent_summary_sha256": _digest(parent / "summary.json"), "source_ticks": source_ticks,
        "selection": "Post-hoc diagnostic candidates from the fixed five-episode study; not held-out confirmation and not a new denominator",
        "new_world_model_calls": 0, "vla_calls": 0, "quality_admitted": False, "human_review_performed": False, "cases": []}
    (output / "study.json").write_text(json.dumps(report, indent=2))
    for key in cases:
        report["cases"].append(_render_case(output / key, key, rows[key]["inputs"], source_ticks))
    report["status"] = "exported_not_reviewed_or_quality_approved"
    (output / "study.json").write_text(json.dumps(report, indent=2))
    (output / "index.html").write_text(_review_page(report))
    return report


def _review_page(report):
    cards = []
    for row in report["cases"]:
        key = html.escape(row["case"], quote=True)
        sheets = " ".join(f'<a href="{key}/{s["file"]}">{s["source_ticks"][0]}–{s["source_ticks"][-1]}</a>' for s in row["sheets"])
        cards.append(f'<section><h2>{key}</h2><video controls preload="metadata" src="{key}/comparison.mp4"></video>'
            f'<p><button data-delta="-1">−1 frame</button> <button data-delta="1">+1 frame</button> '
            f'<input type="range" min="0" max="{report["source_ticks"]}" value="0"><output>0</output> '
            '<select><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></p>'
            f'<p>All-frame thumbnail sheets: {sheets}</p><p><a href="{key}/native_t032.png">Native tick32</a> · '
            f'<a href="{key}/native_t{report["source_ticks"]:03d}.png">Native end</a></p></section>')
    return '''<!doctype html><meta charset="utf-8"><title>Cosmos temporal review</title>
<style>body{font-family:system-ui;background:#181a1e;color:#eee;margin:24px}video{width:100%;max-width:2560px}a{color:#acf}input{width:40%}section{margin-bottom:40px}</style>
<h1>Real / historical0 / seed1 / seed2: temporal review</h1>
<p>One synchronized video per case; each column has wrist above exterior1/exterior2. Frame0 is the recorded reset.
Ticks16/32/48/64 are EEF chunk endpoints. Native spatial layout is preserved (not lossless encoding).
Export and thumbnail inspection are NOT human approval, physical validation or task success.</p>
<p><a href="study.json">Provenance</a> · <a href="REPORT_CN.md">Review notes (after review)</a></p>''' + ''.join(cards) + '''
<script>
document.querySelectorAll('section').forEach(s=>{
 const v=s.querySelector('video'),r=s.querySelector('input'),o=s.querySelector('output');
 const seek=n=>{v.pause();n=Math.max(0,Math.min(+r.max,n));r.value=n;o.textContent=`tick ${n}, ${(n/15).toFixed(3)}s`;v.currentTime=n/15;};
 r.addEventListener('input',()=>seek(+r.value));
 s.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>seek(+r.value+ +b.dataset.delta)));
 s.querySelector('select').addEventListener('change',e=>v.playbackRate=+e.target.value);
 const show=t=>{const n=Math.min(+r.max,Math.round(t*15));r.value=n;o.textContent=`tick ${n}, ${(n/15).toFixed(3)}s`;};
 if(v.requestVideoFrameCallback){const cb=(_,m)=>{show(m.mediaTime);v.requestVideoFrameCallback(cb);};v.requestVideoFrameCallback(cb);}
 else v.addEventListener('timeupdate',()=>show(v.currentTime));
});
</script>'''


def audit_cosmos_temporal_review(root, directory):
    """Audit files and declared inspection coverage, not the truth of judgments."""
    root = Path(root).resolve()
    directory = project_path(root, str(directory))
    report = json.loads((directory / "study.json").read_text())
    parent = project_path(root, report["parent"])
    if (report["status"] != "exported_not_reviewed_or_quality_approved"
            or report["parent_summary_sha256"] != _digest(parent / "summary.json")
            or report["new_world_model_calls"] != 0 or report["vla_calls"] != 0):
        raise ValueError("Parent or read-only review contract changed")
    probes = {}
    for row in report["cases"]:
        key = row["case"]
        if not re.fullmatch(r"eef_ep\d{6}", key):
            raise ValueError("Invalid review case")
        if [t for page in row["sheets"] for t in page["source_ticks"]] != list(range(report["source_ticks"] + 1)):
            raise ValueError("Temporal sheets omit or duplicate frames")
        names = [page["file"] for page in row["sheets"]]
        expected = {"comparison.mp4", "native_t032.png", f"native_t{report['source_ticks']:03d}.png", *names}
        if len(set(names)) != len(names) or set(row["files"]) != expected:
            raise ValueError("Review manifest does not cover every media file")
        for item in row["inputs"].values():
            if _digest(project_path(root, item["path"])) != item["sha256"]:
                raise ValueError("Reviewed input changed")
        for name, sha in row["files"].items():
            if Path(name).name != name or _digest(directory / key / name) != sha:
                raise ValueError("Review artifact changed")
        stream = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames,avg_frame_rate,width,height,codec_name,pix_fmt",
            "-of", "json", str(directory / key / "comparison.mp4")], text=True, timeout=30))["streams"][0]
        if (int(stream["nb_read_frames"]) != report["source_ticks"] + 1 or stream["avg_frame_rate"] != "15/1"
                or (stream["width"], stream["height"]) != (2560, 568)
                or stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"):
            raise ValueError("Actual temporal video differs from its contract")
        probes[key] = stream
    observations = directory / "observations.json"
    if observations.exists():
        notes = json.loads(observations.read_text())
        if (notes.get("reviewer") != "assistant" or notes.get("human_review_performed") is not False
                or notes.get("full_native_frame_review_performed") is not False
                or len(notes["cases"]) != len(report["cases"])
                or {r["case"] for r in notes["cases"]} != set(probes)):
            raise ValueError("Require explicit assistant/thumbnail scope and complete case coverage")
        for note in notes["cases"]:
            row = next(r for r in report["cases"] if r["case"] == note["case"])
            sheets = [p["file"] for p in row["sheets"]]
            if (note.get("inspected_sheets") != sheets or note.get("native_ticks") != [32, report["source_ticks"]]
                    or note.get("quality_admitted") is not False or note.get("task_success") is not None):
                raise ValueError("Incomplete declared evidence or implicit quality approval")
    result = {"status": "passed_export_artifact_audit_not_quality_approval", "cases": probes,
        "study_sha256": _digest(directory / "study.json"), "index_sha256": _digest(directory / "index.html"),
        "observations_sha256": _digest(observations) if observations.exists() else None,
        "new_world_model_calls": 0, "vla_calls": 0, "quality_admitted": False,
        "scope": "File identity, decoded timeline and declared inspection coverage; no certification of review judgments or browser interaction"}
    (directory / "audit.json").write_text(json.dumps(result, indent=2))
    return result
