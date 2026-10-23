#!/usr/bin/env python3
"""Freeze candidate configs and reference contact sheets; never infer success."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.datasets.droid import CAMERAS  # noqa: E402
from evaluation.datasets.success_samples import candidate_suite  # noqa: E402
from vlaworldeval.atomic_json import publish_json_exclusive  # noqa: E402


def main():
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 350, 550, 800, 950])
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Use a new output directory inside this project")
    root = ROOT / "datasets/cosmos3_droid/success"
    candidates = {}
    for path in sorted((root / "meta/episodes").glob("chunk-*/file-*.parquet")):
        for row in pq.read_table(path, filters=[("episode_index", "in", args.episodes)]).to_pylist():
            if row["episode_index"] in candidates:
                raise ValueError("Duplicate episode metadata")
            candidates[row["episode_index"]] = row
    if set(candidates) != set(args.episodes):
        raise ValueError("Requested metadata is not local")
    output.mkdir(parents=True)
    records = []
    templates = ["pi0_cosmos_droid_3gpu_v1", "pi05_cosmos_droid_3gpu_v1",
                 "engineering_groot_cosmos_v1", "groot_n17_cosmos_droid_3gpu_v1",
                 "pi0_ctrl_world_droid_v1", "pi05_ctrl_world_droid_v1",
                 "groot_n15_ctrl_world_droid_v1", "groot_n17_ctrl_world_droid_v1"]
    for episode in args.episodes:
        row = candidates[episode]
        sheet = Image.new("RGB", (5 * 256, 3 * 164), "white")
        draw = ImageDraw.Draw(sheet)
        for camera_index, (name, key) in enumerate(CAMERAS.items()):
            prefix = f"videos/{key}"
            path = root / f"{prefix}/chunk-{row[prefix+'/chunk_index']:03d}/file-{row[prefix+'/file_index']:03d}.mp4"
            for column, frame in enumerate([15, row["length"] // 4, row["length"] // 2,
                                             3 * row["length"] // 4, row["length"] - 1]):
                pixels = subprocess.check_output(["ffmpeg", "-v", "error",
                    "-ss", str(row[prefix+"/from_timestamp"] + frame / 15), "-i", str(path), "-frames:v", "1",
                    "-vf", "scale=256:144", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
                sheet.paste(Image.frombytes("RGB", (256, 144), pixels), (column*256, camera_index*164))
                draw.text((column*256, camera_index*164+144), f"{name} frame={frame}", fill="black")
        sheet.save(output / f"episode_{episode:06d}_reference.png")
        configs = []
        for template_name in templates:
            template = json.loads((ROOT / f"configs/experiments/{template_name}.json").read_text())
            horizon = 15 if "ctrl" in template_name else 16
            suite = candidate_suite(template, episode=episode, length=row["length"], source_horizon=horizon)
            if "ctrl" in template_name:
                suite["runtime"].update(vla_gpus=[3], world_model_gpus=[3])
            else:
                suite["runtime"].update(vla_gpus=[0], world_model_gpus=[0, 1, 2],
                    vla_memory_gib=20, max_process_memory_gib_per_gpu=40)
            path = output / f"{suite['suite_id']}.json"
            publish_json_exclusive(path, suite)
            configs.append(str(path.relative_to(ROOT)))
        records.append({"episode": episode, "length": row["length"], "tasks": row["tasks"],
                        "configs": configs, "reference_review": "pending", "task_success": None})
    publish_json_exclusive(output / "candidates.json", {"schema_version": 1, "records": records,
        "status": "candidate_set_not_success_claim", "automatic_task_scoring": False})
    print(json.dumps({"output": str(output), "episodes": args.episodes, "configs": len(records)*len(templates)}))


if __name__ == "__main__":
    main()
