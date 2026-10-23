#!/usr/bin/env python3
"""Invoke Giga-World-1's upstream inference entrypoint on official Toydata."""

from __future__ import annotations

import os
import argparse
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vlaworldeval.paths import gigaworld1_source  # noqa: E402
from adapters.environment import get_environment  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
GIGA_ROOT = gigaworld1_source()
TOYDATA = ROOT / "datasets/gigaworld1/official/toydata"
OUTPUT = ROOT / "results/official_autoregressive/gigaworld1_toydata_official"


def main() -> None:
    from vlaworldeval.paths import fresh_output_path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = fresh_output_path(ROOT, args.output)
    # These are the same model/config/checkpoint choices as the upstream
    # scripts/infer/run_infer_nano_i2v.sh; only the input/output locations are
    # project-local so that the result is tracked with the other replays.
    command = [
        str(get_environment("world_models.gigaworld1").python),
        str(GIGA_ROOT / "infer/infer_giga_world.py"),
        "--config", str(GIGA_ROOT / "scripts/training/configs/stage_1_post_functrl_wan21.yaml"),
        "--base_model_path", str(GIGA_ROOT / "model/stage1/nano/Giga-World-1-nano-stage1_final-diffusers"),
        "--transformer_model_name_or_path", str(GIGA_ROOT / "model/stage1/nano/Giga-World-1-nano-stage1_final-diffusers"),
        "--checkpoint_path", str(GIGA_ROOT / "model/stage1/nano/Giga-World-1-nano-stage1_scene_lora"),
        "--image_path", str(TOYDATA / "infer_assest/input_image.png"),
        "--control_video_path", str(TOYDATA / "infer_assest/control_video.mp4"),
        "--prompt", "stack the box ears . The scene features a flat, gray surface with a subtle wood-grain texture, serving as the entire visible background. The only object present is a plain brown cardboard box lying open and centered on this surface.",
        "--output_dir", str(output),
        "--sample_name", "toydata_i2v",
        "--seed", "42",
        "--fps", "10",
        "--num_frames", "99",
        "--height", "480",
        "--width", "1920",
        "--num_inference_steps", "20",
        "--guidance_scale", "5.0",
        "--enable_tiling", "True",
    ]
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    env.update({
        "HF_ENABLE_PARALLEL_LOADING": "yes",
        "HF_PARALLEL_LOADING_WORKERS": "8",
        "TOKENIZERS_PARALLELISM": "false",
        "FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:512",
    })
    output.mkdir(parents=True, exist_ok=False)
    print("Running upstream:", " ".join(command), flush=True)
    subprocess.run(command, cwd=GIGA_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
