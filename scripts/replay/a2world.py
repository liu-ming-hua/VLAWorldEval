#!/usr/bin/env python3
"""Autoregressive replay using A2World's bundled official DROID example."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_ASSET = ROOT / "third_party/a2world/world_model/examples/assets/pretrained/droid"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/world_models/a2world/a2world-pretrained.pt"
DEFAULT_BASE = ROOT / "checkpoints/world_models/a2world/base"


def save_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3], got {frames.shape}")
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{frames.shape[2]}x{frames.shape[1]}", "-r", str(fps), "-i", "-",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(stderr[-2000:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-checkpoints", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=ROOT / "results/official_autoregressive/a2world_droid_official")
    parser.add_argument("--sampling-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from vlaworldeval.paths import fresh_output_path
    args.output = fresh_output_path(ROOT, args.output)

    asset = args.asset_dir.resolve()
    image_paths = [asset / "exterior_1.png", asset / "wrist.png", asset / "exterior_2.png"]
    action_path = asset / "actions.npz"
    for path in (*image_paths, action_path, args.checkpoint, args.base_checkpoints):
        if not path.exists():
            raise FileNotFoundError(path)
    actions = np.asarray(np.load(action_path)["actions"], dtype=np.float32)
    if actions.shape != (20, 14):
        raise ValueError(f"official A2World DROID example must be [20,14], got {actions.shape}")
    views = [np.asarray(Image.open(path).convert("RGB")) for path in image_paths]

    from adapters.world_models.a2world import A2WorldAdapter

    args.output.mkdir(parents=True, exist_ok=False)
    adapter = A2WorldAdapter(
        checkpoint=args.checkpoint,
        base_checkpoints=args.base_checkpoints,
        variant="pretrained",
        repo_path=ROOT / "third_party/a2world/world_model",
        device="cuda",
        view_ids=(2, 0, 3),
        width=256,
        height=256,
        num_sampling_steps=args.sampling_steps,
        seed=args.seed,
    )
    adapter.reset(views)
    generated_chunks = []
    chunks = []
    for chunk_index, chunk in enumerate(np.array_split(actions, 1)):
        generated = np.asarray(adapter.rollout(chunk))  # [T,V,H,W,3]
        generated_chunks.append(np.concatenate([generated[:, i] for i in range(generated.shape[1])], axis=2))
        np.save(args.output / f"actions_chunk_{chunk_index:03d}.npy", chunk)
        chunks.append({"chunk_index": chunk_index, "action_shape": list(chunk.shape), "video_shape": list(generated.shape)})

    video = np.concatenate(generated_chunks, axis=0)
    save_video(args.output / "rollout_long.mp4", video, fps=10)
    np.save(args.output / "actions_all.npy", actions)
    metadata = {
        "format_version": 1,
        "model": "A2World-pretrained",
        "protocol": "official bundled A2World DROID replay",
        "source": str(asset),
        "input_views": ["exterior_1", "wrist", "exterior_2"],
        "view_ids": [2, 0, 3],
        "action_representation": "official A2World shared dual-arm delta pose + gripper",
        "action_shape": list(actions.shape),
        "num_action_chunks": len(chunks),
        "num_output_frames": int(len(video)),
        "video_fps": 10,
        "state_output": None,
        "checkpoint": str(args.checkpoint.resolve()),
        "chunks": chunks,
        "note": "The bundled pretrained DROID example contains one complete 20-action chunk; no artificial action repetition is used.",
    }
    args.output.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
