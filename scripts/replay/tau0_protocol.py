#!/usr/bin/env python3
"""Exercise the released τ₀-WM Simulator server with its exact API contract.

This is deliberately a protocol smoke test, not a TacoPlay benchmark: the
only local TacoPlay asset has 7-D actions and cannot be used with the released
``simulator/`` checkpoint. The smoke test reuses its two initial RGB views and
sends a zero 20-D relative-EEF action chunk; the upstream server appends six
fixed controls to form the 26-D model input, so server/client wiring and model
loading can be checked without inventing an action conversion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def first_frame(path: Path) -> np.ndarray:
    encoded = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    )
    return np.asarray(Image.open(BytesIO(encoded)).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=Path,
                        default=ROOT / "datasets/tau0_wm/taco_play/sample_episode_000000")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8002)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/official_autoregressive/tau0_wm_protocol_smoke")
    parser.add_argument("--steps", type=int, default=33)
    args = parser.parse_args()
    from vlaworldeval.paths import fresh_output_path
    args.output = fresh_output_path(ROOT, args.output)

    from adapters.world_models.tau0_wm_simulator import Tau0WMSimulatorAdapter

    episode = args.episode.resolve()
    views = [
        first_frame(episode / "videos/rgb_static.mp4"),
        first_frame(episode / "videos/rgb_gripper.mp4"),
    ]
    actions = np.zeros((args.steps, 20), dtype=np.float32)
    args.output.mkdir(parents=True, exist_ok=False)
    adapter = Tau0WMSimulatorAdapter(
        host=args.server_host,
        port=args.server_port,
        repo_path=ROOT / "third_party/tau-0-wm",
        instruction="protocol smoke test",
        save_sim_video=True,
        sim_path=args.output / "rollout_server.mp4",
    )
    adapter.reset(views)
    terminal = adapter.step(actions)
    np.save(args.output / "actions_all.npy", actions)
    np.save(args.output / "terminal_views.npy", adapter.last_video)
    metadata = {
        "format_version": 1,
        "model": "τ₀-WM Simulator (20-D API / 26-D checkpoint)",
        "protocol": "upstream websocket simulator protocol smoke test",
        "source": str(episode),
        "input_views": ["rgb_static", "rgb_gripper"],
        "action_representation": "20-D relative EEF 6D API; server appends 6 head/waist/velocity controls",
        "action_shape": list(actions.shape),
        "synthetic_zero_action": True,
        "ground_truth_future_rgb_used": False,
        "state_output": None,
        "server_video": "rollout_server.mp4",
        "terminal_shape": list(np.asarray(terminal).shape),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    adapter.close()
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
