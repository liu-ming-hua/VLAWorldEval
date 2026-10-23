#!/usr/bin/env python3
"""Prepare tiny, traceable fixtures from the VLAs' official datasets."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "datasets/vla_official_smoke"


def read_frame(video: Path, index: int) -> np.ndarray:
    # The official DROID and LIBERO files use AV1. PyAV provides the software
    # decoder needed on servers where OpenCV incorrectly selects unsupported
    # hardware AV1 decoding.
    with av.open(str(video)) as container:
        for current, frame in enumerate(container.decode(video=0)):
            if current == index:
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(f"Cannot decode frame {index} from {video}")


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"schema_version": 1}

    bridge_video = ROOT / "datasets/worldgym/official/bridge/test/000000002.mp4"
    bridge_image = OUTPUT / "bridge/frame_000.png"
    save_image(bridge_image, read_frame(bridge_video, 0))
    manifest["bridge"] = {
        "dataset": "WorldGym official Bridge sample",
        "source": str(bridge_video.relative_to(ROOT)),
        "frame_indices": [0],
        "instruction": "put the eggplant in the pot",
        "images": {"primary": str(bridge_image.relative_to(ROOT))},
    }

    droid_root = ROOT / "datasets/cosmos3_droid/success"
    droid_table = pd.read_parquet(droid_root / "data/chunk-000/file-000.parquet")
    droid_episode = pd.read_parquet(
        droid_root / "meta/episodes/chunk-000/file-000.parquet"
    ).iloc[0]
    droid_views = {
        "exterior_image_1_left": "observation.image.exterior_image_1_left",
        "exterior_image_2_left": "observation.image.exterior_image_2_left",
        "wrist_image_left": "observation.image.wrist_image_left",
    }
    droid_images: dict[str, list[str]] = {}
    for short_name, feature_name in droid_views.items():
        video = droid_root / "videos" / feature_name / "chunk-000/file-000.mp4"
        paths = []
        for frame_index in (0, 15):
            target = OUTPUT / "droid" / f"{short_name}_{frame_index:03d}.png"
            save_image(target, read_frame(video, frame_index))
            paths.append(str(target.relative_to(ROOT)))
        droid_images[short_name] = paths

    state_rows = droid_table.iloc[[0, 15]]
    state_path = OUTPUT / "droid/state.npz"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        state_path,
        joint_position=np.stack(state_rows["observation.state.joint_positions"]),
        cartesian_position=np.stack(state_rows["observation.state.cartesian_position"]),
        gripper_position=np.asarray(
            state_rows["observation.state.gripper_position"].tolist(), dtype=np.float32
        ).reshape(2, 1),
    )
    instruction = str(droid_episode["tasks"][0]).split(" | ")[0]
    manifest["droid"] = {
        "dataset": "LeRobot DROID success, official episode 0",
        "source": str(droid_root.relative_to(ROOT)),
        "episode_index": 0,
        "frame_indices": [0, 15],
        "instruction": instruction,
        "images": droid_images,
        "state": str(state_path.relative_to(ROOT)),
    }

    libero_root = ROOT / "datasets/lerobot_libero"
    if not (libero_root / "data/chunk-000/file-000.parquet").is_file():
        # This script is intentionally run with environment/vla/smolvla. Only
        # episode 0 and its two video streams are downloaded.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDataset(
            "lerobot/libero", root=libero_root, episodes=[0], download_videos=True
        )
    libero_table = pd.read_parquet(libero_root / "data/chunk-000/file-000.parquet")
    libero_images: dict[str, str] = {}
    for short_name in ("image", "image2"):
        video = (
            libero_root
            / "videos"
            / f"observation.images.{short_name}"
            / "chunk-000/file-000.mp4"
        )
        target = OUTPUT / "libero" / f"{short_name}_000.png"
        save_image(target, read_frame(video, 0))
        libero_images[short_name] = str(target.relative_to(ROOT))
    libero_state = np.asarray(libero_table.iloc[0]["observation.state"], dtype=np.float32)
    libero_state_path = OUTPUT / "libero/state.npy"
    libero_state_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(libero_state_path, libero_state)
    tasks = pd.read_parquet(libero_root / "meta/tasks.parquet")
    task_index = int(libero_table.iloc[0]["task_index"])
    task_row = tasks.loc[tasks["task_index"] == task_index].iloc[0]
    manifest["libero"] = {
        "dataset": "LeRobot LIBERO, official episode 0",
        "source": str(libero_root.relative_to(ROOT)),
        "episode_index": 0,
        "frame_indices": [0],
        # LeRobot v3 stores the natural-language task in this parquet's index.
        "instruction": str(task_row.name),
        "images": libero_images,
        "state": str(libero_state_path.relative_to(ROOT)),
        "adapter_projection": {
            "camera1": "observation.images.image",
            "camera2": "observation.images.image2",
            "state": "first 6 values required by smolvla_base config.json",
        },
    }

    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
