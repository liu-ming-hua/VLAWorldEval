"""Admission for upstream WorldGym Bridge MP4/NPZ examples."""
import json
from pathlib import Path
import subprocess

import numpy as np

from evaluation.datasets.episode import EpisodeInput


def load_worldgym_episode(root: Path, episode: int, start: int, chunks: int, horizon: int,
                         fps: int, *, instruction: str, replay: bool):
    video = root / f"{episode:09d}.mp4"
    actions = video.with_suffix(".npz")
    with np.load(actions, allow_pickle=False) as packet:
        values = packet["arr_0"]
    if values.ndim != 2 or values.shape[1] != 10 or not np.isfinite(values).all():
        raise ValueError("WorldGym sample requires finite [T,10] actions")
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(video),
    ], text=True))["streams"][0]
    numerator, denominator = map(int, info["avg_frame_rate"].split("/"))
    if numerator / denominator != fps or int(info["nb_frames"]) != len(values):
        raise ValueError("WorldGym video/action lengths or FPS do not match")
    if not 0 <= start < len(values) or not instruction.strip():
        raise ValueError("WorldGym initial frame or instruction is invalid")
    stop = start + chunks * horizon + 1
    if replay and stop > len(values):
        raise ValueError("Incomplete WorldGym replay action window; padding is forbidden")
    pixels = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(video), "-ss", str(start / fps), "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    image = np.frombuffer(pixels, dtype=np.uint8).reshape(info["height"], info["width"], 3).copy()
    if image.shape != (256, 256, 3):
        raise ValueError("This WorldGym acceptance pair requires the official 256x256 sample")
    return EpisodeInput(
        trial={"instruction": instruction, "observation_bundle": {"views": {"bridge.primary": image}}},
        replay_actions=[values[i:i + horizon].copy() for i in range(start + 1, stop, horizon)] if replay else [],
        provenance={"episode_index": episode, "start_frame": start, "end_frame": stop - 1,
                    "fps": fps, "instruction": instruction, "instruction_source": "suite_annotation_not_in_official_npz",
                    "video_path": str(video.resolve()), "action_path": str(actions.resolve()), "length": len(values),
                    "conditioning_frames": 1, "imagined_beyond_sample": not replay and stop > len(values),
                    "replay_action_source": "official WorldGym arr_0; action[t] accompanies frame[t]; skip anchor action"},
    )
