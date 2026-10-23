"""Prepare genuine DreamTrue geometry with the pinned official dataset loader."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from adapters.world_models.dreamtrue import PROTOCOL, SOURCE_REVISION, VIEW_NAMES, validate_condition
from vlaworldeval.paths import project_root


def prepare_official(*, source="droid", index=0, raw_root=None, with_reference=False):
    """One official sample, no retries, synthetic RGB, action renderer or replay loop.

    Uses the official fail-fast data method instead of __getitem__, whose
    exception retry loop can run forever when upstream raw data is missing.
    """
    root = project_root()
    repo = root / "third_party/dreamtrue"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    if revision != SOURCE_REVISION or dirty:
        raise ValueError("DreamTrue preparation requires the pinned source revision")
    sys.path.insert(0, str(repo / "wmvideo"))
    release = root / "datasets/dreamtrue/release"
    if source not in {"agibot", "droid", "robomind"}:
        raise ValueError("Unsupported released source")
    raw = Path(raw_root).resolve() if raw_root else root / "datasets/dreamtrue/raw" / source
    if not raw.is_relative_to(root.resolve()):
        raise ValueError("Raw dataset root must remain project-local")
    os.environ[f"{source.upper()}_DATA_ROOT"] = str(raw)
    os.environ[f"{source.upper()}_CONDITION_ROOT"] = str(release / "condition/conditions" / source)
    os.environ["AGIBOT_VACE_CONDITION_MODE"] = "condition_h5"
    from examples.wanvideo.model_training.data.dataset_factory import build_single_dataset, load_dataset_config
    config = load_dataset_config(str(release / "condition/data-config/config.yaml"))
    selected = next(value for value in config["datasets"].values() if value["type"] == source)
    dataset = build_single_dataset(selected, {
        "num_frames": 101, "height": 240, "width": 320, "repeat": 1,
        "min_interval": 3, "max_interval": 3, "load_workers": 1,
        "pad_short_actions": True, "random_sample_start": False,
        "detail_prompt": True, "use_plucker": True, "seed": 0})
    if not 0 <= index < len(dataset.scenes):
        raise ValueError("Official sample index outside manifest")
    sample = dataset._get_video_frames(index, np.random.default_rng(0))
    clip = dataset.scenes[index]
    condition_path = Path(dataset._condition_h5_path(clip)).resolve()
    if not condition_path.is_relative_to(root.resolve()):
        raise ValueError("Condition path leaves the project")
    packet = pack_official_sample(sample, condition_path=str(condition_path), source=source, clip=clip)
    from vlaworldeval.assets import file_record
    packet["condition"]["source"]["condition_asset"] = file_record(condition_path)
    if source == "droid":
        import h5py
        trajectory = Path(dataset._trajectory_path(clip["source_key"], root=dataset._clip_data_root(clip)))
        with h5py.File(condition_path) as condition, h5py.File(trajectory) as original:
            if condition.attrs["data_key"] != clip["source_key"] or condition.attrs["use_state"]:
                raise ValueError("Condition identity/type differs from the official action-rendered manifest")
            indices = condition["indices"][:].astype(np.int64)
            rendered = condition["qpos"][:, :7]
            commanded = original["action/joint_position"][:][indices, :7]
            error = float(np.max(np.abs(rendered - commanded)))
            if not np.allclose(rendered, commanded, atol=1e-6, rtol=0):
                raise ValueError(f"Raw DROID actions do not match the rendered condition: max error {error}")
        metadata = dataset._metadata(clip["source_key"], root=dataset._clip_data_root(clip))
        packet["condition"]["source"].update(raw_trajectory=str(trajectory),
            action_joint_max_abs_error=error, frames_verified=len(indices),
            camera_paths={"head": metadata["left_mp4_path"], "hand_left": metadata["right_mp4_path"],
                          "hand_right": metadata["wrist_mp4_path"]})
        provenance = trajectory.parent / "dreamtrue-raw-provenance.json"
        if provenance.is_file():
            import json
            packet["condition"]["source"]["raw_provenance"] = {
                "path": str(provenance), **file_record(provenance), "record": json.loads(provenance.read_text())}
    if with_reference:
        count = packet["condition"]["n_real_frames"]
        reference = {name: np.stack([np.asarray(f) for f in sample[f"{name}_video"]])[-count:]
                     for name in VIEW_NAMES}
        return packet, reference
    return packet


def pack_official_sample(sample, *, condition_path, source, clip):
    """Only map official loader outputs; preserve uint16 depth and OpenCV c2w."""
    condition = {"schema_version": 1, "protocol": PROTOCOL,
        "n_real_frames": sample.get("n_real_frames", 101), "pad_mode": "front",
        "source": {"kind": "official_rendered_geometry", "condition_path": condition_path,
                   "dataset": source, "source_key": clip.get("source_key"), "source_revision": SOURCE_REVISION},
        "views": {name: {
            "rgb": np.stack(sample[f"{name}_vace_rgb_video"]),
            "depth": np.stack(sample[f"{name}_vace_depth_video"]),
            "intrinsic": np.asarray(sample[f"{name}_vace_intrinsic"]),
            "extrinsic": np.asarray(sample[f"{name}_vace_extrinsic"]),
        } for name in VIEW_NAMES}}
    validate_condition(condition, num_frames=101, height=240, width=320)
    reset = {"schema_version": 1, "protocol": PROTOCOL, "prompt": sample["prompt"],
             "views": {name: np.asarray(sample[f"{name}_video"][0]) for name in VIEW_NAMES}}
    return {"schema_version": 1, "protocol": PROTOCOL, "reset": reset, "condition": condition}
