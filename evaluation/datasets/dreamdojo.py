"""Deterministic single-episode GR1 packet preparation, without GPU allocation.

Matches groot_dreams/groot_configs.py and WrappedLeRobotSingleDataset's GR1
encoding, including the first-row state-key alias in the pinned release.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PARTS = ("left_arm", "right_arm", "left_hand", "right_hand", "waist")
INLAB_TASKS = ("pnp_handover_plate", "pnp_cucumber", "pnp_corn", "pnp_dragonfruit",
               "pour_items_into_basket", "right_to_left_handover_corn", "mug", "fold_cloth_long", "fold_cloth", "color_puzzles")


def episode_paths(dataset, episode):
    if type(episode) is not int or episode < 0:
        raise ValueError("Episode must be an explicit nonnegative integer")
    info = json.loads((Path(dataset)/"meta/info.json").read_text())
    chunk_size = info["chunks_size"]
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("Invalid official episode chunk size")
    fields = {"episode_index": episode, "episode_chunk": episode//chunk_size,
              "video_key": "observation.images.ego_view_freq20"}
    paths = [Path(dataset)/info[key].format(**fields) for key in ("data_path", "video_path")]
    if any(not path.resolve().is_relative_to(Path(dataset).resolve()) for path in paths):
        raise ValueError("Official metadata must not escape its dataset directory")
    return tuple(paths)


def validate_prepared_packet(packet):
    from adapters.world_models.dreamdojo import SOURCE_REVISION
    source = packet.get("source", {})
    if (source.get("normalization_source") != "pinned_source/shared_meta/GR1_unified_stats.json"
            or source.get("source_revision") != SOURCE_REVISION):
        raise ValueError("Unsupported or superseded GR1 packet; prepare native_v2 with pinned unified statistics")
    actions = np.asarray(packet["actions"])
    if actions.shape not in {(1, 12, 384), (4, 12, 384)}:
        raise ValueError("Only native13 or49-frame GR1 windows are admitted")
    if len(actions) == 4 and source.get("multiround_protocol") != "official_global_alias_12_action_windows":
        raise ValueError("Multi-round GR1 actions require explicit global state-alias provenance")


def normalize_parts(rows, modality, statistics, statistics_modality=None):
    pieces = []
    for part in PARTS:
        field = modality[part]
        start, end = field["start"], field["end"]
        values = np.asarray(rows[field["original_key"]], dtype=np.float32)[..., start:end]
        stat_field = (statistics_modality or modality)[part]
        stats = statistics[stat_field["original_key"]]
        stat_start, stat_end = stat_field["start"], stat_field["end"]
        low = np.asarray(stats["min"], dtype=np.float32)[stat_start:stat_end]
        high = np.asarray(stats["max"], dtype=np.float32)[stat_start:stat_end]
        mask = low != high
        normalized = np.zeros_like(values)
        normalized[..., mask] = (values[..., mask] - low[mask]) / (high[mask] - low[mask])
        normalized[..., mask] = normalized[..., mask] * 2 - 1
        pieces.append(normalized)
    result = np.concatenate(pieces, axis=-1)
    if result.shape[-1] != 29 or not np.isfinite(result).all():
        raise ValueError("Invalid GR1 normalized joints")
    return result


def encode_actions(normalized, initial_state, chunks):
    if chunks not in {1, 4}:
        raise ValueError("Unsupported multi-chunk count; use the13 or49-frame official protocol")
    normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.shape != (chunks * 12 + 2, 29) or np.asarray(initial_state).shape != (29,):
        raise ValueError("GR1 sample requires a prefix baseline plus N*12+1 frames")
    actions = np.zeros((chunks * 12, 384), dtype=np.float32)
    for start in range(1, chunks * 12 + 1, 4):
        actions[start - 1:start + 3, :29] = normalized[start:start + 4] - normalized[start - 1]
    # Upstream `key = action_seq[0:1,:29]; key[:] = state` modifies row 0.
    # Preserve that released behavior, and identify it explicitly in provenance.
    actions[0, :29] = initial_state
    return actions.reshape(chunks, 12, 384)


def prepare_packet(dataset: Path, config_path: Path, output: Path, *, chunks=1, start=0, episode=0):
    if chunks < 1 or start < 0:
        raise ValueError("chunks must be positive and start nonnegative")
    if chunks not in {1, 4}:
        raise ValueError("Only native13 or49-frame GR1 preparation is declared")
    runtime = json.loads(Path(config_path).read_text())
    if chunks == 4 and runtime.get("kwargs", {}).get("native_multiround") is not True:
        raise ValueError("49-frame preparation requires an explicit native_multiround config")
    if output.exists():
        raise FileExistsError(output)
    import pyarrow.parquet as pq
    import cv2
    import torch
    from torchvision.transforms.v2 import functional as VF
    from torchvision.transforms import InterpolationMode
    from evaluation.execution.transport import write_packet
    from adapters.world_models.dreamdojo import SOURCE_REVISION
    from vlaworldeval.assets import verify_snapshot
    from vlaworldeval.paths import project_root

    dataset = Path(dataset).resolve()
    report = verify_snapshot(dataset.parents[1])  # official/<In-lab_Eval>/<dataset>
    metadata = json.loads((dataset / "meta/modality.json").read_text())
    shared_meta = project_root() / "third_party/dreamdojo/shared_meta"
    # The pinned official loader has has_default_meta=True, so GR1 always
    # takes normalization statistics from the source's 44-D unified metadata,
    # even though this evaluation episode has a different 1030-D raw layout.
    statistics = json.loads((shared_meta / "GR1_unified_stats.json").read_text())
    stats_modality = json.loads((shared_meta / "GR1_unified_modality.json").read_text())
    table_path, video_path = episode_paths(dataset, episode)
    table = pq.read_table(table_path)
    indices = start + np.arange(chunks * 12 + 2) * 2
    if indices[-1] >= len(table):
        raise ValueError("Requested sample exceeds the single downloaded episode")
    rows = {name: np.asarray(table[name].take(indices).to_pylist(), dtype=np.float32)
            for name in ("action", "observation.state")}
    normalized = normalize_parts(rows, metadata["action"], statistics, stats_modality["action"])
    state = normalize_parts(rows, metadata["state"], statistics, stats_modality["state"])[0]
    actions = encode_actions(normalized, state, chunks)
    reader = cv2.VideoCapture(str(video_path))
    try:
        fps = reader.get(cv2.CAP_PROP_FPS)
        if fps != 20:
            raise ValueError(f"GR1 sample requires 20Hz source video, got {fps}")
        reader.set(cv2.CAP_PROP_POS_FRAMES, int(indices[1]))
        ok, frame = reader.read()
        if not ok:
            raise ValueError("Cannot read the GR1 initial frame")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        reader.release()
    # Same center crop and float bilinear+antialias resize as official test_transform.
    h, w = frame.shape[:2]
    crop_h, crop_w = (h, int(h * 4 / 3)) if w / h > 4 / 3 else (int(w * 3 / 4), w)
    top, left = (h - crop_h) // 2, (w - crop_w) // 2
    image = torch.from_numpy(frame[top:top + crop_h, left:left + crop_w].copy()).permute(2, 0, 1).float() / 255
    image = VF.resize(image, [480, 640], interpolation=InterpolationMode.BILINEAR, antialias=True)
    image = (image * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    config = json.loads(config_path.read_text())
    if config["action"]["representation"] != "dreamdojo_gr1_slots384":
        raise ValueError("Wrong replay action protocol")
    packet = {
        "protocol": config["action"], "actions": actions,
        "reset_args": {"value": image}, "initial_views": {"ego_view": image},
        "source": {
            "repository": report["repository"], "revision": report["revision"],
            "source_revision": SOURCE_REVISION, "episode": episode, "start_frame": start,
            "fps": 20, "sampled_frame_indices": indices.tolist(),
            "initial_frame_index": int(indices[1]),
            "conversion": "official GR1 joint minmax; stride2; four-frame block deltas; "
                          "29 joints in 384 slots; pinned upstream first-row normalized-state alias preserved",
            "normalization_source": "pinned_source/shared_meta/GR1_unified_stats.json",
            "multiround_protocol": "official_global_alias_12_action_windows" if chunks == 4 else None,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_packet(output, packet)
    return {"ok": True, "model_inference_performed": False, "output": str(output),
            "action_shape": list(actions.shape), "source": packet["source"]}


def compare_official_encoding(dataset: Path, packet_path: Path):
    """CPU comparison against the pinned WrappedLeRobotSingleDataset method.

    Supplies the selected episode's actual raw rows/frames to the official
    transform and __getitem__. Only the two unused GPU metadata allocations
    are made CPU-only; no model output or learned computation is simulated.
    """
    import copy
    import os
    import sys
    from unittest.mock import patch
    import cv2
    import pyarrow.parquet as pq
    import torch
    from evaluation.execution.transport import read_packet
    from vlaworldeval.paths import project_root

    source = project_root() / "third_party/dreamdojo"
    sys.path.insert(0, str(source))
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    from groot_dreams.data.dataset import WrappedLeRobotSingleDataset
    from groot_dreams.data.embodiment_tags import EmbodimentTag
    from groot_dreams.groot_configs import construct_modality_config_and_transforms
    from adapters.world_models.dreamdojo import _source_cwd

    packet = read_packet(packet_path)
    validate_prepared_packet(packet)
    indices = np.asarray(packet["source"]["sampled_frame_indices"])
    chunks = len(packet["actions"])
    if len(indices) != chunks*12+2:
        raise ValueError("Selected frame indices differ from declared native window")
    dataset = Path(dataset).resolve()
    table_path, video_path = episode_paths(dataset, packet["source"]["episode"])
    table = pq.read_table(table_path)
    rows = {name: np.asarray(table[name].take(indices).to_pylist(), dtype=np.float32)
            for name in ("action", "observation.state")}
    reader = cv2.VideoCapture(str(video_path))
    images = []
    try:
        for index in indices:
            reader.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = reader.read()
            if not ok:
                raise ValueError(f"Missing source video frame {index}")
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        reader.release()
    modality = json.loads((dataset / "meta/modality.json").read_text())
    raw = {"video.ego_view_freq20": np.stack(images)}
    for group in ("state", "action"):
        for part in PARTS:
            field = modality[group][part]
            value = rows[field["original_key"]]
            raw[f"{group}.{part}"] = value[:, field["start"]:field["end"]]
            if group == "state":
                raw[f"{group}.{part}"] = raw[f"{group}.{part}"][:1]
    _, _, transform = construct_modality_config_and_transforms(chunks*12+2, "gr1")
    official = WrappedLeRobotSingleDataset.__new__(WrappedLeRobotSingleDataset)
    official._dataset_path, official._dataset_name = dataset, dataset.name
    official._all_steps = np.array([[packet["source"]["episode"], int(indices[0])]])
    official.num_frames, official.transforms = chunks*12+1, transform
    official.get_step_data = lambda *_: copy.deepcopy(raw)
    with _source_cwd(source):
        official.transforms.set_metadata(official._get_metadata(EmbodimentTag.GR1))
        official.transforms.eval()
        with patch.object(torch.Tensor, "cuda", lambda value, *a, **k: value), patch(
            "groot_dreams.data.dataset.randint", side_effect=RuntimeError("Official data transform failed; retry disabled")
        ):
            sample = official[0]
    expected_actions = sample["action"].numpy()
    expected_image = sample["video"][:, 0].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(packet["actions"].reshape(-1, 384), expected_actions)
    np.testing.assert_array_equal(packet["reset_args"]["value"], expected_image)
    return {"ok": True, "model_inference_performed": False, "official_method":
            "WrappedLeRobotSingleDataset.__getitem__", "action_exact_equal": True,
            "initial_image_exact_equal": True, "cuda_metadata_allocations_only_stubbed": True,
            "chunks_compared": chunks}
