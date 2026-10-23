"""Utilities for writing reproducible rollout artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import mediapy as media
import numpy as np

from evaluation.contracts import RobotState


def _to_numpy_action(action: Any) -> np.ndarray:
    """Convert one adapter action chunk to a CPU float32 [H, A] array."""
    if hasattr(action, "detach"):
        action = action.detach().float().cpu().numpy()
    else:
        action = np.asarray(action)
    action = np.asarray(action, dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    if action.ndim != 2:
        raise ValueError(f"Expected an action chunk with shape [H, A], got {action.shape}")
    return np.ascontiguousarray(action)


def pack_action_chunks(action_chunks: Iterable[Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable-length action chunks into arrays suitable for ``npz``.

    Returns:
        actions: Float32 array [T, max_horizon, max_action_dim]. Unused entries
            are NaN-padded.
        lengths: Int32 array [T] containing the valid horizon for each step.
        mask: Bool array [T, max_horizon] marking valid time positions.
    """
    chunks = [_to_numpy_action(chunk) for chunk in action_chunks]
    if not chunks:
        return (
            np.empty((0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 0), dtype=bool),
        )

    max_horizon = max(chunk.shape[0] for chunk in chunks)
    max_action_dim = max(chunk.shape[1] for chunk in chunks)
    actions = np.full(
        (len(chunks), max_horizon, max_action_dim),
        np.nan,
        dtype=np.float32,
    )
    lengths = np.zeros((len(chunks),), dtype=np.int32)
    mask = np.zeros((len(chunks), max_horizon), dtype=bool)
    for index, chunk in enumerate(chunks):
        horizon, action_dim = chunk.shape
        actions[index, :horizon, :action_dim] = chunk
        lengths[index] = horizon
        mask[index, :horizon] = True
    return actions, lengths, mask


def pack_states(states: Iterable[RobotState | None]) -> dict[str, np.ndarray | list[dict]]:
    """Pack typed states without object arrays or pickle-dependent loading."""
    records = list(states)
    flattened: list[np.ndarray | None] = []
    uncertainties: list[np.ndarray | None] = []
    components: list[dict[str, Any]] = []
    for state in records:
        if state is None:
            flattened.append(None)
            uncertainties.append(None)
            components.append({"names": [], "sizes": []})
            continue
        values, names, sizes = state.flattened()
        flattened.append(values)
        if state.uncertainty is None:
            uncertainty = None
        else:
            uncertainty = np.concatenate(
                [state.uncertainty[name].reshape(-1) for name in names]
            ).astype(np.float32, copy=False)
        uncertainties.append(uncertainty)
        components.append({"names": list(names), "sizes": list(sizes)})

    max_width = max((len(value) for value in flattened if value is not None), default=0)
    values_array = np.full((len(records), max_width), np.nan, dtype=np.float32)
    uncertainty_array = np.full_like(values_array, np.nan)
    mask = np.zeros((len(records), max_width), dtype=bool)
    lengths = np.zeros((len(records),), dtype=np.int32)
    sources = np.asarray(["" if state is None else state.source for state in records], dtype=str)
    schemas = np.asarray(["" if state is None else state.schema for state in records], dtype=str)
    for index, value in enumerate(flattened):
        if value is None:
            continue
        width = len(value)
        values_array[index, :width] = value
        mask[index, :width] = True
        lengths[index] = width
        if uncertainties[index] is not None:
            uncertainty_array[index, :width] = uncertainties[index]
    return {
        "state_values": values_array,
        "state_uncertainty": uncertainty_array,
        "state_lengths": lengths,
        "state_mask": mask,
        "state_sources": sources,
        "state_schemas": schemas,
        "components": components,
    }


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "view"


def rollout_artifact_dir(
    video_out_dir: str | Path,
    root_dir: str | Path,
    trial: dict,
    retry_index: int,
) -> Path:
    """Return a collision-free, human-readable directory for one rollout."""
    root = Path(root_dir).resolve()
    source = Path(trial["trial_png"]).resolve()
    relative_parent = source.parent.relative_to(root)
    return Path(video_out_dir) / relative_parent / source.stem / f"retry_{retry_index:03d}"


def save_rollout_artifacts(
    output_dir: str | Path,
    video: np.ndarray,
    action_chunks: Iterable[Any],
    trial: dict,
    *,
    fps: float = 20,
    native_action_chunks: Iterable[Any] | None = None,
    states: Iterable[RobotState | None] | None = None,
    view_videos: dict[str, np.ndarray] | None = None,
    pair_manifest: dict[str, Any] | None = None,
    timings: list[dict[str, float]] | None = None,
    action_bridge_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Write videos, native/bridged actions, states, and provenance metadata.

    ``actions`` remains an alias of ``bridged_actions`` for compatibility with
    format v1 readers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    actions, lengths, mask = pack_action_chunks(action_chunks)
    if action_bridge_metadata is not None and len(action_bridge_metadata) != len(lengths):
        raise ValueError("Action bridge metadata must have one record per WM chunk")
    if native_action_chunks is None:
        native_actions = np.empty((0, 0, 0), dtype=np.float32)
        native_lengths = np.empty((0,), dtype=np.int32)
        native_mask = np.empty((0, 0), dtype=bool)
    else:
        native_actions, native_lengths, native_mask = pack_action_chunks(native_action_chunks)
    action_path = output_dir / "actions.npz"
    np.savez_compressed(
        action_path,
        actions=actions,
        action_lengths=lengths,
        action_mask=mask,
        bridged_actions=actions,
        bridged_action_lengths=lengths,
        bridged_action_mask=mask,
        native_actions=native_actions,
        native_action_lengths=native_lengths,
        native_action_mask=native_mask,
    )

    video_path = output_dir / "rollout.mp4"
    media.write_video(str(video_path), video, fps=fps)

    view_files = {}
    if view_videos:
        views_dir = output_dir / "views"
        views_dir.mkdir(exist_ok=True)
        used_names = set()
        for view_name, view_video in sorted(view_videos.items()):
            filename = _safe_name(view_name) + ".mp4"
            if filename in used_names:
                raise ValueError(f"View names collide after filename sanitization: {view_name!r}")
            used_names.add(filename)
            view_path = views_dir / filename
            media.write_video(str(view_path), np.asarray(view_video), fps=fps)
            view_files[view_name] = str(view_path.relative_to(output_dir))

    state_path = None
    state_components = None
    state_sources = []
    if states is not None:
        packed_states = pack_states(states)
        state_components = packed_states.pop("components")
        state_sources = packed_states["state_sources"].tolist()
        state_path = output_dir / "states.npz"
        np.savez_compressed(state_path, **packed_states)

    metadata = {
        "format_version": 2,
        "video_file": video_path.name,
        "view_files": view_files,
        "action_file": action_path.name,
        "state_file": None if state_path is None else state_path.name,
        "action_semantics": {
            "native_actions": "raw time-major chunks returned by the VLA adapter",
            "bridged_actions": "converted chunks consumed by the world-model adapter",
            "actions": "compatibility alias of bridged_actions",
        },
        "action_array_layout": "actions[outer_step, chunk_step, action_dim]",
        "state_array_layout": "state_values[observation_step, flattened_state_dim]",
        "state_components": state_components,
        "state_sources": state_sources,
        "num_rollout_steps": int(len(lengths)),
        "num_video_frames": int(len(video)),
        "video_fps": float(fps),
        "video_shape": list(video.shape),
        "pair_manifest": pair_manifest,
        "timings": timings or [],
        "action_bridge_metadata": action_bridge_metadata or [],
        "trial": {
            key: value
            for key, value in trial.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    result = {
        "video_path": str(video_path),
        "action_path": str(action_path),
        "metadata_path": str(metadata_path),
    }
    if state_path is not None:
        result["state_path"] = str(state_path)
    return result
