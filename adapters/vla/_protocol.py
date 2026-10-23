"""Small protocol helpers shared by independently packaged VLA adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


from vlaworldeval.paths import project_root

PROJECT_ROOT = project_root()


def project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{name} must be a mapping with the model's official field names; "
            f"received {type(value).__name__}."
        )
    return value


def require_keys(value: Mapping[str, Any], keys: Sequence[str], *, name: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing)}")


def uint8_video(value: Any, *, history: int, key: str, batched: bool) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{key} must have shape [T,H,W,3] or [H,W,3], got {array.shape}")
    if array.shape[0] != history:
        raise ValueError(f"{key} requires exactly {history} frame(s), got {array.shape[0]}")
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255)
    array = array.astype(np.uint8, copy=False)
    return array[None, ...] if batched else array


def float_state(value: Any, *, history: int, key: str, batched: bool) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, ...]
    if array.ndim != 2:
        raise ValueError(f"{key} must have shape [T,D] or [D], got {array.shape}")
    if array.shape[0] != history:
        raise ValueError(f"{key} requires exactly {history} state step(s), got {array.shape[0]}")
    return array[None, ...] if batched else array


def concatenate_actions(actions: Mapping[str, Any], keys: Sequence[str]) -> np.ndarray:
    require_keys(actions, keys, name="model action")
    arrays = []
    for key in keys:
        array = np.asarray(actions[key], dtype=np.float32)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2:
            raise ValueError(f"Action {key!r} must resolve to [H,D], got {array.shape}")
        arrays.append(array)
    horizons = {array.shape[0] for array in arrays}
    if len(horizons) != 1:
        raise ValueError(f"Action components have inconsistent horizons: {sorted(horizons)}")
    return np.concatenate(arrays, axis=-1)
