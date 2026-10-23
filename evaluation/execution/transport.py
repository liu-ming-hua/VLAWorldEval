"""JSON + numeric NPZ transport. No pickle, arbitrary objects or GPU tensors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_packet(path: Path, value: Any, *, compressed=False) -> None:
    if type(compressed) is not bool:
        raise ValueError("Packet compression must be an explicit boolean")
    arrays = {}

    def encode(item):
        if hasattr(item, "detach"):
            item = item.detach().float().cpu().numpy()
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("Object arrays are not allowed in worker transport")
            key = f"array_{len(arrays)}"
            arrays[key] = item
            return {"__array__": key}
        if isinstance(item, dict):
            return {str(k): encode(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(v) for v in item]
        if isinstance(item, np.generic):
            return item.item()
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(f"Unsupported packet value {type(item)}")

    tree = encode(value)
    arrays["__json__"] = np.asarray(json.dumps(tree, allow_nan=False))
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        (np.savez_compressed if compressed else np.savez)(handle, **arrays)
    temporary.replace(path)


def read_packet(path: Path) -> Any:
    with np.load(path, allow_pickle=False) as arrays:
        def decode(item):
            if isinstance(item, dict):
                if set(item) == {"__array__"}:
                    return arrays[item["__array__"]].copy()
                return {k: decode(v) for k, v in item.items()}
            if isinstance(item, list):
                return [decode(v) for v in item]
            return item
        return decode(json.loads(str(arrays["__json__"])))


__all__ = ["read_packet", "write_packet"]
