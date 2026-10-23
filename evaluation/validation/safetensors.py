"""Header-only tensor inventory, not a checksum or measured VRAM profiler."""
from collections import Counter
import json
import math
from pathlib import Path
import struct


def inspect_sharded_tensors(directory):
    directory = Path(directory).resolve()
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if not isinstance(index, dict) or not isinstance(index.get("metadata", {}), dict):
        raise ValueError("Checkpoint index and metadata must be objects")
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
        raise ValueError("Require a nonempty string weight_map")
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "I32": 4,
              "U32": 4, "I64": 8, "U64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}
    seen, dtype_bytes, groups = set(), Counter(), Counter()
    total_elements = 0
    for filename in sorted(set(mapping.values())):
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory) or path.suffix != ".safetensors":
            raise ValueError("Tensor shard must remain inside the checkpoint directory")
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"Truncated safetensors prefix: {filename}")
            length = struct.unpack("<Q", raw)[0]
            if not 2 <= length <= 64 * 1024 * 1024 or 8 + length > path.stat().st_size:
                raise ValueError(f"Invalid safetensors header size: {filename}")
            header = json.loads(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError("Tensor header must be an object")
        intervals = []
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            if name in seen or mapping.get(name) != filename or not isinstance(tensor, dict):
                raise ValueError("Tensor keys/shards differ from the checkpoint index")
            shape, dtype, offsets = tensor.get("shape"), tensor.get("dtype"), tensor.get("data_offsets")
            if (not isinstance(shape, list) or any(type(v) is not int or v < 0 for v in shape)
                    or not isinstance(dtype, str) or dtype not in widths
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(v) is not int or v < 0 for v in offsets)):
                raise ValueError(f"Invalid tensor metadata: {name}")
            elements = math.prod(shape)
            size = offsets[1] - offsets[0]
            if size != elements * widths[dtype] or offsets[1] > path.stat().st_size - 8 - length:
                raise ValueError(f"Tensor offsets/size mismatch: {name}")
            seen.add(name)
            intervals.append(tuple(offsets))
            dtype_bytes[dtype] += size
            groups[".".join(name.split(".")[:2])] += size
            total_elements += elements
        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValueError(f"Non-contiguous or overlapping tensor data: {filename}")
            cursor = end
        if cursor != path.stat().st_size - 8 - length:
            raise ValueError(f"Unexpected trailing tensor data: {filename}")
    if seen != set(mapping):
        raise ValueError("Checkpoint index references missing tensors")
    total_bytes = sum(dtype_bytes.values())
    declared = index.get("metadata", {}).get("total_size")
    if declared is not None and (type(declared) is not int or declared != total_bytes):
        raise ValueError("Checkpoint total_size differs from tensor headers")
    return {"scope": "header_only_not_integrity_or_measured_vram", "tensor_count": len(seen),
            "elements": total_elements, "tensor_bytes": total_bytes,
            "dtype_bytes": dict(dtype_bytes), "module_bytes": dict(groups)}
