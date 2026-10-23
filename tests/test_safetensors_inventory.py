import json
import struct

import pytest

from evaluation.validation.safetensors import inspect_sharded_tensors


def checkpoint(tmp_path, *, shape=None, offsets=None, total=12, mapping=None):
    name = "action_head.model.weight"
    header = {name: {"shape": [2, 3] if shape is None else shape, "dtype": "BF16",
                     "data_offsets": [0, 12] if offsets is None else offsets}}
    encoded = json.dumps(header).encode()
    (tmp_path / "model-00001.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(12))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
        "weight_map": mapping if mapping is not None else {name: "model-00001.safetensors"}}))


def test_header_inventory_does_not_materialize_tensor_data(tmp_path):
    checkpoint(tmp_path)
    report = inspect_sharded_tensors(tmp_path)
    assert report["tensor_count"] == 1
    assert report["elements"] == 6
    assert report["tensor_bytes"] == 12
    assert report["dtype_bytes"] == {"BF16": 12}
    assert report["module_bytes"] == {"action_head.model": 12}
    assert report["scope"] == "header_only_not_integrity_or_measured_vram"


@pytest.mark.parametrize("kwargs", [
    {"shape": [-1, 3]}, {"shape": [True, 3]}, {"offsets": [0, 14]},
    {"offsets": [1, 13]}, {"total": 13},
    {"mapping": {"missing": "model-00001.safetensors"}},
    {"mapping": {"weight": "../outside.safetensors"}},
])
def test_corrupt_or_mismatched_headers_are_rejected(tmp_path, kwargs):
    checkpoint(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        inspect_sharded_tensors(tmp_path)


def test_truncated_header_and_oversized_prefix_rejected(tmp_path):
    checkpoint(tmp_path)
    shard = tmp_path / "model-00001.safetensors"
    shard.write_bytes(b"x")
    with pytest.raises(ValueError, match="prefix"):
        inspect_sharded_tensors(tmp_path)
    shard.write_bytes(struct.pack("<Q", 2**40))
    with pytest.raises(ValueError, match="header size"):
        inspect_sharded_tensors(tmp_path)


def test_shard_symlink_must_not_escape_checkpoint(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    checkpoint(model)
    shard = model / "model-00001.safetensors"
    outside = tmp_path / "outside.safetensors"
    shard.rename(outside)
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="inside"):
        inspect_sharded_tensors(model)
