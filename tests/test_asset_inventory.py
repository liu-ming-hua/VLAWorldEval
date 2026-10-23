import hashlib
import json

from evaluation.validation.asset_inventory import inventory_assets
from vlaworldeval.workspace import initialize_workspace


def test_local_fingerprints_are_not_upstream_verification(tmp_path):
    root = tmp_path / "workspace"
    initialize_workspace(root)
    config = "configs/experiments/openvla_worldgym_v1.json"
    missing = inventory_assets(root, [config])
    assert not missing["ok"] and missing["missing"] and not missing["files"]
    model = root / "checkpoints/vla/openvla-7b"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "weights.bin"}}))
    (model / "weights.bin").write_bytes(b"example")
    (model / ".cache").mkdir()
    (model / ".cache/private-metadata").write_text("not part of the model")
    report = inventory_assets(root, [config])
    assert not report["upstream_checksum_verified"]
    assert len(report["files"]) == 3
    record = report["files"]["checkpoints/vla/openvla-7b/weights.bin"]
    assert record["sha256"] == hashlib.sha256(b"example").hexdigest()
    assert (model / "weights.bin").read_bytes() == b"example"
