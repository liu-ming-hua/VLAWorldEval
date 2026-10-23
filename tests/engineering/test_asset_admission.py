"""Related regression contracts; all original test assertions are retained."""

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.manifests import load_pair_manifest
from evaluation.planning.registry import RuntimeRegistry, load_runtime_registry
from evaluation.validation.asset_inventory import inventory_assets
from vlaworldeval.workspace import initialize_workspace

ROOT = Path(__file__).resolve().parents[2]


def test_openvla_config_without_weights_is_not_admitted(tmp_path):
    pair = load_pair_manifest(ROOT / "configs/pairs/openvla__worldgym.json")
    registry = load_runtime_registry(ROOT)
    folder = tmp_path / "checkpoints/vla/openvla-7b"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    status = registry.asset_status(tmp_path, pair, {"world_model_checkpoint": "wm.pt"}, {})
    assert status["vlas.OpenVLAAdapter.config"]
    assert not status["vlas.OpenVLAAdapter.index"]


@pytest.mark.parametrize("mapping", [[], {}, {"x": None}, {"x": 1}])
def test_bad_weight_indices_are_actionable_errors(tmp_path, mapping):
    pair = load_pair_manifest(ROOT / "configs/pairs/openvla__worldgym.json")
    definition = {"assets": {"checkpoint": {"project_path": "weights"}}}
    registry = RuntimeRegistry(
        {
            "vlas": {pair.vla.adapter: definition},
            "world_models": {pair.world_model.adapter: {"assets": {}}},
        },
        {},
        [],
    )
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="weight_map"):
        registry.asset_status(tmp_path, pair, {}, {})


def test_zero_byte_shard_is_not_admitted(tmp_path):
    pair = load_pair_manifest(ROOT / "configs/pairs/openvla__worldgym.json")
    definition = {"assets": {"checkpoint": {"project_path": "weights"}}}
    registry = RuntimeRegistry(
        {
            "vlas": {pair.vla.adapter: definition},
            "world_models": {pair.world_model.adapter: {"assets": {}}},
        },
        {},
        [],
    )
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "empty.safetensors").touch()
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "empty.safetensors"}})
    )
    assert not all(registry.asset_status(tmp_path, pair, {}, {}).values())


def test_cosmos_fd_admission_requires_own_ancillary_assets(tmp_path):
    pair = load_pair_manifest(ROOT / "configs/pairs/groot_n15__cosmos3_droid.json")
    registry = load_runtime_registry(ROOT)
    status = registry.asset_status(tmp_path, pair, {"world_model_checkpoint": "fd"}, {})
    for name in ("vae", "tokenizer_vocab", "tokenizer_merges", "tokenizer_config", "framework"):
        assert not status[f"world_models.Cosmos3NanoDroidFDAdapter.{name}"]
    assets = registry.models["world_models"]["Cosmos3NanoDroidFDAdapter"]["assets"]
    assert all(
        "tau0" not in str(binding) and "/cosmos3_nano/" not in str(binding)
        for binding in assets.values()
    )


def test_local_fingerprints_are_not_upstream_verification(tmp_path):
    root = tmp_path / "workspace"
    initialize_workspace(root)
    config = "configs/experiments/openvla_worldgym_v1.json"
    missing = inventory_assets(root, [config])
    assert not missing["ok"] and missing["missing"] and not missing["files"]
    model = root / "checkpoints/vla/openvla-7b"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "weights.bin"}})
    )
    (model / "weights.bin").write_bytes(b"example")
    (model / ".cache").mkdir()
    (model / ".cache/private-metadata").write_text("not part of the model")
    report = inventory_assets(root, [config])
    assert not report["upstream_checksum_verified"]
    assert len(report["files"]) == 3
    record = report["files"]["checkpoints/vla/openvla-7b/weights.bin"]
    assert record["sha256"] == hashlib.sha256(b"example").hexdigest()
    assert (model / "weights.bin").read_bytes() == b"example"
