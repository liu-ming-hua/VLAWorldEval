import json
from pathlib import Path

import pytest

from evaluation.planning.registry import RuntimeRegistry, load_runtime_registry
from evaluation.manifests import load_pair_manifest

ROOT = Path(__file__).resolve().parents[1]


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
    registry = RuntimeRegistry({"vlas": {pair.vla.adapter: definition},
                                "world_models": {pair.world_model.adapter: {"assets": {}}}}, {}, [])
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="weight_map"):
        registry.asset_status(tmp_path, pair, {}, {})


def test_zero_byte_shard_is_not_admitted(tmp_path):
    pair = load_pair_manifest(ROOT / "configs/pairs/openvla__worldgym.json")
    definition = {"assets": {"checkpoint": {"project_path": "weights"}}}
    registry = RuntimeRegistry({"vlas": {pair.vla.adapter: definition},
                                "world_models": {pair.world_model.adapter: {"assets": {}}}}, {}, [])
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "empty.safetensors").touch()
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "empty.safetensors"}}))
    assert not all(registry.asset_status(tmp_path, pair, {}, {}).values())


def test_cosmos_fd_admission_requires_own_ancillary_assets(tmp_path):
    pair = load_pair_manifest(ROOT / "configs/pairs/groot_n15__cosmos3_droid.json")
    registry = load_runtime_registry(ROOT)
    status = registry.asset_status(tmp_path, pair, {"world_model_checkpoint": "fd"}, {})
    for name in ("vae", "tokenizer_vocab", "tokenizer_merges", "tokenizer_config", "framework"):
        assert not status[f"world_models.Cosmos3NanoDroidFDAdapter.{name}"]
    assets = registry.models["world_models"]["Cosmos3NanoDroidFDAdapter"]["assets"]
    assert all("tau0" not in str(binding) and "/cosmos3_nano/" not in str(binding)
               for binding in assets.values())
