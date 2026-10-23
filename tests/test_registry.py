"""Runtime registration stays weight-free and preserves constructor protocols."""
import copy
import json
from pathlib import Path
import sys

import pytest

from evaluation.planning.experiments import build_experiment_plan
from evaluation.factories import resolve_workers
from evaluation.manifests import load_pair_manifest
from evaluation.planning.registry import RuntimeRegistry, load_runtime_registry, resolve_bindings

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,module,argument,checkpoint", [
    ("openvla", "openvla", "model_name_or_path", "openvla-7b"),
    ("spatialvla", "spatialvla", "model_name", "spatialvla-4b-224-pt"),
    ("octo", "octo", "model_name", "octo-base-1.5"),
    ("rt1_x", "rt1", "checkpoint_path", "rt1-x/rt_1_x_tf_trained_for_002272480_step"),
])
def test_registered_worldgym_constructors_unchanged(name, module, argument, checkpoint):
    before = set(sys.modules)
    plan = build_experiment_plan(ROOT / f"configs/experiments/{name}_worldgym_v1.json")
    pair = load_pair_manifest(ROOT / f"configs/pairs/{name}__worldgym.json")
    vla, wm = resolve_workers(ROOT, pair, plan["runtime"], plan["sampling"], ROOT / "unused")
    assert vla.module == f"adapters.vla.{module}"
    assert vla.kwargs[argument] == str(ROOT / "checkpoints/vla" / checkpoint)
    assert wm.kwargs == {"checkpoint_path": str(ROOT / plan["runtime"]["world_model_checkpoint"]),
                         "default_cfg": 3, "use_pixel_rope": False, "num_inference_steps": 10}
    assert not any(name.startswith(("torch", "jax", "tensorflow", "transformers", "adapters.vla."))
                   for name in set(sys.modules) - before)


def test_cosmos_constructor_unchanged():
    plan = build_experiment_plan(ROOT / "configs/experiments/droid_framework_v1.json")
    pair = load_pair_manifest(ROOT / "configs/pairs/groot_n15__cosmos3_droid.json")
    vla, wm = resolve_workers(ROOT, pair, plan["runtime"], plan["sampling"], ROOT / "unused")
    assert vla.kwargs == {"model_path": str(ROOT / "checkpoints/vla/groot/n1.5-3b")}
    assert wm.kwargs == {
        "model_path": str(ROOT / plan["runtime"]["world_model_checkpoint"]),
        "runtime_dir": str(ROOT / "unused/workers/cosmos_runtime"),
        "num_inference_steps": 30, "guidance": 1, "shift": 10, "resolution": "480",
        "diffusion_cache": False, "use_torch_compile": False,
    }


@pytest.mark.parametrize("binding", [{"project_path": "../outside"}, {"runtime_path": "bad"},
                                     {"suite_path": "../outside"}, {"shell": "echo unsafe"}])
def test_unsafe_bindings_rejected(tmp_path, binding):
    with pytest.raises(ValueError):
        resolve_bindings({"model": binding}, tmp_path, {"bad": "../outside"}, {}, tmp_path)


@pytest.mark.parametrize("change", ["duplicate", "unknown_dataset", "unknown_binding", "asset_not_path"])
def test_invalid_registry_rejected(tmp_path, change):
    data = json.loads((ROOT / "configs/runtime_registry.json").read_text())
    if change == "duplicate":
        data["pairs"].append(copy.deepcopy(data["pairs"][0]))
    elif change == "unknown_dataset":
        data["pairs"][0]["dataset"] = "unknown"
    elif change == "unknown_binding":
        data["models"]["vlas"]["OpenVLAAdapter"]["kwargs"] = {"model": {"eval": "unsafe"}}
    else:
        data["models"]["vlas"]["OpenVLAAdapter"]["assets"] = {"weights": True}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/runtime_registry.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_runtime_registry(tmp_path)


def test_index_references_missing_shards_are_not_admitted(tmp_path):
    pair = load_pair_manifest(ROOT / "configs/pairs/openvla__worldgym.json")
    definition = {"assets": {"checkpoint": {"project_path": "weights"}}}
    registry = RuntimeRegistry({"vlas": {pair.vla.adapter: definition},
                                "world_models": {pair.world_model.adapter: {"assets": {}}}}, {}, [])
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    assert not all(registry.asset_status(tmp_path, pair, {}, {}).values())


def test_registered_loader_preserves_official_worldgym_packet():
    root = ROOT / "datasets/worldgym/official/bridge/test"
    if not root.is_dir():
        pytest.skip("Local official sample not installed")
    pair = load_pair_manifest(ROOT / "configs/pairs/openvla__worldgym.json")
    registry = load_runtime_registry(ROOT)
    packet = registry.load_episode(registry.pair_profile(pair), root,
                                   {"start_frame": 0, "fps": 20, "instruction": "move"},
                                   {"episode_index": 2, "chunks": 3, "mode": "wm_replay"}, 1)
    assert len(packet.replay_actions) == 3
    assert packet.replay_actions[0].shape == (1, 10)
    assert packet.trial["observation_bundle"].get("robot_state") is None
