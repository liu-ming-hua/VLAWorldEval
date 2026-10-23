"""Related regression contracts; all original test assertions are retained."""

import copy
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from evaluation.factories import resolve_workers
from evaluation.manifests import load_pair_manifest
from evaluation.planning.configuration import ConfigurationError
from evaluation.planning.experiments import build_experiment_plan
from evaluation.planning.registry import (
    RuntimeRegistry,
    load_runtime_registry,
    resolve_bindings,
    validate_bindings,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name,module,argument,checkpoint",
    [
        ("openvla", "openvla", "model_name_or_path", "openvla-7b"),
        ("spatialvla", "spatialvla", "model_name", "spatialvla-4b-224-pt"),
        ("octo", "octo", "model_name", "octo-base-1.5"),
        ("rt1_x", "rt1", "checkpoint_path", "rt1-x/rt_1_x_tf_trained_for_002272480_step"),
    ],
)
def test_registered_worldgym_constructors_unchanged(name, module, argument, checkpoint):
    before = set(sys.modules)
    plan = build_experiment_plan(ROOT / f"configs/experiments/{name}_worldgym_v1.json")
    pair = load_pair_manifest(ROOT / f"configs/pairs/{name}__worldgym.json")
    vla, wm = resolve_workers(ROOT, pair, plan["runtime"], plan["sampling"], ROOT / "unused")
    assert vla.module == f"adapters.vla.{module}"
    assert vla.kwargs[argument] == str(ROOT / "checkpoints/vla" / checkpoint)
    assert wm.kwargs == {
        "checkpoint_path": str(ROOT / plan["runtime"]["world_model_checkpoint"]),
        "default_cfg": 3,
        "use_pixel_rope": False,
        "num_inference_steps": 10,
    }
    assert not any(
        name.startswith(("torch", "jax", "tensorflow", "transformers", "adapters.vla."))
        for name in set(sys.modules) - before
    )


def test_cosmos_constructor_unchanged():
    plan = build_experiment_plan(ROOT / "configs/experiments/droid_framework_v1.json")
    pair = load_pair_manifest(ROOT / "configs/pairs/groot_n15__cosmos3_droid.json")
    vla, wm = resolve_workers(ROOT, pair, plan["runtime"], plan["sampling"], ROOT / "unused")
    assert vla.kwargs == {"model_path": str(ROOT / "checkpoints/vla/groot/n1.5-3b")}
    assert wm.kwargs == {
        "model_path": str(ROOT / plan["runtime"]["world_model_checkpoint"]),
        "runtime_dir": str(ROOT / "unused/workers/cosmos_runtime"),
        "num_inference_steps": 30,
        "guidance": 1,
        "shift": 10,
        "resolution": "480",
        "diffusion_cache": False,
        "use_torch_compile": False,
    }


@pytest.mark.parametrize(
    "binding",
    [
        {"project_path": "../outside"},
        {"runtime_path": "bad"},
        {"suite_path": "../outside"},
        {"shell": "echo unsafe"},
    ],
)
def test_unsafe_bindings_rejected(tmp_path, binding):
    with pytest.raises(ValueError):
        resolve_bindings({"model": binding}, tmp_path, {"bad": "../outside"}, {}, tmp_path)


@pytest.mark.parametrize(
    "change", ["duplicate", "unknown_dataset", "unknown_binding", "asset_not_path"]
)
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
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "missing.safetensors"}})
    )
    assert not all(registry.asset_status(tmp_path, pair, {}, {}).values())


def registry_fixture():
    return json.loads((ROOT / "configs/runtime_registry.json").read_text())


@pytest.mark.parametrize(
    "location,value",
    [
        ((), []),
        (("schema_version",), True),
        (("models",), []),
        (("datasets",), None),
        (("pairs",), {}),
        (("models", "vlas", "OpenVLAAdapter"), []),
        (("models", "vlas", "OpenVLAAdapter", "distributed"), "false"),
        (("models", "vlas", "OpenVLAAdapter", "module"), "adapters..bad"),
        (("models", "vlas", "OpenVLAAdapter", "kwargs"), []),
        (("models", "vlas", "OpenVLAAdapter", "environment"), "  "),
        (("datasets", "worldgym_bridge", "arguments"), {"root": []}),
        (("datasets", "worldgym_bridge", "module"), "os.path"),
        (("pairs", 0), None),
        (("pairs", 0, "id"), []),
        (("pairs", 0, "vlas"), "OpenVLAAdapter"),
        (("pairs", 0, "vlas"), ["OpenVLAAdapter", "OpenVLAAdapter"]),
        (("pairs", 0, "replay_transforms"), {"official_worldgym_actions": []}),
        (("pairs", 0, "sampling_constraints"), {"typo": 1}),
        (("pairs", 0, "provenance"), {"files": {"asset": []}}),
        (("pairs", 0, "evaluation_modes"), ["unimplemented_mode"]),
        (("pairs", 0, "dataset_source_fps"), float("inf")),
    ],
)
def test_bad_registry_shapes_produce_file_and_field_diagnostics(tmp_path, location, value):
    data = registry_fixture()
    if location:
        cursor = data
        for part in location[:-1]:
            cursor = cursor[part]
        cursor[location[-1]] = value
    else:
        data = value
    path = tmp_path / "configs/runtime_registry.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError) as exc:
        load_runtime_registry(tmp_path)
    assert exc.value.details["file"] == str(path)
    assert exc.value.details["issues"][0]["field"].startswith("$")


@pytest.mark.parametrize("error", ["missing", "unknown", "invalid_json"])
def test_registry_schema_missing_and_unknown_fields(tmp_path, error):
    data = registry_fixture()
    if error == "missing":
        del data["models"]["vlas"]["OpenVLAAdapter"]["kwargs"]
    elif error == "unknown":
        data["models"]["vlas"]["OpenVLAAdapter"]["kwrags"] = {}
    path = tmp_path / "configs/runtime_registry.json"
    path.parent.mkdir()
    path.write_text("{" if error == "invalid_json" else json.dumps(data))
    with pytest.raises(ConfigurationError):
        load_runtime_registry(tmp_path)


def test_registry_validation_preserves_all_current_declarations(tmp_path):
    data = registry_fixture()
    before = deepcopy(data)
    path = tmp_path / "configs/runtime_registry.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    registry = load_runtime_registry(tmp_path)
    assert registry.models == before["models"]
    assert registry.datasets == before["datasets"]
    assert registry.pairs == before["pairs"]


@pytest.mark.parametrize("source", ["runtime", "sampling", "runtime_path"])
def test_missing_context_binding_is_an_input_error(tmp_path, source):
    with pytest.raises(ValueError, match="Missing.*field"):
        resolve_bindings({"path": {source: "misspelled"}}, tmp_path, {}, {}, tmp_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_constructor_scalars_must_be_finite(value):
    with pytest.raises(ValueError, match="finite"):
        validate_bindings({"guidance": value})
