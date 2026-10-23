"""Extension mistakes must fail before importing model runtimes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from evaluation.planning.configuration import ConfigurationError
from evaluation.planning.registry import load_runtime_registry, resolve_bindings, validate_bindings

ROOT = Path(__file__).resolve().parents[1]


def registry_fixture():
    return json.loads((ROOT / "configs/runtime_registry.json").read_text())


@pytest.mark.parametrize("location,value", [
    ((), []), (("schema_version",), True), (("models",), []), (("datasets",), None),
    (("pairs",), {}), (("models", "vlas", "OpenVLAAdapter"), []),
    (("models", "vlas", "OpenVLAAdapter", "distributed"), "false"),
    (("models", "vlas", "OpenVLAAdapter", "module"), "adapters..bad"),
    (("models", "vlas", "OpenVLAAdapter", "kwargs"), []),
    (("models", "vlas", "OpenVLAAdapter", "environment"), "  "),
    (("datasets", "worldgym_bridge", "arguments"), {"root": []}),
    (("datasets", "worldgym_bridge", "module"), "os.path"),
    (("pairs", 0), None), (("pairs", 0, "id"), []),
    (("pairs", 0, "vlas"), "OpenVLAAdapter"),
    (("pairs", 0, "vlas"), ["OpenVLAAdapter", "OpenVLAAdapter"]),
    (("pairs", 0, "replay_transforms"), {"official_worldgym_actions": []}),
    (("pairs", 0, "sampling_constraints"), {"typo": 1}),
    (("pairs", 0, "provenance"), {"files": {"asset": []}}),
    (("pairs", 0, "evaluation_modes"), ["unimplemented_mode"]),
    (("pairs", 0, "dataset_source_fps"), float("inf")),
])
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
