"""All catalog consumers share diagnostics; no model stack is required."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from evaluation.planning.catalog import load_model_catalog
from evaluation.planning.configuration import ConfigurationError
from evaluation.planning.inventory import validate_catalog
from evaluation.planning.capabilities import capabilities
from vlaworldeval.workspace import initialize_workspace

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("location,value", [
    ((), []), (("schema_version",), True), (("schema_version",), 2),
    (("description",), []), (("vlas",), []), (("world_models",), {}),
    (("vlas", "openvla"), []), (("vlas", "openvla", "adapter"), "not a class"),
    (("vlas", "openvla", "status"), None), (("vlas", "openvla", "source"), "  "),
    (("vlas", "openvla", "environment"), 42), (("vlas", "openvla", "input"), []),
    (("vlas", "openvla", "native_output"), {}),
    (("vlas", "openvla", "adapter_output"), []),
    (("world_models", "worldgym", "output"), "video"),
    (("world_models", "worldgym", "action_bridge"), False),
])
def test_invalid_catalog_has_structured_file_and_field_error(tmp_path, location, value):
    data = json.loads((ROOT / "configs/model_catalog.json").read_text())
    if location:
        cursor = data
        for key in location[:-1]:
            cursor = cursor[key]
        cursor[location[-1]] = value
    else:
        data = value
    path = tmp_path / "configs/model_catalog.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    for consumer in (load_model_catalog, validate_catalog):
        with pytest.raises(ConfigurationError) as exc:
            consumer(tmp_path)
        assert exc.value.details["file"] == str(path)
        assert exc.value.details["issues"][0]["field"] == (".".join(location) if location else "$")


def test_catalog_reports_syntax_missing_file_and_missing_required_description(tmp_path):
    with pytest.raises(ConfigurationError):
        load_model_catalog(tmp_path)
    path = tmp_path / "configs/model_catalog.json"
    path.parent.mkdir()
    path.write_text("{")
    with pytest.raises(ConfigurationError):
        load_model_catalog(tmp_path)
    data = json.loads((ROOT / "configs/model_catalog.json").read_text())
    del data["vlas"]["openvla"]["training_domain"]
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="training_domain"):
        load_model_catalog(tmp_path)


def test_current_catalog_keeps_descriptions_and_declared_status_without_promotion(tmp_path):
    data = json.loads((ROOT / "configs/model_catalog.json").read_text())
    before = deepcopy(data)
    path = tmp_path / "configs/model_catalog.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    assert load_model_catalog(tmp_path) == before
    assert validate_catalog(tmp_path) == {group: len(before[group]) for group in ("vlas", "world_models")}


@pytest.mark.parametrize("change", ["unknown_level", "missing_scope", "duplicate_pairs", "supported_without_pairs"])
def test_declared_tier_has_explicit_scope_and_pair_subset(tmp_path, change):
    data = json.loads((ROOT / "configs/model_catalog.json").read_text())
    info = data["vlas"]["openvla"]["integration"]
    if change == "unknown_level":
        info["level"] = "all_robots_verified"
    elif change == "missing_scope":
        del info["scope"]
    elif change == "duplicate_pairs":
        info["supported_pair_ids"] *= 2
    else:
        info["supported_pair_ids"] = []
    path = tmp_path / "configs/model_catalog.json"
    path.parent.mkdir()
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigurationError, match="integration"):
        load_model_catalog(tmp_path)


def test_current_model_tiers_do_not_promote_every_registered_pair_or_quality():
    result = capabilities(ROOT)
    assert not result["execution_performed"] and not result["quality_evaluated"]
    entries = result["model_support"]
    eef = entries["world_models"]["cosmos3_nano_droid_fd_20k_ema"]
    assert len(eef["registered_pair_ids"]) == 6
    assert eef["supported_pair_ids"] == ["groot_n15__cosmos3_nano_droid_fd_v1"]
    assert entries["world_models"]["gigaworld1"]["level"] == "incompatible"
    assert not entries["world_models"]["gigaworld1"]["registered_pair_ids"]
    assert entries["world_models"]["cosmos3_droid_joint_fd_best5600"]["level"] == "native_replay"
    for group in entries.values():
        for item in group.values():
            assert not item["execution_performed"] and not item["quality_evaluated"]


@pytest.mark.parametrize("change", ["unclassified", "nonexistent_pair", "incompatible_registered"])
def test_capability_tiers_remain_consistent_with_real_declarations(tmp_path, change):
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace)
    path = workspace / "configs/model_catalog.json"
    data = json.loads(path.read_text())
    if change == "unclassified":
        del data["vlas"]["smolvla"]["integration"]
    elif change == "nonexistent_pair":
        data["vlas"]["openvla"]["integration"]["supported_pair_ids"] = ["fictional_pair"]
    else:
        data["vlas"]["openvla"]["integration"].update(level="incompatible", supported_pair_ids=[])
    path.write_text(json.dumps(data))
    if change == "unclassified":
        assert capabilities(workspace)["model_support"]["vlas"]["smolvla"]["level"] == "unclassified"
    else:
        with pytest.raises(ConfigurationError, match="contradicts"):
            capabilities(workspace)
