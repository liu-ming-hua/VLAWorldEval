import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from adapters.backends.cosmos import CosmosFrameworkRuntime
from adapters.backends.cosmos_actions import restore_robolab_geometry, validate_joint_action
from adapters.vla import GrootN17DroidAdapter, Pi0FastDroidAdapter
from adapters.vla.openpi import Pi0Adapter
from vlaworldeval.cache import runtime_cache_environment
from vlaworldeval.openpi_mirror import check_file


def test_specialists_keep_existing_protocols_and_separate_names():
    assert GrootN17DroidAdapter.environment_key == "vla.groot_n17"
    assert Pi0FastDroidAdapter.config_name == "pi0_fast_droid"
    assert Pi0FastDroidAdapter.required_fields == Pi0Adapter.required_fields
    assert Pi0FastDroidAdapter.expected_action_shape == (10, 8)


def test_openpi_cache_stays_in_project(tmp_path):
    assert runtime_cache_environment(tmp_path)["OPENPI_DATA_HOME"] == str(
        tmp_path / ".cache/openpi"
    )


def test_groot_droid_specialist_uses_its_single_frame_checkpoint_protocol():
    for suffix in ("ctrl_world", "cosmos3_droid"):
        pair = json.loads(Path(f"configs/pairs/groot_n17_droid__{suffix}.json").read_text())
        assert pair["vla"]["history_offsets"] == [0]
        assert pair["vla"]["action"]["horizon"] == 40


def test_official_md5_verification(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"original")
    expected = {
        "size": 8,
        "md5Hash": base64.b64encode(hashlib.md5(b"original").digest()).decode(),
        "generation": "123",
    }
    assert check_file(path, expected)["generation"] == "123"
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="MD5"):
        check_file(path, expected)
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        check_file(path, expected)


def test_joint_fd_protocol_rejects_wrong_semantics_shapes_and_gripper():
    good = np.zeros((32, 8))
    assert validate_joint_action(good, 32).dtype == np.float32
    for bad in (np.zeros((16, 10)), np.full((32, 8), np.nan)):
        with pytest.raises(ValueError, match="finite"):
            validate_joint_action(bad, 32)
    good[0, -1] = 2
    with pytest.raises(ValueError, match="gripper"):
        validate_joint_action(good, 32)


def test_robolab_geometry_restoration_does_not_stretch_views():
    original = np.arange(528 * 640 * 3, dtype=np.uint8).reshape(1, 528, 640, 3)
    restored = restore_robolab_geometry(original)
    assert restored.shape == (1, 540, 640, 3)
    np.testing.assert_array_equal(restored[:, :528], original)
    np.testing.assert_array_equal(restored[:, -1], original[:, -1])
    with pytest.raises(ValueError, match="geometry"):
        restore_robolab_geometry(np.zeros((1, 544, 736, 3), dtype=np.uint8))


def test_exported_backbone_filename_resolves_only_known_local_architecture(tmp_path):
    runtime = CosmosFrameworkRuntime()
    runtime.framework_root = tmp_path
    path = (
        tmp_path
        / "cosmos_framework/model/generator/reasoner/qwen3_vl/configs/Qwen3-VL-8B-Instruct.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert runtime._backbone_config_path(path.name) == path
    with pytest.raises(FileNotFoundError):
        runtime._backbone_config_path("arbitrary.json")
    with pytest.raises(ValueError, match="leaves"):
        runtime._backbone_config_path("../config.json")
