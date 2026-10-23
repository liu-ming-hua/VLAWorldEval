import json

import numpy as np
import pytest

from adapters.vla._protocol import PROJECT_ROOT, concatenate_actions, uint8_video
from adapters.vla.groot import GrootN15Adapter, GrootN17Adapter
from adapters.vla.openpi import Pi0Adapter, Pi05Adapter
from adapters.vla.smolvla import SmolVLAAdapter


def test_new_adapters_declare_independent_environments():
    assert GrootN15Adapter.environment_key == "vla.groot_n15"
    assert GrootN17Adapter.environment_key == "vla.groot_n17"
    assert Pi0Adapter.environment_key == "vla.openpi"
    assert Pi05Adapter.environment_key == "vla.openpi"
    assert SmolVLAAdapter.environment_key == "vla.smolvla"


def test_vla_manifests_reference_official_projects():
    manifests = {
        "groot.json": "github.com/NVIDIA/Isaac-GR00T",
        "openpi.json": "github.com/Physical-Intelligence/openpi",
        "smolvla.json": "github.com/huggingface/lerobot",
    }
    for filename, expected in manifests.items():
        data = json.loads((PROJECT_ROOT / "configs" / "vlas" / filename).read_text())
        serialized = json.dumps(data)
        assert expected in serialized
        assert "FluxVLA" not in serialized


def test_protocol_helpers_preserve_time_major_actions():
    actions = {
        "translation": np.ones((4, 3), dtype=np.float32),
        "gripper": np.zeros((4, 1), dtype=np.float32),
    }
    result = concatenate_actions(actions, ("translation", "gripper"))
    assert result.shape == (4, 4)


def test_protocol_helpers_reject_missing_history():
    with pytest.raises(ValueError, match="requires exactly 2"):
        uint8_video(np.zeros((32, 32, 3), dtype=np.uint8), history=2, key="camera", batched=True)


def test_groot_n15_declares_official_input_resolution():
    assert GrootN15Adapter.input_resolution == (256, 256)


def test_openpi_adapters_declare_checkpoint_action_shapes():
    assert Pi0Adapter.expected_action_shape == (10, 8)
    assert Pi05Adapter.expected_action_shape == (15, 8)
