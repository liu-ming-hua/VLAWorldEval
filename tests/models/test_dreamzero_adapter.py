"""Protocol tests with fixture policy outputs, not DreamZero model inference."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.vla.dreamzero import (
    CAMERAS,
    DreamZeroDroidAdapter,
    decode_dreamzero_action,
    dreamzero_observation,
)


def inputs(frames=1):
    images = {
        name: np.full((frames, 8, 12, 3), i * 30, dtype=np.uint8) for i, name in enumerate(CAMERAS)
    }
    return images, {
        "observation/joint_position": np.arange(7, dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
    }


def actions():
    return {"action.joint_position": np.ones((24, 7)), "action.gripper_position": np.full(24, 1.03)}


def test_official_field_mapping_state_types_and_no_mutation():
    image, state = inputs()
    out = dreamzero_observation(image, "pick", state, first_call=True)
    assert set(CAMERAS.values()).issubset(out)
    assert out["video.exterior_image_2_left"].shape == (1, 8, 12, 3)
    assert out["video.exterior_image_2_left"][0, 0, 0, 0] == 30
    assert out["state.joint_position"].dtype == np.float64
    assert out["state.joint_position"].shape == (1, 7)
    assert out["state.gripper_position"].shape == (1, 1)
    out["state.joint_position"][0, 0] = -100
    assert state["observation/joint_position"][0] == 0


@pytest.mark.parametrize("frames,first", [(0, True), (4, True), (1, False), (3, False), (5, False)])
def test_history_is_not_repeated_or_silently_truncated(frames, first):
    image, state = inputs(frames)
    with pytest.raises(ValueError, match="history padding"):
        dreamzero_observation(image, "pick", state, first_call=first)


@pytest.mark.parametrize(
    "error", ["state_missing", "joint_width", "nan", "gripper", "image_float", "prompt"]
)
def test_bad_inputs_fail_before_policy_call(error):
    image, state = inputs()
    instruction = "pick"
    if error == "state_missing":
        state.pop("observation/joint_position")
    elif error == "joint_width":
        state["observation/joint_position"] = np.zeros(6)
    elif error == "nan":
        state["observation/joint_position"][0] = np.nan
    elif error == "gripper":
        state["observation/gripper_position"] = [2]
    elif error == "image_float":
        image[next(iter(CAMERAS))] = np.zeros((1, 8, 12, 3), dtype=np.float32)
    else:
        instruction = ""
    with pytest.raises(ValueError):
        dreamzero_observation(image, instruction, state, first_call=True)


def test_output_is_decoded_absolute_action_without_hidden_clipping():
    out = decode_dreamzero_action(actions())
    assert out.shape == (24, 8) and out.dtype == np.float32
    assert out[0, -1] == np.float32(1.03)
    assert np.all(out[:, :7] == 1)


@pytest.mark.parametrize("error", ["missing", "shape", "nonfinite"])
def test_bad_outputs_do_not_become_zero_actions(error):
    value = actions()
    if error == "missing":
        value.pop("action.gripper_position")
    elif error == "shape":
        value["action.joint_position"] = np.ones((25, 7))
    else:
        value["action.joint_position"][0, 0] = np.inf
    with pytest.raises(ValueError):
        decode_dreamzero_action(value)


def test_adapter_calls_official_api_reset_and_failed_cache_guard():
    calls = []

    class Policy:
        fail = False

        def lazy_joint_forward_causal(self, batch):
            calls.append(batch.obs)
            if self.fail:
                raise RuntimeError("fixture failure")
            return SimpleNamespace(act=SimpleNamespace(**actions())), None

    adapter = object.__new__(DreamZeroDroidAdapter)
    adapter._closed = False
    adapter._policy = Policy()
    adapter._batch_type = SimpleNamespace
    adapter._inference_context = nullcontext
    adapter.reset()
    image, state = inputs()
    assert adapter.predict(image, "pick", state).shape == (24, 8)
    with pytest.raises(ValueError, match="history padding"):
        adapter.predict(image, "pick", state)
    image, state = inputs(4)
    adapter.predict(image, "pick", state)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="Reset"):
        adapter.predict(image, "different task", state)
    adapter.reset()
    image, state = inputs()
    adapter.predict(image, "different task", state)
    assert calls[-1]["video.wrist_image_left"].shape[0] == 1
    image, state = inputs(4)
    adapter._policy.fail = True
    with pytest.raises(RuntimeError, match="fixture failure"):
        adapter.predict(image, "different task", state)
    with pytest.raises(RuntimeError, match="failed"):
        adapter.predict(image, "different task", state)
    with pytest.raises(RuntimeError, match="recreate"):
        adapter.reset()


def test_unsupported_distributed_size_rejected_before_assets_or_weights(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ValueError, match="IP1/2"):
        DreamZeroDroidAdapter()
