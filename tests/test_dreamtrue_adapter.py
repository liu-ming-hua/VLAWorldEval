"""CPU protocol/transport tests; fixtures are not model inference evidence."""

from types import ModuleType
import sys

import numpy as np
import pytest

from adapters.base import WorldModelAdapter
from adapters.world_models.dreamtrue import (
    DreamTrueAdapter,
    PROTOCOL,
    VIEW_NAMES,
    validate_condition,
)


def condition():
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "n_real_frames": 5,
        "pad_mode": "front",
        "source": {"kind": "official_rendered_geometry", "condition_path": "fixture-only.h5"},
        "views": {
            name: {
                "rgb": np.full((5, 8, 16, 3), i, dtype=np.uint8),
                "depth": np.full((5, 8, 16), 1000, dtype=np.uint16),
                "intrinsic": np.repeat(np.eye(3, dtype=np.float32)[None], 5, axis=0),
                "extrinsic": np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0),
            }
            for i, name in enumerate(VIEW_NAMES)
        },
    }


def adapter():
    result = DreamTrueAdapter()
    # Small fixture dimensions only: public constructor deliberately fixes the
    # official profile; no tiny inference is claimed here.
    result.height, result.width, result.num_frames = 8, 16, 5
    return result


def reset_value():
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "prompt": "fixture prompt",
        "views": {
            name: np.full((8, 16, 3), 50 + i, dtype=np.uint8) for i, name in enumerate(VIEW_NAMES)
        },
    }


def test_import_and_native_boundary():
    model = DreamTrueAdapter()
    assert isinstance(model, WorldModelAdapter)
    assert model.environment_key == "world_models.dreamtrue"
    assert not model.supports_raw_actions and not model.supports_native_replay_v1
    assert model.model is None and model.last_video is None
    with pytest.raises(ValueError, match="official launcher"):
        DreamTrueAdapter(num_inference_steps=1)


def test_close_without_loaded_model_is_lightweight_and_idempotent():
    model = DreamTrueAdapter()
    model.close()
    model.close()
    assert model.model is None and model.last_video is None and model.current_views is None
    with pytest.raises(RuntimeError, match="recreate the worker"):
        model.step(None)


def test_close_releases_model_before_finalization(monkeypatch):
    import weakref
    from types import SimpleNamespace

    model = DreamTrueAdapter()
    class Payload:
        pass
    model.model = Payload()
    ref = weakref.ref(model.model)
    events = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, synchronize=lambda: events.append("sync"),
        empty_cache=lambda: events.append("empty"))))
    model.close()
    assert ref() is None and events == ["sync", "empty"]


@pytest.mark.parametrize("bad", [np.zeros((101, 7)), {}, {"schema_version": 1}, None])
def test_raw_actions_never_become_geometry(bad):
    with pytest.raises(ValueError, match="rendered geometry"):
        validate_condition(bad, num_frames=5, height=8, width=16)


@pytest.mark.parametrize(
    "mutation",
    [
        "version",
        "protocol",
        "extra_view",
        "missing_camera",
        "depth_dtype",
        "rgb_dtype",
        "frames",
        "nonfinite",
        "camera_shape",
        "source",
        "padding",
        "singular",
        "c2w",
    ],
)
def test_strict_geometry_validation(mutation):
    value = condition()
    view = value["views"]["head"]
    if mutation == "version":
        value["schema_version"] = True
    elif mutation == "protocol":
        value["protocol"] = "rawaction"
    elif mutation == "extra_view":
        value["views"]["wrist"] = view
    elif mutation == "missing_camera":
        view.pop("intrinsic")
    elif mutation == "depth_dtype":
        view["depth"] = view["depth"].astype(np.float32)
    elif mutation == "rgb_dtype":
        view["rgb"] = view["rgb"].astype(np.float32)
    elif mutation == "frames":
        view["depth"] = view["depth"][:-1]
    elif mutation == "nonfinite":
        view["extrinsic"][0, 0, 0] = np.nan
    elif mutation == "camera_shape":
        view["intrinsic"] = np.eye(3)
    elif mutation == "source":
        value["source"]["kind"] = "raw_action"
    elif mutation == "padding":
        value["n_real_frames"] = 6
    elif mutation == "singular":
        view["intrinsic"][:] = 0
    elif mutation == "c2w":
        view["extrinsic"][:, 3, 3] = 0
    with pytest.raises(ValueError):
        validate_condition(value, num_frames=5, height=8, width=16)


def test_validation_copies_and_preserves_depth_units():
    value = condition()
    valid = validate_condition(value, num_frames=5, height=8, width=16)
    valid["head"]["depth"][:] = 123
    assert value["views"]["head"]["depth"][0, 0, 0] == 1000


def test_reset_requires_actual_three_view_mapping_and_copies():
    model = adapter()
    with pytest.raises(ValueError):
        model.reset(np.zeros((8, 16, 3), dtype=np.uint8))
    value = reset_value()
    model.reset(value)
    value["views"]["head"][:] = 0
    assert model.current_views["head"][0, 0, 0] == 50
    assert model.last_video is None


def mock_contract(monkeypatch):
    from PIL import Image

    module = ModuleType("examples.wanvideo.model_training.vace_contract")
    module.build_mask_frames = lambda n, w, h, **kw: [
        Image.new("L", (w, h), 0 if i < n - kw["n_real_frames"] + 1 else 255) for i in range(n)
    ]
    monkeypatch.setitem(sys.modules, "examples.wanvideo.model_training.vace_contract", module)


def test_step_dispatches_native_fields_and_returns_time_major_views(monkeypatch):
    mock_contract(monkeypatch)
    model = adapter()
    model.reset(reset_value())
    calls = []

    class Pipeline:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return [
                [np.full((8, 16, 3), i * 20 + t, dtype=np.uint8) for t in range(5)]
                for i in range(3)
            ]

    model.model = Pipeline()
    value = condition()
    value["n_real_frames"] = 3
    result = model.step_output(value)
    assert model.last_video.shape == (3, 3, 8, 16, 3)
    assert tuple(result.views) == VIEW_NAMES
    assert result.views["hand_right"][-1, 0, 0, 0] == 44
    assert result.robot_state is None
    assert result.metadata["temporal_layout"] == "inclusive"
    args = calls[0]
    assert args["vace_scale"] == 1 and args["cfg_scale"] == 1
    assert args["vace_depth_video"][0][0].dtype == np.uint16
    assert [np.asarray(frame)[0, 0] for frame in args["vace_mask_video"][0]] == [0, 0, 0, 255, 255]
    assert np.asarray(args["vace_reference_image"][0][0])[0, 0, 0] == 50
    model.step(value)
    assert np.asarray(calls[1]["vace_reference_image"][0][0])[0, 0, 0] == 4


def test_failed_generation_invalidates_cached_video(monkeypatch):
    mock_contract(monkeypatch)
    model = adapter()
    model.reset(reset_value())

    class Bad:
        def generate(self, **kwargs):
            raise RuntimeError("fixture failure")

    model.model = Bad()
    model.last_video = np.zeros((5, 3, 8, 16, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="fixture failure"):
        model.step(condition())
    assert model.last_video is None
    with pytest.raises(RuntimeError, match="recreate"):
        model.reset(reset_value())


def test_worker_transport_preserves_native_mapping(tmp_path):
    from evaluation.execution.transport import read_packet, write_packet

    value = condition()
    path = tmp_path / "native.npz"
    write_packet(path, value)
    received = read_packet(path)
    validate_condition(received, num_frames=5, height=8, width=16)
    assert received["views"]["head"]["depth"].dtype == np.uint16
