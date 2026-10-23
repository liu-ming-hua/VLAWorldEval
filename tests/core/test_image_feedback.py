"""Visual diagnostics must not reset time, state, actions or random seeds."""

import json
from types import MethodType

import numpy as np
import pytest
from PIL import Image

from evaluation.execution.native_replay import admit_image_anchors
from evaluation.execution.transport import write_packet

pytestmark = pytest.mark.process


@pytest.fixture
def anchor_fixture(tmp_path):
    config = {
        "action": {
            "dimension": 2,
            "horizon": 2,
            "rate_hz": 10,
            "representation": "fixture",
            "frame": "fixture",
            "semantics": "fixture",
        }
    }
    initial = np.zeros((16, 16, 3), np.uint8)
    packet = {
        "actions": np.ones((3, 2, 2)),
        "reset_args": {"value": initial},
        "source": {"fps": 10, "start_frame": 5},
    }
    anchors = {
        "images": np.stack([initial, initial + 10, initial + 20]),
        "source_frame_indices": np.array([5, 7, 9]),
        "source": {"conversion": "synthetic RGB only"},
        "intervention": "ground_truth_visual_context",
    }
    return tmp_path / "anchors.npz", config, packet, anchors


def test_visual_anchor_boundary_alignment(anchor_fixture):
    path, config, packet, anchors = anchor_fixture
    write_packet(path, anchors)
    actual = admit_image_anchors(path, packet, config)
    assert np.array_equal(actual["images"], anchors["images"])


@pytest.mark.parametrize("kind", ["state", "time", "initial", "geometry", "dtype"])
def test_invalid_or_nonvisual_interventions_rejected(anchor_fixture, kind):
    path, config, packet, anchors = anchor_fixture
    if kind == "state":
        anchors["states"] = np.zeros((3, 7))
    elif kind == "time":
        anchors["source_frame_indices"] += 1
    elif kind == "initial":
        anchors["images"][0] += 1
    elif kind == "geometry":
        anchors["images"] = anchors["images"][:, :8]
    else:
        anchors["images"] = anchors["images"].astype(np.float32)
    write_packet(path, anchors)
    with pytest.raises(ValueError):
        admit_image_anchors(path, packet, config)


def test_joint_reanchor_keeps_command_proxy_and_step_seed():
    from adapters.world_models.cosmos3_droid_joint_fd import Cosmos3DroidJointFDAdapter

    wm = object.__new__(Cosmos3DroidJointFDAdapter)
    original = np.zeros((540, 640, 3), np.uint8)
    wm._observation = {
        "observation/image": original,
        "observation/joint_position": np.arange(7),
        "observation/gripper_position": np.array([0.2]),
        "prompt": "test",
    }
    wm.seed, wm._index = 4, 7
    state = wm._observation["observation/joint_position"]
    info = wm.reanchor_image(original + 20)
    assert wm._observation["observation/joint_position"] is state
    assert wm._index == 7 and info["diffusion_seed"] == 11 and info["state_changed"] is False
    assert wm._observation["observation/image"][0, 0, 0] == 20


def test_eef_reanchor_accepts_original_geometry_after_native_border_crop():
    from adapters.world_models.cosmos3_nano_droid_fd import Cosmos3NanoDroidFDAdapter

    wm = object.__new__(Cosmos3NanoDroidFDAdapter)
    wm._conditioning_size = (640, 540)
    wm.current_image = Image.new("RGB", (640, 528))
    wm.seed, wm._step_index = 3, 8
    info = wm.reanchor_image(np.full((540, 640, 3), 60, np.uint8))
    assert wm.current_image.size == (640, 540) and wm._step_index == 8
    assert info["diffusion_seed"] == 11 and info["state_changed"] is False


def test_eef_output_border_restoration_does_not_change_recurrent_image():
    from adapters.world_models.cosmos3_nano_droid_fd import Cosmos3NanoDroidFDAdapter

    wm = object.__new__(Cosmos3NanoDroidFDAdapter)
    wm._conditioning_size = (640, 540)
    wm.current_image = Image.new("RGB", (640, 528))
    wm.last_policy_state = None

    def step(self, action):
        self.last_video = np.zeros((16, 528, 640, 3), np.uint8)
        return self.last_video[-1]

    wm.step = MethodType(step, wm)
    output = wm.step_output(np.zeros((16, 10)))
    assert output.views["droid.cosmos_composite"].shape == (16, 540, 640, 3)
    assert wm.current_image.size == (640, 528)


def test_real_worker_image_feedback_never_restarts_loop(tmp_path, monkeypatch):
    from evaluation.execution.client import ModelWorker
    from tests.support.builders import replay_fixture

    module, config, packet_path, packet = replay_fixture(tmp_path, monkeypatch)

    def launch(*a, **k):
        k.update(module="tests.support.runtime", adapter="FeedbackFixtureWM")
        return ModelWorker(*a, **k)

    monkeypatch.setattr(module, "ModelWorker", launch)
    packet["source"].update(fps=10, start_frame=0)
    write_packet(packet_path, packet)
    images = np.zeros((20, 16, 16, 3), np.uint8)
    anchors = {
        "images": images,
        "source_frame_indices": np.arange(20) * 2,
        "source": {"conversion": "synthetic fixture"},
        "intervention": "ground_truth_visual_context",
    }
    path = tmp_path / "anchors.npz"
    write_packet(path, anchors)
    out = tmp_path / "run"
    report = module.run_native_replay(config, packet_path, out, image_anchors=path)
    assert report["status"] == "completed" and report["world_model_calls"] == 20
    assert len(report["anchor_events"]) == 19
    assert [e["details"]["step_index"] for e in report["anchor_events"]] == list(range(1, 20))
    assert report["vla_calls"] == 0
    assert np.array_equal(np.load(out / "actions.npy"), packet["actions"])
    assert (
        json.loads((out / "summary.json").read_text())["feedback_mode"]
        == "ground_truth_visual_context"
    )
