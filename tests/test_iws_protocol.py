"""CPU-only protocol fixtures; these tests do not perform IWS inference."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.world_models.iws import IWSAdapter, prepare_image, validate_action
from evaluation.datasets.iws import prepare_iws_packet
from evaluation.execution.transport import read_packet
from evaluation.manifests import ActionSpec, VideoSpec

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/replay/iws_native_v1.json"


def test_import_is_lightweight_in_fresh_core_process():
    code = 'import sys; import adapters.world_models.iws; assert "torch" not in sys.modules; assert "lightning" not in sys.modules; assert "interactive_world_sim" not in sys.modules'
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


@pytest.mark.parametrize("shape", [(4,), (2, 4), (1, 7), (0, 4)])
def test_wrong_action_protocol_rejected(shape):
    with pytest.raises(ValueError, match=r"\[1,4\]"):
        validate_action(np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "1", 1j, True])
def test_invalid_action_values_rejected(value):
    with pytest.raises(ValueError):
        validate_action(np.full((1, 4), value))


def test_raw_metric_action_not_clipped_or_mutated_before_checkpoint_normalization():
    raw = np.array([[0.2, -0.3, 1.1, 0.5]], dtype=np.float64)
    value = validate_action(raw)
    assert value.dtype == np.float32 and value[0, 2] > 1
    value[0, 0] = 99
    assert raw[0, 0] == 0.2


def test_official_aspect_crop_not_tiny_128_pixel_crop():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :80] = 255
    image[:, 560:] = 255
    assert prepare_image(image).shape == (128, 128, 3)
    assert not prepare_image(image).any()


@pytest.mark.parametrize("image", [np.zeros((8, 8)), np.zeros((8, 8, 3)), np.zeros((0, 8, 3), np.uint8)])
def test_invalid_image_rejected(image):
    with pytest.raises(ValueError):
        prepare_image(image)


def test_config_declares_native_future_actions_and_frames():
    config = json.loads(CONFIG.read_text())
    action = ActionSpec.from_dict(config["action"])
    video = VideoSpec.from_dict(config["video"])
    assert action.dimension == 4 and action.horizon == 1 and action.rate_hz == 10
    assert video.frames_per_chunk == 1 and video.frame_count(20) == 21
    assert IWSAdapter.environment_key == config["environment"]
    assert config["gpus"] == [0] and not config["instruction_conditioning"]


def test_prepare_preserves_real_command_indices_and_initial_only_image(tmp_path, monkeypatch):
    # In-memory HDF5 facade keeps the core test independent of h5py/weights.
    commands = np.arange(32, dtype=np.float32).reshape(8, 4)
    images = np.zeros((8, 12, 16, 3), dtype=np.uint8)
    images[2] = 77

    class HDF5:
        def __enter__(self):
            return {"action": commands, "obs/images/camera_1_color": images}

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=lambda _: HDF5()))
    monkeypatch.setattr("evaluation.datasets.iws.verify_snapshot", lambda _: {
        "repository": "yixuan1999/interactive-world-sim-min-data",
        "revision": "987c175bcc29ddd0f6e6e3096b2c20df300253b0"})
    dataset = tmp_path / "pusht/val/episode_0.hdf5"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"CPU fixture, not an official episode")
    output = tmp_path / "packet.npz"
    prepare_iws_packet(CONFIG, dataset, output, start=2, chunks=3, action_source="stored_command_diagnostic")
    packet = read_packet(output)
    np.testing.assert_array_equal(packet["actions"], commands[3:6, None])
    np.testing.assert_array_equal(packet["reset_args"]["value"]["initial_action"], commands[2])
    assert packet["source"]["action_source_frame_indices"] == [3, 4, 5]
    assert np.all(packet["initial_views"]["camera_1_color"] == 77)
    with pytest.raises(FileExistsError):
        prepare_iws_packet(CONFIG, dataset, output)
    with pytest.raises(ValueError, match="exceed"):
        prepare_iws_packet(CONFIG, dataset, tmp_path / "bad.npz", chunks=8, action_source="stored_command_diagnostic")


def test_default_preparation_uses_observed_joints_and_never_reads_stored_commands(tmp_path, monkeypatch):
    joints = np.arange(8 * 14, dtype=np.float32).reshape(8, 14)
    bases = np.tile(np.eye(4, dtype=np.float32), (8, 2, 1, 1))
    images = np.zeros((8, 12, 16, 3), dtype=np.uint8)
    calls = []

    class HDF5:
        def __enter__(self):
            # No action/ee_pos fields: accessing them must fail.
            return {"obs/joint_pos": joints, "obs/world_t_robot_base": bases,
                    "obs/images/camera_1_color": images}

        def __exit__(self, *args):
            pass

    def fk(joint_values, base_values):
        calls.append((joint_values.copy(), base_values.copy()))
        return joint_values[:, :4].copy(), {"fixture": "CPU protocol only"}

    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=lambda _: HDF5()))
    monkeypatch.setattr("evaluation.datasets.iws.official_fk_actions", fk)
    monkeypatch.setattr("evaluation.datasets.iws.verify_snapshot", lambda _: {
        "repository": "yixuan1999/interactive-world-sim-min-data",
        "revision": "987c175bcc29ddd0f6e6e3096b2c20df300253b0"})
    dataset = tmp_path / "pusht/val/episode_0.hdf5"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"fixture, not official data")
    output = tmp_path / "packet.npz"
    prepare_iws_packet(CONFIG, dataset, output, start=2, chunks=3)
    packet = read_packet(output)
    np.testing.assert_array_equal(calls[0][0], joints[2:6])
    np.testing.assert_array_equal(calls[0][1], bases[2:6])
    np.testing.assert_array_equal(packet["actions"], joints[3:6, None, :4])
    np.testing.assert_array_equal(packet["reset_args"]["value"]["initial_action"], joints[2, :4])
    assert packet["source"]["action_source"] == "official_fk"


def test_setup_and_cli_help():
    subprocess.run(["bash", "-n", str(ROOT / "scripts/setup/iws.sh")], check=True)
    for script in ("scripts/download/iws.py", "scripts/replay/iws.py"):
        subprocess.run([sys.executable, str(ROOT / script), "--help"], check=True, capture_output=True)
