"""Small CPU tensor parity checks in IRASim's optional model environment."""
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np
import pytest
from adapters.world_models.irasim import IRASimAdapter

_ROOT = Path(__file__).resolve().parents[1]
os.environ.update(XDG_CACHE_HOME=str(_ROOT / ".cache"), TORCH_HOME=str(_ROOT / ".cache/torch"),
                  HF_HOME=str(_ROOT / ".cache/huggingface"), HF_HUB_CACHE=str(_ROOT / ".cache/huggingface/hub"))

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def adapter():
    model = IRASimAdapter.__new__(IRASimAdapter)
    model.device = torch.device("cpu")
    model.num_inference_steps = 50
    model._latent = None
    return model


def test_step_preserves_terminal_latent_and_emits_only_future_frames():
    model = adapter()
    image = np.zeros((288, 512, 3), dtype=np.uint8)
    initial = np.zeros((4, 36, 64), dtype=np.float32)
    model.reset({"image": image, "latent": initial})
    calls = []
    def pipeline(actions, **kwargs):
        assert not torch.is_grad_enabled()
        calls.append((actions.clone(), kwargs["mask_x"].clone()))
        video = torch.zeros((1, 16, 3, 288, 512))
        video[:, 0] = -1
        video[:, -1] = 1
        latent = torch.ones((1, 16, 4, 36, 64)) * len(calls)
        return video, latent
    model.pipeline = pipeline
    terminal = model.step(np.full((15, 2), 0.025, dtype=np.float32))
    torch.testing.assert_close(calls[0][0], torch.full((1, 15, 2), 0.5))
    assert model.last_video.shape == (15, 288, 512, 3)
    assert model.last_video.dtype == np.uint8
    assert (terminal == 255).all()
    assert (model.last_video[0] == 127).all()
    assert tuple(model.last_latents.shape) == (15, 4, 36, 64)
    model.step(np.zeros((15, 2)))
    torch.testing.assert_close(calls[1][1], torch.ones((1, 1, 4, 36, 64)))
    result = model.step_output(np.zeros((15, 2)), output_views=("primary",))
    assert set(result.views) == {"primary"}
    assert result.robot_state is None
    assert not torch.cuda.is_initialized()


def test_reset_encodes_same_scaled_latent_protocol():
    model = adapter()
    captured = []
    def encode(frame):
        captured.append(frame)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: torch.ones((1, 4, 36, 64))))
    model.vae = SimpleNamespace(encode=encode, config=SimpleNamespace(scaling_factor=0.13025))
    model.reset(np.full((288, 512, 3), 255, dtype=np.uint8))
    torch.testing.assert_close(captured[0], torch.ones((1, 3, 288, 512)))
    torch.testing.assert_close(model._latent, torch.full((1, 1, 4, 36, 64), 0.13025))
    assert model.last_video is None


def test_bundled_sample_matches_upstream_dataset_scaling():
    import json
    root = Path(__file__).resolve().parents[1]
    repo = root / "third_party/irasim"
    if not repo.exists():
        pytest.skip("Optional official checkout absent")
    ann = json.loads((repo / "robotdata/opensource_robotdata/languagetable/annotation/test/example.json").read_text())
    raw = np.asarray(ann["actions"][:15])
    np.testing.assert_allclose(IRASimAdapter.validate_actions(raw), (raw * [20., 20.]).astype(np.float32), rtol=1e-6)
    upstream = (repo / "dataset/dataset_2D.py").read_text()
    assert "self.c_act_scaler = [20.0, 20.0]" in upstream
