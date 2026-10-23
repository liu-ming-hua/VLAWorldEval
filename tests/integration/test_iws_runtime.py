"""Torch CPU fixtures for native history/failure semantics, no learned inference."""

import numpy as np
import pytest

from adapters.world_models.iws import IWSAdapter

torch = pytest.importorskip("torch", reason="Core intentionally has no Torch; run in IWS env too")


def fixture_adapter():
    calls = []

    class Normalizer:
        def __getitem__(self, key):
            return self

        def normalize(self, value):
            return value

    class Model:
        normalizer = Normalizer()
        dtype = torch.float32

        def encoder_forward(self, image):
            return torch.ones((1, 4, 2, 2))

        def dynamics_forward(self, latent, actions):
            calls.append((latent.clone(), actions.clone()))
            assert actions.shape[1] == latent.shape[1] + 1
            return latent[:, -1:] + 1

    adapter = object.__new__(IWSAdapter)
    adapter.device = torch.device("cpu")
    adapter.model = Model()
    adapter._render = lambda *a, **kw: torch.full((1, 3, 128, 128), 0.5)
    adapter._closed = adapter._failed = False
    adapter._latent = None
    adapter._actions = []
    adapter.last_video = None
    adapter.last_policy_state = None
    return adapter, calls


def reset(adapter):
    adapter.reset({"image": np.zeros((128, 128, 3), np.uint8), "initial_action": np.zeros(4)})


def test_history_survives_shared_step_output_and_slides_at_ten():
    adapter, calls = fixture_adapter()
    reset(adapter)
    for _ in range(13):
        out = adapter.step_output(np.ones((1, 4)), output_views=("camera_1_color",))
        assert out.views["camera_1_color"].shape == (1, 128, 128, 3)
        assert out.robot_state is None
    assert [x[0].shape[1] for x in calls] == list(range(1, 11)) + [10] * 3
    assert adapter._latent[0, -1, 0, 0, 0] == 14
    reset(adapter)
    assert adapter._latent.shape[1] == 1 and len(adapter._actions) == 1
    assert adapter.last_video is None


def test_failure_requires_recreation_and_invalid_input_does_not_poison():
    adapter, _ = fixture_adapter()
    with pytest.raises(RuntimeError):
        adapter.step(np.zeros((1, 4)))
    reset(adapter)
    with pytest.raises(ValueError):
        adapter.step(np.zeros((2, 4)))
    assert not adapter._failed
    adapter._render = lambda *a, **kw: torch.full((1, 3, 128, 128), float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite"):
        adapter.step(np.zeros((1, 4)))
    with pytest.raises(RuntimeError, match="recreate"):
        reset(adapter)
    adapter.close()
    assert adapter.model is None and adapter._closed
