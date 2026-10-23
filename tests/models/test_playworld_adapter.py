"""Weight-free contract checks; fixture outputs are not model validation."""

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.base import WorldModelAdapter
from adapters.world_models.playworld import VIEW_NAMES, PlayWorldAdapter

pytestmark = pytest.mark.process

ROOT = Path(__file__).resolve().parents[2]


def images():
    return {
        name: np.full((192, 320, 3), index * 40, np.uint8) for index, name in enumerate(VIEW_NAMES)
    }


def adapter():
    model = SimpleNamespace()
    model._bootstrap_current_latent = lambda observation: observation["views"]
    calls = []

    def rollout(state, observation, actions, instruction):
        calls.append((state, observation, actions.copy(), instruction))
        return {
            "frames": np.concatenate(
                [images()[name][None].repeat(5, 0) for name in VIEW_NAMES], axis=1
            ),
            "next_state": {"step": len(calls)},
            "latents": "fixture",
        }

    model.rollout = rollout
    value = object.__new__(PlayWorldAdapter)
    value.model = model
    value._state = None
    value._inference_context = nullcontext
    value.prompt = "put the carrot in the bowl"
    value.action_downsample = 3
    value.last_video = value.last_latents = None
    return value, calls


def test_import_does_not_require_torch_or_initialize_gpu():
    code = "import sys; import adapters.world_models.playworld; assert 'torch' not in sys.modules; assert 'diffusers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_real_adapter_base_and_native_profile():
    assert issubclass(PlayWorldAdapter, WorldModelAdapter)
    config = json.loads((ROOT / "configs/replay/playworld_native_v1.json").read_text())
    assert config["environment"] == PlayWorldAdapter.environment_key
    assert config["action"]["horizon"] == 15
    assert config["action"]["temporal_layout"] == "future"
    assert config["video"] == {"fps": 5, "frames_per_chunk": 5, "temporal_layout": "future"}
    assert config["gpus"] == [2]


def test_reset_and_steps_preserve_official_state_and_three_video_views():
    value, calls = adapter()
    value.reset({"views": images(), "initial_state": np.arange(7), "instruction": "pick carrot"})
    assert np.array_equal(value._state["_initial_robot_state"], np.arange(7))
    action = np.arange(105, dtype=np.float32).reshape(15, 7)
    out = value.step_output(action, output_views=VIEW_NAMES)
    assert list(out.views) == list(VIEW_NAMES)
    assert out.robot_state is None
    assert all(v.shape == (5, 192, 320, 3) for v in out.views.values())
    assert [v[0, 0, 0, 0] for v in out.views.values()] == [0, 40, 80]
    assert value.last_video.shape == (5, 3, 192, 320, 3)
    assert np.array_equal(calls[0][2], action)  # stride/normalization belong to upstream
    assert calls[0][3] == "pick carrot"
    value.step(action)
    assert calls[1][0] == {"step": 1}
    assert value._state == {"step": 2}


@pytest.mark.parametrize("error", ["single", "missing", "float", "size", "four"])
def test_no_view_duplication_or_implicit_resize(error):
    views = images()
    if error == "single":
        views = views[VIEW_NAMES[0]]
    elif error == "missing":
        views.pop("wrist")
    elif error == "float":
        views["wrist"] = views["wrist"].astype(np.float32)
    elif error == "size":
        views["wrist"] = np.zeros((224, 224, 3), np.uint8)
    else:
        views = [*views.values(), views["wrist"]]
    with pytest.raises(ValueError):
        PlayWorldAdapter._views(views)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((13, 7)),
        np.zeros((5, 7)),
        np.zeros((16, 7)),
        np.zeros((15, 8)),
        np.full((15, 7), np.nan),
    ],
)
def test_invalid_action_chunks_not_padded_or_truncated(bad):
    value, calls = adapter()
    value.reset(images())
    with pytest.raises(ValueError):
        value.step(bad)
    assert calls == []
    assert value._state is None


def test_explicit_five_sample_mode_and_missing_reset():
    value, calls = adapter()
    with pytest.raises(RuntimeError, match="reset"):
        value.step(np.zeros((15, 7)))
    value.action_downsample = 1
    value.reset(images())
    value.step(np.zeros((5, 7)))
    assert calls[0][2].shape == (5, 7)


def test_failed_official_call_requires_new_reset():
    value, _ = adapter()
    value.reset(images())

    def fail(*args, **kwargs):
        raise RuntimeError("official failure")

    value.model.rollout = fail
    with pytest.raises(RuntimeError, match="official failure"):
        value.step(np.zeros((15, 7)))
    assert value._state is None and value.last_video is None


def test_reset_requires_instruction_and_clears_old_episode():
    value, _ = adapter()
    value.reset(images())
    with pytest.raises(ValueError, match="instruction"):
        value.reset({"views": images(), "instruction": ""})
    assert value._state is None


def test_replay_cli_help_without_torch():
    result = subprocess.run(
        [sys.executable, "scripts/replay/playworld.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--hold-probe" in result.stdout and "--actions" in result.stdout


def test_memory_budget_has_no_machine_specific_upper_bound(tmp_path):
    with pytest.raises(FileNotFoundError):
        PlayWorldAdapter(memory_gib=24, repo_path=tmp_path / "missing")


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True])
def test_invalid_memory_budget_rejected_before_imports(budget):
    with pytest.raises(ValueError, match="memory_gib"):
        PlayWorldAdapter(memory_gib=budget)


def test_gpu_guard_resolves_unindexed_cuda_and_preserves_worker_limit(monkeypatch):
    calls = []
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        mem_get_info=lambda index: (30 * 2**30, 80 * 2**30),
        get_per_process_memory_fraction=lambda index: 0.2,
        set_per_process_memory_fraction=lambda fraction, index: calls.append((fraction, index)),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    value = object.__new__(PlayWorldAdapter)
    value.device = SimpleNamespace(type="cuda", index=None)
    value.memory_gib = 18
    value._guard_gpu()
    assert calls == [(0.2, 0)]
