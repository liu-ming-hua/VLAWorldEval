"""Padding is a native model condition, never paired physical ground truth."""

from dataclasses import asdict

import numpy as np
import pytest

from evaluation.contracts import WorldModelOutput
from evaluation.manifests import VideoSpec


def test_legacy_spec_shape_serialization_and_assembly_are_unchanged():
    spec = VideoSpec.from_dict({"fps": 4, "frames_per_chunk": 15, "temporal_layout": "future"})
    assert type(spec) is VideoSpec
    assert asdict(spec) == {"fps": 4, "frames_per_chunk": 15, "temporal_layout": "future"}
    initial = np.zeros((1, 12, 12, 3), np.uint8)
    chunks = [np.full((15, 12, 12, 3), i, np.uint8) for i in (1, 2)]
    np.testing.assert_array_equal(
        spec.assemble(initial, chunks), np.concatenate([initial, *chunks])
    )
    assert spec.frame_count(2) == 31


@pytest.mark.parametrize("chunks,tail", [(1, 1), (1, 15), (2, 2), (3, 14)])
def test_tail_retains_all_real_frames_and_no_padded_frames(chunks, tail):
    spec = VideoSpec.from_dict({"fps": 4, "frames_per_chunk": 15, "final_chunk_valid_frames": tail})
    initial = np.zeros((1, 12, 12, 3), np.uint8)
    windows = [np.full((15, 12, 12, 3), i + 1, np.uint8) for i in range(chunks)]
    result = spec.assemble(initial, windows)
    assert len(result) == spec.frame_count(chunks) == 1 + 15 * (chunks - 1) + tail
    assert np.all(result[-tail:] == chunks)
    spec.validate_output(WorldModelOutput(views={"primary": windows[-1]}), ("primary",))
    with pytest.raises(ValueError):
        spec.validate_output(WorldModelOutput(views={"primary": windows[-1][:1]}), ("primary",))


@pytest.mark.parametrize("tail", [0, -1, 16, True, 1.5, None])
def test_invalid_tail_rejected(tail):
    with pytest.raises(ValueError):
        VideoSpec.from_dict({"fps": 4, "frames_per_chunk": 15, "final_chunk_valid_frames": tail})


def test_inclusive_tail_counts_future_transitions_not_the_shared_anchor():
    spec = VideoSpec.from_dict(
        {
            "fps": 5,
            "frames_per_chunk": 5,
            "temporal_layout": "inclusive",
            "final_chunk_valid_frames": 2,
        }
    )
    windows = [np.full((5, 12, 12, 3), i, np.uint8) for i in (1, 2)]
    initial = np.zeros((1, 12, 12, 3), np.uint8)
    value = spec.assemble(initial, windows)
    assert value.shape == (7, 12, 12, 3) and spec.frame_count(2) == 7
    assert np.all(value[:4] == 1) and np.all(value[4:] == 2)
    with pytest.raises(ValueError):
        VideoSpec.from_dict(
            {
                "fps": 5,
                "frames_per_chunk": 5,
                "temporal_layout": "inclusive",
                "final_chunk_valid_frames": 5,
            }
        )


@pytest.mark.parametrize("bad", [None, "endpoint", "policy", "padding"])
def test_ctrl_inclusive_tail_validates_actual_source_ids_and_repeated_condition(bad):
    from evaluation.timeline import ctrl_tail_source_indices

    spec = VideoSpec.from_dict(
        {
            "fps": 5,
            "frames_per_chunk": 5,
            "temporal_layout": "inclusive",
            "final_chunk_valid_frames": 2,
        }
    )
    source = {
        "start_frame": 0,
        "end_frame": 18,
        "last_recorded_source_frame": 18,
        "tail_policy": "repeat_last_recorded_state",
    }
    states = np.arange(7 * 7, dtype=np.float32).reshape(7, 7)
    actual = np.stack([states[[0, 1, 2, 3, 4]], states[[4, 5, 6, 6, 6]]])
    if bad == "endpoint":
        source["last_recorded_source_frame"] = 21
    elif bad == "policy":
        source["tail_policy"] = "invented"
    elif bad == "padding":
        actual[-1, -1] = 999
    if bad:
        with pytest.raises(ValueError):
            ctrl_tail_source_indices(source, spec, actual)
    else:
        np.testing.assert_array_equal(
            ctrl_tail_source_indices(source, spec, actual), [[0, 3, 6, 9, 12], [12, 15, 18, 18, 18]]
        )


@pytest.mark.parametrize("count", [1, 14, 15, 16, 17, 29, 30, 31, 52])
def test_irasim_long_tail_keeps_all_actions_and_repeats_first_real_tail(count):
    from evaluation.datasets.irasim_reference import long_action_windows

    raw = np.arange(count * 2, dtype=np.float32).reshape(count, 2)
    windows, valid = long_action_windows(raw)
    assert windows.shape == ((count + 14) // 15, 15, 2)
    flat = windows.reshape(-1, 2)
    np.testing.assert_array_equal(flat[:count], raw)
    if len(flat) > count:
        np.testing.assert_array_equal(
            flat[count:],
            np.repeat(raw[(count // 15) * 15 : (count // 15) * 15 + 1], len(flat) - count, axis=0),
        )
    spec = VideoSpec.from_dict(
        {"fps": 4, "frames_per_chunk": 15, "final_chunk_valid_frames": valid}
    )
    assert spec.frame_count(len(windows)) == count + 1


def test_shared_executor_crops_padding_before_reference_scores_and_video(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from evaluation.execution import native_replay as module
    from evaluation.execution.transport import write_packet
    from evaluation.reporting.task_scores import _digest

    monkeypatch.setattr(
        module,
        "get_environment",
        lambda _: SimpleNamespace(scope="world_model", exists=True, python=sys.executable),
    )
    action = {
        "dimension": 2,
        "horizon": 15,
        "representation": "xy",
        "frame": "fixture",
        "semantics": "delta",
        "rate_hz": 4,
        "normalization": "none",
        "temporal_layout": "future",
    }
    config = {
        "schema_version": 1,
        "environment": "world_models.fixture",
        "module": "adapters.world_models.irasim",
        "adapter": "IRASimAdapter",
        "kwargs": {},
        "gpus": [],
        "memory_gib": 1,
        "seed": 0,
        "action": action,
        "video": {"fps": 4, "frames_per_chunk": 15, "final_chunk_valid_frames": 2},
        "output_views": ["primary"],
        "reset_operation": "reset",
        "instruction_conditioning": False,
        "worker_options": {},
    }
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps(config))
    first = np.zeros((16, 16, 3), np.uint8)
    input_path = tmp_path / "input.npz"
    write_packet(
        input_path,
        {
            "protocol": action,
            "actions": np.zeros((2, 15, 2)),
            "reset_args": {"value": first},
            "initial_views": {"primary": first},
            "source": {"conversion": "synthetic fixture only"},
        },
    )
    truth = np.concatenate(
        [first[None], np.ones((15, 16, 16, 3), np.uint8), np.full((2, 16, 16, 3), 2, np.uint8)]
    )
    reference = tmp_path / "reference.npz"
    write_packet(
        reference,
        {
            "schema_version": 1,
            "input_sha256": _digest(input_path),
            "views": {"primary": truth},
            "frame_indices": {"primary": np.arange(18)},
            "source": {
                "dataset_domain": "synthetic_fixture",
                "split": "test",
                "episode": "0",
                "conversion": "fixture only",
                "condition_kind": "fixture",
                "frame_axis": "tick",
                "ground_truth_kind": "recorded_execution",
            },
        },
    )

    class Worker:
        ready, failure, steps = {"fixture_only": True}, None, 0

        def __init__(self, *args, **kwargs):
            pass

        def call(self, operation, **kwargs):
            assert "reference" not in kwargs
            if operation == "step":
                self.steps += 1
                frames = np.full((15, 16, 16, 3), self.steps, np.uint8)
                if self.steps == 2:
                    frames[2:] = 255  # Must never enter real-future scores/video.
                return {"views": {"primary": frames}, "metadata": {"fixture_only": True}}

        def close(self):
            return {"returncode": 0, "signals": [], "errors": []}

    videos = []
    monkeypatch.setattr(module, "ModelWorker", Worker)

    def save_video(path, value, **kwargs):
        videos.append(value.copy())
        path.write_bytes(b"fixture, not encoded model output")

    monkeypatch.setattr(module.media, "write_video", save_video)
    result = module.run_native_replay(conf, input_path, tmp_path / "run", reference_path=reference)
    assert result["world_model_calls"] == 2 and result["vla_calls"] == 0
    assert result["frames_per_view"] == 18
    assert result["reference_evaluation"]["metrics"]["primary"]["frames_scored"] == 17
    assert result["reference_evaluation"]["metrics"]["primary"]["mse_01"] == 0
    np.testing.assert_array_equal(videos[0], truth)
