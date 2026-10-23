"""Published validation membership is not paper-test membership."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.datasets.ctrl_world import released_selection


def test_actual_pinned_release_has_all97_distinct_windows_but_only_two_episodes():
    root = Path(__file__).resolve().parents[1]/"third_party/ctrl-world"
    if not (root/".git").exists():
        pytest.skip("Optional pinned Ctrl checkout absent")
    samples, files = released_selection(root)
    assert len(samples) == 97
    assert {s["episode_id"] for s in samples} == {199, 18599}
    assert len(files) == 1
    assert next(iter(files.values()))["sha256"] == "a58c908fc9d787eb52b750e7004967acd1f40c8a72f581dc1852300a9383a100"


def test_changed_validation_asset_is_rejected(tmp_path, monkeypatch):
    from evaluation.datasets import ctrl_world as module
    path = tmp_path/"sample.json"
    path.write_text("changed fixture")
    replies = iter(["a"*40, "b"*40])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: next(replies))
    with pytest.raises(ValueError, match="differs"):
        module.verified_asset(tmp_path, "sample.json")


def test_released_validation_flag_never_relabels_one_selected_window_as_all(tmp_path):
    from scripts.benchmarks.native_replay import main
    with pytest.raises(ValueError, match="all97"):
        main(["--model", "ctrl_world", "--full-ctrl-released-validation", "--dataset", str(tmp_path),
              "--config", "unused", "--output", str(tmp_path/"new"), "--start", "1"])


@pytest.mark.parametrize("problem", [None, "source", "condition", "endpoint", "crop"])
def test_common_admission_accepts_only_audited_clamped_tail(tmp_path, monkeypatch, problem):
    from evaluation.execution import native_replay as module
    from evaluation.execution.transport import write_packet
    from evaluation.datasets.ctrl_world import inclusive_action_windows
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"configs/replay/ctrl_world_native_v2.json").read_text())
    config.update(gpus=[], kwargs={}, environment="world_models.fixture", instruction_conditioning=False)
    config["video"] = {**config["video"], "final_chunk_valid_frames": 2}
    states = np.arange(7*7, dtype=np.float32).reshape(7, 7)
    actions, indices, _ = inclusive_action_windows(states)
    source = {"conversion": "synthetic admission fixture only", "start_frame": 0, "end_frame": 18,
              "last_recorded_source_frame": 18, "tail_policy": "repeat_last_recorded_state",
              "source_frames_per_chunk": 12, "temporal_layout": "inclusive",
              "action_source_frame_indices": indices.tolist()}
    if problem == "source":
        source["action_source_frame_indices"][-1][-1] = 24
    elif problem == "condition":
        actions[-1, -1] += 1
    elif problem == "endpoint":
        source["last_recorded_source_frame"] = 21
    elif problem == "crop":
        config["video"]["final_chunk_valid_frames"] = 1
    first = np.zeros((16, 16, 3), np.uint8)
    packet = {"protocol": config["action"], "actions": actions, "source": source,
              "reset_args": {"image": [first]*3, "first_action": states[0]},
              "initial_views": {name: first for name in config["output_views"]}}
    conf, input_path = tmp_path/"config.json", tmp_path/"input.npz"
    conf.write_text(json.dumps(config))
    write_packet(input_path, packet)
    monkeypatch.setattr(module, "get_environment", lambda _: SimpleNamespace(scope="world_model", exists=True, python=sys.executable))
    if problem:
        with pytest.raises(ValueError):
            module.admit_native_replay(conf, input_path)
    else:
        _, _, video, *_ = module.admit_native_replay(conf, input_path)
        assert video.frame_count(2) == 7
