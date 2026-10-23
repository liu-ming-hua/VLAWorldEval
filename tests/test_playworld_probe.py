"""Core-only bounds/provenance checks for the synthetic action diagnostic."""
import hashlib
import json

import numpy as np
import pytest

from evaluation.datasets.playworld import prepare_single_axis_probe, single_axis_probe_actions
from evaluation.execution.transport import write_packet


def stats():
    return {"state_01": [-1.0] * 7, "state_99": [1.0] * 7}


def test_smooth_probe_reaches_two_cm_at_selected_tick_12_and_keeps_other_dimensions():
    initial = np.array([0.4, 0.2, 0.3, -0.6, 0.1, 0.2, 0.5], np.float32)
    actions = single_axis_probe_actions(initial, stats())
    assert actions.shape == (1, 15, 7) and actions.dtype == np.float32
    np.testing.assert_array_equal(actions[0, 0], initial)
    np.testing.assert_array_equal(actions[0, :, 1:], np.repeat(initial[None, 1:], 15, axis=0))
    np.testing.assert_allclose(actions[0, ::3, 0] - initial[0], [0, 0.003125, 0.01, 0.016875, 0.02], atol=3e-8)
    assert np.all(np.diff(actions[0, :13, 0]) > 0)
    np.testing.assert_array_equal(actions[0, 12:], np.repeat(actions[0, 12:13], 3, axis=0))


@pytest.mark.parametrize("dimension", [0, 3, 6])
def test_out_of_domain_actions_are_rejected_instead_of_clipped(dimension):
    initial = np.zeros(7, np.float32)
    initial[dimension] = 0.99 if dimension == 0 else 1.01
    with pytest.raises(ValueError, match="bounds"):
        single_axis_probe_actions(initial, stats())


@pytest.mark.parametrize("distance", [0, -0.02, 0.03, float("nan")])
def test_probe_cannot_expand_motion_scope(distance):
    with pytest.raises(ValueError, match="2cm"):
        single_axis_probe_actions(np.zeros(7), stats(), displacement_m=distance)


def test_probe_npy_provenance_hash_and_changed_initial_detection(tmp_path):
    baseline = tmp_path / "hold"
    baseline.mkdir()
    initial = np.zeros(7, np.float32)
    original = tmp_path / "original.yaml"
    original.write_bytes(b"fixture source")
    packet = {"actions": np.repeat(initial[None, None], 15, axis=1),
              "reset_args": {"value": {"initial_state": initial}},
              "source": {"input_sha256": {str(original): hashlib.sha256(original.read_bytes()).hexdigest()}}}
    write_packet(baseline.with_suffix(".inputs.npz"), packet)
    (baseline / "summary.json").write_text(json.dumps({"status": "completed", "world_model_calls": 1, "input_sha256": "fixture"}))
    (baseline / "config.json").write_text('{"seed":0}')
    source = tmp_path / "stats.py"
    source.write_text("DATASET_STATS: dict = " + repr({"droid": stats()}))
    output = tmp_path / "probe"
    report = prepare_single_axis_probe(baseline, output, source)
    assert report["synthetic"] is True and report["recorded_action_replay"] is False
    assert report["actions_sha256"] == hashlib.sha256((output / "actions.npy").read_bytes()).hexdigest()
    assert report["normalization_clipping_required"] is False
    original.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="changed since hold"):
        prepare_single_axis_probe(baseline, tmp_path / "changed_probe", source)
