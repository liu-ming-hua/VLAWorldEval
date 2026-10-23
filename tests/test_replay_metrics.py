"""Pixel fidelity and reference admission are not physical-reliability claims."""
from copy import deepcopy

import numpy as np
import pytest

from evaluation.reporting.replay_metrics import score_view, validate_reference


def pair():
    video = np.stack([np.full((16, 16, 3), v, np.uint8) for v in (0, 40, 80)])
    config = {"video": {"fps": 10, "frames_per_chunk": 1, "temporal_layout": "future"}}
    packet = {"actions": np.zeros((2, 1, 4))}
    initial = {"cam": video[:1]}
    ref = {"schema_version": 1, "input_sha256": "abc", "views": {"cam": video},
           "frame_indices": {"cam": np.arange(3)}, "source": {
               "dataset_domain": "fixture", "split": "test", "episode": "0", "conversion": "identity",
               "condition_kind": "recorded_command", "frame_axis": "source_frame",
               "ground_truth_kind": "recorded_execution"}}
    return video, config, packet, initial, ref


def test_exact_match_json_safe_and_initial_not_scored():
    video, *_ = pair()
    pred = video.copy()
    pred[0] = 255
    r = score_view(pred, video)
    assert r["mse_01"] == 0 and r["frames_scored"] == 2
    assert r["mean_frame_psnr_db"] is None and r["perfect_frame_count"] == 2
    assert r["ssim_rgb_gaussian11_sigma1p5_population"] == pytest.approx(1)
    assert r["physical_action_tracking"]["status"] == "not_measured"


def test_frozen_predictor_gets_no_improvement_but_exact_motion_does():
    video, *_ = pair()
    frozen = np.repeat(video[:1], len(video), axis=0)
    assert score_view(frozen, video)["relative_mse_improvement_over_frozen"] == pytest.approx(0)
    assert score_view(video, video)["relative_mse_improvement_over_frozen"] == pytest.approx(1)


@pytest.mark.parametrize("change", ["hash", "initial", "count", "order", "camera", "ground_truth"])
def test_reference_cannot_silently_shift_replace_or_invent_truth(change):
    _, config, packet, initial, ref = pair()
    ref = deepcopy(ref)
    if change == "hash":
        ref["input_sha256"] = "different-action-input"
    elif change == "initial":
        ref["views"]["cam"][0] = 1
    elif change == "count":
        ref["views"]["cam"] = ref["views"]["cam"][:-1]
    elif change == "order":
        ref["frame_indices"]["cam"] = np.array([0, 2, 1])
    elif change == "camera":
        ref["views"]["invented_wrist"] = ref["views"]["cam"]
    else:
        ref["source"]["ground_truth_kind"] = "generated_or_counterfactual_guess"
    with pytest.raises(ValueError):
        validate_reference(ref, config, packet, initial, input_sha256="abc")


def test_reference_admission_matches_exact_timeline():
    _, config, packet, initial, ref = pair()
    assert validate_reference(ref, config, packet, initial, input_sha256="abc") is ref


@pytest.mark.parametrize("bad", [np.zeros((3, 16, 16, 3)), np.zeros((1, 16, 16, 3), np.uint8)])
def test_metrics_reject_wrong_dtype_or_no_future(bad):
    with pytest.raises(ValueError):
        score_view(bad, bad)
