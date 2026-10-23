"""Scores must expose missing tracks and episode-level uncertainty."""
import numpy as np
import pytest

from evaluation.reporting.motion import trajectory_metrics, event_metrics, episode_bootstrap


def track():
    return np.column_stack((np.linspace(.1, .8, 20), np.full(20, .4)))


def test_perfect_track_and_initial_frame_excluded():
    p = track()
    r = p.copy()
    p[0] = [.99, .99]
    valid = np.ones(20, bool)
    result = trajectory_metrics(p, r, valid, valid, fps=10)
    assert result["status"] == "scored" and result["paired_frames"] == 19
    assert result["ade"] == result["fde"] == result["symmetric_hausdorff"] == result["dtw_mean_path_distance"] == 0


def test_offset_is_measured_in_normalized_image_coordinates():
    p = track()
    valid = np.ones(20, bool)
    result = trajectory_metrics(p+[0, .1], p, valid, valid, fps=10)
    assert result["ade"] == pytest.approx(.1)
    assert result["fde"] == pytest.approx(.1)
    assert result["symmetric_hausdorff"] == pytest.approx(.1)
    assert result["speed_wasserstein"] == pytest.approx(0)


def test_frozen_prediction_is_not_good_motion():
    p = track()
    valid = np.ones(20, bool)
    result = trajectory_metrics(np.repeat(p[:1], 20, axis=0), p, valid, valid, fps=10)
    assert result["ade"] > .3 and result["fde"] > .6 and result["speed_wasserstein"] > 0


def test_missing_detection_is_unscored_not_a_perfect_origin():
    p = track()
    result = trajectory_metrics(p, p, np.zeros(20, bool), np.ones(20, bool), fps=10)
    assert result["status"] == "insufficient_coverage" and result["paired_coverage"] == 0
    assert result["ade"] is None and result["fde"] is None


def test_missing_final_frame_does_not_use_last_visible_endpoint():
    valid = np.ones(20, bool)
    pred_valid = valid.copy()
    pred_valid[-1] = False
    result = trajectory_metrics(track(), track(), pred_valid, valid, fps=10)
    assert result["status"] == "scored" and result["fde"] is None


@pytest.mark.parametrize("change", ["nan", "range", "fps", "mask"])
def test_invalid_observations_rejected(change):
    p = track()
    valid = np.ones(20, bool)
    fps = 10
    if change == "nan":
        p[3] = np.nan
    elif change == "range":
        p[3] = 2
    elif change == "fps":
        fps = 0
    else:
        valid = valid.astype(int)
    with pytest.raises(ValueError):
        trajectory_metrics(p, track(), valid, valid, fps=fps)


def test_event_failure_and_all_negative_are_explicit():
    assert event_metrics({"lift": True}, {"lift": False})["false_positive"] == 1
    result = event_metrics({"lift": False}, {"lift": False})
    assert result["agreement"] == 1 and result["f1"] is None and result["all_negative"]
    with pytest.raises(ValueError):
        event_metrics({"lift": 1}, {"lift": True})


def test_bootstrap_uses_episodes_not_views_or_frames():
    result = episode_bootstrap([("a", 0), ("a", 2), ("b", 3)])
    assert result["episodes"] == 2 and result["mean"] == 2
    assert result == episode_bootstrap([("a", 0), ("a", 2), ("b", 3)])
    assert episode_bootstrap([("one", 2)])["ci95"] is None
    assert episode_bootstrap([])["mean"] is None
