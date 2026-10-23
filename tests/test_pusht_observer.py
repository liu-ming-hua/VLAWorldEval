"""Fixed native markers must not hallucinate coordinates on empty images."""
import cv2
import numpy as np

from evaluation.reporting.pusht_observer import markers


def test_empty_image_returns_missing_not_origin():
    assert all(point is None for point in markers(np.zeros((128, 128, 3), np.uint8)).values())


def test_independent_color_markers_and_spatial_identity():
    hsv = np.zeros((128, 128, 3), np.uint8)
    hsv[50:65, 50:65] = [160, 120, 150]
    hsv[60:70, 20:30] = [15, 200, 150]
    hsv[45:55, 95:105] = [15, 200, 150]
    observed = markers(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    assert observed["object"] is not None
    assert observed["left_pusher"][0] < .5 < observed["right_pusher"][0]


def test_ambiguous_duplicate_object_is_not_silently_selected():
    hsv = np.zeros((128, 128, 3), np.uint8)
    hsv[20:30, 20:30] = [160, 120, 150]
    hsv[70:80, 70:80] = [160, 120, 150]
    assert markers(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))["object"] is None
