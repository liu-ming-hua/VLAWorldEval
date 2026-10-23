"""Reference-bound pixel fidelity, deliberately not physical correctness scores.

The replay executor calls this on raw RGB, before lossy video encoding. References
never enter the model worker. Metrics exclude the first/conditioning frame.
"""
from __future__ import annotations

import numpy as np

from evaluation.manifests import VideoSpec


def validate_reference(reference, config, packet, initial, *, input_sha256):
    if (not isinstance(reference, dict)
            or set(reference) != {"schema_version", "input_sha256", "views", "frame_indices", "source"}
            or type(reference["schema_version"]) is not int or reference["schema_version"] != 1
            or reference["input_sha256"] != input_sha256):
        raise ValueError("Reference schema or bound replay input hash differs")
    source = reference["source"]
    if (not isinstance(source, dict) or not all(isinstance(source.get(k), str) and source[k]
            for k in ("dataset_domain", "split", "episode", "conversion", "condition_kind", "frame_axis"))
            or source.get("ground_truth_kind") != "recorded_execution"):
        raise ValueError("Reference requires recorded-execution provenance, domain, split and condition kind")
    video = VideoSpec.from_dict(config["video"])
    count = video.frame_count(len(packet["actions"]))
    if (not isinstance(reference["views"], dict) or set(reference["views"]) != set(initial)
            or not isinstance(reference["frame_indices"], dict) or set(reference["frame_indices"]) != set(initial)):
        raise ValueError("Reference camera identity differs from the replay")
    for name, first in initial.items():
        value = np.asarray(reference["views"][name])
        indices = np.asarray(reference["frame_indices"][name])
        if value.dtype != np.uint8 or value.shape != (count, *first.shape[1:]):
            raise ValueError(f"Reference {name} shape/dtype differs from the exact prediction timeline")
        if (indices.dtype.kind not in "iu" or indices.shape != (count,) or indices[0] < 0
                or np.any(np.diff(indices) <= 0)):
            raise ValueError("Reference source frame indices must be strictly increasing integers")
        if not np.array_equal(value[0], first[0]):
            raise ValueError("Reference frame zero must equal the recorded initial observation")
    return reference


def _ssim(x, y):
    """Wang-style Gaussian 11x11, sigma1.5, population covariance, RGB mean."""
    import cv2
    if min(x.shape[:2]) < 11:
        raise ValueError("SSIM requires image sides >=11")
    x, y = x.astype(np.float64) / 255, y.astype(np.float64) / 255
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5)  # noqa: E731
    ux, uy = blur(x), blur(y)
    vx, vy, cov = blur(x*x) - ux*ux, blur(y*y) - uy*uy, blur(x*y) - ux*uy
    value = ((2*ux*uy + .01**2) * (2*cov + .03**2)
             / ((ux*ux + uy*uy + .01**2) * (vx + vy + .03**2)))
    return float(value[5:-5, 5:-5].mean())


def score_view(prediction, reference):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if (prediction.dtype != np.uint8 or reference.dtype != np.uint8 or prediction.shape != reference.shape
            or prediction.ndim != 4 or prediction.shape[-1] != 3 or len(prediction) < 2):
        raise ValueError("Metrics require matching uint8 [T,H,W,3], including initial and future frames")
    p, r = prediction[1:].astype(np.float64)/255, reference[1:].astype(np.float64)/255
    per_frame = ((p-r)**2).mean(axis=(1, 2, 3))
    baseline = ((reference[0].astype(np.float64)/255 - r)**2).mean(axis=(1, 2, 3))
    psnr = [None if error == 0 else float(-10*np.log10(error)) for error in per_frame]
    mean_mse, base_mse = float(per_frame.mean()), float(baseline.mean())
    ssim = [_ssim(x, y) for x, y in zip(prediction[1:], reference[1:])]
    pred_diff = np.diff(prediction.astype(np.float32)/255, axis=0)
    ref_diff = np.diff(reference.astype(np.float32)/255, axis=0)
    return {
        "frames_scored": len(p), "initial_frame_scored": False, "mse_01": mean_mse,
        "psnr_db_from_mean_mse": None if mean_mse == 0 else float(-10*np.log10(mean_mse)),
        "mean_frame_psnr_db": None if any(v is None for v in psnr) else float(np.mean(psnr)),
        "perfect_frame_count": int((per_frame == 0).sum()),
        "ssim_rgb_gaussian11_sigma1p5_population": float(np.mean(ssim)),
        "frozen_initial_baseline_mse_01": base_mse,
        "relative_mse_improvement_over_frozen": None if base_mse == 0 else 1-mean_mse/base_mse,
        "temporal_rgb_delta_mae_01": float(np.abs(pred_diff-ref_diff).mean()),
        "per_frame": {"mse_01": per_frame.tolist(), "psnr_db": psnr, "ssim": ssim},
        "lpips": {"status": "not_computed", "reason": "Requires separately declared learned evaluator assets"},
        "physical_action_tracking": {"status": "not_measured", "reason": "Pixel fidelity is not robot trajectory or contact truth"},
        "fid_fvd": {"status": "not_computed", "reason": "Small paired samples are not a valid distributional benchmark"},
    }
