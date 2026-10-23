"""Coverage-aware 2-D trajectory/event metrics, not hidden physical-state truth.

Independent implementations of standard distances; exact DTW is deliberately
not labelled as EWMBench's FastDTW/best-of-three/empirically normalized score.
Missing observations are never silently interpolated or converted to zeros.
"""
from __future__ import annotations

import numpy as np


def trajectory_metrics(predicted, reference, predicted_valid, reference_valid, *, fps, minimum_coverage=.8):
    p, r = np.asarray(predicted, dtype=float), np.asarray(reference, dtype=float)
    pv, rv = np.asarray(predicted_valid), np.asarray(reference_valid)
    if (p.ndim != 2 or p.shape[1] != 2 or p.shape != r.shape or len(p) < 2
            or pv.dtype != bool or rv.dtype != bool or pv.shape != (len(p),) or rv.shape != pv.shape
            or isinstance(fps, (bool, np.bool_)) or isinstance(minimum_coverage, (bool, np.bool_))
            or not np.isfinite(fps) or fps <= 0 or not 0 < minimum_coverage <= 1):
        raise ValueError("Tracks require aligned [T,2], boolean visibility masks and positive FPS")
    for values, valid in ((p, pv), (r, rv)):
        if not np.isfinite(values[valid]).all() or np.any((values[valid] < 0) | (values[valid] > 1)):
            raise ValueError("Visible track coordinates must be finite normalized image XY")
    # Never reward the initial conditioning frame.
    mask = pv & rv
    mask[0] = False
    indices = np.flatnonzero(mask)
    coverage = len(indices) / (len(p)-1)
    result = {"status": "scored" if coverage >= minimum_coverage and len(indices) >= 2 else "insufficient_coverage",
              "total_future_frames": len(p)-1, "paired_frames": len(indices), "paired_coverage": coverage,
              "prediction_coverage": float(pv[1:].mean()), "reference_coverage": float(rv[1:].mean()),
              "minimum_coverage": minimum_coverage, "initial_frame_scored": False,
              "coordinate_space": "normalized_image_xy_not_metric_3d", "interpolated_frames": 0,
              "ade": None, "fde": None, "symmetric_hausdorff": None,
              "dtw_mean_path_distance": None, "speed_wasserstein": None,
              "frozen_initial_ade_on_paired_frames": None, "relative_ade_improvement_over_frozen": None}
    if result["status"] != "scored":
        return result
    a, b = p[indices], r[indices]
    if len(a) > 2000:
        raise ValueError("Exact DTW is bounded at 2000 paired frames; explicitly define shorter windows")
    result["ade"] = float(np.linalg.norm(a-b, axis=1).mean())
    if rv[0]:
        baseline = float(np.linalg.norm(r[0]-b, axis=1).mean())
        result["frozen_initial_ade_on_paired_frames"] = baseline
        result["relative_ade_improvement_over_frozen"] = 1-result["ade"]/baseline if baseline else None
    # A missing final detection is not replaced by the last observed point.
    result["fde"] = float(np.linalg.norm(p[-1]-r[-1])) if mask[-1] else None
    distances = np.linalg.norm(a[:, None]-b[None], axis=-1)
    result["symmetric_hausdorff"] = float(max(distances.min(0).max(), distances.min(1).max()))
    previous = np.full(len(b)+1, np.inf)
    previous[0] = 0
    lengths = np.zeros(len(b)+1, dtype=int)
    for row in distances:
        current = np.full(len(b)+1, np.inf)
        path_lengths = np.zeros(len(b)+1, dtype=int)
        for j, cost in enumerate(row, 1):
            options = (previous[j-1], previous[j], current[j-1])
            k = int(np.argmin(options))
            length = (lengths[j-1], lengths[j], path_lengths[j-1])[k]
            current[j], path_lengths[j] = cost+options[k], length+1
        previous, lengths = current, path_lengths
    result["dtw_mean_path_distance"] = float(previous[-1]/lengths[-1])
    adjacent = mask[1:] & mask[:-1]
    # Frame0 is excluded from errors, but is valid context for first velocity.
    adjacent[0] = pv[0] & rv[0] & pv[1] & rv[1]
    if adjacent.any():
        ps = np.linalg.norm(np.diff(p, axis=0)[adjacent], axis=1)*fps
        rs = np.linalg.norm(np.diff(r, axis=0)[adjacent], axis=1)*fps
        result["speed_wasserstein"] = float(np.abs(np.sort(ps)-np.sort(rs)).mean())
    return result


def event_metrics(predicted, reference):
    """Matched observable events, not force/contact ground truth inferred from RGB."""
    if (not isinstance(predicted, dict) or not isinstance(reference, dict)
            or set(predicted) != set(reference) or not predicted
            or any(type(v) is not bool for v in [*predicted.values(), *reference.values()])):
        raise ValueError("Events require matching nonempty names and explicit boolean observations")
    tp = sum(predicted[k] and reference[k] for k in reference)
    fp = sum(predicted[k] and not reference[k] for k in reference)
    fn = sum(not predicted[k] and reference[k] for k in reference)
    return {"events": len(reference), "agreement": sum(predicted[k] == reference[k] for k in reference)/len(reference),
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "all_negative": not any(reference.values()), "physical_contact_force_measured": False}


def episode_bootstrap(records, *, seed=0, repetitions=2000):
    """Views/windows/seeds are first averaged within episode, never resampled as independent frames."""
    if type(repetitions) is not int or not 100 <= repetitions <= 100000:
        raise ValueError("Bootstrap repetitions must be an integer in [100,100000]")
    groups = {}
    for episode, value in records:
        if not isinstance(episode, str) or not episode or not np.isfinite(value):
            raise ValueError("Bootstrap requires explicit source-episode keys and finite values")
        groups.setdefault(episode, []).append(value)
    values = np.asarray([np.mean(v) for v in groups.values()])
    if not len(values):
        return {"episodes": 0, "mean": None, "ci95": None}
    result = {"episodes": len(values), "mean": float(values.mean()), "ci95": None,
              "resampling_unit": "source_episode", "seed": seed, "repetitions": repetitions}
    if len(values) >= 2:
        rng = np.random.default_rng(seed)
        samples = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(repetitions)]
        result["ci95"] = np.quantile(samples, [.025, .975]).tolist()
    return result
