"""Local streaming video diagnostics, explicitly not task-success evaluation."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DiagnosticThresholds:
    near_static_mae: float = 0.5
    jump_mae: float = 35.0
    static_fraction: float = 0.95
    noise_ratio: float = 4.0

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Diagnostic thresholds must be finite nonnegative numbers")
        if not 0 <= self.static_fraction <= 1 or self.noise_ratio < 1:
            raise ValueError("static_fraction must be in [0,1]; noise_ratio must be >= 1")


def diagnose_frames(frames, *, fps, thresholds=None):
    """Keep two frames and scalar accumulators only; all intensities are 0..255.

    Duplicate counts use sampled decoded pixels (codecs may alter equality).
    High-frequency growth is a review hint, not a flower-screen classifier.
    """
    thresholds = thresholds or DiagnosticThresholds()
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    count = duplicates = still = jumps = run = longest = 0
    total_mae = max_mae = noise_sum = max_noise = 0.0
    previous = shape = initial_noise = None
    for frame in frames:
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) < 2:
            raise ValueError("Diagnostics require uint8 RGB [H,W,3], H/W >= 2")
        if shape is not None and array.shape != shape:
            raise ValueError("Video frame dimensions changed")
        shape = array.shape
        # Spatial metrics use a deterministic bounded-resolution sample.
        sample = array[::max(1, math.ceil(shape[0] / 256)), ::max(1, math.ceil(shape[1] / 256))].astype(np.float32)
        gray = sample.mean(axis=-1)
        noise = float((np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2)
        initial_noise = noise if initial_noise is None else initial_noise
        noise_sum += noise
        max_noise = max(max_noise, noise)
        if previous is not None:
            mae = float(np.abs(sample - previous).mean())
            total_mae += mae
            max_mae = max(max_mae, mae)
            duplicates += int(np.array_equal(sample, previous))
            near_static = mae <= thresholds.near_static_mae
            still += int(near_static)
            jumps += int(mae >= thresholds.jump_mae)
            run = run + 1 if near_static else 0
            longest = max(longest, run)
        previous = sample
        count += 1
    if not count:
        raise ValueError("Video contains no decoded frames")
    transitions = count - 1
    fraction = still / transitions if transitions else None
    growth = max_noise / max(initial_noise, 1.0)
    warnings = []
    if fraction is not None and fraction >= thresholds.static_fraction:
        warnings.append("near_static_video")
    if jumps:
        warnings.append("large_frame_jump")
    if growth >= thresholds.noise_ratio and max_noise >= 20:
        warnings.append("high_frequency_growth_review_required")
    return {
        "frames": count, "fps": float(fps), "duration_seconds": count / fps,
        "sampled_shape": list(previous.shape), "transitions": transitions,
        "mean_transition_mae": total_mae / transitions if transitions else None,
        "max_transition_mae": max_mae if transitions else None,
        "sampled_duplicate_transition_fraction": duplicates / transitions if transitions else None,
        "near_static_transition_fraction": fraction,
        "longest_near_static_seconds": longest / fps,
        "large_jump_transitions": jumps, "mean_spatial_gradient": noise_sum / count,
        "max_spatial_gradient": max_noise, "spatial_gradient_growth": growth,
        "warnings": warnings,
    }


def diagnose_video(path, *, thresholds=None):
    import cv2
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    def frames():
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        result = diagnose_frames(frames(), fps=fps, thresholds=thresholds)
        if expected > 0 and result["frames"] != expected:
            raise ValueError(f"Incomplete video decode: {result['frames']} of {expected} frames")
        return result
    finally:
        capture.release()


def diagnose_artifact(directory, *, thresholds=None):
    """Read-only inspection of an artifact-v2 job; never imports a model."""
    directory = Path(directory).resolve()
    metadata_path = directory / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata = json.loads(metadata_bytes)
    if metadata.get("format_version") != 2:
        raise ValueError("Video diagnostics require artifact format_version=2")
    thresholds = thresholds or DiagnosticThresholds()
    files = metadata.get("view_files") or {"primary": metadata["video_file"]}
    results = {}
    for name, relative in files.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Video path leaves artifact directory")
        results[name] = {"file": relative, **diagnose_video(path, thresholds=thresholds)}
    return {
        "schema_version": 1, "kind": "video_diagnostics", "artifact": str(directory),
        "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "thresholds": asdict(thresholds), "views": results,
        "task_success_evaluated": False, "physical_validity_evaluated": False,
        "limitations": "Heuristic sampled-pixel diagnostics; warnings are not model failure or task scores.",
    }
