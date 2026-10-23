"""Domain-specific visible PushT markers, not a universal gripper detector.

Fixed HSV thresholds are applied independently to prediction and recorded RGB.
Missing or ambiguous components remain unobserved; never use future kinematics
to position a predicted marker. Calibration/semantic review is reported, not assumed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def markers(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Marker observer requires RGB uint8 images")
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    masks = {"object": ((h >= 145) & (h <= 175) & (s >= 55) & (v >= 50)),
             "pusher": (((h <= 28) | (h >= 177)) & (s >= 135) & (v >= 60))}
    result = {}
    for name, mask in masks.items():
        _, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        candidates = [(int(st[cv2.CC_STAT_AREA]), center) for st, center in zip(stats[1:], centroids[1:])
                      if 5 <= st[cv2.CC_STAT_AREA] <= image.shape[0]*image.shape[1]*.2]
        bins = {"object": candidates} if name == "object" else {
            "left_pusher": [x for x in candidates if x[1][0] < image.shape[1]/2],
            "right_pusher": [x for x in candidates if x[1][0] >= image.shape[1]/2]}
        for label, values in bins.items():
            values.sort(key=lambda pair: pair[0], reverse=True)
            valid = bool(values) and (len(values) == 1 or values[0][0] >= 2*values[1][0])
            result[label] = (values[0][1]/[image.shape[1], image.shape[0]]).tolist() if valid else None
    return result


def observe_replay(summary_path, reference_path, output):
    from evaluation.reporting.motion_detector import validate_annotation, score_annotations
    from evaluation.reporting.task_scores import _digest
    from evaluation.execution.transport import read_packet
    from evaluation.datasets.native_reference import indexed_video
    from vlaworldeval.atomic_json import publish_json_exclusive
    from vlaworldeval.paths import project_root
    import mediapy as media
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(project_root()):
        raise ValueError("Observer output must be a new project-local directory")
    reference = read_packet(Path(reference_path))
    if reference["source"]["dataset_domain"] != "aloha_pusht" or set(reference["views"]) != {"camera_1_color"}:
        raise ValueError("Color-marker observer is only declared for native ALOHA PushT camera1")
    value = {"schema_version": 1, "summary_sha256": _digest(Path(summary_path)),
        "reference_sha256": _digest(Path(reference_path)), "views": {"camera_1_color": {}},
        "evaluator": {"identity": "native ALOHA PushT color markers v1",
            "source_sha256": _digest(Path(__file__)), "human_review_status": "pending",
            "semantic_scope": "stationary camera1, pink T object, orange opposing pushers; not universal left/right robot identity",
            "identity_rule": "pushers occupy opposing image halves; crossing invalidates this protocol",
            "coordinate_space": "normalized_image_xy", "threshold_selection": "fixed before full-split motion scoring",
            "rgb_range": "uint8", "pink_hsv": [[145, 175], [55, 255], [50, 255]],
            "orange_hsv": "H<=28 or H>=177; S>=135,V>=60; component area5..20%image; max/second>=2"}}
    validate_annotation(value, Path(summary_path), Path(reference_path))
    gt = reference["views"]["camera_1_color"]
    pred = indexed_video(Path(summary_path).parent/"camera_1_color.mp4", range(len(gt)))
    observations = {side: [markers(frame) for frame in frames] for side, frames in (("prediction", pred), ("reference", gt))}
    tracks = {}
    for name in ("object", "left_pusher", "right_pusher"):
        tracks[name] = {side: [r[name] if r[name] is not None else [0., 0.] for r in observations[side]]
                        for side in observations}
        tracks[name].update({side+"_valid": [r[name] is not None for r in observations[side]] for side in observations})
    value["views"]["camera_1_color"] = tracks
    obj = tracks["object"]
    if all(obj[side+"_valid"][0] and obj[side+"_valid"][-1] for side in ("prediction", "reference")):
        # Observable displacement only: this is not grasp/task success or force.
        event = "object_endpoint_displacement_above_0p05_image_xy"
        value["events"] = {side: {event: bool(np.linalg.norm(np.asarray(obj[side][-1])-obj[side][0]) >= .05)}
                           for side in ("prediction", "reference")}
        value["evaluator"]["event_rubric"] = "Visible object centroid endpoint displacement>=0.05 normalized image XY; not rotation/contact/task success"
    value["evaluator"]["initial_frame_sha256"] = hashlib.sha256(gt[0].tobytes()).hexdigest()
    output.mkdir(parents=True)
    annotation = output/"tracks.json"
    publish_json_exclusive(annotation, value)
    fps = json.loads(Path(summary_path).parent.joinpath("config.json").read_text())["video"]["fps"]
    frames = []
    colors = {"object": (0, 255, 0), "left_pusher": (255, 80, 0), "right_pusher": (0, 80, 255)}
    for i in range(len(gt)):
        pair = []
        for side, frame in (("prediction", pred[i]), ("reference", gt[i])):
            frame = frame.copy()
            for name, point in observations[side][i].items():
                if point is not None:
                    xy = tuple((np.asarray(point)*[frame.shape[1], frame.shape[0]]).round().astype(int))
                    cv2.circle(frame, xy, 2, colors[name], 1)
            pair.append(frame)
        frames.append(np.concatenate(pair, axis=1))
    media.write_video(output/"tracking_overlay.mp4", np.stack(frames), fps=fps)
    return score_annotations(summary_path, reference_path, annotation, output/"motion.json")
