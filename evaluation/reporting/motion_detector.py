"""Optional offline detector-backed replay evaluation in the registered video environment.

Uses the released EWMBench detector, NOT its licensed evaluation implementation.
Identity is detector class, not left/right image position or unstable tracker IDs.
Detection coverage and unvalidated cross-embodiment transfer remain explicit.
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.reporting.motion import trajectory_metrics, event_metrics
from evaluation.reporting.task_scores import _digest
from evaluation.execution.reliability import clean_worker_exit
from vlaworldeval.atomic_json import publish_json_exclusive


def validate_annotation(value, summary_path, reference_path):
    summary = json.loads(summary_path.read_text())
    if (value.get("schema_version") != 1 or value.get("summary_sha256") != _digest(summary_path)
            or value.get("reference_sha256") != _digest(reference_path)
            or summary.get("status") != "completed" or not clean_worker_exit(summary.get("cleanup"))
            or summary["reference_evaluation"]["reference_sha256"] != _digest(reference_path)):
        raise ValueError("Motion annotations require the same clean replay and recorded reference")
    if not isinstance(value.get("evaluator"), dict) or not value["evaluator"].get("identity"):
        raise ValueError("Every observation requires evaluator provenance")
    if not isinstance(value.get("views"), dict) or not value["views"]:
        raise ValueError("Motion annotation views are required")
    for name, tracks in value["views"].items():
        if name not in summary["reference_evaluation"]["metrics"] or not isinstance(tracks, dict):
            raise ValueError("Annotation view does not match the replay")
        if _digest(summary_path.parent / f"{name}.mp4") != summary["files"][f"{name}.mp4"]:
            raise ValueError("Video changed since prediction")
        count = summary["reference_evaluation"]["metrics"][name]["frames_scored"]+1
        for track in tracks.values():
            if any(len(track[k]) != count for k in ("prediction", "reference", "prediction_valid", "reference_valid")):
                raise ValueError("Track frames must match the exact replay timeline")
    return summary


def score_annotations(summary_path, reference_path, annotations_path, output):
    summary_path, reference_path, annotations_path, output = map(Path, (summary_path, reference_path, annotations_path, output))
    if output.exists():
        raise FileExistsError(output)
    value = json.loads(annotations_path.read_text())
    validate_annotation(value, summary_path, reference_path)
    summary = json.loads(summary_path.read_text())
    from evaluation.reporting.native_reference import admitted_config
    from vlaworldeval.paths import project_root
    config = admitted_config(project_root(), summary_path, summary)
    tracks = {}
    for view, entities in value["views"].items():
        tracks[view] = {name: trajectory_metrics(track["prediction"], track["reference"],
            track["prediction_valid"], track["reference_valid"], fps=config["video"]["fps"])
            for name, track in entities.items()}
    events = value.get("events")
    result = {"schema_version": 1, "status": "completed", "summary_sha256": _digest(summary_path),
              "summary_path": str(summary_path.resolve()), "reference_path": str(reference_path.resolve()),
              "annotations_path": str(annotations_path.resolve()),
              "reference_sha256": _digest(reference_path), "annotations_sha256": _digest(annotations_path),
              "evaluator": value["evaluator"], "tracks": tracks,
              "unannotated_views": sorted(set(summary["reference_evaluation"]["metrics"])-set(tracks)),
              "events": None if events is None else event_metrics(events["prediction"], events["reference"]),
              "event_status": "not_annotated" if events is None else "observed_2d_events_not_contact_force",
              "physical_action_tracking_verified": False,
              "official_ewmbench_score": False,
              "metric_protocol": "normalized image XY; exact DTW mean path distance; no best-of-N, no missing-point interpolation"}
    from evaluation.reporting.checkpoint_metrics import metric_source_fingerprints
    result["implementation_sha256"] = metric_source_fingerprints("motion")
    publish_json_exclusive(output, result)
    return result


def detect_replay(summary_path, reference_path, output, *, device="cpu", confidence=.8):
    """Shared artifacts only. No WM/VLA invocation and no implicit weight download."""
    from vlaworldeval.paths import project_root
    from vlaworldeval.assets import verify_snapshot
    from evaluation.execution.transport import read_packet
    from evaluation.datasets.native_reference import indexed_video
    from ultralytics import YOLO
    import cv2
    root = project_root()
    output = Path(output).resolve()
    summary_path, reference_path = Path(summary_path), Path(reference_path)
    if output.exists() or not output.is_relative_to(root):
        raise ValueError("Motion extraction requires a new project-local output directory")
    if not 0 < confidence <= 1 or device not in {"cpu", "0", "1", "2", "3"}:
        raise ValueError("Explicit confidence and allowed evaluator device required")
    directory = root / "checkpoints/evaluators/ewmbench_gripper_v01"
    verified = verify_snapshot(directory)
    if (verified["repository"] != "agibot-world/EWMBench-model"
            or verified["revision"] != "85a49f8118ec656c9d511c5236582ac7fde16fbc"):
        raise ValueError("Requires the pinned official EWMBench gripper detector")
    reference = read_packet(reference_path)
    value = {"schema_version": 1, "summary_sha256": _digest(summary_path),
             "reference_sha256": _digest(reference_path), "views": {},
             "evaluator": {"identity": "EWMBench released YOLO detector, per-class highest confidence",
                 "snapshot": verified, "confidence": confidence, "resize_wh": [640, 480],
                 "native_training_domain": "AgiBot", "cross_domain_calibration": "not_validated",
                 "human_review_status": "pending", "device": device, "interpolation": False}}
    # Validate evidence before loading trusted checkpoint code or decoding frames.
    value["views"] = {name: {} for name in reference["views"]}
    validate_annotation(value, summary_path, reference_path)
    model = YOLO(str(directory / "yoloworld-EWMBench-v0.1.pt"))
    names = dict(model.names)
    value["evaluator"]["class_names"] = names
    output.mkdir(parents=True)
    for name, gt in reference["views"].items():
        pred = indexed_video(summary_path.parent / f"{name}.mp4", range(len(gt)))
        tracks = {str(label): {"prediction": [], "reference": [], "prediction_valid": [], "reference_valid": []}
                  for label in names.values()}
        for side, frames in (("prediction", pred), ("reference", gt)):
            for frame in frames:
                # Ultralytics ndarray input is BGR; transport/reference images are RGB.
                bgr = cv2.cvtColor(cv2.resize(frame, (640, 480)), cv2.COLOR_RGB2BGR)
                result = model.predict(bgr, device=device, conf=confidence, verbose=False)[0]
                selected = {}
                for box, cls, score in zip(result.boxes.xywh.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                           result.boxes.conf.cpu().numpy()):
                    cls = int(cls)
                    if cls not in selected or score > selected[cls][1]:
                        selected[cls] = (box[:2]/[640, 480], float(score))
                for cls, label in names.items():
                    item = selected.get(cls)
                    tracks[str(label)][side].append([0., 0.] if item is None else item[0].tolist())
                    tracks[str(label)][side+"_valid"].append(item is not None)
        value["views"][name] = tracks
    annotations = output / "tracks.json"
    publish_json_exclusive(annotations, value)
    return score_annotations(summary_path, reference_path, annotations, output / "motion.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--provider", choices=["ewmbench", "pusht_color"], default="ewmbench")
    args = parser.parse_args()
    import os
    from vlaworldeval.paths import project_root
    from vlaworldeval.cache import runtime_cache_environment
    os.environ.update(runtime_cache_environment(project_root()), HF_HUB_OFFLINE="1",
                      YOLO_CONFIG_DIR=str(project_root()/".cache/ultralytics"))
    if args.provider == "pusht_color":
        from evaluation.reporting.pusht_observer import observe_replay
        result = observe_replay(args.summary, args.reference, args.output)
    else:
        result = detect_replay(args.summary, args.reference, args.output, device=args.device)
    print(json.dumps(result))
