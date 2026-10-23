"""Pinned native-domain evaluation subsets, direct mirror transport only.

This does not download models or silently replace inaccessible evaluation data.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from vlaworldeval.assets import retryable_transport_error  # noqa: E402

DATASETS = {
    "irasim_native_test_raw": {
        "repo": "fangqi/IRASim", "revision": "dfcbf85c27b2c5d041dbf5df3df272111e758b48",
        "destination": "datasets/irasim/native_raw_v1",
        "patterns": ["languagetable_train_data.tar.gz.sfpart_*"],
    },
    "cosmos20k_holdout": {
        "repo": "nvidia/Cosmos3-DROID", "revision": "5c11a20accb11497270a5247a7f1e66ad04c956c",
        "destination": "datasets/cosmos3_droid", "patterns": ["success/meta/*", "failure/meta/*"],
    },
    "irasim_annotations_prefix": {
        "repo": "fangqi/IRASim", "revision": "dfcbf85c27b2c5d041dbf5df3df272111e758b48",
        "destination": "datasets/irasim/annotation_transport_v1",
        "patterns": ["languagetable_train_data.tar.gz.sfpart_aa"],
    },
    "dreamdojo_eval100": {
        "repo": "nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1",
        "revision": "9f68429ad12535d30d1aae33b7945c79c830c8c4",
        "destination": "datasets/dreamdojo/official_eval100_v1", "patterns": ["In-lab_Eval/*/meta/*"],
    },
    "dreamdojo_eval_metadata": {
        "repo": "nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1",
        "revision": "9f68429ad12535d30d1aae33b7945c79c830c8c4",
        "destination": "datasets/dreamdojo/eval_metadata_v1", "patterns": ["In-lab_Eval/*/meta/*"],
    },
    "cotracker3": {
        "repo": "facebook/cotracker3", "repo_type": "model",
        "revision": "bf55ea50d4390e1820a267f131cd6587240fb2c5",
        "destination": "checkpoints/evaluators/cotracker3", "patterns": ["scaled_offline.pth", "README.md"],
    },
    "grounding_dino": {
        "repo": "IDEA-Research/grounding-dino-tiny", "repo_type": "model",
        "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "destination": "checkpoints/evaluators/grounding_dino_tiny",
        "patterns": ["*.json", "vocab.txt", "model.safetensors", "README.md"],
    },
    "irasim_full_languagetable": {
        "repo": "fangqi/IRASim", "revision": "dfcbf85c27b2c5d041dbf5df3df272111e758b48",
        "destination": "datasets/irasim/full_native_v1",
        "patterns": ["languagetable_train_data.tar.gz.sfpart_*", "languagetable_evaluation_data.tar.gz.sfpart_*"],
    },
    "iws_full_pusht": {
        "repo": "yixuan1999/interactive-world-sim-data",
        "revision": "6c3010b4d8c0fae8b683856ca1c550869d5bdb04",
        "destination": "datasets/iws/full_native_v1",
        "patterns": ["pusht/val/*.hdf5"],
    },
    "dreamdojo_full_inlab": {
        "repo": "nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1",
        "revision": "9f68429ad12535d30d1aae33b7945c79c830c8c4",
        "destination": "datasets/dreamdojo/full_native_v1",
        "patterns": ["In-lab_Eval/*"],
    },
    "ewmbench_detector": {
        "repo": "agibot-world/EWMBench-model", "repo_type": "model",
        "revision": "85a49f8118ec656c9d511c5236582ac7fde16fbc",
        "destination": "checkpoints/evaluators/ewmbench_gripper_v01",
        "patterns": ["yoloworld-EWMBench-v0.1.pt", "README.md"],
    },
    "dreamdojo": {
        "repo": "nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1",
        "revision": "9f68429ad12535d30d1aae33b7945c79c830c8c4",
        "destination": "datasets/dreamdojo/evaluation_v1",
        "patterns": [f"In-lab_Eval/gr1_unified.{task}_robot/{suffix}"
                     for task in ("color_puzzles", "fold_cloth_long", "mug")
                     for suffix in ("meta/*.json", "meta/tasks.jsonl", "data/chunk-000/episode_000000.parquet",
                                    "videos/chunk-000/observation.images.ego_view_freq20/episode_000000.mp4")],
    },
    "iws": {
        "repo": "yixuan1999/interactive-world-sim-min-data",
        "revision": "987c175bcc29ddd0f6e6e3096b2c20df300253b0",
        "destination": "datasets/iws/evaluation_v1",
        "patterns": [f"pusht/val/episode_{i}.hdf5" for i in range(3)],
    },
    "playworld": {
        "repo": "tennyyyin/playworld_dataset_preview",
        "revision": "bc4ea8f9bf88b8266c2aadf66e608b78aa035b17",
        "destination": "datasets/playworld/evaluation_v1",
        "patterns": ["v0_2025_12_28_2000/annotation/val/9.json",
                     "v0_2025_12_28_2000/videos/val/9/*.mp4",
                     "v0_2025_12_28_2000/metainfo/*"],
    },
}


def download_with_retry(operation, entry, persist, *, attempts=3, delay=30):
    """Bounded transport retries; access/integrity failures are not repair permission."""
    if type(attempts) is not int or not 1 <= attempts <= 10 or type(delay) not in (int, float) or not math.isfinite(delay) or not 0 <= delay <= 60:
        raise ValueError("Use1..10 attempts with finite0..60-second retry delay")
    entry.update(status="downloading", attempts=[], substitute_used=False)
    try:
        for index in range(attempts):
            attempt = {"attempt": index+1, "status": "downloading"}
            entry["attempts"].append(attempt)
            persist()
            try:
                result = operation()
            except Exception as exc:
                retryable = retryable_transport_error(exc)
                attempt.update(status="failed", error_type=type(exc).__name__, message=str(exc), transport_retryable=retryable)
                entry["status"] = "retrying_transport" if retryable and index+1 < attempts else "transport_failed" if retryable else "unavailable"
                persist()
                if entry["status"] != "retrying_transport":
                    return None
                time.sleep(delay)
                entry["status"] = "downloading"
            else:
                attempt["status"] = "verified"
                entry.update(result, status="verified")
                persist()
                return result
    except BaseException as exc:
        entry.update(status="failed" if isinstance(exc, Exception) else "interrupted", interruption_type=type(exc).__name__)
        persist()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*DATASETS, "all"], default="all",
        help="all preserves the small dreamdojo/iws/playworld pilot; full archives require explicit model selection")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="frozen Cosmos20k heldout audit; default is the current capability-study audit")
    parser.add_argument("--attempts", type=int, default=3, help="1..10 transport attempts; authorization/integrity errors are not retried")
    parser.add_argument("--retry-delay", type=float, default=30, help="bounded0..60-second delay between resumable transport attempts")
    args = parser.parse_args(argv)
    if not 1 <= args.attempts <= 10 or not math.isfinite(args.retry_delay) or not 0 <= args.retry_delay <= 60:
        raise ValueError("Use1..10 attempts with finite0..60-second retry delay")
    from vlaworldeval.workspace import direct_network_environment
    from vlaworldeval.assets import download_snapshot
    from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT) or args.output.exists():
        raise ValueError("Choose a new report inside this project")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = direct_network_environment(ROOT)
    for name in list(os.environ):
        if name.lower().endswith("_proxy"):
            os.environ.pop(name)
    os.environ.update(environment, HF_ENDPOINT="https://hf-mirror.com")
    report = {"schema_version": 1, "endpoint": "https://hf-mirror.com", "proxies_disabled": True,
              "scope": "pinned_declared_patterns_not_an_automatic_full_test_set_claim", "datasets": {}}
    publish_json_exclusive(args.output, report)
    # Never let adding a full400GB release silently expand the legacy default.
    selection = ["dreamdojo", "iws", "playworld"] if args.model == "all" else [args.model]
    for name in selection:
        spec = dict(DATASETS[name])
        try:
            if name == "dreamdojo_eval100":
                from evaluation.planning.checkpoint import dreamdojo_eval100_selection
                selected = dreamdojo_eval100_selection(ROOT/"datasets/dreamdojo/eval_metadata_v1")
                spec["patterns"] = [*spec["patterns"], *sorted({case[k] for case in selected["samples"] for k in ("parquet", "video")})]
            elif name == "cosmos20k_holdout":
                import json
                selection_path = (args.selection or ROOT/"results/checkpoint_capability_v1/cosmos20k_holdout_audit.json").resolve()
                if not selection_path.is_relative_to(ROOT):
                    raise ValueError("Selection report must be project-local")
                audit = json.loads(selection_path.read_text())
                if audit["expected_episodes"] != 2157 or len(audit["episodes"]) != 2157:
                    raise ValueError("Reconstruct the complete documented20k-EMA holdout before downloading")
                from vlaworldeval.assets import file_record
                if any(file_record(Path(p))["sha256"] != sha for p, sha in audit["metadata_sha256"].items()):
                    raise ValueError("Holdout metadata changed after selection")
                spec["patterns"] = [*spec["patterns"], *sorted({p for case in audit["episodes"] for p in case["payload_paths"]})]
            report["datasets"][name] = {}
            download_with_retry(lambda: download_snapshot(spec["repo"], ROOT / spec["destination"], root=ROOT,
                revision=spec["revision"], patterns=spec["patterns"], repo_type=spec.get("repo_type", "dataset")),
                report["datasets"][name], lambda: replace_owned_json(args.output, report),
                attempts=args.attempts, delay=args.retry_delay)
        except Exception as exc:
            report["datasets"][name] = {"status": "precondition_failed", "error_type": type(exc).__name__,
                                        "message": str(exc), "substitute_used": False}
        replace_owned_json(args.output, report)
        print(name, report["datasets"][name]["status"], flush=True)
    return 0 if all(v["status"] == "verified" for v in report["datasets"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
