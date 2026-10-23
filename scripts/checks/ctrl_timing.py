#!/usr/bin/env python3
"""Corrected Ctrl validation; model calls use the shared native executor."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "run", "compare", "reference", "audit", "render"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--attempt", help="Explicit new retry name; retain original interrupted attempt")
    parser.add_argument("--mode", default="autoregressive", choices=["autoregressive", "ground_truth_visual_context"])
    args = parser.parse_args()
    from evaluation.planning.registry import project_path
    directory = project_path(ROOT, str(args.directory))
    if args.operation == "prepare":
        from evaluation.planning.ctrl_validation import prepare_ctrl_validation
        study = prepare_ctrl_validation(ROOT, directory)
        result = {"status": "prepared", "cases": len(study["cases"]), "planned_world_model_calls": study["planned_world_model_calls"]}
    elif args.operation in {"audit", "render"}:
        from evaluation.reporting.ctrl_validation import audit_ctrl_validation, render_ctrl_validation
        result = audit_ctrl_validation(ROOT, directory) if args.operation == "audit" else render_ctrl_validation(ROOT, directory, cases=(args.case,) if args.case else ())
    elif args.operation == "reference":
        from evaluation.execution.native_replay import admit_native_replay
        from evaluation.execution.client import ModelWorker
        from evaluation.execution.reliability import clean_worker_exit
        study = json.loads((directory / "study.json").read_text())
        case = next(c for c in study["cases"] if c["id"] == "ctrl_ep795_smoke")
        output = directory / "official_reference"
        config, spec, _, packet, _, kwargs, options = admit_native_replay(Path(case["config"]), Path(case["input"]), suite_dir=output)
        output.mkdir(exist_ok=False)
        worker = ModelWorker(spec.python, output / "worker", module=config["module"], adapter=config["adapter"],
            kwargs=kwargs, gpus=config["gpus"], memory_gib=config["memory_gib"], options=options)
        try:
            worker.call("seed", seed=config["seed"])
            worker.call(config["reset_operation"], **packet["reset_args"])
            result = worker.call("verify_ctrl_reference", action=packet["actions"][0], first_action=packet["reset_args"]["first_action"])
        finally:
            cleanup = worker.close()
        result["cleanup"] = cleanup
        if not clean_worker_exit(cleanup):
            result["status"] = "failed"
        (output / "report.json").write_text(json.dumps(result, indent=2))
    elif args.operation == "compare":
        from evaluation.reporting.image_feedback import compare_image_feedback
        result = compare_image_feedback(ROOT, directory, directory / "paired_comparison", cases=("ctrl_ep795",))
    else:
        from evaluation.execution.native_replay import run_native_replay
        from evaluation.execution.reliability import run_signal_handlers
        from evaluation.reporting.task_scores import _digest
        study = json.loads((directory / "study.json").read_text())
        case = next(c for c in study["cases"] if c["id"] == args.case)
        if args.mode not in case["modes"]:
            raise ValueError("Unplanned experiment mode")
        for key in ("input", "config", "anchors"):
            if key in case and _digest(Path(case[key])) != case[key + "_sha256"]:
                raise ValueError("Frozen input/config changed")
        output = directory / case["id"] / args.mode
        if args.attempt:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.attempt):
                raise ValueError("Unsafe attempt name")
            original = output / "summary.json"
            if not original.exists() or json.loads(original.read_text())["status"] != "failed":
                raise ValueError("Retry requires an explicitly failed original run")
            output = directory / case["id"] / "attempts" / args.attempt / args.mode
        with run_signal_handlers():
            result = run_native_replay(Path(case["config"]), Path(case["input"]), output,
                image_anchors=Path(case["anchors"]) if args.mode == "ground_truth_visual_context" else None)
        if args.attempt and result["status"] == "completed":
            from vlaworldeval.atomic_json import publish_json_exclusive
            publish_json_exclusive(directory / case["id"] / ("selected_" + args.mode + ".json"),
                {"relative_path": str(output.relative_to(directory / case["id"])),
                 "report_sha256": _digest(output / "summary.json"), "prior_report_sha256": _digest(original),
                 "reason": "Explicit clean rerun after interrupted original; no failed evidence overwritten"})
        result = {key: result[key] for key in ("status", "world_model_calls", "vla_calls")}
    display = {key: result[key] for key in ("status", "world_model_calls", "vla_calls", "planned_world_model_calls", "partial_selection") if key in result}
    if "cases" in result:
        display["cases"] = len(result["cases"]) if isinstance(result["cases"], list) else result["cases"]
    print(json.dumps(display), flush=True)
    return 0 if result["status"] in {"prepared", "completed", "passed", "comparisons_built_not_quality_approved", "passed_engineering_not_quality_approval", "rendered_not_quality_approved"} else 1


if __name__ == "__main__":
    sys.exit(main())
