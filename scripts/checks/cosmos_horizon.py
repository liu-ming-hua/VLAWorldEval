#!/usr/bin/env python3
"""Source-time Cosmos qualification using shared native replay; no VLA calls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "run", "audit", "render", "reference-motion", "summarize", "temporal-review", "audit-review"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--case", help="Render one completed model/episode without inference")
    parser.add_argument("--output", type=Path, help="Fresh output directory for temporal-review")
    parser.add_argument("--resource-overrides", type=Path, help="Explicit environment-to-GiB JSON; allocator limits only")
    args = parser.parse_args()
    for value, operation, flag in ((args.output, "temporal-review", "--output"),
            (args.case, "render", "--case"), (args.resource_overrides, "run", "--resource-overrides")):
        if value is not None and args.operation != operation:
            parser.error(f"{flag} is only valid for {operation}")
    if args.operation == "temporal-review" and args.output is None:
        parser.error("temporal-review requires --output")
    from evaluation.planning.registry import project_path
    directory = project_path(ROOT, str(args.directory))
    if args.operation == "audit-review":
        from evaluation.reporting.cosmos_temporal_review import audit_cosmos_temporal_review
        result = audit_cosmos_temporal_review(ROOT, directory)
    elif args.operation == "temporal-review":
        from evaluation.reporting.cosmos_temporal_review import export_cosmos_temporal_review
        result = export_cosmos_temporal_review(ROOT, directory, args.output)
    elif args.operation == "prepare":
        from evaluation.planning.cosmos_horizon import prepare_cosmos_horizon
        study = prepare_cosmos_horizon(ROOT, directory)
        result = {"status": "prepared", "jobs": len(study["jobs"]), "planned_new_world_model_calls": study["planned_new_world_model_calls"]}
    elif args.operation == "run":
        from evaluation.execution.native_study import run_native_study
        overrides = None if args.resource_overrides is None else json.loads(project_path(ROOT, str(args.resource_overrides)).read_text())
        result = run_native_study(ROOT, directory, resource_overrides=overrides)
    elif args.operation == "reference-motion":
        from evaluation.reporting.cosmos_horizon import write_reference_motion
        result = write_reference_motion(directory)
    elif args.operation == "summarize":
        from evaluation.reporting.cosmos_horizon import summarize_horizon_study
        result = summarize_horizon_study(ROOT, directory)
    else:
        from evaluation.reporting.cosmos_horizon import audit_horizon_study, render_horizon_study
        result = audit_horizon_study(ROOT, directory) if args.operation == "audit" else render_horizon_study(ROOT, directory, selection=(args.case,) if args.case else ())
    display = {key: result[key] for key in ("status", "jobs", "planned_new_world_model_calls", "new_world_model_calls", "world_model_calls", "vla_calls") if key in result}
    if isinstance(display.get("jobs"), list):
        display["jobs"] = len(display["jobs"])
    if "cases" in result:
        display["cases"] = len(result["cases"])
    print(json.dumps(display), flush=True)


if __name__ == "__main__":
    main()
