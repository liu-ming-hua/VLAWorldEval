#!/usr/bin/env python3
"""Recorded-event Cosmos controls using the shared, frozen native replay queue."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "freeze", "run", "audit", "render", "finalize"])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--wait", action="store_true", help="Wait up to two hours for the existing queue; finalize only")
    args = parser.parse_args()
    if args.wait and args.operation != "finalize":
        parser.error("--wait is only valid for finalize")
    from evaluation.planning.registry import project_path
    directory = project_path(ROOT, str(args.directory))
    if args.operation in {"prepare", "freeze"}:
        from evaluation.planning.cosmos_events import prepare_event_study, freeze_event_study
        result = (prepare_event_study if args.operation == "prepare" else freeze_event_study)(ROOT, directory)
    elif args.operation == "run":
        from evaluation.execution.native_study import run_native_study
        result = run_native_study(ROOT, directory)
    elif args.operation == "finalize":
        from evaluation.reporting.cosmos_events import finalize_event_study
        result = finalize_event_study(ROOT, directory, wait=args.wait)
    else:
        from evaluation.reporting.cosmos_events import audit_event_study, render_event_study
        result = (audit_event_study if args.operation == "audit" else render_event_study)(ROOT, directory)
    print(json.dumps({key: len(value) if isinstance(value, list) else value
        for key, value in result.items() if key in {"status", "jobs", "cases", "world_model_calls", "planned_new_world_model_calls", "vla_calls"}}), flush=True)


if __name__ == "__main__":
    main()
