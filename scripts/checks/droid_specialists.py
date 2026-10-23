#!/usr/bin/env python3
"""Verify downloaded specialist identity, then run the existing unified evaluator."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.execution.workflow import evaluate_suite  # noqa: E402
from vlaworldeval.assets import verify_snapshot  # noqa: E402
from vlaworldeval.atomic_json import publish_json_exclusive  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["groot_n17_droid", "pi0_fast_droid"], required=True)
    parser.add_argument("--world-model", choices=["ctrl", "cosmos"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists() or args.wait_seconds < 0:
        raise ValueError("Use a new project-local output file and a nonnegative wait")
    checkpoint = ROOT / ("checkpoints/vla/groot/n1.7-droid" if args.model == "groot_n17_droid"
                         else "checkpoints/vla/openpi/pi0-fast-droid")
    deadline = time.monotonic() + args.wait_seconds
    while not (checkpoint / ".vlaworldeval-download.json").is_file():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f"Checkpoint download has not been verified: {checkpoint}")
        time.sleep(5)
    identity = verify_snapshot(checkpoint)
    expected = "nvidia/GR00T-N1.7-DROID" if args.model == "groot_n17_droid" else "RomanticQq/pi0_fast_droid"
    if identity["repository"] != expected:
        raise ValueError("Specialist checkpoint repository identity mismatch")
    official = None
    if args.model == "pi0_fast_droid":
        from vlaworldeval.openpi_mirror import verify_openpi_mirror
        official = verify_openpi_mirror(checkpoint)
    config = ROOT / f"configs/experiments/{args.model}_{args.world_model}_v1.json"
    report = evaluate_suite(config, experiments=["loop_smoke", "loop_full_window"], execute=True)
    report["specialist_asset"] = identity
    report["official_source_verification"] = official
    output.parent.mkdir(parents=True, exist_ok=True)
    publish_json_exclusive(output, report)
    print(json.dumps({"status": report["status"], "ok": report["ok"], "report": str(output)}), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
