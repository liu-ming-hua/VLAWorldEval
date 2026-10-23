#!/usr/bin/env python3
"""Compatibility command; prefer `vlaworldeval policy-check --model cosmos_droid`."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from evaluation.execution.policy_check import run_policy_check
    from evaluation.execution.reliability import RunInterrupted

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=350)
    parser.add_argument("--start", type=int, default=15)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--memory-gib", type=float, default=16)
    args = parser.parse_args()
    try:
        report = run_policy_check(ROOT, "cosmos_droid", args.output, episode=args.episode,
            start=args.start, gpus=args.gpus, memory_gib=args.memory_gib, execute=True)
    except RunInterrupted as exc:
        return 128 + exc.signum
    print(json.dumps({"status": report["status"], "output": report["output"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
