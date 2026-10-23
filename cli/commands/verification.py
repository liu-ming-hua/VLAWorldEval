"""Verification command handlers; model imports remain lazy."""

from cli.output import print_json
from contextlib import redirect_stdout
import sys


def accept(args, root):
    from evaluation.validation.acceptance import run_acceptance
    from evaluation.execution.reliability import RunInterrupted, run_signal_handlers

    try:
        with run_signal_handlers(), redirect_stdout(sys.stderr):
            report = run_acceptance(
                args.matrix, pairs=args.pair, level=args.level, execute=args.execute
            )
        print_json(report)
        return 0 if report["ok"] else 1
    except RunInterrupted as exc:
        print(f"{exc}; inspect results/acceptance for the interrupted report.", file=sys.stderr)
        return 128 + exc.signum


def env_audit(args, root):
    from evaluation.validation.environment_audit import audit_environments

    report = audit_environments(
        keys=args.environment, baseline=args.baseline, check_dependencies=args.dependencies
    )
    print_json(report)
    return 0 if report["ok"] else 1


def engineering_accept(args, root):
    from evaluation.validation.engineering import run_engineering_acceptance

    report = run_engineering_acceptance(root, args.output, timeout=args.timeout)
    print_json(report)
    return 0 if report["status"] == "passed" else 1
