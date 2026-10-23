"""Execution command handlers; model imports remain lazy."""

from cli.output import print_json
from contextlib import redirect_stdout
import sys


def policy_check(args, root):
    from evaluation.execution.policy_check import run_policy_check
    from evaluation.execution.reliability import RunInterrupted

    try:
        with redirect_stdout(sys.stderr):
            report = run_policy_check(root, args.model, args.output, dataset=args.dataset,
                episode=args.episode, start=args.start, gpus=args.gpus, memory_gib=args.memory_gib,
                execute=args.execute, attention_backend=args.attention_backend, runtime_check=args.runtime_check)
    except RunInterrupted as exc:
        print(f"{exc}; inspect policy-check report and worker logs.", file=sys.stderr)
        return 128 + exc.signum
    print_json(report)
    return 0 if report["status"] in {"passed", "inputs_checked", "runtime_checked"} else 1


def run(args, root):
    from evaluation.execution.executor import execute_suite
    from evaluation.execution.reliability import RunInterrupted, run_signal_handlers

    try:
        with run_signal_handlers(), redirect_stdout(sys.stderr):
            result = execute_suite(
                args.config,
                experiments=args.experiment,
                seeds=args.seed,
                admit_only=args.admit_only,
            )
        print_json(result)
    except RunInterrupted as exc:
        print(f"{exc}; inspect the attempt's summary/logs if created.", file=sys.stderr)
        return 128 + exc.signum
    return 0


def evaluate(args, root):
    from evaluation.execution.workflow import evaluate_suite
    from evaluation.execution.reliability import RunInterrupted, run_signal_handlers

    try:
        with run_signal_handlers(), redirect_stdout(sys.stderr):
            report = evaluate_suite(
                args.config, experiments=args.experiment, seeds=args.seed, execute=args.execute
            )
    except RunInterrupted as exc:
        print(f"{exc}; inspect the attempt summary and worker logs.", file=sys.stderr)
        return 128 + exc.signum
    print_json(report)
    return 0 if report["ok"] else 1


def replay(args, root):
    from evaluation.execution.native_replay import run_native_replay
    from evaluation.execution.reliability import RunInterrupted, run_signal_handlers

    try:
        with run_signal_handlers(), redirect_stdout(sys.stderr):
            report = run_native_replay(
                args.config, args.input, args.output, admit_only=args.admit_only,
                image_anchors=getattr(args, "image_anchors", None),
                reference_path=getattr(args, "reference", None)
            )
    except RunInterrupted as exc:
        print(f"{exc}; inspect the replay summary if created.", file=sys.stderr)
        return 128 + exc.signum
    print_json(report)
    return 0 if report.get("ok", report.get("status") == "completed") else 1


def prepare_replay(args, root):
    from evaluation.execution.native_replay import prepare_ctrl_packet

    report = prepare_ctrl_packet(
        args.config,
        args.dataset,
        args.output,
        episode=args.episode,
        start=args.start,
        chunks=args.chunks,
    )
    print_json(report)
    return 0
