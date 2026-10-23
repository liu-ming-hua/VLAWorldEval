"""Reports command handlers; model imports remain lazy."""

from cli.output import print_json


def results(args, root):
    from evaluation.reporting.results import collect_results

    print_json(collect_results(args.root))
    return 0


def inspect(args, root):
    from evaluation.reporting.video_diagnostics import DiagnosticThresholds, diagnose_artifact

    report = diagnose_artifact(
        args.artifact,
        thresholds=DiagnosticThresholds(
            near_static_mae=args.near_static_mae, jump_mae=args.jump_mae
        ),
    )
    print_json(report)
    return 0


def scores(args, root):
    from evaluation.reporting.task_scores import load_annotations, summarize_scores

    print_json(summarize_scores(load_annotations(args.annotations)))
    return 0


def regression_report(args, root):
    from evaluation.reporting.regression_report import summarize_regressions

    report = summarize_regressions(root, args.report)
    print_json(report)
    return 0 if report["ok"] else 1


def replay_report(args, root):
    from evaluation.reporting.native_reference import summarize_native_references
    result = summarize_native_references(root, args.directory, args.output, plan_path=args.plan,
                                        metrics_directory=getattr(args, "metrics", None),
                                        additional_directories=getattr(args, "additional_directory", ()))
    print_json(result)
    return 0


def motion_eval(args, root):
    import json
    import os
    import subprocess
    from evaluation.planning.registry import project_path
    from adapters.environment import get_environment
    from vlaworldeval.cache import runtime_cache_environment
    summary = project_path(root, str(args.summary))
    reference = project_path(root, str(args.reference))
    output = project_path(root, str(args.output))
    if args.annotations:
        from evaluation.reporting.motion_detector import score_annotations
        result = score_annotations(summary, reference, project_path(root, str(args.annotations)), output)
    else:
        runtime = get_environment("evaluators.video")
        environment = {**os.environ, **runtime_cache_environment(root), "CUDA_VISIBLE_DEVICES": "",
                       "YOLO_CONFIG_DIR": str(root/".cache/ultralytics"), "HF_HUB_OFFLINE": "1"}
        child = subprocess.run([str(runtime.python), "-m", "evaluation.reporting.motion_detector",
            "--summary", str(summary), "--reference", str(reference), "--output", str(output), "--provider", args.provider],
            cwd=root, env=environment, capture_output=True, text=True, timeout=3600)
        if child.returncode:
            raise ValueError("Motion evaluator failed: "+child.stderr[-4000:]+child.stdout[-1000:])
        result = json.loads((output/"motion.json").read_text())
    print_json(result)
    return 0
