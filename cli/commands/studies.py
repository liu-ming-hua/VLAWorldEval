"""DROID fixed-set preparation and resource/review-gated queue management."""
from cli.output import print_json
from contextlib import redirect_stdout
import sys


def native_study(args, root):
    if args.operation == "stream":
        from evaluation.execution.native_stream import run_native_stream
        from evaluation.execution.reliability import run_signal_handlers
        if args.plan is None or args.batch_directory is None:
            raise ValueError("stream requires --plan and --batch-directory")
        with run_signal_handlers(), redirect_stdout(sys.stderr):
            result = run_native_stream(root, args.plan, args.batch_directory, args.directory,
                                       gpus=args.gpus, previous_directory=args.previous_directory)
    elif args.operation == "batch":
        from evaluation.planning.native_batches import freeze_ready_batch
        if args.plan is None or args.prepared_progress is None:
            raise ValueError("batch requires --plan and --prepared-progress")
        result = freeze_ready_batch(root, args.plan, args.prepared_progress, args.directory,
                                    limit=args.batch_limit)
    elif args.operation == "subset":
        from evaluation.planning.native_subset import freeze_reference_subset
        if args.plan is None:
            raise ValueError("subset requires --plan and a new --directory")
        result = freeze_reference_subset(root, args.plan, args.directory,
                                        limit=args.case_limit, seed=args.sample_seed)
    elif args.operation == "score":
        from evaluation.reporting.checkpoint_metrics import score_checkpoint_study
        if args.output is None or args.plan is None:
            raise ValueError("score requires the complete --plan and a new --output journal directory")
        result = score_checkpoint_study(root, args.plan, args.directory, args.output,
            watch=args.watch, workers=args.workers, provider=args.motion_provider,
            timeout_seconds=getattr(args, "metric_timeout_seconds", 3600),
            metric_gpus=getattr(args, "metric_gpus", ()),
            frame_batch_size=getattr(args, "metric_frame_batch_size", 1),
            additional_directories=getattr(args, "additional_directory", ()))
    elif args.operation == "remainder":
        from evaluation.planning.native_reference import prepare_reference_remainder
        if args.plan is None:
            raise ValueError("remainder requires the original full --plan")
        result = prepare_reference_remainder(root, args.plan, args.source_directory, args.directory, args.gpus)
    elif args.operation == "repartition":
        from evaluation.planning.native_reference import prepare_reference_repartition
        result = prepare_reference_repartition(root, args.source_directory, args.directory, args.gpus)
    elif args.operation == "retry":
        from evaluation.planning.native_reference import prepare_reference_retry
        if args.output is None:
            raise ValueError("retry requires --output pointing to a new directory")
        result = prepare_reference_retry(root, args.directory, args.output)
    elif args.operation == "prepare":
        from evaluation.planning.native_reference import prepare_reference_study
        if args.plan is None:
            raise ValueError("prepare requires --plan")
        if getattr(args, "prepared_manifest", None) is not None:
            from evaluation.planning.native_batches import validate_ready_batch
            validate_ready_batch(root, args.plan, args.prepared_manifest)
        result = prepare_reference_study(root, args.plan, args.directory)
    elif args.operation == "run":
        from evaluation.execution.native_study import run_native_study
        from evaluation.execution.reliability import run_signal_handlers
        overrides = None
        if getattr(args, "memory_gib", None) is not None:
            import json
            from pathlib import Path
            from evaluation.planning.registry import project_path
            study = json.loads((project_path(root, str(args.directory))/"study.json").read_text())
            overrides = {json.loads(Path(job["config"]).read_text())["environment"]: args.memory_gib
                         for job in study["jobs"]}
        with run_signal_handlers(), redirect_stdout(sys.stderr):
            result = run_native_study(root, args.directory, watch=args.watch, resource_overrides=overrides,
                                     exclusive_gpu=getattr(args, "exclusive_gpu", False),
                                     gpu_override=getattr(args, "gpu", None),
                                     gpu_map=getattr(args, "gpus", None),
                                     reuse_worker=getattr(args, "reuse_worker", False))
    else:
        import json
        from evaluation.planning.registry import project_path
        result = json.loads((project_path(root, str(args.directory)) / "progress.json").read_text())
    print_json(result)
    return 0 if result.get("status", "completed") == "completed" else 1


def droid_study(args, root):
    if args.operation == "compare":
        from evaluation.reporting.positive_controls import compare_frozen_study
        if args.output is None:
            raise ValueError("compare requires --output pointing to a fresh directory")
        with redirect_stdout(sys.stderr):
            result = compare_frozen_study(root, args.directory, args.output)
    elif args.operation == "prepare":
        from evaluation.planning.droid_study import prepare_study
        result = prepare_study(args.config, args.directory)
    elif args.operation == "run":
        from evaluation.execution.droid_study import run_study
        from evaluation.execution.reliability import RunInterrupted, run_signal_handlers
        try:
            with run_signal_handlers(), redirect_stdout(sys.stderr):
                result = run_study(args.directory, phase=args.phase, watch=args.watch)
        except RunInterrupted as exc:
            print(f"{exc}; inspect study progress and attempt logs before resuming.", file=sys.stderr)
            return 128 + exc.signum
    else:
        import json
        from evaluation.planning.registry import project_path
        directory = project_path(root, str(args.directory))
        result = {p.name: json.loads(p.read_text()) for p in directory.glob("*_progress.json")}
    print_json(result)
    return 1 if result.get("status") == "finished_with_failures" else 0
