"""Workspace command handlers; model imports remain lazy."""

from cli.output import print_json


def init(args, root):
    from vlaworldeval.workspace import initialize_workspace

    print_json(initialize_workspace(args.directory))
    return 0


def verify_assets(args, root):
    from vlaworldeval.assets import verify_snapshot

    print_json(verify_snapshot(args.directory))
    return 0


def configure(args, root):
    from vlaworldeval.workspace import configure_suite

    print_json(
        configure_suite(
            args.config,
            args.output,
            root=root,
            vla_gpus=args.vla_gpus,
            world_model_gpus=args.wm_gpus,
            budget=args.budget_gib,
            vla_budget=args.vla_budget_gib,
            chunks=args.chunks,
        )
    )
    return 0


def demo(args, root):
    from vlaworldeval.demo import run_demo

    print_json(run_demo(args.output, chunks=args.chunks))
    return 0


def cleanup_results(args, root):
    from vlaworldeval.maintenance import archive_cleanup, plan_cleanup

    plan = plan_cleanup(root)
    report = archive_cleanup(root, plan) if args.apply else plan
    print_json(report)
    return 0
