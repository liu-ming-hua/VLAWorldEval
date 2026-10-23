"""Project command handlers; model imports remain lazy."""

from cli.output import print_json
import json


def _models(root) -> int:
    from adapters.environment import load_environment_registry
    from evaluation.planning.catalog import load_model_catalog

    environments = load_environment_registry()
    catalog = load_model_catalog(root)
    for group in ("vlas", "world_models"):
        print(f"{group}:")
        for key, entry in catalog[group].items():
            runtime = environments.get(entry["environment"])
            availability = (
                "interpreter_present" if runtime and runtime.exists else "interpreter_missing"
            )
            level = entry.get("integration", {}).get("level", "unclassified")
            print(f"  {key:24} {level:15} {entry['adapter']} {availability}")
    print(
        "Tiers describe declared engineering routes, not execution or quality. "
        "Interpreter presence is not model/asset admission; use capabilities for exact pairs and boundaries. "
        "Incompatible is scoped to the current robot-action loop, not native control-video inference."
    )
    return 0


def _environments(root) -> int:
    from adapters.environment import load_environment_registry

    for key, spec in sorted(load_environment_registry().items()):
        status = "available" if spec.exists else "missing"
        print(f"{key:32} {spec.status:10} {status:9} {spec.path}")
    return 0


def _validate(root, require_assets=False) -> int:
    from evaluation.planning.inventory import validate_catalog, validate_datasets, validate_pairs

    print(f"Model catalog OK: {validate_catalog(root)}")
    datasets = validate_datasets(root, require_assets=require_assets)
    print(
        f"Dataset inventory OK: {len(datasets)} manifests; assets required={require_assets}; "
        f"missing={sum(d['status'] == 'missing_assets' for d in datasets)}"
    )
    print(f"Pair protocols OK: {len(validate_pairs(root))} manifests")
    from evaluation.planning.experiments import build_experiment_plan
    from evaluation.planning.registry import load_runtime_registry

    registry = load_runtime_registry(root)
    print(f"Runtime registry OK: {sum(len(p['vlas']) for p in registry.pairs)} executable pairs")
    for path in sorted((root / "configs/experiments").glob("*.json")):
        plan = build_experiment_plan(path)
        print(
            f"Experiment plan OK: {plan['suite_id']}, {len(plan['jobs'])} planned jobs (not executed)"
        )
    from evaluation.validation.acceptance import load_acceptance_matrix

    for path in sorted((root / "configs/acceptance").glob("*.json")):
        matrix = load_acceptance_matrix(path)
        print(f"Acceptance matrix OK: {path.name}, {len(matrix['entries'])} entries (not executed)")
    return 0


def models(args, root):
    return _models(root)


def environments(args, root):
    return _environments(root)


def validate(args, root):
    return _validate(root, require_assets=args.assets)


def runtimes(args, root):
    from evaluation.planning.registry import load_runtime_registry

    registry = load_runtime_registry(root)
    print_json({"models": registry.models, "datasets": registry.datasets, "pairs": registry.pairs})
    return 0


def capabilities(args, root):
    from evaluation.planning.capabilities import capabilities

    print_json(capabilities(root))
    return 0


def plan(args, root):
    from evaluation.planning.experiments import build_experiment_plan

    print_json(build_experiment_plan(args.config))
    return 0


def doctor(args, root):
    from evaluation.planning.diagnostics import diagnose_suite, format_diagnostics

    report = diagnose_suite(
        args.config,
        experiments=args.experiment,
        seeds=args.seed,
        check_data=args.data,
        check_gpu=args.gpu,
    )
    print(
        json.dumps(report, indent=2, ensure_ascii=False)
        if args.json
        else format_diagnostics(report)
    )
    return 0 if report["ok"] else 2
