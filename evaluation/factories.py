"""Declared runtime factories; no model-specific imports in the parent."""
from dataclasses import dataclass
from pathlib import Path
from evaluation.planning.registry import load_runtime_registry, resolve_bindings


@dataclass
class WorkerSpec:
    module: str
    adapter: str
    kwargs: dict


def resolve_workers(root: Path, pair, runtime, sampling, suite_dir: Path):
    registry = load_runtime_registry(root)
    registry.pair_profile(pair)
    specs = []
    for role, adapter in (("vlas", pair.vla.adapter), ("world_models", pair.world_model.adapter)):
        definition = registry.models[role][adapter]
        specs.append(WorkerSpec(definition["module"], adapter,
                                resolve_bindings(definition["kwargs"], root, runtime, sampling, suite_dir)))
    return tuple(specs)
