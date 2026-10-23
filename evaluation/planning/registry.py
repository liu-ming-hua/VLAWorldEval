"""Local runtime definitions: explicit pair admission, safe declarative bindings."""
from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path


def validate_bindings(bindings, *, paths_only=False):
    if not isinstance(bindings, dict):
        raise ValueError("Bindings must be an object")
    for name, binding in bindings.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"Invalid constructor argument: {name}")
        if isinstance(binding, dict):
            allowed = {"project_path", "runtime_path"} if paths_only else {
                "project_path", "runtime_path", "suite_path", "sampling", "runtime"}
            if len(binding) != 1 or set(binding) - allowed:
                raise ValueError(f"Invalid binding for {name}")
            value = next(iter(binding.values()))
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid binding value for {name}")
        elif paths_only or isinstance(binding, (list, tuple)):
            raise ValueError(f"Invalid binding for {name}")
        elif type(binding) not in (str, bool, int, float, type(None)) or (
                type(binding) in (int, float) and not math.isfinite(binding)):
            raise ValueError(f"Binding {name} requires a finite JSON scalar or declarative reference")


def project_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Project path must be a nonempty string")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Runtime path leaves the project: {value}")
    return path


def resolve_bindings(bindings, root, runtime, sampling, suite_dir):
    validate_bindings(bindings)
    result = {}
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            result[name] = binding
            continue
        if len(binding) != 1:
            raise ValueError(f"Invalid binding for {name}")
        source, value = next(iter(binding.items()))
        if source == "project_path":
            result[name] = str(project_path(root, value))
        elif source == "runtime_path":
            if value not in runtime:
                raise ValueError(f"Missing runtime field {value!r} for binding {name}")
            result[name] = str(project_path(root, runtime[value]))
        elif source == "suite_path":
            if Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError(f"Invalid suite path: {value}")
            result[name] = str(suite_dir / value)
        elif source in {"sampling", "runtime"}:
            context = sampling if source == "sampling" else runtime
            if value not in context:
                raise ValueError(f"Missing {source} field {value!r} for binding {name}")
            result[name] = context[value]
        else:
            raise ValueError(f"Unknown binding source: {source}")
    return result


@dataclass(frozen=True)
class RuntimeRegistry:
    models: dict
    datasets: dict
    pairs: list

    def asset_paths(self, root, pair, runtime, sampling):
        paths = {}
        for role, adapter in (("vlas", pair.vla.adapter), ("world_models", pair.world_model.adapter)):
            for name, binding in self.models[role][adapter].get("assets", {}).items():
                paths[f"{role}.{adapter}.{name}"] = resolve_bindings({name: binding}, root, runtime, sampling, root)[name]
        return paths

    def pair_profile(self, pair):
        matches = [profile for profile in self.pairs
                   if pair.vla.adapter in profile["vlas"] and pair.world_model.adapter == profile["world_model"]]
        if len(matches) != 1:
            raise ValueError(f"No unique registered runtime profile for {pair.pair_id}")
        return matches[0]

    def asset_status(self, root, pair, runtime, sampling):
        status = {}
        for role, adapter in (("vlas", pair.vla.adapter), ("world_models", pair.world_model.adapter)):
            for name, binding in self.models[role][adapter].get("assets", {}).items():
                path = Path(resolve_bindings({name: binding}, root, runtime, sampling, root)[name])
                present = ((path.is_file() and path.stat().st_size > 0)
                           or (path.is_dir() and any(path.iterdir())))
                status[f"{role}.{adapter}.{name}"] = present
                indices = list(path.glob("*.index.json")) if path.is_dir() else (
                    [path] if path.name.endswith(".index.json") and present else [])
                for index in indices:
                    content = json.loads(index.read_text(encoding="utf-8"))
                    if not isinstance(content, dict):
                        raise ValueError(f"Checkpoint index must be an object: {index}")
                    if "weight_map" not in content:
                        continue
                    mapping = content["weight_map"]
                    if (not isinstance(mapping, dict) or not mapping
                            or any(not isinstance(name, str) or not name for name in mapping.values())):
                        raise ValueError(f"Invalid checkpoint weight_map: {index}")
                    shards = set(mapping.values())
                    status[f"{role}.{adapter}.{name}.{index.name}.shards"] = bool(shards) and all(
                        project_path(root, str(index.parent / shard)).is_file()
                        and project_path(root, str(index.parent / shard)).stat().st_size > 0
                        for shard in shards)
        return status

    def load_episode(self, profile, root, dataset, job, horizon):
        definition = self.datasets[profile["dataset"]]
        loader = getattr(importlib.import_module(definition["module"]), definition["loader"])
        context = {"root": root, "episode": job["episode_index"], "start": dataset["start_frame"],
                   "chunks": job["chunks"], "horizon": horizon, "fps": dataset["fps"],
                   "instruction": dataset.get("instruction"), "replay": job["mode"] == "wm_replay"}
        return loader(**{argument: context[source] for argument, source in definition["arguments"].items()})


def load_runtime_registry(root: Path) -> RuntimeRegistry:
    from evaluation.planning.registry_schema import load_runtime_data
    data = load_runtime_data(root / "configs/runtime_registry.json", validate_bindings)
    return RuntimeRegistry(data["models"], data["datasets"], data["pairs"])
