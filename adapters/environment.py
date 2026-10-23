"""Unified registry for model-specific Python environments.

Legacy evaluators can run both adapters in one compatible interpreter.
Cross-environment acceptance instead launches isolated workers; their dependency
stacks do not need to be merged. This registry resolves either runtime without
importing model-specific packages.
"""

from __future__ import annotations

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib
from dataclasses import dataclass
from pathlib import Path
import json
import os


from vlaworldeval.paths import project_root

PROJECT_ROOT = project_root()
REGISTRY_PATH = PROJECT_ROOT / "environment" / "registry.toml"


@dataclass(frozen=True)
class EnvironmentSpec:
    key: str
    path: Path
    status: str
    scope: str
    package_profile: str
    notes: str = ""

    @property
    def python(self) -> Path:
        return self.path / "bin" / "python"

    @property
    def exists(self) -> bool:
        return self.python.is_file()


def _flatten(table: dict, prefix: tuple[str, ...] = ()) -> dict[str, dict]:
    flattened: dict[str, dict] = {}
    for key, value in table.items():
        current = prefix + (key,)
        if isinstance(value, dict) and "path" in value:
            flattened[".".join(current)] = value
        elif isinstance(value, dict):
            flattened.update(_flatten(value, current))
    return flattened


def load_environment_registry(path: str | Path = REGISTRY_PATH) -> dict[str, EnvironmentSpec]:
    registry_path = Path(path)
    raw = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    specs = {}
    for key, value in _flatten(raw.get("environments", {})).items():
        env_path = Path(value["path"])
        if not env_path.is_absolute():
            env_path = PROJECT_ROOT / env_path
        specs[key] = EnvironmentSpec(
            key=key,
            path=env_path,
            status=value.get("status", "unknown"),
            scope=value.get("scope", "unknown"),
            package_profile=value.get("package_profile", ""),
            notes=value.get("notes", ""),
        )
    override_file = os.environ.get("VLAWORLDEVAL_ENVIRONMENTS")
    if override_file:
        source = Path(override_file).expanduser()
        if not source.is_absolute():
            source = PROJECT_ROOT / source
        overrides = json.loads(source.read_text())
        if (not isinstance(overrides, dict) or set(overrides) != {"schema_version", "paths"}
                or type(overrides["schema_version"]) is not int or overrides["schema_version"] != 1
                or not isinstance(overrides["paths"], dict)):
            raise ValueError("Environment override requires schema_version=1 and paths object")
        for key, value in overrides["paths"].items():
            if key not in specs:
                raise ValueError(f"Unknown override environment: {key}")
            if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_dir():
                raise ValueError(f"Environment override must be an existing absolute directory: {key}")
            old = specs[key]
            specs[key] = EnvironmentSpec(key, Path(value).resolve(), "overridden", old.scope,
                                         old.package_profile, f"Explicit override from {source}")
    return specs


def get_environment(key: str) -> EnvironmentSpec:
    try:
        return load_environment_registry()[key]
    except KeyError as exc:
        known = ", ".join(sorted(load_environment_registry()))
        raise KeyError(f"Unknown environment {key!r}; known environments: {known}") from exc


def adapter_environment(adapter: object) -> EnvironmentSpec:
    key = getattr(adapter, "environment_key", None)
    if not key:
        raise ValueError(f"{type(adapter).__name__} does not declare environment_key")
    return get_environment(key)


__all__ = [
    "EnvironmentSpec",
    "PROJECT_ROOT",
    "REGISTRY_PATH",
    "adapter_environment",
    "get_environment",
    "load_environment_registry",
]
