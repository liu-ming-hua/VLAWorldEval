"""Portable asset workspaces, with packaged templates and no implicit downloads."""
from __future__ import annotations

import hashlib
import json
import math
import os
from importlib import metadata
from pathlib import Path


def template_files() -> dict[str, bytes]:
    """Use distribution records, not sys.prefix (also works with pip --target)."""
    distribution = metadata.distribution("vlaworldeval")
    files = {}
    prefix = "share/vlaworldeval/"
    for record in distribution.files or ():
        name = str(record).replace("\\", "/")
        if prefix in name:
            relative = name.split(prefix, 1)[1]
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError(f"Unsafe distribution template: {relative}")
            location = distribution.locate_file(record)
            if not location.is_file():
                # pip --target relocates data files under target/share but
                # retains their original prefix-relative paths in RECORD.
                location = distribution.locate_file(prefix + relative)
            files[relative] = location.read_bytes()
    if not files:  # Editable installation: resources remain in the source tree.
        source = Path(__file__).resolve().parents[1]
        for folder, patterns in (("configs", ("*.json", "*.yaml")),
                                 ("environment/locks", ("*.txt", "*.in")),
                                 ("environment/requirements", ("*.txt",))):
            for pattern in patterns:
                for path in (source / folder).rglob(pattern):
                    files[path.relative_to(source).as_posix()] = path.read_bytes()
        registry = source / "environment/registry.toml"
        if registry.is_file():
            files["environment/registry.toml"] = registry.read_bytes()
        for path in (source / "datasets").glob("*/manifest.json"):
            files[path.relative_to(source).as_posix()] = path.read_bytes()
    if "configs/runtime_registry.json" not in files or "environment/registry.toml" not in files:
        raise ValueError("Installation lacks workspace templates; reinstall a complete wheel or source checkout")
    return files


def _portable_template(name: str, content: bytes, destination: Path) -> bytes:
    if name == "environment/registry.toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib
        from adapters.environment import _flatten
        entries = _flatten(tomllib.loads(content.decode())["environments"])
        lines = ["# Generated asset workspace. Environments must be installed separately.\n"]
        for key, spec in entries.items():
            path = spec["path"]
            if Path(path).is_absolute():
                path = "environment/" + key.replace(".", "/")
            lines.append(f"[environments.{key}]\n")
            for field, value in (("path", path), ("status", "configured"),
                                 ("scope", spec["scope"]),
                                 ("package_profile", spec["package_profile"]),
                                 ("notes", "Template only; runtime availability is checked from bin/python.")):
                lines.append(f"{field} = {json.dumps(value)}\n")
            lines.append("\n")
        return "".join(lines).encode()
    return content


def initialize_workspace(destination: Path) -> dict:
    """Create a fresh empty directory only; never merge into user files."""
    destination = Path(destination).absolute()
    if destination.is_symlink() or (destination.exists() and
                                   (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError("Workspace destination must be absent or an empty non-symlink directory")
    files = template_files()  # Resolve all resources before creating anything.
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, content in sorted(files.items()):
        content = _portable_template(name, content, destination)
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    for name in ("checkpoints/vla", "checkpoints/world_models", "datasets", "results",
                 "third_party", ".cache", ".uv-cache"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "workspace": str(destination.resolve()),
              "package_version": metadata.version("vlaworldeval"), "templates": hashes,
              "assets_downloaded": False, "environments_created": False,
              "next": "Set VLAWORLDEVAL_ROOT to this directory; use models/capabilities/plan. "
                      "Real model setup requires the source checkout and licensed assets."}
    (destination / "workspace.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def configure_suite(source: Path, destination: Path, *, root: Path,
                    vla_gpus=None, world_model_gpus=None, budget=None,
                    vla_budget=None, chunks=None) -> dict:
    """Write a validated independent suite without overwriting the original."""
    from evaluation.planning.configuration import load_suite
    from evaluation.planning.experiments import build_experiment_plan
    destination = Path(destination)
    if not destination.is_absolute():
        destination = root / destination
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError("New suite must be inside the asset workspace")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Output already exists; choose a new config filename")
    source = Path(source)
    if not source.is_absolute():
        source = root / source
    value = load_suite(source)
    runtime = value["runtime"]
    for field, indices in (("vla_gpus", vla_gpus), ("world_model_gpus", world_model_gpus)):
        if indices is not None:
            if not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
                raise ValueError(f"{field} requires unique nonnegative GPU indices")
            runtime[field] = indices
    for field, number in (("max_process_memory_gib_per_gpu", budget), ("vla_memory_gib", vla_budget)):
        if number is not None:
            if isinstance(number, bool) or not math.isfinite(number) or number <= 0:
                raise ValueError(f"{field} must be finite and positive")
            runtime[field] = number
    if chunks is not None:
        if type(chunks) is not int or chunks <= 0:
            raise ValueError("chunks must be a positive integer")
        for experiment in value["experiments"]:
            experiment["chunks"] = chunks
    if not destination.stem or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in destination.stem):
        raise ValueError("Config filename must use letters, numbers, underscores or hyphens")
    value["suite_id"] = destination.stem
    value["output_root"] = "results/experiments/" + destination.stem
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects the source and existing results. Remove only
    # our own newly-created file if semantic validation fails.
    with destination.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        plan = build_experiment_plan(destination, root=root)
    except BaseException:
        destination.unlink()
        raise
    return {"config": str(destination), "suite_id": value["suite_id"],
            "jobs": len(plan["jobs"]), "executed": False,
            "next": "doctor --data --gpu before run; budgets are limits, not measured requirements"}


def direct_network_environment(root: Path) -> dict[str, str]:
    """Proxy-free mirror/cache settings shared by asset tools."""
    env = dict(os.environ)
    from vlaworldeval.cache import runtime_cache_environment
    env.update(runtime_cache_environment(root))
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update(HF_ENDPOINT=env.get("HF_ENDPOINT", "https://hf-mirror.com"),
               HF_HOME=str(root / ".cache/huggingface"), UV_CACHE_DIR=str(root / ".uv-cache"),
               HF_HUB_DISABLE_XET="1", HF_HUB_DOWNLOAD_TIMEOUT="600",
               UV_INDEX_URL=env.get("PYPI_MIRROR", "https://pypi.tuna.tsinghua.edu.cn/simple"))
    return env
