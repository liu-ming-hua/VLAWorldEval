"""One asset-root convention for source and installed-package operation."""
import os
from pathlib import Path


def project_root():
    """Explicit asset project wins; otherwise preserve source-checkout defaults.

    Importing the library/CPU demo does not require model assets. Asset commands
    check the returned location separately and give a setup hint if absent.
    """
    value = os.environ.get("VLAWORLDEVAL_ROOT")
    if value is not None:
        root = Path(value).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise ValueError("VLAWORLDEVAL_ROOT must be an existing absolute project directory")
        return root.resolve()
    return Path(__file__).resolve().parents[1]


def require_asset_project(root):
    required = ("configs/runtime_registry.json", "environment/registry.toml")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("Missing project assets: " + ", ".join(missing)
                         + "; set VLAWORLDEVAL_ROOT to the asset project directory")


def fresh_output_path(root, value):
    """Preflight only; callers must atomically mkdir(exist_ok=False) before writing."""
    root = Path(root).resolve()
    raw = root / value
    if raw.is_symlink():
        raise ValueError("Output must not be a symlink")
    output = raw.resolve()
    if not output.is_relative_to(root):
        raise ValueError("Output must remain inside the project")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new experiment directory: {output}")
    return output


def gigaworld1_source():
    """Portable source default; preserve explicitly registered legacy .venv roots."""
    value = os.environ.get("GIGAWORLD1_REPO")
    if value:
        path = Path(value).expanduser()
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("GIGAWORLD1_REPO must be an existing absolute source directory")
        return path.resolve()
    local = project_root() / "third_party/giga-world-1"
    if local.is_dir():
        return local.resolve()
    from adapters.environment import get_environment
    environment = get_environment("world_models.gigaworld1")
    if environment.path.name == ".venv":
        return environment.path.parent.resolve()
    return local
