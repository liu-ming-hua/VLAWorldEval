"""Prepare separately obtained WorldGym code without importing any model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

from vlaworldeval.asset_lock import destination_locked

REVISION = "709665b29f5c95124e7dbc05b0d4d47cd353de97"
SOURCE_HASHES = {
    "model.py": "a33d68ff4f8aca11059381536c381d6c52c94edbff6d603b43a4ee497be2169a",
    "diffusion.py": "60a7f9e34ff8d301ec06ff306705f531c5b9604e2a091301fbb1970e9f0ba643",
    "vae.py": "c0ec194295a40aa2cd992ab5f1f3568e68da802ece0b3d20d25745eabce54d7a",
    "world_model.py": "8171fca40ea924b48f73331ee5cd0c2a8462d3e9de5f9ce649ba5513fecb1ce5",
}
RUNTIME_HASHES = {**SOURCE_HASHES,
    "__init__.py": hashlib.sha256(b"").hexdigest(),
    "vae.py": "9f870ce84c24db35a9bf51bd9f1cefe3689a544ab4fadc5c261f7a66287f0ebf",
    "world_model.py": "6bcf4c633c4fcdefaaec88aeec649986a0870d866738fe07f16638bbff6bdb15",
}


def runtime_directory(root):
    return Path(root) / "third_party/worldgym_runtime"


def runtime_contents(source):
    """Exact-version compatibility edits; network math is unchanged."""
    result = {"__init__.py": b""}
    for name, expected in SOURCE_HASHES.items():
        path = Path(source) / "src/world_model_eval" / name
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"WorldGym upstream source differs from pinned revision: {path}")
        text = data.decode()
        if name == "world_model.py":
            text = text.replace("weights_only=True)", 'weights_only=True, map_location="cpu")')
        elif name == "vae.py":
            text = text.replace("import einops\n", "import einops\nimport os\n")
            text = text.replace(
                '"stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae"',
                'os.environ["WORLDGYM_VAE_PATH"]',
            )
        result[name] = text.encode()
    return result


def verify_runtime(root):
    directory = runtime_directory(root)
    if directory.is_symlink() or not directory.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("WorldGym runtime must stay inside the project")
    marker = directory / "runtime.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("WorldGym runtime missing; run bash scripts/setup/worldgym_runtime.sh")
    value = json.loads(marker.read_text())
    if (not isinstance(value, dict) or value.get("revision") != REVISION or value.get("schema_version") != 1
            or value.get("files") != RUNTIME_HASHES):
        raise ValueError("Invalid WorldGym runtime manifest; use a separately reviewed runtime")
    for name, expected in RUNTIME_HASHES.items():
        path = directory / "worldgym_runtime" / name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"WorldGym runtime integrity mismatch: {path}")
    return value


@destination_locked("destination")
def prepare_runtime(root, source, destination):
    root, destination = Path(root).resolve(), Path(destination)
    if destination.resolve() != runtime_directory(root).resolve() or destination.is_symlink():
        raise ValueError("WorldGym runtime destination must be the project-local runtime directory")
    contents = runtime_contents(source)
    manifest = {"schema_version": 1, "revision": REVISION,
                "source": str(Path(source).resolve()),
                "patches": ["cpu_checkpoint_loading", "explicit_local_vae"],
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
                "redistribution_permission_verified": False}
    if destination.exists():
        existing = verify_runtime(root)
        if existing["files"] != manifest["files"]:
            raise ValueError("Existing WorldGym runtime differs; preserving it")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".worldgym-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "runtime"
        package = stage / "worldgym_runtime"
        package.mkdir(parents=True)
        for name, data in contents.items():
            (package / name).write_bytes(data)
        (stage / "runtime.json").write_text(json.dumps(manifest, indent=2))
        stage.rename(destination)
    return verify_runtime(root)
