"""Optional native-domain parity; requires separately acquired assets or runtimes."""

import json
import shlex
from pathlib import Path

import pytest


def profiles():
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts/setup/ctrl_world.sh").read_text()
    values = {}
    for line in script.replace("\\\n", " ").splitlines():
        if line.startswith("download_model "):
            args = shlex.split(line)
            values[args[1]] = [
                args[index + 1] for index, value in enumerate(args) if value == "--allow"
            ]
    assert "shift 2" in script and '"${local_dir}" "$@"' in script
    return root, values


def test_installer_allow_lists_match_existing_verified_runtime_manifests():
    root, values = profiles()
    found = 0
    for folder in ("ctrl_world", "svd", "clip"):
        path = root / f"checkpoints/world_models/ctrl_world/{folder}/.vlaworldeval-download.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            assert values[manifest["repository"]] == manifest["allow_patterns"]
            found += 1
    if not found:
        pytest.skip("Optional runtime snapshots not installed")
