"""The normal installer must reuse the same precisely scoped asset manifests."""
import json
from pathlib import Path
import shlex

import pytest


def profiles():
    root = Path(__file__).resolve().parents[1]
    script = (root/"scripts/setup/ctrl_world.sh").read_text()
    values = {}
    for line in script.replace("\\\n", " ").splitlines():
        if line.startswith("download_model "):
            args = shlex.split(line)
            values[args[1]] = [args[index+1] for index, value in enumerate(args) if value == "--allow"]
    assert 'shift 2' in script and '"${local_dir}" "$@"' in script
    return root, values


def test_installer_downloads_only_the_selected_runtime_exports():
    _, values = profiles()
    assert values["yjguo/Ctrl-World"] == ["checkpoint-10000.pt"]
    assert len(values["stabilityai/stable-video-diffusion-img2vid"]) == 9
    assert len(values["openai/clip-vit-base-patch32"]) == 6
    assert all("fp16" not in name and not name.endswith((".h5", ".msgpack"))
               for names in values.values() for name in names)


def test_installer_allow_lists_match_existing_verified_runtime_manifests():
    root, values = profiles()
    found = 0
    for folder in ("ctrl_world", "svd", "clip"):
        path = root/f"checkpoints/world_models/ctrl_world/{folder}/.vlaworldeval-download.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            assert values[manifest["repository"]] == manifest["allow_patterns"]
            found += 1
    if not found:
        pytest.skip("Optional runtime snapshots not installed")
