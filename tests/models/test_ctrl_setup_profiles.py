"""The normal installer must reuse the same precisely scoped asset manifests."""

import shlex
from pathlib import Path


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


def test_installer_downloads_only_the_selected_runtime_exports():
    _, values = profiles()
    assert values["yjguo/Ctrl-World"] == ["checkpoint-10000.pt"]
    assert len(values["stabilityai/stable-video-diffusion-img2vid"]) == 9
    assert len(values["openai/clip-vit-base-patch32"]) == 6
    assert all(
        "fp16" not in name and not name.endswith((".h5", ".msgpack"))
        for names in values.values()
        for name in names
    )
