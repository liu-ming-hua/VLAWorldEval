"""Exercise environment path guards without downloading or installing packages."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.process


def fixture(tmp_path):
    root = tmp_path / "project"
    script = root / "scripts/setup/worldgym.sh"
    script.parent.mkdir(parents=True)
    original = Path(__file__).resolve().parents[2] / "scripts/setup/worldgym.sh"
    shutil.copyfile(original, script)
    return root, script


@pytest.mark.parametrize("kind", ["outside", "traversal", "symlink"])
def test_worldgym_environment_cannot_escape_project(tmp_path, kind):
    root, script = fixture(tmp_path)
    outside = tmp_path / "outside"
    if kind == "outside":
        target = outside
    elif kind == "traversal":
        target = root / "environment/../../outside"
    else:
        (root / "environment").mkdir()
        target = root / "environment/link"
        target.symlink_to(outside, target_is_directory=True)
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "WORLDGYM_ENV_DIR": str(target), "UV_BIN": "/bin/false"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "must remain inside" in result.stderr
    assert not outside.exists()
    assert not (root / "checkpoints").exists()


def test_worldgym_can_create_a_separate_project_local_environment(tmp_path):
    root, script = fixture(tmp_path)
    target = root / "environment/verification/worldgym"
    # A stub stops at the first uv operation, before any real installation.
    uv = tmp_path / "uv-stub"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >&2\nexit 7\n')
    uv.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)],
        env={**os.environ, "WORLDGYM_ENV_DIR": str(target), "UV_BIN": str(uv)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert f"venv\n{target}\n--python\n3.11" in result.stderr
    assert not target.exists()


def test_shared_profile_is_portable_and_compatible_with_openvla():
    root = Path(__file__).resolve().parents[2]
    lock = root / "environment/locks/worldgym-openvla-py311.txt"
    pins = {}
    for line in lock.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", line)
        name, version = line.split("==")
        assert name not in pins
        pins[name] = version
    assert pins["torch"] == "2.6.0"
    assert pins["torchvision"] == "0.21.0"
    assert pins["transformers"] == "4.49.0"
    assert pins["huggingface-hub"] == "0.36.2"
    assert (
        "vlaworldeval" not in pins
    )  # Install current project, not an absolute developer checkout.
    for name in ("worldgym.sh", "vla_envs.sh"):
        assert lock.name in (root / "scripts/setup" / name).read_text()
