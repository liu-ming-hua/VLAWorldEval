"""Downloaded runtime trees and raw samples must not enter a framework commit."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.process


def test_new_model_assets_are_ignored_but_public_declarations_remain_visible(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is only required for this source-boundary check")
    root = Path(__file__).resolve().parents[2]
    subprocess.run([git, "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_bytes((root / ".gitignore").read_bytes())
    ignored = {
        "third_party/dreamdojo/README.md",
        "third_party/iws/pyproject.toml",
        "third_party/iws_dependencies/yixuan_utilities/setup.py",
        "third_party/playworld/openworld/__init__.py",
        "third_party/dreamtrue/wmvideo/run.py",
        "third_party/irasim/models/irasim.py",
        "datasets/dreamtrue/raw/droid/episode/trajectory.h5",
        "datasets/dreamtrue/raw/droid/episode/camera.mp4",
        "datasets/dreamtrue/condition/condition-mini.tar.gz",
        "datasets/dreamdojo/prepared/native.packet",
        "datasets/iws/pusht/val/episode.hdf5",
        "datasets/playworld/benchmark/case/initialization.yaml",
        "datasets/irasim/languagetable/sample.json",
        "datasets/cosmos3_droid/success/data/episode.parquet",
        "checkpoints/world_models/dreamdojo/reason/config.json",
        "environment/world_models/dreamtrue/pyvenv.cfg",
    }
    declarations = {
        "third_party/README.md",
        "datasets/README.md",
        "datasets/iws/manifest.json",
        "datasets/dreamdojo/manifest.json",
        "datasets/dreamtrue/manifest.json",
        "datasets/playworld/manifest.json",
        "datasets/irasim/manifest.json",
        "datasets/iws/README.md",
        "environment/requirements/iws.txt",
        "adapters/world_models/iws.py",
        "scripts/setup/iws.sh",
    }
    paths = sorted(ignored | declarations)
    result = subprocess.run(
        [git, "-C", str(tmp_path), "check-ignore", "--no-index", "--stdin"],
        input="\n".join(paths) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert set(result.stdout.splitlines()) == ignored
