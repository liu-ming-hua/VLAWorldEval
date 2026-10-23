"""Optional native-domain parity; requires separately acquired assets or runtimes."""

import shutil
from pathlib import Path

import pytest

from vlaworldeval.worldgym_runtime import prepare_runtime, runtime_directory, verify_runtime


def test_worldgym_runtime_preparation_preserves_network_and_is_idempotent(tmp_path):
    source = Path("third_party/world-model-eval")
    if not source.is_dir():
        pytest.skip("Optional upstream checkout not shipped with the framework")
    report = prepare_runtime(tmp_path, source, runtime_directory(tmp_path))
    assert not report["redistribution_permission_verified"]
    assert prepare_runtime(tmp_path, source, runtime_directory(tmp_path)) == report
    package = runtime_directory(tmp_path) / "worldgym_runtime"
    for name in ("model.py", "diffusion.py"):
        assert (package / name).read_bytes() == (
            source / "src/world_model_eval" / name
        ).read_bytes()
    assert 'map_location="cpu"' in (package / "world_model.py").read_text()
    assert 'os.environ["WORLDGYM_VAE_PATH"]' in (package / "vae.py").read_text()
    (package / "model.py").write_text("changed")
    with pytest.raises(ValueError, match="integrity"):
        verify_runtime(tmp_path)
    with pytest.raises(ValueError, match="integrity"):
        prepare_runtime(tmp_path, source, runtime_directory(tmp_path))
    assert (package / "model.py").read_text() == "changed"


def test_runtime_rejects_modified_upstream_without_overwriting(tmp_path):
    source = Path("third_party/world-model-eval/src/world_model_eval")
    if not source.is_dir():
        pytest.skip("Optional upstream checkout not shipped with the framework")
    destination = tmp_path / "source/src/world_model_eval"
    destination.mkdir(parents=True)
    for name in ("model.py", "diffusion.py", "vae.py", "world_model.py"):
        shutil.copyfile(source / name, destination / name)
    (destination / "model.py").write_text("modified upstream")
    with pytest.raises(ValueError, match="pinned revision"):
        prepare_runtime(tmp_path, tmp_path / "source", runtime_directory(tmp_path))
    assert not runtime_directory(tmp_path).exists()
