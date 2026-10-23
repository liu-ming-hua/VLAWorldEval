"""Import probes must be isolated and never count as GPU/model acceptance."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.validation.native_runtime import probe_native_runtime

pytestmark = pytest.mark.process


def test_real_child_probe_has_no_model_or_gpu(tmp_path):
    report = probe_native_runtime(
        sys.executable,
        root=tmp_path,
        source=Path(json.__file__).parent,
        package="json",
        imports=("math",),
    )
    assert report["ok"] is True
    assert report["cuda_initialized"] is False
    assert report["model_inference_performed"] is False
    assert report["package_python_sources"]["files"] > 0


def test_wrong_package_owner_stops_imports(tmp_path):
    report = probe_native_runtime(
        sys.executable,
        root=tmp_path,
        source=tmp_path / "not_json",
        package="json",
        imports=("math",),
    )
    assert report["ok"] is False
    assert not report["modules"]
    assert "does not resolve" in report["errors"][0]["message"]


def test_import_failure_is_structured(tmp_path):
    report = probe_native_runtime(
        sys.executable,
        root=tmp_path,
        source=Path(json.__file__).parent,
        package="json",
        imports=("missing_native_fixture_123456789",),
    )
    assert report["ok"] is False
    assert report["errors"][0]["type"] == "ModuleNotFoundError"


def test_probe_removes_parent_paths_proxies_and_gpu_visibility(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "untrusted_fixture")
    monkeypatch.setenv("HTTPS_PROXY", "http://fixture")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    def run(command, **kwargs):
        assert command[1:3] == ["-I", "-c"]
        env = kwargs["env"]
        assert "PYTHONPATH" not in env and "HTTPS_PROXY" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["XDG_CACHE_HOME"].startswith(str(tmp_path))
        return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="fixture warning")

    monkeypatch.setattr(subprocess, "run", run)
    report = probe_native_runtime(
        "fixture-python", root=tmp_path, source=tmp_path, package="fixture", imports=()
    )
    assert report["ok"] is True
    assert report["stderr"] == "fixture warning"


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired("probe", 45),
        OSError("missing interpreter"),
    ],
)
def test_probe_failure_cannot_be_passed(tmp_path, monkeypatch, failure):
    def fail(*a, **kw):
        raise failure

    monkeypatch.setattr(subprocess, "run", fail)
    report = probe_native_runtime(
        "fixture", root=tmp_path, source=tmp_path, package="fixture", imports=()
    )
    assert report["ok"] is False
    assert report["errors"]
