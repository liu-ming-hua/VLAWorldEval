import json
from pathlib import Path

import pytest

from vlaworldeval.demo import run_demo
from vlaworldeval.paths import project_root, require_asset_project


def test_cpu_demo_uses_shared_loop_and_non_one_to_one_video(tmp_path):
    report = run_demo(tmp_path / "demo", chunks=3)
    assert report["audit"]["ok"] and not report["task_success_evaluated"]
    metadata = json.loads((tmp_path / "demo/metadata.json").read_text())
    assert metadata["num_video_frames"] == 10 and metadata["video_fps"] == 30
    assert metadata["pair_manifest"]["world_model"]["action"]["horizon"] == 2
    assert (tmp_path / "demo/video_diagnostics.json").is_file()
    with pytest.raises(FileExistsError):
        run_demo(tmp_path / "demo")


def test_asset_root_is_explicit_and_checked(tmp_path, monkeypatch):
    monkeypatch.setenv("VLAWORLDEVAL_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    with pytest.raises(ValueError, match="set VLAWORLDEVAL_ROOT"):
        require_asset_project(project_root())
    monkeypatch.setenv("VLAWORLDEVAL_ROOT", "relative")
    with pytest.raises(ValueError, match="absolute"):
        project_root()


def test_default_dependencies_do_not_install_model_stacks():
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    metadata = tomllib.loads(Path("pyproject.toml").read_text())
    deps = metadata["project"]["dependencies"]
    assert not any(dep.startswith(("torch", "jax", "transformers", "tensorflow", "openai")) for dep in deps)
    assert "worldgym" in metadata["project"]["optional-dependencies"]


def test_invalid_asset_root_has_structured_cli_error(tmp_path):
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "cli", "models"],
                            env={**os.environ, "VLAWORLDEVAL_ROOT": "relative"},
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"]["code"] == "input_error"
    assert "Traceback" not in result.stderr


def test_capabilities_do_not_claim_all_combinations_or_execution():
    from evaluation.planning.capabilities import capabilities
    report = capabilities(Path(__file__).resolve().parents[1])
    assert len(report["executable_pairs"]) == 16
    assert not report["arbitrary_composition_supported"] and not report["execution_performed"]
    assert "a2world" in report["catalogued_without_unified_pair"]["world_models"]


def test_missing_droid_dependency_is_an_actionable_data_error(tmp_path, monkeypatch):
    import builtins
    from evaluation.datasets.droid import load_droid_episode
    original = builtins.__import__
    def importing(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ModuleNotFoundError("pyarrow")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ValueError, match="CORE_DROID=1"):
        load_droid_episode(tmp_path, 0, 0, 1)


def test_installed_worker_stages_only_framework_code(tmp_path):
    from evaluation.execution.client import worker_code_path
    site = tmp_path / "site-packages"
    site.mkdir()
    for name in ("adapters", "cli", "evaluation", "models", "vlaworldeval", "numpy"):
        (site / name).mkdir()
        (site / name / "__init__.py").write_text("")
    worker = tmp_path / "worker"
    worker.mkdir()
    stage = worker_code_path(worker, site)
    assert stage == worker / "framework_code"
    assert not (stage / "numpy").exists()
    assert not (stage / "models").exists()
    assert (stage / "evaluation/__init__.py").is_file()
