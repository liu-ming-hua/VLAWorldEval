import json
import hashlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from evaluation.planning.inventory import validate_datasets
from vlaworldeval.assets import MANIFEST, download_snapshot, file_record, verify_snapshot
from vlaworldeval.workspace import configure_suite, initialize_workspace, direct_network_environment
from vlaworldeval.paths import gigaworld1_source

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_has_portable_declarations_not_machine_assets(tmp_path):
    destination = tmp_path / "workspace"
    report = initialize_workspace(destination)
    assert not report["assets_downloaded"] and not report["environments_created"]
    registry = (destination / "environment/registry.toml").read_text()
    assert "/data1/liuming" not in registry
    assert 'status = "ready"' not in registry
    assert not list((destination / "checkpoints").rglob("*.safetensors"))
    assert (destination / "configs/benchmarks/native_domains_v1.json").is_file()
    assert (destination / "environment/requirements/video_metrics.txt").is_file()
    result = subprocess.run([sys.executable, "-m", "cli", "validate"], cwd=ROOT,
                            env={**os.environ, "VLAWORLDEVAL_ROOT": str(destination)},
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "16 executable pairs" in result.stdout
    assert all(entry["status"] != "present" for entry in validate_datasets(destination))


def test_workspace_protects_existing_files_and_symlinks(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    file = destination / "user.txt"
    file.write_text("preserve")
    with pytest.raises(ValueError, match="empty"):
        initialize_workspace(destination)
    assert file.read_text() == "preserve"
    link = tmp_path / "link"
    link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink"):
        initialize_workspace(link)


def test_configure_suite_validates_and_does_not_overwrite(tmp_path):
    workspace = tmp_path / "workspace"
    initialize_workspace(workspace)
    source = workspace / "configs/experiments/openvla_worldgym_v1.json"
    before = source.read_bytes()
    destination = workspace / "configs/experiments/my_machine.json"
    report = configure_suite(source, destination, root=workspace, vla_gpus=[2],
                             world_model_gpus=[3], budget=10, vla_budget=8, chunks=6)
    value = json.loads(destination.read_text())
    assert value["runtime"]["vla_gpus"] == [2]
    assert value["runtime"]["world_model_gpus"] == [3]
    assert all(e["chunks"] == 6 for e in value["experiments"])
    assert report["executed"] is False
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="already exists"):
        configure_suite(source, destination, root=workspace)


@pytest.mark.parametrize("kwargs", [{"vla_gpus": [0, 0]}, {"budget": float("nan")},
                                    {"chunks": 0}, {"world_model_gpus": [0, 1]}])
def test_configure_rejects_invalid_hardware_and_removes_own_invalid_output(tmp_path, kwargs):
    initialize_workspace(tmp_path / "workspace")
    root = tmp_path / "workspace"
    destination = root / "configs/experiments/invalid.json"
    with pytest.raises(ValueError):
        configure_suite(root / "configs/experiments/openvla_worldgym_v1.json",
                        destination, root=root, **kwargs)
    assert not destination.exists()


def test_direct_network_caches_and_proxies(tmp_path, monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(key, "http://fake-proxy")
    env = direct_network_environment(tmp_path)
    assert not any(key.lower().endswith("_proxy") for key in env if key.lower() != "no_proxy")
    assert env["HF_HOME"].startswith(str(tmp_path))
    assert env["UV_CACHE_DIR"].startswith(str(tmp_path))
    for key in ("XDG_CACHE_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "MODELSCOPE_CACHE", "TRANSFORMERS_CACHE",
                "MPLCONFIGDIR", "TFHUB_CACHE_DIR", "NUMBA_CACHE_DIR", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR", "CUDA_CACHE_PATH", "JAX_COMPILATION_CACHE_DIR"):
        assert Path(env[key]).is_relative_to(tmp_path)


def make_manifest(root):
    root.mkdir()
    (root / "weights.bin").write_bytes(b"example weights")
    report = {"schema_version": 1, "repository": "example/model", "repo_type": "model",
              "requested_revision": "main", "revision": "a" * 40, "allow_patterns": [],
              "files": {"weights.bin": file_record(root / "weights.bin")}}
    (root / MANIFEST).write_text(json.dumps(report))
    return report


def test_asset_verification_is_offline_and_detects_corruption(tmp_path):
    directory = tmp_path / "model"
    make_manifest(directory)
    assert verify_snapshot(directory)["files_verified"] == 1
    assert download_snapshot("example/model", directory, root=tmp_path)["ok"]
    with pytest.raises(ValueError, match="identity/revision"):
        download_snapshot("example/other", directory, root=tmp_path)
    (directory / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupted"):
        verify_snapshot(directory)


def test_selected_verification_is_explicitly_not_full_snapshot(tmp_path):
    directory = tmp_path/"data"
    report = make_manifest(directory)
    (directory/"other.bin").write_bytes(b"second file")
    report["files"]["other.bin"] = file_record(directory/"other.bin")
    (directory/MANIFEST).write_text(json.dumps(report))
    (directory/"other.bin").unlink()
    selected = verify_snapshot(directory, names=["weights.bin"])
    assert selected["files_verified"] == 1 and selected["snapshot_files_declared"] == 2
    assert selected["verification_scope"] == "selected_files_only"
    with pytest.raises(ValueError, match="corrupted"):
        verify_snapshot(directory)
    for names in (["unknown"], [], ["weights.bin", "weights.bin"]):
        with pytest.raises(ValueError):
            verify_snapshot(directory, names=names)


@pytest.mark.parametrize("mode", ["exact_names", "glob"])
def test_large_exact_manifest_fetches_only_resolved_immutable_files(tmp_path, monkeypatch, mode):
    names = [f"video_{i:04d}.bin" for i in range(513 if mode == "exact_names" else 100)]
    selected = set(names)
    calls = []
    payload = b"native payload"
    digest = hashlib.sha256(payload).hexdigest()

    class API:
        def __init__(self, endpoint):
            pass

        def repo_info(self, **kwargs):
            return types.SimpleNamespace(sha="b"*40, siblings=[
                types.SimpleNamespace(rfilename=name, size=len(payload), lfs={"sha256": digest})
                for name in [*names, "unselected.bin"]])

    def fetch(**kwargs):
        assert kwargs["revision"] == "b"*40 and kwargs["endpoint"] == "https://hf-mirror.com"
        assert kwargs["filename"] in selected
        calls.append(kwargs["filename"])
        Path(kwargs["local_dir"], kwargs["filename"]).write_bytes(payload)

    def wrong_branch(**kwargs):
        pytest.fail("Large explicit manifests must not invoke Hub's quadratic snapshot filter")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        HfApi=API, snapshot_download=wrong_branch, hf_hub_download=fetch))
    from vlaworldeval import assets
    monkeypatch.setattr(assets, "fnmatch", types.SimpleNamespace(translate=assets.fnmatch.translate,
        fnmatchcase=lambda *args: pytest.fail("Exact-name manifests must not recompile fnmatch patterns")))
    result = download_snapshot("example/native", tmp_path/"data", root=tmp_path,
                               revision="b"*40, patterns=names if mode == "exact_names" else ["video_*.bin"])
    assert result["files_verified"] == len(calls) == len(selected)
    assert not (tmp_path/"data/unselected.bin").exists()


def test_asset_manifest_cannot_escape_directory(tmp_path):
    directory = tmp_path / "model"
    report = make_manifest(directory)
    report["files"] = {"../outside.bin": report["files"]["weights.bin"]}
    (directory / MANIFEST).write_text(json.dumps(report))
    with pytest.raises(ValueError, match="leaves"):
        verify_snapshot(directory)


def test_snapshot_manifest_symlink_is_rejected_without_reading_target(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    outside = tmp_path / "private.json"
    outside.write_text("this is not a model manifest")
    (directory / MANIFEST).symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        verify_snapshot(directory)
    with pytest.raises(ValueError, match="symlink"):
        download_snapshot("example/model", directory, root=tmp_path)
    assert outside.read_text() == "this is not a model manifest"


def test_download_resolves_commit_and_does_not_mark_incomplete_snapshot(tmp_path, monkeypatch):
    calls = []

    class API:
        def __init__(self, endpoint):
            assert endpoint == "https://hf-mirror.com"

        def repo_info(self, **kwargs):
            assert kwargs["files_metadata"] is True
            return types.SimpleNamespace(sha="b" * 40, siblings=[
                types.SimpleNamespace(rfilename="config.json", size=2, lfs=None,
                                      blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest()),
                types.SimpleNamespace(rfilename="weights.bin", size=7, lfs={"sha256": "c" * 64})])

        def list_repo_files(self, **kwargs):
            assert kwargs["revision"] == "b" * 40
            return ["config.json", "weights.bin"]

    def incomplete(**kwargs):
        calls.append(kwargs)
        assert kwargs["revision"] == "b" * 40
        assert kwargs["endpoint"] == "https://hf-mirror.com"
        assert kwargs["cache_dir"] == str(tmp_path / ".cache/huggingface/hub")
        assert kwargs["allow_patterns"] is None
        Path(kwargs["local_dir"], "config.json").write_text("{}")

    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        HfApi=API, snapshot_download=incomplete))
    directory = tmp_path / "model"
    with pytest.raises(ValueError, match="incomplete"):
        download_snapshot("example/model", directory, root=tmp_path)
    assert calls and not (directory / MANIFEST).exists()


def test_hf_publication_interruption_retains_data_and_allows_retry(tmp_path, monkeypatch):
    import vlaworldeval.assets as module
    class API:
        def __init__(self, **kwargs):
            pass
        def repo_info(self, **kwargs):
            return types.SimpleNamespace(sha="b" * 40, siblings=[
                types.SimpleNamespace(rfilename="config.json", size=2, lfs=None,
                                      blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest())])
    def snapshot(**kwargs):
        Path(kwargs["local_dir"], "config.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=API, snapshot_download=snapshot))
    directory = tmp_path / "model"
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("simulated commit interruption")
    with monkeypatch.context() as scope:
        scope.setattr(module, "publish_json_exclusive", interrupted)
        with pytest.raises(KeyboardInterrupt):
            download_snapshot("example/model", directory, root=tmp_path)
    assert not (directory / MANIFEST).exists()
    assert (directory / "config.json").read_text() == "{}"
    assert download_snapshot("example/model", directory, root=tmp_path)["ok"]


def test_source_bootstrap_preserves_non_git_user_directory(tmp_path):
    source = tmp_path / "user_source"
    source.mkdir()
    (source / "important.py").write_text("preserve")
    result = subprocess.run(["bash", "-c", 'source "$1"; ensure_source "$2" "$3" "$4"',
                             "test", str(ROOT / "scripts/setup/common.sh"), str(source),
                             "https://invalid.example/repo.git", "a" * 40],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "not deleting" in result.stderr
    assert (source / "important.py").read_text() == "preserve"


def test_hf_download_never_overwrites_different_legacy_weights(tmp_path, monkeypatch):
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "weights.bin").write_bytes(b"user-owned")
    remote = types.SimpleNamespace(rfilename="weights.bin", size=10,
                                   lfs={"sha256": "a" * 64})
    class API:
        def __init__(self, **kwargs):
            pass
        def repo_info(self, **kwargs):
            return types.SimpleNamespace(sha="b" * 40, siblings=[remote])
    def forbidden(**kwargs):
        pytest.fail("snapshot_download must not run before legacy checks pass")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        HfApi=API, snapshot_download=forbidden))
    with pytest.raises(ValueError, match="preserving"):
        download_snapshot("example/model", directory, root=tmp_path)
    assert (directory / "weights.bin").read_bytes() == b"user-owned"
    assert not (directory / MANIFEST).exists()


def test_bad_core_option_fails_before_installation():
    result = subprocess.run(["bash", "scripts/setup/core.sh"], cwd=ROOT,
                            env={**os.environ, "CORE_DEV": "typo"},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "no environment changes" in result.stderr


def test_giga_repository_override_is_explicit_and_validated(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGAWORLD1_REPO", str(tmp_path))
    assert gigaworld1_source() == tmp_path.resolve()
    monkeypatch.setenv("GIGAWORLD1_REPO", "relative")
    with pytest.raises(ValueError, match="absolute"):
        gigaworld1_source()


def test_new_public_guides_have_resolvable_local_links():
    import re
    for name in ("en/quickstart.md", "zh/quickstart.md", "en/release.md", "zh/release.md"):
        path = ROOT / "docs" / name
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not target.startswith(("https://", "http://", "#")):
                assert (path.parent / target.split("#")[0]).exists(), f"Broken link in {name}: {target}"


def test_environment_override_preserves_registry_and_rejects_unknown_keys(tmp_path, monkeypatch):
    from adapters.environment import load_environment_registry
    source = ROOT / "environment/registry.toml"
    original = source.read_bytes()
    overrides = tmp_path / "environments.json"
    overrides.write_text(json.dumps({"schema_version": 1, "paths": {"vla.openvla": str(tmp_path)}}))
    monkeypatch.setenv("VLAWORLDEVAL_ENVIRONMENTS", str(overrides))
    assert load_environment_registry()["vla.openvla"].path == tmp_path
    assert source.read_bytes() == original
    overrides.write_text(json.dumps({"schema_version": 1, "paths": {"unknown": str(tmp_path)}}))
    with pytest.raises(ValueError, match="Unknown override"):
        load_environment_registry()


@pytest.mark.parametrize("bad", [None, [], {"schema_version": True, "paths": {}},
                                  {"schema_version": 1, "paths": {"vla.openvla": "relative"}}])
def test_bad_environment_maps_are_actionable_errors(tmp_path, monkeypatch, bad):
    from adapters.environment import load_environment_registry
    source = tmp_path / "environments.json"
    source.write_text(json.dumps(bad))
    monkeypatch.setenv("VLAWORLDEVAL_ENVIRONMENTS", str(source))
    with pytest.raises(ValueError):
        load_environment_registry()
