"""Core-only download tests: shared locks/hashes and honest data boundaries."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("playworld_download", ROOT / "scripts/download/playworld.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_network_configuration_removes_all_proxy_cases(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "fixture")
    monkeypatch.setenv("http_proxy", "fixture")
    monkeypatch.setenv("All_Proxy", "fixture")
    download.configure_network(tmp_path)
    import os
    assert not any(key.lower().endswith("_proxy") for key in os.environ)
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert os.environ["HF_HOME"].startswith(str(tmp_path))


def test_model_and_data_use_shared_fixed_revision_snapshot(tmp_path, monkeypatch):
    import vlaworldeval.assets as assets

    calls = []
    monkeypatch.setattr(assets, "download_snapshot", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(download, "reuse_base", lambda root, name, repo: root / name)
    monkeypatch.setattr(sys, "argv", ["download", "--root", str(tmp_path), "--preview"])
    download.main()
    assert len(calls) == 3
    assert calls[0][0][0] == "tennyyyin/open-world-checkpoints"
    assert calls[0][1]["patterns"] == ["wm/checkpoint-120000.pt", "action_adapter/model2_15_9.pth"]
    assert calls[0][1]["revision"] == download.MODEL_REVISION
    assert calls[1][1]["repo_type"] == "dataset"
    assert calls[2][1]["patterns"] == download.PREVIEW_PATTERNS
    assert "checkpoint-100000.pt" not in calls[2][1]["patterns"]


def test_config_only_base_cannot_be_silently_reused(tmp_path, monkeypatch):
    source = tmp_path / "checkpoints/world_models/ctrl_world/clip"
    source.mkdir(parents=True)
    (source / "config.json").write_text("{}")
    remote = SimpleNamespace(rfilename="config.json", size=2, blob_id="bad", lfs=None)
    api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40, siblings=[remote]))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: api))
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    with pytest.raises(ValueError, match="missing required components"):
        download.reuse_base(tmp_path, "clip", "openai/clip-vit-base-patch32")
    assert not (tmp_path / "checkpoints/world_models/playworld/clip").exists()


def test_broken_symlink_fails_preserving_target(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: None))
    target = tmp_path / "checkpoints/world_models/playworld/clip"
    target.parent.mkdir(parents=True)
    target.symlink_to("missing")
    with pytest.raises(ValueError, match="Broken base symlink"):
        download.reuse_base(tmp_path, "clip", "openai/clip-vit-base-patch32")
    assert target.is_symlink()


def test_existing_unmarked_target_still_checks_official_hash(tmp_path, monkeypatch):
    source = tmp_path / "checkpoints/world_models/playworld/clip"
    source.mkdir(parents=True)
    filenames = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "special_tokens_map.json",
                 "vocab.json", "merges.txt"]
    for name in filenames:
        (source / name).write_bytes(b"{}")
    remotes = [SimpleNamespace(rfilename=name, size=2, blob_id="wrong", lfs=None) for name in filenames]
    api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40, siblings=remotes))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: api))
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    with pytest.raises(ValueError, match="checksum mismatch"):
        download.reuse_base(tmp_path, "clip", "openai/clip-vit-base-patch32")
    assert not (source.parent / "clip.reference.json").exists()


def test_environment_snapshot_does_not_mask_failed_dependency_check(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("playworld_replay", ROOT / "scripts/replay/playworld.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    results = iter([SimpleNamespace(returncode=1, stdout="fixture stdout", stderr="fixture dependency mismatch"),
                    SimpleNamespace(returncode=1, stdout="", stderr="no GPU fixture")])
    monkeypatch.setattr(replay.subprocess, "run", lambda *a, **kw: next(results))
    monkeypatch.setattr(replay.metadata, "distributions", lambda: [SimpleNamespace(metadata={"Name":"fixture"},version="1")])
    output = tmp_path / "snapshot"
    report = replay.environment_snapshot(output)
    saved = json.loads((output / "environment.json").read_text())
    assert report["dependency_check"]["passed"] is False
    assert saved["dependency_check"]["exit_code"] == 1
    assert saved["dependency_check"]["stderr"] == "fixture dependency mismatch"
    assert saved["packages"] == {"fixture":"1"}
