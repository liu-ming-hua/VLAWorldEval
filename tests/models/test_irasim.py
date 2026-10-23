"""Core-only admission/contract checks: no torch, downloads or GPUs required."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from adapters.world_models.irasim import IRASimAdapter, verify_checkpoint_provenance
from evaluation.manifests import ActionSpec, VideoSpec

pytestmark = pytest.mark.process

ROOT = Path(__file__).resolve().parents[2]


def test_raw_actions_match_official_dataset_scaler():
    raw = np.arange(30, dtype=np.float64).reshape(15, 2) / 1000
    original = raw.copy()
    actual = IRASimAdapter.validate_actions(raw)
    np.testing.assert_allclose(actual, raw * np.array([20.0, 20.0]), rtol=1e-6)
    np.testing.assert_array_equal(raw, original)
    assert actual.dtype == np.float32


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((16, 2)),
        np.zeros((15, 7)),
        np.full((15, 2), np.nan),
        np.full((15, 2), np.inf),
        np.full((15, 2), "0"),
        np.ones((15, 2), dtype=bool),
    ],
)
def test_actions_reject_invalid_protocol(value):
    with pytest.raises(ValueError, match=r"\[15,2\]"):
        IRASimAdapter.validate_actions(value)


def test_replay_timing_and_worker_contract():
    config = json.loads((ROOT / "configs/replay/irasim_native_v1.json").read_text())
    action = ActionSpec.from_dict(config["action"])
    video = VideoSpec.from_dict(config["video"])
    assert (action.horizon, action.dimension, action.advance_steps) == (15, 2, 15)
    assert video.frame_count(2) == 31
    initial = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    chunks = [np.ones((15, 2, 2, 3), dtype=np.uint8) * n for n in (1, 2)]
    output = video.assemble(initial, chunks)
    assert len(output) == 31
    np.testing.assert_array_equal(output[16:], chunks[1])
    assert config["reset_operation"] == "reset"
    assert config["instruction_conditioning"] is False
    assert config["environment"] == IRASimAdapter.environment_key


def test_missing_provenance_fails_before_pickle(tmp_path):
    checkpoint = tmp_path / "untrusted.pt"
    checkpoint.write_bytes(b"not a pickle")
    with pytest.raises(FileNotFoundError):
        verify_checkpoint_provenance(checkpoint)
    (tmp_path / "checkpoint.json").write_text(json.dumps({"repository": "other/repo"}))
    with pytest.raises(ValueError, match="provenance"):
        verify_checkpoint_provenance(checkpoint)


def downloader():
    spec = importlib.util.spec_from_file_location(
        "irasim_downloader_test", ROOT / "scripts/download/irasim.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_archive_concatenates_lazily(monkeypatch, tmp_path):
    module = downloader()
    calls = []

    def fake_snapshot(repo, directory, **kwargs):
        calls.append(kwargs)
        directory.mkdir(parents=True)
        (directory / kwargs["patterns"][0]).write_bytes(b"abc" if len(calls) == 1 else b"def")
        return {"revision": "a" * 40}

    monkeypatch.setattr(module, "download_snapshot", fake_snapshot)
    with module.ShardStream("main", tmp_path) as stream:
        assert stream.read(2) == b"ab"
        assert len(calls) == 1
        assert stream.read(3) == b"cde"
        assert len(calls) == 2
        assert calls[1]["revision"] == "a" * 40
        assert all(c["repo_type"] == "dataset" for c in calls)
        with pytest.raises(ValueError, match="Unbounded"):
            stream.read()


def test_adapter_import_does_not_require_torch():
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from adapters.world_models.irasim import IRASimAdapter; assert 'torch' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_download_forces_mirror_and_project_cache_without_mixed_case_proxies(monkeypatch, tmp_path):
    module = downloader()
    environment = {
        "HF_ENDPOINT": "https://huggingface.co",
        "HtTp_PrOxY": "http://invalid",
        "NO_PROXY": "*",
        "CUSTOM_proxy": "http://invalid",
        "KEEP": "value",
    }
    monkeypatch.setattr(module.os, "environ", environment)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    module.configure_environment()
    assert environment["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert not any(key.lower().endswith("_proxy") for key in environment)
    for key in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "UV_CACHE_DIR",
    ):
        assert Path(environment[key]).is_relative_to(tmp_path)
    assert environment["KEEP"] == "value"
