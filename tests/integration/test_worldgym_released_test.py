"""Entire tiny released test is not a substitute for full paper validation."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from evaluation.execution.transport import read_packet
from evaluation.planning.checkpoint import prepare_worldgym_released_test


def assets(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2]
    original = source / "datasets/worldgym/official/bridge/test"
    if not original.is_dir():
        pytest.skip("Optional original release data absent")
    dataset = tmp_path / "sample/test"
    dataset.mkdir(parents=True)
    for name in ("000000002.mp4", "000000002.npz"):
        shutil.copyfile(original / name, dataset / name)
    config = tmp_path / "config.json"
    shutil.copyfile(source / "configs/replay/worldgym_bridge_single_step_v1.json", config)
    from evaluation.datasets import native_reference

    monkeypatch.setattr(native_reference, "project_root", lambda: tmp_path)
    return dataset, config


def test_entire_released_sample_retains_last_real_action_and_one_episode_denominator(
    tmp_path, monkeypatch
):
    dataset, config = assets(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    result = prepare_worldgym_released_test(tmp_path, dataset, config, output)
    assert result["released_test_episodes"] == 1 and result["cases"] == 3
    plan = json.loads((output / "plan.json").read_text())
    expected = np.load(dataset / "000000002.npz")["arr_0"][1:]
    for seed, case in enumerate(plan["cases"]):
        target = tmp_path / case["prepared"]
        packet, reference = read_packet(target / "input.npz"), read_packet(target / "reference.npz")
        np.testing.assert_array_equal(packet["actions"][:, 0], expected)
        assert packet["actions"].shape == (21, 1, 10)
        assert len(reference["views"]["primary"]) == 22
        assert json.loads((target / "config.json").read_text())["seed"] == seed
    coverage = json.loads((output / "coverage.json").read_text())
    assert coverage["expected_released_test_episodes"] == 1
    assert not coverage["full_paper_test_completed"] and not coverage["model_inference_performed"]


@pytest.mark.parametrize("issue", ["corrupt", "extra", "duplicate_seed", "short_config"])
def test_changed_release_or_ambiguous_sampling_never_creates_a_full_test_claim(
    tmp_path, monkeypatch, issue
):
    dataset, config = assets(tmp_path, monkeypatch)
    seeds = [0, 0] if issue == "duplicate_seed" else [0]
    if issue == "corrupt":
        (dataset / "000000002.npz").write_bytes(b"changed fixture")
    elif issue == "extra":
        (dataset / "000000003.mp4").write_bytes(b"undeclared fixture")
    elif issue == "short_config":
        data = json.loads(config.read_text())
        data["action"]["horizon"] = 4
        config.write_text(json.dumps(data))
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        prepare_worldgym_released_test(tmp_path, dataset, config, output, seeds=seeds)
    assert not output.exists()
