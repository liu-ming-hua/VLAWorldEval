"""A debug subset cannot pass complete-split admission."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.planning.checkpoint import prepare_iws_validation
from vlaworldeval.assets import MANIFEST
from evaluation.execution.transport import write_packet, read_packet
from evaluation.reporting.task_scores import _digest


def test_capability_scope_accounts_for_every_catalogued_world_model():
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root/"configs/model_catalog.json").read_text())
    scope = json.loads((root/"configs/benchmarks/checkpoint_capability_v1.json").read_text())
    assert set(scope["models"]) == set(catalog["world_models"])
    assert scope["full_test_set_completed"] is False


def fixture(tmp_path, monkeypatch, *, mini=False, missing=False):
    from evaluation.datasets import native_reference
    from evaluation.planning import checkpoint
    dataset = tmp_path/"data"
    dataset.mkdir()
    names = [f"pusht/val/episode_{i}.hdf5" for i in range(9 if missing else 10)]
    (dataset/MANIFEST).write_text(json.dumps({"files": dict.fromkeys(names, {})}))
    monkeypatch.setattr(checkpoint, "verify_snapshot", lambda _: {
        "repository": "yixuan1999/interactive-world-sim-min-data" if mini else "yixuan1999/interactive-world-sim-data",
        "revision": "6c3010b4d8c0fae8b683856ca1c550869d5bdb04"})

    class H5:
        def __enter__(self):
            return {"obs/joint_pos": np.zeros((200, 14))}

        def __exit__(self, *args):
            pass

    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=lambda _: H5()))

    def prepare(model, source, config, target, **kwargs):
        target.mkdir(parents=True)
        write_packet(target/"input.npz", {"actions": np.zeros((192, 1, 4)), "fixture": True})
        write_packet(target/"reference.npz", {"views": {"camera": np.zeros((193, 12, 12, 3), np.uint8)},
            "frame_indices": {"camera": np.arange(193)}})
        (target/"config.json").write_text(json.dumps({"gpus": [0], "seed": 0}))

    monkeypatch.setattr(native_reference, "prepare_case", prepare)
    return dataset


def test_full_split_all_seeds_horizons_and_references_bound(tmp_path, monkeypatch):
    dataset = fixture(tmp_path, monkeypatch)
    output = tmp_path/"prepared"
    result = prepare_iws_validation(tmp_path, dataset, tmp_path/"unused_config", output)
    assert result["cases"] == 60 and result["prepared_episodes"] == 10
    plan = json.loads((output/"plan.json").read_text())
    assert len({case["id"] for case in plan["cases"]}) == 60
    assert sum(len(json.loads((output/f"plan_gpu{i}.json").read_text())["cases"]) for i in range(4)) == 60
    for case in plan["cases"]:
        target = tmp_path/case["prepared"]
        packet = read_packet(target/"input.npz")
        ref = read_packet(target/"reference.npz")
        assert ref["input_sha256"] == _digest(target/"input.npz")
        assert len(ref["views"]["camera"]) == len(packet["actions"])+1
    coverage = json.loads((output/"coverage.json").read_text())
    assert not coverage["full_test_set_completed"] and not coverage["model_inference_performed"]


@pytest.mark.parametrize("problem", ["mini", "missing", "seeds"])
def test_cannot_relabel_subset_or_invalid_seed_as_complete(tmp_path, monkeypatch, problem):
    dataset = fixture(tmp_path, monkeypatch, mini=problem == "mini", missing=problem == "missing")
    output = tmp_path/"prepared"
    with pytest.raises(ValueError):
        prepare_iws_validation(tmp_path, dataset, tmp_path/"unused", output,
                               seeds=[0, 0] if problem == "seeds" else [0])
    assert not output.exists()


def dreamdojo_metadata(tmp_path, monkeypatch):
    from evaluation.planning import checkpoint
    from evaluation.datasets.dreamdojo import INLAB_TASKS
    monkeypatch.setattr(checkpoint, "verify_snapshot", lambda _: {
        "repository": "nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1",
        "revision": "9f68429ad12535d30d1aae33b7945c79c830c8c4"})
    for task in INLAB_TASKS:
        directory = tmp_path/"In-lab_Eval"/f"gr1_unified.{task}_robot"/"meta"
        directory.mkdir(parents=True)
        info = {"chunks_size": 1000, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
        (directory/"info.json").write_text(json.dumps(info))
        (directory/"episodes.jsonl").write_text('\n'.join(json.dumps(x) for x in [
            {"episode_index": 0, "length": 100}, {"episode_index": 1, "length": 108}]))
    return tmp_path


def test_dreamdojo_exact_uniform_selection_preserves_task_and_step_order(tmp_path, monkeypatch):
    from evaluation.planning.checkpoint import dreamdojo_eval100_selection
    root = dreamdojo_metadata(tmp_path, monkeypatch)
    result = dreamdojo_eval100_selection(root)
    assert result["expected_samples"] == len(result["samples"]) == 100
    assert [(r["episode"], r["start"]) for r in result["samples"][:3]] == [(0, 0), (0, 1), (1, 0)]
    assert result["samples"][9]["start"] == 8
    assert result["samples"][10]["task"] == "pnp_cucumber"
    assert all(r["chunks"] == 4 for r in result["samples"])
    assert result["protocol"]["restrict_len"] == 100


@pytest.mark.parametrize("issue", ["short", "duplicate", "escape"])
def test_dreamdojo_selection_rejects_bad_metadata_instead_of_replacing_samples(tmp_path, monkeypatch, issue):
    from evaluation.planning.checkpoint import dreamdojo_eval100_selection
    root = dreamdojo_metadata(tmp_path, monkeypatch)
    directory = root/"In-lab_Eval/gr1_unified.pnp_handover_plate_robot/meta"
    if issue == "short":
        (directory/"episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 98}))
    elif issue == "duplicate":
        line = json.dumps({"episode_index": 0, "length": 1000})
        (directory/"episodes.jsonl").write_text(line+'\n'+line)
    else:
        info = json.loads((directory/"info.json").read_text())
        info["data_path"] = "../../outside.parquet"
        (directory/"info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError):
        dreamdojo_eval100_selection(root)
