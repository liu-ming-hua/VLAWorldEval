"""Full membership and interrupted preparation never turn into a tiny test set."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.planning import cosmos
from evaluation.reporting.task_scores import _digest
from vlaworldeval.assets import MANIFEST


def selection_fixture(tmp_path, monkeypatch):
    class Generator:
        def manual_seed(self, seed):
            assert seed == 42
            return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=Generator,
        randperm=lambda count, generator: np.arange(count)))
    rows, metadata = [], {}
    for subset, count in (("success", 57639), ("failure", 14268)):
        info = tmp_path/subset/"meta/info.json"
        info.parent.mkdir(parents=True)
        info.write_text(json.dumps({"total_episodes": count, "fps": 15}))
        metadata[str(info)] = _digest(info)
        rows.extend({"id": f"cosmos20k_{subset}_{index:06d}", "subset": subset, "episode": index,
                     "length": 34, "complete16_action_chunks": 2} for index in range(round(count*.03)))
    return {"model": "cosmos3_nano_droid_fd_20k_ema", "expected_episodes": 2157,
            "episodes": rows, "metadata_sha256": metadata}


def test_complete_selection_requires_exact_order_and_bound_metadata(tmp_path, monkeypatch):
    selection = selection_fixture(tmp_path, monkeypatch)
    cosmos.validate_selection(selection, tmp_path)
    selection["episodes"][0], selection["episodes"][1] = selection["episodes"][1], selection["episodes"][0]
    with pytest.raises(ValueError, match="ordered"):
        cosmos.validate_selection(selection, tmp_path)
    selection["episodes"].reverse()
    with pytest.raises(ValueError):
        cosmos.validate_selection(selection, tmp_path)


@pytest.mark.parametrize("issue", ["small", "changed", "omitted_metadata", "chunks", "other_checkpoint"])
def test_cannot_relabel_another_cohort_as_full(tmp_path, monkeypatch, issue):
    selection = selection_fixture(tmp_path, monkeypatch)
    if issue == "small":
        selection["episodes"] = selection["episodes"][:1]
    elif issue == "changed":
        (tmp_path/"success/meta/info.json").write_text("{}")
    elif issue == "omitted_metadata":
        selection["metadata_sha256"].pop(next(iter(selection["metadata_sha256"])))
    elif issue == "chunks":
        selection["episodes"][0]["complete16_action_chunks"] = 1
    else:
        selection["model"] = "cosmos3_droid_joint_fd_best5600"
    with pytest.raises(ValueError):
        cosmos.validate_selection(selection, tmp_path)


def test_preparation_failure_keeps_denominator_and_resumes_only_bound_completed_cases(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    for relative in ("evaluation/planning/cosmos.py", "evaluation/datasets/cosmos_reference.py",
                     "evaluation/datasets/droid.py", "evaluation/datasets/video.py",
                     "adapters/actions/droid.py", "evaluation/execution/transport.py"):
        target = tmp_path/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source/relative).read_bytes())
    dataset = tmp_path/"dataset"
    selection = selection_fixture(dataset, monkeypatch)
    (dataset/MANIFEST).write_text("{}")
    audit = tmp_path/"selection.json"
    audit.write_text(json.dumps(selection))
    config = tmp_path/"config.json"
    config.write_text(json.dumps({"seed": 0, "kwargs": {"seed": 0}}))
    calls = []

    def prepare(root, data, cfg, target, **kwargs):
        calls.append(kwargs["episode"])
        if kwargs["episode"] == 1:
            raise OSError("controlled fixture interruption before writing case")
        target.mkdir(parents=True)
        for name in ("input.npz", "reference.npz", "config.json"):
            (target/name).write_bytes(b"fixture only; no model output")
        return {"chunks": 2, "source_length": 34}

    monkeypatch.setattr(cosmos, "prepare_native_case", prepare)
    monkeypatch.setattr(cosmos, "native_eligibility", lambda *_: {"fixture_only": True})
    output = tmp_path/"prepared"
    for _ in range(2):
        with pytest.raises(OSError, match="controlled"):
            cosmos.prepare_holdout(tmp_path, dataset, audit, config, output)
        state = json.loads((output/"progress.json").read_text())
        assert state["expected_cases"] == 2157 and len(state["completed"]) == 1
        assert state["status"] == "failed"
        assert not (output/"plan.json").exists() and not (output/"coverage.json").exists()
    assert calls == [0, 1, 1]
    (output/"cases/cosmos20k_success_000000_seed0/input.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        cosmos.prepare_holdout(tmp_path, dataset, audit, config, output)


def test_short_selected_episodes_remain_explicitly_accounted(tmp_path, monkeypatch):
    selection = selection_fixture(tmp_path, monkeypatch)
    row = selection["episodes"][0]
    row.update(length=4, complete16_action_chunks=0)
    cosmos.validate_selection(selection, tmp_path)
    eligible, short = cosmos.partition_selection(selection)
    assert len(eligible)+len(short) == 2157
    assert len(eligible) == 2156 and len(short) == 1
    assert short[0]["id"] == row["id"] and short[0]["available_transitions"] == 3
    assert short[0]["status"] == "no_complete_native_window"
    row.update(length=1, complete16_action_chunks=0)
    cosmos.validate_selection(selection, tmp_path)
    assert cosmos.partition_selection(selection)[1][0]["available_transitions"] == 0


def test_completed_plan_preserves_short_episode_denominator(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1]
    for relative in ("evaluation/planning/cosmos.py", "evaluation/datasets/cosmos_reference.py",
                     "evaluation/datasets/droid.py", "evaluation/datasets/video.py",
                     "adapters/actions/droid.py", "evaluation/execution/transport.py"):
        target = tmp_path/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source/relative).read_bytes())
    dataset = tmp_path/"dataset"
    selection = selection_fixture(dataset, monkeypatch)
    selection["episodes"][0].update(length=4, complete16_action_chunks=0)
    (dataset/MANIFEST).write_text("{}")
    audit = tmp_path/"selection.json"
    audit.write_text(json.dumps(selection))
    config = tmp_path/"config.json"
    config.write_text(json.dumps({"seed": 0, "kwargs": {"seed": 0}}))

    def prepare(root, data, cfg, target, **kwargs):
        target.mkdir(parents=True)
        for name in ("input.npz", "reference.npz", "config.json"):
            (target/name).write_bytes(b"synthetic protocol fixture, not a model result")
        return {"chunks": 2, "source_length": 34}

    # Test plan accounting, not expensive media decoding or journal frequency.
    replace = cosmos.replace_owned_json
    monkeypatch.setattr(cosmos, "replace_owned_json", lambda path, value:
        replace(path, value) if value["status"] == "prepared_not_evaluated" else None)
    monkeypatch.setattr(cosmos, "prepare_native_case", prepare)
    monkeypatch.setattr(cosmos, "native_eligibility", lambda *_: {"fixture_only": True})
    output = tmp_path/"prepared"
    result = cosmos.prepare_holdout(tmp_path, dataset, audit, config, output)
    plan = json.loads((output/"plan.json").read_text())
    assert result["selected_episodes"] == 2157 and result["expected_cases"] == 2156
    assert result["all_selected_episodes_accounted"] and not result["full_test_set_completed"]
    assert len(plan["cases"]) == 2156
    assert len(plan["unavailable"]["native_short_episodes"]) == 1
    assert len(plan["cases"])+len(plan["unavailable"]["native_short_episodes"]) == 2157
    assert selection["episodes"][0]["id"]+"_seed0" not in {case["id"] for case in plan["cases"]}
