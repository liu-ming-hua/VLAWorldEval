"""Resume full cohorts without resampling or duplicate inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.execution.transport import write_packet
from evaluation.planning.native_reference import prepare_reference_remainder


def fixture(tmp_path, monkeypatch):
    from evaluation.reporting import checkpoint_metrics
    cases = []
    for index in range(5):
        prepared = tmp_path/f"datasets/case{index}"
        prepared.mkdir(parents=True)
        (prepared/"config.json").write_text("{}")
        write_packet(prepared/"input.npz", {"actions": np.zeros((index+1, 1, 2))})
        write_packet(prepared/"reference.npz", {"fixture": True})
        cases.append({"id": f"case{index}", "prepared": str(prepared.relative_to(tmp_path))})
    plan = tmp_path/"plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "scope": "full population",
                               "cases": cases, "unavailable": {"physical_branches": "not released"}}))
    saved = tmp_path/"runs/case0"
    saved.mkdir(parents=True)
    (saved/"prediction.mp4").write_bytes(b"saved prediction")
    sha = hashlib.sha256((saved/"prediction.mp4").read_bytes()).hexdigest()
    summary = saved/"summary.json"
    summary.write_text(json.dumps({"files": {"prediction.mp4": sha}}))
    monkeypatch.setattr(checkpoint_metrics, "select_replays", lambda *a, **k:
                        ({"case0": (summary, tmp_path/"datasets/case0/reference.npz")}, []))
    return plan, saved


def test_remainder_preserves_full_population_and_disjoint_shards(tmp_path, monkeypatch):
    plan, _ = fixture(tmp_path, monkeypatch)
    result = prepare_reference_remainder(tmp_path, plan, ["runs"], "pending", [0, 1])
    assert result["planned_cases"] == 5 and result["pending_cases"] == 4
    assert set(result["reused_cases"]) == {"case0"}
    assert not result["inference_performed"] and not result["full_metrics_completed"]
    ids = []
    for shard in result["shards"]:
        frozen = json.loads((tmp_path/shard["plan"]).read_text())
        ids.extend(case["id"] for case in frozen["cases"])
        assert frozen["unavailable"] == {"physical_branches": "not released"}
    assert len(ids) == len(set(ids)) == 4
    assert set(ids) | set(result["reused_cases"]) == {f"case{i}" for i in range(5)}
    assert json.loads(plan.read_text())["scope"] == "full population"
    with pytest.raises(FileExistsError):
        prepare_reference_remainder(tmp_path, plan, ["runs"], "pending", [0])


def test_remainder_rejects_modified_completed_output(tmp_path, monkeypatch):
    plan, saved = fixture(tmp_path, monkeypatch)
    (saved/"prediction.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        prepare_reference_remainder(tmp_path, plan, ["runs"], "pending", [0])
    assert not (tmp_path/"pending").exists()


@pytest.mark.parametrize("gpus", [[], [0, 0], [-1], [True]])
def test_remainder_requires_valid_explicit_devices(tmp_path, monkeypatch, gpus):
    plan, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="target GPUs"):
        prepare_reference_remainder(tmp_path, plan, ["runs"], "pending", gpus)


def test_remainder_cli_dispatch_is_explicit():
    from cli.parser import build_parser
    args = build_parser(Path.cwd()).parse_args([
        "native-study", "remainder", "--plan", "full.json", "--directory", "pending",
        "--source-directory", "old1", "--source-directory", "old2", "--gpus", "0", "1"])
    assert args.operation == "remainder" and len(args.source_directory) == 2
