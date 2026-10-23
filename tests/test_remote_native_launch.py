"""Remote campaign resource boundaries must not change scientific coverage."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "launch_remote_native", Path(__file__).resolve().parents[1] / "scripts/ops/launch_remote_native.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reject_handoff_outside_four_authorized_gpus(tmp_path):
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    (handoff / "handoff.json").write_text(json.dumps({
        "source_root": "/original", "pending": [], "gpus": [0, 4]}))
    with pytest.raises(ValueError, match="restricted"):
        MODULE.launch(tmp_path, "irasim", handoff)


def test_cosmos_entire_cohort_uses_one_four_gpu_group(tmp_path, monkeypatch):
    corpus = tmp_path / "datasets/cosmos3_droid_prepared/full_holdout_v2"
    corpus.mkdir(parents=True)
    cases = [{"id": str(i), "prepared": f"datasets/case{i}"} for i in range(2145)]
    plan = {"schema_version": 1, "scope": "full", "cases": cases, "unavailable": {"short": 12}}
    (corpus / "plan.json").write_text(json.dumps(plan))
    (corpus / "progress.json").write_text(json.dumps({"completed": {
        c["id"]: {"chunks": 2, "sha256": {"input.npz": "verified"}} for c in cases}}))
    verified, prepared, commands = [], [], []
    monkeypatch.setattr(MODULE, "_digest", lambda p: verified.append(p) or "verified")
    monkeypatch.setattr(MODULE, "prepare_reference_study", lambda *args: prepared.append(args))

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if command[1] == "has-session" else 0)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    MODULE.launch(tmp_path, "cosmos")
    assert len(verified) == 2145
    assert len(prepared) == 1 and prepared[0][1] == corpus / "plan.json"
    starts = [c for c in commands if c[1] == "new-session"]
    assert len(starts) == 1
    assert "--gpus 0 1 2 3" in starts[0][-1]
    assert "quad1" not in starts[0][-1]
    assert json.loads((corpus / "plan.json").read_text()) == plan


def test_irasim_can_launch_one_verified_shard_without_other_shards_arriving(tmp_path, monkeypatch):
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    jobs = [{"id": str(i), **{key: f"/original/datasets/case{i}/{key}.npz"
                             for key in ("input", "config", "reference")},
             **{key + "_sha256": "verified" for key in ("input", "config", "reference")}}
            for i in range(4)]
    (handoff / "handoff.json").write_text(json.dumps({
        "source_root": "/original", "pending": jobs, "gpus": [0, 1, 2, 3]}))
    (handoff / "gpu1_plan.json").write_text(json.dumps({
        "cases": [{"id": "1", "prepared": "datasets/case1"}]}))
    verified, prepared, commands = [], [], []
    monkeypatch.setattr(MODULE, "_digest", lambda p: verified.append(p) or "verified")
    monkeypatch.setattr(MODULE, "prepare_reference_study", lambda *args: prepared.append(args))

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if command[1] == "has-session" else 0)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    MODULE.launch(tmp_path, "irasim", handoff, shard="gpu1")
    assert len(verified) == 3 and all(p.parent.name == "case1" for p in verified)
    assert len(prepared) == 1
    assert "--gpu 1 --memory-gib 12" in commands[-1][-1]


def test_cosmos_budget_binds_all400_cases_and_uses_fresh_namespace(tmp_path, monkeypatch):
    cases = [{"id": str(i), "prepared": f"datasets/case{i}"} for i in range(400)]
    plan = tmp_path/"plan.json"
    plan.write_text(json.dumps({"cases": cases}))
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"plan_sha256": "verified", "cases": {
        c["id"]: {"sha256": {name: "verified" for name in
                              ("input.npz", "config.json", "reference.npz")}} for c in cases}}))
    verified, commands = [], []
    monkeypatch.setattr(MODULE, "_digest", lambda p: verified.append(p) or "verified")
    monkeypatch.setattr(MODULE, "prepare_reference_study", lambda *args: None)

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if command[1] == "has-session" else 0)

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    MODULE.launch(tmp_path, "cosmos", native_plan=plan, manifest=manifest,
                  campaign_directory=tmp_path/"new_campaign")
    assert len(verified) == 1201
    assert "vwe-budget400-cosmos-quad0" in commands[-1]
    assert "new_campaign/cosmos_runs/quad0" in commands[-1][-1]
    assert "--gpus 0 1 2 3" in commands[-1][-1]


@pytest.mark.parametrize("state,raises", [("completed", False), ("failed", True)])
def test_budget_dependency_gate_never_ignores_failed_irasim(tmp_path, monkeypatch, state, raises):
    for gpu in range(4):
        path = tmp_path/"irasim_runs"/f"gpu{gpu}"
        path.mkdir(parents=True)
        (path/"progress.json").write_text(json.dumps({"study_sha256": "verified", "status": state}))
    monkeypatch.setattr(MODULE, "_digest", lambda p: "verified")
    if raises:
        with pytest.raises(ValueError, match="stopped/failed"):
            MODULE.wait_irasim(tmp_path, tmp_path)
    else:
        MODULE.wait_irasim(tmp_path, tmp_path)
