"""Reference inputs must be frozen as carefully as model inputs."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.execution.transport import write_packet
from evaluation.planning.native_reference import prepare_reference_study


def inputs(root):
    directory = root / "datasets/prepared"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps({"environment": "world_models.fixture"}))
    write_packet(directory / "input.npz", {"actions": np.zeros((2, 1, 4))})
    write_packet(directory / "reference.npz", {"fixture": True})
    plan = {"schema_version": 1, "scope": "fixture", "cases": [{"id": "case0", "prepared": "datasets/prepared"}],
            "unavailable": {"gated": "Fixture missing data is not a completed experiment"}}
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


def test_freeze_includes_reference_hash_only_needed_environment_and_no_inference(tmp_path, monkeypatch):
    from evaluation.planning import native_reference as module
    path, _ = inputs(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(module, "load_environment_registry", lambda: {
        "world_models.fixture": SimpleNamespace(exists=True, path=runtime),
        "vla.unused": SimpleNamespace(exists=True, path=runtime)})
    out = tmp_path / "study"
    r = prepare_reference_study(tmp_path, path, out)
    assert not r["model_inference_performed"] and r["vla_calls"] == 0
    assert r["jobs"][0]["reference_sha256"] and r["jobs"][0]["chunks"] == 2
    assert set(json.loads((out/"environments.json").read_text())["paths"]) == {"world_models.fixture"}
    with pytest.raises(FileExistsError):
        prepare_reference_study(tmp_path, path, out)


@pytest.mark.parametrize("change", ["duplicate", "escape", "missing_reference", "unavailable_environment"])
def test_bad_plan_never_starts_model_or_creates_output(tmp_path, monkeypatch, change):
    from evaluation.planning import native_reference as module
    path, plan = inputs(tmp_path)
    monkeypatch.setattr(module, "load_environment_registry", lambda: {})
    if change == "duplicate":
        plan["cases"] *= 2
    elif change == "escape":
        plan["cases"][0]["prepared"] = "../outside"
    elif change == "missing_reference":
        (tmp_path/"datasets/prepared/reference.npz").unlink()
    path.write_text(json.dumps(plan))
    out = tmp_path / "study"
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_reference_study(tmp_path, path, out)
    assert not out.exists()
