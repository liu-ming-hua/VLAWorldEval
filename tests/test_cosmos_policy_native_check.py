"""The native check must not hide shutdown failures behind valid actions."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("clean", [True, False])
def test_native_check_requires_inference_and_clean_exit(tmp_path, monkeypatch, clean):
    from scripts.checks import cosmos_policy_droid as check
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["check", "--output", str(tmp_path / "run")])
    monkeypatch.setattr("vlaworldeval.assets.verify_snapshot", lambda *a: {
        "repository": "nvidia/Cosmos3-Nano-Policy-DROID", "revision": "805c0d6d46196ecdc789213a6de72303c383895b"})
    monkeypatch.setattr("evaluation.execution.admission.policy_resource_admission", lambda *a: {})
    monkeypatch.setattr("evaluation.validation.native_runtime.probe_native_runtime", lambda *a, **kw: {"ok": True})
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setattr("adapters.environment.get_environment", lambda *a: SimpleNamespace(python=interpreter))
    state = {"observation/joint_position": np.zeros(7), "observation/gripper_position": np.zeros(1)}
    trial = {"observation_bundle": {"views": {f"droid.{key}": np.zeros((8, 8, 3), dtype=np.uint8)
        for key in ("exterior_1", "exterior_2", "wrist")}}, "instruction": "fixture",
        "initial_state": SimpleNamespace(values=state)}
    monkeypatch.setattr("evaluation.datasets.droid.load_openpi_droid_episode",
                        lambda *a: SimpleNamespace(trial=trial, provenance={"fixture": True}))

    class Worker:
        memory = []
        ready = {"fixture": True}
        failure = None

        def __init__(self, *args, **kwargs):
            assert kwargs["options"].shutdown_grace_seconds == 30

        def call(self, *args, **kwargs):
            return np.zeros((32, 8), dtype=np.float32)

        def close(self):
            return {"returncode": 0 if clean else -15, "errors": [], "signals": [] if clean else ["SIGTERM"]}

    monkeypatch.setattr("evaluation.execution.client.ModelWorker", Worker)
    assert check.main() == (0 if clean else 1)
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["inference_passed"] is True
    assert report["cleanup_passed"] is clean
    assert report["status"] == ("passed" if clean else "failed")
    assert report["world_model_calls"] == 0
    assert report["policy_calls"] == 1
