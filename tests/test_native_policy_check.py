"""Native startup is distinct from input checking, and cannot hide cleanup failure."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.execution.policy_check import POLICIES, run_policy_check
from evaluation.execution.reliability import RunInterrupted


@pytest.fixture
def native_check(tmp_path, monkeypatch):
    state = {"observation/joint_position": np.zeros(7), "observation/gripper_position": np.zeros(1)}
    images = {f"droid.{key}": np.zeros((8, 8, 3), dtype=np.uint8)
              for key in ("exterior_1", "exterior_2", "wrist")}
    trial = {"observation_bundle": {"views": images}, "instruction": "fixture",
             "initial_state": SimpleNamespace(values=state)}
    monkeypatch.setattr("evaluation.datasets.droid.load_openpi_droid_episode",
                        lambda *a: SimpleNamespace(trial=trial, provenance={"fixture": True}))
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setattr("adapters.environment.get_environment", lambda *a: SimpleNamespace(python=interpreter))
    calls = []

    def verify(path):
        calls.append("assets")
        spec = POLICIES["cosmos_droid"] if "cosmos3" in str(path) else POLICIES["dreamzero_droid"]
        if path.name == "umt5-tokenizer":
            return {"repository": "google/umt5-xxl", "revision": "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"}
        return {"repository": spec.repository, "revision": spec.revision}

    monkeypatch.setattr("vlaworldeval.assets.verify_snapshot", verify)
    monkeypatch.setattr("evaluation.execution.admission.policy_resource_admission", lambda *a: {"fixture": True})
    monkeypatch.setattr("evaluation.validation.native_runtime.probe_native_runtime", lambda *a, **kw: {"ok": True})

    class Worker:
        memory = []
        ready = {"fixture": True}
        failure = None
        cleanup = {"returncode": 0, "errors": [], "signals": []}
        shape = None
        exception = None

        def __init__(self, *a, **kw):
            calls.append("worker")
            self.horizon = 24 if kw["adapter"] == "DreamZeroDroidAdapter" else 32
            assert kw["options"].shutdown_grace_seconds == 30
            assert kw["kwargs"].get("attention_backend", "TE") == "TE"

        def call(self, operation, **kwargs):
            calls.append(operation)
            if operation == "reset":
                return None
            assert set(kwargs) == {"image", "instruction", "state"}
            if self.exception:
                raise self.exception
            return np.zeros(self.shape or (self.horizon, 8), dtype=np.float32)

        def close(self):
            calls.append("close")
            return self.cleanup

    monkeypatch.setattr("evaluation.execution.client.ModelWorker", Worker)
    return tmp_path, Worker, calls


@pytest.mark.parametrize("model", list(POLICIES))
def test_native_input_only_never_verifies_weights_or_starts_worker(native_check, model):
    root, _, calls = native_check
    report = run_policy_check(root, model, "input")
    assert report["status"] == "inputs_checked"
    assert report["model_inference_performed"] is False
    assert report["asset_integrity_verified"] is False
    assert not calls


@pytest.mark.parametrize("model", list(POLICIES))
@pytest.mark.parametrize("clean", [True, False])
def test_native_success_requires_clean_exit(native_check, model, clean):
    root, Worker, calls = native_check
    if not clean:
        Worker.cleanup = {"returncode": 0, "errors": [], "signals": ["SIGTERM"]}
    report = run_policy_check(root, model, "run", execute=True)
    assert report["status"] == ("passed" if clean else "failed")
    assert report["inference_passed"] is True
    assert report["cleanup_passed"] is clean
    assert report["action_shape"] == [POLICIES[model].horizon, 8]
    assert report["world_model_calls"] == 0
    assert calls[-3:] == ["reset", "predict", "close"]
    assert json.loads((root / "run/report.json").read_text()) == report


def test_busy_gpu_rejects_before_asset_hash_and_worker(native_check, monkeypatch):
    from evaluation.execution.admission import ResourceAdmissionError
    root, _, calls = native_check

    def reject(*a):
        raise ResourceAdmissionError("fixture busy")

    monkeypatch.setattr("evaluation.execution.admission.policy_resource_admission", reject)
    report = run_policy_check(root, "dreamzero_droid", "busy", execute=True)
    assert report["status"] == "resource_rejected"
    assert report["model_inference_performed"] is False
    assert not calls
    assert (root / "busy/report.json").is_file()


def test_runtime_rejection_never_hashes_weights_or_launches_worker(native_check, monkeypatch):
    root, _, calls = native_check
    monkeypatch.setattr("evaluation.validation.native_runtime.probe_native_runtime",
                        lambda *a, **kw: {"ok": False, "errors": [{"module": "fixture_missing_TE"}]})
    report = run_policy_check(root, "dreamzero_droid", "runtime_bad", execute=True)
    assert report["status"] == "runtime_rejected"
    assert not report["model_inference_performed"]
    assert not report["asset_integrity_verified"]
    assert not calls


def test_runtime_check_without_gpu_or_weights(native_check, monkeypatch):
    root, _, calls = native_check
    def no_gpu(*a):
        pytest.fail("Runtime import check must not query or allocate GPUs")
    monkeypatch.setattr("evaluation.execution.admission.policy_resource_admission", no_gpu)
    report = run_policy_check(root, "dreamzero_droid", "runtime_only", runtime_check=True)
    assert report["status"] == "runtime_checked"
    assert report["runtime_probe"]["ok"] is True
    assert not report["model_inference_performed"]
    assert not calls


@pytest.mark.parametrize("shape", [(8, 8), (24, 7)])
def test_invalid_action_still_closes_worker(native_check, shape):
    root, Worker, calls = native_check
    Worker.shape = shape
    report = run_policy_check(root, "dreamzero_droid", "invalid", execute=True)
    assert report["status"] == "failed"
    assert report["cleanup_passed"] is True
    assert not (root / "invalid/actions.npz").exists()
    assert calls[-1] == "close"


def test_cancellation_saved_and_reraised(native_check):
    import signal
    root, Worker, calls = native_check
    Worker.exception = RunInterrupted(signal.SIGTERM)
    with pytest.raises(RunInterrupted):
        run_policy_check(root, "dreamzero_droid", "cancel", execute=True)
    report = json.loads((root / "cancel/report.json").read_text())
    assert report["status"] == "interrupted"
    assert report["cleanup_passed"] is True
    assert calls[-1] == "close"


def test_cancellation_during_cleanup_is_not_swallowed(native_check, monkeypatch):
    import signal
    root, Worker, _ = native_check

    def interrupted_close(self):
        raise RunInterrupted(signal.SIGTERM)

    monkeypatch.setattr(Worker, "close", interrupted_close)
    with pytest.raises(RunInterrupted):
        run_policy_check(root, "dreamzero_droid", "cancel_cleanup", execute=True)
    report = json.loads((root / "cancel_cleanup/report.json").read_text())
    assert report["status"] == "interrupted"
    assert report["cleanup_passed"] is False


@pytest.mark.parametrize("kwargs", [
    {"gpus": [0, 0]}, {"gpus": [4]}, {"gpus": [0, 1, 2]}, {"memory_gib": float("nan")},
    {"episode": -1}, {"start": True}, {"attention_backend": "fake"},
])
def test_invalid_config_has_no_output(native_check, kwargs):
    root, _, calls = native_check
    with pytest.raises(ValueError):
        run_policy_check(root, "dreamzero_droid", "invalid", **kwargs)
    assert not (root / "invalid").exists()
    assert not calls


def test_existing_output_preserved(native_check):
    root, _, calls = native_check
    (root / "existing").mkdir()
    with pytest.raises(FileExistsError):
        run_policy_check(root, "dreamzero_droid", "existing")
    assert not calls
