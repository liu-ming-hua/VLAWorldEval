"""Real CPU subprocess failures, deadlines and lifecycle cleanup."""

import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from evaluation.execution.client import ModelWorker
from evaluation.execution.reliability import (
    RunInterrupted,
    WorkerError,
    WorkerOptions,
    WorkerTimeoutError,
    exception_details,
    resolve_worker_options,
    run_signal_handlers,
)
from evaluation.planning.experiments import build_experiment_plan

pytestmark = pytest.mark.process


def create(tmp_path, adapter="SlowFixtureVLA", kwargs=None, options=None):
    return ModelWorker(
        Path(sys.executable),
        tmp_path / "worker",
        module="tests.support.runtime",
        adapter=adapter,
        kwargs=kwargs or {},
        gpus=[],
        memory_gib=0,
        options=options
        or WorkerOptions(
            startup_timeout_seconds=15,
            request_timeout_seconds=0.2,
            shutdown_grace_seconds=0.1,
            terminate_grace_seconds=0.1,
        ),
    )


def not_running(pid):
    try:
        # An orphaned zombie is dead; reaping it belongs to the host's init.
        return Path(f"/proc/{pid}/stat").read_text().split(") ")[1][0] == "Z"
    except FileNotFoundError:
        return True


@pytest.mark.parametrize("value", [0, -1, True, math.inf, math.nan, "3"])
def test_invalid_deadlines_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        WorkerOptions(request_timeout_seconds=value)


def test_common_options_role_override_and_typo_rejection():
    options = resolve_worker_options(
        {
            "worker_options": {"startup_timeout_seconds": 60},
            "vla_worker_options": {"request_timeout_seconds": 7},
        },
        "vla",
    )
    assert asdict(options) == {
        "startup_timeout_seconds": 60,
        "request_timeout_seconds": 7,
        "shutdown_grace_seconds": 5,
        "terminate_grace_seconds": 5,
    }
    with pytest.raises(ValueError, match="Unknown"):
        resolve_worker_options({"worker_options": {"request_timout_seconds": 3}}, "vla")


def test_suite_rejects_invalid_options_before_model_start(tmp_path):
    config = json.loads(Path("configs/experiments/openvla_worldgym_v1.json").read_text())
    config["runtime"]["worker_options"] = {"startup_timeout_seconds": 0}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="startup_timeout"):
        build_experiment_plan(path)


def test_request_timeout_reaps_worker_and_forbids_reuse(tmp_path):
    worker = create(tmp_path)
    with pytest.raises(WorkerTimeoutError) as caught:
        worker.call("predict", image=None, instruction="slow", state=None)
    assert isinstance(caught.value, TimeoutError)
    assert caught.value.details["code"] == "request_timeout"
    assert caught.value.details["operation"] == "predict"
    assert worker.process.poll() is not None and worker.log.closed
    assert len(list(worker.directory.glob("request_*.npz"))) == 1
    error = json.loads((worker.directory / "error.json").read_text())
    with pytest.raises(WorkerError, match="cannot be reused"):
        worker.call("reset", value=None)
    assert json.loads((worker.directory / "error.json").read_text()) == error
    assert worker.close() == worker.close()


def test_startup_timeout_and_sigkill_escalation(tmp_path):
    # Allow imports to finish before delaying in the constructor.
    marker = tmp_path / "pid"
    with pytest.raises(WorkerTimeoutError) as caught:
        create(
            tmp_path,
            kwargs={"delay": 30, "marker": str(marker), "ignore_term": True},
            options=WorkerOptions(
                startup_timeout_seconds=3,
                request_timeout_seconds=1,
                shutdown_grace_seconds=0.1,
                terminate_grace_seconds=0.1,
            ),
        )
    assert caught.value.details["code"] == "startup_timeout"
    assert marker.exists(), "Fixture imports should finish before the startup deadline"
    assert not_running(int(marker.read_text()))
    lifecycle = json.loads((tmp_path / "worker/lifecycle.json").read_text())
    assert "SIGKILL" in lifecycle["cleanup"]["signals"]


@pytest.mark.parametrize(
    "adapter,code,instruction",
    [
        ("StartupFailureVLA", "startup_failed", None),
        ("SlowFixtureVLA", "worker_exited", "exit"),
        ("SlowFixtureVLA", "out_of_memory", "oom"),
    ],
)
def test_structured_model_errors(tmp_path, adapter, code, instruction):
    worker = None
    with pytest.raises(WorkerError) as caught:
        worker = create(tmp_path, adapter=adapter)
        worker.call("predict", image=None, instruction=instruction, state=None)
    assert caught.value.details["code"] == code
    if worker:
        assert worker.closed and worker.process.poll() is not None
    assert json.loads((tmp_path / "worker/error.json").read_text())["code"] == code


def test_missing_python_launch_failure_is_recorded(tmp_path):
    with pytest.raises(WorkerError) as caught:
        ModelWorker(
            tmp_path / "missing-python",
            tmp_path / "worker",
            module="tests.support.runtime",
            adapter="FixtureVLA",
            kwargs={},
            gpus=[],
            memory_gib=0,
        )
    assert caught.value.details["code"] == "launch_failed"
    lifecycle = json.loads((tmp_path / "worker/lifecycle.json").read_text())
    assert lifecycle["status"] == "failed" and lifecycle["cleanup"]["returncode"] is None


def test_dead_launcher_children_are_cleaned_without_touching_parent(tmp_path):
    marker = tmp_path / "child_pid"
    worker = create(tmp_path, adapter="ChildFixtureVLA", kwargs={"marker": str(marker)})
    with pytest.raises(WorkerError):
        worker.call("predict", image=None, instruction="exit", state=None)
    assert not_running(int(marker.read_text()))
    assert worker.process.poll() == 24
    assert os.getpid() != worker.process.pid


def test_normal_close_is_idempotent(tmp_path):
    worker = create(
        tmp_path,
        adapter="FixtureVLA",
        options=WorkerOptions(
            startup_timeout_seconds=15,
            request_timeout_seconds=1,
            shutdown_grace_seconds=2,
            terminate_grace_seconds=0.1,
        ),
    )
    cleanup = worker.close()
    assert worker.close() is cleanup
    assert cleanup["returncode"] == 0 and not cleanup["errors"]
    assert worker.process.stdin.closed and worker.log.closed


def test_zero_exit_launcher_with_forced_child_cleanup_is_not_clean(tmp_path):
    marker = tmp_path / "leaked_child_pid"
    worker = create(
        tmp_path,
        adapter="CleanLauncherLeakingChildVLA",
        kwargs={"marker": str(marker)},
        options=WorkerOptions(
            startup_timeout_seconds=15,
            request_timeout_seconds=1,
            shutdown_grace_seconds=2,
            terminate_grace_seconds=0.1,
        ),
    )
    cleanup = worker.close()
    assert cleanup["returncode"] == 0
    assert cleanup["signals"]
    assert not_running(int(marker.read_text()))
    assert worker.failure["code"] == "shutdown_failed"
    lifecycle = json.loads((tmp_path / "worker/lifecycle.json").read_text())
    assert lifecycle["status"] == "failed"


@pytest.mark.parametrize(
    "cleanup,expected",
    [
        ({"returncode": 0, "errors": [], "signals": []}, True),
        ({"returncode": 0, "errors": [], "signals": ["SIGTERM"]}, False),
        ({"returncode": 0, "errors": ["close failed"], "signals": []}, False),
        ({"returncode": -15, "errors": [], "signals": []}, False),
        ({"returncode": 0, "errors": []}, False),
        ({"returncode": False, "errors": [], "signals": []}, False),
        (None, False),
    ],
)
def test_clean_exit_requires_complete_unforced_evidence(cleanup, expected):
    from evaluation.execution.reliability import clean_worker_exit

    assert clean_worker_exit(cleanup) is expected


def test_corrupt_response_is_classified_and_retained(tmp_path):
    worker = create(tmp_path)
    response = worker.directory / "response_000001.npz"
    response.write_bytes(b"not a numeric packet")
    with pytest.raises(WorkerError) as caught:
        worker.call("predict", image=None, instruction="slow", state=None)
    assert caught.value.details["code"] == "protocol_error"
    assert response.exists() and worker.process.poll() is not None


def test_keyboard_interrupt_closes_worker_and_preserves_original(tmp_path, monkeypatch):
    worker = create(tmp_path)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("test interruption")

    monkeypatch.setattr(worker, "_wait", interrupted)
    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        worker.call("predict", image=None, instruction="slow", state=None)
    assert worker.process.poll() is not None
    assert worker.failure["code"] == "interrupted"


def test_cleanup_does_not_kill_an_unrelated_session(tmp_path):
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        worker = create(tmp_path)
        with pytest.raises(WorkerTimeoutError):
            worker.call("predict", image=None, instruction="slow", state=None)
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=5)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_cli_signals_unwind_worker_cleanup_and_restore_handlers(tmp_path, sig):
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    worker = create(tmp_path, adapter="FixtureVLA")
    with pytest.raises(RunInterrupted) as caught:
        with run_signal_handlers():
            try:
                os.kill(os.getpid(), sig)
            finally:
                worker.close()
    assert exception_details(caught.value)["signal"] == signal.Signals(sig).name
    assert worker.process.poll() is not None
    assert all(signal.getsignal(s) == handler for s, handler in previous.items())
