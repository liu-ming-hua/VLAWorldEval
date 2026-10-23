import json
import sys

import numpy as np
import pytest

from evaluation.execution.reliability import WorkerOptions, clean_worker_exit
from evaluation.execution.worker_session import NativeWorkerSession


def acquire(session, runtime_dir, **changes):
    config = dict(
        module="tests.support.runtime",
        adapter="ReusableFixtureWM",
        kwargs={"runtime_dir": str(runtime_dir)},
        gpus=[],
        memory_gib=1,
        options=WorkerOptions(startup_timeout_seconds=20, request_timeout_seconds=10),
    )
    config.update(changes)
    return session.acquire(sys.executable, **config)


def test_real_cpu_worker_reused_with_isolated_seed_state_and_directories(tmp_path):
    with NativeWorkerSession(tmp_path / "session") as session:
        outputs, pids = [], []
        for index in range(2):
            worker, reused = acquire(session, tmp_path / f"case{index}/runtime")
            assert reused == bool(index)
            pids.append(worker.process.pid)
            worker.call("seed", seed=42)
            worker.call("reset", value=np.zeros((16, 16, 3), np.uint8))
            result = worker.call(
                "step", action=np.zeros((1, 1)), output_views=["primary"], state_schema=None
            )
            outputs.append(result["views"]["primary"])
            path = tmp_path / f"case{index}/summary.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(
                json.dumps({"status": "generated_pending_worker_close", "worker_session": {}})
            )
            session.track_report(path)
            assert "cleanup" not in json.loads(path.read_text())
        assert pids[0] == pids[1]
        np.testing.assert_array_equal(outputs[0], outputs[1])
    assert session.clean
    assert json.loads((tmp_path / "session/session_close.json").read_text())["model_loads"] == 1
    for index in range(2):
        report = json.loads((tmp_path / f"case{index}/summary.json").read_text())
        assert report["status"] == "completed" and clean_worker_exit(report["cleanup"])


def test_reuse_refuses_scientific_or_runtime_change(tmp_path):
    with NativeWorkerSession(tmp_path / "session") as session:
        acquire(session, tmp_path / "case0")
        with pytest.raises(ValueError, match="cannot change"):
            acquire(session, tmp_path / "case1", memory_gib=2)


def test_resident_budget_is_not_admitted_twice(tmp_path):
    session = NativeWorkerSession(tmp_path / "session")
    assert not session.owns_resources([], 1)
    with session:
        acquire(session, tmp_path / "case0")
        assert session.owns_resources([], 1)
        assert not session.owns_resources([0], 1)
        assert not session.owns_resources([], 2)
    assert not session.owns_resources([], 1)


def test_unsupported_adapter_cannot_silently_reuse(tmp_path):
    with NativeWorkerSession(tmp_path / "session") as session:
        with pytest.raises(ValueError, match="declare"):
            acquire(session, tmp_path / "case0", adapter="NoReuseFixtureWM")


def test_dead_worker_cannot_certify_earlier_generated_cases(tmp_path):
    from evaluation.execution.reliability import WorkerError

    path = tmp_path / "summary.json"
    with NativeWorkerSession(tmp_path / "session") as session:
        worker, _ = acquire(session, tmp_path / "case0")
        path.write_text(
            json.dumps({"status": "generated_pending_worker_close", "worker_session": {}})
        )
        session.track_report(path)
        with pytest.raises(WorkerError):
            worker.call("invalid_fixture_operation")
    assert not session.clean
    assert json.loads(path.read_text())["status"] == "failed"
