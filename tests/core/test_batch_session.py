import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from evaluation.execution.batch_session import BatchedNativeSession
from evaluation.execution.reliability import WorkerOptions


class FakeWorker:
    def __init__(self, *args, **kwargs):
        self.ready = {"load_seconds": 1.0, "pid": 42}
        self.memory = None
        self.failure = None
        self.batches = []

    def call(self, operation, *, items):
        assert operation == "native_batch_step"
        self.batches.append(items)
        return [
            {
                "views": {
                    "droid.cosmos_composite": np.full(
                        (16, 8, 8, 3),
                        int(item["image"][0, 0, 0]) + int(item["action"][0, 0]),
                        np.uint8,
                    )
                },
                "metadata": {},
                "robot_state": None,
            }
            for item in items
        ]

    def close(self):
        return {"signals": [], "errors": [], "returncode": 0, "group_alive_after_cleanup": False}


def acquire(session, path, **overrides):
    kwargs = dict(
        module="adapters.world_models.cosmos3_droid_native",
        adapter="Cosmos3DroidFDNativeAdapter",
        kwargs={"diffusion_cache": False, "inference_batch_size": 2, "runtime_dir": str(path)},
        gpus=[0, 1, 2, 3],
        memory_gib=60,
        options=WorkerOptions(),
    )
    kwargs.update(overrides)
    return session.acquire(Path("/python"), **kwargs)[0]


def test_batch_routes_independent_history_seed_and_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr("evaluation.execution.batch_session.ModelWorker", FakeWorker)
    session = BatchedNativeSession(tmp_path / "session", batch_size=2)

    def replay(index):
        proxy = acquire(session, tmp_path / f"case{index}")
        proxy.call("seed", seed=index * 100)
        proxy.call("reset", value=np.full((8, 8, 3), index * 10, np.uint8))
        for _ in range(3):
            result = proxy.call(
                "step",
                action=np.full((16, 10), index + 1),
                output_views=["droid.cosmos_composite"],
                state_schema=None,
            )
        return result["views"]["droid.cosmos_composite"][-1, 0, 0, 0]

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(replay, [0, 1]))
    session.close()
    assert results == [3, 16]
    assert session.generated_chunks == 6
    assert any(len(batch) == 2 for batch in session.worker.batches)
    items = [item for batch in session.worker.batches for item in batch]
    assert sorted(item["seed"] for item in items) == [0, 1, 2, 100, 101, 102]
    assert len({i["runtime_dir"] for i in items}) == 6
    assert session.clean


def test_batch_rejects_mixed_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr("evaluation.execution.batch_session.ModelWorker", FakeWorker)
    with BatchedNativeSession(tmp_path / "session", batch_size=2) as session:
        acquire(session, tmp_path / "a")
        with pytest.raises(ValueError, match="Cannot mix"):
            acquire(session, tmp_path / "b", memory_gib=20)


def test_batch_clean_close_certifies_reports(monkeypatch, tmp_path):
    monkeypatch.setattr("evaluation.execution.batch_session.ModelWorker", FakeWorker)
    with BatchedNativeSession(tmp_path / "session", batch_size=2) as session:
        acquire(session, tmp_path / "a")
        path = tmp_path / "summary.json"
        path.write_text(
            json.dumps({"status": "generated_pending_worker_close", "worker_session": {}})
        )
        session.track_report(path)
    assert json.loads(path.read_text())["status"] == "completed"


@pytest.mark.parametrize("mutate_science", [False, True])
def test_batch_config_admission_only_allows_runtime_changes(tmp_path, mutate_science):
    from evaluation.reporting.checkpoint_metrics import admitted_resource_config
    from evaluation.reporting.task_scores import _digest

    original = tmp_path / "original.json"
    config = {
        "adapter": "Cosmos3DroidFDNativeAdapter",
        "gpus": [0, 1, 2, 3],
        "memory_gib": 16,
        "kwargs": {"num_inference_steps": 30},
    }
    original.write_text(json.dumps(config))
    effective = tmp_path / "effective.json"
    effective.write_text(
        json.dumps(
            {
                **config,
                "memory_gib": 72,
                "kwargs": {
                    "num_inference_steps": 5 if mutate_science else 30,
                    "inference_batch_size": 4,
                },
            }
        )
    )
    path = tmp_path / "campaign/runs/case0/summary.json"
    path.parent.mkdir(parents=True)
    (path.parent / "config.json").write_bytes(effective.read_bytes())
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    (tmp_path / "campaign/batch_admission.json").write_text(
        json.dumps(
            {
                "kind": "native_cross_trajectory_batch",
                "source_plan": str(plan),
                "source_plan_sha256": _digest(plan),
                "batch_size": 4,
                "cases": {
                    "case0": {
                        "original_config": str(original),
                        "original_sha256": _digest(original),
                        "effective_config": str(effective),
                        "effective_sha256": _digest(effective),
                    }
                },
            }
        )
    )
    assert (
        admitted_resource_config(
            tmp_path,
            path,
            {"config_sha256": _digest(effective)},
            {"config_sha256": _digest(original)},
            "case0",
        )
        is not mutate_science
    )


def test_batch_failure_releases_all_pending_callers(monkeypatch, tmp_path):
    class BrokenWorker(FakeWorker):
        def call(self, *args, **kwargs):
            self.failure = {"error": "deliberate batch failure"}
            raise RuntimeError("deliberate batch failure")

    monkeypatch.setattr("evaluation.execution.batch_session.ModelWorker", BrokenWorker)
    with BatchedNativeSession(tmp_path / "session", batch_size=2) as session:
        proxies = [acquire(session, tmp_path / f"case{i}") for i in range(2)]

        def predict(proxy):
            proxy.call("seed", seed=0)
            proxy.call("reset", value=np.zeros((8, 8, 3), np.uint8))
            with pytest.raises(RuntimeError, match="deliberate batch failure"):
                proxy.call(
                    "step",
                    action=np.zeros((16, 10)),
                    output_views=["droid.cosmos_composite"],
                    state_schema=None,
                )

        with ThreadPoolExecutor(2) as pool:
            list(pool.map(predict, proxies))
    assert not session.clean


def test_batch_stream_reads_report_not_reference_path_when_reusing(monkeypatch, tmp_path):
    from evaluation.execution import native_batch_stream
    from evaluation.reporting.task_scores import _digest

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "fixture",
                "unavailable": {},
                "cases": [{"id": "case0", "prepared": "fixture"}],
            }
        )
    )
    old = tmp_path / "old/case0"
    old.mkdir(parents=True)
    video = old / "primary.mp4"
    video.write_bytes(b"fixture")
    report = old / "summary.json"
    report.write_text(json.dumps({"files": {"primary.mp4": _digest(video)}}))
    reference = tmp_path / "reference.npz"
    monkeypatch.setattr(native_batch_stream, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        native_batch_stream, "select_replays", lambda *a, **k: ({"case0": (report, reference)}, [])
    )
    native_batch_stream.run(plan, tmp_path / "new", [tmp_path / "old"], batch_size=8)
    progress = json.loads((tmp_path / "new/progress.json").read_text())
    assert progress["status"] == "completed"
    assert progress["certified_completed"] == 1
