"""Queue orchestration fixtures; no checkpoint or model-quality claims."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.execution.transport import write_packet
from evaluation.planning.native_batches import freeze_ready_batch, FILES
from evaluation.reporting.task_scores import _digest


@pytest.mark.parametrize("clean", [True, False])
def test_stream_shares_one_session_and_requires_clean_shutdown(tmp_path, monkeypatch, clean):
    from evaluation.execution import native_stream as module
    cases, records = [], {}
    for i in range(2):
        d = tmp_path/"datasets"/str(i)
        d.mkdir(parents=True)
        (d/"config.json").write_text(json.dumps({"environment": "world_models.fixture"}))
        write_packet(d/"input.npz", {"actions": np.zeros((1, 1, 1))})
        write_packet(d/"reference.npz", {"fixture": True})
        cases.append({"id": f"case{i}", "prepared": str(d.relative_to(tmp_path))})
        records[f"case{i}"] = {"chunks": 1, "sha256": {f: _digest(d/f) for f in FILES}}
    plan = tmp_path/"plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "scope": "fixture", "cases": cases, "unavailable": {}}))
    preparation = tmp_path/"preparation.json"
    preparation.write_text(json.dumps({"frozen": {"plan_sha256": _digest(plan)}, "completed": records}))
    batches = tmp_path/"batches"
    for _ in range(2):
        freeze_ready_batch(tmp_path, plan, preparation, batches, limit=1)
    env = tmp_path/"runtime"
    env.mkdir()
    monkeypatch.setattr("evaluation.planning.native_reference.load_environment_registry", lambda: {
        "world_models.fixture": SimpleNamespace(exists=True, path=env)})
    sessions, used = [], []

    class Session:
        worker = True

        def __init__(self, directory, **kwargs):
            self.clean = False
            self.callback = kwargs["on_generated"]
            sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.clean = clean

    def generate(root, directory, **kwargs):
        session = kwargs["worker_session"]
        used.append(session)
        study = json.loads((directory/"study.json").read_text())
        state = {"status": "generated_pending_worker_close", "jobs": {}}
        for job in study["jobs"]:
            report = directory/"runs"/job["id"]/"summary.json"
            session.callback(report)
            state["jobs"][job["id"]] = {"status": "generated_pending_worker_close"}
        (directory/"progress.json").write_text(json.dumps(state))
        return {"status": "generated_pending_worker_close"}

    monkeypatch.setattr(module, "NativeWorkerSession", Session)
    monkeypatch.setattr(module, "run_native_study", generate)
    output = tmp_path/"stream"
    if clean:
        result = module.run_native_stream(tmp_path, plan, batches, output, gpus=[0, 1, 2, 3])
        assert result["status"] == "completed" and result["certified_completed"] == 2
    else:
        with pytest.raises(ValueError, match="exit cleanly"):
            module.run_native_stream(tmp_path, plan, batches, output, gpus=[0, 1, 2, 3])
        assert json.loads((output/"progress.json").read_text())["status"] == "failed"
    assert len(sessions) == 1 and used == [sessions[0], sessions[0]]
