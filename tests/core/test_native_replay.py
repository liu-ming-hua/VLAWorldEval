import hashlib
import json

import numpy as np
import pytest

from evaluation.execution.transport import write_packet
from tests.support.builders import replay_fixture

pytestmark = pytest.mark.process


def test_native_replay_runtime_binding_is_output_scoped(tmp_path, monkeypatch):
    module, config, packet, _ = replay_fixture(tmp_path, monkeypatch)
    value = json.loads(config.read_text())
    value["kwargs"]["runtime_dir"] = {"suite_path": "runtime"}
    config.write_text(json.dumps(value))
    admitted = module.admit_native_replay(config, packet, suite_dir=tmp_path / "output")
    assert admitted[5]["runtime_dir"] == str(tmp_path / "output/runtime")
    assert not (tmp_path / "runtime").exists()


def test_native_replay_long_real_worker_and_no_overwrite(tmp_path, monkeypatch):
    module, config, packet, _ = replay_fixture(tmp_path, monkeypatch)
    output = tmp_path / "result"
    assert module.run_native_replay(config, packet, output, admit_only=True)["executed"] is False
    assert not output.exists()
    report = module.run_native_replay(config, packet, output)
    assert report["status"] == "completed"
    assert report["world_model_calls"] == 20 and report["vla_calls"] == 0
    assert report["frames_per_view"] == 41
    assert report["cleanup"]["returncode"] == 0 and not report["cleanup"]["errors"]
    assert np.load(output / "actions.npy", allow_pickle=False).shape == (20, 2, 2)
    for filename, digest in report["files"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        module.run_native_replay(config, packet, output)


def test_native_replay_failure_is_preserved(tmp_path, monkeypatch):
    module, config, packet, _ = replay_fixture(tmp_path, monkeypatch, negative=True)
    with pytest.raises(RuntimeError, match="deliberate fixture failure"):
        module.run_native_replay(config, packet, tmp_path / "failed")
    report = json.loads((tmp_path / "failed/summary.json").read_text())
    assert report["status"] == "failed" and report["world_model_calls"] == 3
    assert report["cleanup"]["returncode"] is not None


def test_native_replay_rejects_protocol_and_frame_counts(tmp_path, monkeypatch):
    module, config, packet_path, packet = replay_fixture(tmp_path, monkeypatch)
    packet["protocol"] = {**packet["protocol"], "frame": "wrong_frame"}
    write_packet(packet_path, packet)
    with pytest.raises(ValueError, match="protocol differs"):
        module.run_native_replay(config, packet_path, tmp_path / "bad", admit_only=True)


def test_native_replay_wrong_future_count_fails(tmp_path, monkeypatch):
    module, config, packet, _ = replay_fixture(tmp_path, monkeypatch)
    data = json.loads(config.read_text())
    data["video"]["frames_per_chunk"] = 3
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="frames; expected"):
        module.run_native_replay(config, packet, tmp_path / "failed")
    report = json.loads((tmp_path / "failed/summary.json").read_text())
    assert report["status"] == "failed" and report["world_model_calls"] == 0


def test_native_replay_requires_instruction_when_enabled(tmp_path, monkeypatch):
    module, config, packet, _ = replay_fixture(tmp_path, monkeypatch)
    data = json.loads(config.read_text())
    data["instruction_conditioning"] = True
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source instruction"):
        module.run_native_replay(config, packet, tmp_path / "bad", admit_only=True)


def test_native_replay_unclean_shutdown_not_success(tmp_path, monkeypatch):
    module, config, packet, _ = replay_fixture(tmp_path, monkeypatch)
    from evaluation.execution.client import ModelWorker

    def launch(*args, **kwargs):
        kwargs.update(module="tests.support.runtime", adapter="BrokenCloseFixtureWM")
        return ModelWorker(*args, **kwargs)

    monkeypatch.setattr(module, "ModelWorker", launch)
    report = module.run_native_replay(config, packet, tmp_path / "result")
    assert report["status"] == "failed" and report["world_model_calls"] == 20
    assert report["error"]["code"] == "shutdown_failed"
