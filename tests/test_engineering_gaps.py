import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

from evaluation.validation.environment_audit import compare_environment, audit_environments
from evaluation.reporting.task_scores import load_annotations, summarize_scores
from evaluation.execution.transport import write_packet


def test_environment_drift_catches_packages_python_and_sources():
    baseline = {"python": [3, 11, 2], "packages": {"a": "1", "b": "2"}, "source_requirements": []}
    current = {"python": [3, 11, 3], "packages": {"a": "2", "c": "1"},
               "source_requirements": [{"package": "a", "commit": "changed"}]}
    drift = compare_environment(current, baseline)
    assert drift["added"] == ["c"] and drift["removed"] == ["b"]
    assert drift["changed"]["a"] == {"expected": "1", "actual": "2"}
    assert drift["python_changed"] and drift["sources_changed"]
    assert not any(compare_environment(baseline, baseline).values())


def test_environment_probe_is_stdlib_only(monkeypatch, tmp_path):
    import evaluation.validation.environment_audit as module
    monkeypatch.setattr(module, "load_environment_registry", lambda: {
        "test": SimpleNamespace(path=Path(sys.executable).parent, python=Path(sys.executable),
                                package_profile="test", exists=True)})
    report = audit_environments(keys=["test"])
    assert report["ok"] and not report["portable_lock"] and not report["fresh_install_verified"]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(report))
    assert audit_environments(keys=["test"], baseline=baseline)["ok"]
    with pytest.raises(ValueError, match="Unknown environments"):
        audit_environments(keys=["other"])


def annotations(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"fixture": true}')
    records = [{"policy": policy, "task": "pick", "episode": "e1", "seed": "0", "domain": domain,
                "evaluator": "human", "rubric": "pick-v1", "evidence": evidence.name,
                "evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(), "score": score}
               for policy, domain, score in [("a", "real", 1), ("a", "world_model", 1),
                                              ("b", "real", 0), ("b", "world_model", 0)]]
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"schema_version": 1, "records": records}))
    return path, records


def test_local_scores_are_evidence_bound_and_matched(tmp_path):
    path, records = annotations(tmp_path)
    report = summarize_scores(load_annotations(path))
    assert report["pairwise_concordance"] == 1
    assert not report["scientific_validation_claimed"]
    records[-1]["episode"] = "different"
    assert summarize_scores(records)["pairwise_concordance"] is None
    records[-1]["score"] = None
    assert summarize_scores(records)["groups"][-1]["mean_score"] is None
    (tmp_path / "evidence.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_annotations(path)


@pytest.mark.parametrize("score", [True, -1, 2, float("nan"), None])
def test_invalid_or_unexplained_scores_rejected(tmp_path, score):
    path, records = annotations(tmp_path)
    records[0]["score"] = score
    path.write_text(json.dumps({"schema_version": 1, "records": records}))
    with pytest.raises(ValueError):
        load_annotations(path)


def replay_fixture(tmp_path, monkeypatch, *, negative=False):
    import evaluation.execution.native_replay as module
    from evaluation.execution.client import ModelWorker
    monkeypatch.setattr(module, "get_environment", lambda key: SimpleNamespace(
        scope="world_model", exists=True, python=Path(sys.executable)))
    # Production admission allows only trusted WM package modules. Test the
    # runner with a real persistent fixture worker, not a mock stepping loop.
    def launch(*args, **kwargs):
        kwargs.update(module="tests.runtime_fixtures", adapter="FixtureWM")
        return ModelWorker(*args, **kwargs)
    monkeypatch.setattr(module, "ModelWorker", launch)
    action = {"dimension": 2, "horizon": 2, "representation": "fixture", "frame": "fixture",
              "semantics": "fixture", "rate_hz": 10}
    config = {"schema_version": 1, "environment": "fixture", "module": "adapters.world_models.fixture",
              "adapter": "FixtureWM", "kwargs": {}, "gpus": [], "memory_gib": 1, "seed": 0,
              "action": action, "video": {"fps": 10, "frames_per_chunk": 2},
              "output_views": ["front", "wrist"], "reset_operation": "reset", "worker_options": {},
              "instruction_conditioning": False}
    config_path, packet_path = tmp_path / "config.json", tmp_path / "input.npz"
    config_path.write_text(json.dumps(config))
    actions = np.ones((20, 2, 2), dtype=np.float32)
    if negative:
        actions[3] = -1
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    packet = {"protocol": action, "actions": actions, "reset_args": {"value": frame},
              "initial_views": {"front": frame, "wrist": frame},
              "source": {"conversion": "synthetic fixture, not robot evidence"}}
    write_packet(packet_path, packet)
    return module, config_path, packet_path, packet


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
        kwargs.update(module="tests.runtime_fixtures", adapter="BrokenCloseFixtureWM")
        return ModelWorker(*args, **kwargs)
    monkeypatch.setattr(module, "ModelWorker", launch)
    report = module.run_native_replay(config, packet, tmp_path / "result")
    assert report["status"] == "failed" and report["world_model_calls"] == 20
    assert report["error"]["code"] == "shutdown_failed"


@pytest.mark.parametrize("payload", [[], None, {"schema_version": True, "records": []}, {"schema_version": 1, "records": [[]]}])
def test_malformed_annotations_raise_input_errors(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_annotations(path)
