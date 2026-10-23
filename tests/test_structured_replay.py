"""One executor handles structured and numeric conditions; fixtures are not model evidence."""
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from evaluation.execution.transport import read_packet, write_packet
from evaluation.reporting.task_scores import _digest


def test_structured_replay_scores_without_sending_future_images_to_worker(tmp_path, monkeypatch):
    from evaluation.execution import native_replay as module
    fake = ModuleType("adapters.world_models.reference_fixture")
    class Adapter:
        native_condition_protocol = "fixture.geometry.v1"

        @staticmethod
        def validate_replay_condition(value):
            if set(value) != {"control"}:
                raise ValueError("not native geometry")
    fake.Adapter = Adapter
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(module, "get_environment", lambda _: SimpleNamespace(scope="world_model", exists=True, python=sys.executable))
    config = {"schema_version": 2, "environment": "world_models.fixture", "module": fake.__name__,
        "adapter": "Adapter", "kwargs": {}, "gpus": [], "memory_gib": 1, "seed": 0,
        "action": {"kind": "structured", "protocol": Adapter.native_condition_protocol},
        "video": {"fps": 5, "frames_per_chunk": 3, "temporal_layout": "inclusive"},
        "output_views": ["cam"], "reset_operation": "reset", "instruction_conditioning": False, "worker_options": {}}
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps(config)+"\n")
    pixels = np.stack([np.full((16, 16, 3), i, np.uint8) for i in (0, 10, 20)])
    packet = {"protocol": config["action"], "actions": [{"control": np.zeros(2)}],
              "reset_args": {"value": pixels[0]}, "initial_views": {"cam": pixels[0]},
              "source": {"conversion": "fixture only"}}
    input_path = tmp_path / "input.npz"
    write_packet(input_path, packet)
    ref = {"schema_version": 1, "input_sha256": _digest(input_path), "views": {"cam": pixels},
           "frame_indices": {"cam": np.arange(3)}, "source": {"dataset_domain": "fixture", "split": "test",
             "episode": "0", "conversion": "fixture", "condition_kind": "geometry", "frame_axis": "tick",
             "ground_truth_kind": "recorded_execution"}}
    reference = tmp_path / "reference.npz"
    write_packet(reference, ref)
    calls = []
    class Worker:
        ready = {"fixture": True}
        failure = None

        def __init__(self, *a, **kw):
            pass

        def call(self, operation, **kwargs):
            calls.append((operation, kwargs))
            if operation == "step":
                assert set(kwargs) == {"action", "output_views", "state_schema"}
                assert set(kwargs["action"]) == {"control"}
                return {"views": {"cam": pixels.copy()}, "metadata": {"fixture": True}}

        def close(self):
            return {"signals": [], "errors": [], "returncode": 0}
    monkeypatch.setattr(module, "ModelWorker", Worker)
    monkeypatch.setattr(module.media, "write_video", lambda path, *a, **kw: path.write_bytes(b"fixture-not-video"))
    out = tmp_path / "run"
    result = module.run_native_replay(conf, input_path, out, reference_path=reference)
    assert result["status"] == "completed" and result["world_model_calls"] == 1 and result["vla_calls"] == 0
    assert result["reference_evaluation"]["metrics"]["cam"]["mse_01"] == 0
    assert (out / "config.json").read_bytes() == conf.read_bytes()
    assert read_packet(out / "conditions.npz")["protocol"] == packet["protocol"]
    assert all("reference" not in args for _, args in calls)
    with pytest.raises(ValueError, match="not declared"):
        module.run_native_replay(conf, input_path, tmp_path / "bad", image_anchors=reference)


def test_structured_adapter_boundary_is_real_and_numeric_v1_unchanged():
    from adapters.world_models.dreamtrue import DreamTrueAdapter, PROTOCOL
    assert DreamTrueAdapter.native_condition_protocol == PROTOCOL
    with pytest.raises(ValueError, match="rendered geometry"):
        DreamTrueAdapter.validate_replay_condition(np.zeros((13, 7)))
