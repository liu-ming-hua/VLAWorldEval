"""Explicit raw-window/advance/stitch contracts; no model weights needed."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from evaluation.manifests import VideoSpec, PairManifest, load_pair_manifest
from evaluation.evaluator import RolloutTrace
from evaluation.contracts import ObservationBundle, WorldModelOutput


@pytest.mark.parametrize("count", [1, 2, 24])
def test_inclusive_stitch_follows_official_boundary_ownership(count):
    spec = VideoSpec(5, 5, "inclusive")
    # A predicted boundary differs across windows: assert next chunk owns it.
    chunks = [np.full((5, 2, 2, 3), i + 10, np.uint8) for i in range(count)]
    initial = np.full((1, 2, 2, 3), 255, np.uint8)
    full = spec.assemble(initial, chunks)
    assert len(full) == 4 * count + 1 == spec.frame_count(count)
    assert full[0, 0, 0, 0] == 10  # Never inject recorded initial pixels.
    assert full[-1, 0, 0, 0] == count + 9
    for i in range(count):
        assert np.all(full[i * 4:(i + 1) * 4] == i + 10)


def test_rollout_trace_and_native_executor_use_same_stitching():
    pair = load_pair_manifest("configs/pairs/pi0__ctrl_world.json")
    obs = ObservationBundle(views={n: np.zeros((2, 2, 3), np.uint8) for n in pair.world_model.output_views}, instruction="test")
    outputs = [WorldModelOutput(views={n: np.full((5, 2, 2, 3), 10 + i, np.uint8) for n in obs.views}) for i in range(3)]
    trace = RolloutTrace(pair, [obs], world_outputs=outputs)
    for name, frames in trace.view_videos().items():
        assert np.array_equal(frames, pair.world_model.video.assemble(obs.views[name], [v.views[name] for v in outputs]))
        assert len(frames) == 13


def test_inclusive_pixel_diagnostics_assign_boundary_to_next_chunk():
    from evaluation.reporting.image_feedback import pixel_diagnostics
    real = np.zeros((9, 16, 16, 3), np.uint8)
    ar = real + np.arange(9, dtype=np.uint8)[:, None, None, None]
    result = pixel_diagnostics(real, ar, real, frames_per_chunk=4, temporal_layout="inclusive")
    assert result["per_chunk"][0]["ar_mean"] == 1.5
    assert result["per_chunk"][1]["ar_mean"] == 6
    assert result["by_frame_offset"]["autoregressive"] == [2, 3, 4, 5]
    assert result["final_terminal_only"]["autoregressive"] == 8


def test_real_native_worker_inclusive_frames_and_image_anchors(tmp_path, monkeypatch):
    import mediapy as media
    from evaluation.execution.client import ModelWorker
    from evaluation.execution.transport import write_packet
    from tests.test_engineering_gaps import replay_fixture
    module, config_path, packet_path, packet = replay_fixture(tmp_path, monkeypatch)
    def launch(*args, **kwargs):
        kwargs.update(module="tests.runtime_fixtures", adapter="InclusiveFixtureWM")
        return ModelWorker(*args, **kwargs)
    monkeypatch.setattr(module, "ModelWorker", launch)
    config = json.loads(config_path.read_text())
    config["action"].update(horizon=5, rate_hz=5, temporal_layout="inclusive")
    config["video"].update(fps=5, frames_per_chunk=5, temporal_layout="inclusive")
    packet.update(protocol=config["action"], actions=np.ones((3, 5, 2), np.float32))
    packet["source"].update(fps=15, start_frame=15)
    config_path.write_text(json.dumps(config))
    write_packet(packet_path, packet)
    anchors = {"images": np.zeros((3, 16, 16, 3), np.uint8), "source_frame_indices": np.array([15, 27, 39]),
        "source": {"conversion": "fixture RGB only"}, "intervention": "ground_truth_visual_context"}
    anchor_path = tmp_path / "anchors.npz"
    write_packet(anchor_path, anchors)
    report = module.run_native_replay(config_path, packet_path, tmp_path / "result", image_anchors=anchor_path)
    assert report["world_model_calls"] == 3 and report["frames_per_view"] == 13
    assert [x["source_frame"] for x in report["anchor_events"]] == [27, 39]
    video = np.asarray(media.read_video(tmp_path / "result/front.mp4"))
    assert len(video) == 13
    # No real zero frame; next prediction owns boundary4,8; final endpoint kept.
    assert np.allclose(video[[0, 4, 8, 12]].mean(axis=(1, 2, 3)), [20, 40, 60, 64], atol=2)
    assert report["cleanup"]["returncode"] == 0


@pytest.mark.parametrize("fault", ["legacy", "source_stride", "overlap", "indices"])
def test_ctrl_native_admission_rejects_legacy_or_misaligned_packets(tmp_path, monkeypatch, fault):
    from evaluation.execution.transport import write_packet
    from tests.test_engineering_gaps import replay_fixture
    module, config_path, packet_path, packet = replay_fixture(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text())
    config.update(adapter="CtrlWorldAdapter", reset_operation="reset_official_replay")
    config["action"].update(dimension=7, horizon=5, rate_hz=5, temporal_layout="inclusive")
    config["video"].update(fps=5, frames_per_chunk=5, temporal_layout="inclusive")
    packet.update(protocol=config["action"], actions=np.ones((2, 5, 7), np.float32),
        reset_args={"image": [np.zeros((16, 16, 3), np.uint8)] * 3, "first_action": np.ones(7)})
    packet["source"].update(fps=15, start_frame=15, source_frames_per_chunk=12, temporal_layout="inclusive",
        action_source_frame_indices=[[15, 18, 21, 24, 27], [27, 30, 33, 36, 39]])
    if fault == "legacy":
        config["video"].pop("temporal_layout")
    elif fault == "source_stride":
        packet["source"]["source_frames_per_chunk"] = 15
    elif fault == "indices":
        packet["source"]["action_source_frame_indices"][1][0] = 30
    else:
        packet["actions"][1, 0] = 2
    config_path.write_text(json.dumps(config))
    write_packet(packet_path, packet)
    with pytest.raises(ValueError, match="Legacy Ctrl|overlapping"):
        module.run_native_replay(config_path, packet_path, tmp_path / "never_created", admit_only=True)
    assert not (tmp_path / "never_created").exists()


def test_legacy_pair_cannot_silently_claim_corrected_timing():
    pair = json.loads(Path("configs/pairs/pi0__ctrl_world.json").read_text())
    pair["world_model"]["video"].pop("temporal_layout")
    with pytest.raises(ValueError, match="Legacy Ctrl timing"):
        PairManifest.from_dict(pair)


def test_retry_selection_is_explicit_and_hash_bound(tmp_path):
    from evaluation.reporting.ctrl_validation import case_run_directory
    from evaluation.reporting.task_scores import _digest
    base = tmp_path / "case"
    run = base / "attempts/retry_01/autoregressive"
    run.mkdir(parents=True)
    report = run / "summary.json"
    report.write_text('{"status":"completed"}')
    pointer = base / "selected_autoregressive.json"
    assert case_run_directory(tmp_path, "case", "autoregressive") == base / "autoregressive"
    pointer.write_text(json.dumps({"relative_path": "attempts/retry_01/autoregressive", "report_sha256": _digest(report)}))
    assert case_run_directory(tmp_path, "case", "autoregressive") == run
    report.write_text('{"status":"failed"}')
    with pytest.raises(ValueError, match="changed"):
        case_run_directory(tmp_path, "case", "autoregressive")
    pointer.write_text(json.dumps({"relative_path": "../escape", "report_sha256": "invalid"}))
    with pytest.raises(ValueError, match="leaves"):
        case_run_directory(tmp_path, "case", "autoregressive")


def test_policy_history_excludes_duplicate_window_start_and_proxy_endpoint():
    from adapters.base import VLAAdapter, WorldModelAdapter
    from evaluation.evaluator import PolicyWorldModelEvaluator
    from tests.test_pilworld_mode import setup
    pair, _, obs, _ = setup("ctrl")
    class Policy(VLAAdapter):
        adapter_identity = "GrootN17Adapter"
        def __init__(self):
            self.inputs = []
        def reset(self, initial_state=None):
            pass
        def predict(self, image, instruction, state=None):
            self.inputs.append(image["wrist_image_left"][:, 0, 0, 0].copy())
            values = np.zeros((40, 17), np.float32)
            values[:, 10:] = state["joint_position"]
            values[12:, 10] += .2  # Unexecuted replanning tail must be ignored.
            return values
    class World(WorldModelAdapter):
        adapter_identity = "CtrlWorldAdapter"
        def reset(self, image):
            self.count = 0
        def step(self, action):
            raise NotImplementedError
        def step_output(self, action, **kwargs):
            self.count += 1
            frames = np.stack([np.full((192, 320, 3), x, np.uint8) for x in
                [239] + list(range(4 * (self.count - 1) + 1, 4 * self.count + 1))])
            return WorldModelOutput(views={n: frames for n in pair.world_model.output_views})
    policy = Policy()
    trace = PolicyWorldModelEvaluator(policy, World(), pair_manifest=pair).rollout_detailed(
        {"observation_bundle": obs, "instruction": "move"}, 10)
    assert len(trace.primary_video()) == 41
    assert policy.inputs[1].tolist() == [0, 4]
    assert policy.inputs[2].tolist() == [3, 8]
    assert all(239 not in values for values in policy.inputs)
    assert np.allclose(trace.final_state.values["joint_position"], obs.robot_state.values["joint_position"])


@pytest.mark.parametrize("kind", ["action", "video"])
def test_inclusive_requires_two_samples(kind):
    if kind == "video":
        with pytest.raises(ValueError, match="inclusive"):
            VideoSpec(5, 1, "inclusive")
    else:
        spec = load_pair_manifest("configs/pairs/pi0__ctrl_world.json").world_model.action
        with pytest.raises(ValueError, match="inclusive"):
            replace(spec, horizon=1)


def test_actual_ctrl_source_indices_and_no_tail_padding():
    root = Path("datasets/cosmos3_droid/success")
    if not root.is_dir():
        pytest.skip("separately acquired dataset absent")
    pytest.importorskip("pyarrow")
    from evaluation.datasets.droid import load_droid_episode
    data = load_droid_episode(root, 795, 15, 24, protocol="ctrl_world")
    indices = np.asarray(data.provenance["action_source_frame_indices"])
    assert np.array_equal(indices[0], [15, 18, 21, 24, 27])
    assert indices[-1, -1] == 303
    assert data.provenance["end_frame"] == 303
    assert np.array_equal(np.stack(data.replay_actions)[:-1, -1], np.stack(data.replay_actions)[1:, 0])
    with pytest.raises(ValueError, match="no tail padding"):
        load_droid_episode(root, 795, 15, 25, protocol="ctrl_world")


def test_camera_vae_protocol_in_optional_ctrl_environment():
    python = Path("environment/world_models/ctrl_world/bin/python")
    if not python.exists():
        pytest.skip("optional Torch runtime absent")
    # Real torch on CPU, stub VAE only. Checks actual adapter methods and RNG.
    code = '''
from types import SimpleNamespace
import torch, numpy as np
from adapters.world_models.ctrl_world import CtrlWorldAdapter
wm=object.__new__(CtrlWorldAdapter)
wm.device=torch.device('cpu'); wm.dtype=torch.float32
wm.height=16; wm.width=16; wm.num_frames=5; wm.decode_chunk_size=7
calls=[]
class VAE:
 config=SimpleNamespace(scaling_factor=1.)
 def encode(self,x):
  calls.append(('encode',tuple(x.shape)))
  v=torch.zeros((1,4,2,2))+len(calls)
  return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda:v))
 def decode(self,x,num_frames):
  calls.append(('decode',tuple(x.shape),num_frames))
  return SimpleNamespace(sample=torch.zeros((len(x),3,16,16)))
wm.pipeline=SimpleNamespace(vae=VAE())
latent=wm._encode_views([np.zeros((16,16,3),np.uint8) for _ in range(3)])
assert latent.shape==(1,4,6,2)
assert [x[1] for x in calls]==[(1,3,16,16)]*3
video=wm._decode_latents(torch.zeros((5,3,4,2,2)))
assert video.shape==(5,3,16,16,3)
assert [x[2] for x in calls if x[0]=='decode']==[7,7,1]
wm.num_history=6; wm._memory_latents=[latent.clone() for _ in range(27)]
wm._memory_actions=[np.zeros(7) for _ in range(27)]; wm._current_latent=latent
state=torch.random.get_rng_state().clone()
info=wm.reanchor_image([np.zeros((16,16,3),np.uint8) for _ in range(3)])
assert torch.equal(state,torch.random.get_rng_state()) and info['step_index']==3
assert len(wm._memory_actions)==27
print('camera encode/decode and reanchor verified')
'''
    subprocess.run([str(python), "-c", code], check=True, timeout=60)
