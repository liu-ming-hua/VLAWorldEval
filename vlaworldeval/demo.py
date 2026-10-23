"""Weight-free example using the real typed loop, artifacts and audit code.

The moving square is a synthetic fixture, not a pretrained world model or a
robotics benchmark. It tests integration, not policy/world-model quality.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from adapters.base import VLAAdapter, WorldModelAdapter
from evaluation.reporting.artifact_checks import audit_job
from evaluation.reporting.artifacts import save_rollout_artifacts
from evaluation.contracts import ObservationBundle, WorldModelOutput
from evaluation.evaluator import PolicyWorldModelEvaluator
from evaluation.manifests import PairManifest
from evaluation.reporting.video_diagnostics import diagnose_artifact


class DemoPolicy(VLAAdapter):
    def reset(self, initial_state=None):
        pass

    def predict(self, image, instruction, state=None):
        # Respond to the generated observation rather than a fixed action list.
        pixels = np.argwhere(image["front"][..., 0] > 200)
        x = float(pixels[:, 1].mean())
        return np.tile([1.0 if x < 45 else -1.0, 0.25], (2, 1)).astype(np.float32)


class DemoWorldModel(WorldModelAdapter):
    def reset(self, image):
        self.position = np.array([12.0, 12.0])

    @staticmethod
    def render(position):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x, y = np.clip(position, 4, 58).astype(int)
        frame[y - 3:y + 3, x - 3:x + 3] = [255, 80, 30]
        return frame

    def step_output(self, action, *, output_views=(), state_schema=None):
        target = self.position + np.asarray(action).sum(axis=0)
        front = np.stack([self.render(self.position + (target - self.position) * t / 3) for t in (1, 2, 3)])
        self.position = target
        return WorldModelOutput(views={"front": front, "wrist": front[:, :, ::-1].copy()})

    def step(self, action):
        return self.step_output(action).terminal_views["front"]


def demo_manifest():
    action = {"dimension": 2, "horizon": 2, "representation": "synthetic_xy",
              "frame": "synthetic_pixels", "semantics": "delta", "rate_hz": 20}
    views = {"front": "front", "wrist": "wrist"}
    return PairManifest.from_dict({
        "pair_id": "synthetic_engineering_demo", "schema_version": 1,
        "vla": {"adapter": "DemoPolicy", "view_mapping": views, "view_mode": "mapping",
                "squeeze_single_frame": True, "action": action},
        "world_model": {"adapter": "DemoWorldModel", "view_mapping": views,
                        "view_mode": "mapping", "action": action,
                        "output_views": list(views), "video": {"fps": 30, "frames_per_chunk": 3}},
        "action_bridge": {"name": "identity", "source": action, "target": action},
        "state_provider": {"type": "none", "state_schema": "none"},
        "metadata": {"synthetic": True, "task_success_evaluated": False},
    })


def _hashes(views):
    return {name: hashlib.sha256(value[-1].tobytes()).hexdigest() for name, value in views.items()}


def run_demo(output, *, chunks=5):
    if type(chunks) is not int or not 1 <= chunks <= 1000:
        raise ValueError("Demo chunks must be an integer in [1,1000]")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite another run.
    def write(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    manifest = demo_manifest()
    job = {"chunks": chunks, "mode": "policy_loop", "state_provider": "none"}
    write("pair_manifest.json", manifest.to_dict())
    write("job.json", job)
    (output / "partial").mkdir()
    def on_step(trace):
        index = len(trace.world_outputs) - 1
        write(f"partial/chunk_{index:06d}.json", {
            "input_view_hashes": _hashes(trace.observations[-2].views),
            "output_terminal_view_hashes": _hashes(trace.world_outputs[-1].views),
            "observation_input_source": "initial_observation" if index == 0 else "previous_generated_terminal_views",
        })
    try:
        initial = DemoWorldModel.render(np.array([12, 12]))
        observation = ObservationBundle(views={"front": initial, "wrist": initial[:, ::-1].copy()},
                                        instruction="move the synthetic square")
        evaluator = PolicyWorldModelEvaluator(DemoPolicy(), DemoWorldModel(), pair_manifest=manifest)
        trace = evaluator.rollout_detailed({"observation_bundle": observation,
                                           "instruction": observation.instruction}, chunks, on_step=on_step)
        save_rollout_artifacts(output, trace.primary_video(), [a.values for a in trace.bridged_actions],
                              {"synthetic": True, "partial": False}, fps=manifest.world_model.video.fps,
                              native_action_chunks=[a.values for a in trace.native_actions],
                              states=[o.robot_state for o in trace.observations], view_videos=trace.view_videos(),
                              pair_manifest=manifest.to_dict(), timings=trace.timings)
        record = {"output": str(output), "status": "completed", "completed_chunks": chunks,
                  "world_model_calls": chunks, "vla_calls": chunks}
        audit = audit_job(record)
        diagnostics = diagnose_artifact(output)
        write("video_diagnostics.json", diagnostics)
        report = {"schema_version": 1, "status": "completed" if audit["ok"] else "failed_audit",
                  "synthetic": True, "gpu_used": False, "model_weights_used": False,
                  "task_success_evaluated": False, "job": record, "audit": audit}
        write("summary.json", report)
        if not audit["ok"]:
            raise ValueError(f"Synthetic artifact audit failed: {audit['failures']}")
        return report
    except BaseException as exc:
        write("status.json", {"status": "failed", "error": str(exc)})
        raise
