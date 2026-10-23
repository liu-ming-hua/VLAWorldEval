"""One native DROID policy call, never a WM loop, task score or compatibility grant."""
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from vlaworldeval.paths import fresh_output_path


@dataclass(frozen=True)
class NativePolicy:
    module: str
    adapter: str
    environment: str
    checkpoint: str
    repository: str
    revision: str
    horizon: int
    gpus: tuple
    memory_gib: float
    source: str
    package: str


POLICIES = {
    "cosmos_droid": NativePolicy(
        "adapters.vla.cosmos_droid", "CosmosDroidPolicyAdapter", "vla.cosmos_policy_droid",
        "checkpoints/vla/cosmos3/nano-policy-droid", "nvidia/Cosmos3-Nano-Policy-DROID",
        "805c0d6d46196ecdc789213a6de72303c383895b", 32, (0, 1, 2, 3), 16,
        "third_party/cosmos-framework", "cosmos_framework"),
    "dreamzero_droid": NativePolicy(
        "adapters.vla.dreamzero", "DreamZeroDroidAdapter", "vla.dreamzero",
        "checkpoints/vla/dreamzero/droid", "GEAR-Dreams/DreamZero-DROID",
        "96ad344138c66e82536422432ad742f015784942", 24, (0, 1), 64,
        "third_party/dreamzero", "groot"),
}


def _validate_current_input(image, instruction, state):
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty DROID instruction is required")
    for key, value in image.items():
        value = np.asarray(value)
        if value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 1 or value.dtype != np.uint8:
            raise ValueError(f"{key} requires one current uint8 RGB [H,W,3] image")
    for key, width in (("observation/joint_position", 7), ("observation/gripper_position", 1)):
        value = np.asarray(state[key])
        if value.shape != (width,) or not np.isfinite(value).all():
            raise ValueError(f"{key} requires a finite measured [{width}] state")
        if width == 1 and np.any((value < 0) | (value > 1)):
            raise ValueError("Measured DROID gripper state must be in [0,1]")


def run_policy_check(root, model, output, *, dataset="datasets/cosmos3_droid/success",
                     episode=350, start=15, gpus=None, memory_gib=None, execute=False,
                     attention_backend=None, runtime_check=False):
    """Default is input-only. Execution requires memory and full snapshot checks.

    Only the current three images and measured joint/gripper state reach predict;
    no future recorded actions or images are passed to the policy.
    """
    from adapters.environment import get_environment
    from evaluation.datasets.droid import load_openpi_droid_episode
    from evaluation.execution.admission import ResourceAdmissionError, policy_resource_admission
    from evaluation.execution.client import ModelWorker
    from evaluation.execution.executor import fingerprint, git_checkout_evidence, write_json
    from evaluation.execution.reliability import WorkerOptions, clean_worker_exit, exception_details, run_signal_handlers
    from evaluation.execution.transport import write_packet
    from evaluation.planning.registry import project_path
    from vlaworldeval.assets import verify_snapshot
    from evaluation.validation.native_runtime import probe_native_runtime

    if model not in POLICIES:
        raise ValueError(f"Unknown native policy: {model}")
    spec = POLICIES[model]
    gpus = list(spec.gpus if gpus is None else gpus)
    memory_gib = spec.memory_gib if memory_gib is None else memory_gib
    if (not gpus or any(type(g) is not int or g not in range(4) for g in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError("Use unique GPU IDs from 0–3")
    if model == "dreamzero_droid" and len(gpus) not in (1, 2):
        raise ValueError("DreamZero supports IP1/2, not weight sharding")
    if isinstance(memory_gib, bool) or not isinstance(memory_gib, (int, float)) or not math.isfinite(memory_gib) or memory_gib <= 0:
        raise ValueError("memory_gib must be finite and positive")
    if any(type(v) is not int or v < 0 for v in (episode, start)):
        raise ValueError("episode/start must be nonnegative integers")
    if attention_backend not in (None, "TE", "torch") or (attention_backend is not None and model != "dreamzero_droid"):
        raise ValueError("attention_backend is an explicit DreamZero-only TE/torch selection")
    root = Path(root).resolve()
    output = fresh_output_path(root, output)
    dataset = project_path(root, str(dataset))
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "failed", "model": model,
              "scope": "native_policy_single_call_not_closed_loop", "task_success": None,
              "model_inference_performed": False, "asset_integrity_verified": False,
              "inference_passed": False,
              "world_model_calls": 0, "policy_calls": 0, "cleanup_passed": None,
              "execution_requested": execute, "expected_action_shape": [spec.horizon, 8],
              "runtime_check_requested": runtime_check or execute,
              "source_code": fingerprint(Path(__file__)), "output": str(output),
              "gpu_ids": gpus, "memory_gib_per_rank": memory_gib,
              "checkpoint": str(root / spec.checkpoint), "expected_revision": spec.revision,
              "dataset": str(dataset), "episode": episode, "start": start,
              "attention_backend": (attention_backend or "TE") if model == "dreamzero_droid" else None}
    worker = None
    with run_signal_handlers():
        try:
            code_root = Path(__file__).resolve().parents[2]
            report["framework_sources"] = [fingerprint(path) for package in
                ("adapters", "evaluation", "vlaworldeval") for path in sorted((code_root / package).rglob("*.py"))]
            data = load_openpi_droid_episode(dataset, episode, start, 1)
            views = data.trial["observation_bundle"]["views"]
            image = {f"observation/{name}": views[f"droid.{key}"] for name, key in (
                ("exterior_image_1_left", "exterior_1"), ("exterior_image_2_left", "exterior_2"),
                ("wrist_image_left", "wrist"))}
            state = {key: data.trial["initial_state"].values[key] for key in (
                "observation/joint_position", "observation/gripper_position")}
            instruction = data.trial["instruction"]
            # Weight-free validation shared by both released joint-space policies.
            _validate_current_input(image, instruction, state)
            environment = get_environment(spec.environment)
            if not Path(environment.python).is_file():
                raise FileNotFoundError(f"Missing isolated interpreter: {environment.python}")
            report.update(source=data.provenance, environment=str(environment.python),
                          input_shapes={"images": {k: list(v.shape) for k, v in image.items()},
                                        "state": {k: list(np.asarray(v).shape) for k, v in state.items()}})
            write_packet(output / "input.npz", {"image": image, "state": state,
                         "instruction": instruction, "source": data.provenance})
            if execute:
                # Reject busy devices before hashing tens of GB or starting any worker.
                report["resources"] = policy_resource_admission(gpus, memory_gib)
            if runtime_check or execute:
                imports = ["torch", "torchvision", "transformers"]
                if model == "dreamzero_droid" and attention_backend != "torch":
                    imports += ["transformer_engine.pytorch", "transformer_engine_torch"]
                if model == "cosmos_droid":
                    imports += ["natten"]
                report["runtime_probe"] = probe_native_runtime(environment.python, root=root,
                    source=root / spec.source, package=spec.package, imports=imports)
                report["upstream_source"] = git_checkout_evidence(root / spec.source)
                if not report["runtime_probe"]["ok"]:
                    report.update(status="runtime_rejected", error={"code": "runtime_import_failed",
                        "message": "Isolated runtime imports failed; inspect runtime_probe. No model was loaded."})
                    return report
            if not execute:
                report.update(status="runtime_checked" if runtime_check else "inputs_checked", unchecked=["GPU memory",
                              *([] if runtime_check else ["runtime dependencies"]), "GPU kernels and model operations",
                              "checkpoint integrity", "native model execution"])
                return report
            for relative, repository, revision in [(spec.checkpoint, spec.repository, spec.revision)] + (
                [("checkpoints/vla/dreamzero/umt5-tokenizer", "google/umt5-xxl",
                  "66cb9e7e85526fe440a945569e42c72fb6cbc0ad")] if model == "dreamzero_droid" else []):
                assets = verify_snapshot(root / relative)
                if assets["repository"] != repository or assets.get("revision") != revision:
                    raise ValueError(f"Snapshot identity/revision mismatch: {relative}")
                report.setdefault("verified_assets", {})[relative] = assets
            report["asset_integrity_verified"] = True
            kwargs = {"model_path": str(root / spec.checkpoint), "runtime_dir": str(output / "runtime")}
            if model == "dreamzero_droid":
                kwargs.update(tokenizer_path=str(root / "checkpoints/vla/dreamzero/umt5-tokenizer"),
                              repo_path=str(root / "third_party/dreamzero"),
                              attention_backend=attention_backend or "TE", seed=0)
            else:
                kwargs.update(decode_preview=False, seed=0)
            worker = ModelWorker(environment.python, output / "worker", module=spec.module,
                                 adapter=spec.adapter, kwargs=kwargs, gpus=gpus, memory_gib=memory_gib,
                                 options=WorkerOptions(shutdown_grace_seconds=30))
            report["ready"] = worker.ready
            worker.call("reset", value=None)
            report["model_inference_performed"] = True
            actions = np.asarray(worker.call("predict", image=image, instruction=instruction, state=state))
            report["policy_calls"] = 1
            if actions.shape != (spec.horizon, 8) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid native policy action shape/values: {actions.shape}")
            write_packet(output / "actions.npz", {"actions": actions})
            report.update(status="passed", inference_passed=True, action_shape=list(actions.shape),
                          raw_gripper_range=[float(actions[:, -1].min()), float(actions[:, -1].max())],
                          raw_gripper_out_of_range=bool(np.any((actions[:, -1] < 0) | (actions[:, -1] > 1))),
                          memory=worker.memory)
        except (Exception, KeyboardInterrupt) as exc:
            report.update(status="resource_rejected" if isinstance(exc, ResourceAdmissionError) else "failed",
                          error=exception_details(exc))
            if isinstance(exc, KeyboardInterrupt):
                report["status"] = "interrupted"
                raise
        finally:
            cleanup_interruption = None
            if worker is not None:
                try:
                    cleanup = worker.close()
                    report["cleanup"] = cleanup
                    report["cleanup_passed"] = not worker.failure and clean_worker_exit(cleanup)
                except (Exception, KeyboardInterrupt) as exc:
                    report.update(cleanup_passed=False, cleanup_error=exception_details(exc))
                    if isinstance(exc, KeyboardInterrupt):
                        cleanup_interruption = exc
                        report["status"] = "interrupted"
                if not report["cleanup_passed"] and report["status"] != "interrupted":
                    report["status"] = "failed"
            write_json(output / "report.json", report)
            if cleanup_interruption is not None:
                raise cleanup_interruption
    return report
