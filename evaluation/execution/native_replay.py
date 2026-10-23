"""World-model-only replay using the same persistent worker as policy suites.

Inputs are already converted to a declared native protocol. This entry never
guesses robot/action mappings or substitutes recorded future images for output.
Trusted adapter config executes local code, just like the normal runtime registry.
"""
from __future__ import annotations

import json
import hashlib
import importlib
import math
from pathlib import Path
import re
import time

import mediapy as media
import numpy as np

from adapters.environment import get_environment
from evaluation.contracts import WorldModelOutput, as_rgb_history
from evaluation.manifests import ActionSpec, VideoSpec
from evaluation.planning.registry import resolve_bindings, validate_bindings
from evaluation.execution.reliability import WorkerOptions, clean_worker_exit, exception_details
from evaluation.execution.client import ModelWorker
from evaluation.reporting.task_scores import _digest
from evaluation.execution.transport import read_packet, write_packet
from evaluation.reporting.video_diagnostics import diagnose_frames
from vlaworldeval.paths import project_root


def prepare_ctrl_packet(config_path: Path, dataset: Path, output: Path, *, episode=0, start=0, chunks=20):
    """Prepare Ctrl's native inputs using shared, independent camera admission."""
    from PIL import Image
    from evaluation.datasets.droid import load_droid_episode
    from evaluation.execution.transport import write_packet
    from evaluation.planning.registry import project_path
    config = json.loads(config_path.read_text())
    if config.get("adapter") != "CtrlWorldAdapter" or config.get("reset_operation") != "reset_official_replay":
        raise ValueError("prepare-replay currently supports only the CtrlWorld official reset profile")
    if config.get("action", {}).get("dimension") != 7 or config["action"].get("horizon") != 5:
        raise ValueError("Ctrl preparation requires 5x7 actions")
    if chunks < 1 or start < 0 or episode < 0:
        raise ValueError("Invalid dataset window")
    dataset = project_path(project_root(), str(dataset))
    if output.exists():
        raise FileExistsError(output)
    data = load_droid_episode(dataset, episode, start, chunks, protocol="ctrl_world")
    source_views = data.trial["observation_bundle"]["views"]
    names = ("exterior_1", "exterior_2", "wrist")
    if config["output_views"] != list(names):
        raise ValueError("Ctrl prepared camera order must be exterior_1/exterior_2/wrist")
    width, height = config["kwargs"].get("width", 320), config["kwargs"].get("height", 192)
    views = {name: np.asarray(Image.fromarray(source_views[f"droid.{name}"]).resize((width, height)))
             for name in names}
    packet = {"protocol": config["action"], "actions": np.stack(data.replay_actions),
              "reset_args": {"image": [views[name] for name in names], "first_action": data.replay_actions[0][0]},
              "initial_views": views, "source": {**data.provenance,
                  "conversion": "Ctrl-World absolute observed EEF XYZ/RPY/gripper, down_sample=3; repeated official memory seed"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_packet(output, packet)
    return {"ok": True, "model_inference_performed": False, "output": str(output),
            "action_shape": list(packet["actions"].shape), "source": packet["source"]}


def admit_native_replay(config_path: Path, packet_path: Path, *, suite_dir=None, resident_session=None):
    config = json.loads(config_path.read_text())
    fields = {"schema_version", "environment", "module", "adapter", "kwargs", "gpus", "memory_gib", "seed",
              "action", "video", "output_views", "reset_operation", "worker_options", "instruction_conditioning"}
    if (not isinstance(config, dict) or set(config) != fields
            or type(config["schema_version"]) is not int or config["schema_version"] not in (1, 2)):
        raise ValueError("Native replay config fields/schema do not match v1/v2")
    structured = config["schema_version"] == 2
    if not isinstance(config["module"], str) or not isinstance(config["environment"], str):
        raise ValueError("Native replay module/environment must be strings")
    if not isinstance(config["action"], dict) or not isinstance(config["video"], dict):
        raise ValueError("Native action/video specifications must be objects")
    if not isinstance(config["worker_options"], dict):
        raise ValueError("Native worker_options must be an object")
    if not re.fullmatch(r"adapters\.world_models\.[a-zA-Z_][a-zA-Z0-9_]*", config["module"]):
        raise ValueError("Replay module must be adapters.world_models.<module>")
    if not isinstance(config["adapter"], str) or not config["adapter"].isidentifier():
        raise ValueError("Invalid adapter class")
    spec = get_environment(config["environment"])
    if spec.scope != "world_model" or not spec.exists:
        raise ValueError("Replay requires an available registered world-model environment")
    gpus, budget = config["gpus"], config["memory_gib"]
    if (not isinstance(gpus, list) or any(type(g) is not int or g < 0 for g in gpus)
            or len(set(gpus)) != len(gpus)):
        raise ValueError("gpus must be unique nonnegative integer IDs")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("memory_gib must be finite and positive")
    if gpus and not (resident_session is not None and resident_session.owns_resources(gpus, budget)):
        from evaluation.execution.admission import resource_admission
        resource_admission({"world_model_gpus": gpus, "max_process_memory_gib_per_gpu": budget}, False)
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")
    if structured:
        if (set(config["action"]) != {"kind", "protocol"} or config["action"]["kind"] != "structured"
                or not isinstance(config["action"]["protocol"], str) or not config["action"]["protocol"]):
            raise ValueError("Structured replay requires an explicit nonnumeric protocol")
        adapter_type = getattr(importlib.import_module(config["module"]), config["adapter"])
        if (getattr(adapter_type, "native_condition_protocol", None) != config["action"]["protocol"]
                or not callable(getattr(adapter_type, "validate_replay_condition", None))):
            raise ValueError("Adapter does not declare this structured replay protocol")
        action = None
    else:
        for field in ("dimension", "horizon"):
            if type(config["action"].get(field)) is not int or config["action"][field] < 1:
                raise ValueError(f"Native action {field} must be a positive integer")
        rate = config["action"].get("rate_hz")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Native action rate_hz must be finite and positive")
        action = ActionSpec.from_dict(config["action"])
    video = VideoSpec.from_dict(config["video"])
    if not structured and config["adapter"] == "CtrlWorldAdapter":
        if action.temporal_layout != "inclusive" or video.temporal_layout != "inclusive" or action.horizon != 5 or video.frames_per_chunk != 5:
            raise ValueError("Legacy Ctrl timing is unsupported: use inclusive 5-sample action/video windows")
    names = config["output_views"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("output_views must be unique safe camera names")
    if config["reset_operation"] not in {"reset", "reset_official_replay"}:
        raise ValueError("Unknown native reset operation")
    options = WorkerOptions(**config["worker_options"])
    packet = read_packet(packet_path)
    if not isinstance(packet, dict) or set(packet) != {"protocol", "actions", "reset_args", "initial_views", "source"}:
        raise ValueError("Native packet requires protocol/actions/reset_args/initial_views/source")
    if packet["protocol"] != config["action"]:
        raise ValueError("Prepared action protocol differs from model config")
    if not isinstance(packet["source"], dict) or not packet["source"].get("conversion"):
        raise ValueError("Prepared packet must identify its source conversion")
    if structured:
        actions = packet["actions"]
        if not isinstance(actions, list) or not actions:
            raise ValueError("Structured replay needs a nonempty list of conditions")
        for chunk in actions:
            adapter_type.validate_replay_condition(chunk)
    else:
        actions = np.asarray(packet["actions"])
        if actions.ndim != 3 or actions.shape[0] < 1:
            raise ValueError("Replay actions must be nonempty [chunks,horizon,dimension]")
        if not (np.issubdtype(actions.dtype, np.integer) or np.issubdtype(actions.dtype, np.floating)):
            raise ValueError("Replay actions must be real numeric values")
        for chunk in actions:
            action.wrap(chunk, source="dataset_ground_truth")
    if not structured and config["adapter"] == "CtrlWorldAdapter":
        source = packet["source"]
        expected = source["start_frame"] + np.arange(len(actions))[:, None] * 12 + np.arange(5) * 3
        if hasattr(video, "final_chunk_valid_frames"):
            from evaluation.timeline import ctrl_tail_source_indices
            expected = ctrl_tail_source_indices(source, video, actions)
        if (source.get("source_frames_per_chunk") != 12 or source.get("temporal_layout") != "inclusive"
                or not np.array_equal(source.get("action_source_frame_indices"), expected)
                or not np.array_equal(actions[:-1, -1], actions[1:, 0])):
            raise ValueError("Ctrl actions require audited overlapping source indices and identical shared endpoints")
    if not isinstance(packet["initial_views"], dict) or set(packet["initial_views"]) != set(names):
        raise ValueError("Initial views must match output_views exactly")
    initial = {name: as_rgb_history(packet["initial_views"][name], name=name)[-1:]
               for name in names}
    reset_keys = {"value"} if config["reset_operation"] == "reset" else {"image", "first_action"}
    if not isinstance(packet["reset_args"], dict) or set(packet["reset_args"]) != reset_keys:
        raise ValueError("reset_args do not match the declared reset operation")
    validate_bindings(config["kwargs"])
    kwargs = resolve_bindings(config["kwargs"], project_root(), {}, {}, suite_dir or config_path.parent)
    for name, binding in config["kwargs"].items():
        if isinstance(binding, dict) and "project_path" in binding:
            path = Path(kwargs[name])
            if not path.exists() or (path.is_dir() and not any(path.iterdir())):
                raise ValueError(f"Missing or empty replay asset: {path}")
    if type(config["instruction_conditioning"]) is not bool:
        raise ValueError("instruction_conditioning must be boolean")
    if config["instruction_conditioning"]:
        instruction = packet["source"].get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Text-conditioned replay needs a source instruction")
        if "prompt" in kwargs:
            raise ValueError("Do not override prompt alongside instruction_conditioning")
        kwargs["prompt"] = instruction
    return config, spec, video, packet, initial, kwargs, options


def admit_image_anchors(path, packet, config):
    """Only recorded RGB changes; never accept replacement states/actions."""
    anchors = read_packet(path)
    if not isinstance(anchors, dict) or set(anchors) != {"images", "source_frame_indices", "source", "intervention"}:
        raise ValueError("Image anchor packet fields do not match the visual-only diagnostic")
    if anchors["intervention"] != "ground_truth_visual_context":
        raise ValueError("Unknown visual intervention")
    images = np.asarray(anchors["images"])
    first = packet["reset_args"].get("image", packet["reset_args"].get("value"))
    if isinstance(first, dict):
        first = first["observation/image"]
    first = np.asarray(first)
    if images.dtype != np.uint8 or images.shape != (len(packet["actions"]), *first.shape):
        raise ValueError("Image anchor count/dtype/geometry differs from the declared initial input")
    if not np.array_equal(images[0], first):
        raise ValueError("First anchor must equal the original initial image exactly")
    action = ActionSpec.from_dict(config["action"])
    stride = packet["source"]["fps"] * action.advance_steps / action.rate_hz
    if int(stride) != stride:
        raise ValueError("Image anchors require integer source-tick boundaries")
    expected = packet["source"]["start_frame"] + np.arange(len(images)) * int(stride)
    if not np.array_equal(anchors["source_frame_indices"], expected):
        raise ValueError("Image anchors do not align with the action chunk boundaries")
    if not isinstance(anchors["source"], dict) or not anchors["source"].get("conversion"):
        raise ValueError("Image anchors require explicit real-image source provenance")
    return anchors


def run_native_replay(config_path: Path, packet_path: Path, output: Path, *, admit_only=False, image_anchors=None,
                      reference_path=None, worker_session=None):
    config, spec, video, packet, initial, kwargs, options = admit_native_replay(
        config_path, packet_path, suite_dir=output, resident_session=worker_session)
    if config["schema_version"] == 2 and image_anchors is not None:
        raise ValueError("Image-anchor intervention is not declared for structured replay")
    anchors = admit_image_anchors(image_anchors, packet, config) if image_anchors is not None else None
    reference = None
    if reference_path is not None:
        if image_anchors is not None:
            raise ValueError("Ground-truth evaluation must not inject future reference images into inference")
        from evaluation.reporting.replay_metrics import validate_reference
        reference = validate_reference(read_packet(reference_path), config, packet, initial,
                                       input_sha256=_digest(packet_path))
    if admit_only:
        return {"ok": True, "executed": False, "chunks": len(packet["actions"]),
                "action_shape": None if config["schema_version"] == 2 else list(packet["actions"].shape),
                "reference_validated": reference is not None}
    output.mkdir(parents=True, exist_ok=False)
    worker = None
    report = {"schema_version": 1, "status": "running", "vla_calls": 0, "world_model_calls": 0,
              "task_success_evaluated": False, "input_sha256": _digest(packet_path),
              "config_sha256": _digest(config_path), "source": packet["source"], "files": {}, "timings": []}
    report.update(feedback_mode="autoregressive" if anchors is None else "ground_truth_visual_context",
                  image_anchors_sha256=None if anchors is None else _digest(image_anchors),
                  anchor_events=[], chunk_prediction_sha256=[], scientific_quality_verified=False)
    code_root = Path(__file__).resolve().parents[2]
    report["framework_code_sha256"] = {str(path.relative_to(code_root)): _digest(path)
        for package in ("adapters", "evaluation", "vlaworldeval")
        for path in sorted((code_root / package).rglob("*.py"))}
    # Preserve the exact admitted bytes, including whitespace, for hash-bound reports.
    (output / "config.json").write_bytes(config_path.read_bytes())
    frames = {name: [value] for name, value in initial.items()}
    try:
        worker_args = dict(module=config["module"], adapter=config["adapter"], kwargs=kwargs,
                           gpus=config["gpus"], memory_gib=config["memory_gib"], options=options)
        if worker_session is None:
            worker = ModelWorker(spec.python, output / "worker", **worker_args)
        else:
            worker, reused = worker_session.acquire(spec.python, **worker_args)
            report["worker_session"] = {"directory": str(worker_session.directory), "model_reused": reused,
                                        "completion_requires_clean_session_close": True}
        report["runtime"] = worker.ready
        if worker_session is not None:
            report["runtime"] = {**worker.ready, "load_seconds": 0.0 if reused else worker.ready["load_seconds"],
                                 "session_load_seconds": worker.ready["load_seconds"]}
        worker.call("seed", seed=config["seed"])
        worker.call(config["reset_operation"], **packet["reset_args"])
        for index, chunk in enumerate(packet["actions"]):
            if anchors is not None and index > 0:
                details = worker.call("reanchor_image", image=anchors["images"][index])
                report["anchor_events"].append({"chunk": index,
                    "source_frame": int(anchors["source_frame_indices"][index]), "details": details})
            started = time.perf_counter()
            result = worker.call("step", action=chunk, output_views=config["output_views"], state_schema=None)
            prediction = WorldModelOutput(views=result["views"], metadata=result["metadata"])
            video.validate_output(prediction, tuple(config["output_views"]))
            report["chunk_prediction_sha256"].append({name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in prediction.views.items()})
            for name, value in prediction.views.items():
                if value.shape[1:] != initial[name].shape[1:]:
                    raise ValueError(f"Generated view {name} changed image geometry")
                frames[name].append(value)
            report["world_model_calls"] += 1
            report["timings"].append(time.perf_counter() - started)
        report["video_diagnostics"] = {}
        reference_metrics = {}
        for name, chunks in frames.items():
            full = video.assemble(chunks[0], chunks[1:])
            if reference is not None:
                from evaluation.reporting.replay_metrics import score_view
                reference_metrics[name] = score_view(full, reference["views"][name])
            path = output / f"{name}.mp4"
            media.write_video(path, full, fps=video.fps)
            report["files"][path.name] = _digest(path)
            report["video_diagnostics"][name] = diagnose_frames(full, fps=video.fps)
            if reference is not None:
                comparison = output / f"{name}_prediction_vs_reference.mp4"
                media.write_video(comparison, np.concatenate([full, reference["views"][name]], axis=2), fps=video.fps)
                report["files"][comparison.name] = _digest(comparison)
        if reference is not None:
            report["reference_evaluation"] = {"reference_sha256": _digest(reference_path),
                "reference_source": reference["source"], "metrics": reference_metrics,
                "prediction_domain": "raw_uint8_before_video_encoding", "initial_frame_scored": False,
                "reference_sent_to_worker": False, "physical_action_tracking_verified": False}
        if config["schema_version"] == 2:
            write_packet(output / "conditions.npz", {"actions": packet["actions"], "protocol": packet["protocol"]})
            report["files"]["conditions.npz"] = _digest(output / "conditions.npz")
        else:
            np.save(output / "actions.npy", packet["actions"], allow_pickle=False)
            report["files"]["actions.npy"] = _digest(output / "actions.npy")
        report.update(status="completed" if worker_session is None else "generated_pending_worker_close",
                      frames_per_view=video.frame_count(len(packet["actions"])),
                      video_fps=video.fps, temporal_layout=video.temporal_layout,
                      raw_frames_per_chunk=video.frames_per_chunk, advance_frames_per_chunk=video.advance_frames,
                      saved_initial_frame_source="generated_window_start" if video.temporal_layout == "inclusive" else "recorded_initial_image")
    except BaseException as exc:
        report.update(status="failed", error=exception_details(exc))
        raise
    finally:
        if worker is not None:
            report["worker_memory_peak"] = getattr(worker, "memory", None)
            if worker_session is None:
                cleanup = worker.close()
                report["cleanup"] = cleanup
                if worker.failure or not clean_worker_exit(cleanup):
                    report.update(status="failed", error=worker.failure or {"code": "cleanup_failed"})
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        if worker_session is not None and "worker_session" in report:
            worker_session.track_report(output/"summary.json")
    return report
