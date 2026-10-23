"""Strict full Language-Table preparation with reviewed upstream loader methods."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.execution.transport import read_packet, write_packet
from evaluation.reporting.task_scores import _digest
from vlaworldeval.asset_lock import asset_destination_lock
from vlaworldeval.assets import file_record
from vlaworldeval.atomic_json import publish_json_exclusive, replace_owned_json

COUNTS = (4562, 5243)
SOURCE_REVISION = "dfcbf85c27b2c5d041dbf5df3df272111e758b48"


def source_method(repo, name, namespace):
    path = Path(repo)/"dataset/dataset_2D.py"
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "Dataset_2D")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def short_selection(annotations, repo):
    """Compare every published window with actual upstream indexing, no fallback."""
    annotations, repo = Path(annotations), Path(repo)
    coverage = json.loads((annotations/"coverage.json").read_text())
    episodes, clips = COUNTS
    if (coverage.get("repository") != "fangqi/IRASim" or coverage.get("revision") != SOURCE_REVISION
            or coverage.get("test_episodes") != episodes or len(coverage.get("files", {})) != episodes
            or coverage.get("official_short_clips") != clips or len(coverage.get("clips", [])) != clips):
        raise ValueError("Complete pinned IRASim annotation selection required")
    method = source_method(repo, "_load_and_process_ann_file", {"json": json})
    context = SimpleNamespace(args=SimpleNamespace(num_frames=16), start_frame_interval=16, sequence_interval=1)
    selected, empty = [], []
    for episode, record in sorted(coverage["files"].items()):
        path = annotations/record["path"]
        if (len(episode) != 6 or not episode.isdigit() or record["path"] != f"annotation/test/{episode}.json"
                or _digest(path) != record["sha256"]):
            raise ValueError("Annotation identity/checksum differs from frozen selection")
        native = method(context, str(path))
        if not native:
            empty.append(episode)
        selected.extend({"episode": episode, "start": row["frame_ids"][0],
                         "source_frame_indices": row["frame_ids"]} for row in native)
    declared = sorted(coverage["clips"], key=lambda row: (row["episode"], row["start"]))
    if selected != declared or len(selected) != clips:
        raise ValueError("Frozen windows differ from upstream Dataset_2D indexing; no replacement")
    return {"schema_version": 1, "samples": selected, "test_episodes": episodes, "official_short_clips": clips,
            "episodes_without_short_window": empty, "selection_sha256": _digest(annotations/"coverage.json"),
            "source_method": "Dataset_2D._load_and_process_ann_file", "upstream_sha256": _digest(repo/"dataset/dataset_2D.py"),
            "all_indices_equal": True, "model_inference_performed": False, "full_test_set_completed": False}


def official_video_frames(repo, dataset, annotation, indices):
    """Call get_video, not __getitem__ (which silently replaces invalid samples)."""
    import torch
    import torchvision.transforms as transforms
    from decord import VideoReader, cpu
    from adapters.world_models.irasim import load_official_module
    native = load_official_module(Path(repo), "dataset/video_transforms.py", "_irasim_reference_transforms")
    context = SimpleNamespace(video_path=str(dataset), args=SimpleNamespace(normalize=True),
        preprocess=transforms.Compose([native.ToTensorVideo(), native.Resize_Preprocess((288, 512)),
                                      transforms.Normalize([.5]*3, [.5]*3, inplace=True)]))
    method = source_method(repo, "get_video", {"os": os, "np": np, "torch": torch,
                                              "VideoReader": VideoReader, "cpu": cpu})
    value = method(context, annotation, list(indices))
    if tuple(value.shape) != (len(indices), 3, 288, 512) or not torch.isfinite(value).all():
        raise ValueError("Unexpected native RGB loader output")
    return ((value/2+.5).clamp(0, 1)*255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def load_episode(dataset, episode):
    import torch
    from decord import VideoReader, cpu
    path = Path(dataset)/f"annotation/test/{episode}.json"
    annotation = json.loads(path.read_text())
    if (annotation.get("episode_id") != episode
            or annotation.get("video_path") != f"videos/test/{episode}/rgb.mp4"
            or annotation.get("latent_video_path") != f"latent_videos/test/{episode}/0.pt"):
        raise ValueError("Native episode path/identity differs")
    actions = np.asarray(annotation["actions"], dtype=np.float32)
    latents = torch.load(Path(dataset)/annotation["latent_video_path"], map_location="cpu", weights_only=True)["obs"]
    video_frames = len(VideoReader(str(Path(dataset)/annotation["video_path"]), ctx=cpu(0), num_threads=2))
    if (actions.ndim != 2 or actions.shape[1] != 2 or not np.isfinite(actions).all()
            or tuple(latents.shape) != (video_frames, 4, 36, 64)
            or not torch.isfinite(latents).all() or len(actions) != video_frames-1):
        raise ValueError("Source RGB/latent/action timeline differs; no substitute sample")
    return annotation, actions, latents


def long_action_windows(actions):
    """Upstream long-tail rule: repeat its first real action, then trim video."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 2 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Finite nonempty native XY action trajectory required")
    windows = []
    for start in range(0, len(actions), 15):
        segment = actions[start:start+15]
        if len(segment) < 15:
            segment = np.concatenate([segment, np.repeat(segment[:1], 15-len(segment), axis=0)])
        windows.append(segment.copy())
    return np.stack(windows), (len(actions)-1)%15+1


def compare_long_protocol(annotations, repo):
    """Execute upstream pure tail assignments, not its learned generator."""
    import torch
    import inspect
    from einops import repeat
    from adapters.world_models.irasim import IRASimAdapter
    selected = short_selection(annotations, repo)
    source = Path(repo)/"evaluate/generate_long_video.py"
    assignments = {node.targets[0].id: node for node in ast.walk(ast.parse(source.read_text()))
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id in {"false_action", "seg_false_action", "com_action"}
                   and (node.targets[0].id != "com_action" or (
                       isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                       and isinstance(node.value.func.value, ast.Name)
                       and node.value.func.value.id == "torch" and node.value.func.attr == "cat"))}
    if set(assignments) != {"false_action", "seg_false_action", "com_action"}:
        raise ValueError("Upstream native tail statements changed")
    code = compile(ast.Module(body=[assignments[name] for name in
        ("false_action", "seg_false_action", "com_action")], type_ignores=[]), str(source), "exec")
    coverage = json.loads((Path(annotations)/"coverage.json").read_text())
    episodes, calls, frames, padded = 0, 0, 0, 0
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Upstream JSON torch.tensor default must be float32")
    for record in coverage["files"].values():
        annotation = json.loads((Path(annotations)/record["path"]).read_text())
        raw = np.asarray(annotation["actions"], np.float32)
        windows, valid = long_action_windows(raw)
        action = torch.tensor(annotation["actions"])*np.array([20., 20.])
        expected = [action[start:start+15] for start in range(0, len(action)-14, 15)]
        offset = len(action)//15*15
        if offset < len(action):
            context = {"torch": torch, "repeat": repeat, "args": SimpleNamespace(num_frames=16),
                       "seg_action": action[offset:], "true_action": len(action)-offset}
            exec(code, context)
            expected.append(context["com_action"])
        actual = np.stack([IRASimAdapter.validate_actions(window) for window in windows])
        np.testing.assert_array_equal(actual, torch.stack(expected).float().numpy())
        if 1+15*(len(windows)-1)+valid != len(raw)+1:
            raise ValueError("Native long output crop differs from recorded timeline")
        episodes += 1
        calls += len(windows)
        frames += len(raw)
        padded += len(windows)*15-len(raw)
    return {"schema_version": 1, "test_episodes": episodes, "native_model_calls": calls,
        "real_future_frames": frames, "padded_model_actions": padded,
        "all_padded_scaled_actions_exact_equal": True, "all_cropped_lengths_equal": True,
        "annotation_manifest_sha256": selected["selection_sha256"],
        "upstream_long_script_sha256": _digest(source), "adapter_sha256": _digest(Path(inspect.getfile(IRASimAdapter))),
        "scope": "annotation/action/index parity only; no future media verification or learned model calls",
        "model_inference_performed": False, "full_test_set_completed": False}


def write_native_case(dataset, repo, config, target, sample, episode_data, provenance, *, mode):
    from evaluation.datasets.native_reference import save_reference
    from evaluation.reporting.replay_metrics import validate_reference
    annotation, actions, latents = episode_data
    indices = sample["source_frame_indices"]
    frames = official_video_frames(repo, dataset, annotation, indices)
    start = sample["start"]
    if mode == "long":
        windows, valid = long_action_windows(actions)
        if indices != list(range(len(actions)+1)) or start != 0 or config["video"]["final_chunk_valid_frames"] != valid:
            raise ValueError("Full long trajectory must retain every recorded action/frame")
    elif mode == "short":
        windows = actions[start:start+15][None]
    else:
        raise ValueError("Unknown IRASim native evaluation protocol")
    packet = {"protocol": config["action"], "actions": windows,
              "reset_args": {"value": {"image": frames[0], "latent": latents[start].float().numpy().copy()}},
              "initial_views": {"primary": frames[0]},
              "source": {**provenance, "initial_rgb_frame_index": start, "initial_latent_frame_index": start,
                  "source_frame_indices": indices, "future_latents_sent_to_worker": False,
                  "timing_calibrated": False, "time_unit": "native_dataset_tick", "benchmark_presentation_hz": 4,
                  "raw_action_count": len(indices)-1, "model_action_count": int(windows.shape[0]*15),
                  "tail_padding": "repeat_first_real_tail_action" if mode == "long" else "none",
                  "padded_predictions_scored": False,
                  "conversion": "Official decord/get_video float ToTensorVideo/resize/normalize; initial pre-encoded latent only; raw XY commands, adapter scales20 once; native long-tail padding trimmed before scoring"}}
    target.mkdir(parents=True)
    write_packet(target/"input.npz", packet)
    save_reference(target/"input.npz", target/"reference.npz", {"primary": frames}, {"primary": indices},
        domain="language_table", split=f"official_full_{mode}_test", episode=sample["episode"],
        condition_kind="recorded_command", conversion=packet["source"]["conversion"], timing_calibrated=False)
    write_packet(target/"reference.npz", read_packet(target/"reference.npz"), compressed=True)
    publish_json_exclusive(target/"config.json", config)
    validate_reference(read_packet(target/"reference.npz"), config, packet, {"primary": frames[:1]},
                       input_sha256=_digest(target/"input.npz"))


def write_short_case(*args):
    return write_native_case(*args, mode="short")


def write_long_case(*args):
    return write_native_case(*args, mode="long")


def prepare_full_short(*args):
    return prepare_full_test(*args, mode="short")


def prepare_full_long(*args):
    return prepare_full_test(*args, mode="long")


def prepare_full_test(root, dataset, annotations, config_path, output, *, mode):
    """Every official short window or every full test trajectory, never a subset."""
    from evaluation.planning.registry import project_path
    root = Path(root).resolve()
    dataset, annotations, config_path, output = [project_path(root, str(p))
                                                for p in (dataset, annotations, config_path, output)]
    repo = root/"third_party/irasim"
    selection = short_selection(annotations, repo)
    if mode not in {"short", "long"}:
        raise ValueError("Choose native short or long evaluation")
    if mode == "long":
        source = json.loads((annotations/"coverage.json").read_text())
        selection = {**selection, "samples": [{"episode": episode, "start": 0,
            "source_frame_indices": list(range(record["actions"]+1))}
            for episode, record in sorted(source["files"].items())],
            "short_window_indices_exact_equal": selection["all_indices_equal"],
            "source_method": "generate_long_video: sorted test annotations, every action from frame0; no short-window exclusions",
            "upstream_long_source_sha256": _digest(repo/"evaluate/generate_long_video.py")}
        selection.pop("all_indices_equal")
    expected_cases = COUNTS[1] if mode == "short" else COUNTS[0]
    assets = json.loads((dataset/"coverage.json").read_text())
    if (assets.get("status") != "extracted_not_inferred" or assets.get("test_episodes") != COUNTS[0]
            or assets.get("official_short_clips") != COUNTS[1]
            or assets.get("annotation_manifest_sha256") != selection["selection_sha256"]
            or assets.get("source_snapshot", {}).get("revision") != SOURCE_REVISION):
        raise ValueError("Full native assets must be extracted and bound to these annotations")
    config = json.loads(config_path.read_text())
    if (config.get("adapter") != "IRASimAdapter" or config.get("action", {}).get("horizon") != 15
            or config["action"].get("dimension") != 2 or config.get("seed") != 3407
            or config.get("kwargs", {}).get("num_inference_steps") != 50):
        raise ValueError("Full short protocol freezes native15x2 actions, seed3407 and50 denoising steps")
    code = [root/"evaluation/datasets/irasim_reference.py", root/"evaluation/execution/transport.py", repo/"dataset/dataset_2D.py",
            repo/"dataset/video_transforms.py", repo/"evaluate/generate_long_video.py",
            root/"evaluation/manifests.py", root/"evaluation/timeline.py"]
    frozen = {"assets_manifest_sha256": _digest(dataset/"coverage.json"),
              "selection_sha256": selection["selection_sha256"], "config_sha256": _digest(config_path),
              "protocol": mode, "code_sha256": {str(path.relative_to(root)): _digest(path) for path in code}}
    with asset_destination_lock(root, output):
        output.mkdir(parents=True, exist_ok=True)
        journal = output/"progress.json"
        if journal.exists():
            progress = json.loads(journal.read_text())
            if progress.get("frozen") != frozen:
                raise ValueError("Cannot resume changed native preparation")
        else:
            if any(output.iterdir()):
                raise ValueError("Choose an empty prepared directory")
            progress = {"schema_version": 1, "status": "preparing", "frozen": frozen,
                        "expected_cases": expected_cases, "completed": {}}
            publish_json_exclusive(journal, progress)
        cases, cached_episode, episode_data, provenance = [], None, None, None
        try:
            for index, sample in enumerate(selection["samples"]):
                if any(_digest(root/name) != sha for name, sha in frozen["code_sha256"].items()):
                    raise ValueError("Native preprocessing source changed during preparation")
                episode = sample["episode"]
                if episode != cached_episode:
                    names = [f"annotation/test/{episode}.json", f"videos/test/{episode}/rgb.mp4", f"latent_videos/test/{episode}/0.pt"]
                    records = {name: file_record(dataset/name) for name in names}
                    if any(records[name] != assets["files"].get(name) for name in names):
                        raise ValueError("Extracted episode bytes differ from full snapshot provenance")
                    episode_data = load_episode(dataset, episode)
                    provenance = {"episode": episode, "files": records, "frozen_preparation": frozen}
                    cached_episode = episode
                identifier = (f"irasim_lt_{episode}_start{sample['start']:04d}_seed3407" if mode == "short"
                              else f"irasim_lt_{episode}_full_long_seed3407")
                target = output/"cases"/identifier
                if identifier in progress["completed"]:
                    if any(_digest(target/name) != sha for name, sha in progress["completed"][identifier].items()):
                        raise ValueError("Previously prepared case changed")
                else:
                    runtime = {**config, "gpus": [index % 4]}
                    if mode == "long":
                        valid = (len(sample["source_frame_indices"])-2)%15+1
                        runtime["video"] = {**runtime["video"], "final_chunk_valid_frames": valid}
                    writer = write_short_case if mode == "short" else write_long_case
                    writer(dataset, repo, runtime, target, sample, episode_data, provenance)
                    progress["completed"][identifier] = {name: _digest(target/name)
                        for name in ("input.npz", "reference.npz", "config.json")}
                    progress.update(status="preparing")
                    progress.pop("error", None)
                    replace_owned_json(journal, progress)
                cases.append({"id": identifier, "prepared": str(target.relative_to(root))})
            if len(cases) != expected_cases or set(progress["completed"]) != {case["id"] for case in cases}:
                raise ValueError("Incomplete full native-test membership")
            plan = {"schema_version": 1, "scope": ("Every official Language-Table16-frame stride16 short test window" if mode == "short"
                else "Every full Language-Table test trajectory, including all real tail actions; padding outputs trimmed"),
                    "cases": cases, "unavailable": ({"long_trajectory_protocol": "separate protocol, not completed by these short windows",
                    "no_official_short_window": selection["episodes_without_short_window"]} if mode == "short" else {})}
            plans = {"plan.json": plan, **{f"plan_gpu{gpu}.json": {**plan, "cases": cases[gpu::4]} for gpu in range(4)}}
            for name, value in plans.items():
                path = output/name
                if path.exists():
                    if json.loads(path.read_text()) != value:
                        raise ValueError("Previously published native plan changed")
                else:
                    publish_json_exclusive(path, value)
            if not (output/"coverage.json").exists():
                publish_json_exclusive(output/"coverage.json", {**selection, "prepared_cases": len(cases),
                    "frozen": frozen, "protocol": mode, "full_long_trajectory_test_completed": False})
            progress["status"] = "prepared_not_evaluated"
            replace_owned_json(journal, progress)
        except BaseException as exc:
            progress.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error=f"{type(exc).__name__}: {exc}")
            replace_owned_json(journal, progress)
            raise
    return {"plan": str(output/"plan.json"), "prepared_cases": len(cases), "model_inference_performed": False}
