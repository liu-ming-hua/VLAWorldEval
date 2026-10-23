"""Model-native preparation only; all inference belongs to the shared replay executor."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evaluation.execution.transport import read_packet, write_packet
from evaluation.reporting.task_scores import _digest
from vlaworldeval.paths import project_root


def indexed_video(path, indices, transform=None, *, decoder="ffmpeg", timeout=3600):
    if decoder == "ffmpeg":
        from evaluation.datasets.video import decode_indexed_video
        return decode_indexed_video(path, indices, transform, timeout=timeout)
    indices = np.asarray(list(indices), dtype=np.int64)
    if len(indices) == 0 or indices[0] < 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("Video indices must be nonempty and strictly increasing")
    if decoder != "opencv":
        raise ValueError("Unknown native video decoder")
    import cv2
    reader = cv2.VideoCapture(str(path))
    frames = []
    try:
        for index in indices:
            reader.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = reader.read()
            if not ok:
                raise ValueError(f"Missing real future frame {index}: {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(rgb) if transform else rgb)
    finally:
        reader.release()
    return np.stack(frames)


def save_reference(packet_path, output, views, indices, *, domain, split, episode,
                   conversion, condition_kind, frame_axis="source_frame", **extra):
    if output.exists():
        raise FileExistsError(output)
    source = {"dataset_domain": domain, "split": split, "episode": str(episode),
              "conversion": conversion, "condition_kind": condition_kind, "frame_axis": frame_axis,
              "ground_truth_kind": "recorded_execution", **extra}
    write_packet(output, {"schema_version": 1, "input_sha256": _digest(packet_path), "views": views,
                          "frame_indices": {k: np.asarray(indices[k], dtype=np.int64) for k in views},
                          "source": source})


def prepare_case(model, dataset, config_path, directory, *, start=0, chunks=1, episode=0, gpu=None):
    """No model generation or fallback samples; refuses overwriting existing evidence."""
    from PIL import Image
    root = project_root()
    dataset, config_path, directory = Path(dataset), Path(config_path), Path(directory)
    if not directory.resolve().is_relative_to(root) or directory.exists():
        raise ValueError("Prepare into a new project-local directory")
    directory.mkdir(parents=True)
    config = json.loads(config_path.read_text())
    if gpu is not None:
        if type(gpu) is not int or gpu not in (0, 1, 2, 3):
            raise ValueError("This benchmark only admits GPU0–3")
        config["gpus"] = [gpu]
    packet_path, reference_path = directory / "input.npz", directory / "reference.npz"
    common = {"domain": model, "split": "official_example_split_not_independently_verified",
              "episode": episode, "condition_kind": "recorded_command"}
    if model == "dreamtrue":
        from evaluation.datasets.dreamtrue import prepare_official
        from adapters.world_models.dreamtrue import PROTOCOL, VIEW_NAMES
        native, views = prepare_official(index=episode, raw_root=dataset, with_reference=True)
        config.update(schema_version=2, action={"kind": "structured", "protocol": PROTOCOL},
                      video={"fps": 5, "frames_per_chunk": len(views["head"]), "temporal_layout": "inclusive"})
        config["output_views"] = list(VIEW_NAMES)
        packet = {"protocol": config["action"], "actions": [native["condition"]],
                  "reset_args": {"value": native["reset"]}, "initial_views": native["reset"]["views"],
                  "source": {**native["condition"]["source"], "conversion": "Pinned official geometry and raw RGB loader, front padding removed only for scoring"}}
        write_packet(packet_path, packet)
        indices = {k: np.arange(len(v)) for k, v in views.items()}
        common.update(domain="droid", condition_kind="action_rendered_geometry",
                      episode=packet["source"]["source_key"], frame_axis="official_sample_tick",
                      conversion=packet["source"]["conversion"])
    elif model == "iws":
        import h5py
        from evaluation.datasets.iws import prepare_iws_packet
        from adapters.world_models.iws import prepare_image
        prepare_iws_packet(config_path, dataset, packet_path, start=start, chunks=chunks)
        ix = np.arange(start, start+chunks+1)
        with h5py.File(dataset) as f:
            views = {"camera_1_color": np.stack([prepare_image(f["obs/images/camera_1_color"][i]) for i in ix])}
        indices = {"camera_1_color": ix}
        common.update(domain="aloha_pusht", split="official_val", episode=dataset.name,
                      condition_kind="future_measured_kinematics",
                      conversion="Exact same camera crop/area resize as input, source tick+1 aligns each prediction")
    elif model == "irasim":
        from evaluation.datasets.irasim import prepare_sample
        import torch
        from torchvision.transforms.functional import resize
        prepare_sample(config_path, dataset, packet_path)
        views = {"primary": indexed_video(dataset / "videos/test/example.mp4", range(16),
            lambda a: resize(torch.from_numpy(a.copy()).permute(2, 0, 1), [288, 512],
                             antialias=True).permute(1, 2, 0).numpy())}
        indices = {"primary": np.arange(16)}
        common.update(domain="language_table", split="official_bundled_test_example",
                      conversion="Official frame0 latent; RGB torchvision antialias resize; future frames1..15",
                      timing_calibrated=False)
    elif model == "dreamdojo":
        from evaluation.datasets.dreamdojo import prepare_packet, episode_paths
        import torch
        from torchvision.transforms.v2 import functional as VF
        from torchvision.transforms import InterpolationMode
        prepare_packet(dataset, config_path, packet_path, start=start, chunks=chunks, episode=episode)
        packet = read_packet(packet_path)
        ix = packet["source"]["sampled_frame_indices"][1:]
        def transform(a):
            h, w = a.shape[:2]
            ch, cw = (h, int(h*4/3)) if w/h > 4/3 else (int(w*3/4), w)
            top, left = (h-ch)//2, (w-cw)//2
            x = torch.from_numpy(a[top:top+ch, left:left+cw].copy()).permute(2, 0, 1).float()/255
            x = VF.resize(x, [480, 640], interpolation=InterpolationMode.BILINEAR, antialias=True)
            return (x*255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        _, video_path = episode_paths(dataset, episode)
        views = {"ego_view": indexed_video(video_path, ix, transform, decoder="opencv")}
        indices = {"ego_view": np.asarray(ix)}
        common.update(domain="gr1", split="official_in_lab_eval", episode=f"{dataset.name}/{episode}",
                      conversion=f"Pinned source transform, stride2; initial selected source frame; next{chunks*12} real frames with global official action alias",
                      condition_kind="recorded_joint_targets_with_initial_state_alias")
    elif model == "worldgym":
        from evaluation.datasets.worldgym import load_worldgym_episode
        horizon = config["action"]["horizon"]
        data = load_worldgym_episode(dataset, episode, start, chunks, horizon, 20,
                                    instruction="recorded action replay", replay=True)
        first = data.trial["observation_bundle"]["views"]["bridge.primary"]
        packet = {"protocol": config["action"], "actions": np.stack(data.replay_actions),
                  "reset_args": {"value": first}, "initial_views": {"primary": first},
                  "source": {**data.provenance, "conversion": "Unmodified official 10D actions, skip initial action per upstream replay"}}
        write_packet(packet_path, packet)
        ix = np.arange(start, start+chunks*horizon+1)
        views = {"primary": indexed_video(dataset / f"{episode:09d}.mp4", ix)}
        indices = {"primary": ix}
        common.update(domain="bridge", split=dataset.name, conversion=packet["source"]["conversion"])
    elif model == "ctrl_world":
        from evaluation.execution.native_replay import prepare_ctrl_packet
        prepare_ctrl_packet(config_path, dataset, packet_path, episode=episode, start=start, chunks=chunks)
        packet = read_packet(packet_path)
        ix = np.arange(start, start+12*chunks+1, 3)
        views, indices = {}, {}
        for key, info in packet["source"]["videos"].items():
            name = key.removeprefix("droid.")
            shift = int(np.ceil(info["offset_seconds"]*15 - 1e-5))
            values = indexed_video(info["path"], ix+shift,
                lambda a: np.asarray(Image.fromarray(a).resize((320, 192))))
            views[name], indices[name] = values, ix+shift
        common.update(domain="droid", split="local_source_split_training_overlap_unknown",
                      conversion="Independent camera offsets; native inclusive 5-sample windows at source0,3,6,..; no guessed 1-second time warp",
                      condition_kind="future_measured_kinematics")
    else:
        raise ValueError("Model lacks a recorded native reference provider; no synthetic fallback")
    save_reference(packet_path, reference_path, views, indices, **common)
    effective = directory / "config.json"
    effective.write_text(json.dumps(config, indent=2))
    from evaluation.reporting.replay_metrics import validate_reference
    packet = read_packet(packet_path)
    initial = {k: np.asarray(v)[None] for k, v in packet["initial_views"].items()}
    validate_reference(read_packet(reference_path), config, packet, initial, input_sha256=_digest(packet_path))
    result = {"model": model, "input": str(packet_path), "reference": str(reference_path),
              "config": str(effective), "frames": {k: len(v) for k, v in views.items()}, "source": common}
    (directory / "prepared.json").write_text(json.dumps(result, indent=2))
    return result
