"""Prepare the bundled official Language-Table sample for shared native replay."""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from torchvision.transforms.functional import resize


def prepare_sample(config_path: Path, dataset: Path, output: Path):
    from evaluation.execution.transport import write_packet, read_packet
    from vlaworldeval.assets import file_record
    provenance = json.loads((dataset / "sample.json").read_text())
    if (provenance.get("repository") != "bytedance/IRASim"
            or provenance.get("revision") != "c72b6dade6fcd65971e0aa8ab49ea39b15108c90"):
        raise ValueError("Official sample provenance differs")
    for name in ("annotation/test/example.json", "videos/test/example.mp4", "latent_videos/test/example.pt"):
        expected = provenance["files"][name]
        if file_record(dataset / name) != {key: expected[key] for key in ("size", "sha256")}:
            raise ValueError(f"Sample checksum differs: {name}")
    config = json.loads(config_path.read_text())
    if config["adapter"] != "IRASimAdapter" or config["action"]["horizon"] != 15 or config["action"]["dimension"] != 2:
        raise ValueError("Expected official IRASim Language-Table 15x2 protocol")
    annotation = dataset / "annotation/test/example.json"
    ann = json.loads(annotation.read_text())
    if ann.get("video_path") != "videos/test/example.mp4" or ann.get("latent_video_path") != "latent_videos/test/example.pt":
        raise ValueError("Official RGB/latent annotation paths differ")
    actions = np.asarray(ann["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2 or len(actions) < 15 or not np.isfinite(actions).all():
        raise ValueError("Invalid official Language-Table actions")
    # Latent sample is tensor-only: never enable pickle object loading here.
    latent_history = torch.load(dataset / "latent_videos/test/example.pt", map_location="cpu", weights_only=True)["obs"]
    latent = latent_history[0]
    if tuple(latent.shape) != (4, 36, 64):
        raise ValueError("Invalid initial sample latent")
    with imageio.get_reader(dataset / "videos/test/example.mp4") as reader:
        video_metadata = reader.get_meta_data()
        video_frames = reader.count_frames()
        image = reader.get_data(0)
    if len(latent_history) != video_frames or len(actions) != video_frames - 1:
        raise ValueError("Official RGB/latent/action frame counts differ")
    image = resize(torch.from_numpy(image.copy()).permute(2, 0, 1), [288, 512], antialias=True).permute(1, 2, 0).numpy()
    packet = {
        "protocol": config["action"], "actions": actions[:15][None],
        "reset_args": {"value": {"image": image, "latent": latent.numpy()}},
        "initial_views": {"primary": image},
        "source": {
            "dataset": "bytedance/IRASim bundled Language-Table example",
            "source_commit": "c72b6dade6fcd65971e0aa8ab49ea39b15108c90",
            "annotation": file_record(annotation), "fps": 4,
            "sample_provenance": provenance,
            "timing_calibrated": False,
            "time_unit": "native_dataset_tick",
            "benchmark_presentation_hz": 4,
            "physical_duration_seconds": None,
            "timing_note": "ActionSpec.rate_hz=4 is benchmark presentation only; never use for physical-time fairness comparisons",
            "container_video_fps": video_metadata.get("fps"),
            "source_video_frames": video_frames,
            "source_latent_frames": len(latent_history),
            "initial_rgb_frame_index": 0,
            "initial_latent_frame_index": 0,
            "sequence_interval": 1,
            "conversion": "First 15 raw XY deltas; adapter multiplies by 20 as Dataset_2D; official pre-encoded frame 0",
            "unused_tail_actions": len(actions) - 15,
            "prompt_supported": False, "state_prediction_supported": False
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = read_packet(output)
        if (previous["protocol"] != packet["protocol"] or previous["source"] != packet["source"]
                or not np.array_equal(previous["actions"], packet["actions"])
                or not np.array_equal(previous["initial_views"]["primary"], image)
                or not np.array_equal(previous["reset_args"]["value"]["image"], image)
                or not np.array_equal(previous["reset_args"]["value"]["latent"], latent.numpy())):
            raise ValueError("Existing prepared packet differs; choose a fresh output")
    else:
        write_packet(output, packet)
    return {"ok": True, "executed": False, "packet": str(output), "action_shape": [1, 15, 2]}


def check_initial_context(dataset: Path, vae_path: Path, output: Path):
    """CPU-only VAE reconstruction check of the official RGB/latent pairing."""
    from diffusers import AutoencoderKL
    from vlaworldeval.assets import verify_snapshot
    verify_snapshot(vae_path)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    latent = torch.load(dataset / "latent_videos/test/example.pt", weights_only=True, map_location="cpu")["obs"][0]
    vae = AutoencoderKL.from_pretrained(str(vae_path), subfolder="vae", local_files_only=True).eval()
    with torch.inference_mode():
        decoded = vae.decode(latent[None] / vae.config.scaling_factor).sample[0]
    decoded = ((decoded / 2 + 0.5).clamp(0, 1) * 255).permute(1, 2, 0).numpy()
    with imageio.get_reader(dataset / "videos/test/example.mp4") as reader:
        frames = np.stack([resize(torch.from_numpy(frame.copy()).permute(2, 0, 1), [288, 512], antialias=True)
                           .permute(1, 2, 0).numpy() for frame in reader])
    mse = ((frames.astype(np.float32) - decoded[None]) ** 2).mean(axis=(1, 2, 3))
    report = {"device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
              "initial_latent_index": 0, "annotated_initial_rgb_index": 0,
              "closest_rgb_frame_index": int(mse.argmin()),
              "initial_rgb_psnr_db": float(10 * np.log10(255**2 / mse[0])),
              "mse_by_rgb_frame": mse.tolist(),
              "note": "Lossy VAE reconstruction checks alignment; it is not pixel equality or generation quality"}
    imageio.imwrite(output / "decoded_initial_latent.png", decoded.astype(np.uint8))
    imageio.imwrite(output / "official_initial_rgb.png", frames[0])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
