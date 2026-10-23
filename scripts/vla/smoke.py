#!/usr/bin/env python3
"""Run one VLA adapter on a fixture derived from its official data domain."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datasets/vla_official_smoke/manifest.json"


def image(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    value = Image.open(ROOT / path).convert("RGB")
    if size is not None:
        value = value.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(value)


def droid_eef_9d(cartesian_position: np.ndarray) -> np.ndarray:
    # Exact convention used by Isaac-GR00T N1.7's official DROID downloader.
    from scipy.spatial.transform import Rotation

    correction = np.asarray([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
    cartesian_position = np.asarray(cartesian_position, dtype=np.float64)
    rotation = Rotation.from_euler("XYZ", cartesian_position[3:]).as_matrix()
    rot6d = (rotation @ correction)[:2, :].reshape(6)
    return np.concatenate([cartesian_position[:3], rot6d]).astype(np.float32)


def configure_memory_limit() -> None:
    # A800 has 80 GiB. 0.23 leaves a little headroom below the requested
    # 20-GiB per-process ceiling, including on the most occupied physical GPU.
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.23, device=0)
        torch.cuda.reset_peak_memory_stats(0)


def build_case(name: str, manifest: dict[str, Any]):
    if name in {"openvla", "spatialvla", "octo", "rt1_x"}:
        sample = manifest["bridge"]
        return image(sample["images"]["primary"]), sample["instruction"], None, sample

    if name in {"groot_n15", "groot_n17", "pi0_droid", "pi05_droid"}:
        sample = manifest["droid"]
        states = np.load(ROOT / sample["state"])
        current = 1
        if name == "groot_n15":
            images = {
                "video.exterior_image_1": image(
                    sample["images"]["exterior_image_1_left"][current], (256, 256)
                ),
                "video.exterior_image_2": image(
                    sample["images"]["exterior_image_2_left"][current], (256, 256)
                ),
                "video.wrist_image": image(
                    sample["images"]["wrist_image_left"][current], (256, 256)
                ),
            }
            state = {
                "state.eef_position": states["cartesian_position"][current, :3],
                "state.eef_rotation": states["cartesian_position"][current, 3:],
                "state.gripper_position": states["gripper_position"][current],
            }
        elif name == "groot_n17":
            images = {
                "exterior_image_1_left": np.stack(
                    [image(path) for path in sample["images"]["exterior_image_1_left"]]
                ),
                "wrist_image_left": np.stack(
                    [image(path) for path in sample["images"]["wrist_image_left"]]
                ),
            }
            state = {
                "eef_9d": droid_eef_9d(states["cartesian_position"][current]),
                "gripper_position": states["gripper_position"][current],
                "joint_position": states["joint_position"][current],
            }
        else:
            images = {
                "observation/exterior_image_1_left": image(
                    sample["images"]["exterior_image_1_left"][current]
                ),
                "observation/wrist_image_left": image(
                    sample["images"]["wrist_image_left"][current]
                ),
            }
            state = {
                "observation/joint_position": states["joint_position"][current],
                "observation/gripper_position": states["gripper_position"][current],
            }
        return images, sample["instruction"], state, sample

    if name == "smolvla":
        sample = manifest["libero"]
        # The official base checkpoint accepts missing configured cameras, but
        # its released normalization statistics are 6-D while LIBERO state is
        # 8-D. Keep the exact, explicit projection in the fixture manifest.
        images = {
            "observation.images.camera1": image(sample["images"]["image"]),
            "observation.images.camera2": image(sample["images"]["image2"]),
        }
        state = {"observation.state": np.load(ROOT / sample["state"])[:6]}
        return images, sample["instruction"], state, sample
    raise KeyError(name)


def create_adapter(name: str):
    if name == "openvla":
        from adapters.vla.openvla import OpenVLAAdapter

        return OpenVLAAdapter(str(ROOT / "checkpoints/vla/openvla-7b"))
    if name == "spatialvla":
        from adapters.vla.spatialvla import SpatialVLAAdapter

        return SpatialVLAAdapter(str(ROOT / "checkpoints/vla/spatialvla-4b-224-pt"))
    if name == "octo":
        from adapters.vla.octo import OctoAdapter

        return OctoAdapter()
    if name == "rt1_x":
        from adapters.vla.rt1 import RT1Adapter

        return RT1Adapter(ROOT / "checkpoints/vla/rt1-x/rt_1_x_tf_trained_for_002272480_step")
    if name == "groot_n15":
        from adapters.vla.groot import GrootN15Adapter

        return GrootN15Adapter()
    if name == "groot_n17":
        from adapters.vla.groot import GrootN17Adapter

        return GrootN17Adapter()
    if name == "pi0_droid":
        from adapters.vla.openpi import Pi0Adapter

        return Pi0Adapter()
    if name == "pi05_droid":
        from adapters.vla.openpi import Pi05Adapter

        return Pi05Adapter()
    if name == "smolvla":
        from adapters.vla.smolvla import SmolVLAAdapter

        return SmolVLAAdapter()
    raise KeyError(name)


def classify_error(name: str, exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if name == "groot_n17" and (
        "Cosmos-Reason2-2B" in message
        or "gated repo" in message.lower()
        or "couldn't find them in the cached files" in message.lower()
    ):
        return "blocked"
    return "failed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result: dict[str, Any] = {
        "model": args.model,
        "status": "failed",
        "physical_gpu": os.environ.get("VLA_PHYSICAL_GPU"),
        "memory_limit_gib": 0.23 * 80,
    }
    try:
        configure_memory_limit()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        inputs, instruction, state, sample = build_case(args.model, manifest)
        result.update(
            dataset=sample["dataset"],
            dataset_source=sample["source"],
            instruction=instruction,
        )
        adapter = create_adapter(args.model)
        adapter.reset(state)
        action = adapter.predict(inputs, instruction, state)
        values = action.detach().float().cpu().numpy()
        finite = bool(np.isfinite(values).all())
        result.update(
            status="passed" if finite else "failed",
            action_shape=list(values.shape),
            action_finite=finite,
            action_min=float(values.min()),
            action_max=float(values.max()),
            action_mean=float(values.mean()),
            action_std=float(values.std()),
        )
        try:
            import torch

            if torch.cuda.is_available():
                result["torch_peak_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
        except Exception:
            pass
    except Exception as exc:
        result.update(
            status=classify_error(args.model, exc),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    result["elapsed_seconds"] = time.monotonic() - started
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result["status"] in {"passed", "blocked"} else 1)


if __name__ == "__main__":
    main()
