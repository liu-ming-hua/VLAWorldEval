"""DROID three-camera ↔ Cosmos composite view conversion."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from PIL import Image

from adapters.observations.base import ViewBridge
from evaluation.contracts import ObservationBundle, WorldModelOutput, as_rgb_history
from evaluation.manifests import ViewBridgeSpec


DEFAULT_VIEWS = {
    "exterior_1": "droid.exterior_1",
    "exterior_2": "droid.exterior_2",
    "wrist": "droid.wrist",
    "composite": "droid.cosmos_composite",
}


def _resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR))


def compose_droid_cosmos_view(
    exterior_1: Any,
    exterior_2: Any,
    wrist: Any,
) -> np.ndarray:
    """Match Cosmos' DROID wrist-top/exteriors-bottom layout for each frame."""
    first = as_rgb_history(exterior_1, name="exterior_1")
    second = as_rgb_history(exterior_2, name="exterior_2")
    wrist_history = as_rgb_history(wrist, name="wrist")
    if not (len(first) == len(second) == len(wrist_history)):
        raise ValueError("DROID view histories must have the same number of frames")
    composed = []
    for left_frame, right_frame, wrist_frame in zip(first, second, wrist_history):
        height, width = wrist_frame.shape[:2]
        half_size = (width // 2, height // 2)
        bottom = np.concatenate(
            [_resize(left_frame, half_size), _resize(right_frame, half_size)], axis=1
        )
        composed.append(np.concatenate([wrist_frame, bottom], axis=0))
    return np.stack(composed)


def split_droid_cosmos_view(composite: Any) -> dict[str, np.ndarray]:
    """Recover three policy streams from a generated Cosmos composite video."""
    history = as_rgb_history(composite, name="droid.cosmos_composite")
    wrist_frames, exterior_1_frames, exterior_2_frames = [], [], []
    for frame in history:
        total_height, width = frame.shape[:2]
        wrist_height = int(round(total_height * 2.0 / 3.0))
        if wrist_height <= 0 or wrist_height >= total_height or width < 2:
            raise ValueError(f"Invalid DROID Cosmos composite shape {frame.shape}")
        wrist = frame[:wrist_height]
        bottom = frame[wrist_height:]
        midpoint = width // 2
        target_size = (width, wrist_height)
        wrist_frames.append(wrist)
        exterior_1_frames.append(_resize(bottom[:, :midpoint], target_size))
        exterior_2_frames.append(_resize(bottom[:, midpoint:], target_size))
    return {
        "wrist": np.stack(wrist_frames),
        "exterior_1": np.stack(exterior_1_frames),
        "exterior_2": np.stack(exterior_2_frames),
    }


class DroidCosmosViewBridge(ViewBridge):
    """Keep DROID policy views and derive/recover Cosmos' composed stream."""

    def __init__(self, spec: ViewBridgeSpec):
        super().__init__(spec)
        self.names = {**DEFAULT_VIEWS, **dict(spec.config.get("views", {}))}

    def _require(self, views: Mapping[str, Any], names: tuple[str, ...]) -> None:
        missing = [self.names[name] for name in names if self.names[name] not in views]
        if missing:
            raise ValueError(f"DROID view bridge is missing views: {', '.join(missing)}")

    def prepare_observation(self, observation: ObservationBundle) -> ObservationBundle:
        views = dict(observation.views)
        self._require(views, ("exterior_1", "exterior_2", "wrist"))
        views[self.names["composite"]] = compose_droid_cosmos_view(
            views[self.names["exterior_1"]],
            views[self.names["exterior_2"]],
            views[self.names["wrist"]],
        )
        self._composite_shape = views[self.names["composite"]].shape[1:3]
        return ObservationBundle(
            views=views,
            instruction=observation.instruction,
            robot_state=observation.robot_state,
            step_index=observation.step_index,
            timestamp=observation.timestamp,
            metadata={**observation.metadata, "view_bridge": self.spec.name},
        )

    def process_world_output(self, output: WorldModelOutput) -> WorldModelOutput:
        self._require(output.views, ("composite",))
        views = dict(output.views)
        # Official preprocessing does not upscale and pads bottom/right only.
        # Undo that geometry before splitting the camera streams.
        if hasattr(self, "_composite_shape"):
            history = as_rgb_history(views[self.names["composite"]], name="droid.cosmos_composite")
            original_h, original_w = self._composite_shape
            target_h, target_w = history.shape[1:3]
            if (target_h == original_h // 16 * 16
                    and target_w == original_w // 16 * 16):
                # Official action decoding crops latent padding before VAE
                # decode. Original dimensions are floored to 16 pixels.
                # Restore only missing bottom/right borders; do not stretch
                # the wrist/exterior boundary or claim missing pixels are real.
                views[self.names["composite"]] = np.pad(
                    history, ((0, 0), (0, original_h - target_h),
                              (0, original_w - target_w), (0, 0)), mode="edge",
                )
            else:
                scale = min(target_w / original_w, target_h / original_h, 1.0)
                content_h = int(original_h * scale + 0.5)
                content_w = int(original_w * scale + 0.5)
                views[self.names["composite"]] = np.stack([
                    _resize(frame[:content_h, :content_w], (original_w, original_h))
                    for frame in history
                ])
        split = split_droid_cosmos_view(views[self.names["composite"]])
        views[self.names["wrist"]] = split["wrist"]
        views[self.names["exterior_1"]] = split["exterior_1"]
        views[self.names["exterior_2"]] = split["exterior_2"]
        return WorldModelOutput(
            views=views,
            robot_state=output.robot_state,
            metadata={**output.metadata, "view_bridge": self.spec.name},
        )


def split_ctrl_decoded_video(decoded):
    """VAE NCHW -> RGB time/view/H/W/C, without reshaping channel order."""
    array = np.asarray(decoded)
    if array.ndim != 4 or array.shape[1] != 3 or array.shape[2] % 3:
        raise ValueError("Ctrl VAE output must be [T,3,3H,W]")
    frames, _, triple_height, width = array.shape
    return array.transpose(0, 2, 3, 1).reshape(frames, 3, triple_height // 3, width, 3)


class DroidCtrlViewBridge(ViewBridge):
    """Canonical DROID views at the released Ctrl-World image resolution."""

    def prepare_observation(self, observation):
        from dataclasses import replace
        views = {name: np.stack([np.asarray(Image.fromarray(frame).resize((320, 192)))
                                for frame in history])
                 for name, history in observation.views.items()}
        return replace(observation, views=views)


__all__ = [
    "DroidCosmosViewBridge",
    "compose_droid_cosmos_view",
    "split_droid_cosmos_view",
]
