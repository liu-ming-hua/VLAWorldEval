"""Adapter for the official A2World multi-view action-conditioned world model.

A2World consumes one initial RGB frame per camera and a 20-step, 14-D action
chunk. The ``libero`` checkpoint additionally consumes pose/history features
for A2World-sim. Those features condition generation; the released pipeline
does not return a robot-state prediction head. The adapter therefore exposes
``last_video`` and keeps ``last_state``/``last_policy_state`` unset.

The upstream runtime is imported lazily from
``third_party/a2world/world_model`` so importing the project registry does not
require Megatron, Transformer Engine, or CUDA extensions.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from adapters.base import WorldModelAdapter

ActionMapper = Callable[[np.ndarray], np.ndarray]


class A2WorldAdapter(WorldModelAdapter):
    """Wrap A2World or its history-aware A2World-sim checkpoint."""

    environment_key = "world_models.a2world"
    model_name = "A2World"
    action_dim = 14
    action_horizon = 20
    view_names = ("exterior_1", "wrist", "exterior_2")

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        base_checkpoints: str | Path,
        variant: str = "pretrained",
        repo_path: str | Path = "third_party/a2world/world_model",
        device: str | torch.device = "cuda",
        view_ids: Sequence[int] | None = None,
        width: int = 256,
        height: int = 256,
        fps: int = 10,
        history: bool = True,
        history_length: int = 20,
        num_sampling_steps: int = 35,
        guidance: float = 0.0,
        seed: int = 0,
        action_mapper: ActionMapper | None = None,
    ):
        if variant not in {"pretrained", "libero"}:
            raise ValueError("variant must be 'pretrained' or 'libero'")
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        if history_length < 1:
            raise ValueError("history_length must be positive")

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.base_checkpoints = Path(base_checkpoints).expanduser().resolve()
        self.variant = variant
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.device = torch.device(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.use_history = bool(history) and variant == "libero"
        self.history_length = int(history_length)
        self.num_sampling_steps = int(num_sampling_steps)
        self.guidance = float(guidance)
        self.seed = int(seed)
        self.action_mapper = action_mapper
        self.view_ids = tuple(int(value) for value in view_ids) if view_ids is not None else None

        self.pipeline: Any | None = None
        self.current_frames: list[np.ndarray] = []
        self.history_frames: list[list[np.ndarray]] = []
        self.history_actions = np.zeros((max(self.history_length - 1, 0), self.action_dim), dtype=np.float32)
        self.last_video: np.ndarray | None = None  # [T, V, H, W, 3], future frames only
        self.last_video_with_condition: np.ndarray | None = None
        self.last_actions: np.ndarray | None = None  # [20, 14], consumed by A2World
        self.last_state: None = None
        self.last_policy_state: None = None

        if not self.checkpoint.exists():
            raise FileNotFoundError(f"A2World checkpoint does not exist: {self.checkpoint}")
        if not self.base_checkpoints.is_dir():
            raise FileNotFoundError(f"A2World base checkpoints directory does not exist: {self.base_checkpoints}")
        self._add_upstream_to_path()
        self._load_pipeline()

    def _add_upstream_to_path(self) -> None:
        if not self.repo_path.is_dir():
            raise FileNotFoundError(
                f"A2World runtime not found at {self.repo_path}; run scripts/setup/a2world.sh"
            )
        source = str(self.repo_path)
        if source not in sys.path:
            sys.path.insert(0, source)

    def _load_pipeline(self) -> None:
        try:
            from a2world.checkpoints import EXPECTED_ACTION_DIM, inspect_checkpoint
            from cosmos_predict2.configs.base.config_multiview_action import (
                get_cosmos_predict2_multiview_pipeline,
                get_cosmos_predict2_multiview_state_pred_pipeline,
            )
            from cosmos_predict2.pipelines.video2world_multiview_action import (
                MultiviewVideo2WorldActionConditionedPipeline,
            )
            from cosmos_predict2.pipelines.video2world_multiview_action_state_pred import (
                MultiviewVideo2WorldActionConditionedStatePredPipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "A2World runtime dependencies are unavailable; run scripts/setup/a2world.sh "
                "inside environment/world_models/a2world"
            ) from exc

        checkpoint_info = inspect_checkpoint(self.checkpoint)
        if checkpoint_info.variant != self.variant:
            raise ValueError(
                f"A2World checkpoint variant is {checkpoint_info.variant!r}, "
                f"but adapter was configured with {self.variant!r}"
            )

        config_getter = (
            get_cosmos_predict2_multiview_state_pred_pipeline
            if self.variant == "libero"
            else get_cosmos_predict2_multiview_pipeline
        )
        config = config_getter(model_size="2B", resolution="720", fps=self.fps, views=3, frames=21)
        config.net.action_dim = EXPECTED_ACTION_DIM
        config.net.use_history = self.use_history
        config.guardrail_config.enabled = False
        config.prompt_refiner_config.enabled = False

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A2WorldAdapter requires CUDA; torch.cuda.is_available() is False")

        pipeline_type = (
            MultiviewVideo2WorldActionConditionedStatePredPipeline
            if self.variant == "libero"
            else MultiviewVideo2WorldActionConditionedPipeline
        )
        old_args = os.environ.get("COSMOS_PREDICT2_ARGS")
        os.environ["COSMOS_PREDICT2_ARGS"] = f"--checkpoints {self.base_checkpoints}"
        try:
            self.pipeline = pipeline_type.from_config(
                config=config,
                dit_path=str(self.checkpoint),
                use_text_encoder=False,
                device=str(self.device),
                torch_dtype=torch.bfloat16,
                load_ema_to_reg=False,
                load_prompt_refiner=False,
            )
        finally:
            if old_args is None:
                os.environ.pop("COSMOS_PREDICT2_ARGS", None)
            else:
                os.environ["COSMOS_PREDICT2_ARGS"] = old_args

    @staticmethod
    def _as_rgb(value: Any) -> np.ndarray:
        if isinstance(value, (str, Path)):
            value = Image.open(value)
        if isinstance(value, Image.Image):
            value = np.asarray(value.convert("RGB"))
        elif torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"A2World view must be an RGB HWC image, got {value.shape}")
        if np.issubdtype(value.dtype, np.floating) and float(np.nanmax(value)) <= 1.0 + 1e-3:
            value = value * 255.0
        value = np.clip(value, 0, 255).astype(np.uint8)
        image = Image.fromarray(value, mode="RGB")
        return np.asarray(image, dtype=np.uint8)

    def _parse_views(self, image: Any) -> list[np.ndarray]:
        if isinstance(image, dict):
            if "views" not in image:
                raise ValueError("A2World reset dict must contain a 'views' entry")
            if image.get("view_ids") is not None:
                self.view_ids = tuple(int(value) for value in image["view_ids"])
            image = image["views"]
        if isinstance(image, (str, Path, Image.Image)) or torch.is_tensor(image):
            image = [image]
        elif isinstance(image, np.ndarray) and image.ndim == 4:
            image = list(image)
        elif isinstance(image, np.ndarray):
            image = [image]
        if not isinstance(image, Sequence) or isinstance(image, (bytes, bytearray)):
            raise TypeError("A2World reset expects a sequence of 2 or 3 RGB views")
        views = [self._as_rgb(value) for value in image]
        views = [
            np.asarray(Image.fromarray(view, mode="RGB").resize((self.width, self.height), Image.Resampling.BILINEAR))
            for view in views
        ]
        if len(views) not in {2, 3}:
            raise ValueError(
                f"A2World requires 2 or 3 synchronized views, received {len(views)}; "
                "do not silently duplicate a missing camera"
            )
        if self.view_ids is None:
            self.view_ids = (2, 0) if len(views) == 2 else (2, 0, 3)
        if len(self.view_ids) != len(views):
            raise ValueError("view_ids must contain one camera embedding ID per view")
        if any(value < 0 or value > 3 for value in self.view_ids):
            raise ValueError("A2World view IDs must be in [0, 3]")
        return views

    def reset(self, image: Any) -> None:
        self.current_frames = self._parse_views(image)
        self.history_frames = [
            [frame.copy() for _ in range(self.history_length)] for frame in self.current_frames
        ]
        self.history_actions = np.zeros(
            (max(self.history_length - 1, 0), self.action_dim), dtype=np.float32
        )
        self.last_video = None
        self.last_video_with_condition = None
        self.last_actions = None
        self.last_state = None
        self.last_policy_state = None

    def _prepare_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        values = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"A2World action must have shape [T, A], got {values.shape}")
        if values.shape[1] != self.action_dim:
            if self.action_mapper is None:
                raise ValueError(
                    f"A2World expects 14-D actions, got {values.shape[1]}; provide action_mapper"
                )
            values = np.asarray(self.action_mapper(values), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.action_dim:
            raise ValueError(f"action_mapper must return [T, 14], got {values.shape}")
        if values.shape[0] < self.action_horizon:
            values = np.concatenate(
                [values, np.repeat(values[-1:], self.action_horizon - values.shape[0], axis=0)], axis=0
            )
        elif values.shape[0] > self.action_horizon:
            values = values[: self.action_horizon]
        return np.ascontiguousarray(values)

    @staticmethod
    def _pack_conditioning_frames(frames: list[np.ndarray]) -> torch.Tensor:
        packed = np.zeros((len(frames) * 21, *frames[0].shape), dtype=np.uint8)
        for view_index, frame in enumerate(frames):
            packed[view_index * 21] = frame
        return torch.from_numpy(packed).permute(3, 0, 1, 2).contiguous()

    @torch.inference_mode()
    def step(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        if self.pipeline is None or not self.current_frames:
            raise RuntimeError("Call A2WorldAdapter.reset(views) before step(action)")
        from einops import rearrange
        from a2world.history import action_path_features, sample_history_indices

        actions = self._prepare_action(action)
        history_tensor = None
        state = None
        if self.use_history:
            indices = sample_history_indices(self.history_actions, self.history_length)
            selected = [np.stack([view[index] for index in indices]) for view in self.history_frames]
            history_video = np.concatenate(selected, axis=0)
            history_tensor = torch.from_numpy(history_video).permute(3, 0, 1, 2).unsqueeze(0)
            state = action_path_features(self.history_actions, actions, indices)[None]

        conditioning = self._pack_conditioning_frames(self.current_frames)
        latent_view_indices = torch.tensor(self.view_ids, dtype=torch.int64).repeat_interleave(6)
        common = dict(
            video_tensor=conditioning,
            input_path=None,
            actions=actions,
            blacks=None,
            history=history_tensor,
            latent_view_indices=latent_view_indices,
            num_conditional_frames=1,
            guidance=self.guidance,
            n_views=len(self.current_frames),
            fps=self.fps,
            num_sampling_step=self.num_sampling_steps,
            seed=self.seed,
        )
        if self.variant == "libero":
            video = self.pipeline(state=state, **common)
        else:
            video = self.pipeline(**common)
        if video is None:
            raise RuntimeError("A2World rollout was rejected by the guardrail")

        per_view = rearrange(video, "b c (v t) h w -> b c v t h w", v=len(self.current_frames))
        generated = (
            (per_view[0].permute(1, 2, 3, 4, 0).cpu().float().numpy() + 1.0) * 127.5
        ).clip(0, 255).astype(np.uint8)
        self.last_video_with_condition = generated.copy()
        future = generated[:, 1:] if generated.shape[1] > 1 else generated
        self.last_video = future.transpose(1, 0, 2, 3, 4).copy()
        self.last_actions = actions.copy()
        self.current_frames = [view[-1].copy() for view in generated]

        if self.use_history:
            for view_index, view in enumerate(generated):
                self.history_frames[view_index].extend(view[1:])
            self.history_actions = np.concatenate([self.history_actions, actions], axis=0)
            max_history_actions = max(self.history_length - 1, 0)
            if max_history_actions:
                self.history_actions = self.history_actions[-max_history_actions:]
            for view_index in range(len(self.history_frames)):
                self.history_frames[view_index] = self.history_frames[view_index][-self.history_length :]

        # A2World predicts video only. There is intentionally no state head.
        self.last_state = None
        self.last_policy_state = None
        return self.current_frames[0]

    def rollout(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        """Generate a future multi-view video with layout ``[T,V,H,W,3]``."""
        self.step(action)
        if self.last_video is None:
            raise RuntimeError("A2World produced no video")
        return self.last_video

    def close(self) -> None:
        self.pipeline = None


__all__ = ["A2WorldAdapter"]
