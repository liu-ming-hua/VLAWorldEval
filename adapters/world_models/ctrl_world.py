"""Adapter for the official Ctrl-World action-conditioned world model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import importlib
import json
import sys
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image

from adapters.base import WorldModelAdapter

ActionMapper = Callable[[np.ndarray], np.ndarray]


class CtrlWorldAdapter(WorldModelAdapter):
    """Wrap Ctrl-World's official PyTorch inference implementation.

    The released DROID checkpoint expects the three RGB views vertically
    composed into one latent with spatial height ``3H/8``. The official VAE
    encodes each camera independently before stacking latent height; decoding
    also separates camera latents before the upstream chunked VAE calls.
    A call consumes five inclusive
    action frames (XYZ, RPY, gripper) and predicts five vertically-composed
    frames.
    """

    environment_key = "world_models.ctrl_world"

    def __init__(
        self,
        *,
        svd_model_path: str | Path,
        clip_model_path: str | Path,
        checkpoint_path: str | Path,
        repo_path: str | Path = "third_party/ctrl-world",
        data_stat_path: str | Path | None = None,
        action_mapper: ActionMapper | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        height: int = 192,
        width: int = 320,
        num_history: int = 6,
        num_frames: int = 5,
        num_inference_steps: int = 50,
        decode_chunk_size: int = 7,
        guidance_scale: float = 1.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        prompt: str | None = None,
    ):
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.svd_model_path = Path(svd_model_path).expanduser().resolve()
        self.clip_model_path = Path(clip_model_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.data_stat_path = Path(data_stat_path).expanduser().resolve() if data_stat_path else self.repo_path / "dataset_meta_info/droid/stat.json"
        self.action_mapper = action_mapper
        self.device = torch.device(device)
        self.dtype = dtype
        self.height = height
        self.width = width
        self.num_history = num_history
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.decode_chunk_size = decode_chunk_size
        self.guidance_scale = guidance_scale
        self.fps = fps
        self.motion_bucket_id = motion_bucket_id
        self.prompt = prompt
        self.last_video: np.ndarray | None = None
        self.last_latents: torch.Tensor | None = None
        self._current_latent: torch.Tensor | None = None
        # Ctrl-World does not consume the last six consecutive observations.
        # The official replay code keeps an append-only memory and retrieves
        # [0, 0, -8, -6, -4, -2] at every call.  Keeping this memory explicit
        # avoids silently changing the checkpoint's temporal protocol.
        self._memory_latents: list[torch.Tensor] = []
        self._memory_actions: list[np.ndarray] = []
        self._official_history_indices = (0, 0, -8, -6, -4, -2)

        for path, label in (
            (self.repo_path, "Ctrl-World repo_path"),
            (self.svd_model_path, "SVD model path"),
            (self.clip_model_path, "CLIP model path"),
            (self.checkpoint_path, "Ctrl-World checkpoint path"),
            (self.data_stat_path, "Ctrl-World data stat path"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CtrlWorldAdapter requires CUDA; torch.cuda.is_available() is False")

        self._import_runtime()
        self._load_model()

    def _import_runtime(self) -> None:
        repo = str(self.repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        self._ctrl_world = importlib.import_module("models.ctrl_world")
        self._pipeline_cls = importlib.import_module("models.pipeline_ctrl_world").CtrlWorldDiffusionPipeline

    def _load_model(self) -> None:
        stat = json.loads(self.data_stat_path.read_text(encoding="utf-8"))
        self.state_p01 = np.asarray(stat["state_01"], dtype=np.float32)
        self.state_p99 = np.asarray(stat["state_99"], dtype=np.float32)
        args = SimpleNamespace(
            svd_model_path=str(self.svd_model_path),
            clip_model_path=str(self.clip_model_path),
            num_history=self.num_history,
            num_frames=self.num_frames,
            action_dim=7,
            text_cond=self.prompt is not None,
            frame_level_cond=True,
            dtype=self.dtype,
        )
        self.model = self._ctrl_world.CrtlWorld(args)
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
        self.model.to(self.device).to(self.dtype).eval()
        self.pipeline = self.model.pipeline

    def _views(self, image: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> list[np.ndarray]:
        if isinstance(image, (list, tuple)):
            views = [np.asarray(v) for v in image]
        else:
            arr = np.asarray(image)
            if arr.ndim == 4:
                views = [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 3 and arr.shape[0] % 3 == 0 and arr.shape[0] > self.height:
                views = list(np.split(arr, 3, axis=0))
            else:
                raise ValueError("Ctrl-World reset expects three views: [3,H,W,3] or vertical [3H,W,3]")
        if len(views) != 3 or any(v.ndim != 3 or v.shape[-1] != 3 for v in views):
            raise ValueError("Ctrl-World released checkpoint requires exactly three RGB views")
        return [np.asarray(Image.fromarray(v.astype(np.uint8)).resize((self.width, self.height))) for v in views]

    @torch.inference_mode()
    def _encode_views(self, views: list[np.ndarray]) -> torch.Tensor:
        # Upstream get_traj_info encodes each camera separately, then stacks
        # latent height. Encoding a stitched RGB canvas is NOT equivalent.
        latents = []
        for view in views:
            batch = torch.from_numpy(view[None]).to(self.device, dtype=self.dtype)
            batch = batch.permute(0, 3, 1, 2) / 255.0 * 2 - 1
            latents.append(self.pipeline.vae.encode(batch).latent_dist.sample().mul_(self.pipeline.vae.config.scaling_factor))
        return torch.cat(latents, dim=2)

    def reset(self, image: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
        if isinstance(image, Mapping):
            if set(image) != {"views", "first_action", "instruction"}:
                raise ValueError("Ctrl reset bundle requires views/first_action/instruction")
            self.prompt = str(image["instruction"])
            self.reset_official_replay(image["views"], image["first_action"])
            return
        current = self._encode_views(self._views(image))
        self._reset_official_memory(current, np.zeros(7, dtype=np.float32))
        self.last_video = None
        self.last_latents = None

    def _reset_official_memory(self, latent: torch.Tensor, first_action: np.ndarray) -> None:
        """Seed the same 24-slot memory used by official Ctrl-World replay."""
        action = self._normalize_action(np.asarray(first_action, dtype=np.float32))
        seed_length = self.num_history * 4
        self._memory_latents = [latent.clone() for _ in range(seed_length)]
        self._memory_actions = [action.copy() for _ in range(seed_length)]
        self._current_latent = latent

    def reset_official_replay(
        self,
        image: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
        first_action: np.ndarray,
    ) -> None:
        """Initialize the released DROID replay protocol.

        The upstream script repeats the first latent/action 24 times, then
        retrieves six sparse entries.  It does not use six real pre-roll
        observations from the dataset.
        """
        current = self._encode_views(self._views(image))
        self._reset_official_memory(current, first_action)
        self.last_video = None
        self.last_latents = None

    def reset_with_history(
        self,
        images: list[np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...]],
        actions: np.ndarray,
    ) -> None:
        """Compatibility wrapper using the official sparse-memory protocol.

        Older callers passed six observations here.  Those observations are
        intentionally not used as six consecutive history slots: that differs
        from the released replay implementation and can produce corrupted
        frames.  The latest image/action seeds the repeated initial memory.
        """
        if len(images) != self.num_history:
            raise ValueError(f"Expected {self.num_history} history observations, got {len(images)}")
        raw_actions = np.asarray(actions, dtype=np.float32)
        if raw_actions.shape != (self.num_history, 7):
            raise ValueError(
                f"Expected history actions with shape [{self.num_history},7], got {raw_actions.shape}"
            )
        encoded = [self._encode_views(self._views(image)) for image in images]
        self._reset_official_memory(encoded[-1], raw_actions[-1])
        self.last_video = None
        self.last_latents = None

    def reanchor_image(self, image):
        """Replace current/terminal visual memory, preserving actions and RNG.

        Called at every diagnostic chunk boundary, so previous terminal slots
        are already real. Keep the official sparse indices and memory length;
        do not reseed a 24-slot history at each boundary.
        """
        if self._current_latent is None or len(self._memory_latents) < self.num_history * 4:
            raise RuntimeError("Reset before visual reanchor")
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        # VAE posterior sampling must not advance the diffusion RNG stream in
        # only one arm of this paired diagnostic.
        with torch.random.fork_rng(devices=devices):
            current = self._encode_views(self._views(image))
        self._current_latent = current
        self._memory_latents[-1] = current.clone()
        return {"step_index": len(self._memory_latents) - self.num_history * 4,
                "state_changed": False, "history_entries": len(self._memory_latents),
                "diffusion_rng_preserved": True,
                "visual_history": "initial repeated seed plus real chunk terminals"}

    def _prepare_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        arr = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Ctrl-World action must be [H,A], got {arr.shape}")
        if arr.shape[1] != 7:
            if self.action_mapper is None:
                raise ValueError(f"Ctrl-World expects 7-D Cartesian actions, got {arr.shape[1]}; provide action_mapper")
            arr = np.asarray(self.action_mapper(arr), dtype=np.float32)
        if arr.shape[1] != 7:
            raise ValueError(f"action_mapper must return [H,7], got {arr.shape}")
        if arr.shape[0] != self.num_frames:
            raise ValueError(f"Ctrl requires exactly {self.num_frames} inclusive samples; no implicit action padding/truncation")
        return np.stack([self._normalize_action(row) for row in arr], axis=0)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize one raw DROID Cartesian action using Ctrl-World statistics."""
        row = np.asarray(action, dtype=np.float32)
        return np.clip(2.0 * (row - self.state_p01) / (self.state_p99 - self.state_p01 + 1e-8) - 1.0, -1.0, 1.0)

    @torch.inference_mode()
    def step(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        if self._current_latent is None:
            raise RuntimeError("Call CtrlWorldAdapter.reset(image) before step(action)")
        future = self._prepare_action(action)
        if len(self._memory_latents) < self.num_history * 4:
            raise RuntimeError("Ctrl-World official memory was not initialized")
        history_indices = [
            index if index >= 0 else len(self._memory_latents) + index
            for index in self._official_history_indices
        ]
        history_actions = np.stack([self._memory_actions[index] for index in history_indices], axis=0)
        action_tensor = torch.from_numpy(np.concatenate([history_actions, future], axis=0)).unsqueeze(0).to(self.device, dtype=self.dtype)
        if self.prompt is not None:
            action_hidden = self.model.action_encoder(action_tensor, self.prompt, self.model.tokenizer, self.model.text_encoder, frame_level_cond=True)
        else:
            action_hidden = self.model.action_encoder(action_tensor)
        history = torch.cat([self._memory_latents[index] for index in history_indices], dim=0).unsqueeze(0)
        _, predicted = self._pipeline_cls.__call__(
            self.pipeline,
            image=self._current_latent,
            text=action_hidden,
            width=self.width,
            height=self.height * 3,
            num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps,
            decode_chunk_size=self.decode_chunk_size,
            max_guidance_scale=self.guidance_scale,
            fps=self.fps,
            motion_bucket_id=self.motion_bucket_id,
            history=history,
            output_type="latent",
            return_dict=False,
            frame_level_cond=True,
        )
        # Upstream Ctrl-World has returned both [B,C,F,H,W] and [B,F,C,H,W]
        # layouts across runtime versions. Normalize here before carrying the
        # terminal latent into the next autoregressive call.
        if predicted.ndim != 5:
            raise RuntimeError(f"Unexpected Ctrl-World latent shape: {tuple(predicted.shape)}")
        if predicted.shape[2] == self.num_frames:
            predicted_bcf = predicted
        elif predicted.shape[1] == self.num_frames:
            predicted_bcf = predicted.permute(0, 2, 1, 3, 4)
        else:
            raise RuntimeError(
                f"Cannot identify Ctrl-World temporal dimension in latent shape {tuple(predicted.shape)} "
                f"for num_frames={self.num_frames}"
            )
        stacked_latents = predicted_bcf[0].permute(1, 0, 2, 3)
        per_view = stacked_latents.reshape(
            self.num_frames, 4, 3, self.height // 8, self.width // 8
        ).permute(0, 2, 1, 3, 4)
        terminal_latent = predicted_bcf[:, :, -1].detach()
        self._current_latent = terminal_latent
        # Official replay appends only the terminal latent/action of the
        # generated chunk, not all five intermediate frames.
        self._memory_latents.append(terminal_latent.clone())
        self._memory_actions.append(future[-1].copy())
        self.last_latents = per_view.detach()
        decoded = self._decode_latents(per_view)
        self.last_video = decoded
        return decoded[-1, 0]

    @torch.inference_mode()
    def _decode_latents(self, per_view):
        """Official view-major VAE decoding, separate from model generation."""
        # Match forward_wm: view-major flatten then decode_chunk_size batches,
        # including its temporal chunk boundaries (not a vertical RGB decode).
        x = per_view.permute(1, 0, 2, 3, 4).flatten(0, 1)
        batches = []
        for offset in range(0, len(x), self.decode_chunk_size):
            chunk = x[offset:offset + self.decode_chunk_size] / self.pipeline.vae.config.scaling_factor
            batches.append(self.pipeline.vae.decode(chunk, num_frames=len(chunk)).sample)
        decoded = torch.cat(batches).reshape(3, self.num_frames, 3, self.height, self.width)
        decoded = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255).float().cpu().numpy().astype(np.uint8)
        decoded = decoded.transpose(1, 0, 3, 4, 2)
        return decoded

    def rollout(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        self.step(action)
        if self.last_video is None:
            raise RuntimeError("Ctrl-World produced no video")
        return self.last_video


__all__ = ["CtrlWorldAdapter"]
