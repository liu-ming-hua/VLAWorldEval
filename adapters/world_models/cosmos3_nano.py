"""Adapter for NVIDIA Cosmos3-Nano forward-dynamics generation.

Cosmos3-Nano is a local Diffusers pipeline rather than a separate HTTP server.
In ``forward_dynamics`` mode it consumes an initial RGB observation and a
domain-specific action trajectory, then returns future video frames. The
released model does not expose a robot-state prediction head; the adapter
therefore keeps ``last_state`` unset and advances its visual observation with
the terminal generated frame.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from adapters.base import WorldModelAdapter

ActionMapper = Callable[[np.ndarray], np.ndarray]


class Cosmos3NanoAdapter(WorldModelAdapter):
    """Wrap the official ``Cosmos3OmniPipeline`` in forward-dynamics mode."""

    environment_key = "world_models.cosmos3_nano"
    model_name = "Cosmos3-Nano"
    view_names = ("single_view",)

    DOMAIN_ACTION_DIMS = {
        "av": 9,
        "camera_pose": 9,
        "umi": 10,
        "bridge_orig_lerobot": 10,
        "droid_lerobot": 10,
        "robomind-franka": 10,
        "robomind-franka-dual": 20,
        "robomind-ur": 10,
        "galbot": 30,
        "agibotworld": 29,
        "agibot_gear_gripper": 29,
        "agibot_gear_gripper_ext": 29,
        "fractal": 10,
        "hand_pose": 57,
    }

    def __init__(
        self,
        *,
        model_path: str | Path,
        domain_name: str = "bridge_orig_lerobot",
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        resolution_tier: int = 480,
        view_point: str = "ego_view",
        fps: float = 5.0,
        num_inference_steps: int = 30,
        guidance_scale: float = 1.0,
        use_system_prompt: bool = False,
        enable_safety_check: bool = False,
        prompt: str = "",
        action_mapper: ActionMapper | None = None,
        seed: int | None = 0,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        self.domain_name = domain_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.resolution_tier = int(resolution_tier)
        self.view_point = view_point
        self.fps = float(fps)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.use_system_prompt = bool(use_system_prompt)
        self.enable_safety_check = bool(enable_safety_check)
        self.prompt = str(prompt)
        self.action_mapper = action_mapper
        self.seed = seed
        self.pipe: Any | None = None
        self.current_image: Image.Image | None = None
        self.last_video: np.ndarray | None = None  # (K, H, W, 3), uint8
        self.last_video_with_condition: np.ndarray | None = None
        self.last_actions: np.ndarray | None = None
        self.last_state: None = None
        self.last_policy_state: None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Cosmos3-Nano model path does not exist: {self.model_path}")
        if domain_name not in self.DOMAIN_ACTION_DIMS:
            raise ValueError(
                f"Unsupported Cosmos3 domain {domain_name!r}; choose from "
                f"{sorted(self.DOMAIN_ACTION_DIMS)}"
            )
        if self.resolution_tier not in (256, 480, 704, 720):
            raise ValueError("resolution_tier must be one of 256, 480, 704, 720")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Cosmos3NanoAdapter requires CUDA; torch.cuda.is_available() is False")
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            from diffusers import Cosmos3OmniPipeline
            from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        except ImportError as exc:
            raise ImportError(
                "Cosmos3 Diffusers support is unavailable; run scripts/setup/cosmos3_nano.sh"
            ) from exc

        # Diffusers accepts ``device_map="cuda"`` rather than an indexed
        # string such as ``"cuda:3"``.  The caller can select the physical
        # GPU with CUDA_VISIBLE_DEVICES; set the current device as well for
        # direct multi-GPU launches.
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.pipe = Cosmos3OmniPipeline.from_pretrained(
            str(self.model_path),
            torch_dtype=self.dtype,
            device_map="cuda" if self.device.type == "cuda" else None,
            enable_safety_checker=self.enable_safety_check,
        )
        if self.device.type != "cuda":
            self.pipe.to(self.device)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            flow_shift=10.0,
            use_karras_sigmas=False,
        )

    @staticmethod
    def _to_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise ValueError("Cosmos3 reset accepts one RGB view, not a multi-view batch")
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Cosmos3 reset expects an RGB HWC image, got {arr.shape}")
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.0 + 1e-3:
            arr = np.clip(arr * 255.0, 0.0, 255.0)
        return Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB")

    def reset(self, image: Any) -> None:
        self.current_image = self._to_image(image)
        self.last_video = None
        self.last_video_with_condition = None
        self.last_actions = None
        self.last_state = None
        self.last_policy_state = None

    def _prepare_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        arr = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise ValueError(f"Cosmos3 action must have shape [T,A], got {arr.shape}")
        if arr.shape[1] != self.DOMAIN_ACTION_DIMS[self.domain_name]:
            if self.action_mapper is None:
                expected = self.DOMAIN_ACTION_DIMS[self.domain_name]
                raise ValueError(
                    f"Cosmos3 domain {self.domain_name!r} expects {expected}-D actions, "
                    f"got {arr.shape[1]}; provide action_mapper."
                )
            arr = np.asarray(self.action_mapper(arr), dtype=np.float32)
        expected = self.DOMAIN_ACTION_DIMS[self.domain_name]
        if arr.ndim != 2 or arr.shape[1] != expected:
            raise ValueError(f"action_mapper must return [T,{expected}], got {arr.shape}")
        if arr.shape[0] < 16:
            arr = np.concatenate([arr, np.repeat(arr[-1:], 16 - arr.shape[0], axis=0)], axis=0)
        if arr.shape[0] > 400:
            arr = arr[:400]
        return np.ascontiguousarray(arr)

    @staticmethod
    def _video_uint8(video: Any) -> np.ndarray:
        if isinstance(video, list):
            arr = np.stack([np.asarray(frame) for frame in video], axis=0)
        elif torch.is_tensor(video):
            arr = video.detach().float().cpu().numpy()
        else:
            arr = np.asarray(video)
        if arr.ndim != 4:
            raise ValueError(f"Cosmos3 result video must be [T,H,W,3], got {arr.shape}")
        if arr.shape[-1] != 3 and arr.shape[1] == 3:
            arr = arr.transpose(0, 2, 3, 1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Cosmos3 result video must end in RGB channels, got {arr.shape}")
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.0 + 1e-3:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).round().astype(np.uint8)

    @torch.inference_mode()
    def step(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        if self.pipe is None or self.current_image is None:
            raise RuntimeError("Call Cosmos3NanoAdapter.reset(image) before step(action)")
        actions = self._prepare_action(action)
        from diffusers import CosmosActionCondition

        condition = CosmosActionCondition(
            mode="forward_dynamics",
            chunk_size=int(actions.shape[0]),
            domain_name=self.domain_name,
            resolution_tier=self.resolution_tier,
            raw_actions=torch.from_numpy(actions),
            image=self.current_image,
            view_point=self.view_point,
        )
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(self.seed))
        result = self.pipe(
            prompt=self.prompt,
            action=condition,
            fps=self.fps,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            use_system_prompt=self.use_system_prompt,
            output_type="np",
            enable_safety_check=self.enable_safety_check,
            generator=generator,
        )
        full_video = self._video_uint8(result.video)
        future = full_video[1:] if full_video.shape[0] == actions.shape[0] + 1 else full_video
        if future.shape[0] > actions.shape[0]:
            future = future[: actions.shape[0]]
        if future.shape[0] == 0:
            raise RuntimeError("Cosmos3 returned no future frames")
        self.last_video_with_condition = full_video
        self.last_video = future
        self.last_actions = actions.copy()
        self.current_image = Image.fromarray(future[-1], mode="RGB")
        return future[-1]

    def rollout(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        """Generate one chunk and return future RGB frames ``(K,H,W,3)``."""
        self.step(action)
        if self.last_video is None:
            raise RuntimeError("Cosmos3 produced no video")
        return self.last_video


__all__ = ["Cosmos3NanoAdapter"]
