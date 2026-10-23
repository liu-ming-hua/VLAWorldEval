"""Client adapter for the released τ₀-WM Action-Conditioned Video Simulator.

The released ``simulator/`` checkpoint on Hugging Face is the 26-D whole-body
relative-EEF simulator. Its websocket API consumes synchronized RGB views as
``[V,C,H,W]`` in ``[-1, 1]`` and a 20-D dual-arm relative-EEF action chunk;
the server appends six fixed head/waist/velocity controls before feeding the
26-D model input. The chunk has at most 33 steps.
The server returns the terminal generated frame and a reward; it does not
return a robot state. Full generated videos are saved by the server when
requested in the payload, not returned through the default response.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from adapters.base import WorldModelAdapter


class Tau0WMSimulatorAdapter(WorldModelAdapter):
    """Wrap ``web_infer_utils.simulator.server_sim`` through its websocket client."""

    environment_key = "world_models.tau0_wm_simulator"
    model_name = "τ₀-WM Simulator (20-D API / 26-D model)"
    action_dim = 20
    action_horizon = 33
    view_names = ("static", "gripper")

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8002,
        repo_path: str | Path = "third_party/tau-0-wm",
        instruction: str = "",
        image_size: tuple[int, int] = (192, 256),
        num_inference_steps: int = 25,
        execution_step: int = 33,
        sample_solver: str = "unipc",
        shift: float = 5.0,
        guide_scale: float = 1.0,
        action_mapper: Any | None = None,
        save_sim_video: bool = False,
        sim_path: str | Path | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.instruction = instruction
        self.height, self.width = map(int, image_size)
        self.num_inference_steps = int(num_inference_steps)
        self.execution_step = min(int(execution_step), self.action_horizon)
        self.sample_solver = sample_solver
        self.shift = float(shift)
        self.guide_scale = float(guide_scale)
        self.action_mapper = action_mapper
        self.save_sim_video = bool(save_sim_video)
        self.sim_path = str(Path(sim_path).expanduser().resolve()) if sim_path else ""
        self._client: Any | None = None
        self._views: list[np.ndarray] | None = None
        self._needs_reset = True
        self.last_video: np.ndarray | None = None  # [1,V,H,W,3], uint8
        self.last_reward: np.ndarray | float | None = None
        self.last_state = None
        self.last_policy_state = None

        if not self.repo_path.is_dir():
            raise FileNotFoundError(
                f"τ₀-WM source not found at {self.repo_path}; run the setup script first"
            )
        source = str(self.repo_path)
        if source not in sys.path:
            sys.path.insert(0, source)

    def _load_client(self) -> None:
        if self._client is not None:
            return
        try:
            from web_infer_utils.openpi_client.websocket_client_policy import (
                WebsocketClientPolicy,
            )
        except ImportError as exc:
            raise ImportError(
                "τ₀-WM client dependencies are unavailable; use the tau0_wm_simulator environment"
            ) from exc
        self._client = WebsocketClientPolicy(host=self.host, port=self.port)

    @staticmethod
    def _as_rgb(value: Any) -> np.ndarray:
        if isinstance(value, Image.Image):
            value = np.asarray(value.convert("RGB"))
        elif torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"τ₀-WM view must be an HWC RGB image, got {arr.shape}")
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0 + 1e-3:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _parse_views(self, image: Any) -> list[np.ndarray]:
        if isinstance(image, dict):
            image = image.get("views", image)
        if isinstance(image, (str, Path, Image.Image)) or torch.is_tensor(image):
            image = [image]
        elif isinstance(image, np.ndarray) and image.ndim == 4:
            image = list(image)
        elif isinstance(image, np.ndarray):
            image = [image]
        if not isinstance(image, Sequence):
            raise TypeError("τ₀-WM reset expects one or more RGB views")
        views = []
        for value in image:
            rgb = self._as_rgb(value)
            rgb = np.asarray(
                Image.fromarray(rgb, mode="RGB").resize(
                    (self.width, self.height), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
            views.append(rgb)
        if not views:
            raise ValueError("τ₀-WM reset received no views")
        return views

    def reset(self, image: Any) -> None:
        self._views = self._parse_views(image)
        self._needs_reset = True
        self.last_video = None
        self.last_reward = None
        self.last_state = None
        self.last_policy_state = None

    def _prepare_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        arr = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise ValueError(f"τ₀-WM action must have shape [T,A], got {arr.shape}")
        if arr.shape[1] != self.action_dim:
            if self.action_mapper is None:
                raise ValueError(
                    f"τ₀-WM simulator API expects {self.action_dim}-D dual-arm relative-EEF actions; got {arr.shape[1]}"
                )
            arr = np.asarray(self.action_mapper(arr), dtype=np.float32)
        if arr.shape[1] != self.action_dim:
            raise ValueError(f"action_mapper must return [T,{self.action_dim}], got {arr.shape}")
        if arr.shape[0] < self.action_horizon:
            arr = np.concatenate([arr, np.repeat(arr[-1:], self.action_horizon - arr.shape[0], axis=0)])
        return np.ascontiguousarray(arr[: self.action_horizon])

    @staticmethod
    def _terminal_views(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[1] != 3:
            raise ValueError(f"τ₀-WM server terminal frame must be [V,3,H,W], got {arr.shape}")
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr + 1.0) * 127.5 if float(np.nanmin(arr)) < 0 else arr * 255.0
        return np.clip(arr, 0, 255).round().astype(np.uint8).transpose(0, 2, 3, 1)[None]

    def step(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        if self._views is None:
            raise RuntimeError("Call Tau0WMSimulatorAdapter.reset(views) before step(action)")
        self._load_client()
        obs = np.stack([view.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0 for view in self._views])
        actions = self._prepare_action(action)
        prompt = self.instruction
        if self._needs_reset:
            prompt = "<reset>" + prompt
            self._needs_reset = False
        response = self._client.infer(
            {
                "obs": obs,
                "prompt": prompt,
                "actions": actions,
                "num_inference_steps": self.num_inference_steps,
                "execution_step": min(self.execution_step, actions.shape[0]),
                "sample_solver": self.sample_solver,
                "shift": self.shift,
                "guide_scale": self.guide_scale,
                "save_sim_video": self.save_sim_video,
                "sim_path": self.sim_path,
            }
        )
        results = response["results"]
        self.last_video = self._terminal_views(results[0])
        self.last_reward = np.asarray(results[1])
        self.last_state = None
        self.last_policy_state = None
        return self.last_video[0, 0]

    def rollout(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        self.step(action)
        if self.last_video is None:
            raise RuntimeError("τ₀-WM returned no terminal frame")
        return self.last_video

    def close(self) -> None:
        if self._client is not None:
            websocket = getattr(self._client, "_ws", None)
            if websocket is not None:
                websocket.close()
        self._client = None


__all__ = ["Tau0WMSimulatorAdapter"]
