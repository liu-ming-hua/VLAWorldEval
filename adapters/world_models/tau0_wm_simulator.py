"""Client adapter for the released τ₀-WM Action-Conditioned Video Simulator.

The released ``simulator/`` checkpoint on Hugging Face is the 26-D whole-body
relative-EEF simulator. Its websocket API consumes synchronized RGB views as
``[V,C,H,W]`` in ``[-1, 1]`` and a 20-D dual-arm relative-EEF action chunk;
the server appends six fixed head/waist/velocity controls before feeding the
26-D model input. The chunk has at most 33 steps.
The server returns the terminal generated frame and a reward; it does not
return a robot state. Full generated videos are saved by the server when
requested in the payload, not returned through the default response.
The optional project transport patch exposes full video with return_full_video=True.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    import torch
from PIL import Image

from adapters.base import WorldModelAdapter


class Tau0WMSimulatorAdapter(WorldModelAdapter):
    """Wrap ``web_infer_utils.simulator.server_sim`` through its websocket client."""

    environment_key = "world_models.tau0_wm_simulator"
    model_name = "τ₀-WM Simulator (20-D API / 26-D model)"
    action_dim = 20
    action_horizon = 33
    supports_full_video = False
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
        return_full_video: bool = False,
        sim_path: str | Path | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.instruction = instruction
        self.height, self.width = map(int, image_size)
        self.num_inference_steps = int(num_inference_steps)
        self.execution_step = int(execution_step)
        if not 1 <= self.execution_step <= self.action_horizon:
            raise ValueError("execution_step must be in 1..33")
        self.sample_solver = sample_solver
        self.shift = float(shift)
        self.guide_scale = float(guide_scale)
        self.action_mapper = action_mapper
        self.return_full_video = bool(return_full_video)
        self.supports_full_video = self.return_full_video
        self.save_sim_video = bool(save_sim_video)
        self.sim_path = str(Path(sim_path).expanduser().resolve()) if sim_path else ""
        self._client: Any | None = None
        self._views: list[np.ndarray] | None = None
        self._needs_reset = True
        self.last_video: np.ndarray | None = None  # [T,V,H,W,3], uint8
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
        elif hasattr(value, "detach"):
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
        if isinstance(image, (str, Path, Image.Image)) or hasattr(image, "detach"):
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
                Image.fromarray(rgb).resize(
                    (self.width, self.height), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
            views.append(rgb)
        if len(views) != len(self.view_names):
            raise ValueError("This τ₀-WM profile requires two independently supplied RGB views")
        return views

    def reset(self, image: Any) -> None:
        self._views = self._parse_views(image)
        self._needs_reset = True
        self.last_video = None
        self.last_reward = None
        self.last_state = None
        self.last_policy_state = None

    def _prepare_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        arr = action.detach().float().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
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
        if arr.ndim != 2 or arr.shape[1] != self.action_dim:
            raise ValueError(f"action_mapper must return [T,{self.action_dim}], got {arr.shape}")
        if not 1 <= len(arr) <= self.execution_step:
            raise ValueError("Action length must be 1..execution_step; refusing padding or truncation")
        if not np.isfinite(arr).all():
            raise ValueError("Actions must be finite")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _terminal_views(value: Any) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[1] != 3:
            raise ValueError(f"τ₀-WM server terminal frame must be [V,3,H,W], got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("Server terminal frame must be finite")
        if np.issubdtype(arr.dtype, np.floating):
            if float(arr.min()) < -1.001 or float(arr.max()) > 1.001:
                raise ValueError("Official server float RGB must lie in [-1,1]")
            # The released server always returns normalized RGB, even if every pixel is positive.
            arr = (arr + 1.0) * 127.5
        elif arr.dtype != np.uint8:
            raise ValueError("Server frame must be normalized floating RGB or uint8")
        return np.clip(arr, 0, 255).round().astype(np.uint8).transpose(0, 2, 3, 1)[None]

    def step(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        if self._views is None:
            raise RuntimeError("Call Tau0WMSimulatorAdapter.reset(views) before step(action)")
        actions = self._prepare_action(action)
        self._load_client()
        obs = np.stack([view.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0 for view in self._views])
        prompt = self.instruction
        if self._needs_reset:
            prompt = "<reset>" + prompt
        response = self._client.infer(
            {
                "obs": obs,
                "prompt": prompt,
                "actions": actions,
                "num_inference_steps": self.num_inference_steps,
                "execution_step": actions.shape[0],
                "sample_solver": self.sample_solver,
                "shift": self.shift,
                "guide_scale": self.guide_scale,
                "save_sim_video": self.save_sim_video,
                "return_full_video": self.return_full_video,
                "sim_path": self.sim_path,
            }
        )
        results = response["results"]
        terminal = self._terminal_views(results[0])
        if terminal.shape[1:] != (len(self.view_names), self.height, self.width, 3):
            raise ValueError("Server returned unexpected view count or image geometry")
        video = terminal
        if self.return_full_video:
            if len(results) != 3 or not isinstance(results[2], dict):
                raise ValueError("Server lacks the full-video extension; apply scripts/setup/patches/tau0_full_video.patch and restart the server")
            payload = results[2]
            if (payload.get("protocol") != "vlaworldeval.tau0.video.v1"
                    or payload.get("layout") != "VCTHW"
                    or payload.get("range") != "[-1,1]"
                    or payload.get("includes_context_frame") is not True):
                raise ValueError("Unsupported Tau0 full-video protocol")
            raw = np.asarray(payload["video"])
            if raw.ndim != 5 or raw.shape[:2] != (len(self.view_names), 3) or raw.shape[2] < 2:
                raise ValueError("Full video must be [V,3,T,H,W] with context and future")
            frames = [self._terminal_views(raw[:, :, i])[0] for i in range(raw.shape[2])]
            decoded = np.stack(frames)
            if decoded.shape[2:] != (self.height, self.width, 3):
                raise ValueError("Full-video image geometry differs from request")
            if not np.array_equal(decoded[-1:], terminal):
                raise ValueError("Full-video terminal frame disagrees with terminal response")
            video = decoded[1:]  # Drop decoded context, retain all future frames.
        self.last_video = video
        self._views = [frame.copy() for frame in terminal[0]]
        self._needs_reset = False
        self.last_reward = np.asarray(results[1])
        self.last_state = None
        self.last_policy_state = None
        return terminal[0, 0]

    def step_output(self, action, *, output_views=(), state_schema=None):
        output = super().step_output(action, output_views=output_views, state_schema=state_schema)
        output.metadata.update(
            output_kind="future_video" if self.return_full_video else "terminal_frame_only",
            full_video_available=self.return_full_video,
            context_frame_removed=self.return_full_video,
            feedback="generated_terminal_frame",
        )
        return output

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
