"""Client adapter for the official GE-Sim 2.0 world-model server.

GE-Sim 2.0 is intentionally kept as a server-backed adapter. The released
checkpoint is coupled to the G01/OmniPicker embodiment and its closed-loop
conditioning needs an official episode bundle (three cameras, calibration and
robot-state files). The adapter therefore does not silently fabricate camera
calibration from a single RGB image.

The official server returns both imagined video and Pose-Expert state. The
generic ``WorldModelAdapter`` contract returns the terminal head-camera frame
from ``step`` for legacy evaluators; the complete multi-view video and state
remain available as ``last_video`` and ``last_state``.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adapters.base import WorldModelAdapter

ActionMapper = Callable[[np.ndarray], np.ndarray]


class GESimV2Adapter(WorldModelAdapter):
    """Wrap ``gesim.WorldModelEnv`` from GE-Sim 2.0."""

    environment_key = "world_models.ge_sim_v2"
    model_name = "GE-Sim 2.0"
    action_dim = 16
    state_dim = 16
    view_names = ("head", "left_wrist", "right_wrist")

    def __init__(
        self,
        *,
        server_url: str = "http://localhost:9000",
        episode_path: str | Path | None = None,
        repo_path: str | Path | None = "third_party/ge-sim-v2",
        chunk_size: int = 25,
        compress_actions: bool = True,
        conditioning: str = "action",
        task: str | None = None,
        timeout: float = 300.0,
        action_mapper: ActionMapper | None = None,
    ):
        self.server_url = server_url
        self.episode_path = Path(episode_path).expanduser().resolve() if episode_path else None
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self.chunk_size = int(chunk_size)
        self.compress_actions = bool(compress_actions)
        self.conditioning = conditioning
        self.task = task
        self.timeout = float(timeout)
        self.action_mapper = action_mapper

        self.env: Any | None = None
        self.last_video: np.ndarray | None = None  # (T, V, H, W, 3), uint8
        self.last_frames: np.ndarray | None = None  # (T, 3, V, H, W), float32 [0, 1]
        self.last_state: np.ndarray | None = None  # (T, 16), WM layout
        self.last_policy_state: np.ndarray | None = None  # (16,), policy layout
        self.last_observation: Any | None = None
        self.last_info: Any | None = None

        if self.conditioning not in ("action", "episode"):
            raise ValueError("conditioning must be 'action' or 'episode'")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        self._load_runtime()

    def _load_runtime(self) -> None:
        """Import the official client lazily so registry inspection stays light."""
        try:
            gesim = importlib.import_module("gesim")
        except ModuleNotFoundError as exc:
            if self.repo_path is None:
                raise ImportError(
                    "GE-Sim is not installed. Run scripts/setup/ge_sim_v2.sh "
                    "inside the GE-Sim environment."
                ) from exc
            source = self.repo_path / "src"
            if not source.is_dir():
                raise FileNotFoundError(f"GE-Sim source package not found: {source}") from exc
            source_str = str(source)
            if source_str not in sys.path:
                sys.path.insert(0, source_str)
            gesim = importlib.import_module("gesim")
        self._env_cls = gesim.WorldModelEnv

    def _make_env(self) -> Any:
        return self._env_cls(
            self.server_url,
            chunk_size=self.chunk_size,
            keep_frames=True,
            compress_actions=self.compress_actions,
            timeout=self.timeout,
        )

    @staticmethod
    def _as_episode_path(value: Any) -> Path | None:
        if isinstance(value, (str, Path)):
            path = Path(value).expanduser()
            if path.is_dir():
                return path.resolve()
        return None

    def reset(self, image: Any = None) -> None:
        """Reset from an official episode bundle.

        A plain RGB image is rejected because GE-Sim also needs camera
        calibration and G01 kinematics for action-conditioned conditioning.
        """
        episode = self._as_episode_path(image) or self.episode_path
        if episode is None:
            raise ValueError(
                "GE-Sim 2.0 reset requires an official episode bundle directory; "
                "pass episode_path=... or reset(bundle_path)."
            )
        if not episode.is_dir():
            raise FileNotFoundError(f"GE-Sim episode bundle does not exist: {episode}")

        self.close()
        self.env = self._make_env()
        self.last_observation = self.env.reset(
            episode,
            task=self.task,
            conditioning=self.conditioning,
        )
        self.last_video = None
        self.last_frames = None
        self.last_state = None
        self.last_policy_state = np.asarray(self.last_observation.state, dtype=np.float32).copy()
        self.last_info = None

    def _prepare_action(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        arr = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"GE-Sim action must have shape [H,A], got {arr.shape}")
        if arr.shape[1] != self.action_dim:
            if self.action_mapper is None:
                raise ValueError(
                    f"GE-Sim 2.0 expects {self.action_dim}-D absolute joint actions, "
                    f"got {arr.shape[1]}; provide action_mapper."
                )
            arr = np.asarray(self.action_mapper(arr), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.action_dim:
            raise ValueError(f"action_mapper must return [H,{self.action_dim}], got {arr.shape}")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _convert_video(frames: np.ndarray) -> np.ndarray:
        """Convert ``(T,3,V,H,W)`` float frames to ``(T,V,H,W,3)`` uint8."""
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 5 or frames.shape[1] != 3 or frames.shape[2] < 3:
            raise ValueError(f"GE-Sim frames must be (T,3,V,H,W), got {frames.shape}")
        return (
            np.clip(frames[:, :, :3], 0.0, 1.0) * 255.0
        ).round().astype(np.uint8).transpose(0, 2, 3, 4, 1)

    def step(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        """Generate a chunk and return the terminal head-camera RGB frame."""
        if self.env is None:
            raise RuntimeError("Call GESimV2Adapter.reset(bundle_path) before step(action)")
        actions = self._prepare_action(action)
        observation, _reward, state, info = self.env.step(actions)
        frames = np.asarray(info.frames, dtype=np.float32)
        if frames.shape[0] == 0:
            raise RuntimeError("GE-Sim returned no frames")

        self.last_frames = frames
        self.last_video = self._convert_video(frames)
        self.last_state = None if state is None else np.asarray(state, dtype=np.float32).copy()
        self.last_observation = observation
        self.last_policy_state = np.asarray(observation.state, dtype=np.float32).copy()
        self.last_info = info
        return self.last_video[-1, 0]

    def rollout(self, action: torch.Tensor | np.ndarray) -> np.ndarray:
        """Generate one chunk and return all views as ``(T,V,H,W,3)`` uint8."""
        self.step(action)
        if self.last_video is None:
            raise RuntimeError("GE-Sim produced no video")
        return self.last_video

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


__all__ = ["GESimV2Adapter"]
