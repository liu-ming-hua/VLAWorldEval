"""Paper SVD PlayWorld, using the pinned OpenWorld VidWM implementation."""
from __future__ import annotations

from collections.abc import Mapping
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from adapters.base import WorldModelAdapter

SOURCE_REVISION = "457942696e322cb24b00450d854fbf9875e01888"
VIEW_NAMES = ("exterior_right", "exterior_left", "wrist")


class PlayWorldAdapter(WorldModelAdapter):
    """Three-view DROID SVD flow-map model; no learned robot-state output.

    Input is exactly 15 raw absolute XYZ/RPY/gripper states at 15 Hz (or five
    already-sampled states with action_downsample=1). The official runtime
    selects [0,3,6,9,12], normalizes, and predicts five frames. No implicit
    action padding, delta conversion, or policy compatibility is claimed.
    """

    environment_key = "world_models.playworld"
    view_names = VIEW_NAMES

    def __init__(self, *, checkpoint_path="checkpoints/world_models/playworld/wm/checkpoint-120000.pt",
                 svd_model_path="checkpoints/world_models/playworld/svd",
                 clip_model_path="checkpoints/world_models/playworld/clip",
                 repo_path="third_party/playworld", device="cuda", prompt=None,
                 action_downsample=3, num_inference_steps=4, decode_chunk_size=5,
                 memory_gib=18.0):
        if type(action_downsample) is not int or action_downsample not in (1, 3):
            raise ValueError("PlayWorld supports action_downsample=1 or 3 only")
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if type(decode_chunk_size) is not int or decode_chunk_size not in (1, 5):
            raise ValueError("decode_chunk_size must be 1 or 5 to keep temporal VAE camera groups separate")
        if isinstance(memory_gib, bool) or not np.isfinite(memory_gib) or memory_gib <= 0:
            raise ValueError("memory_gib must be finite and positive")
        self.repo_path = Path(repo_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.svd_model_path = Path(svd_model_path).resolve()
        self.clip_model_path = Path(clip_model_path).resolve()
        for path in (self.repo_path, self.checkpoint_path, self.svd_model_path, self.clip_model_path):
            if not path.exists():
                raise FileNotFoundError(path)
        revision = subprocess.check_output(["git", "-C", str(self.repo_path), "rev-parse", "HEAD"], text=True).strip()
        if revision != SOURCE_REVISION:
            raise ValueError(f"PlayWorld source must be pinned at {SOURCE_REVISION}; got {revision}")
        if subprocess.check_output(["git", "-C", str(self.repo_path), "status", "--porcelain"], text=True).strip():
            raise ValueError("PlayWorld source is modified; refusing unverified runtime")
        import torch
        self.device = torch.device(device)
        self._inference_context = torch.inference_mode
        self.prompt = prompt
        self.action_downsample = action_downsample
        self.memory_gib = memory_gib
        self._state = None
        self.last_video = None
        self.last_latents = None
        self._guard_gpu()
        self._load_runtime(num_inference_steps, decode_chunk_size)

    def _guard_gpu(self):
        import torch
        if self.device.type != "cuda":
            return
        if not torch.cuda.is_available():
            raise RuntimeError("PlayWorld GPU inference requires CUDA")
        index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        if free < (self.memory_gib + 3.5) * 2**30:
            raise RuntimeError(f"PlayWorld needs {self.memory_gib + 3.5:g} GiB free; observed {free / 2**30:.2f}")
        # Respect a tighter cap already established by the persistent worker.
        fraction = min(torch.cuda.get_per_process_memory_fraction(index), self.memory_gib * 2**30 / total)
        torch.cuda.set_per_process_memory_fraction(fraction, index)

    def _load_runtime(self, steps, decode_chunk_size):
        import torch
        for package in ("openworld", "vidwm"):
            for name, module in tuple(sys.modules.items()):
                if name == package or name.startswith(package + "."):
                    source = getattr(module, "__file__", None)
                    if source and not Path(source).resolve().is_relative_to(self.repo_path):
                        raise RuntimeError(f"Conflicting {package} import: {source}; use the isolated worker")
        sys.path.insert(0, str(self.repo_path))
        runtime = importlib.import_module("openworld.world_models.vidwm_world_model")
        cfg = runtime.VidWMConfig(
            repo_path=str(self.repo_path / "external/vidwm"),
            svd_model_path=str(self.svd_model_path), clip_model_path=str(self.clip_model_path),
            device=str(self.device), action_downsample=self.action_downsample,
            num_inference_steps=steps, decode_chunk_size=decode_chunk_size,
            view_order=self.view_names, action_stats_preset="droid",
        )
        self.model = runtime.VidWMWorldModel(cfg)
        self.model.load_checkpoint(str(self.checkpoint_path))
        # Upstream loads with strict=False. Reject missing/unexpected/shape
        # mismatched learned tensors instead of permitting random parameters.
        checkpoint = torch.load(self.checkpoint_path, map_location="meta", weights_only=True, mmap=True)
        for prefix, module in (("unet.", self.model.pipeline.unet), ("action_encoder.", self.model.action_encoder)):
            actual = {key[len(prefix):]: tuple(value.shape) for key, value in checkpoint.items() if key.startswith(prefix)}
            expected = {key: tuple(value.shape) for key, value in module.state_dict().items()}
            if actual != expected:
                missing = sorted(set(expected) - set(actual))
                extra = sorted(set(actual) - set(expected))
                raise ValueError(f"Incomplete PlayWorld {prefix} weights: missing={missing[:8]} extra={extra[:8]}; check shapes")
        del checkpoint
        self.model.pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)

    @staticmethod
    def _actions(value, rows):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (rows, 7) or not np.isfinite(array).all():
            raise ValueError(f"Expected finite raw absolute XYZ/RPY/gripper [{rows},7], got {array.shape}")
        return array

    @staticmethod
    def _views(value):
        if isinstance(value, Mapping):
            if set(value) != set(VIEW_NAMES):
                raise ValueError(f"PlayWorld requires named views {VIEW_NAMES}")
            frames = [value[name] for name in VIEW_NAMES]
        elif isinstance(value, (list, tuple)):
            frames = list(value)
        else:
            value = np.asarray(value)
            if value.ndim == 4 and value.shape[0] == 3:
                frames = list(value)
            elif value.shape == (576, 320, 3):
                frames = list(np.split(value, 3, axis=0))
            else:
                raise ValueError("PlayWorld reset requires three RGB views or a [576,320,3] stack")
        if len(frames) != 3:
            raise ValueError("PlayWorld requires exactly three real camera views")
        result = {}
        for name, frame in zip(VIEW_NAMES, frames):
            frame = np.asarray(frame)
            if frame.dtype != np.uint8 or frame.shape != (192, 320, 3):
                raise ValueError(f"{name}: expected uint8 [192,320,3]; official bootstrap does not resize")
            result[name] = frame.copy()
        return result

    def reset(self, image):
        self._state = None
        self.last_video = self.last_latents = None
        with self._inference_context():
            self._reset(image)

    def _reset(self, image):
        initial_state = None
        if isinstance(image, Mapping) and "views" in image:
            if set(image) - {"views", "initial_state", "instruction"}:
                raise ValueError("Unknown PlayWorld reset bundle field")
            if "instruction" in image:
                self.prompt = image["instruction"]
            if "initial_state" in image:
                initial_state = self._actions(np.asarray(image["initial_state"])[None], 1)[0]
            image = image["views"]
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Official text-conditioned PlayWorld requires an instruction")
        views = self._views(image)
        latent = self.model._bootstrap_current_latent({"views": views})
        self._state = {"current_latent": latent}
        if initial_state is not None:
            self._state["_initial_robot_state"] = initial_state
        self.last_video = self.last_latents = None

    def step(self, action):
        try:
            with self._inference_context():
                return self._step(action)
        except BaseException:
            # Official history is mutable; a partially failed call is unsafe
            # to retry as if it had not advanced.
            self._state = None
            self.last_video = self.last_latents = None
            raise

    def _step(self, action):
        if self._state is None:
            raise RuntimeError("Call reset before PlayWorld step")
        actions = self._actions(action, 5 * self.action_downsample)
        output = self.model.rollout(self._state, None, actions, instruction=self.prompt)
        frames = np.asarray(output["frames"])
        if frames.shape != (5, 576, 320, 3) or frames.dtype != np.uint8:
            raise RuntimeError(f"Unexpected official PlayWorld output: {frames.shape} {frames.dtype}")
        self._state = output["next_state"]
        self.last_latents = output["latents"]
        self.last_video = frames.reshape(5, 3, 192, 320, 3)
        return frames[-1]

    def close(self):
        self._state = self.last_video = self.last_latents = None
        if hasattr(self, "model"):
            del self.model
        if self.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
