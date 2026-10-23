"""Headless PushT adapter using the released IWS latent dynamics and decoder.

Actions are absolute world-frame [left_x,left_y,right_x,right_y] in metres.
Only reset sees a recorded image; subsequent steps retain predicted latents.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from adapters.base import WorldModelAdapter

SOURCE_REVISION = "3ba69b41d070d7a56e79c287da5b24a7d90742aa"
MODEL_REVISION = "6705a00b8366d2399e41154c60bf6bb5c1e42a60"


def verify_assets(checkpoint: Path, source: Path):
    from vlaworldeval.assets import verify_snapshot

    report = verify_snapshot(checkpoint.parent.parent.parent)
    if report["repository"] != "yixuan1999/interactive-world-sim-checkpoints" or report["revision"] != MODEL_REVISION:
        raise ValueError("IWS checkpoint must match the pinned official repository/revision")
    import json
    manifest = json.loads((checkpoint.parent.parent.parent / ".vlaworldeval-download.json").read_text())
    required = {"pusht_cam1/checkpoints/best.ckpt", "pusht_cam1/.hydra/config.yaml"}
    if not required <= set(manifest["files"]) or checkpoint.relative_to(checkpoint.parent.parent.parent).as_posix() != "pusht_cam1/checkpoints/best.ckpt":
        raise ValueError("Pinned IWS checkpoint and config must both be covered by the manifest")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError("IWS source revision differs from the pinned commit")
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=True)
    return report


def prepare_image(image):
    """Official aspect-ratio centre crop followed by area resize to 128²."""
    import cv2

    value = np.asarray(image)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("IWS image must be nonempty uint8 HWC RGB")
    h, w = value.shape[:2]
    side = min(h, w)
    value = value[(h-side)//2:(h-side)//2+side, (w-side)//2:(w-side)//2+side]
    return cv2.resize(value, (128, 128), interpolation=cv2.INTER_AREA)


def validate_action(action):
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    value = np.asarray(action)
    if value.shape != (1, 4) or value.dtype.kind not in "fi" or not np.isfinite(value).all():
        raise ValueError("IWS requires finite real [1,4] absolute world XY actions in metres")
    return value.astype(np.float32, copy=True)


class IWSAdapter(WorldModelAdapter):
    environment_key = "world_models.iws"
    model_name = "Interactive World Simulator"
    action_dim = 4
    action_horizon = 1
    view_names = ("camera_1_color",)

    def __init__(self, *, checkpoint_path="checkpoints/world_models/iws/pusht_cam1/checkpoints/best.ckpt",
                 repo_path="third_party/iws", device="cuda:0"):
        import torch
        from omegaconf import OmegaConf

        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index not in {None, 0}:
            raise ValueError("Official IWS runtime requires cuda:0; CPU supports import/packet checks only")
        checkpoint = Path(checkpoint_path).resolve()
        source = Path(repo_path).resolve()
        if not checkpoint.is_file() or not source.is_dir():
            raise FileNotFoundError("Run scripts/setup/iws.sh to install pinned IWS assets")
        self.asset_verification = verify_assets(checkpoint, source)
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(0)
            if free < 4 * 2**30:
                raise RuntimeError("IWS requires at least 4 GiB free on GPU 0 before model loading")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from interactive_world_sim.algorithms.latent_dynamics.latent_world_model import LatentWorldModel
        from interactive_world_sim.algorithms.common.diffusion_helper import render_img_cm

        # Resolvers are used by the trusted, revision-pinned upstream Hydra config.
        OmegaConf.register_new_resolver("eval", lambda expr: eval(expr, {"__builtins__": {}, "len": len, "int": int}), replace=True)
        OmegaConf.register_new_resolver("torch", lambda name: {"float": torch.float32, "float32": torch.float32}[name], replace=True)
        text = (checkpoint.parent.parent / ".hydra/config.yaml").read_text()
        text = text.replace("_target_: algorithms.", "_target_: interactive_world_sim.algorithms.")
        text = text.replace("_target_: utils.", "_target_: interactive_world_sim.utils.")
        config = OmegaConf.create(text)
        config.algorithm.load_ae = None
        config.n_frames = config.algorithm.n_frames = 10
        config.algorithm.diffusion.sampling_timesteps = 10
        if "diffusion" in config.algorithm.dynamics and "sampling_timesteps" in config.algorithm.dynamics.diffusion:
            config.algorithm.dynamics.diffusion.sampling_timesteps = 10
        # Metrics are training-only and otherwise download I3D from Dropbox.
        config.algorithm.metrics = []
        if list(config.algorithm.obs_keys) != list(self.view_names) or config.algorithm.action_dim != 4:
            raise ValueError("IWS adapter supports the official pusht_cam1 checkpoint only")
        self.model = LatentWorldModel(config.algorithm)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        # Keep every inference weight/normalizer, omit only the unused I3D metric.
        state = {key: value for key, value in state.items() if not key.startswith("validation_fvd_model.")}
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        self.model.dynamics = self.model.dynamics.to(self.model.dtype).eval()
        from torch.nn.attention import SDPBackend
        from interactive_world_sim.algorithms.models.attention import Attention
        # Upstream forces Flash on A100, although released weights use FP32.
        # Math SDPA supports FP32 and leaves model weights/attention semantics intact.
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.cuda_backends = [SDPBackend.MATH]
        self._render = render_img_cm
        self._latent = None
        self._actions = []
        self.last_video = None
        self.last_policy_state = None
        self._failed = False
        self._closed = False

    def reset(self, image):
        """Seed one frame and its corresponding raw action, without padding."""
        import torch

        if self._closed or self._failed:
            raise RuntimeError("IWS is closed or failed; recreate the adapter")
        if not isinstance(image, dict) or set(image) != {"image", "initial_action"}:
            raise ValueError("IWS reset requires image and initial_action (raw world XY)")
        first_action = validate_action(np.asarray(image["initial_action"])[None])
        rgb = prepare_image(image["image"])
        value = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].to(self.device).float() / 255
        with torch.inference_mode():
            value = self.model.normalizer[self.view_names[0]].normalize(value)
            latent = self.model.encoder_forward(value)[None]
            initial = self.model.normalizer["action"].normalize(torch.from_numpy(first_action).to(self.device))
        self._latent = latent
        self._actions = [initial.clone()]
        self.last_video = None

    def step(self, action):
        import torch

        if self._closed or self._failed or self._latent is None:
            raise RuntimeError("Reset IWS before step; recreate after a failed call")
        raw = validate_action(action)
        try:
            with torch.inference_mode():
                normalized = self.model.normalizer["action"].normalize(torch.from_numpy(raw).to(self.device))
                # Official keyboard/dataset replay bounds normalized commands.
                normalized = normalized.clamp(-1, 1)
                history = (self._actions + [normalized])[-11:]
                actions = torch.cat(history, dim=0)[None].to(dtype=self.model.dtype)
                future = self.model.dynamics_forward(self._latent, actions)
                latent = torch.cat([self._latent, future], dim=1)[:, -10:]
                frame = self._render(self.model, latent[:, -1], 128,
                                     normalizer=self.model.normalizer, num_views=1)
                if not torch.isfinite(frame).all() or future.shape[1] != 1:
                    raise RuntimeError("IWS produced nonfinite pixels or an invalid prediction horizon")
                video = (frame.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            self._latent, self._actions = latent, history[-10:]
            self.last_video = video
            return video[-1].copy()
        except BaseException:
            self._failed = True
            self.last_video = None
            raise

    def close(self):
        self._latent = None
        self._actions = []
        self.last_video = None
        self.model = None
        self._closed = True
