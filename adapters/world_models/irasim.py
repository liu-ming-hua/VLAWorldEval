"""Official public IRASim Language-Table Frame-Ada inference adapter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from adapters.base import WorldModelAdapter

OFFICIAL_SHARD_SHA256 = (
    "f465bec7ef777809b54c327c17ecd606e445c68a3374144afcc5d7c3bf36aad1",
    "5903c87b8c518e504366dbe0ff2ce2c9a83e376a9c77f2b65d3bfd7ea3b1a3c7",
    "7582a2f74aed8931efa99ce5b7611be198648d524e0804f7e7dfc06576b1b036",
    "4a726283c4abed6b716c48916b5aa447bd22858f7fb24c9f9b2324bb9f83b47e",
    "ecf2be477a7ce4adb4d3134f3d35a7d145371571ea7709dd58cd9578b11ae8c6",
    "dbb0896b496ce5905b0e5cc67016d38d870749d87e5aa714f152451f28e167d8",
)


def verify_checkpoint_provenance(checkpoint: Path):
    """Check the extracted member hash and the verified official shard chain."""
    from vlaworldeval.assets import file_record
    info = json.loads((checkpoint.parent / "checkpoint.json").read_text())
    if (info.get("repository") != "fangqi/IRASim"
            or info.get("member") != "opensource_robotdata/languagetable/checkpoints/frame_ada/0300000.pt"
            or info.get("revision") != "dfcbf85c27b2c5d041dbf5df3df272111e758b48"
            or info.get("file", {}).get("size") != 10869764914
            or not info.get("shards")
            or len(info["shards"]) > len(OFFICIAL_SHARD_SHA256)
            or any(part.get("revision") != info["revision"] for part in info["shards"])
            or info.get("file") != file_record(checkpoint)):
        raise ValueError("Checkpoint lacks verified official extraction provenance")
    # These records were published only after shared download_snapshot verified
    # each complete shard against HF's official LFS SHA256.
    for index, part in enumerate(info["shards"]):
        if part["name"] != f"languagetable_checkpoints_data.tar.gz.sfpart_a{chr(97 + index)}":
            raise ValueError("Unexpected official checkpoint shard order")
        manifest_path = checkpoint.parents[3] / ".cache/irasim/checkpoint_shards" / f"part_{index:02d}" / ".vlaworldeval-download.json"
        manifest = json.loads(manifest_path.read_text())
        record = manifest.get("files", {}).get(part["name"], {})
        if (manifest.get("repository") != "fangqi/IRASim" or manifest.get("repo_type") != "dataset"
                or manifest.get("revision") != info["revision"]
                or record.get("size") != 2147483648
                or record.get("sha256") != OFFICIAL_SHARD_SHA256[index]
                or record.get("remote_checksum") != {"algorithm": "sha256", "value": record.get("sha256")}):
            raise ValueError("Checkpoint shard lacks official checksum provenance")
    return info


def load_official_module(repo: Path, relative: str, name: str):
    """Load upstream leaf modules without polluting generic models/dataset names."""
    spec = importlib.util.spec_from_file_location(name, repo / relative)
    module = importlib.util.module_from_spec(spec)
    # dataclass introspection requires the module to be registered while loading.
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class IRASimAdapter(WorldModelAdapter):
    """15 metric XY actions -> 15 new RGB frames, retaining terminal latent.

    Action scaling (x20), resolution (288x512), FP32, and the 16-frame
    conditioned window follow upstream Language-Table Frame-Ada evaluation.
    This checkpoint has no text input or robot-state prediction.
    """

    environment_key = "world_models.irasim"
    view_names = ("primary",)

    def __init__(self, *, checkpoint_path="checkpoints/world_models/irasim/languagetable_frame_ada.pt",
                 vae_model_path="checkpoints/world_models/irasim/sdxl_vae",
                 repo_path="third_party/irasim", device="cuda", num_inference_steps=50):
        import torch
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        self.device = torch.device(device)
        self.repo_path = Path(repo_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.vae_model_path = Path(vae_model_path).resolve()
        self.num_inference_steps = num_inference_steps
        for path in (self.repo_path / "models/irasim.py", self.checkpoint_path,
                     self.vae_model_path / "vae/config.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
        self._latent = None
        self.last_video = None
        self.last_latents = None
        self._load_runtime()

    def _load_runtime(self):
        verify_checkpoint_provenance(self.checkpoint_path)
        import torch
        from diffusers import AutoencoderKL, PNDMScheduler
        official = load_official_module(self.repo_path, "models/irasim.py", "_irasim_public_model")
        pipeline = load_official_module(self.repo_path, "sample/pipeline_trajectory2videogen.py", "_irasim_public_pipeline")
        args = SimpleNamespace(dataset="languagetable", final_frame_ada=False)
        model = official.IRASim_models["IRASim-XL/2"](
            input_size=[36, 64], num_frames=16, learn_sigma=False, extras=3,
            attention_mode="math", args=args)
        # Official trusted archive; mmap avoids materializing unused optimizer tensors.
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
        state = checkpoint["ema"] if "ema" in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.vae = AutoencoderKL.from_pretrained(str(self.vae_model_path), subfolder="vae",
                                               local_files_only=True).to(self.device).eval().requires_grad_(False)
        scheduler = PNDMScheduler.from_pretrained(
            str(self.repo_path / "pretrained_models/scheduler"), local_files_only=True,
            beta_start=0.0001, beta_end=0.02, beta_schedule="linear", variance_type="learned_range")
        self.pipeline = pipeline.Trajectory2VideoGenPipeline(vae=self.vae, scheduler=scheduler, transformer=self.model)

    @staticmethod
    def validate_actions(action):
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        array = np.asarray(action)
        if (array.shape != (15, 2) or array.dtype.kind not in "fiu" or not np.isfinite(array).all()):
            raise ValueError("IRASim Language-Table requires finite [15,2] raw metric XY actions")
        with np.errstate(over="ignore"):
            scaled = array.astype(np.float32) * 20.0
        if not np.isfinite(scaled).all():
            raise ValueError("IRASim actions overflow float32 after scaling")
        return scaled

    def reset(self, image):
        import torch
        with torch.inference_mode():
            self._reset(image)

    def _reset(self, image):
        import torch
        from torchvision.transforms.functional import resize
        # The official bundled sample contains pre-encoded initial latents.
        # A reset bundle preserves that exact context for native replay.
        latent = None
        if isinstance(image, dict):
            if set(image) != {"image", "latent"}:
                raise ValueError("IRASim reset bundle requires image and latent")
            image, latent = image["image"], image["latent"]
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("reset requires uint8 RGB [H,W,3]")
        if latent is None:
            frame = torch.from_numpy(array.copy()).permute(2, 0, 1).float().div(255)
            frame = resize(frame, [288, 512], antialias=True).mul(2).sub(1)[None].to(self.device)
            self._latent = self.vae.encode(frame).latent_dist.sample().mul(self.vae.config.scaling_factor)[:, None]
        else:
            latent = torch.as_tensor(np.asarray(latent), dtype=torch.float32, device=self.device)
            if latent.shape != (4, 36, 64) or not torch.isfinite(latent).all():
                raise ValueError("Bundled initial latent must be finite [4,36,64]")
            self._latent = latent[None, None].clone()
        self.last_video = None
        self.last_latents = None

    def step(self, action):
        import torch
        with torch.inference_mode():
            return self._step(action)

    def _step(self, action):
        import torch
        if self._latent is None:
            raise RuntimeError("Call reset before step")
        actions = torch.from_numpy(self.validate_actions(action))[None].to(self.device)
        videos, latents = self.pipeline(
            actions, mask_x=self._latent, video_length=16, height=288, width=512,
            num_inference_steps=self.num_inference_steps, guidance_scale=1.0,
            device=self.device, return_dict=False, output_type="both")
        if tuple(videos.shape) != (1, 16, 3, 288, 512) or tuple(latents.shape) != (1, 16, 4, 36, 64):
            raise RuntimeError("Unexpected official IRASim output geometry")
        if not torch.isfinite(videos).all() or not torch.isfinite(latents).all():
            raise RuntimeError("Official IRASim output contains nonfinite values")
        # Keep the latent autoregressive context, rather than re-encoding RGB.
        self._latent = latents[:, -1:].detach().clone()
        self.last_latents = latents[0, 1:].detach()
        self.last_video = ((videos[0, 1:] / 2 + 0.5).clamp(0, 1).mul(255)
                           .to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
        return self.last_video[-1].copy()
