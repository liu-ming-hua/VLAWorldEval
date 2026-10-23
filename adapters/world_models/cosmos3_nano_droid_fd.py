"""Community Cosmos3-Nano DROID FD export via the official framework runtime.

This adapter is deliberately separate from :mod:`cosmos3_nano`, which wraps
the generic Diffusers base checkpoint.  The released DROID post-train uses the
Cosmos Framework export/runtime, a fixed 17-frame/16-action window, and raw
metric DROID actions with no normalization.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adapters.base import WorldModelAdapter
from adapters.backends.cosmos import CosmosFrameworkRuntime
from adapters.backends.cosmos import context_parallel_degree as context_parallel_degree


from vlaworldeval.paths import project_root

PROJECT_ROOT = project_root()
DEFAULT_FRAMEWORK_ROOT = PROJECT_ROOT / "third_party" / "cosmos-framework"
DEFAULT_VAE_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "world_models"
    / "cosmos3_nano_droid_fd"
    / "vae"
    / "Wan2.2_VAE.pth"
)
DEFAULT_TEXT_TOKENIZER = PROJECT_ROOT / "checkpoints/world_models/cosmos3_nano_droid_fd/text_tokenizer"



class Cosmos3NanoDroidFDAdapter(WorldModelAdapter, CosmosFrameworkRuntime):
    """Run the released 20k-EMA DROID post-train through Cosmos Framework.

    Instantiate this class inside the registered Cosmos DROID FD environment.
    The framework pipeline is loaded once and reused across calls to ``step``.
    A call consumes exactly ``[16, 10]`` raw DROID actions and produces the 16
    future frames of the three-view composite canvas.
    """

    environment_key = "world_models.cosmos3_nano_droid_fd"
    model_name = "Cosmos3-Nano-DROID-FD-20k-EMA"
    view_names = ("droid.cosmos_composite",)
    action_shape = (16, 10)
    fps = 15

    def __init__(
        self,
        *,
        model_path: str | Path,
        vae_path: str | Path = DEFAULT_VAE_PATH,
        text_tokenizer_path: str | Path = DEFAULT_TEXT_TOKENIZER,
        framework_root: str | Path = DEFAULT_FRAMEWORK_ROOT,
        num_inference_steps: int = 30,
        guidance: float = 1.0,
        shift: float = 10.0,
        seed: int = 0,
        resolution: str = "480",
        use_torch_compile: bool = False,
        diffusion_cache: bool = True,
        inference_batch_size: int = 1,
        runtime_dir: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.vae_path = Path(vae_path).expanduser().resolve()
        self.text_tokenizer_path = Path(text_tokenizer_path).expanduser().resolve()
        self.framework_root = Path(framework_root).expanduser().resolve()
        self.num_inference_steps = int(num_inference_steps)
        self.guidance = float(guidance)
        self.shift = float(shift)
        self.seed = int(seed)
        self.resolution = str(resolution)
        self.use_torch_compile = bool(use_torch_compile)
        self.diffusion_cache = bool(diffusion_cache)
        if type(inference_batch_size) is not int or inference_batch_size < 1:
            raise ValueError("inference_batch_size must be a positive integer")
        self.inference_batch_size = inference_batch_size
        self._temporary_dir: tempfile.TemporaryDirectory[str] | None = None
        if int(os.environ.get("WORLD_SIZE", "1")) > 1 and runtime_dir is None:
            raise ValueError("Distributed inference requires a shared runtime_dir")
        if runtime_dir is None:
            self._temporary_dir = tempfile.TemporaryDirectory(prefix="vlaworldeval-cosmos3-droid-")
            self.runtime_dir = Path(self._temporary_dir.name)
        else:
            self.runtime_dir = Path(runtime_dir).expanduser().resolve()
            self.runtime_dir.mkdir(parents=True, exist_ok=True)

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"DROID FD checkpoint not found: {self.model_path}")
        if not (self.model_path / "config.json").is_file():
            raise FileNotFoundError(f"DROID FD config.json not found in {self.model_path}")
        if not self.vae_path.is_file():
            raise FileNotFoundError(
                f"Wan2.2 VAE not found: {self.vae_path}. Pass vae_path explicitly."
            )
        if not (self.framework_root / "cosmos_framework").is_dir():
            raise FileNotFoundError(f"Cosmos Framework source not found: {self.framework_root}")
        for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
            if not (self.text_tokenizer_path / name).is_file():
                raise FileNotFoundError(f"Local Cosmos text tokenizer missing: {self.text_tokenizer_path / name}")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be positive")
        if self.resolution != "480":
            raise ValueError("The released DROID FD recipe is aligned to resolution tier '480'")
        self.checkpoint_contract = self.validate_checkpoint_contract(self.model_path)

        self.current_image: Image.Image | None = None
        self.last_video: np.ndarray | None = None
        self.last_video_with_condition: np.ndarray | None = None
        self.last_actions: np.ndarray | None = None
        self.last_state = None
        self.last_policy_state = None
        self._step_index = 0
        self._pipe: Any | None = None
        self._setup_args: Any | None = None
        self._sample_overrides_cls: Any | None = None
        self._decode_video: Any | None = None
        self._load_pipeline()

    @classmethod
    def validate_checkpoint_contract(cls, model_path: str | Path) -> dict[str, Any]:
        """Validate static fields that distinguish the DROID post-train export."""
        root = Path(model_path).expanduser().resolve()
        config_path = root / "config.json"
        index_path = root / "model.safetensors.index.json"
        if not config_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"Incomplete Cosmos3 DROID FD export under {root}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        model = payload.get("model", {}).get("config", {})
        tokenizer = model.get("tokenizer", {})
        expected = {
            "action_gen": True,
            "max_action_dim": 64,
            "encode_exact_durations": [17],
        }
        actual = {
            "action_gen": model.get("action_gen"),
            "max_action_dim": model.get("max_action_dim"),
            "encode_exact_durations": tokenizer.get("encode_exact_durations"),
        }
        if actual != expected:
            raise ValueError(f"Unexpected Cosmos3 DROID FD checkpoint contract: {actual}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index.get("weight_map", {}).values()))
        missing = [name for name in shards if not (root / name).is_file()]
        if not shards or missing:
            raise FileNotFoundError(f"Missing Cosmos3 DROID FD weight shards: {missing}")
        return {**actual, "num_shards": len(shards), "action_shape": list(cls.action_shape)}

    @staticmethod
    def _to_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        arr = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"DROID FD reset expects one RGB composite [H,W,3], got {arr.shape}")
        if np.issubdtype(arr.dtype, np.floating) and float(np.max(arr)) <= 1.0 + 1e-3:
            arr = arr * 255.0
        return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8), mode="RGB")

    def reset(self, image: Any) -> None:
        self.current_image = self._to_image(image)
        self._conditioning_size = self.current_image.size
        self.last_video = None
        self.last_video_with_condition = None
        self.last_actions = None
        self.last_state = None
        self.last_policy_state = None
        self._step_index = 0

    @classmethod
    def validate_action(cls, action: Any) -> np.ndarray:
        arr = action.detach().float().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != cls.action_shape:
            raise ValueError(
                f"Cosmos3 DROID FD requires exactly raw action {cls.action_shape}, got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("Cosmos3 DROID FD action contains NaN or infinity")
        return np.ascontiguousarray(arr)

    def reanchor_image(self, image):
        if self.current_image is None:
            raise RuntimeError("Reset before visual reanchor")
        replacement = self._to_image(image)
        if replacement.size != self._conditioning_size:
            raise ValueError("Visual reanchor must preserve conditioning geometry")
        self.current_image = replacement
        return {"step_index": self._step_index, "state_changed": False,
                "diffusion_seed": self.seed + self._step_index}

    def step_output(self, action, *, output_views=(), state_schema=None):
        # Canonical observation geometry only; recurrent current_image remains
        # the unmodified native decoded terminal, as in upstream inference.
        from adapters.backends.cosmos_actions import restore_robolab_geometry
        from evaluation.contracts import WorldModelOutput
        output = super().step_output(action, output_views=output_views, state_schema=state_schema)
        width, height = self._conditioning_size
        return WorldModelOutput(
            views={name: restore_robolab_geometry(video, height, width) for name, video in output.views.items()},
            robot_state=output.robot_state,
            metadata={**output.metadata, "missing_border_pixels": "edge-filled, not predicted"})

    def step(self, action: Any) -> np.ndarray:
        if self.current_image is None or self._pipe is None or self._setup_args is None:
            raise RuntimeError("Call reset(composite_image) before step(action)")
        actions = self.validate_action(action)
        step_name = f"step_{self._step_index:05d}"
        inputs_dir = self.runtime_dir / "inputs" / step_name
        inputs_dir.mkdir(parents=True, exist_ok=True)
        image_path = inputs_dir / "condition.png"
        action_path = inputs_dir / "actions.json"
        import torch.distributed as dist
        distributed = dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        if rank == 0:
            self.current_image.save(image_path)
            action_path.write_text(json.dumps(actions.tolist()), encoding="utf-8")
        if distributed:
            dist.barrier()

        sample = self._sample_overrides_cls(
            name=step_name,
            model_mode="forward_dynamics",
            vision_path=str(image_path),
            action_path=str(action_path),
            domain_name="droid_lerobot",
            action_chunk_size=16,
            image_size=480,
            aspect_ratio="4,3",
            resolution=self.resolution,
            fps=self.fps,
            num_frames=17,
            num_steps=self.num_inference_steps,
            guidance=self.guidance,
            shift=self.shift,
            prompt="",
            seed=self.seed + self._step_index,
        )
        sample.output_dir = self.runtime_dir / "outputs" / step_name
        sample.download(sample.output_dir / "inputs")
        sample_args = sample.build_sample(model_config=self._pipe.model_config)
        outputs = self._pipe.generate([sample_args])
        result = [None, None]
        if rank == 0:
            if not outputs or outputs[0].status != "success" or not outputs[0].outputs:
                result[1] = outputs[0].message if outputs else "no output returned"
            else:
                result[0] = next(
                    (str(path) for path in outputs[0].outputs[0].files if Path(path).suffix == ".mp4"),
                    None,
                )
                if result[0] is None:
                    result[1] = "no video artifact returned"
        if distributed:
            dist.broadcast_object_list(result, src=0)
        if result[1]:
            raise RuntimeError(f"Cosmos3 DROID FD inference failed: {result[1]}")
        video_path = Path(result[0])
        try:
            frames, _ = self._decode_video(video_path)
            full_video = frames.cpu().numpy().astype(np.uint8, copy=False)
        except ModuleNotFoundError as exc:
            if exc.name != "torchcodec":
                raise
            # Decoding a saved MP4 is not model preprocessing. Keep PyAV as
            # a CPU fallback rather than requiring a different Torch ABI.
            import av
            with av.open(str(video_path)) as container:
                full_video = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
        future = full_video[1:] if len(full_video) == 17 else full_video
        if future.shape[0] != 16:
            raise RuntimeError(f"Expected 16 generated frames, got {future.shape}")

        self.last_video_with_condition = full_video
        self.last_video = future
        self.last_actions = actions.copy()
        self.current_image = Image.fromarray(future[-1], mode="RGB")
        self._step_index += 1
        return future[-1]

    def rollout(self, action: Any) -> np.ndarray:
        self.step(action)
        assert self.last_video is not None
        return self.last_video


__all__ = ["Cosmos3NanoDroidFDAdapter"]
