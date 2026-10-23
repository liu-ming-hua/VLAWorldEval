"""Adapter for the local Giga-World-1 inference implementation.

Giga-World-1 is a controllable video generator.  Its released FunCtrl
interface consumes a first image, text, and an optional camera/Plucker control
video; it does not consume robot actions directly.  ``action_to_control_video``
is therefore an explicit integration point for a task-specific action-to-
camera/control representation.

The adapter deliberately supports an action-agnostic smoke mode when
``allow_action_agnostic=True``.  That mode is useful for checking model
loading and video generation, but must not be used as an action-conditioned
VLA evaluation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from adapters.base import WorldModelAdapter
from vlaworldeval.paths import gigaworld1_source


ControlVideoBuilder = Callable[[np.ndarray, np.ndarray, Any], Any]


class GigaWorld1Adapter(WorldModelAdapter):
    """Wrap Giga-World-1's local Diffusers inference pipeline.

    Parameters
    ----------
    repo_path:
        Local Giga-World-1 checkout.  Its ``gigaworld`` package is imported
        from this directory and is not copied into this project.
    base_model_path:
        Giga Diffusers-format checkpoint containing VAE, T5, and image
        encoder components.
    config_path:
        Giga YAML config, normally the matching stage-1 config.
    checkpoint_path:
        Optional Giga scene-LoRA checkpoint directory.
    control_video_path:
        Optional precomputed Plücker/Ray-Map control video.  For real
        action-conditioned evaluation, prefer ``action_to_control_video``.
    action_to_control_video:
        Callback ``(current_image, action, previous_control) -> control``.
        The returned value can be a video path or an array accepted by the
        Giga pipeline.
    allow_action_agnostic:
        Permit a static/no control video.  This is explicitly a smoke mode;
        actions are recorded by the evaluator but do not affect Giga output.
    """

    environment_key = "world_models.gigaworld1"

    def __init__(
        self,
        *,
        repo_path: str | Path | None = None,
        base_model_path: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
        transformer_model_path: str | Path | None = None,
        prompt: str = "",
        control_video_path: str | Path | None = None,
        action_to_control_video: ControlVideoBuilder | None = None,
        allow_action_agnostic: bool = False,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_frames: int = 33,
        height: int = 480,
        width: int = 1920,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        fps: int = 10,
        seed: int = 42,
        output_size: tuple[int, int] = (256, 256),
    ):
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else gigaworld1_source()
        self.base_model_path = Path(base_model_path).expanduser()
        self.config_path = Path(config_path).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self.transformer_model_path = (
            Path(transformer_model_path).expanduser() if transformer_model_path else self.base_model_path
        )
        self.prompt = prompt
        self.control_video_path = Path(control_video_path).expanduser() if control_video_path else None
        self.action_to_control_video = action_to_control_video
        self.allow_action_agnostic = allow_action_agnostic
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.fps = fps
        self.seed = seed
        self.output_size = output_size
        self._current_image: np.ndarray | None = None
        self._control_video: Any = None
        self.last_video: np.ndarray | None = None

        for path, label in (
            (self.repo_path, "Giga-World-1 repo_path"),
            (self.base_model_path, "Giga base_model_path"),
            (self.config_path, "Giga config_path"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Giga checkpoint_path does not exist: {self.checkpoint_path}")
        if self.control_video_path is not None and not self.control_video_path.exists():
            raise FileNotFoundError(f"Giga control_video_path does not exist: {self.control_video_path}")
        if not torch.cuda.is_available() and self.device.type == "cuda":
            raise RuntimeError("GigaWorld1Adapter requires CUDA; torch.cuda.is_available() is False")

        self._load_pipeline()

    def _load_external_inference_module(self):
        repo = str(self.repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        infer_path = self.repo_path / "infer" / "infer_giga_world.py"
        spec = importlib.util.spec_from_file_location("vlaworldeval_giga_infer", infer_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Giga inference module: {infer_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_pipeline(self) -> None:
        giga_infer = self._load_external_inference_module()
        from accelerate.utils import set_seed
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from diffusers.utils import load_video
        from omegaconf import OmegaConf
        from peft import set_peft_model_state_dict
        from transformers import AutoTokenizer, UMT5EncoderModel

        from gigaworld.modules.gigaworld_kernels import (
            replace_all_norms_with_flash_norms,
            replace_rmsnorm_with_fp32,
            replace_rope_with_flash_rope,
        )
        from gigaworld.modules.transformer_functrl_gigaworld import GigaworldTransformer3DModelFunCtrl
        from gigaworld.pipelines.pipeline_gigaworld_functrl import GigaworldFunCtrlPipeline
        from gigaworld.utils.train_config import Args
        from gigaworld.utils.utils_base import NORM_LAYER_PREFIXES, load_model_checkpoint

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        config = OmegaConf.load(str(self.config_path))
        args = OmegaConf.merge(OmegaConf.structured(Args), config)
        args.model_config.pretrained_model_name_or_path = str(self.base_model_path)
        args.model_config.transformer_model_name_or_path = str(self.transformer_model_path)
        args.model_config.load_checkpoints_custom = self.checkpoint_path is not None
        args.model_config.load_model_path = str(self.checkpoint_path) if self.checkpoint_path else None

        set_seed(self.seed)
        negative_prompt = args.data_config.negative_prompt
        weight_dtype = self.dtype

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.base_model_path),
            subfolder="tokenizer",
            revision=args.model_config.revision,
        )
        scheduler_path = self.repo_path / "scripts" / "accelerate_configs" / "scheduler_config.json"
        scheduler = UniPCMultistepScheduler.from_pretrained(str(scheduler_path))
        vae = AutoencoderKLWan.from_pretrained(
            str(self.base_model_path),
            subfolder="vae",
            revision=args.model_config.revision,
            variant=args.model_config.variant,
            torch_dtype=weight_dtype,
            device_map=self.device,
        )
        vae.enable_tiling()
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(self.base_model_path),
            subfolder="text_encoder",
            revision=args.model_config.revision,
            variant=args.model_config.variant,
            dtype=weight_dtype,
            device_map=self.device,
        )

        transformer_kwargs = {
            "has_multi_term_memory_patch": args.training_config.has_multi_term_memory_patch,
            "zero_history_timestep": args.training_config.zero_history_timestep,
            "restrict_self_attn": args.training_config.restrict_self_attn,
            "guidance_cross_attn": args.training_config.guidance_cross_attn,
            "is_train_restrict_lora": args.training_config.is_train_restrict_lora,
            "restrict_lora": args.training_config.restrict_lora,
            "restrict_lora_rank": args.training_config.restrict_lora_rank,
            "is_amplify_history": args.training_config.is_amplify_history,
            "history_scale_mode": args.training_config.history_scale_mode,
            "model_type": args.model_config.model_type,
        }
        transformer = GigaworldTransformer3DModelFunCtrl.from_pretrained(
            str(self.transformer_model_path),
            subfolder="transformer" if (self.transformer_model_path / "transformer").exists() else None,
            transformer_additional_kwargs=transformer_kwargs,
        )
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
        transformer.requires_grad_(False)

        if self.checkpoint_path is not None:
            transformer_lora_config = giga_infer.build_lora_config(args, transformer)
            transformer.add_adapter(transformer_lora_config)
            load_model_checkpoint(
                args=args,
                checkpoint_path=str(self.checkpoint_path),
                transformer=transformer,
                pipeline_class=GigaworldFunCtrlPipeline,
                norm_layer_prefixes=NORM_LAYER_PREFIXES,
                convert_unet_state_dict_to_peft_fn=giga_infer.convert_unet_state_dict_to_peft,
                set_peft_model_state_dict_fn=set_peft_model_state_dict,
                cast_training_params_fn=giga_infer.cast_training_params,
            )
        for name, parameter in transformer.named_parameters():
            if any(pattern in name for pattern in transformer.__class__._keep_in_fp32_modules):
                parameter.data = parameter.data.to(torch.float32)
            else:
                parameter.data = parameter.data.to(weight_dtype)
        transformer.to(self.device).eval()
        vae.eval()
        text_encoder.eval()

        self.pipe = GigaworldFunCtrlPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            pipeline_type=args.model_config.model_type,
        ).to(self.device)
        self.negative_prompt = negative_prompt
        self._load_video = load_video

    def reset(self, image: np.ndarray) -> None:
        self._current_image = np.asarray(image, dtype=np.uint8).copy()
        self.last_video = None
        if self.control_video_path is not None:
            self._control_video = self._load_video(str(self.control_video_path))
        else:
            self._control_video = None

    def _resolve_control_video(self, action: np.ndarray) -> Any:
        if self.action_to_control_video is not None:
            return self.action_to_control_video(self._current_image, action, self._control_video)
        if self._control_video is not None:
            if not self.allow_action_agnostic:
                raise RuntimeError(
                    "A static Giga control video is not action-conditioned. "
                    "Provide action_to_control_video or set allow_action_agnostic=True "
                    "for a smoke test."
                )
            return self._control_video
        raise RuntimeError(
            "Giga-World-1 FunCtrl requires a control video. Provide control_video_path "
            "or action_to_control_video."
        )

    @torch.inference_mode()
    def step(self, action: torch.Tensor) -> np.ndarray:
        if self._current_image is None:
            raise RuntimeError("Call GigaWorld1Adapter.reset(image) before step(action)")
        action_np = action.detach().float().cpu().numpy() if torch.is_tensor(action) else np.asarray(action)
        control_video = self._resolve_control_video(action_np)
        image = Image.fromarray(self._current_image).resize((self.width, self.height))
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        kwargs = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "image": image,
            "guidance_scale": self.guidance_scale,
            "num_frames": self.num_frames,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "generator": generator,
            "output_type": "np",
        }
        if control_video is not None:
            kwargs["control_video"] = control_video
        generated = self.pipe(**kwargs).frames[0]
        self.last_video = np.asarray(generated)
        terminal = np.asarray(generated[-1])
        terminal = np.clip(terminal * 255 if terminal.max() <= 1.0 else terminal, 0, 255).astype(np.uint8)
        terminal = np.asarray(Image.fromarray(terminal).resize(self.output_size))
        self._current_image = terminal
        self.seed += 1
        return terminal


# Backward-compatible alias. Use GigaWorld1Adapter for all new code so that
# future GigaWorld versions can coexist without ambiguous adapter names.
GigaWorldAdapter = GigaWorld1Adapter
