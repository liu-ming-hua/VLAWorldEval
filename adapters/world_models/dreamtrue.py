"""DreamTrue's versioned native three-view geometry interface.

The persistent worker can transport this mapping. Shared native_replay v1
currently admits only numeric action chunks and cannot represent this protocol.
No action conversion, geometry renderer, or private replay loop is provided.
"""
from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from adapters.base import WorldModelAdapter
from vlaworldeval.paths import project_root

SOURCE_REVISION = "991e2963c015be9c5b7f21695bb82bfaaceffed8"
CHECKPOINT_SHA256 = "e7c34b3028404d5b24865d7fde05d18e97d82d4e30c4a5de1656481c7a33e919"
BASE_REVISION = "539c162b1387eac9dc4c20bd3f74671309e76a4c"
PROTOCOL = "dreamtrue.rendered_geometry.threeviews.v1"
VIEW_NAMES = ("head", "hand_left", "hand_right")


def complete_offload_map(mapping, *, group_norm, wrapped_module):
    """Compatibility extension: upstream added a Plücker GroupNorm after its map."""
    return {**mapping, group_norm: wrapped_module}


def wrap_camera_encoder(encoder, *, device, torch, enable_vram_management, wrapped_module):
    """Keep the small trained camera branch on one device, including GN.

    Native encode_mask_cam chooses its input device from the first parameter;
    generic CPU offloading of just Conv3d would move that input back to CPU.
    Explicit native wrappers use the compute device for every lifecycle state.
    """
    config = {f"{stage}_{field}": value for stage in ("offload", "onload", "preparing", "computation")
              for field, value in (("device", device), ("dtype", torch.bfloat16))}
    mapping = complete_offload_map({torch.nn.Conv3d: wrapped_module},
                                  group_norm=torch.nn.GroupNorm, wrapped_module=wrapped_module)
    return enable_vram_management(encoder.to(device=device, dtype=torch.bfloat16), mapping, vram_config=config)


def classify_checkpoint_shapes(shapes):
    """Header-only release audit; unknown or unpaired tensors fail closed."""
    pairs, full = {}, []
    counts = {"dit": 0, "vace": 0}
    for key, shape in shapes.items():
        match = re.fullmatch(r"pipe\.(dit|vace)\.(.+)", key)
        if match is None:
            raise ValueError(f"Unmatched DreamTrue checkpoint key: {key}")
        branch, name = match.groups()
        counts[branch] += 1
        lora = re.fullmatch(r"(.+)\.lora_([AB])\.default\.weight", name)
        if lora:
            target, side = lora.groups()
            pairs.setdefault((branch, target), {})[side] = tuple(shape)
        elif "lora_" in name:
            raise ValueError(f"Unrecognized LoRA tensor: {key}")
        else:
            full.append((branch, name, key))
    for target, pair in pairs.items():
        if set(pair) != {"A", "B"} or len(pair["A"]) != 2 or len(pair["B"]) != 2:
            raise ValueError(f"Unpaired/unsupported LoRA factors: {target}")
        if pair["A"][0] != pair["B"][1]:
            raise ValueError(f"LoRA rank mismatch: {target}")
    return pairs, full, counts


def audit_dual_lora(pipe, checkpoint, native_load):
    """Observe every native fusion load, without replacing its arithmetic.

    No full base clones or duplicate LoRA matrix products are constructed.
    Native per-target load_state_dict calls are observed and their actual final
    weights must exactly equal the native fused tensors passed to that call.
    """
    import torch
    from safetensors import safe_open
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        shapes = {key: weights.get_slice(key).get_shape() for key in weights.keys()}
        pairs, full, counts = classify_checkpoint_shapes(shapes)
        modules = {branch: dict(getattr(pipe, branch).named_modules()) for branch in ("dit", "vace")}
        hooks, applied = [], {}
        targets = []
        for (branch, name), pair in pairs.items():
            module = modules[branch].get(name)
            if module is None or not hasattr(module, "weight"):
                raise ValueError(f"LoRA target module missing: {branch}.{name}")
            expected_shape = (pair["B"][0], pair["A"][1])
            if tuple(module.weight.shape) != expected_shape:
                raise ValueError(f"LoRA target shape mismatch: {branch}.{name}")
            targets.append((f"{branch}.{name}", module))
        state = {branch: getattr(pipe, branch).state_dict() for branch in ("dit", "vace")}
        for branch, name, key in full:
            if name not in state[branch] or tuple(state[branch][name].shape) != tuple(shapes[key]):
                raise ValueError(f"Non-LoRA target missing/shape mismatch: {key}")

        def observe(name, module, original):
            def load(values, *args, **kwargs):
                result = original(values, *args, **kwargs)
                expected = values["weight"].to(dtype=module.weight.dtype, device=module.weight.device)
                if not torch.equal(module.weight, expected):
                    raise RuntimeError(f"Native fused weight was not applied exactly: {name}")
                applied[name] = applied.get(name, 0) + 1
                return result
            return load

        try:
            for name, module in targets:
                previous = module.__dict__.get("load_state_dict")
                original = module.load_state_dict
                hooks.append((module, previous))
                module.load_state_dict = observe(name, module, original)
            native_load()
        finally:
            for module, previous in hooks:
                if previous is None:
                    delattr(module, "load_state_dict")
                else:
                    module.load_state_dict = previous
        if set(applied) != {name for name, _ in targets} or any(n != 1 for n in applied.values()):
            raise RuntimeError("Native LoRA loader dropped or duplicated one or more target modules")
        after = {branch: getattr(pipe, branch).state_dict() for branch in ("dit", "vace")}
        full_checked = []
        for branch, name, key in full:
            actual = after[branch][name]
            expected = weights.get_tensor(key).to(device=actual.device, dtype=actual.dtype)
            if not torch.equal(actual, expected):
                raise RuntimeError(f"Non-LoRA release tensor was not applied exactly: {key}")
            full_checked.append(key)
    coverage = {}
    for branch in ("dit", "vace", "text_encoder", "vae"):
        model = getattr(pipe, branch)
        params = list(model.named_parameters())
        if not params or any(parameter.is_meta for _, parameter in params):
            raise RuntimeError(f"Official base has missing/unmaterialized parameters: {branch}")
        coverage[branch] = {"parameter_tensors": len(params), "parameters": sum(p.numel() for _, p in params),
                            "dtypes": sorted({str(p.dtype) for _, p in params}), "meta_tensors": 0}
    return {"checkpoint_keys": len(shapes), "prefix_counts": counts, "unmatched_keys": 0,
        "lora_pairs": len(pairs), "target_shapes_verified": len(targets),
        "native_fusion_calls_verified": len(applied), "native_fused_weight_equality": True,
        "targets_applied_once": sorted(applied), "non_lora_verified_equal": full_checked,
        "all_keys_accounted_for": len(shapes) == 2 * len(pairs) + len(full_checked),
        "materialized_parameter_coverage": coverage,
        "base_loader": "pinned official load_model uses load_state_dict(assign=True), strict=True by default"}


def validate_condition(value, *, num_frames, height, width):
    fields = {"schema_version", "protocol", "views", "source", "n_real_frames", "pad_mode"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("DreamTrue requires a versioned rendered geometry mapping; raw actions are unsupported")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["protocol"] != PROTOCOL:
        raise ValueError("Unsupported DreamTrue condition schema/protocol")
    if (type(value["n_real_frames"]) is not int or not 1 <= value["n_real_frames"] <= num_frames
            or value["pad_mode"] != "front"):
        raise ValueError("Native v1 requires explicit n_real_frames and official front padding")
    source = value["source"]
    if (not isinstance(source, Mapping) or not isinstance(source.get("condition_path"), str)
            or not source["condition_path"] or source.get("kind") != "official_rendered_geometry"):
        raise ValueError("Condition source must identify official_rendered_geometry and condition_path")
    views = value["views"]
    if not isinstance(views, Mapping) or set(views) != set(VIEW_NAMES):
        raise ValueError(f"Expected exact official view slots {VIEW_NAMES}")
    result = {}
    for name in VIEW_NAMES:
        view = views[name]
        if not isinstance(view, Mapping) or set(view) != {"rgb", "depth", "intrinsic", "extrinsic"}:
            raise ValueError(f"{name} requires rgb/depth/intrinsic/extrinsic")
        rgb, depth = np.asarray(view["rgb"]), np.asarray(view["depth"])
        if rgb.dtype != np.uint8 or rgb.shape != (num_frames, height, width, 3):
            raise ValueError(f"{name}.rgb must be uint8 [T,H,W,3]")
        if depth.dtype != np.uint16 or depth.shape != (num_frames, height, width):
            raise ValueError(f"{name}.depth must be official masked uint16 depth [T,H,W]")
        cameras = {}
        for key, shape in (("intrinsic", (num_frames, 3, 3)), ("extrinsic", (num_frames, 4, 4))):
            array = np.asarray(view[key])
            if array.shape != shape or not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise ValueError(f"{name}.{key} must be finite float {shape}")
            cameras[key] = array.copy()
        if (np.any(np.abs(np.linalg.det(cameras["intrinsic"])) < 1e-12)
                or not np.allclose(cameras["extrinsic"][:, 3], [0, 0, 0, 1])):
            raise ValueError(f"{name}: invalid camera matrices")
        result[name] = {"rgb": rgb.copy(), "depth": depth.copy(), **cameras}
    return result


class DreamTrueAdapter(WorldModelAdapter):
    environment_key = "world_models.dreamtrue"
    model_name = "DreamTrue"
    native_protocol = PROTOCOL
    view_names = VIEW_NAMES  # Logical slots; DROID maps exterior left/right/gripper here.
    supports_raw_actions = False
    supports_native_replay_v1 = False
    native_condition_protocol = PROTOCOL

    @staticmethod
    def validate_replay_condition(value):
        return validate_condition(value, num_frames=101, height=240, width=320)

    def __init__(self, *, checkpoint_path="checkpoints/world_models/dreamtrue/wmvideo/checkpoint-wmvideo.safetensors",
                 base_model_path="checkpoints/world_models/dreamtrue/Wan-AI/Wan2.1-VACE-14B",
                 repo_path="third_party/dreamtrue", height=240, width=320, num_frames=101,
                 num_inference_steps=50, seed=0, device="cuda", cpu_offload=False, vram_limit=None, audit_path=None):
        if (height, width, num_frames, num_inference_steps) != (240, 320, 101, 50):
            raise ValueError("Native v1 fixes official launcher geometry/frames/steps to 240/320/101/50")
        self.height, self.width, self.num_frames = height, width, num_frames
        self.num_inference_steps, self.seed, self.device = num_inference_steps, seed, device
        if type(cpu_offload) is not bool:
            raise ValueError("cpu_offload must be boolean; precision stays BF16")
        self.cpu_offload = cpu_offload
        if vram_limit is not None and (isinstance(vram_limit, bool) or not isinstance(vram_limit, (int, float))
                                       or not math.isfinite(vram_limit) or vram_limit <= 0):
            raise ValueError("vram_limit must be a finite positive GiB threshold")
        self.vram_limit = vram_limit
        self.audit_path = self._path(audit_path) if audit_path is not None else None
        self.checkpoint_path = self._path(checkpoint_path)
        self.base_model_path = self._path(base_model_path)
        self.repo_path = self._path(repo_path)
        self.model = None
        self.checkpoint_audit = None
        self.current_views = None
        self.prompt = None
        self.last_video = None
        self.last_policy_state = None
        self._failed = False

    @staticmethod
    def _path(value):
        path = Path(value)
        path = (project_root() / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(project_root().resolve()):
            raise ValueError("DreamTrue assets/source must remain inside the project")
        return path

    def reset(self, image):
        if self._failed:
            raise RuntimeError("DreamTrue generation failed; recreate the worker")
        if (not isinstance(image, Mapping) or set(image) != {"schema_version", "protocol", "views", "prompt"}
                or type(image["schema_version"]) is not int or image["schema_version"] != 1
                or image["protocol"] != PROTOCOL):
            raise ValueError("reset requires schema_version/protocol/views/prompt, not one RGB image")
        if not isinstance(image["prompt"], str) or not image["prompt"].strip():
            raise ValueError("DreamTrue requires the actual source prompt")
        if not isinstance(image["views"], Mapping) or set(image["views"]) != set(VIEW_NAMES):
            raise ValueError("reset needs all three official view slots")
        images = {}
        for name in VIEW_NAMES:
            array = np.asarray(image["views"][name])
            if array.dtype != np.uint8 or array.shape != (self.height, self.width, 3):
                raise ValueError(f"{name}: expected uint8 initial RGB [{self.height},{self.width},3]")
            images[name] = array.copy()
        self.current_views, self.prompt = images, image["prompt"]
        self.last_video = None

    def _load_model(self):
        if self.model is not None:
            return
        revision = subprocess.check_output(["git", "-C", str(self.repo_path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(self.repo_path), "status", "--porcelain"], text=True).strip()
        if revision != SOURCE_REVISION or dirty:
            raise RuntimeError("DreamTrue runtime must be the clean pinned official source revision")
        from vlaworldeval.assets import file_record
        if not self.checkpoint_path.is_file() or file_record(self.checkpoint_path)["sha256"] != CHECKPOINT_SHA256:
            raise ValueError("DreamTrue dual-LoRA checkpoint does not match official release SHA256")
        from vlaworldeval.assets import verify_snapshot
        report = verify_snapshot(self.base_model_path)
        if report["repository"] != "Wan-AI/Wan2.1-VACE-14B" or report["revision"] != BASE_REVISION:
            raise ValueError("Exact Wan-AI/Wan2.1-VACE-14B base required")
        shards = sorted(self.base_model_path.glob("diffusion_pytorch_model-*-of-00007.safetensors"))
        required = [f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors" for i in range(1, 8)]
        required += ["models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl/tokenizer_config.json"]
        if len(shards) != 7 or any(not (self.base_model_path / name).is_file() for name in required):
            raise FileNotFoundError("DreamTrue requires all seven VACE-14B shards, T5, VAE and tokenizer")
        import torch
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Official DreamTrue inference requires CUDA with BF16")
        # The released diffusion tensors alone exceed a 20 GiB GPU allocation.
        # Read the verified full tensor shapes. Offload is an explicit profile;
        # precision never changes to satisfy admission.
        from safetensors import safe_open
        sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
        resident = 0
        for file in self.base_model_path.glob("diffusion_pytorch_model*.safetensors"):
            with safe_open(file, framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    tensor = weights.get_slice(key)
                    if tensor.get_dtype() not in sizes:
                        raise ValueError("Unsupported official base dtype")
                    resident += int(np.prod(tensor.get_shape())) * 2  # official BF16 load
        free, _ = torch.cuda.mem_get_info(torch.device(self.device))
        minimum = resident + 10 * 2**30  # T5, VAE, activations and workspaces
        if not self.cpu_offload and free < minimum:
            raise RuntimeError(f"DreamTrue resource constraint: free={free / 2**30:.2f} GiB; "
                               f"BF16 diffusion plus reserve requires >= {minimum / 2**30:.2f} GiB")
        for key in list(os.environ):
            if key.lower().endswith("_proxy"):
                os.environ.pop(key)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_VACE_DIT="false")
        sys.path.insert(0, str(self.repo_path / "wmvideo"))
        module = importlib.import_module("Script.Demo.inference_demo_v2v_vace")
        base = self.base_model_path

        class AuditedInference(module.WanV2VVaceInferenceModule):
            def _load_dual_lora(native_self, checkpoint_path, alpha):
                native_self.checkpoint_audit = audit_dual_lora(native_self.pipe, checkpoint_path,
                    lambda: super(AuditedInference, native_self)._load_dual_lora(checkpoint_path, alpha))

        self.model = AuditedInference(
            model_id_with_origin_paths=(f"Wan-AI/Wan2.1-VACE-14B:{base}/diffusion_pytorch_model*.safetensors,"
                f"Wan-AI/Wan2.1-I2V-14B-480P:{base}/models_t5_umt5-xxl-enc-bf16.pth,"
                f"Wan-AI/Wan2.1-I2V-14B-480P:{base}/Wan2.1_VAE.pth"),
            tokenizer_path=str(base / "google/umt5-xxl"), dual_lora_checkpoint=str(self.checkpoint_path),
            use_plucker=True, device="cpu" if self.cpu_offload else self.device, torch_dtype=torch.bfloat16,
            vram_limit=self.vram_limit)
        self.checkpoint_audit = self.model.checkpoint_audit
        if self.audit_path is not None:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("x") as stream:
                json.dump({"ok": True, "checkpoint_sha256": CHECKPOINT_SHA256, **self.checkpoint_audit}, stream, indent=2)
        print("DreamTrue checkpoint audit:", {key: value for key, value in self.checkpoint_audit.items()
              if key not in {"targets_applied_once", "non_lora_verified_equal"}}, flush=True)
        if self.cpu_offload:
            # Load and fuse the exact official dual-LoRA on CPU first, retaining
            # its BF16 arithmetic. Then use the release's own layer wrappers;
            # never substitute disk-offload's default float8 preparing dtype.
            from diffsynth.core.vram import AutoWrappedModule, enable_vram_management
            from diffsynth.configs import VRAM_MANAGEMENT_MODULE_MAPS
            from diffsynth.models.model_loader import ModelPool
            pool = ModelPool()
            config = {"offload_dtype": torch.bfloat16, "offload_device": "cpu",
                      "onload_dtype": torch.bfloat16, "onload_device": "cpu",
                      "preparing_dtype": torch.bfloat16, "preparing_device": "cpu",
                      "computation_dtype": torch.bfloat16, "computation_device": self.device}
            for name in ("text_encoder", "dit", "vace", "vae"):
                model = getattr(self.model.pipe, name)
                key = type(model).__module__ + "." + type(model).__name__
                if key not in VRAM_MANAGEMENT_MODULE_MAPS:
                    raise RuntimeError(f"Official CPU offload module map missing: {key}")
                mapping = {pool.import_model_class(src): pool.import_model_class(dst)
                           for src, dst in VRAM_MANAGEMENT_MODULE_MAPS[key].items()}
                mapping = complete_offload_map(mapping, group_norm=torch.nn.GroupNorm, wrapped_module=AutoWrappedModule)
                # The release's new camera encoder uses its first parameter's
                # device to move input, and its GroupNorm is absent from the
                # generic VACE module map. Keep this small trained encoder on
                # the compute device as a whole, avoiding CPU-input/GPU-conv
                # mismatch while the large pretrained modules are offloaded.
                camera_encoder = model.mask_cam_embedding if name == "vace" else None
                if camera_encoder is not None:
                    model.mask_cam_embedding = None
                setattr(self.model.pipe, name, enable_vram_management(model, mapping, vram_config=config,
                                                                    vram_limit=self.vram_limit))
                if camera_encoder is not None:
                    model.mask_cam_embedding = wrap_camera_encoder(camera_encoder, device=self.device, torch=torch,
                        enable_vram_management=enable_vram_management, wrapped_module=AutoWrappedModule)
            self.model.pipe.device = torch.device(self.device)
            self.model.pipe.vram_management_enabled = True

    def close(self):
        """Release large wrapped modules before Python interpreter finalization."""
        import gc

        loaded = self.model is not None
        self.model = None
        self.current_views = None
        self.last_video = None
        self._failed = True
        if loaded:
            gc.collect()
            import torch
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

    def step(self, action):
        self.last_video = None
        if self._failed:
            raise RuntimeError("DreamTrue generation failed; recreate the worker")
        if self.current_views is None:
            raise RuntimeError("Call reset with three real initial views first")
        condition = validate_condition(action, num_frames=self.num_frames, height=self.height, width=self.width)
        try:
            self._load_model()
        except Exception:
            if self.model is not None:
                self._failed = True  # Never reuse partially wrapped/loaded weights.
            raise
        from PIL import Image
        from examples.wanvideo.model_training.vace_contract import build_mask_frames
        frames = [condition[name] for name in VIEW_NAMES]
        try:
            generated = self.model.generate(
                prompt=self.prompt,
                vace_reference_image=[[Image.fromarray(self.current_views[name])] for name in VIEW_NAMES],
                vace_rgb_video=[[Image.fromarray(frame) for frame in v["rgb"]] for v in frames],
                vace_depth_video=[list(v["depth"]) for v in frames],
                vace_mask_video=[build_mask_frames(self.num_frames, self.width, self.height,
                    n_real_frames=action["n_real_frames"], pad_mode=action["pad_mode"]) for _ in VIEW_NAMES],
                vace_intrinsic=[v["intrinsic"] for v in frames],
                vace_extrinsic=[v["extrinsic"] for v in frames],
                height=self.height, width=self.width, num_frames=self.num_frames,
                seed=self.seed, cfg_scale=1.0, num_inference_steps=self.num_inference_steps,
                vace_scale=1.0, tiled=False)
            if not isinstance(generated, (list, tuple)) or len(generated) != 3:
                raise RuntimeError("DreamTrue did not return three view videos")
            video = np.stack([np.stack([np.asarray(f) for f in view]) for view in generated], axis=1)
            if video.dtype != np.uint8 or video.shape != (self.num_frames, 3, self.height, self.width, 3):
                raise RuntimeError(f"Invalid DreamTrue video shape/dtype: {video.shape}, {video.dtype}")
        except Exception:
            self._failed = True
            raise
        self.last_video = video[-action["n_real_frames"]:]  # official front-padding trim; inclusive reference
        self.current_views = {name: video[-1, index].copy() for index, name in enumerate(VIEW_NAMES)}
        return video[-1].copy()

    def step_output(self, action, *, output_views=(), state_schema=None):
        if output_views and tuple(output_views) != VIEW_NAMES:
            raise ValueError("DreamTrue output view slots must retain official order")
        if state_schema is not None:
            raise ValueError("DreamTrue has no predicted robot-state head")
        result = super().step_output(action, output_views=VIEW_NAMES)
        result.metadata.update(protocol=PROTOCOL, temporal_layout="inclusive", padding_trimmed=True,
                               source_revision=SOURCE_REVISION, model_inference_performed=True,
                               cpu_offload=self.cpu_offload, vram_limit=self.vram_limit,
                               condition_source=dict(action["source"]), output_views=list(VIEW_NAMES),
                               base_revision=BASE_REVISION, dual_lora_sha256=CHECKPOINT_SHA256,
                               checkpoint_audit=self.checkpoint_audit)
        return result
