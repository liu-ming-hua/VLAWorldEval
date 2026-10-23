"""DreamDojo 2B GR1, using NVIDIA's Video2WorldInference in a persistent worker.

The native input is twelve already encoded 384-D GR1 actions. This adapter
does not guess a conversion from a seven-dimensional end-effector policy.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import json

import numpy as np

from adapters.base import WorldModelAdapter
from vlaworldeval.paths import project_root

SOURCE_REVISION = "02f119b759d5c7f84a399fdeea3c6e82e7ed6cff"
EXPERIMENT = "dreamdojo_2b_480_640_gr1"
CONFIG = "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py"
MODEL_REVISION = "89d029e10816d2995d700cb8ba06f171e0504203"
MODEL_PREFIX = "2B_GR1_post-train/iter_000050000/model"
COMPATIBILITY_PROFILE = "dreamdojo_2b_gr1_wan_vae_torch271_cu126_v1"
TOKENIZER_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
TOKENIZER_SIZE = 507609880
TOKENIZER_OFFICIAL_REVISION = "85f8ae7bfe8f5525c8d103429524dcf12f98bf7b"
TOKENIZER_TRANSPORT_REVISION = "37ec512624d61f7aa208f7ea8140a131f93afc9a"


def validate_tokenizer_identity(manifest):
    """After verify_snapshot, require exact current official tokenizer bytes."""
    if (manifest.get("repository") != "Wan-AI/Wan2.1-T2V-1.3B"
            or manifest.get("revision") != TOKENIZER_TRANSPORT_REVISION
            or set(manifest.get("files", {})) != {"Wan2.1_VAE.pth"}):
        raise ValueError("Unexpected tokenizer transport identity")
    record = manifest["files"]["Wan2.1_VAE.pth"]
    if record.get("sha256") != TOKENIZER_SHA256 or record.get("size") != TOKENIZER_SIZE:
        raise ValueError("Tokenizer bytes differ from pinned current official Cosmos tokenizer")
    return {"repository": "nvidia/Cosmos-Predict2.5-2B", "revision": TOKENIZER_OFFICIAL_REVISION,
            "filename": "tokenizer.pth", "sha256": TOKENIZER_SHA256, "size": TOKENIZER_SIZE,
            "transport_repository": manifest["repository"], "transport_revision": manifest["revision"],
            "historical_registry_revision_verified": False}


def validate_attention_features(*, dropout_p=0.0, alibi_slopes=None, softcap=0.0):
    """The isolated Flash wheel supports forward inference with these off only."""
    if dropout_p != 0.0 or alibi_slopes is not None or softcap != 0.0:
        raise ValueError("DreamDojo inference-only Flash requires dropout=0, ALiBi=None, softcap=0")


@contextmanager
def restricted_flash_attention():
    # Guard the pinned interface's dispatch globals, including aliases imported
    # by Transformers/TE before this context. Never install this wheel globally.
    import flash_attn.flash_attn_interface as interface
    originals = {}
    def guarded(original, positions):
        def call(*args, **kwargs):
            values = {key: kwargs.get(key, args[pos] if len(args) > pos else default)
                      for key, pos, default in positions}
            validate_attention_features(**values)
            return original(*args, **kwargs)
        return call
    try:
        for name, offset in (("_wrapped_flash_attn_forward", 0),
                             ("_wrapped_flash_attn_varlen_forward", 4)):
            originals[name] = getattr(interface, name)
            positions = (("dropout_p", 3 + offset, 0.0),
                         ("softcap", 8 + offset, 0.0),
                         ("alibi_slopes", 9 + offset, None))
            setattr(interface, name, guarded(originals[name], positions))
        yield
    finally:
        for name, original in originals.items():
            setattr(interface, name, original)


def verify_pinned_checkpoint(assets: Path, checkpoint: Path):
    from vlaworldeval.assets import MANIFEST, verify_snapshot
    report = verify_snapshot(assets)
    if report["repository"] != "nvidia/DreamDojo" or report["revision"] != MODEL_REVISION:
        raise ValueError("DreamDojo requires the pinned official checkpoint revision")
    manifest = json.loads((assets / MANIFEST).read_text())
    expected = {f"{MODEL_PREFIX}/.metadata", "2B_GR1_post-train/latest_checkpoint.txt"}
    expected.update(f"{MODEL_PREFIX}/__{rank}_0.distcp" for rank in range(128))
    if set(manifest["files"]) != expected:
        raise ValueError("Pinned DreamDojo manifest must cover all 128 model rank shards")
    if checkpoint != (assets / MODEL_PREFIX).resolve():
        raise ValueError("Checkpoint must be the verified pinned GR1 DCP model directory")
    return report


def validate_tensor_coverage(expected, available, *, ema=True):
    """Check every learned parameter; training-only tensors may be extra.

    Only the official net. -> net_ema. prefix conversion is permitted. No
    learned parameter may be dropped or initialized from a random fallback.
    """
    if not expected:
        raise ValueError("Cannot validate an empty learned-parameter inventory")
    missing, mismatch = [], []
    for key, shape in expected.items():
        if not key.startswith("net."):
            raise ValueError(f"Unexpected learned parameter outside net: {key}")
        source = "net_ema." + key[4:] if ema else key
        if source not in available:
            missing.append(source)
        elif tuple(available[source]) != tuple(shape):
            mismatch.append((source, tuple(shape), tuple(available[source])))
    if missing or mismatch:
        raise ValueError(f"Incomplete learned checkpoint tensors: missing={missing[:8]}, shape_mismatch={mismatch[:8]}")
    return {"learned_parameters_verified": len(expected), "ema_to_net": ema}


def learned_parameter_inventory(module):
    """Use PyTorch's state-dict hooks, not wrapper-internal parameter names.

    Activation checkpoint wrappers remove their private prefix in state_dict.
    keep_vars preserves Parameter identity so every learned object can be
    checked without accidentally treating a buffer as a learned parameter.
    """
    parameters = {id(value) for value in module.parameters()}
    expected, seen = {}, set()
    for name, value in module.state_dict(keep_vars=True).items():
        if id(value) in parameters:
            expected["net." + name] = tuple(value.shape)
            seen.add(id(value))
    if not parameters or seen != parameters:
        raise ValueError("Official state-dict inventory does not cover every learned Parameter object")
    return expected


def validate_parameter_shapes(expected, available):
    if not expected:
        raise ValueError("Empty learned-parameter inventory")
    missing = [key for key in expected if key not in available]
    wrong = [key for key in expected if key in available and tuple(expected[key]) != tuple(available[key])]
    if missing or wrong:
        raise ValueError(f"Incomplete auxiliary learned tensors: missing={missing[:8]}, shape_mismatch={wrong[:8]}")


def reject_training_only_lam(module, args):
    raise RuntimeError("Training-only LAM has no verified checkpoint and must never run in native GR1 inference")


def preload_cuda_libraries():
    """Expose PyPI cuDNN split libraries to TE in an existing worker process."""
    import ctypes
    import sysconfig
    directory = Path(sysconfig.get_path("purelib")) / "nvidia/cudnn/lib"
    # TE 2.2 requires split cuDNN SONAMEs, while wheels lack a global ldconfig entry.
    for name in ("libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_ops.so.9",
                 "libcudnn_adv.so.9", "libcudnn_cnn.so.9", "libcudnn_heuristic.so.9",
                 "libcudnn_engines_runtime_compiled.so.9", "libcudnn_engines_precompiled.so.9"):
        path = directory / name
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def checkpoint_directory(path: str | Path) -> Path:
    """Resolve official checkpoint roots, iterations, or model/ DCP.

    Zero-length *unreferenced* distributed rank shards are official and valid.
    The DCP reader checks referenced storage ranges when loading the model.
    """
    path = Path(path).expanduser().resolve()
    if path.is_file():
        raise ValueError(f"Only pinned DCP directories are supported; unverified PT is refused: {path}")
    if (path / "2B_GR1_post-train").is_dir():
        path /= "2B_GR1_post-train"
    latest = path / "latest_checkpoint.txt"
    if latest.is_file():
        name = latest.read_text().strip()
        if not name.startswith("iter_") or not name[5:].isdigit():
            raise ValueError("Unsafe or invalid latest_checkpoint.txt")
        path /= name
    if (path / "model").is_dir():
        path /= "model"
    if ((path / ".metadata").is_file() and (path / ".metadata").stat().st_size
            and any(p.stat().st_size for p in path.glob("*.distcp") if p.is_file())):
        return path
    raise FileNotFoundError(f"Incomplete DreamDojo DCP checkpoint: {path}")


@contextmanager
def _source_cwd(repo: Path):
    previous = Path.cwd()
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(previous)


class DreamDojoAdapter(WorldModelAdapter):
    environment_key = "world_models.dreamdojo"
    model_name = "DreamDojo-2B-GR1"
    view_names = ("ego_view",)

    def __init__(self, *, checkpoint_path, repo_path=None, assets_path=None,
                 device="cuda:0", num_inference_steps=35, guidance=0, seed=0, native_multiround=False):
        root = project_root()
        self.repo_path = Path(repo_path or root / "third_party/dreamdojo").resolve()
        self.assets_path = Path(assets_path or root / "checkpoints/world_models/dreamdojo").resolve()
        self.checkpoint_path = checkpoint_directory(checkpoint_path)
        self.checkpoint_verification = verify_pinned_checkpoint(self.assets_path, self.checkpoint_path)
        self.device = str(device)
        if not self.device.startswith("cuda"):
            raise ValueError("Official DreamDojo inference requires CUDA")
        if type(num_inference_steps) is not int or num_inference_steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        if not np.isfinite(guidance) or guidance < 0:
            raise ValueError("guidance must be finite and nonnegative")
        if not (self.repo_path / CONFIG).is_file():
            raise FileNotFoundError(f"DreamDojo source missing: {self.repo_path}")
        revision = subprocess.check_output(
            ["git", "-C", str(self.repo_path), "rev-parse", "HEAD"], text=True).strip()
        if revision != SOURCE_REVISION:
            raise ValueError(f"DreamDojo source revision mismatch: {revision}")
        changes = subprocess.check_output(
            ["git", "-C", str(self.repo_path), "status", "--porcelain"], text=True)
        if changes:
            raise ValueError("DreamDojo requires a clean pinned source checkout")
        self.num_inference_steps, self.guidance, self.seed = num_inference_steps, guidance, seed
        if type(native_multiround) is not bool:
            raise ValueError("native_multiround must be an explicit boolean")
        self.native_multiround = native_multiround
        self.pipe = None
        self.current_image = None
        self.last_video = None
        self.last_video_with_condition = None
        self.last_actions = None
        self.last_state = None
        self.last_policy_state = None
        self._chunk_index = 0

    def _load_pipeline(self):
        import torch
        preload_cuda_libraries()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no DreamDojo inference was performed")
        for key in list(os.environ):
            if key.lower().endswith("_proxy"):
                os.environ.pop(key)
        root = project_root()
        from vlaworldeval.cache import runtime_cache_environment
        os.environ.update(runtime_cache_environment(root))
        os.environ.update(HF_ENDPOINT="https://hf-mirror.com", HF_HUB_OFFLINE="1",
                          HF_HOME=str(root / ".cache/huggingface"),
                          XDG_CACHE_HOME=str(root / ".cache"))
        if os.environ.get("COSMOS_SMOKE", "0").lower() in {"1", "true", "yes", "y"}:
            raise RuntimeError("COSMOS_SMOKE skips weight loading and is forbidden")
        torch.cuda.set_device(self.device)
        sys.path.insert(0, str(self.repo_path))
        from cosmos_predict2._src.imaginaire import flags
        if flags.SMOKE or flags.INTERNAL:
            raise RuntimeError("DreamDojo requires public runtime with SMOKE disabled")
        from cosmos_predict2._src.imaginaire.utils import checkpoint_db
        from vlaworldeval.assets import verify_snapshot
        for component, repository, revision in (
            ("tokenizer", "Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a"),
            ("reason", "nvidia/Cosmos-Reason1-7B", "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"),
        ):
            report = verify_snapshot(self.assets_path / component)
            if report["repository"] != repository or report["revision"] != revision:
                raise ValueError(f"Wrong pinned auxiliary asset: {component}")
        from vlaworldeval.assets import MANIFEST
        self.tokenizer_identity = validate_tokenizer_identity(
            json.loads((self.assets_path / "tokenizer" / MANIFEST).read_text()))
        from cosmos_predict2._src.predict2.utils import model_loader
        reason_config = json.loads((self.assets_path / "reason/config.json").read_text())
        validate_attention_features(dropout_p=reason_config["attention_dropout"])
        from torch.distributed.checkpoint import FileSystemReader
        metadata = FileSystemReader(str(self.checkpoint_path)).read_metadata()
        for storage in metadata.storage_data.values():
            shard = self.checkpoint_path / storage.relative_path
            if (not shard.resolve().is_relative_to(self.checkpoint_path) or not shard.is_file()
                    or shard.stat().st_size < storage.offset + storage.length):
                raise ValueError(f"Missing or truncated referenced DCP storage: {storage.relative_path}")
        original_load = model_loader.load_model_state_dict_from_checkpoint
        original_planner = model_loader.DefaultLoadPlanner
        original_module_load = torch.nn.Module.load_state_dict
        self.auxiliary_verification = {}
        def strict_auxiliary_load(module, state, *args, **kwargs):
            if type(module).__name__ in {"QwenVLBaseModel", "WanVAE_"}:
                expected = {name: tuple(value.shape) for name, value in module.named_parameters()}
                available = {name: tuple(value.shape) for name, value in state.items()
                             if hasattr(value, "shape")}
                validate_parameter_shapes(expected, available)
                self.auxiliary_verification[type(module).__name__] = len(expected)
            return original_module_load(module, state, *args, **kwargs)
        def strict_load(model, config, checkpoint, load_ema_to_reg=False, *args, **kwargs):
            if Path(checkpoint).resolve() != self.checkpoint_path or not load_ema_to_reg:
                raise ValueError("Only the verified GR1 EMA checkpoint may be loaded")
            expected = learned_parameter_inventory(model.net)
            available = {name: tuple(value.size) for name, value in metadata.state_dict_metadata.items()
                         if hasattr(value, "size")}
            self.tensor_verification = validate_tensor_coverage(expected, available, ema=True)
            return original_load(model, config, checkpoint, load_ema_to_reg, *args, **kwargs)
        def strict_planner(*args, **kwargs):
            kwargs["allow_partial_load"] = False
            return original_planner(*args, **kwargs)
        # Official experiment registration eagerly resolves *unused* 2B/14B
        # pretrained defaults. Bind these defaults to the explicitly supplied
        # posttrained checkpoint; load_model_from_checkpoint overrides load_path.
        # Tokenizer and text encoder still require their actual, separate assets.
        original = checkpoint_db.get_checkpoint_path
        def local_checkpoint(uri):
            uri = str(uri).rstrip("/")
            if not uri.startswith(("s3://", "hf://")):
                return original(uri)
            record = checkpoint_db.get_checkpoint_by_s3(uri)
            hf = record.hf
            if hf is None:
                raise FileNotFoundError(f"No public local asset for {record.name}")
            if getattr(hf, "filename", None) == "tokenizer.pth":
                target = self.assets_path / "tokenizer/Wan2.1_VAE.pth"
            elif hf.repository == "nvidia/Cosmos-Reason1-7B":
                if hf.revision != "3210bec0495fdc7a8d3dbb8d58da5711eab4b423":
                    raise ValueError("Official Qwen processor/Reason registry revision changed")
                target = self.assets_path / "reason"
            elif hf.repository.startswith("nvidia/Cosmos-Predict2"):
                target = self.checkpoint_path
            else:
                raise FileNotFoundError(f"Unconfigured local asset: {record.name}")
            if not target.exists():
                raise FileNotFoundError(f"DreamDojo asset missing: {target}")
            return str(target)
        checkpoint_db.get_checkpoint_path = local_checkpoint
        model_loader.load_model_state_dict_from_checkpoint = strict_load
        model_loader.DefaultLoadPlanner = strict_planner
        torch.nn.Module.load_state_dict = strict_auxiliary_load
        try:
            from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
            with _source_cwd(self.repo_path):
                self.pipe = Video2WorldInference(
                    experiment_name=EXPERIMENT, ckpt_path=str(self.checkpoint_path),
                    s3_credential_path="", context_parallel_size=1, config_file=CONFIG,
                    offload_diffusion_model=True, offload_text_encoder=True, offload_tokenizer=True,
                )
                # Upstream eagerly constructs LAM, but generate_vid2world's
                # LAM calls are commented out. Never allow its unverified
                # training-only parameters to enter an inference computation.
                lam = self.pipe.model.lam
                self.training_only_lam_parameters = sum(p.numel() for p in lam.parameters())
                for module in lam.modules():
                    module.register_forward_pre_hook(reject_training_only_lam)
                if (not getattr(self, "tensor_verification", None)
                        or set(self.auxiliary_verification) != {"QwenVLBaseModel", "WanVAE_"}):
                    self.pipe = None
                    raise RuntimeError("Required learned-parameter load verification did not run")
        finally:
            checkpoint_db.get_checkpoint_path = original
            model_loader.load_model_state_dict_from_checkpoint = original_load
            model_loader.DefaultLoadPlanner = original_planner
            torch.nn.Module.load_state_dict = original_module_load

    @staticmethod
    def _image(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.dtype != np.uint8 or value.shape != (480, 640, 3):
            raise ValueError(f"DreamDojo needs uint8 RGB [480,640,3], got {value.shape}/{value.dtype}")
        return value.copy()

    def reset(self, image):
        self.current_image = self._image(image)
        self.last_video = self.last_video_with_condition = self.last_actions = None
        self.last_state = self.last_policy_state = None
        self._chunk_index = 0

    @staticmethod
    def _actions(value):
        if isinstance(value, (list, tuple)):
            # np.asarray([True, 1.0]) already promotes bool to float. Preserve
            # Python scalar types before NumPy's common-dtype inference too.
            scalars = np.asarray(value, dtype=object)
            if any(isinstance(item, (bool, np.bool_, str, bytes, complex, np.complexfloating))
                   for item in scalars.flat):
                raise ValueError("GR1 actions must be real numeric values, not bool/string/complex")
        if hasattr(value, "detach"):
            dtype = str(value.dtype)
            if dtype == "torch.bool" or dtype.startswith("torch.complex"):
                raise ValueError("GR1 actions must be real numeric values, not bool/string/complex")
            value = value.detach().cpu()
            if dtype == "torch.bfloat16":
                value = value.float()
            value = value.numpy()
        value = np.asarray(value)
        if value.dtype.kind not in "iuf":
            raise ValueError("GR1 actions must be real numeric values, not bool/string/complex")
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (12, 384) or not np.isfinite(value).all():
            raise ValueError("DreamDojo needs finite native GR1 actions [12,384]")
        if np.any(value[:, 29:] != 0):
            raise ValueError("GR1 action slots [29:384] must be zero")
        return np.ascontiguousarray(value)

    def step(self, action):
        if self.current_image is None:
            raise RuntimeError("Call reset(image) before step(action)")
        if self._chunk_index and not getattr(self, "native_multiround", False):
            raise RuntimeError("Only one native chunk per reset is validated; multi-chunk state alias is unsupported")
        if self._chunk_index >= 4:
            raise RuntimeError("Declared native49-frame protocol is limited to four12-action rounds")
        actions = self._actions(action)
        if self.pipe is None:
            self._load_pipeline()
        import torch
        with (torch.inference_mode(), torch.cuda.device(self.device),
              _source_cwd(self.repo_path), restricted_flash_attention()):
            video_input = torch.zeros((1, 3, 13, 480, 640), dtype=torch.uint8)
            video_input[0, :, 0] = torch.from_numpy(self.current_image.copy()).permute(2, 0, 1)
            result = self.pipe.generate_vid2world(
                prompt="", input_path=video_input, action=torch.from_numpy(actions),
                guidance=self.guidance, num_video_frames=13, num_latent_conditional_frames=1,
                resolution="480,640", seed=(self.seed + self._chunk_index * (12 if getattr(self, "native_multiround", False) else 1)) % 2**32,
                num_steps=self.num_inference_steps,
            )
            if tuple(result.shape) != (1, 3, 13, 480, 640) or not torch.isfinite(result).all():
                raise RuntimeError(f"Invalid DreamDojo output: {tuple(result.shape)}")
            video = ((result[0].float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            video = video.permute(1, 2, 3, 0).cpu().numpy()
        self.last_video_with_condition = video
        self.last_video = video[1:].copy()  # twelve future frames, exclusive timing
        self.current_image = video[-1].copy()
        self.last_actions = actions.copy()
        self._chunk_index += 1
        return self.current_image.copy()

    def step_output(self, action, *, output_views=(), state_schema=None):
        from dataclasses import replace
        result = super().step_output(action, output_views=output_views, state_schema=state_schema)
        import torch
        metadata = {
            **result.metadata, "source_revision": SOURCE_REVISION, "model_revision": MODEL_REVISION,
            "compatibility_profile": COMPATIBILITY_PROFILE, "exact_paper_reproduction": False,
            "asset_fidelity_status": "current_official_tokenizer_bytes_verified_historical_revision_unavailable",
            "tokenizer_identity": self.tokenizer_identity,
            "learned_tensor_coverage": self.tensor_verification,
            "auxiliary_learned_tensor_coverage": self.auxiliary_verification,
            "training_only_lam": {"forward_forbidden": True,
                                  "excluded_parameter_elements": self.training_only_lam_parameters},
            "max_chunks_per_reset": 4 if getattr(self, "native_multiround", False) else 1,
            "multiround_protocol": "official_global_action_alias_then12-action_windows_seed_at_offsets0_12_24_36" if getattr(self, "native_multiround", False) else None,
            "state_source": "recorded_state_in_native_action",
            "flash_attention_profile": "forward_only_dropout0_alibi_none_softcap0_sm80",
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(self.device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(self.device)),
        }
        # Shared RPC deletes consumed response packets. Persist only this small
        # diagnostic record in its existing console log, never a private loop.
        print("DREAMDOJO_STEP_EVIDENCE=" + json.dumps(metadata, allow_nan=False), flush=True)
        return replace(result, metadata=metadata)
