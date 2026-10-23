"""Experimental DreamZero DROID policy; native GPU acceptance is still required."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from adapters.base import VLAAdapter
from adapters.vla._protocol import project_path, require_keys, require_mapping
from vlaworldeval.paths import project_root

CAMERAS = {
    "observation/exterior_image_1_left": "video.exterior_image_1_left",
    "observation/exterior_image_2_left": "video.exterior_image_2_left",
    "observation/wrist_image_left": "video.wrist_image_left",
}


def dreamzero_observation(image, instruction, state, *, first_call):
    """Match official Batch fields; never repeat images or invent absent state."""
    images = require_mapping(image, name="DreamZero images")
    states = require_mapping(state, name="DreamZero state")
    require_keys(images, tuple(CAMERAS), name="DreamZero images")
    require_keys(states, ("observation/joint_position", "observation/gripper_position"), name="DreamZero state")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("DreamZero requires a nonempty instruction")
    count = 1 if first_call else 4
    observation = {"annotation.language.action_text": instruction}
    for source, target in CAMERAS.items():
        value = np.asarray(images[source])
        if value.ndim == 3:
            value = value[None]
        if (value.ndim != 4 or value.shape[0] != count or value.shape[-1] != 3
                or min(value.shape[1:3]) < 1 or value.dtype != np.uint8):
            raise ValueError(f"{source} requires uint8 [{count},H,W,3]; no synthetic history padding")
        observation[target] = value.copy()
    for source, target, width in (("observation/joint_position", "state.joint_position", 7),
                                 ("observation/gripper_position", "state.gripper_position", 1)):
        value = np.asarray(states[source], dtype=np.float64)
        if value.shape == (width,):
            value = value[None]
        if value.shape != (1, width) or not np.isfinite(value).all():
            raise ValueError(f"{source} requires finite current state [1,{width}]")
        if width == 1 and np.any((value < 0) | (value > 1)):
            raise ValueError("Measured DROID gripper state must be in [0,1]")
        observation[target] = value.copy()
    return observation


def decode_dreamzero_action(action):
    arrays = []
    for name, width in (("action.joint_position", 7), ("action.gripper_position", 1)):
        value = action.get(name) if isinstance(action, dict) else getattr(action, name, None)
        if value is None:
            raise ValueError(f"Missing decoded DreamZero output {name}; zero-action fallback is disabled")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if width == 1 and value.shape == (24,):
            value = value[:, None]
        if value.shape != (24, width) or not np.isfinite(value).all():
            raise ValueError(f"Decoded DreamZero {name} must be finite [24,{width}]")
        arrays.append(value)
    return np.concatenate(arrays, axis=-1)


class DreamZeroDroidAdapter(VLAAdapter):
    environment_key = "vla.dreamzero"
    expected_action_shape = (24, 8)

    def __init__(self, model_path="checkpoints/vla/dreamzero/droid", *,
                 tokenizer_path="checkpoints/vla/dreamzero/umt5-tokenizer",
                 repo_path="third_party/dreamzero", runtime_dir=None,
                 attention_backend="TE", seed=0):
        if attention_backend not in {"TE", "torch"}:
            raise ValueError("Select TE explicitly, or upstream torch attention as an experimental alternative")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0,2**32)")
        size = int(os.environ.get("WORLD_SIZE", "1"))
        if size not in (1, 2):
            raise ValueError("DreamZero supports IP1/2, not weight sharding or arbitrary GPU counts")
        self.model_path, self.tokenizer_path, self.repo_path = map(project_path, (model_path, tokenizer_path, repo_path))
        if json.loads((self.model_path / "config.json").read_text()).get("action_horizon") != 24:
            raise ValueError("This adapter admits only the released 24-action DROID checkpoint")
        for name in ("spiece.model", "tokenizer_config.json"):
            if not (self.tokenizer_path / name).is_file():
                raise FileNotFoundError(self.tokenizer_path / name)
        if not self.repo_path.is_dir():
            raise FileNotFoundError(self.repo_path)
        if size > 1 and runtime_dir is None:
            raise ValueError("Distributed DreamZero requires a shared runtime_dir")
        self.runtime_dir = project_path(runtime_dir) if runtime_dir else None
        if self.runtime_dir is not None and not self.runtime_dir.resolve().is_relative_to(project_root()):
            raise ValueError("runtime_dir must remain inside the project")
        self.attention_backend, self.seed = attention_backend, seed
        self._policy = self._batch_type = None
        self._owns_single_group = False
        self._closed = False
        try:
            self._load_policy(size)
        except BaseException:
            if self._owns_single_group:
                self.close()
            raise
        self.reset()

    def _load_policy(self, size):
        # Import the isolated runtime only here, never in metadata/contract checks.
        if self.attention_backend == "TE" and any(importlib.util.find_spec(name) is None
                for name in ("transformer_engine", "transformer_engine_torch")):
            raise ImportError("Official TE attention is unavailable; install its compatible runtime or explicitly select attention_backend='torch'. No silent fallback.")
        os.environ["ATTENTION_BACKEND"] = self.attention_backend
        os.environ["ENABLE_DIT_CACHE"] = "false"
        os.environ["ENABLE_TENSORRT"] = "false"
        os.environ.pop("LOAD_TRT_ENGINE", None)
        import torch
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh
        import groot
        if not Path(groot.__file__).resolve().is_relative_to(self.repo_path.resolve()):
            raise ImportError("The environment resolves a different groot package; use the isolated DreamZero environment")
        if not torch.cuda.is_available():
            raise RuntimeError("DreamZero native policy requires CUDA")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        from evaluation.validation.safetensors import inspect_sharded_tensors
        payload = inspect_sharded_tensors(self.model_path)["tensor_bytes"]
        free, _ = torch.cuda.mem_get_info()
        if free < payload + 8 * 2**30:
            raise RuntimeError("Insufficient free GPU memory for full DreamZero weights per rank plus estimated headroom; run scripts/checks/dreamzero.py --gpu first")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if not dist.is_initialized():
            if size > 1:
                dist.init_process_group("nccl")
            else:
                if self.runtime_dir is None:
                    cache = project_root() / ".cache/runtime"
                    cache.mkdir(parents=True, exist_ok=True)
                    self.runtime_dir = Path(tempfile.mkdtemp(prefix="dreamzero-", dir=cache))
                self.runtime_dir.mkdir(parents=True, exist_ok=True)
                dist.init_process_group("nccl", init_method=(self.runtime_dir / "rendezvous").as_uri(), rank=0, world_size=1)
                self._owns_single_group = True
        if dist.get_world_size() != size:
            raise ValueError("DreamZero process-group size differs from the requested IP size")
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from groot.vla.data.schema import EmbodimentTag
        from tianshou.data import Batch
        mesh = init_device_mesh("cuda", (size,), mesh_dim_names=("ip",))
        self._batch_type = Batch
        self._inference_context = torch.no_grad
        self._policy = GrootSimPolicy(embodiment_tag=EmbodimentTag("oxe_droid"),
            model_path=str(self.model_path), device="cuda", device_mesh=mesh,
            tokenizer_path_override=str(self.tokenizer_path))
        if getattr(self._policy.trained_model.action_head, "ip_size", None) != size:
            raise RuntimeError("Upstream DreamZero skipped inference parallel initialization")

    def reset(self, initial_state=None):
        if self._closed:
            raise RuntimeError("DreamZero adapter is closed")
        if getattr(self, "_failed", False):
            raise RuntimeError("DreamZero inference failed; recreate the worker rather than reusing partial cache state")
        # The official causal head resets its cache on a one-frame input.
        self._first_call, self._failed, self._instruction = True, False, None

    def predict(self, image, instruction, state=None):
        if self._closed or self._failed:
            raise RuntimeError("DreamZero is closed or inference failed; inspect/recreate the worker")
        if self._instruction is not None and instruction != self._instruction:
            raise ValueError("Reset the DreamZero episode before changing instruction")
        obs = dreamzero_observation(image, instruction, state, first_call=self._first_call)
        try:
            with self._inference_context():
                result, _video_latents = self._policy.lazy_joint_forward_causal(self._batch_type(obs=obs))
            actions = decode_dreamzero_action(result.act)
        except Exception:
            self._failed = True
            raise
        self._first_call, self._instruction = False, instruction
        return actions  # Unclipped decoded absolute targets, not state measurements.

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._policy = None
        import torch.distributed as dist
        if self._owns_single_group and dist.is_initialized():
            dist.destroy_process_group()
        # For torchrun, the common worker owns collective shutdown.
