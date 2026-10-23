"""Official Cosmos DROID policy via the published RoboLab action/state protocol."""
from pathlib import Path
import json
import tempfile
import os

import numpy as np

from adapters.base import VLAAdapter
from adapters.backends.cosmos import CosmosFrameworkRuntime
from adapters.backends.cosmos_actions import build_joint_sample, generate_joint_sample, decode_joint_video
from adapters.vla._protocol import project_path, require_keys, require_mapping
from adapters.world_models.cosmos3_nano_droid_fd import (
    DEFAULT_FRAMEWORK_ROOT, DEFAULT_VAE_PATH, DEFAULT_TEXT_TOKENIZER,
)


class CosmosDroidPolicyAdapter(VLAAdapter, CosmosFrameworkRuntime):
    environment_key = "vla.cosmos_policy_droid"
    expected_action_shape = (32, 8)

    def __init__(self, model_path="checkpoints/vla/cosmos3/nano-policy-droid", *,
                 runtime_dir=None, num_inference_steps=4, guidance=3.0, shift=5.0,
                 seed=0, decode_preview=True, format_prompt_as_json=False):
        self.model_path = project_path(model_path)
        if runtime_dir is None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Distributed Cosmos requires an explicit shared runtime_dir")
        config = json.loads((self.model_path / "config.json").read_text())
        if not config.get("model", {}).get("config", {}).get("action_gen"):
            raise ValueError("Cosmos DROID policy requires an action-enabled checkpoint")
        self.vae_path = DEFAULT_VAE_PATH
        self.text_tokenizer_path = DEFAULT_TEXT_TOKENIZER
        self.framework_root = DEFAULT_FRAMEWORK_ROOT
        cache = project_path(".cache/runtime")
        cache.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = project_path(runtime_dir) if runtime_dir else Path(tempfile.mkdtemp(prefix="cosmos-policy-", dir=cache))
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.num_inference_steps, self.guidance, self.shift = num_inference_steps, guidance, shift
        self.seed, self.resolution = seed, "480"
        self.use_torch_compile = self.diffusion_cache = False
        self.decode_preview = decode_preview
        self.format_prompt_as_json = format_prompt_as_json
        self.last_prediction_video = None
        self._index = 0
        self._load_pipeline()

    def reset(self, initial_state=None):
        self._index = 0
        self.last_prediction_video = None

    def predict(self, image, instruction, state=None):
        import torch
        images = require_mapping(image, name="Cosmos DROID images")
        states = require_mapping(state, name="Cosmos DROID joint/gripper state")
        require_keys(images, ("observation/exterior_image_1_left", "observation/exterior_image_2_left",
                              "observation/wrist_image_left"), name="Cosmos DROID images")
        require_keys(states, ("observation/joint_position", "observation/gripper_position"), name="Cosmos DROID state")
        sample = build_joint_sample({**images, **states, "prompt": instruction},
                                    format_prompt_as_json=self.format_prompt_as_json)
        generated = generate_joint_sample(self, sample, self.seed + self._index)
        actions = generated["action"][0][1:, :8].detach().float().cpu().numpy()
        actions[:, -1] = 1.0 - actions[:, -1]  # upstream RoboLab output convention
        if actions.shape != self.expected_action_shape or not np.isfinite(actions).all():
            raise ValueError(f"Unexpected Cosmos DROID policy output: {actions.shape}")
        if self.decode_preview:
            self.last_prediction_video = decode_joint_video(self, generated)
        self._index += 1
        return torch.as_tensor(actions, device="cuda")
