"""Community DROID absolute-joint FD: 32 targets + one clean current state row."""
from pathlib import Path
import json
import tempfile
import os

from adapters.base import WorldModelAdapter
from adapters.backends.cosmos import CosmosFrameworkRuntime
from adapters.backends.cosmos_actions import build_joint_sample, generate_joint_sample, decode_joint_video, validate_joint_action, restore_robolab_geometry
from adapters.vla._protocol import project_path
from adapters.world_models.cosmos3_nano_droid_fd import DEFAULT_FRAMEWORK_ROOT, DEFAULT_VAE_PATH, DEFAULT_TEXT_TOKENIZER


class Cosmos3DroidJointFDAdapter(WorldModelAdapter, CosmosFrameworkRuntime):
    environment_key = "world_models.cosmos3_droid_joint_fd"
    action_shape = (32, 8)
    view_names = ("droid.cosmos_composite",)
    fps = 15

    def __init__(self, model_path="checkpoints/world_models/cosmos3_nano_droid_joint_fd/community_anshks_v1_best5600/best-iter5600",
                 *, runtime_dir=None, num_inference_steps=30, guidance=1.0, shift=5.0, seed=0):
        self.model_path = project_path(model_path)
        if runtime_dir is None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Distributed Cosmos requires an explicit shared runtime_dir")
        config = json.loads((self.model_path / "config.json").read_text())["model"]["config"]
        if config["tokenizer"]["encode_exact_durations"] != [33] or config["max_action_dim"] != 64:
            raise ValueError("Joint FD requires the released 33-frame/64-padded-dim export")
        self.vae_path, self.text_tokenizer_path = DEFAULT_VAE_PATH, DEFAULT_TEXT_TOKENIZER
        self.framework_root = DEFAULT_FRAMEWORK_ROOT
        cache = project_path(".cache/runtime")
        cache.mkdir(parents=True, exist_ok=True)
        self.runtime_dir = project_path(runtime_dir) if runtime_dir else Path(tempfile.mkdtemp(prefix="cosmos-joint-fd-", dir=cache))
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.num_inference_steps, self.guidance, self.shift = num_inference_steps, guidance, shift
        self.seed, self.resolution = seed, "480"
        self.use_torch_compile = self.diffusion_cache = False
        self.last_state = self.last_policy_state = None
        self.last_video = self.last_video_with_condition = None
        self._observation = None
        self._index = 0
        self._load_pipeline()

    def reset(self, observation):
        required = ("observation/image", "observation/joint_position", "observation/gripper_position", "prompt")
        if not isinstance(observation, dict) or any(key not in observation for key in required):
            raise ValueError("Joint FD reset requires image, measured initial joints/gripper and prompt")
        self._observation = dict(observation)
        self._index = 0
        self.last_state = self.last_policy_state = None
        self.last_video = self.last_video_with_condition = None

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("Call reset before joint FD step")
        actions = validate_joint_action(action, 32)
        sample = build_joint_sample(self._observation, mode="forward_dynamics", targets=actions)
        generated = generate_joint_sample(self, sample, self.seed + self._index)
        video = restore_robolab_geometry(decode_joint_video(self, generated))
        if len(video) != 33:
            raise ValueError(f"Joint FD must decode 33 frames, got {len(video)}")
        self.last_video_with_condition = video
        self.last_video = video[1:]
        self._observation.update({"observation/image": video[-1],
            "observation/joint_position": actions[-1, :7],
            "observation/gripper_position": actions[-1, 7:]})
        # This proprioception update is an ideal-following command proxy,
        # NOT predicted state. The WorldModelAdapter state output remains None.
        self._index += 1
        return self.last_video

    def reanchor_image(self, image):
        import numpy as np
        if self._observation is None:
            raise RuntimeError("Reset before visual reanchor")
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.shape != self._observation["observation/image"].shape:
            raise ValueError("Visual reanchor requires the original uint8 composite geometry")
        self._observation["observation/image"] = image.copy()
        return {"step_index": self._index, "state_changed": False,
                "diffusion_seed": self.seed + self._index}
