"""Official Hugging Face LeRobot SmolVLA adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    import torch

from adapters.base import VLAAdapter
from adapters.vla._protocol import project_path, require_keys, require_mapping


class SmolVLAAdapter(VLAAdapter):
    environment_key = "vla.smolvla"

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/vla/smolvla/base",
        *,
        backbone_path: str = "checkpoints/vla/smolvla/smolvlm2-500m-video-instruct",
        device: str = "cuda",
    ):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        checkpoint = project_path(checkpoint_path)
        backbone = project_path(backbone_path)
        self.device = device
        config = SmolVLAConfig.from_pretrained(checkpoint, local_files_only=True)
        config.vlm_model_name = str(backbone)
        config.device = device
        self.policy = SmolVLAPolicy.from_pretrained(
            checkpoint, config=config, local_files_only=True
        ).to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {"tokenizer_name": str(backbone)},
            },
        )
        self.image_keys = tuple(
            key
            for key, feature in self.policy.config.input_features.items()
            if feature.type.name == "VISUAL"
        )
        self.state_key = "observation.state"

    def reset(self, initial_state: Any = None) -> None:
        self.policy.reset()

    @staticmethod
    def _image_tensor(value: Any, key: str) -> torch.Tensor:
        import torch
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"{key} must have shape [H,W,3] or [3,H,W], got {array.shape}")
        tensor = torch.as_tensor(array)
        if tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] != 3:
            raise ValueError(f"{key} must be RGB, got {tuple(tensor.shape)}")
        tensor = tensor.float()
        if tensor.numel() and tensor.max() > 1:
            tensor = tensor / 255.0
        return tensor

    def predict(self, image: Any, instruction: str, state: Any = None) -> torch.Tensor:
        import torch
        images = require_mapping(image, name="SmolVLA image")
        present_image_keys = tuple(key for key in self.image_keys if key in images)
        if not present_image_keys:
            raise ValueError(
                "SmolVLA image must contain at least one configured visual key; "
                f"expected one of {self.image_keys}"
            )
        if isinstance(state, Mapping):
            require_keys(state, (self.state_key,), name="SmolVLA state")
            state_value = state[self.state_key]
        elif state is None:
            raise ValueError("SmolVLA requires observation.state; no state was provided")
        else:
            state_value = state

        # The upstream SmolVLA implementation explicitly supports a variable
        # number of present cameras and masks missing configured views.  Keep
        # that behavior here instead of requiring every camera in config.json.
        frame: dict[str, Any] = {
            key: self._image_tensor(images[key], key) for key in present_image_keys
        }
        frame[self.state_key] = torch.as_tensor(state_value, dtype=torch.float32)
        frame["task"] = instruction
        batch = self.preprocess(frame)
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocess(actions)
        if isinstance(actions, Mapping):
            actions = actions["action"]
        actions = torch.as_tensor(actions, dtype=torch.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(f"SmolVLA actions must have shape [H,A], got {tuple(actions.shape)}")
        return actions.to(self.device)


__all__ = ["SmolVLAAdapter"]
