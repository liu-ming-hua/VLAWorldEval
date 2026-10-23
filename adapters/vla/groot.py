"""Official NVIDIA Isaac-GR00T N1.5 and N1.7 policy adapters."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    import torch

from adapters.base import VLAAdapter
from adapters.vla._protocol import (
    concatenate_actions,
    float_state,
    project_path,
    require_keys,
    require_mapping,
    uint8_video,
)


class GrootN15Adapter(VLAAdapter):
    """GR00T N1.5 using its official Isaac-GR00T runtime and DROID protocol."""

    environment_key = "vla.groot_n15"
    input_resolution = (256, 256)

    def __init__(
        self,
        model_path: str = "checkpoints/vla/groot/n1.5-3b",
        *,
        data_config: str = "oxe_droid",
        embodiment_tag: str = "oxe_droid",
        denoising_steps: int | None = None,
        device: str = "cuda",
    ):
        from gr00t.experiment.data_config import load_data_config
        from gr00t.model.policy import Gr00tPolicy

        config = load_data_config(data_config)
        self.policy = Gr00tPolicy(
            model_path=str(project_path(model_path)),
            embodiment_tag=embodiment_tag,
            modality_config=config.modality_config(),
            modality_transform=config.transform(),
            denoising_steps=denoising_steps,
            device=device,
        )
        self.device = device
        self.modality_config = config.modality_config()

    def reset(self, initial_state: Any = None) -> None:
        return None

    def predict(self, image: Any, instruction: str, state: Any = None) -> torch.Tensor:
        import torch
        images = require_mapping(image, name="GR00T N1.5 image")
        states = require_mapping(state, name="GR00T N1.5 state")
        video_keys = self.modality_config["video"].modality_keys
        state_keys = self.modality_config["state"].modality_keys
        require_keys(images, video_keys, name="GR00T N1.5 image")
        require_keys(states, state_keys, name="GR00T N1.5 state")

        observation: dict[str, Any] = {}
        video_history = len(self.modality_config["video"].delta_indices)
        state_history = len(self.modality_config["state"].delta_indices)
        for key in video_keys:
            frames = uint8_video(
                images[key], history=video_history, key=key, batched=False
            )
            if frames.shape[1:3] != self.input_resolution:
                raise ValueError(
                    f"{key} must use the official OXE-DROID resolution "
                    f"{self.input_resolution}, got {frames.shape[1:3]}"
                )
            observation[key] = frames
        for key in state_keys:
            observation[key] = float_state(
                states[key], history=state_history, key=key, batched=False
            )
        language_key = self.modality_config["language"].modality_keys[0]
        observation[language_key] = np.asarray([instruction], dtype=object)

        action = self.policy.get_action(observation)
        keys = self.modality_config["action"].modality_keys
        result = concatenate_actions(action, keys)
        return torch.as_tensor(result, dtype=torch.float32, device=self.device)


class GrootN17Adapter(VLAAdapter):
    """GR00T N1.7 using its official two-frame DROID observation protocol."""

    environment_key = "vla.groot_n17"

    def __init__(
        self,
        model_path: str = "checkpoints/vla/groot/n1.7-3b",
        *,
        embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        device: str = "cuda",
        strict: bool = True,
    ):
        # The official N1.7 package exposes these guards so a cached gated
        # Cosmos backbone is resolved to its local snapshot before
        # Transformers constructs the tokenizer. Without them, Transformers
        # still calls model_info() from its Mistral-regex check in offline mode.
        os.environ.setdefault("GROOT_HF_LOCAL_FIRST", "1")
        os.environ.setdefault("GROOT_PATCH_MISTRAL", "1")
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag,
            model_path=str(project_path(model_path)),
            device=device,
            strict=strict,
        )
        self.device = device
        self.modality_config = self.policy.get_modality_config()

    def reset(self, initial_state: Any = None) -> None:
        self.policy.reset()

    def predict(self, image: Any, instruction: str, state: Any = None) -> torch.Tensor:
        import torch
        images = require_mapping(image, name="GR00T N1.7 image")
        states = require_mapping(state, name="GR00T N1.7 state")
        video_cfg = self.modality_config["video"]
        state_cfg = self.modality_config["state"]
        require_keys(images, video_cfg.modality_keys, name="GR00T N1.7 image")
        require_keys(states, state_cfg.modality_keys, name="GR00T N1.7 state")

        video = {
            key: uint8_video(
                images[key], history=len(video_cfg.delta_indices), key=key, batched=True
            )
            for key in video_cfg.modality_keys
        }
        state_data = {
            key: float_state(
                states[key], history=len(state_cfg.delta_indices), key=key, batched=True
            )
            for key in state_cfg.modality_keys
        }
        language_key = self.policy.language_key
        observation = {
            "video": video,
            "state": state_data,
            "language": {language_key: [[instruction]]},
        }
        action, _ = self.policy.get_action(observation)
        keys = self.modality_config["action"].modality_keys
        result = concatenate_actions(action, keys)
        return torch.as_tensor(result, dtype=torch.float32, device=self.device)


class GrootN17DroidAdapter(GrootN17Adapter):
    """DROID post-trained checkpoint, distinct from the N1.7 base baseline."""

    def __init__(self, model_path="checkpoints/vla/groot/n1.7-droid", **kwargs):
        super().__init__(model_path=model_path, **kwargs)


__all__ = ["GrootN15Adapter", "GrootN17Adapter", "GrootN17DroidAdapter"]
