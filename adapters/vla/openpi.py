"""Official Physical Intelligence OpenPI adapters for pi0 and pi0.5."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, TYPE_CHECKING

import numpy as np
if TYPE_CHECKING:
    import torch

from adapters.base import VLAAdapter
from adapters.vla._protocol import project_path, require_keys, require_mapping


class OpenPIAdapter(VLAAdapter):
    environment_key = "vla.openpi"
    config_name = ""
    default_checkpoint = ""
    required_fields: tuple[str, ...] = ()
    expected_action_shape: tuple[int, int] = (0, 0)

    def __init__(
        self,
        checkpoint_path: str | None = None,
        *,
        config_name: str | None = None,
        output_device: str = "cuda",
        sample_kwargs: Mapping[str, Any] | None = None,
    ):
        from openpi.policies import policy_config
        from openpi.training import config as training_config

        self.output_device = output_device
        self.config_name = config_name or type(self).config_name
        checkpoint = project_path(checkpoint_path or self.default_checkpoint)
        config = training_config.get_config(self.config_name)
        if self.config_name == "pi0_fast_droid":
            tokenizer = project_path("checkpoints/vla/openpi/fast-tokenizer")
            if not (tokenizer / "processor_config.json").is_file():
                raise FileNotFoundError(f"Local official FAST tokenizer missing: {tokenizer}")
            config = replace(config, model=replace(config.model,
                fast_model_tokenizer_kwargs={"fast_tokenizer_path": str(tokenizer)}))
        self.policy = policy_config.create_trained_policy(
            config,
            checkpoint,
            sample_kwargs=dict(sample_kwargs or {}),
        )

    def reset(self, initial_state: Any = None) -> None:
        return None

    def predict(self, image: Any, instruction: str, state: Any = None) -> torch.Tensor:
        import torch
        images = require_mapping(image, name=f"{self.config_name} image")
        robot_state = require_mapping(state, name=f"{self.config_name} state")
        observation = {**robot_state, **images, "prompt": instruction}
        require_keys(observation, self.required_fields, name=f"{self.config_name} observation")
        joint_position = np.asarray(observation["observation/joint_position"])
        gripper_position = np.asarray(observation["observation/gripper_position"])
        if joint_position.shape != (7,):
            raise ValueError(
                "observation/joint_position must have shape [7], "
                f"got {joint_position.shape}"
            )
        if gripper_position.shape not in {(), (1,)}:
            raise ValueError(
                "observation/gripper_position must be a scalar or shape [1], "
                f"got {gripper_position.shape}"
            )
        result = self.policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != self.expected_action_shape:
            raise ValueError(
                f"{self.config_name} actions must have shape "
                f"{self.expected_action_shape}, got {actions.shape}"
            )
        return torch.as_tensor(actions, dtype=torch.float32, device=self.output_device)


class Pi0Adapter(OpenPIAdapter):
    """Official pi0-DROID policy: 2 RGB views + 8-D state -> [10,8]."""

    config_name = "pi0_droid"
    default_checkpoint = "checkpoints/vla/openpi/pi0-droid"
    expected_action_shape = (10, 8)
    required_fields = (
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
    )


class Pi05Adapter(OpenPIAdapter):
    """Official pi0.5-DROID policy: 2 RGB views + 8-D state -> [15,8]."""

    config_name = "pi05_droid"
    default_checkpoint = "checkpoints/vla/openpi/pi05-droid"
    expected_action_shape = (15, 8)
    required_fields = Pi0Adapter.required_fields


class Pi0FastDroidAdapter(OpenPIAdapter):
    """Official pi0-FAST DROID JAX policy, with a local FAST tokenizer."""

    config_name = "pi0_fast_droid"
    default_checkpoint = "checkpoints/vla/openpi/pi0-fast-droid"
    expected_action_shape = (10, 8)
    required_fields = Pi0Adapter.required_fields


__all__ = ["OpenPIAdapter", "Pi0Adapter", "Pi05Adapter", "Pi0FastDroidAdapter"]
