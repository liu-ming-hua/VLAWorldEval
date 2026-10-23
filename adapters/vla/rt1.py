"""RT-1-X adapter.

The released RT-1-X checkpoint is a TensorFlow SavedModel. It is tempting to
load it with the now-archived legacy JAX implementation, but that
implementation expects a Flax msgpack checkpoint and is therefore not
compatible with the official ``saved_model.pb`` release. This adapter follows
the official TF-Agents SavedModel policy interface instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from vlaworldeval.paths import project_root

import numpy as np

# RT-1-X is loaded on CPU through TensorFlow. Hiding CUDA from TensorFlow is
# important because the same process uses the selected CUDA device for
# WorldGym and for the returned PyTorch action tensor.
os.environ.setdefault(
    "TFHUB_CACHE_DIR",
    str(project_root() / ".cache" / "tfhub"),
)
import tensorflow as tf

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    # TensorFlow may already have initialized its device list when this module
    # is imported from an interactive process. Normal CLI execution reaches
    # this call early enough.
    pass

import tensorflow_hub as hub
import torch
from tf_agents.policies import py_tf_eager_policy
from tf_agents.trajectories import time_step as ts

from adapters.actions.bridge import rescale_bridge_action
from adapters.base import VLAAdapter


class _SavedModelRT1Policy:
    """Thin wrapper around the official RT-1-X TF-Agents SavedModel."""

    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = str(checkpoint_path)
        self.policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            self.checkpoint_path,
            load_specs_from_pbtxt=True,
            use_tf_function=True,
        )
        self.state = self.policy.get_initial_state(batch_size=1)

    def reset(self) -> None:
        self.state = self.policy.get_initial_state(batch_size=1)

    def action(self, image: np.ndarray, embedding: np.ndarray, instruction: str) -> dict:
        # The SavedModel spec is [256, 320, 3] uint8. RT-1-X expects the
        # padded resize used by the official SimplerEnv policy.
        image_tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
        image_tensor = tf.image.resize_with_pad(
            image_tensor,
            target_height=256,
            target_width=320,
            method=tf.image.ResizeMethod.BILINEAR,
        )
        image_tensor = tf.cast(tf.clip_by_value(image_tensor, 0, 255), tf.uint8)
        observation = {
            "image": image_tensor,
            "natural_language_embedding": tf.convert_to_tensor(
                embedding, dtype=tf.float32
            ),
            "natural_language_instruction": tf.convert_to_tensor(instruction),
        }
        time_step = ts.restart(observation)
        policy_step = self.policy.action(time_step, self.state)
        self.state = policy_step.state
        return {name: np.asarray(value) for name, value in policy_step.action.items()}


class RT1Adapter(VLAAdapter):
    """Official RT-1-X TensorFlow policy mapped to WorldGym's 10-D action."""

    environment_key = "vla.rt1"

    def __init__(self, checkpoint_path: str | Path, history_len: int = 15, text_encoder_path: str | None = None):
        del history_len  # History is maintained by the SavedModel policy state.
        checkpoint = Path(checkpoint_path)
        if not (checkpoint / "saved_model.pb").exists():
            raise ValueError(
                "RT-1-X adapter requires the official TensorFlow SavedModel "
                f"(saved_model.pb), but none was found under {checkpoint}."
            )
        self.policy = _SavedModelRT1Policy(checkpoint)
        self._use = None
        self._instruction = None
        self._embedding = None
        self.text_encoder_path = text_encoder_path or "https://tfhub.dev/google/universal-sentence-encoder-large/5"

    def reset(self, initial_state: Any = None) -> None:
        del initial_state
        self.policy.reset()
        self._instruction = None
        self._embedding = None

    def _instruction_embedding(self, instruction: str) -> np.ndarray:
        if self._use is None:
            self._use = hub.load(self.text_encoder_path)
        if instruction != self._instruction:
            self._instruction = instruction
            self._embedding = self._use([instruction]).numpy()[0].astype(np.float32)
        return self._embedding

    @staticmethod
    def _to_worldgym_action(raw: dict) -> torch.Tensor:
        """Convert RT-1-X physical action fields to WorldGym's 10-D range."""
        world_vector = np.asarray(raw["world_vector"], dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw["rotation_delta"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(raw["gripper_closedness_action"], dtype=np.float32).reshape(-1)

        # RT-1-X emits metres and radians with bounds [-2, 2] and
        # [-pi/2, pi/2]. First normalize to the Open-X/Bridge [-1, 1]
        # convention, then apply the same WorldGym bridge conversion as the
        # other VLA adapters.
        action7 = torch.tensor(
            [
                *(world_vector[:3] / 2.0),
                *(rotation_delta[:3] / (np.pi / 2.0)),
                float(gripper[0]),
            ],
            device="cuda",
            dtype=torch.float32,
        )
        padded = torch.cat([action7, action7.new_zeros(3)])
        return rescale_bridge_action(
            padded,
            wv_lo=-1.0,
            wv_hi=1.0,
            rd_lo=-1.0,
            rd_hi=1.0,
        )

    def predict(self, image: np.ndarray, instruction: str, state: Any = None) -> torch.Tensor:
        del state
        raw_action = self.policy.action(
            np.asarray(image, dtype=np.uint8),
            self._instruction_embedding(instruction),
            instruction,
        )
        return self._to_worldgym_action(raw_action).unsqueeze(0)

    def update_state(self, state: Any, action: torch.Tensor, observation: np.ndarray) -> Any:
        del action, observation
        return state


__all__ = ["RT1Adapter"]
