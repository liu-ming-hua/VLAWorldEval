from typing import Any
from pathlib import Path

import jax
import numpy as np
import torch
from octo.model.octo_model import OctoModel

from adapters.actions.bridge import rescale_bridge_action
from adapters.base import VLAAdapter
from vlaworldeval.paths import project_root


def _normalize(actions, statistics):
    low = np.asarray(statistics["mean"] - 10 * statistics["std"])
    high = np.asarray(statistics["mean"] + 10 * statistics["std"])
    mask = np.asarray(statistics.get("mask", np.ones_like(low)), dtype=bool)
    return np.where(mask, 2 * (actions - low) / (high - low) - 1, actions)


class OctoAdapter(VLAAdapter):
    environment_key = "vla.octo"

    def __init__(self, model_name: str = "checkpoints/vla/octo-base-1.5"):
        candidate = Path(model_name).expanduser()
        if not candidate.is_absolute():
            candidate = project_root() / candidate
        checkpoint = str(candidate) if candidate.exists() else f"hf://rail-berkeley/{model_name}"
        self.model = OctoModel.load_pretrained(checkpoint)

    def reset(self, initial_state: Any = None) -> None:
        pass

    def predict(self, image: np.ndarray, instruction: str, state: Any = None) -> torch.Tensor:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = {"image_primary": image[np.newaxis, np.newaxis, ...], "timestep_pad_mask": np.array([[True]])}
        tasks = self.model.create_tasks(texts=[prompt])
        actions = self.model.sample_actions(
            inputs, tasks,
            unnormalization_statistics=self.model.dataset_statistics["bridge_dataset"]["action"],
            rng=jax.random.PRNGKey(0),
        )[0]
        out = torch.zeros((actions.shape[0], 10), device="cuda", dtype=torch.float32)
        for i, raw in enumerate(actions):
            a = torch.as_tensor(_normalize(raw, self.model.dataset_statistics["bridge_dataset"]["action"]), device="cuda", dtype=torch.float32)
            a = torch.cat([a, a.new_zeros(3)])
            out[i] = rescale_bridge_action(a, wv_lo=-1, wv_hi=1, rd_lo=-1, rd_hi=1)
        return out
