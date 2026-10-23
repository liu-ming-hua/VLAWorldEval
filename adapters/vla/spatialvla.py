from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from adapters.actions.bridge import rescale_bridge_action
from adapters.base import VLAAdapter


class SpatialVLAAdapter(VLAAdapter):
    environment_key = "vla.spatialvla"

    def __init__(self, model_name: str = "spatialvla-4b-224-pt"):
        model_id = model_name if "/" in model_name else f"IPEC-COMMUNITY/{model_name}"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).eval().cuda()

    def reset(self, initial_state: Any = None) -> None:
        pass

    def predict(self, image: np.ndarray, instruction: str, state: Any = None) -> torch.Tensor:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(images=[Image.fromarray(image)], text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)
        outputs = self.model.predict_action(inputs)
        actions = self.processor.decode_actions(outputs, unnorm_key="bridge_orig/1.0.0")["actions"]
        stats = self.processor.statistics["bridge_orig/1.0.0"]["action"]
        low, high = np.asarray(stats["q01"]), np.asarray(stats["q99"])
        mask = np.asarray(stats.get("mask", np.ones_like(low)), dtype=bool)
        actions = np.where(mask, 2 * (actions - low) / (high - low) - 1, actions)
        out = torch.zeros((actions.shape[0], 10), device="cuda", dtype=torch.float32)
        for i, raw in enumerate(actions):
            a = torch.as_tensor(raw, device="cuda", dtype=torch.float32)
            a = torch.cat([a, a.new_zeros(3)])
            out[i] = rescale_bridge_action(a, wv_lo=-1, wv_hi=1, rd_lo=-1, rd_hi=1)
        return out
