from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from adapters.actions.bridge import rescale_bridge_action
from adapters.base import VLAAdapter


class OpenVLAAdapter(VLAAdapter):
    environment_key = "vla.openvla"

    def __init__(self, model_name_or_path: str = "openvla-7b"):
        model_id = model_name_or_path if Path(model_name_or_path).exists() else f"openvla/{model_name_or_path}"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda().eval()

    def reset(self, initial_state: Any = None) -> None:
        pass

    @torch.no_grad()
    def predict(self, image: np.ndarray, instruction: str, state: Any = None) -> torch.Tensor:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, Image.fromarray(image)).to(device="cuda", dtype=torch.bfloat16)
        actions = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        action = torch.as_tensor(actions, device="cuda", dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = torch.cat([action, torch.zeros((*action.shape[:-1], 3), device="cuda")], dim=-1)
        return torch.stack([rescale_bridge_action(a) for a in action])
