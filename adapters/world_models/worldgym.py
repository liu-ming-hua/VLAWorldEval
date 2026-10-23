from pathlib import Path

import numpy as np
import torch

from adapters.base import WorldModelAdapter
from vlaworldeval.paths import project_root
from vlaworldeval.worldgym_runtime import runtime_directory, verify_runtime


class WorldGymAdapter(WorldModelAdapter):
    """Adapter around the repository's pretrained diffusion WorldModel."""

    environment_key = "world_models.worldgym"

    def __init__(self, checkpoint_path: str | Path, num_inference_steps: int = 10, **kwargs):
        import os
        import sys
        root = project_root()
        verify_runtime(root)
        source = str(runtime_directory(root))
        if source not in sys.path:
            sys.path.insert(0, source)
        os.environ.setdefault("WORLDGYM_VAE_PATH", str(root / "checkpoints/world_models/worldgym/vae/sd3_vae"))
        from worldgym_runtime.world_model import WorldModel
        self.model = WorldModel(str(checkpoint_path), **kwargs)
        self.model.diffusion.sampling_timesteps = num_inference_steps
        self.last_video: np.ndarray | None = None
        self.last_policy_state = None

    def reset(self, image: np.ndarray) -> None:
        image = np.asarray(image).copy()
        x = torch.from_numpy(image).cuda().float() / 255.0
        self.model.reset(x)
        self.last_video = None
        self.last_policy_state = None

    def step(self, action: torch.Tensor) -> np.ndarray:
        return self.rollout(action)[-1]

    def rollout(self, action: torch.Tensor) -> np.ndarray:
        """Generate all frames conditioned on an action chunk.

        ``action`` is ``[H,10]`` (or ``[10]`` for one action). The
        terminal frame remains available through ``step`` for legacy callers,
        while closed-loop evaluators can consume the complete imagined video.
        """
        action = torch.as_tensor(action, device="cuda", dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if getattr(self.model, "chunk_size", None) != action.shape[0]:
            self.model.chunk_size = action.shape[0]
        generated = {}
        start = self.model.curr_frame
        for index, x in self.model.generate_chunk(action):
            generated[int(index)] = x[0, -1].detach().cpu().numpy()
        if not generated:
            raise RuntimeError("WorldGym produced no frame for the action chunk")
        if sorted(generated) != list(range(start, start + action.shape[0])):
            raise RuntimeError(f"WorldGym generated incomplete frame indices: {sorted(generated)}")
        self.last_video = np.clip(np.stack([generated[i] for i in sorted(generated)]) * 255, 0, 255).astype(np.uint8)
        return self.last_video
