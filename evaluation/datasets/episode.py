"""Dataset-independent payload returned by admitted episode loaders."""
from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeInput:
    trial: dict
    replay_actions: list[np.ndarray]
    provenance: dict
