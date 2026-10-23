"""Bridge/OpenX action post-processing shared by compatible VLA adapters."""

from __future__ import annotations

import torch


def rescale_bridge_action(
    action: torch.Tensor,
    *,
    wv_lo: float = -0.05,
    wv_hi: float = 0.05,
    wv_post_scale_max: float = 1.75,
    wv_post_scale_min: float = -1.75,
    rd_lo: float = -0.25,
    rd_hi: float = 0.25,
    rd_post_scale_max: float = 1.4,
    rd_post_scale_min: float = -1.4,
) -> torch.Tensor:
    """Convert Bridge policy values to the WorldGym checkpoint range.

    The first seven values are ``xyz, rpy, gripper``. The WorldGym model uses
    ten values in total; the remaining three are reserved/padded by each VLA
    adapter.
    """

    action = action.clone()
    action[..., :3] = (action[..., :3] - wv_lo) / (wv_hi - wv_lo) * (
        wv_post_scale_max - wv_post_scale_min
    ) + wv_post_scale_min
    action[..., :3] = torch.clamp(
        action[..., :3], wv_post_scale_min, wv_post_scale_max
    )
    action[..., 3:6] = (action[..., 3:6] - rd_lo) / (rd_hi - rd_lo) * (
        rd_post_scale_max - rd_post_scale_min
    ) + rd_post_scale_min
    action[..., 3:6] = torch.clamp(
        action[..., 3:6], rd_post_scale_min, rd_post_scale_max
    )
    action[..., 6] = torch.where(action[..., 6] > 0.8, -1.0, 1.0)
    return action


__all__ = ["rescale_bridge_action"]
