"""Generic action-space bridges for the arm/gripper evaluation stack."""

from .base import (
    ActionBridge,
    CallableActionBridge,
    IdentityActionBridge,
    build_action_bridge,
)
def rescale_bridge_action(*args, **kwargs):
    """Legacy Torch conversion; keep model dependencies out of core imports."""
    from .bridge import rescale_bridge_action as implementation
    return implementation(*args, **kwargs)

__all__ = [
    "ActionBridge",
    "CallableActionBridge",
    "IdentityActionBridge",
    "build_action_bridge",
    "rescale_bridge_action",
]
