"""Observation/view bridges shared by VLA × world-model pairs."""

from .base import IdentityViewBridge, ViewBridge, build_view_bridge
from .droid import DroidCosmosViewBridge

__all__ = [
    "DroidCosmosViewBridge",
    "IdentityViewBridge",
    "ViewBridge",
    "build_view_bridge",
]
