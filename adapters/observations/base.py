"""View-layout bridge contracts."""

from __future__ import annotations

from abc import ABC

from evaluation.contracts import ObservationBundle, WorldModelOutput
from evaluation.manifests import ViewBridgeSpec


class ViewBridge(ABC):
    """Create world-model views and recover policy views after generation."""

    def __init__(self, spec: ViewBridgeSpec):
        self.spec = spec

    def prepare_observation(self, observation: ObservationBundle) -> ObservationBundle:
        return observation

    def process_world_output(self, output: WorldModelOutput) -> WorldModelOutput:
        return output


class IdentityViewBridge(ViewBridge):
    pass


def build_view_bridge(spec: ViewBridgeSpec) -> ViewBridge:
    if ":" in spec.name:
        from adapters.plugins import build_bridge_plugin
        return build_bridge_plugin(spec.name, category="observations", base=ViewBridge, spec=spec)
    if spec.name == "identity":
        return IdentityViewBridge(spec)
    if spec.name == "droid_cosmos_composite_v1":
        from adapters.observations.droid import DroidCosmosViewBridge

        return DroidCosmosViewBridge(spec)
    if spec.name == "droid_ctrl_views_v1":
        from adapters.observations.droid import DroidCtrlViewBridge
        return DroidCtrlViewBridge(spec)
    raise ValueError(f"Unknown view bridge {spec.name!r}")


__all__ = ["IdentityViewBridge", "ViewBridge", "build_view_bridge"]
