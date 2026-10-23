"""Action-space bridge contracts used between policy and world-model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from evaluation.contracts import ActionChunk, ObservationBundle
from evaluation.manifests import ActionBridgeSpec


class ActionBridge(ABC):
    """Convert one typed VLA action chunk into a world-model action chunk."""

    def __init__(self, spec: ActionBridgeSpec):
        self.spec = spec

    @abstractmethod
    def convert(
        self,
        action: ActionChunk,
        *,
        observation: ObservationBundle,
        step_index: int,
    ) -> ActionChunk:
        pass


class IdentityActionBridge(ActionBridge):
    """Auditable no-op bridge for already compatible action protocols."""

    def __init__(self, spec: ActionBridgeSpec):
        super().__init__(spec)
        if spec.source != spec.target:
            raise ValueError("IdentityActionBridge requires identical source and target specs")

    def convert(
        self,
        action: ActionChunk,
        *,
        observation: ObservationBundle,
        step_index: int,
    ) -> ActionChunk:
        self.spec.source.wrap(action.values, source=action.source)
        return self.spec.target.wrap(action.values.copy(), source=self.spec.name)


class CallableActionBridge(ActionBridge):
    """Bridge backed by an explicit conversion function.

    The callback receives the native ``ActionChunk`` and current observation,
    so conversions that need the current EEF pose cannot silently omit it.
    """

    def __init__(
        self,
        spec: ActionBridgeSpec,
        converter: Callable[[ActionChunk, ObservationBundle, int], Any],
    ):
        super().__init__(spec)
        self.converter = converter

    def convert(
        self,
        action: ActionChunk,
        *,
        observation: ObservationBundle,
        step_index: int,
    ) -> ActionChunk:
        self.spec.source.wrap(action.values, source=action.source)
        converted = self.converter(action, observation, step_index)
        return self.spec.target.wrap(converted, source=self.spec.name)


def build_action_bridge(
    spec: ActionBridgeSpec,
    *,
    converter: Callable[[ActionChunk, ObservationBundle, int], Any] | None = None,
) -> ActionBridge:
    if ":" in spec.name:
        if converter is not None:
            raise ValueError("Bridge plugin does not accept a separate converter")
        from adapters.plugins import build_bridge_plugin
        return build_bridge_plugin(spec.name, category="actions", base=ActionBridge, spec=spec)
    if spec.name == "identity":
        if converter is not None:
            raise ValueError("identity action bridge does not accept a converter")
        return IdentityActionBridge(spec)
    if spec.name == "openpi_to_cosmos3_droid_v1":
        if converter is not None:
            raise ValueError(f"{spec.name} uses its versioned built-in converter")
        from adapters.actions.openpi_droid import OpenPICosmosDroidActionBridge
        return OpenPICosmosDroidActionBridge(spec)
    if spec.name == "groot_n17_to_cosmos3_droid_v1":
        if converter is not None:
            raise ValueError("N1.7 bridge uses its built-in converter")
        from adapters.actions.groot_n17_droid import GrootN17CosmosDroidActionBridge
        return GrootN17CosmosDroidActionBridge(spec)
    if spec.name == "droid_policy_to_ctrl_world_v2":
        if converter is not None:
            raise ValueError("DROID/Ctrl bridge uses its built-in converter")
        from adapters.actions.droid_ctrl import DroidCtrlActionBridge
        return DroidCtrlActionBridge(spec)
    if spec.name == "groot_n15_to_cosmos3_droid_v1":
        if converter is not None:
            raise ValueError(f"{spec.name} uses its versioned built-in converter")
        from adapters.actions.droid import GrootN15CosmosDroidActionBridge

        return GrootN15CosmosDroidActionBridge(spec)
    if converter is None:
        raise ValueError(
            f"Action bridge {spec.name!r} requires an explicit converter implementation"
        )
    return CallableActionBridge(spec, converter)


__all__ = [
    "ActionBridge",
    "CallableActionBridge",
    "IdentityActionBridge",
    "build_action_bridge",
]
