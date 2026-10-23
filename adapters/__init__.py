from importlib import import_module

from .base import VLAAdapter, WorldModelAdapter

__all__ = [
    "VLAAdapter",
    "WorldModelAdapter",
    "OpenVLAAdapter",
    "SpatialVLAAdapter",
    "RT1Adapter",
    "OctoAdapter",
    "GrootN15Adapter",
    "GrootN17Adapter",
    "OpenPIAdapter",
    "Pi0Adapter",
    "Pi05Adapter",
    "SmolVLAAdapter",
    "WorldGymAdapter",
    "GigaWorldAdapter",
    "GigaWorld1Adapter",
    "CtrlWorldAdapter",
    "GESimV2Adapter",
    "Cosmos3NanoAdapter",
    "Cosmos3NanoDroidFDAdapter",
    "A2WorldAdapter",
    "Tau0WMSimulatorAdapter",
]


_LAZY_VLA_IMPORTS = {
    "OpenVLAAdapter": ("adapters.vla.openvla", "OpenVLAAdapter"),
    "SpatialVLAAdapter": ("adapters.vla.spatialvla", "SpatialVLAAdapter"),
    "RT1Adapter": ("adapters.vla.rt1", "RT1Adapter"),
    "OctoAdapter": ("adapters.vla.octo", "OctoAdapter"),
    "GrootN15Adapter": ("adapters.vla.groot", "GrootN15Adapter"),
    "GrootN17Adapter": ("adapters.vla.groot", "GrootN17Adapter"),
    "OpenPIAdapter": ("adapters.vla.openpi", "OpenPIAdapter"),
    "Pi0Adapter": ("adapters.vla.openpi", "Pi0Adapter"),
    "Pi05Adapter": ("adapters.vla.openpi", "Pi05Adapter"),
    "SmolVLAAdapter": ("adapters.vla.smolvla", "SmolVLAAdapter"),
}

_LAZY_WORLD_MODEL_IMPORTS = {
    "WorldGymAdapter": ("adapters.world_models.worldgym", "WorldGymAdapter"),
    "GigaWorld1Adapter": ("adapters.world_models.gigaworld1", "GigaWorld1Adapter"),
    "CtrlWorldAdapter": ("adapters.world_models.ctrl_world", "CtrlWorldAdapter"),
    "GESimV2Adapter": ("adapters.world_models.ge_sim_v2", "GESimV2Adapter"),
    "Cosmos3NanoAdapter": ("adapters.world_models.cosmos3_nano", "Cosmos3NanoAdapter"),
    "Cosmos3NanoDroidFDAdapter": (
        "adapters.world_models.cosmos3_nano_droid_fd",
        "Cosmos3NanoDroidFDAdapter",
    ),
    "A2WorldAdapter": ("adapters.world_models.a2world", "A2WorldAdapter"),
    "Tau0WMSimulatorAdapter": ("adapters.world_models.tau0_wm_simulator", "Tau0WMSimulatorAdapter"),
}


def __getattr__(name):
    """Load model-specific dependencies only when that adapter is requested."""
    if name == "GigaWorldAdapter":
        value = __getattr__("GigaWorld1Adapter")
    elif name in _LAZY_VLA_IMPORTS:
        module_name, attribute_name = _LAZY_VLA_IMPORTS[name]
        try:
            value = getattr(import_module(module_name), attribute_name)
        except ImportError as exc:
            raise ImportError(
                f"{name} is not available in the current environment. "
                "Use its environment registry entry and the matching script "
                "under scripts/setup/."
            ) from exc
    elif name in _LAZY_WORLD_MODEL_IMPORTS:
        module_name, attribute_name = _LAZY_WORLD_MODEL_IMPORTS[name]
        try:
            value = getattr(import_module(module_name), attribute_name)
        except ImportError as exc:
            raise ImportError(
                f"{name} is not available in the current environment. "
                "Run the setup script for this world-model environment."
            ) from exc
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
