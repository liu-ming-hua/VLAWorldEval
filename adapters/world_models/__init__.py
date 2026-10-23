"""Lazy exports for model-specific world-model adapters."""

from importlib import import_module


_IMPORTS = {
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
    "Cosmos3DroidJointFDAdapter": ("adapters.world_models.cosmos3_droid_joint_fd", "Cosmos3DroidJointFDAdapter"),
    "Tau0WMSimulatorAdapter": ("adapters.world_models.tau0_wm_simulator", "Tau0WMSimulatorAdapter"),
    "IWSAdapter": ("adapters.world_models.iws", "IWSAdapter"),
    "DreamDojoAdapter": ("adapters.world_models.dreamdojo", "DreamDojoAdapter"),
    "PlayWorldAdapter": ("adapters.world_models.playworld", "PlayWorldAdapter"),
    "DreamTrueAdapter": ("adapters.world_models.dreamtrue", "DreamTrueAdapter"),
    "IRASimAdapter": ("adapters.world_models.irasim", "IRASimAdapter"),
}

__all__ = [
    "WorldGymAdapter",
    "GigaWorld1Adapter",
    "GigaWorldAdapter",
    "CtrlWorldAdapter",
    "GESimV2Adapter",
    "Cosmos3NanoAdapter",
    "Cosmos3NanoDroidFDAdapter",
    "A2WorldAdapter",
    "Cosmos3DroidJointFDAdapter",
    "Tau0WMSimulatorAdapter",
    "IWSAdapter",
    "DreamDojoAdapter",
    "PlayWorldAdapter",
    "DreamTrueAdapter",
    "IRASimAdapter",
]


def __getattr__(name):
    if name == "GigaWorldAdapter":
        value = __getattr__("GigaWorld1Adapter")
    elif name in _IMPORTS:
        module_name, attribute_name = _IMPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
