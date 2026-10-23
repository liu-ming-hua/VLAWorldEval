from importlib import import_module

__all__ = [
    "OpenVLAAdapter",
    "SpatialVLAAdapter",
    "RT1Adapter",
    "OctoAdapter",
    "GrootN15Adapter",
    "GrootN17Adapter",
    "GrootN17DroidAdapter",
    "OpenPIAdapter",
    "Pi0Adapter",
    "Pi05Adapter",
    "Pi0FastDroidAdapter",
    "SmolVLAAdapter",
    "CosmosDroidPolicyAdapter",
]

_LAZY_IMPORTS = {
    "OpenVLAAdapter": ("adapters.vla.openvla", "OpenVLAAdapter"),
    "SpatialVLAAdapter": ("adapters.vla.spatialvla", "SpatialVLAAdapter"),
    "RT1Adapter": ("adapters.vla.rt1", "RT1Adapter"),
    "OctoAdapter": ("adapters.vla.octo", "OctoAdapter"),
    "GrootN15Adapter": ("adapters.vla.groot", "GrootN15Adapter"),
    "GrootN17Adapter": ("adapters.vla.groot", "GrootN17Adapter"),
    "GrootN17DroidAdapter": ("adapters.vla.groot", "GrootN17DroidAdapter"),
    "OpenPIAdapter": ("adapters.vla.openpi", "OpenPIAdapter"),
    "Pi0Adapter": ("adapters.vla.openpi", "Pi0Adapter"),
    "Pi05Adapter": ("adapters.vla.openpi", "Pi05Adapter"),
    "Pi0FastDroidAdapter": ("adapters.vla.openpi", "Pi0FastDroidAdapter"),
    "SmolVLAAdapter": ("adapters.vla.smolvla", "SmolVLAAdapter"),
    "CosmosDroidPolicyAdapter": ("adapters.vla.cosmos_droid", "CosmosDroidPolicyAdapter"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(name)
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
