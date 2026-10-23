"""Explicit local bridge plugins, scoped by adapter category."""
from importlib import import_module
import inspect


def build_bridge_plugin(name, *, category, base, spec):
    module, separator, attribute = name.partition(":")
    prefix = f"adapters.{category}."
    if not separator or not module.startswith(prefix) or not all(part.isidentifier() for part in module.split(".")) or not attribute.isidentifier():
        raise ValueError(f"Bridge plugin must be adapters.{category}.<module>:<class>")
    try:
        implementation = getattr(import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Bridge plugin is unavailable: {name}") from exc
    if not isinstance(implementation, type) or not issubclass(implementation, base) or inspect.isabstract(implementation):
        raise ValueError(f"Bridge plugin must implement {base.__name__}")
    try:
        return implementation(spec)
    except TypeError as exc:
        raise ValueError(f"Bridge plugin must accept its spec as the sole constructor argument: {name}") from exc
