"""Physics sub-package for FEMCubes.

This package exposes the composite :class:`Physics` class used by the
solver stack.  The internal underscored modules split the physics layer
into mesh/parameter storage, model lifecycle management, and global
assembly helpers.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Physics"]

_PUBLIC_EXPORTS = {
    "Physics": ("femcubes.core.physics.physics", "Physics"),
}


def __getattr__(name: str):
    """Lazily resolve public physics re-exports on first access."""
    try:
        module_name, attr_name = _PUBLIC_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    """Return the module attributes plus lazy public exports."""
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from .physics import Physics
