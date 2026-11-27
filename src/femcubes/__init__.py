"""FEMCubes -- modular multi-physics finite-element framework built on FEniCSx.

The top-level package re-exports the main public entry points used by
examples and user scripts.  These exports are resolved lazily so that
lightweight submodule imports, such as ``femcubes.core.params_template``,
do not pull in the full mesh / physics / solver stack during package
initialisation.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Mesh", "Physics", "TimeSolver", "LinearSolver", "ParamsTemplate"]

_PUBLIC_EXPORTS = {
    "Mesh": ("femcubes.core.mesh.mesh", "Mesh"),
    "Physics": ("femcubes.core.physics.physics", "Physics"),
    "TimeSolver": ("femcubes.core.solvers.time_solver", "TimeSolver"),
    "LinearSolver": ("femcubes.core.solvers.solver_enums", "LinearSolver"),
    "ParamsTemplate": ("femcubes.core.params_template", "ParamsTemplate"),
}


def __getattr__(name: str):
    """Lazily resolve public re-exports on first access."""
    if name == "__version__":
        # The version is set from the git tag when the package is installed.
        from importlib.metadata import version

        return version("femcubes")
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
    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.physics.physics import Physics
    from femcubes.core.solvers.solver_enums import LinearSolver
    from femcubes.core.solvers.time_solver import TimeSolver
