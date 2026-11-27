"""Core framework package for FEMCubes.

The :mod:`femcubes.core` package contains the shared infrastructure that
the rest of FEMCubes builds on:

* :class:`~femcubes.core.mesh.mesh.Mesh` for geometry, fields, output,
  and visualization
* :class:`~femcubes.core.physics.physics.Physics` for model assembly and
  global weak-form aggregation
* :class:`~femcubes.core.solvers.time_solver.TimeSolver` for nonlinear
  and transient solves
* :class:`~femcubes.core.params_template.ParamsTemplate` for simulation
  configuration
* :class:`~femcubes.core.solvers.solver_enums.LinearSolver` for selecting
  the linear-solver backend

These re-exports are resolved lazily so importing lightweight core
submodules does not eagerly import ``Mesh`` and trigger generator
discovery during package initialisation.
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
    """Lazily resolve public core re-exports on first access."""
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
