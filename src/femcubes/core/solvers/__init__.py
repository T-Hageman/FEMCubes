"""Solver sub-package for FEMCubes.

The solver stack consists of:

* :class:`~femcubes.core.solvers.step_solver.StepSolver` for one
  nonlinear staggered solve
* :class:`~femcubes.core.solvers.time_solver.TimeSolver` for transient
  time stepping
* :class:`~femcubes.core.solvers.solver_enums.LinearSolver` for choosing
  the PETSc linear-solver backend
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["LinearSolver", "StepSolver", "TimeSolver"]

_PUBLIC_EXPORTS = {
    "LinearSolver": ("femcubes.core.solvers.solver_enums", "LinearSolver"),
    "StepSolver": ("femcubes.core.solvers.step_solver", "StepSolver"),
    "TimeSolver": ("femcubes.core.solvers.time_solver", "TimeSolver"),
}


def __getattr__(name: str):
    """Lazily resolve public solver re-exports on first access."""
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
    from .solver_enums import LinearSolver
    from .step_solver import StepSolver
    from .time_solver import TimeSolver
