"""Continuation must converge the original equations within one iteration budget."""

import numpy as np
import pytest

from femcubes.core.solvers.step_solver import StepSolver
from femcubes.core.solvers.tests.test_newton_trust_region import _Params, _Physics, _stress_problem


def _nonmonotone_response(x):
    # Newton from zero is attracted to a nonzero local minimum of ||F||.
    return (np.array([x[0]**3 - 2*x[0] + 2, x[1]-1, x[2]+1]),
            np.diag([3*x[0]**2 - 2, 1, 1]))


@pytest.mark.parametrize("ghosted", [False, True])
def test_continuation_crosses_a_residual_barrier_and_solves_original_equations(ghosted):
    physics = _Physics(_nonmonotone_response, [0, 0, 0], ghosted=ghosted)
    params = _Params()
    params.Step_Solvers = {0: "NewtonContinuation"}
    params.max_it = 100
    solver = StepSolver(params, physics)
    try:
        solver.Solve_Step()
        residual, _ = _nonmonotone_response(physics.x)
        assert np.linalg.norm(residual)/np.sqrt(3) < params.abs_tol
        assert physics.x[0] < -1
    finally:
        solver._cleanup_solver()
        physics.close()


def test_auxiliary_convergence_cannot_accept_a_nonzero_physical_residual():
    physics = _Physics(_nonmonotone_response, [0, 0, 0])
    params = _Params()
    params.Step_Solvers = {0: "NewtonContinuation"}
    params.max_it = 36
    solver = StepSolver(params, physics)
    original_solve = solver._solve_linear_system
    solves = []

    def count_solve(*args):
        solves.append(None)
        return original_solve(*args)

    # Count actual linear corrections to protect the global iteration budget.
    solver._solve_linear_system = count_solve
    try:
        with pytest.raises(RuntimeError, match="did not converge after 36 iterations"):
            solver.Solve_Step()
        assert len(solves) == params.max_it
        assert np.linalg.norm(_nonmonotone_response(physics.x)[0])/np.sqrt(3) > params.abs_tol
    finally:
        solver._cleanup_solver()
        physics.close()


@pytest.mark.parametrize("method", ["NewtonTrustRegion", "NewtonContinuation"])
def test_solver_updates_are_independent_of_petsc_ghost_storage(method):
    response, start = _stress_problem()
    updates = []
    for ghosted in (False, True):
        physics = _Physics(response, start, ghosted=ghosted)
        params = _Params()
        params.Step_Solvers = {0: method}
        params.max_it = 150
        solver = StepSolver(params, physics)
        try:
            solver.Solve_Step()
            updates.append(np.array(physics.updates))
        finally:
            solver._cleanup_solver()
            physics.close()
    np.testing.assert_allclose(updates[0], updates[1], rtol=1e-10, atol=1e-12)
