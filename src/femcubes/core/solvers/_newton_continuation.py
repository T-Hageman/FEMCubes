"""Internal auxiliary-equilibrium continuation for difficult Newton solves."""

import time

import numpy as np
from petsc4py import PETSc

from femcubes.core.solvers._newton_trust_region import _NewtonTrustRegion
from femcubes.core.utils.mpi_utils import mprint


def _solve_newton_continuation(solver, step, residual_scale):
    """PRIVATE: Approach F=0 through F + eta*W*(x-anchor)=0.

    W is the positive initial tangent diagonal. Auxiliary equilibria provide
    starting iterates only; convergence is always tested on F itself. The
    displacement offset is maintained through the normal distributed update
    interface, so no field layout or component ordering is assumed here.
    """
    K, F = solver.physics.Get_Kmat_Fvec(step)
    solver._initialize_solver(F, step)
    diagonal = solver._positive_tangent_diagonal(K, F)
    scaling = diagonal.copy()
    scaling.getArray()[:] = np.sqrt(diagonal.getArray(readonly=True))
    offset = F.duplicate()
    offset.set(0.0)
    work = F.duplicate()
    eta = 0.0
    radius = None
    stage_iterations = 0
    stage_reference = None
    stage_residuals = []
    final_residual = np.inf
    shifted = None

    def augment(vector):
        if eta:
            work.pointwiseMult(diagonal, offset)
            vector.axpy(eta, work)

    def residual_evaluator(requested_step):
        vector = solver.physics.Get_Fvec(requested_step)
        augment(vector)
        return vector

    def update_solution(increment, requested_step):
        solver._update_solution(increment, requested_step)
        if eta:
            offset.axpy(1.0, increment)

    try:
        for iteration in range(solver.max_iterations):
            started = time.time()
            operator = K
            if eta:
                shifted = K.copy()
                diagonal.copy(result=work)
                work.scale(eta)
                shifted.setDiagonal(work, addv=PETSc.InsertMode.ADD_VALUES)
                shifted.assemble()
                operator = shifted
                augment(F)
            workspace = solver._solve_linear_system(operator, F, step)
            radius = solver._perform_newton_trust_step(
                operator, F, workspace.du, scaling, radius, step,
                residual_evaluator=residual_evaluator, update_solution=update_solution,
            )
            K, F = solver.physics.Get_Kmat_Fvec(step)
            if shifted is not None:
                shifted.destroy()
                shifted = None
            physical = solver._check_innerstep_convergence(F, step, iteration, residual_scale, started)
            final_residual = physical.norm
            if physical.converged:
                mprint(f"\t\tConverged in {iteration+1} iterations!")
                return

            if not eta:
                if iteration + 1 >= 20:
                    eta = 1.0
                    radius = None
                    mprint(f"\t\tStarting auxiliary continuation: damping = {eta:.6e}")
                continue

            stage_iterations += 1
            auxiliary = F.copy()
            try:
                augment(auxiliary)
                auxiliary_norm = solver._compute_residual_norm(auxiliary, step)
            finally:
                auxiliary.destroy()
            if stage_reference is None:
                stage_reference = auxiliary_norm
            stage_residuals.append(auxiliary_norm)
            auxiliary_converged = (
                auxiliary_norm <= solver.absolute_tolerance
                or auxiliary_norm <= solver.relative_tolerance * stage_reference
            )
            if auxiliary_converged:
                offset.set(0.0)
                if stage_iterations <= 5:
                    eta *= 0.5
            elif stage_iterations >= 35 or (
                stage_iterations >= 10 and auxiliary_norm > 0.9 * min(stage_residuals[-6:-1])
            ):
                # Retry this auxiliary equilibrium from its last accepted anchor.
                offset.copy(result=work)
                work.scale(-1.0)
                solver._update_solution(work, step)
                offset.set(0.0)
                K, F = solver.physics.Get_Kmat_Fvec(step)
                final_residual = solver._compute_residual_norm(F, step)
                eta *= 10.0
            else:
                continue
            stage_iterations = 0
            stage_reference = None
            stage_residuals.clear()
            radius = None
            mprint(f"\t\tAuxiliary continuation: damping = {eta:.6e}, physical RMS = {final_residual:.6e}")

        solver._raise_innerstep_nonconvergence(step, final_residual)
    finally:
        for vector in (diagonal, scaling, offset, work, shifted):
            if vector is not None:
                solver._destroy_petsc_object(vector)


class _NewtonContinuation(_NewtonTrustRegion):
    """PRIVATE: Newton trust-region iterations with auxiliary damping (``Step_Solvers``: ``"NewtonContinuation"``)."""

    def solve_innerstep_newton_continuation(self, step: int, residual_scale: np.ndarray) -> None:
        """Solve with auxiliary damping when ordinary trust-region updates stall.

        Acceptance uses the original physical residual. All auxiliary solves
        share the max_it budget. With line_search=False, use full-step Newton.
        """
        if not self.LineSearch:
            return self.solve_innerstep_newton(step, residual_scale)
        return _solve_newton_continuation(self, step, residual_scale)
