"""Internal Newton trust-region solver of one staggered step (scaled dogleg updates)."""

import time

import numpy as np

from femcubes.core.utils.mpi_utils import mprint
from femcubes.core.solvers._step_solver_base import _BestIterate, _StepSolverBase


class _NewtonTrustRegion(_StepSolverBase):
    """PRIVATE: Newton iterations with scaled dogleg trust-region updates (``Step_Solvers``: ``"NewtonTrustRegion"``)."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_innerstep_newton_trust_region(self, step: int, residual_scale: np.ndarray) -> None:
        """Solve using a scaled dogleg trust region for the residual norm.

        Uses the exact, possibly nonsymmetric Jacobian. The Cauchy direction
        is based on J.T*F, so no elastic-energy potential is assumed. With
        line_search=False this delegates to ordinary full-step Newton.
        """
        if not self.LineSearch:
            return self.solve_innerstep_newton(step, residual_scale)
        K, F = self.physics.Get_Kmat_Fvec(step)
        self._initialize_solver(F, step)
        scaling = self._positive_tangent_diagonal(K, F)
        owned = scaling.getArray()
        owned[:] = np.sqrt(owned)
        radius = None
        final_residual = np.inf
        best = _BestIterate()
        try:
            for iteration in range(self.max_iterations):
                started = time.time()
                workspace = self._solve_linear_system(K, F, step)
                self._limit_positive_update(workspace.du, step)
                self._record_field_updates(workspace.du, step)
                radius = self._perform_newton_trust_step(K, F, workspace.du, scaling, radius, step)
                K, F = self.physics.Get_Kmat_Fvec(step)
                residual = self._check_innerstep_convergence(F, step, iteration, residual_scale, started)
                final_residual = residual.norm
                if residual.converged:
                    mprint(f"\t\tConverged in {iteration+1} iterations!")
                    return
                if self._track_best_iterate(best, residual.norm, step):
                    break
            if self._accept_best_iterate(best, step):
                return
            self._raise_innerstep_nonconvergence(step, final_residual, iteration + 1)
        finally:
            self._destroy_petsc_object(scaling)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _perform_newton_trust_step(self, K, F, newton_direction, scaling, radius, step,
                                 residual_evaluator=None, update_solution=None):
        """PRIVATE: Try dogleg corrections; retain the smallest trial on exhaustion.

        Returns the radius for the next iteration, or None after exhaustion so
        that the next iteration restarts from its full Newton step.
        """
        residual_evaluator = self.physics.Get_Fvec if residual_evaluator is None else residual_evaluator
        update_solution = self._update_solution if update_solution is None else update_solution
        # Residual assembly can overwrite F in place during trial evaluations.
        base = F.copy()
        newton = newton_direction.copy()
        gradient = F.duplicate()
        cauchy = F.duplicate()
        action = F.duplicate()
        current = F.duplicate()
        current.set(0.0)
        try:
            newton.pointwiseMult(newton, scaling)
            newton_norm = newton.norm()
            if newton_norm == 0.0:
                update_solution(newton_direction, step)
                return radius
            if radius is None:
                radius = newton_norm

            K.multTranspose(base, gradient)
            gradient.pointwiseDivide(gradient, scaling)
            cauchy.pointwiseDivide(gradient, scaling)
            K.mult(cauchy, action)
            denominator = action.dot(action)
            if not np.isfinite(denominator) or denominator <= 0.0:
                self._perform_residual_backtracking(base, newton_direction, step, residual_evaluator, update_solution)
                return radius
            gradient.copy(result=cauchy)
            cauchy.scale(-gradient.dot(gradient)/denominator)
            base_squared = base.norm()**2

            for trial in range(20):
                candidate = self._dogleg_increment(newton, cauchy, radius)
                step_norm = candidate.norm()
                candidate.pointwiseDivide(candidate, scaling)
                change = candidate.copy()
                try:
                    change.axpy(-1.0, current)
                    update_solution(change, step)
                    candidate.copy(result=current)
                finally:
                    self._destroy_petsc_object(change)
                    self._destroy_petsc_object(candidate)

                trial_squared = residual_evaluator(step).norm()**2
                K.mult(current, action)
                action.axpy(1.0, base)
                predicted = base_squared-action.norm()**2
                ratio = (base_squared-trial_squared)/predicted if predicted > 0.0 else -np.inf
                if not np.isfinite(ratio):
                    ratio = -np.inf
                mprint(f"\t\t\tTrust trial {trial+1}: radius = {radius:.6e}, |F|2 = {0.5*trial_squared:.6e}, agreement = {ratio:.6e}")
                if ratio < 0.25:
                    radius = max(0.25*step_norm, np.finfo(float).tiny)
                elif ratio > 0.75 and step_norm >= 0.9*radius:
                    radius *= 2.0
                if ratio > 1e-3:
                    return radius

            mprint("\t\t\tTrust-region trials exhausted; accepting the smallest tried correction for recovery")
            # The exhausted radius is vanishingly small and the ratio test cannot
            # grow it again; restart the next iteration from the Newton step.
            return None
        finally:
            for vector in (base, newton, gradient, cauchy, action, current):
                self._destroy_petsc_object(vector)

    def _dogleg_increment(self, newton, cauchy, radius):
        """PRIVATE: Return a scaled-coordinate step on the Cauchy--Newton dogleg."""
        if newton.norm() <= radius:
            return newton.copy()
        cauchy_norm = cauchy.norm()
        if cauchy_norm >= radius:
            return self._copy_scaled_vector(cauchy, radius/cauchy_norm)
        difference = newton.copy()
        try:
            difference.axpy(-1.0, cauchy)
            a = difference.dot(difference)
            b = cauchy.dot(difference)
            remaining = max(radius**2-cauchy_norm**2, 0.0)
            root = np.sqrt(b*b+a*remaining)
            fraction = remaining/(root+b) if b > 0.0 else (root-b)/a
            result = cauchy.copy()
            result.axpy(fraction, difference)
            return result
        finally:
            self._destroy_petsc_object(difference)
