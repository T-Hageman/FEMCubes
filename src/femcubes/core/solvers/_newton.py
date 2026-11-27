"""Internal Newton--Raphson solver of one staggered step, with the residual-norm line search."""

import time

import numpy as np

from femcubes.core.utils.mpi_utils import mprint
from femcubes.core.solvers._step_solver_base import _BestIterate, _StepSolverBase


class _Newton(_StepSolverBase):
    """PRIVATE: Newton--Raphson iterations with optional backtracking line search (``Step_Solvers``: ``"Newton"``)."""

    def solve_innerstep_newton(self, step: int, residual_scale: np.ndarray) -> None:
        """Solve one staggered model step with Newton--Raphson."""
        K, F = self.physics.Get_Kmat_Fvec(step)
        self._initialize_solver(F, step)
        final_residual = np.inf
        best = _BestIterate()
        failing_line_searches = 0

        for iteration in range(self.max_iterations):
            t_start_iteration = time.time()
            workspace = self._solve_linear_system(K, F, step)
            if self.LineSearch:
                # the line search limits each trial step itself (see Perform_LineSearch)
                self._record_limited_field_updates(workspace.du, step)
                self.Perform_LineSearch(F, workspace.du, step)
                if self.line_search_nonfinite_abort > 0:
                    outcome = self.last_line_search_outcome
                    failing = outcome.nonfinite_trials > 0 and not outcome.merit_decreased
                    failing_line_searches = failing_line_searches + 1 if failing else 0
                    if failing_line_searches >= self.line_search_nonfinite_abort:
                        mprint(f"\t\tWARNING: Line search met non-finite residuals without reducing the merit "
                               f"in {failing_line_searches} consecutive iterations; aborting step {step}.")
                        raise RuntimeError(
                            f"Staggered step {step} line search produced non-finite residuals without "
                            f"descent in {failing_line_searches} consecutive Newton iterations."
                        )
            else:
                self._limit_positive_update(workspace.du, step)
                self._record_field_updates(workspace.du, step)
                self._update_solution(workspace.du, step)
            K, F = self.physics.Get_Kmat_Fvec(step)

            residual = self._check_innerstep_convergence(F,step,iteration,residual_scale,t_start_iteration)
            final_residual = residual.norm
            if residual.converged:
                mprint(f"\t\tConverged in {iteration+1} iterations!")
                return
            if self._track_best_iterate(best, residual.norm, step):
                break

        if self._accept_best_iterate(best, step):
            return
        self._raise_innerstep_nonconvergence(step, final_residual, iteration + 1)

    def Perform_LineSearch(self, F_old, du, step):
        """Apply safeguarded residual-norm backtracking line search.

        The trial iterates are ``u0 + alpha du``; when the step solves fields with
        bounds or update caps (``Positive_Fields``, ``Fraction_Fields``,
        ``Max_Field_Update``), they are ``u0 + limit(alpha du)`` instead, with the
        limiting of the Newton solvers evaluated at ``u0``. Limiting the full step
        first and then scaling it would search along a direction that need not
        decrease the residual norm; the limited path follows ``du`` for small
        ``alpha``, so a Newton direction ``du`` decreases it for small enough steps.

        Parameters
        ----------
        F_old : PETSc.Vec
            Residual vector before the trial update.
        du : PETSc.Vec
            Current nonlinear search direction (not yet limited).
        step : int
            Staggered solution step being updated.
        """
        return self._perform_residual_backtracking(
            F_old, du, step, self.physics.Get_Fvec, self._update_solution, limited=self._has_limited_fields(step),
        )
