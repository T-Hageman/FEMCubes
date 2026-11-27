"""Internal limited-memory BFGS solver of one staggered step, with its directional line search."""

import time

import numpy as np

from femcubes.core.utils.mpi_utils import mprint
from femcubes.core.solvers._step_solver_base import _StepSolverBase


class _LBFGS(_StepSolverBase):
    """PRIVATE: Limited-memory BFGS iterations preconditioned by the initial tangent (``Step_Solvers``: ``"LBFGS"``)."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve_innerstep_bfgs(self, step: int, residual_scale: np.ndarray) -> None:
        """Solve one staggered model step with limited-memory BFGS."""
        K, F = self.physics.Get_Kmat_Fvec(step)
        self._initialize_solver(F, step)
        history = []
        use_history = True
        best_residual = self._compute_residual_norm(F, step)
        stagnant_iterations = 0
        stagnation_recovery = False
        tight_recovery_search = False
        final_residual = np.inf

        try:
            for iteration in range(self.max_iterations):
                t_start_iteration = time.time()
                active_history = history if use_history else []
                direction = self._lbfgs_direction(K, F, step, active_history)
                K, F, direction = self._ensure_lbfgs_descent_direction(K, F, direction, step, active_history)

                try:
                    F_old = F.copy()
                    try:
                        alpha = self._apply_innerstep_line_search("LBFGS",F,direction,step,tight_recovery=tight_recovery_search,)
                        tight_recovery_search = False
                        F = self.physics.Get_Fvec(step)
                        history_updated = self._update_lbfgs_history(history, K, direction, alpha, F_old, F)
                        use_history = bool(history)
                        if not history_updated:
                            mprint("\t\t\tRefreshing l-BFGS base tangent while retaining its valid history after a rejected curvature update")
                            K, F = self.physics.Get_Kmat_Fvec(step)
                    finally:
                        self._destroy_petsc_object(F_old)

                    residual = self._check_innerstep_convergence(F,step,iteration,residual_scale,t_start_iteration)
                    final_residual = residual.norm
                    if residual.converged:
                        mprint(f"\t\tConverged in {iteration+1} iterations!")
                        return

                    if residual.norm < best_residual:
                        best_residual = residual.norm
                        stagnant_iterations = 0
                        if stagnation_recovery:
                            stagnation_recovery = False
                            use_history = history_updated
                            mprint("\t\t\tLeaving l-BFGS recovery after the fresh active history improved the best F2 residual")
                    else:
                        stagnant_iterations += 1

                    if (stagnant_iterations >= self.LBFGS_StagnationIterations):
                        K, F = self.physics.Get_Kmat_Fvec(step)
                        use_history = bool(history)
                        stagnation_recovery = True
                        tight_recovery_search = True
                        stagnant_iterations = 0
                        mprint("\t\t\tRefreshing the l-BFGS base tangent at the current iterate while retaining valid correction pairs after sustained stagnation")
                finally:
                    self._destroy_petsc_object(direction)
        finally:
            self._destroy_lbfgs_history(history)

        self._raise_innerstep_nonconvergence(step, final_residual)

    def Perform_LBFGS_LineSearch(self, F_old, direction, step, curvature_tolerance=None, prefer_residual=False):
        """Apply a safeguarded directional-gradient line search for l-BFGS.

        The search seeks a root of ``F(x + alpha*p).p`` using secant
        interpolation inside a sign-changing bracket.  This directly
        promotes the positive-curvature correction pairs required by
        l-BFGS while requiring residual-only assembly.

        ``curvature_tolerance`` overrides the configured strong-Wolfe
        curvature scale.  It is used internally to tighten the first search
        after refreshing the base tangent because of F2 stagnation.

        When ``prefer_residual`` is true, reaching the curvature tolerance
        ends the search but the lowest-F2 positive-curvature trial is
        restored.  This prevents a recovery search from discarding an
        earlier trial with a substantially smaller residual merely to move
        closer to the directional stationary point.
        """
        alpha_min = float(self.LineSearch_Lims[0])
        alpha_max = float(self.LineSearch_Lims[1])
        if (not np.isfinite(alpha_min) or not np.isfinite(alpha_max) or alpha_min <= 0.0 or alpha_max < alpha_min):
            raise ValueError("line_search_lims must contain finite positive values [alpha_min, alpha_max] with alpha_min <= alpha_max.")

        psi0 = F_old.dot(direction)
        force_norm0 = F_old.norm()
        mprint(f"\t\tStarting l-BFGS Directional Line Search: F.p = {psi0:.6e}"
        )
        if force_norm0 <= 1.0e-30:
            mprint("\t\t\tAccepted alpha = 0.000000")
            return 0.0

        if not np.isfinite(psi0) or psi0 >= 0.0:
            self._update_solution_scaled(direction, alpha_min, step)
            mprint(f"\t\t\tForced minimum alpha = {alpha_min:.6f}: direction is not a finite descent direction")
            return alpha_min

        if curvature_tolerance is None:
            curvature_tolerance = self.LBFGS_LineSearch_Tolerance
        if (not np.isfinite(curvature_tolerance) or not 0.0 < curvature_tolerance < 1.0):
            raise ValueError("curvature_tolerance must be finite and between 0 and 1.")
        projection_tolerance = curvature_tolerance * abs(psi0)
        residual_limit = self.LBFGS_LineSearch_MaxResidualGrowth * force_norm0
        direction_norm = direction.norm()

        current_alpha = 0.0
        alpha = min(max(1.0, alpha_min), alpha_max)
        lower_alpha = 0.0
        lower_psi = psi0
        upper_alpha = None
        upper_psi = None
        best_alpha = None
        best_force_norm = np.inf
        best_projection = np.inf
        best_residual_alpha = None
        best_residual_norm = force_norm0
        residual_backtracking = False
        evaluated_alphas = []

        for _ in range(self.LBFGS_LineSearch_MaxIter):
            evaluated_alphas.append(alpha)
            self._update_solution_scaled(direction, alpha - current_alpha, step)
            current_alpha = alpha

            F_trial = self.physics.Get_Fvec(step)
            psi = F_trial.dot(direction)
            force_norm = F_trial.norm()
            curvature = alpha * (psi - psi0)
            curvature_scale = alpha * direction_norm * (force_norm + force_norm0)
            positive_curvature = (np.isfinite(curvature) and curvature > 1.0e-12 * curvature_scale)
            residual_safe = (np.isfinite(force_norm) and force_norm <= residual_limit)

            mprint(f"\t\t\tTrial alpha = {alpha:.6f}, F.p = {psi:.6e}, |F|2 = {force_norm:.6e}")

            if (np.isfinite(force_norm) and force_norm < best_residual_norm):
                best_residual_norm = force_norm
                best_residual_alpha = alpha

            if (np.isfinite(psi) and positive_curvature and residual_safe):
                projection = abs(psi)
                if (force_norm < best_force_norm or (force_norm == best_force_norm and projection < best_projection)):
                    best_force_norm = force_norm
                    best_projection = projection
                    best_alpha = alpha
                if projection <= projection_tolerance:
                    if prefer_residual:
                        break
                    mprint(f"\t\t\tAccepted alpha = {alpha:.6f}")
                    return alpha

            if (not residual_backtracking and alpha >= alpha_max and np.isfinite(psi) and psi < 0.0 and force_norm >= force_norm0):
                residual_backtracking = True

            if residual_backtracking:
                if (np.isfinite(force_norm) and force_norm < force_norm0):
                    mprint(f"\t\t\tAccepted residual-reducing alpha = {alpha:.6f}")
                    return alpha
                next_alpha = max(alpha_min, 0.5 * alpha)
            elif not np.isfinite(psi):
                next_alpha = max(alpha_min, 0.5 * alpha)
            else:
                if psi >= 0.0:
                    if upper_alpha is None or alpha < upper_alpha:
                        upper_alpha = alpha
                        upper_psi = psi
                elif alpha > lower_alpha and (upper_alpha is None or alpha < upper_alpha):
                    lower_alpha = alpha
                    lower_psi = psi

                if upper_alpha is not None:
                    bracket_width = upper_alpha - lower_alpha
                    if bracket_width <= 1.0e-12 * max(1.0, alpha_max):
                        break

                    denominator = upper_psi - lower_psi
                    if np.isfinite(denominator) and abs(denominator) > 1.0e-30:
                        next_alpha = upper_alpha - upper_psi * bracket_width / denominator
                    else:
                        next_alpha = 0.5 * (lower_alpha + upper_alpha)

                    margin = 0.1 * bracket_width
                    if (not np.isfinite(next_alpha) or next_alpha <= lower_alpha + margin or next_alpha >= upper_alpha - margin):
                        next_alpha = 0.5 * (lower_alpha + upper_alpha)
                elif positive_curvature and residual_safe:
                    if alpha >= alpha_max:
                        if force_norm < force_norm0:
                            mprint(f"\t\t\tAccepted residual-reducing boundary alpha = {alpha:.6f}")
                            return alpha
                        residual_backtracking = True
                        next_alpha = max(alpha_min, 0.5 * alpha)
                    else:
                        next_alpha = min(alpha_max, 2.0 * alpha)
                else:
                    next_alpha = max(alpha_min, 0.5 * alpha)

            next_alpha = min(max(next_alpha, alpha_min), alpha_max)
            if abs(next_alpha - alpha) <= 1.0e-14 * max(1.0, alpha):
                break
            if any(abs(next_alpha - previous_alpha) <= 1.0e-14 * max(1.0, next_alpha) for previous_alpha in evaluated_alphas):
                break
            alpha = next_alpha

        if best_alpha is not None:
            if best_alpha != current_alpha:
                self._update_solution_scaled(direction, best_alpha - current_alpha, step)
            mprint(f"\t\t\tAccepted lowest-residual curvature alpha = {best_alpha:.6f}")
            return best_alpha

        if best_residual_alpha is not None:
            if best_residual_alpha != current_alpha:
                self._update_solution_scaled(direction, best_residual_alpha - current_alpha, step)
            mprint(f"\t\t\tAccepted best residual-reducing alpha = {best_residual_alpha:.6f}")
            return best_residual_alpha

        if current_alpha != alpha_min:
            self._update_solution_scaled(direction, alpha_min - current_alpha, step)
        mprint(f"\t\t\tForced minimum alpha = {alpha_min:.6f}: no curvature-producing trial was found")
        return alpha_min

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _lbfgs_direction(self, K, F, step, history):
        """PRIVATE: Return an owned l-BFGS search direction.

        The tangent assembled at the start of the staggered step provides
        the initial inverse-Hessian action.  Subsequent curvature corrections
        use the standard limited-memory two-loop recursion.
        """
        q = F.copy()
        alphas = []
        try:
            for s_vec, y_vec, rho in reversed(history):
                alpha = rho * s_vec.dot(q)
                alphas.append(alpha)
                q.axpy(-alpha, y_vec)

            initial_scale = self._lbfgs_initial_inverse_scale(K, step, history)
            workspace = self._solve_linear_system(K, q, step)
            direction = workspace.du.copy()
            direction.scale(-1.0)
            direction.scale(initial_scale)

            for (s_vec, y_vec, rho), alpha in zip(history, reversed(alphas)):
                beta = rho * y_vec.dot(direction)
                direction.axpy(alpha - beta, s_vec)

            direction.scale(-1.0)
            return direction
        finally:
            self._destroy_petsc_object(q)

    def _lbfgs_initial_inverse_scale(self, K, step, history):
        """PRIVATE: Scale the tangent inverse to the latest secant curvature."""
        if not history:
            return 1.0

        _s_vec, y_vec, rho = history[-1]
        workspace = self._solve_linear_system(K, y_vec, step)
        y_h0_y = -y_vec.dot(workspace.du)
        curvature = 1.0 / rho
        if (not np.isfinite(y_h0_y) or not np.isfinite(curvature) or y_h0_y <= 0.0 or curvature <= 0.0):
            return 1.0

        scale = curvature / y_h0_y
        if not np.isfinite(scale) or scale <= 0.0:
            return 1.0
        return scale

    def _ensure_lbfgs_descent_direction(self, K, F, direction, step, history):
        """PRIVATE: Refresh or repair a non-descent l-BFGS direction."""
        directional_derivative = F.dot(direction)
        if (np.isfinite(directional_derivative)and directional_derivative < 0.0):
            return K, F, direction

        mprint(f"\t\t\tRefreshing l-BFGS base tangent: non-descent direction (F.p = {directional_derivative:.6e})")
        self._destroy_petsc_object(direction)
        K, F = self.physics.Get_Kmat_Fvec(step)
        direction = self._lbfgs_direction(K, F, step, history)
        directional_derivative = F.dot(direction)

        if (history and (not np.isfinite(directional_derivative) or directional_derivative >= 0.0)):
            mprint("\t\t\tIgnoring l-BFGS corrections for this direction while retaining their history")
            self._destroy_petsc_object(direction)
            direction = self._lbfgs_direction(K, F, step, [])
            directional_derivative = F.dot(direction)

        if (not np.isfinite(directional_derivative) or directional_derivative >= 0.0):
            self._destroy_petsc_object(direction)
            direction = self._diagonally_scaled_descent_direction(K, F)
            mprint("\t\t\tUsing an absolute-diagonal scaled descent fallback direction")

        return K, F, direction

    def _diagonally_scaled_descent_direction(self, K, F):
        """PRIVATE: Return ``-abs(diag(K))^-1 F`` as an owned vector.

        The absolute diagonal is a positive mixed-field scale even when a
        softening tangent is indefinite.  It therefore supplies a genuine
        descent direction without mixing displacement and phase-field units
        as an unpreconditioned ``-F`` fallback would.
        """
        diagonal = F.duplicate()
        K.getDiagonal(diagonal)
        with diagonal.localForm() as diagonal_local:
            values = diagonal_local.array
            values[:] = np.abs(values)
            invalid = ~np.isfinite(values) | (values <= np.finfo(float).tiny)
            values[invalid] = 1.0

        direction = F.copy()
        direction.pointwiseDivide(F, diagonal)
        direction.scale(-1.0)
        self._destroy_petsc_object(diagonal)
        return direction

    def _update_lbfgs_history(self, history, K, direction, alpha, F_old, F_new):
        """PRIVATE: Add a valid, Powell-damped correction pair.

        When the measured secant curvature is too small or negative, the
        residual difference is blended with the positive base-tangent
        action.  This is the standard Powell damping safeguard: it keeps the
        l-BFGS inverse positive definite while still allowing the correction
        history to develop through a non-convex fracture increment.
        """
        s_vec = self._copy_scaled_vector(direction, alpha)
        y_vec = F_new.copy()
        y_vec.axpy(-1.0, F_old)

        curvature = s_vec.dot(y_vec)
        base_action = F_new.duplicate()
        K.mult(s_vec, base_action)
        base_curvature = s_vec.dot(base_action)
        damping_threshold = self.LBFGS_PowellDamping
        if (np.isfinite(base_curvature) and base_curvature > 0.0 and (not np.isfinite(curvature) or curvature < damping_threshold * base_curvature)):
            theta = (1.0 - damping_threshold) * base_curvature / (base_curvature - curvature)

            y_vec.scale(theta)
            y_vec.axpy(1.0 - theta, base_action)
            curvature = s_vec.dot(y_vec)
            mprint(f"\t\t\tPowell-damped l-BFGS update: s.y = {curvature:.6e}")

        curvature_scale = s_vec.norm() * y_vec.norm()
        curvature_limit = 1.0e-10 * curvature_scale
        if not np.isfinite(curvature) or curvature <= curvature_limit:
            mprint(f"\t\t\tSkipping l-BFGS update: insufficient positive curvature (s.y = {curvature:.6e})")
            self._destroy_petsc_object(s_vec)
            self._destroy_petsc_object(y_vec)
            self._destroy_petsc_object(base_action)
            return False

        self._destroy_petsc_object(base_action)
        history.append((s_vec, y_vec, 1.0 / curvature))
        if len(history) > self.LBFGS_History:
            old_s, old_y, _ = history.pop(0)
            self._destroy_petsc_object(old_s)
            self._destroy_petsc_object(old_y)
        return True

    def _destroy_lbfgs_history(self, history):
        """PRIVATE: Release all PETSc vectors owned by an l-BFGS history."""
        for s_vec, y_vec, _ in history:
            self._destroy_petsc_object(s_vec)
            self._destroy_petsc_object(y_vec)
        history.clear()
