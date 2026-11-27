"""Nonlinear staggered-solve driver for one time step."""

import time

import numpy as np

from femcubes.core.utils.mpi_utils import mprint
from femcubes.core.solvers._constrained_step import _ConstrainedStep
from femcubes.core.solvers._lbfgs import _LBFGS
from femcubes.core.solvers._newton import _Newton
from femcubes.core.solvers._newton_continuation import _NewtonContinuation
from femcubes.core.solvers._newton_trust_region import _NewtonTrustRegion
from femcubes.core.solvers._step_solver_base import LineSearchOutcome  # noqa: F401 - part of this module's public API


class StepSolver(_Newton, _NewtonContinuation, _NewtonTrustRegion, _LBFGS, _ConstrainedStep):
    """Staggered nonlinear solver for a single time step.

    Iterates over staggered solution steps, performing Newton--Raphson,
    Newton trust-region/continuation, or limited-memory BFGS iterations with optional line search within each
    step.  Multi-pass iteration handles coupling between staggered fields.
    It is used directly for one-off nonlinear solves and is also the base
    class for :class:`femcubes.core.solvers.time_solver.TimeSolver`.

    The nonlinear methods live in separate internal modules, one mixin per
    method (``_Newton``, ``_NewtonTrustRegion``, ``_NewtonContinuation``,
    ``_LBFGS``, and ``_ConstrainedStep`` for ``Solve_Constrained_Step``), on
    top of the shared helpers of ``_StepSolverBase``; this class holds the
    configuration and the staggered loop.

    Parameters
    ----------
    params : ParamsTemplate
        Must define ``Solution_Steps``, ``n_passes``, ``max_it``,
        ``rel_tol``, ``abs_tol``, ``line_search``, and
        ``line_search_lims``.  A step converges when the RMS residual is
        below ``abs_tol``, or when it is below ``rel_tol`` times the
        residual after the first update of the step.  Optional settings:

        ``n_passes_force`` (default False)
            Perform all ``n_passes`` staggered passes even when an earlier
            pass has converged.
        ``rel_tol_max_residual`` (default: no limit)
            Accept the relative criterion only while the RMS residual is
            also below this value, so a step whose first update overshoots
            is not accepted on the relative criterion alone.
        ``Stagnation_Iterations`` (default 0: off)
            End a Newton or Newton trust-region solve early when its RMS
            residual has not fallen below half of its smallest earlier
            value for this many iterations (e.g. Newton cycling between a
            few states), instead of spending ``max_it`` iterations before
            the time step is cut.
        ``Stagnation_Accept_Factor`` (default: off, at least 1)
            When such a solve ends without converging (stagnated or at
            ``max_it``), restore the iterate with the smallest RMS residual
            and accept it if that residual is below this factor times
            ``abs_tol``.
        ``Positive_Fields`` (default: none)
            Fields (e.g. concentrations) whose positive values must stay
            positive.  The Newton solvers take each decrease of such a
            value in log space, smoothly capped at the factor
            ``Positive_Update_Factor`` (default 1e-4, in (0, 1)) per
            iteration, so a full step can neither drive it negative nor
            switch abruptly between limited and unlimited updates.  The
            Newton line search limits every trial step this way (see
            ``Perform_LineSearch``).
        ``Fraction_Fields`` (default: none)
            Fields (e.g. surface coverages) kept in (0, 1) the same way:
            decreases as for ``Positive_Fields``, increases with the
            distance ``1 - u`` to the upper bound.
        ``Max_Field_Update`` (default: none)
            Maps fields (e.g. potentials) to the largest change per
            iteration.  When it is exceeded, the whole Newton update is
            scaled down smoothly (damped Newton), keeping its direction.
        ``DEBUG_FIELD_CONVERGENCE`` (default False)
            Print, after every Newton iteration, the RMS residual over the
            rows of each field and the largest change of each field by the
            last update, to find the fields that keep a step from
            converging.
        ``Step_Solvers``
            Maps each staggered step to ``"Newton"``,
            ``"NewtonTrustRegion"``, ``"NewtonContinuation"`` or
            ``"LBFGS"``; unspecified steps use Newton.
        ``LBFGS_History`` (default 10)
            Number of correction pairs retained by l-BFGS.  The directional
            l-BFGS line search is configured by
            ``LBFGS_LineSearch_MaxIter``, ``LBFGS_LineSearch_Tolerance``,
            and ``LBFGS_LineSearch_MaxResidualGrowth``.
            ``LBFGS_StagnationIterations`` (default 5) controls when the
            base tangent is refreshed at the current iterate, and
            ``LBFGS_RecoveryLineSearchTolerance`` the tighter directional
            search used for the first direction from that fresh tangent.
    physics : Physics
        Assembled physics object.

    Public API
    ----------
    ``Solve_Step()``
        Solve all staggered fields for the current time level.
    ``get_runtime_timings()``
        Return cumulative rank-local timings for profiled solver phases.
    ``get_runtime_timing_counts()``
        Return cumulative rank-local invocation counts for profiled phases.
    ``reset_runtime_timings()``
        Clear cumulative phase timings and invocation counts.
    ``solve_innerstep_newton(step, residual_scale)``
        Solve one staggered model step with Newton--Raphson.
    ``solve_innerstep_newton_trust_region(step, residual_scale)``
        Solve one staggered model step with scaled dogleg trust-region updates.
    ``solve_innerstep_newton_continuation(step, residual_scale)``
        Recover stalled solves through auxiliary equilibria with added damping.
    ``solve_innerstep_bfgs(step, residual_scale)``
        Solve one staggered model step with limited-memory BFGS.
    ``Perform_LineSearch(F_old, du, step)``
        Apply residual-norm backtracking for Newton.
    ``Perform_LBFGS_LineSearch(F_old, direction, step)``
        Apply the directional-gradient secant search for l-BFGS.

    Notes
    -----
    The high-level solve order is:

    1. loop over staggered passes
    2. loop over solution steps within each pass
    3. assemble the initial ``K`` and ``F`` for the current step
    4. solve for a Newton or l-BFGS increment ``du``
    5. update the mesh fields
    6. refresh ``K``/``F`` for Newton; l-BFGS normally refreshes only
       ``F``, but also refreshes ``K`` after a rejected curvature update
    7. test convergence from the RMS force-vector L2 norm
    8. perform a post-pass consistency check across all steps
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, params, physics) -> None:
        """Create a staggered nonlinear solver bound to one ``Physics`` object."""
        super().__init__(params, physics)

        self.n_Staggered_Steps: int = max(set(params.Solution_Steps.values()))+1
        if self.n_Staggered_Steps > 1:
            self.n_passes = params.n_passes
            self.n_passes_force = getattr(params, "n_passes_force", False)
        else:
            self.n_passes = 1
        # Optional early abort of a diverging staggered iteration: when the
        # post-pass residual of any step grows by more than
        # ``staggered_divergence_factor`` for ``staggered_divergence_passes``
        # consecutive passes, Solve_Step raises RuntimeError so that TimeSolver
        # resets the step with a smaller dt. Disabled (0) by default; without
        # it a diverging staggered loop is only caught once an inner Newton
        # solve blows up, which can take many expensive passes.
        self.staggered_divergence_passes = int(getattr(params, "staggered_divergence_passes", 0))
        self.staggered_divergence_factor = float(getattr(params, "staggered_divergence_factor", 1.5))
        if self.staggered_divergence_passes < 0:
            raise ValueError("staggered_divergence_passes must be a non-negative integer.")
        if self.staggered_divergence_factor <= 1.0:
            raise ValueError("staggered_divergence_factor must be greater than one.")
        # Optional early abort of a stagnating staggered iteration: when no
        # step reduces its post-pass residual to below
        # ``staggered_stagnation_factor`` times its previous value for
        # ``staggered_stagnation_passes`` consecutive passes, Solve_Step raises
        # RuntimeError. Disabled (0) by default.
        self.staggered_stagnation_passes = int(getattr(params, "staggered_stagnation_passes", 0))
        self.staggered_stagnation_factor = float(getattr(params, "staggered_stagnation_factor", 0.9))
        if self.staggered_stagnation_passes < 0:
            raise ValueError("staggered_stagnation_passes must be a non-negative integer.")
        if not 0.0 < self.staggered_stagnation_factor < 1.0:
            raise ValueError("staggered_stagnation_factor must lie strictly between zero and one.")
        # Optional early abort of a Newton solve whose residual backtracking
        # keeps meeting non-finite trial residuals without reducing the merit,
        # for ``line_search_nonfinite_abort`` consecutive iterations. Disabled
        # (0) by default: an occasional non-finite trial at a long step is
        # normal, whereas repeated ones without any descent indicate a step
        # that cannot be recovered at the current time increment.
        self.line_search_nonfinite_abort = int(getattr(params, "line_search_nonfinite_abort", 0))
        if self.line_search_nonfinite_abort < 0:
            raise ValueError("line_search_nonfinite_abort must be a non-negative integer.")
        self.last_line_search_outcome = None
        
        # Solver parameters
        self.SolverMethod = self._normalize_step_solvers(getattr(params, "Step_Solvers", None))
        
        self.LBFGS_History = getattr(params, "LBFGS_History", 10)
        self.LBFGS_LineSearch_MaxIter = getattr(params, "LBFGS_LineSearch_MaxIter", 20)
        self.LBFGS_LineSearch_Tolerance = float(getattr(params, "LBFGS_LineSearch_Tolerance", 0.9))
        self.LBFGS_LineSearch_MaxResidualGrowth = float(getattr(params, "LBFGS_LineSearch_MaxResidualGrowth", 2.0))
        self.LBFGS_StagnationIterations = getattr(params, "LBFGS_StagnationIterations", 5)
        self.LBFGS_RecoveryLineSearchTolerance = float(getattr(params, "LBFGS_RecoveryLineSearchTolerance", 0.1))
        self.LBFGS_PowellDamping = float(getattr(params, "LBFGS_PowellDamping", 0.2))

        self.max_iterations = params.max_it
        self.relative_tolerance = params.rel_tol
        self.absolute_tolerance = params.abs_tol
        self.relative_tolerance_max_residual = float(getattr(params, "rel_tol_max_residual", np.inf))
        self.StagnationIterations = int(getattr(params, "Stagnation_Iterations", 0))
        if self.StagnationIterations < 0:
            raise ValueError(f"Stagnation_Iterations must not be negative, not {self.StagnationIterations}.")
        self.StagnationAcceptFactor = getattr(params, "Stagnation_Accept_Factor", None)
        if self.StagnationAcceptFactor is not None:
            self.StagnationAcceptFactor = float(self.StagnationAcceptFactor)
            if not (np.isfinite(self.StagnationAcceptFactor) and self.StagnationAcceptFactor >= 1.0):
                raise ValueError(f"Stagnation_Accept_Factor must be finite and at least 1, not {self.StagnationAcceptFactor}.")
        self.DEBUG_FIELD_CONVERGENCE = bool(getattr(params, "DEBUG_FIELD_CONVERGENCE", False))
        self._field_updates = {}
        self._field_update_locations = {}
        self.PositiveFields = list(getattr(params, "Positive_Fields", []))
        self.FractionFields = list(getattr(params, "Fraction_Fields", []))
        self.PositiveUpdateFactor = float(getattr(params, "Positive_Update_Factor", 1.0e-4))
        if not 0.0 < self.PositiveUpdateFactor < 1.0:
            raise ValueError(f"Positive_Update_Factor must lie in (0, 1), not {self.PositiveUpdateFactor}.")
        for option, names in (("Positive_Fields", self.PositiveFields), ("Fraction_Fields", self.FractionFields)):
            if names:
                unknown = sorted(set(names) - set(physics.mesh.fields))
                if unknown:
                    raise ValueError(f"{option} contains unknown fields: {unknown}.")
        both = sorted(set(self.PositiveFields) & set(self.FractionFields))
        if both:
            raise ValueError(f"Fields listed in both Positive_Fields and Fraction_Fields: {both}.")
        self.MaxFieldUpdate = {name: float(limit) for name, limit in dict(getattr(params, "Max_Field_Update", {})).items()}
        if self.MaxFieldUpdate:
            unknown = sorted(set(self.MaxFieldUpdate) - set(physics.mesh.fields))
            if unknown:
                raise ValueError(f"Max_Field_Update contains unknown fields: {unknown}.")
            invalid = {name: limit for name, limit in self.MaxFieldUpdate.items() if not limit > 0.0}
            if invalid:
                raise ValueError(f"Max_Field_Update limits must be positive: {invalid}.")
            overlap = sorted(set(self.MaxFieldUpdate) & (set(self.PositiveFields) | set(self.FractionFields)))
            if overlap:
                raise ValueError(f"Fields with a Max_Field_Update must not also be Positive_Fields or Fraction_Fields: {overlap}.")
        self.LineSearch = params.line_search
        self.LineSearch_Lims = params.line_search_lims

        self._debug_plotting = getattr(params, "_DEBUG_PLOTTING", False)
        self.DEBUG_MEMORY = getattr(params, "DEBUG_MEMORY", False)

    def get_runtime_timings(self) -> dict[str, float]:
        """Return cumulative rank-local solver-phase wall times."""

        return self._runtime_profiler.snapshot()

    def get_runtime_timing_counts(self) -> dict[str, int]:
        """Return cumulative rank-local solver-phase invocation counts."""

        return self._runtime_profiler.snapshot_counts()

    def reset_runtime_timings(self) -> None:
        """Clear cumulative rank-local solver-phase profiling data."""

        self._runtime_profiler.reset()

    def Solve_Step(self) -> None:
        """Solve the nonlinear system for one full time step.

        The method updates fields in-place on ``self.physics.mesh``.  It
        does not advance time or commit history variables; those belong
        to :class:`TimeSolver`.
        """
        residual_scale = np.full(self.n_Staggered_Steps, np.nan)
        previous_post_pass_norms = None
        growing_passes = 0
        stagnant_passes = 0

        for passes in range(self.n_passes):
            if self.n_passes > 1:
                mprint(f" Pass {passes+1}/{self.n_passes}:")
                
            pass_converged = True
            for step in range(self.n_Staggered_Steps):
                if self.n_Staggered_Steps > 1:
                    mprint(f"\tStaggered Step {step}...")

                fields_to_solve = self.physics.get_fields_for_step(step)
                if not fields_to_solve:
                    mprint("Warning: No fields to solve for this step")
                    return

                solver_method = self.SolverMethod[step]
                mprint(f"\t\tUsing {solver_method} nonlinear solver")
                self._field_updates.pop(step, None)
                self._field_update_locations.pop(step, None)

                t_start_step = time.time()
                if solver_method == "Newton":
                    self.solve_innerstep_newton(step, residual_scale)
                elif solver_method == "NewtonTrustRegion":
                    self.solve_innerstep_newton_trust_region(step, residual_scale)
                elif solver_method == "NewtonContinuation":
                    self.solve_innerstep_newton_continuation(step, residual_scale)
                else:
                    self.solve_innerstep_bfgs(step, residual_scale)

                if self.DEBUG_PRINT_TIMING:
                    mprint(f"\t\t Staggered Step duration: {time.time() - t_start_step:.4f} s")

            # Recompute residuals after all steps to ensure cross-step consistency
            post_pass_norms = []
            for step in range(self.n_Staggered_Steps):
                F_check = self.physics.Get_Fvec(step)
                residual = self._residual_state(F_check, step, residual_scale)
                mprint(f"\tPost-pass Step {step}: RMS Residual = {residual.norm:.6e}, Relative = {residual.relative:.6e}")
                post_pass_norms.append(residual.norm)
                if not residual.converged:
                    pass_converged = False

            if self.n_passes > 1 and pass_converged and not self.n_passes_force:
                mprint(" All staggered steps converged after post-pass check; ending multi-pass early.")
                break

            if self.staggered_divergence_passes > 0 and previous_post_pass_norms is not None:
                grew = any(
                    np.isfinite(previous) and previous > 0.0
                    and (not np.isfinite(current) or current > self.staggered_divergence_factor * previous)
                    for previous, current in zip(previous_post_pass_norms, post_pass_norms)
                )
                growing_passes = growing_passes + 1 if grew else 0
                if growing_passes >= self.staggered_divergence_passes:
                    mprint(f" WARNING: Post-pass residual grew by more than a factor "
                           f"{self.staggered_divergence_factor:g} in {growing_passes} consecutive passes; "
                           "aborting the staggered iteration.")
                    raise RuntimeError(
                        f"Staggered iteration diverged over {growing_passes} passes "
                        f"(post-pass residuals {previous_post_pass_norms} -> {post_pass_norms})."
                    )
            if self.staggered_stagnation_passes > 0 and previous_post_pass_norms is not None:
                improved = any(
                    np.isfinite(current) and current <= self.staggered_stagnation_factor * previous
                    for previous, current in zip(previous_post_pass_norms, post_pass_norms)
                )
                stagnant_passes = 0 if improved else stagnant_passes + 1
                if stagnant_passes >= self.staggered_stagnation_passes:
                    mprint(f" WARNING: No post-pass residual decreased by a factor "
                           f"{self.staggered_stagnation_factor:g} in {stagnant_passes} consecutive passes; "
                           "aborting the staggered iteration.")
                    raise RuntimeError(
                        f"Staggered iteration stagnated over {stagnant_passes} passes "
                        f"(post-pass residuals {previous_post_pass_norms} -> {post_pass_norms})."
                    )
            previous_post_pass_norms = post_pass_norms

        if self.n_Staggered_Steps > 1 and not pass_converged:
            mprint(f" WARNING: Reached {self.n_passes} staggered passes with residuals above tolerance; retaining the current iterate.")

        if self.DEBUG_MEMORY:
            self._print_memory_debug_summary()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _normalize_step_solvers(self, configured_solvers):
        """PRIVATE: Validate and complete the per-step nonlinear solver map."""
        if configured_solvers is None:
            configured_solvers = {}
        if not isinstance(configured_solvers, dict):
            raise TypeError("Step_Solvers must be a dictionary keyed by step number.")

        invalid_steps = set(configured_solvers) - set(range(self.n_Staggered_Steps))
        if invalid_steps:
            raise ValueError(f"Step_Solvers contains unknown step numbers: {sorted(invalid_steps)}.")

        solver_methods = {}
        for step in range(self.n_Staggered_Steps):
            method = configured_solvers.get(step, "Newton")
            if method not in ("Newton", "NewtonTrustRegion", "NewtonContinuation", "LBFGS"):
                raise ValueError(f"Invalid solver method '{method}' for step {step}. Must be 'Newton', 'NewtonTrustRegion', 'NewtonContinuation' or 'LBFGS'.")
            solver_methods[step] = method
        return solver_methods
