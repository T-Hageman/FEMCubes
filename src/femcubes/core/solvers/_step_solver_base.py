"""Internal helpers shared by the nonlinear solvers of one staggered step."""

from dataclasses import dataclass, field
import gc
import time

from dolfinx import fem
from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

from femcubes.core.utils.mpi_utils import comm, mprint
from femcubes.core.solvers._lin_solvers import _LinSolvers


@dataclass(frozen=True)
class LineSearchOutcome:
    """Outcome of one Newton residual-backtracking line search.

    ``alpha`` is the accepted step length, ``nonfinite_trials`` the number of
    trial step lengths whose residual merit was NaN or infinite, and
    ``merit_decreased`` whether the accepted step reduced the merit below its
    starting value.
    """

    alpha: float
    nonfinite_trials: int
    merit_decreased: bool


@dataclass(frozen=True)
class _ResidualState:
    """PRIVATE: Residual values used by every nonlinear solution method."""

    norm: float
    relative: float
    converged: bool


@dataclass
class _BestIterate:
    """PRIVATE: The smallest residual of a nonlinear solve so far, the running minimum after each iteration, and
    the fields of the best iterate (kept only while ``Stagnation_Accept_Factor`` could accept it)."""

    norm: float = np.inf
    minima: list = field(default_factory=list)
    state: dict | None = None


class _StepSolverBase(_LinSolvers):
    """PRIVATE: Residual norms, convergence tests, update limiting and line-search helpers of the nonlinear solvers.

    The solver mixins (:class:`_Newton`, :class:`_NewtonTrustRegion`, :class:`_NewtonContinuation`, :class:`_LBFGS`,
    :class:`_ConstrainedStep`) inherit from this class, and
    :class:`femcubes.core.solvers.step_solver.StepSolver` combines them. The attributes used here (tolerances,
    ``PositiveFields``, ``LineSearch``, ...) are set by ``StepSolver.__init__``.
    """

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _perform_residual_backtracking(self, F_old, du, step, residual_evaluator, update_solution, limited=False):
        """PRIVATE: Backtrack using the supplied residual and incremental update; with ``limited``, along the path
        ``u0 + limit(alpha du)`` of ``_limit_positive_update`` (evaluated at ``u0``) instead of ``u0 + alpha du``."""
        phi0 = 0.5 * F_old.norm()**2
        mprint(f"\t\tStarting Line Search: Initial merit = {phi0:.6e}")
        if phi0 <= 1.0e-30:
            mprint("\t\t\tAccepted alpha = 0.000000")
            self.last_line_search_outcome = LineSearchOutcome(0.0, 0, True)
            return 0.0

        alpha = 1.0
        alpha_min = float(self.LineSearch_Lims[0])
        shrink = 0.5
        alpha_prev = 0.0
        best_alpha = 0.0
        best_phi = phi0
        merit_tol = 1.0e-12 * phi0
        nonfinite_trials = 0
        applied = None          # limited path: the increment from u0 that is currently applied

        def move_to(target):
            nonlocal alpha_prev, applied
            if limited:
                if applied is not None:         # back to u0, where the limiting is evaluated
                    applied.scale(-1.0)
                    update_solution(applied, step)
                    self._destroy_petsc_object(applied)
                    applied = None
                if target > 0.0:
                    applied = self._copy_scaled_vector(du, target)
                    self._limit_positive_update(applied, step)
                    update_solution(applied, step)
            else:
                change = self._copy_scaled_vector(du, target - alpha_prev)
                try:
                    update_solution(change, step)
                finally:
                    self._destroy_petsc_object(change)
            alpha_prev = target

        try:
            while alpha >= alpha_min:
                move_to(alpha)

                F_trial = residual_evaluator(step)
                phi = 0.5 * F_trial.norm()**2
                mprint(f"\t\t\tTrial alpha = {alpha:.6f}, |F|2 = {phi:.6e}")
                if not np.isfinite(phi):
                    nonfinite_trials += 1

                if phi < best_phi:
                    best_phi = phi
                    best_alpha = alpha

                if phi <= phi0 - merit_tol:
                    mprint(f"\t\t\tAccepted alpha = {alpha:.6f}")
                    self.last_line_search_outcome = LineSearchOutcome(alpha, nonfinite_trials, True)
                    return alpha

                alpha *= shrink

            if best_alpha > 0.0 and best_alpha != alpha_prev:
                move_to(best_alpha)

            mprint(f"\t\t\tAccepted alpha = {alpha_prev:.6f}")
            self.last_line_search_outcome = LineSearchOutcome(alpha_prev, nonfinite_trials, best_phi < phi0)
            return alpha_prev
        finally:
            if applied is not None:
                self._destroy_petsc_object(applied)

    def _apply_innerstep_line_search(
        self,
        solver_method,
        F,
        direction,
        step,
        *,
        tight_recovery=False,
    ):
        """PRIVATE: Apply the configured update policy for either solver."""
        if not self.LineSearch:
            self._update_solution(direction, step)
            return 1.0

        if solver_method == "Newton":
            return self.Perform_LineSearch(F, direction, step)
        if solver_method == "LBFGS":
            if tight_recovery:
                return self.Perform_LBFGS_LineSearch(
                    F,
                    direction,
                    step,
                    curvature_tolerance=(
                        self.LBFGS_RecoveryLineSearchTolerance
                    ),
                    prefer_residual=True,
                )
            return self.Perform_LBFGS_LineSearch(F, direction, step)
        raise ValueError(
            f"Unsupported nonlinear solver method '{solver_method}'."
        )

    def _residual_state(self, F, step, residual_scale):
        """PRIVATE: Evaluate the common absolute and relative criteria."""
        residual_norm = self._compute_residual_norm(F, step)
        if np.isnan(residual_scale[step]):
            residual_scale[step] = residual_norm
        relative_residual = (
            residual_norm / (residual_scale[step] + 1.0e-16)
        )
        converged = (
            residual_norm < self.absolute_tolerance
            or (relative_residual < self.relative_tolerance
                and residual_norm < self.relative_tolerance_max_residual)
        )
        return _ResidualState(
            norm=float(residual_norm),
            relative=float(relative_residual),
            converged=bool(converged),
        )

    def _field_residual_squares(self, F, step):
        """PRIVATE: Sum of squares and number of the (row-scaled) residual rows of each field of ``step`` (collective)."""
        mesh = self.physics.mesh
        offsets = mesh.get_step_block_offsets(step)
        scale = mesh.Get_Residual_Row_Scale(step)
        local = F.getArray(readonly=True) * (1.0 if scale is None else scale)
        sums = np.array([[np.dot(local[a:b], local[a:b]), b - a] for a, b in zip(offsets, offsets[1:])], dtype=float)
        comm.Allreduce(MPI.IN_PLACE, sums, op=MPI.SUM)
        return dict(zip(mesh.step_fields.get(step, []), sums))

    def _record_field_updates(self, du, step):
        """PRIVATE: Record the largest change of each field of ``step`` by a Newton update, and where it occurs, for
        ``DEBUG_FIELD_CONVERGENCE`` (collective)."""
        if not self.DEBUG_FIELD_CONVERGENCE:
            return
        mesh = self.physics.mesh
        names = mesh.step_fields.get(step, [])
        offsets = mesh.get_step_block_offsets(step)
        owned = du.getArray(readonly=True)
        largest = np.array([np.max(np.abs(owned[a:b]), initial=0.0) for a, b in zip(offsets, offsets[1:])], dtype=float)
        self._field_update_locations[step] = self._largest_update_locations(owned, step, largest)
        comm.Allreduce(MPI.IN_PLACE, largest, op=MPI.MAX)
        self._field_updates[step] = dict(zip(names, largest))

    def _record_limited_field_updates(self, du, step):
        """PRIVATE: ``_record_field_updates`` for the limited full step ``limit(du)``, leaving ``du`` unchanged."""
        if not self.DEBUG_FIELD_CONVERGENCE:
            return
        limited = du.copy()
        try:
            self._limit_positive_update(limited, step)
            self._record_field_updates(limited, step)
        finally:
            self._destroy_petsc_object(limited)

    def _largest_update_locations(self, owned, step, local_largest):
        """PRIVATE: Coordinates of the node with the largest change of each mesh field of ``step`` (None for
        global fields and empty blocks), from the rank that holds it (collective)."""
        mesh = self.physics.mesh
        names = mesh.step_fields.get(step, [])
        offsets = mesh.get_step_block_offsets(step)
        local = []
        for name, a, b, value in zip(names, offsets, offsets[1:], local_largest):
            location = None
            if b > a and not mesh.Is_Global_Field(name):
                space = mesh.fields[name].function_space
                node = int(np.argmax(np.abs(owned[a:b]))) // space.dofmap.index_map_bs
                location = self._dof_location(mesh, space, node)
            local.append((value, location))
        gathered = comm.allgather(local)
        return {name: max((rank[i] for rank in gathered), key=lambda entry: entry[0])[1] for i, name in enumerate(names)}

    @staticmethod
    def _dof_location(mesh, space, node):
        """PRIVATE: Coordinates of one local DOF node: its interpolation point, or for elements without
        pointwise evaluation (Bernstein) the midpoint of the mesh entity carrying it; None when neither exists."""
        try:
            points = space.tabulate_dof_coordinates()
        except RuntimeError:
            try:
                points = mesh._entity_dof_points(space)
            except ValueError:      # Several DOFs per mesh entity (degree > 2)
                return None
        return tuple(float(x) for x in points[node][: mesh.mesh.geometry.dim])

    def _track_best_iterate(self, best, norm, step):
        """PRIVATE: Record the residual of a nonconverged iterate; keep the iterate itself while it is the best one
        that ``Stagnation_Accept_Factor`` could accept. Returns whether the solve stagnates (``Stagnation_Iterations``)."""
        if not (self.StagnationIterations or self.StagnationAcceptFactor):
            return False
        if norm < best.norm:
            best.norm = norm
            if self.StagnationAcceptFactor is not None and norm < self.StagnationAcceptFactor * self.absolute_tolerance:
                mesh = self.physics.mesh
                best.state = {name: mesh.fields[name].x.array.copy() for name in mesh.step_fields.get(step, [])}
        best.minima.append(best.norm)
        n = self.StagnationIterations
        if n and len(best.minima) > n and best.minima[-1] > 0.5 * best.minima[-1 - n]:
            mprint(f"\t\tStagnated: the RMS residual has not halved in {n} iterations.")
            return True
        return False

    def _accept_best_iterate(self, best, step):
        """PRIVATE: Restore and accept the best iterate of a nonconverged solve if ``Stagnation_Accept_Factor`` allows it."""
        if best.state is None:
            return False
        mesh = self.physics.mesh
        for name, values in best.state.items():
            mesh.fields[name].x.array[:] = values
        mprint(f"\t\tAccepted the best iterate: RMS residual {best.norm:.6e} < Stagnation_Accept_Factor * abs_tol.")
        return True

    def _print_field_convergence(self, F, step):
        """PRIVATE: Print the RMS residual over the rows of each field and its change by the last update (``DEBUG_FIELD_CONVERGENCE``)."""
        updates = self._field_updates.get(step, {})
        locations = self._field_update_locations.get(step, {})
        for name, (total, rows) in self._field_residual_squares(F, step).items():
            rms = np.sqrt(total / rows) if rows else 0.0
            change = f"{updates[name]:.3e}" if name in updates else "-"
            where = "" if locations.get(name) is None else " at (" + ", ".join(f"{x:.4g}" for x in locations[name]) + ")"
            mprint(f"\t\t\t{name}: RMS residual = {rms:.3e}, last update max |du| = {change}{where}")

    def _check_innerstep_convergence(
        self,
        F,
        step,
        iteration,
        residual_scale,
        iteration_start,
    ):
        """PRIVATE: Report and validate one method-independent residual."""
        residual = self._residual_state(F, step, residual_scale)
        mprint(
            f"\t\tIteration {iteration:2d}: "
            f"RMS Residual = {residual.norm:.6e}, "
            f"Relative = {residual.relative:.6e}"
        )
        if self.DEBUG_FIELD_CONVERGENCE:
            self._print_field_convergence(F, step)

        if self.DEBUG_PRINT_TIMING:
            mprint(
                "\t\t\t Iteration duration: "
                f"{time.time() - iteration_start:.4f} s"
            )

        if self._debug_plotting:
            with self._runtime_profiler.measure("plotting"):
                self.physics.mesh.Plot_Fields()

        if (
            not residual.converged
            and iteration > 0
            and residual.relative > 1.0e4
        ):
            mprint(
                f"\t\tWARNING: Diverging residual in step {step}!"
            )
            mprint(
                f"\t\tCurrent RMS residual: {residual.norm:.6e}, "
                f"Relative: {residual.relative:.6e}"
            )
            raise RuntimeError(
                f"Staggered step {step} diverged after "
                f"{iteration+1} iterations."
            )

        return residual

    def _raise_innerstep_nonconvergence(self, step, residual_norm, iterations=None):
        """PRIVATE: Raise the common nonconvergence error (iteration limit, or stagnation after ``iterations``)."""
        iterations = self.max_iterations if iterations is None else iterations
        mprint(
            f"\t\tWARNING: Did not converge in "
            f"{iterations} iterations!"
        )
        mprint(f"\t\tFinal RMS residual: {residual_norm:.6e}")
        raise RuntimeError(
            f"Staggered step {step} did not converge after "
            f"{iterations} iterations."
        )

    def _compute_residual_norm(self, F, step=None):
        """PRIVATE: Compute the RMS Euclidean norm of the force vector (rows scaled as the mesh asks, see ``_scaled_norm``)."""
        global_dofs = F.getSize() if hasattr(F, "getSize") else 1
        return self._scaled_norm(F, step) / np.sqrt(max(global_dofs, 1))

    def _scaled_norm(self, F, step=None):
        """PRIVATE: Euclidean norm of a step residual, its owned rows scaled by ``mesh.Get_Residual_Row_Scale`` (axisymmetric meshes)."""
        scale = None if step is None else self.physics.mesh.Get_Residual_Row_Scale(step)
        if scale is None:
            return F.norm()
        local = F.getArray(readonly=True) * scale
        return float(np.sqrt(F.getComm().tompi4py().allreduce(float(np.dot(local, local)), op=MPI.SUM)))

    def _has_limited_fields(self, step):
        """PRIVATE: Whether ``_limit_positive_update`` can change updates of ``step``."""
        bounded = set(self.PositiveFields) | set(self.FractionFields) | set(self.MaxFieldUpdate)
        return bool(bounded) and bool(bounded & set(self.physics.mesh.step_fields.get(step, [])))

    def _limit_positive_update(self, du, step):
        """PRIVATE: Take the approach to a bound of ``Positive_Fields`` and ``Fraction_Fields`` in log space, smoothly capped.

        Towards the lower bound 0 (``u > 0``, ``du < 0``; both options), ``du`` becomes
        ``u (exp(-L tanh(r / L)) - 1)`` with ``r = -du / u`` and
        ``L = -ln(Positive_Update_Factor)``: the Newton step for ``|r| << 1``, never
        negative, and at most the factor ``Positive_Update_Factor`` smaller per iteration,
        without a kink (a hard limit makes Newton cycle where a trace concentration is
        repeatedly limited). Towards the upper bound 1 of ``Fraction_Fields``
        (``u < 1``, ``du > 0``) the same applies to ``1 - u``. Before these, the whole
        update is scaled by ``tanh(m) / m``, with ``m`` the largest change of a
        ``Max_Field_Update`` field relative to its cap ``c``: a damped Newton step that keeps
        the direction (capping the potential alone, while the concentrations take the full
        step, mixes inconsistent updates). Only owned entries of the step's block layout are
        modified.
        """
        if not self._has_limited_fields(step):
            return
        bounded = set(self.PositiveFields) | set(self.FractionFields)
        mesh = self.physics.mesh
        field_names = mesh.step_fields.get(step, [])
        limit = -np.log(self.PositiveUpdateFactor)

        def towards_zero(distance, step_towards):
            # step_towards > 0 moves the distance to the bound towards 0; returns the (negative) change of the distance
            moving = (distance > 0.0) & (step_towards > 0.0)
            relative = np.where(moving, step_towards / np.where(moving, distance, 1.0), 0.0)
            return moving, distance * np.expm1(-limit * np.tanh(relative / limit))

        offsets = mesh.get_step_block_offsets(step)
        with du.localForm() as loc_du:
            increment = loc_du.array
            # Damped Newton: scale the whole update by tanh(m) / m, m = the largest change of a
            # Max_Field_Update field relative to its cap (all ranks); this keeps the Newton direction.
            ratio = 0.0
            for field_name, start, end in zip(field_names, offsets, offsets[1:]):
                if field_name in self.MaxFieldUpdate and end > start:
                    ratio = max(ratio, float(np.max(np.abs(increment[start:end]))) / self.MaxFieldUpdate[field_name])
            ratio = comm.allreduce(ratio, op=MPI.MAX)
            if ratio > 0.0:
                increment[:] *= np.tanh(ratio) / ratio
            for field_name, start, end in zip(field_names, offsets, offsets[1:]):
                if field_name not in bounded:
                    continue
                block = increment[start:end].copy()
                u = mesh.fields[field_name].x.array[: end - start]
                moving, change = towards_zero(u, -block)
                limited = np.where(moving, change, block)
                if field_name in self.FractionFields:
                    moving, change = towards_zero(1.0 - u, block)
                    limited = np.where(moving, -change, limited)
                increment[start:end] = limited

    def _update_solution(self, du, step):
        """PRIVATE: Push a linear-solver increment back into the mesh fields."""
        self.physics.mesh.update_step_from_vector(du, step)

    def _update_solution_scaled(self, du, scale, step):
        """PRIVATE: Apply a scaled update and release the temporary vector.
        NOTE: This is needed to avoid memory leaks due to PETSc allocating 
        a temporary vector but never freeing it uf done in a single-line-statement"""
        scaled_du = self._copy_scaled_vector(du, scale)
        try:
            self._update_solution(scaled_du, step)
        finally:
            self._destroy_petsc_object(scaled_du)

    def _copy_scaled_vector(self, vec, scale):
        """PRIVATE: Return an owned vector copy scaled by ``scale``."""
        scaled_vec = vec.copy()
        scaled_vec.scale(scale)
        return scaled_vec

    @staticmethod
    def _destroy_petsc_object(obj) -> None:
        """PRIVATE: Destroy PETSc-like temporary objects when supported."""
        obj.destroy()

    def _print_memory_debug_summary(self):
        """PRIVATE: Print coarse live-object diagnostics after a completed solve."""
        gc.collect()
        rss_mb = self._resident_memory_mb()
        counts = self._tracked_memory_object_counts()

        rss_text = "unavailable" if rss_mb is None else f"{rss_mb:.1f} MB"
        counts_text = ", ".join(f"{name}={count}" for name, count in counts.items())
        mprint(f"\tMemory after completed solve step: RSS={rss_text}; {counts_text}")

    def _resident_memory_mb(self):
        """PRIVATE: Return current resident memory for this process on Linux."""
        try:
            with open("/proc/self/status", encoding="utf-8") as status_file:
                for line in status_file:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            return None
        return None

    def _tracked_memory_object_counts(self):
        """PRIVATE: Count visible Python wrappers for common native objects."""
        targets = (
            ("PETSc.Vec", PETSc.Vec),
            ("PETSc.Mat", PETSc.Mat),
            ("PETSc.KSP", PETSc.KSP),
            ("PETSc.PC", PETSc.PC),
            ("PETSc.IS", PETSc.IS),
            ("fem.Function", fem.Function),
            ("fem.Expression", fem.Expression),
        )
        objects = gc.get_objects()
        counts = {"PythonObjects": len(objects)}
        counts.update({name: 0 for name, _ in targets})
        for obj in objects:
            for name, target_type in targets:
                if isinstance(obj, target_type):
                    counts[name] += 1
        return counts
