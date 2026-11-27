"""Transient time-marching solver built on top of ``StepSolver``."""

from time import perf_counter

import numpy as np

from femcubes.core.physics.physics import Physics
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.mesh.refinement.config import normalize_refinement_config
from femcubes.core.solvers.step_solver import StepSolver
from mpi4py import MPI as _MPI
from femcubes.core.utils.mpi_utils import mprint, comm as _mpi_comm

class TimeSolver(StepSolver):
    """Time-marching driver for transient simulations.

    Inherits `StepSolver` (per-step nonlinear solves)
    and adds a time loop with adaptive time-step control, measure
    tracking, and output scheduling.

    Parameters
    ----------
    params : ParamsTemplate
        Must define ``start_time``, ``end_time``, ``dt``, and optionally
        ``TimeStepControl``, ``MeshRefinement``, and ``JIT_WARMUP``.
    physics : Physics
        Assembled physics object (models already initialised).

    Public API
    ----------
    ``warmup_jit()``
        Precompile forms before entering the time loop.
    ``run()``
        Execute the full time-stepping solve.
    ``Update_Measures()``
        Refresh global measures after a converged step solve.
    ``Update_TimeIncrements()``
        Update ``params.dt`` from the configured time-step policy.
    ``Check_Step_Validity()``
        Accept or reject the current step based on the configured policy.

    Notes
    -----
    ``TimeSolver`` owns the simulation clock.  It repeatedly calls
    :meth:`Solve_Step`, updates global measures, applies optional
    adaptive time-step control and mesh refinement, and commits the converged
    state. A successful remesh re-solves the same time increment without
    changing ``t`` or ``dt``.
    """

    def __init__(self, params, physics) -> None:
        """Create a transient solver for one mesh / physics configuration."""
        super().__init__(params, physics)
        
        self.t: float = params.start_time
        self.params.t = self.t
        tsc = getattr(params, "TimeStepControl", {"Type": "None"})
        if tsc.get("Type") not in ("None", "StepWise", "Measure"):
            raise ValueError(f"Unknown TimeStepControl Type {tsc.get('Type')!r}; choose from 'None', 'StepWise' and "
                             "'Measure'.")

        self.refinement_config = normalize_refinement_config(
            getattr(params, "MeshRefinement", None)
        )
        mesh_comm = getattr(getattr(physics.mesh, "mesh", None), "comm", None)
        self._refinement_comm = _mpi_comm if mesh_comm is None else mesh_comm
        self._validate_collective_refinement_state()
        self.refinement_event_count = int(
            getattr(physics.mesh, "remesh_epoch", 0)
        )
        self.last_retry_reason = "accept"
        
        # JIT warmup option
        self.jit_warmup = getattr(params, 'JIT_WARMUP', True)

        # Time-step reduction after a failed Solve_Step: ``reset_dt_factor``
        # (default 4) at the first failure of a step, ``reset_dt_factor_repeated``
        # (default: the same) at every further consecutive failure of that step.
        # A larger repeated factor shortens the search for a stable increment
        # when the first failure already shows the step to be far too large.
        self.reset_dt_factor = float(getattr(params, "reset_dt_factor", 4.0))
        self.reset_dt_factor_repeated = float(getattr(params, "reset_dt_factor_repeated", self.reset_dt_factor))
        if not np.isfinite(self.reset_dt_factor) or self.reset_dt_factor <= 1.0:
            raise ValueError("reset_dt_factor must be a finite number greater than one.")
        if not np.isfinite(self.reset_dt_factor_repeated) or self.reset_dt_factor_repeated <= 1.0:
            raise ValueError("reset_dt_factor_repeated must be a finite number greater than one.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warmup_jit(self) -> None:
        """Pre-assemble each staggered step once to warm up JIT compilation.

        This is primarily a convenience for interactive or timing-
        sensitive runs, so the first real step does not include form
        compilation cost. The assembled PETSc objects remain owned by
        ``Physics`` so they can be reused safely by the normal assembly
        path.
        """
        if self.jit_warmup:
            mprint("Performing JIT warmup (precompiling forms)...")
            for step in range(self.n_Staggered_Steps):
                self.physics.Get_Kmat_Fvec(step)
            mprint("JIT warmup complete.")
    
    def run(self) -> None:
        """Run the transient solve, including same-increment remesh retries."""
        self.warmup_jit()
        
        Step: int = 0
        while self.t < self.params.end_time:
            self.Update_TimeIncrements()
            mprint(f"Step {Step}, time: {self.t:.4e} s, dt: {self.params.dt:.4e} s")
            
            # Solve for current Step
            tsc = getattr(self.params, "TimeStepControl", {"Type": "None"})
            dt_min = tsc.get("dt_min", 1.0e-8) if isinstance(tsc, dict) else 1.0e-8
            refinement_levels = 0
            consecutive_resets = 0
            while True:
                error_local = 0
                try:
                    self.Solve_Step()
                except Exception:
                    import traceback
                    from femcubes.core.utils.mpi_utils import rank as _rank
                    if _rank == 0:
                        print(f"  ERROR (rank {_rank}): Solve_Step raised an exception at step {Step}:")
                        print(traceback.format_exc())
                    error_local = 1

                error_any = bool(_mpi_comm.allreduce(error_local, op=_MPI.MAX))
                if error_any:
                    factor = self.reset_dt_factor if consecutive_resets == 0 else self.reset_dt_factor_repeated
                    consecutive_resets += 1
                    new_dt = max(self.params.dt / factor, dt_min)
                    if new_dt <= dt_min and self.params.dt <= dt_min:
                        mprint(f"  Time step already at minimum ({dt_min:.4e} s). "
                               f"Cannot reduce further — aborting simulation.")
                        _mpi_comm.Abort(1)
                    mprint(f"  Resetting step and reducing dt: "
                           f"{self.params.dt:.4e} → {new_dt:.4e} s")
                    self.params.dt = new_dt
                    self.last_retry_reason = "time_step"
                    self.physics.Reset_Step()
                    continue

                consecutive_resets = 0
                self.Update_Measures()
                step_valid = self.Check_Step_Validity()
                retry_reason = "accept" if step_valid else "time_step"
                self._require_collective_retry_reason(retry_reason)
                if not step_valid:
                    self.physics.Reset_Step()
                    continue

                if self._should_attempt_refinement(Step, refinement_levels):
                    refinement_start = perf_counter()
                    with self._runtime_profiler.measure("refinement"):
                        result = self.physics.mesh.Refine_From_Indicator(
                            self.refinement_config
                        )
                        installed = self._record_collective_refinement_result(result)
                        if installed:
                            self._require_collective_retry_reason("remesh")
                            refinement_levels += 1
                            refresh_start = perf_counter()
                            self._refresh_after_remesh()
                            refresh_seconds = perf_counter() - refresh_start
                            self._report_successful_refinement(
                                result,
                                refresh_seconds=refresh_seconds,
                                total_seconds=perf_counter() - refinement_start,
                            )
                    self._runtime_profiler.reclassify(
                        getattr(result, "visualization_seconds", 0.0),
                        source="refinement",
                        target="plotting",
                    )
                    if installed:
                        continue

                self._require_collective_retry_reason("accept")
                break
            
            # Commit Step
            self.physics.commit()
            
            self.t += self.params.dt
            if abs(self.t - self.params.end_time) <= 1.0e-9 * self.params.dt:
                self.t = self.params.end_time   # no extra step of round-off length
            self.params.t = self.t
            Step += 1
            
            # Outputs
            with self._runtime_profiler.measure("plotting"):
                self.physics.mesh.Plot_Fields()
                self.physics.mesh.Plot_Measures()
            with self._runtime_profiler.measure("output"):
                self.physics.mesh.Write_Outputs(Step, self.t)
        
    def Update_Measures(self) -> None:
        """Update global measures through the assembled physics object."""
        self.physics.Update_Global_Measures()
    
    def Update_TimeIncrements(self) -> None:
        """Update the time increment according to the configured control policy.

        Supported policies are currently:

        ``"None"``
            Leave ``params.dt`` unchanged.
        ``"StepWise"``
            Grow ``dt`` by ``dt_grow_factor`` per step towards the largest
            step of a piecewise-constant table (``time_list``, ``dt_list``);
            a smaller table value applies at once.
        ``"Measure"``
            Grow or shrink ``dt`` based on a monitored global measure: a
            step whose measure exceeds ``Target_Change`` is repeated with
            ``dt`` divided by ``dt_cut_factor`` (default 2, down to
            ``dt_min``); otherwise ``dt`` grows by ``dt_grow_factor``
            (default 1.2) per step up to the scheduled maximum.

        With every policy, a step that would pass ``end_time`` is shortened
        to end there.
        """
        if hasattr(self.params, "TimeStepControl"):
            tsc = self.params.TimeStepControl
        else:
            tsc = {"Type": "None"}
            
        if tsc["Type"] == "None":
            self.params.dt = self.params.dt
        if tsc["Type"] == "StepWise":
            dt_target = self._scheduled_dt(tsc["time_list"], tsc["dt_list"])
            # Grow toward the table target rather than jumping straight to it.
            # If dt was reduced after a failed step it recovers at dt_grow_factor
            # per step; if it was never reduced the first multiply already
            # exceeds dt_target and the min() clamps it — same as before.
            # If the table target decreases at a boundary (e.g. new phase uses
            # a smaller dt) we honour that immediately by also applying the min.
            dt_grow_factor = tsc.get("dt_grow_factor", 1.2)
            self.params.dt = min(self.params.dt * dt_grow_factor, dt_target)
            self._land_on_zero()
        if tsc["Type"] == "Measure":
            dt_old = self.params.dt
            
            
            time_list = tsc.get("time_list", None)
            if time_list is None:
                dt_target = tsc.get("dt_max", 600.0)
            else:
                dt_target = self._scheduled_dt(time_list, tsc["dt_list"])
            
            dt_grow_factor = tsc.get("dt_grow_factor", 1.2)
            self.params.dt = min(dt_old * dt_grow_factor, dt_target)  # increase time step if below target
            
            
            self._land_on_zero()
            if self.params.t == 0.0 and time_list is not None:
                # check time_list for t=0, and then set dt, otherwise don't touch dt
                dt_list = tsc["dt_list"]
                for i in range(len(time_list)):
                    if time_list[i] == 0.0:
                        self.params.dt = dt_list[i]
                        break
        self._land_on_end()

    def _scheduled_dt(self, time_list, dt_list) -> float:
        """PRIVATE: The largest step of the schedule at the current time: dt_list[i] from time_list[i] on, the
        last entry of dt_list beyond its length (time_list may also list the end of the last interval), and
        dt_list[0] before time_list[0]."""
        for i in range(len(time_list) - 1, -1, -1):
            if self.t >= time_list[i]:
                return dt_list[min(i, len(dt_list) - 1)]
        return dt_list[0]

    def _land_on_end(self) -> None:
        """PRIVATE: shorten the step that would pass ``end_time`` so that the run ends at ``end_time`` exactly
        (a step that would stop just short of it, by less than 1e-9 of the step, is also landed on it)."""
        remaining = self.params.end_time - self.params.t
        if 0.0 < remaining <= self.params.dt * (1.0 + 1.0e-9):
            self.params.dt = remaining

    def _land_on_zero(self) -> None:
        """PRIVATE: shorten a step from t < 0 that would cross t = 0, so that the step ends at t = 0 exactly
        (the end of an initialisation period with negative time)."""
        if self.params.t < 0.0 and self.params.t + self.params.dt > 0.0:
            self.params.dt = -self.params.t

    def Check_Step_Validity(self) -> bool:
        """Check whether the current step should be accepted or retried.

        Returns
        -------
        bool
            ``True`` if the step is accepted, ``False`` if the caller
            should reset the fields and retry with a smaller ``dt``.
        """
        if hasattr(self.params, "TimeStepControl"):
            tsc = self.params.TimeStepControl
        else:
            tsc = {"Type": "None"}
        
        if tsc["Type"] == "Measure":
            measure_name = tsc["Measure"]
            target_change = tsc["Target_Change"]
            if measure_name not in self.params.Global_Measures:
                raise ValueError(f"TimeStepControl Measure {measure_name!r} is not a global measure of the models; "
                                 f"available: {sorted(self.params.Global_Measures)}.")
            if self.params.dt <= 1.05*tsc.get("dt_min", 1.0e-8):
                mprint(f"  Time step {self.params.dt:.4e} s is at or below minimum. Accepting step to avoid excessively small time steps.")
                return True  # Already at minimum time step
            measure_value = self.params.Global_Measures[measure_name]
            if measure_value > target_change:
                dt_cut_factor = tsc.get("dt_cut_factor", 2.0)
                if not dt_cut_factor > 1.0:
                    raise ValueError("TimeStepControl dt_cut_factor must be larger than 1.")
                self.params.dt = max(self.params.dt / dt_cut_factor, tsc.get("dt_min", 1.0e-8))
                mprint(f"  Step invalid due to {measure_name} = {measure_value:.4e} exceeding target {target_change:.4e}")
                mprint(f"  Reducing time step to {self.params.dt:.4e} s")
                return False
            else:
                mprint(f"  Step valid with {measure_name} = {measure_value:.4e} within target {target_change:.4e}")
        
        return True

    # ------------------------------------------------------------------
    # Internal Helpers: Mesh Refinement Retry Lifecycle
    # ------------------------------------------------------------------

    def _validate_collective_refinement_state(self) -> None:
        """PRIVATE: Require one schedule and starting epoch on all mesh ranks."""

        epoch = getattr(self.physics.mesh, "remesh_epoch", 0)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("physics.mesh.remesh_epoch must be a nonnegative integer.")
        states = self._refinement_comm.allgather((self.refinement_config, epoch))
        if any(state != states[0] for state in states):
            raise ValueError(
                "Mesh-refinement configuration and starting epoch must be "
                "identical on every mesh rank."
            )

    def _should_attempt_refinement(self, step: int, levels_this_step: int) -> bool:
        """PRIVATE: Apply start, interval, per-step, and global event limits."""

        config = self.refinement_config
        if not config.enabled or step < config.start_step:
            return False
        if (step - config.start_step) % config.check_interval != 0:
            return False
        if levels_this_step >= config.max_levels_per_step:
            return False
        if (
            config.max_events is not None
            and self.refinement_event_count >= config.max_events
        ):
            return False
        return True

    def _record_collective_refinement_result(self, result) -> bool:
        """PRIVATE: Validate one manager outcome and advance the event count."""

        required = ("installed", "epoch_before", "epoch_after")
        missing = [name for name in required if not hasattr(result, name)]
        if missing:
            raise TypeError(
                "Refinement result is missing attribute(s): " + ", ".join(missing)
            )
        if not isinstance(result.installed, bool):
            raise TypeError("Refinement result installed must be a bool.")

        local_outcome = (
            result.installed,
            result.epoch_before,
            result.epoch_after,
        )
        outcomes = self._refinement_comm.allgather(local_outcome)
        if any(outcome != outcomes[0] for outcome in outcomes):
            raise RuntimeError(
                "Mesh ranks disagree on whether refinement occurred or on its epoch."
            )

        installed, epoch_before, epoch_after = outcomes[0]
        for name, value in (
            ("epoch_before", epoch_before),
            ("epoch_after", epoch_after),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Refinement result {name} must be a nonnegative integer.")
        if epoch_before != self.refinement_event_count:
            raise RuntimeError(
                "Refinement result starts from epoch "
                f"{epoch_before}, expected {self.refinement_event_count}."
            )
        expected_after = epoch_before + int(installed)
        if epoch_after != expected_after:
            raise RuntimeError(
                f"Refinement result ended at epoch {epoch_after}, expected "
                f"{expected_after}."
            )

        self.refinement_event_count = epoch_after
        return installed

    def _require_collective_retry_reason(self, reason: str) -> None:
        """PRIVATE: Reject divergent accept/time-step/remesh control flow."""

        reasons = self._refinement_comm.allgather(reason)
        if any(candidate != reasons[0] for candidate in reasons):
            raise RuntimeError(
                f"Mesh ranks disagree on solver retry reason: {reasons}."
            )
        self.last_retry_reason = reasons[0]

    def _refresh_after_remesh(self) -> None:
        """PRIVATE: Rebuild model state and destroy stale solver workspaces."""

        local_errors = []
        try:
            self.physics.Post_Remesh()
        except Exception as exc:
            local_errors.append(f"Post_Remesh: {type(exc).__name__}: {exc}")
        try:
            self._cleanup_solver()
        except Exception as exc:  # pragma: no cover - defensive PETSc cleanup guard
            local_errors.append(f"solver cleanup: {type(exc).__name__}: {exc}")

        local_error = "; ".join(local_errors) if local_errors else None
        errors = self._refinement_comm.allgather(local_error)
        if any(error is not None for error in errors):
            details = "; ".join(
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            )
            raise RuntimeError(f"Post-remesh refresh failed: {details}")

    def _report_successful_refinement(
        self,
        result,
        *,
        refresh_seconds: float = 0.0,
        total_seconds: float | None = None,
    ) -> None:
        """PRIVATE: Collectively report topology and maximum rank wall times."""

        report = getattr(result, "report", None)
        required = (
            "global_marked_cells",
            "global_cells_before",
            "global_cells_after",
            "old_min_h",
            "old_max_h",
            "new_min_h",
            "new_max_h",
            "imbalance_ratio_after",
            "topology_seconds",
        )
        if report is None or any(not hasattr(report, name) for name in required):
            mprint(
                f"  Mesh refinement epoch {self.refinement_event_count} "
                "installed; re-solving the current time increment."
            )
            return

        timings = {
            "preparation": float(getattr(result, "preparation_seconds", 0.0)),
            "indicator": float(getattr(result, "indicator_seconds", 0.0)),
            "buffer": float(getattr(result, "buffer_seconds", 0.0)),
            "topology": float(report.topology_seconds),
            "markers": float(getattr(result, "marker_transfer_seconds", 0.0)),
            "CG/DG transfer": float(getattr(result, "standard_transfer_seconds", 0.0)),
            "quadrature transfer": float(getattr(result, "quadrature_transfer_seconds", 0.0)),
            "redistribution": float(getattr(report, "repartition_seconds", 0.0)),
            "install": float(getattr(result, "install_seconds", 0.0)),
            "visualization": float(getattr(result, "visualization_seconds", 0.0)),
            "refresh": float(refresh_seconds),
            "total": float(getattr(result, "total_seconds", 0.0) if total_seconds is None else total_seconds),
        }
        transfer_phases = {
            "spaces": "space_creation_seconds",
            "search/setup": "search_seconds",
            "reconstruction": "reconstruction_seconds",
            "interpolation": "interpolation_seconds",
            "copy": "copy_seconds",
            "synchronization": "synchronization_seconds",
            "diagnostics": "diagnostics_seconds",
        }
        initial_diagnostics = getattr(result, "initial_transfer_diagnostics", {})
        for group, families in (("CG/DG", {"CG", "DG"}), ("quadrature", {"Quadrature"})):
            diagnostics = [diagnostic for diagnostic in initial_diagnostics.values() if diagnostic.family in families]
            for phase, attribute in transfer_phases.items():
                timings[f"{group} {phase}"] = sum(float(getattr(diagnostic, attribute, 0.0)) for diagnostic in diagnostics)
            measured = sum(timings[f"{group} {phase}"] for phase in transfer_phases)
            timings[f"{group} other"] = max(0.0, timings[f"{group} transfer"] - measured)
        local_seconds = np.array(list(timings.values()), dtype=np.float64)
        maximum_seconds = np.empty_like(local_seconds)
        self._refinement_comm.Allreduce(local_seconds, maximum_seconds, op=_MPI.MAX)
        timings = dict(zip(timings, maximum_seconds))
        added_cells = report.global_cells_after - report.global_cells_before
        indicator = self.refinement_config.indicator
        indicator_text = "marked "
        if indicator is not None:
            indicator_text = (
                f"indicator {indicator.field} ({indicator.comparison} "
                f"{indicator.threshold:.3e}) marked "
            )
        imbalance_before = float(
            getattr(
                report,
                "imbalance_ratio_before_repartition",
                report.imbalance_ratio_after,
            )
        )
        repartitioned = bool(getattr(report, "repartitioned", False))
        imbalance_text = f"{report.imbalance_ratio_after:.2f}"
        if repartitioned:
            imbalance_text = (
                f"{imbalance_before:.2f} -> {report.imbalance_ratio_after:.2f}"
            )
        mprint(
            f"  Mesh refinement epoch {self.refinement_event_count}: "
            f"{indicator_text}{report.global_marked_cells} cells; global cells "
            f"{report.global_cells_before} -> {report.global_cells_after} "
            f"(+{added_cells}); h [{report.old_min_h:.3e}, "
            f"{report.old_max_h:.3e}] -> [{report.new_min_h:.3e}, "
            f"{report.new_max_h:.3e}]; owned-cell imbalance "
            f"{imbalance_text}; total {timings['total']:.3f} s."
        )
        stage_text = ", ".join(
            f"{name} {timings[name]:.3f} s"
            for name in (
                "preparation", "indicator", "buffer", "topology", "markers",
                "CG/DG transfer", "quadrature transfer", "redistribution",
            )
        )
        mprint(
            f"  Refinement timings (max across ranks): {stage_text}, "
            f"install {timings['install']:.3f} s "
            f"(including visualization {timings['visualization']:.3f} s), "
            f"refresh {timings['refresh']:.3f} s."
        )
        for group in ("CG/DG", "quadrature"):
            details = ", ".join(
                f"{phase} {timings[f'{group} {phase}']:.3f} s"
                for phase in (*transfer_phases, "other")
                if group == "quadrature" or phase != "reconstruction"
            )
            mprint(f"    Initial {group} transfer: {details}.")
        mprint("  Re-solving the current time increment on the refined mesh.")
