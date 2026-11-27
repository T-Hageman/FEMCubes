"""Tests for the public ``TimeSolver`` API."""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from femcubes.core.mesh.refinement import (
    FieldThresholdIndicatorConfig,
    RefinementConfig,
)
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.step_solver import StepSolver
from femcubes.core.utils.runtime_profiler import RuntimeProfiler

time_solver_module = importlib.import_module("femcubes.core.solvers.time_solver")


class _Destroyable:
    def __init__(self):
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1


class _MeshStub:
    def __init__(self):
        self.remesh_epoch = 0
        self.plots = 0
        self.measure_plots = 0
        self.outputs = []

    def Plot_Fields(self):
        self.plots += 1

    def Plot_Measures(self):
        self.measure_plots += 1

    def Write_Outputs(self, step, t):
        self.outputs.append((step, t))


class _PhysicsStub:
    def __init__(self, steps=1):
        self.mesh = _MeshStub()
        self.reset_calls = 0
        self.commit_calls = 0
        self.measure_calls = 0
        self.post_remesh_calls = 0
        self.reset_epochs = []
        self.kf_calls = []
        self.steps = steps

    def Get_Kmat_Fvec(self, step):
        self.kf_calls.append(step)
        return _Destroyable(), _Destroyable()

    def Reset_Step(self):
        self.reset_calls += 1
        self.reset_epochs.append(self.mesh.remesh_epoch)

    def Post_Remesh(self):
        self.post_remesh_calls += 1
        if hasattr(self, "events"):
            self.events.append("post_remesh")

    def commit(self):
        self.commit_calls += 1
        if hasattr(self, "events"):
            self.events.append("commit")

    def Update_Global_Measures(self):
        self.measure_calls += 1


class _PhysicsWarmupStub(_PhysicsStub):
    def __init__(self, steps=1):
        super().__init__(steps=steps)
        self.returned = []

    def Get_Kmat_Fvec(self, step):
        self.kf_calls.append(step)
        pair = (_Destroyable(), _Destroyable())
        self.returned.append(pair)
        return pair


class _TrackingTimeSolver(TimeSolver):
    def __init__(
        self,
        params,
        physics,
        validity_sequence=None,
        reduce_dt_on_invalid=False,
    ):
        super().__init__(params, physics)
        self.solve_calls = 0
        self.validity_sequence = list(validity_sequence or [])
        self.reduce_dt_on_invalid = reduce_dt_on_invalid
        self.warmup_calls = 0
        self.cleanup_calls = 0

    def warmup_jit(self):
        self.warmup_calls += 1

    def Solve_Step(self):
        self.solve_calls += 1
        if hasattr(self.physics, "events"):
            self.physics.events.append("solve")

    def Check_Step_Validity(self):
        if self.validity_sequence:
            valid = self.validity_sequence.pop(0)
            if not valid and self.reduce_dt_on_invalid:
                self.params.dt /= 2.0
            return valid
        return True

    def _cleanup_solver(self):
        self.cleanup_calls += 1
        if hasattr(self.physics, "events"):
            self.physics.events.append("cleanup")
        super()._cleanup_solver()


class _RefiningMeshStub(_MeshStub):
    def __init__(self, outcomes):
        super().__init__()
        self.outcomes = list(outcomes)
        self.refine_calls = 0

    def Refine_From_Indicator(self, config):
        self.refine_calls += 1
        installed = bool(self.outcomes.pop(0)) if self.outcomes else False
        epoch_before = self.remesh_epoch
        if installed:
            self.remesh_epoch += 1
        return SimpleNamespace(
            installed=installed,
            epoch_before=epoch_before,
            epoch_after=self.remesh_epoch,
        )


class _RefiningPhysicsStub(_PhysicsStub):
    def __init__(self, outcomes):
        super().__init__()
        self.mesh = _RefiningMeshStub(outcomes)
        self.events = []


class _Params:
    Solution_Steps = {"u": 0}
    n_passes = 1
    max_it = 1
    rel_tol = 1.0e-6
    abs_tol = 1.0e-8
    line_search = False
    line_search_lims = [0.1, 1.0]
    DEBUG_PRINT_TIMING = False

    start_time = 0.0
    end_time = 1.0
    dt = 0.5
    t = 0.0
    JIT_WARMUP = False

    Global_Measures = {}


def _make_params():
    params = SimpleNamespace()
    for name, value in _Params.__dict__.items():
        if not name.startswith("__"):
            setattr(params, name, value)
    return params


def _refinement_config(**overrides):
    values = {
        "enabled": True,
        "indicator": FieldThresholdIndicatorConfig(
            field="indicator",
            threshold=0.5,
        ),
        "target_h": 0.1,
        "buffer_layers": 0,
    }
    values.update(overrides)
    return RefinementConfig(**values)


class TestTimeSolverPublicApi:
    def test_warmup_jit_calls_each_step_without_destroying_physics_owned_objects(self):
        params = _make_params()
        params.Solution_Steps = {"u": 0, "p": 1}
        params.start_time = 0.0
        params.end_time = 0.0
        params.dt = 1.0
        params.JIT_WARMUP = True
        params.DEBUG_PRINT_TIMING = False
        physics = _PhysicsWarmupStub(steps=2)

        solver = TimeSolver(params, physics)
        solver.warmup_jit()

        assert physics.kf_calls == [0, 1]
        assert all(K.destroy_calls == 0 and F.destroy_calls == 0 for K, F in physics.returned)

    def test_update_measures_forwards_to_physics(self):
        params = _make_params()
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)

        solver.Update_Measures()

        assert physics.measure_calls == 1

    def test_run_accumulates_plotting_and_output_timings(self):
        params = _make_params()
        params.end_time = 0.5
        physics = _PhysicsStub()
        solver = _TrackingTimeSolver(params, physics)
        clock_values = iter((1.0, 1.25, 2.0, 2.5))
        solver._runtime_profiler = RuntimeProfiler(
            enabled=True,
            clock=lambda: next(clock_values),
        )

        solver.run()

        timings = solver.get_runtime_timings()
        assert timings["plotting"] == pytest.approx(0.25)
        assert timings["output"] == pytest.approx(0.5)

    def test_update_time_increments_stepwise(self):
        params = _make_params()
        params.TimeStepControl = {"Type": "StepWise", "time_list": [0.0, 1.0, 3.0], "dt_list": [0.2, 0.5]}
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)
        solver.t = 1.5

        solver.Update_TimeIncrements()

        assert solver.params.dt == 0.5

    def test_update_time_increments_stepwise_uses_last_interval_after_range(self):
        params = _make_params()
        params.TimeStepControl = {"Type": "StepWise", "time_list": [0.0, 1.0, 3.0], "dt_list": [0.2, 0.5]}
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)
        solver.t = 4.0

        solver.Update_TimeIncrements()

        assert solver.params.dt == 0.5

    @pytest.mark.parametrize("policy", ["StepWise", "Measure"])
    def test_schedule_uses_first_entry_before_time_list(self, policy):
        params = _make_params()
        params.dt = 1.0
        params.TimeStepControl = {"Type": policy, "time_list": [0.0, 1.0], "dt_list": [0.2, 0.5],
                                  "Measure": "CrackChange", "Target_Change": 1.0}
        solver = TimeSolver(params, _PhysicsStub())
        solver.t = solver.params.t = -1.0

        solver.Update_TimeIncrements()

        assert solver.params.dt == 0.2

    def test_measure_without_time_list_passes_t_zero(self):
        params = _make_params()
        params.dt = 0.1
        params.TimeStepControl = {"Type": "Measure", "dt_max": 0.3, "Measure": "CrackChange", "Target_Change": 1.0}
        solver = TimeSolver(params, _PhysicsStub())

        solver.Update_TimeIncrements()      # t = 0: no KeyError for the missing time_list

        assert solver.params.dt == pytest.approx(0.12)

    @pytest.mark.parametrize("settings, match", [
        ({"TimeStepControl": {"Type": "Adaptive"}}, "Unknown TimeStepControl Type"),
        ({"Linear_Solver": "CG"}, "Unknown Linear_Solver"),
    ])
    def test_unknown_solver_settings_are_rejected(self, settings, match):
        params = _make_params()
        for name, value in settings.items():
            setattr(params, name, value)
        with pytest.raises(ValueError, match=match):
            TimeSolver(params, _PhysicsStub())

    def test_unknown_measure_is_rejected(self):
        params = _make_params()
        params.TimeStepControl = {"Type": "Measure", "Measure": "Crack_Lenght_Change", "Target_Change": 1.0}
        params.Global_Measures = {"Crack_Length_Change": 0.0}
        solver = TimeSolver(params, _PhysicsStub())
        with pytest.raises(ValueError, match="not a global measure"):
            solver.Check_Step_Validity()

    def test_update_time_increments_stepwise_lands_on_zero(self):
        params = _make_params()
        params.dt = 0.05
        params.TimeStepControl = {"Type": "StepWise", "time_list": [-1.0, 0.0, 1.0e-6], "dt_list": [0.05, 1.0e-6, 10.0]}
        solver = TimeSolver(params, _PhysicsStub())
        solver.t = solver.params.t = -0.02

        solver.Update_TimeIncrements()
        assert solver.params.dt == pytest.approx(0.02)

        solver.t = solver.params.t = 0.0
        solver.Update_TimeIncrements()
        assert solver.params.dt == 1.0e-6

    def test_update_time_increments_measure_crosses_zero(self):
        params = _make_params()
        params.dt = 0.4
        params.TimeStepControl = {"Type": "Measure", "dt_max": 1.0, "time_list": [-1.0, 0.0], "dt_list": [0.2, 0.3]}
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)
        solver.t = -0.2
        solver.params.t = -0.2

        solver.Update_TimeIncrements()

        assert solver.params.dt == 0.2

    def test_update_time_increments_measure_uses_zero_override(self):
        params = _make_params()
        params.dt = 0.2
        params.TimeStepControl = {"Type": "Measure", "dt_max": 1.0, "time_list": [0.0, 1.0], "dt_list": [0.05, 0.2]}
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)
        solver.t = 0.0
        solver.params.t = 0.0

        solver.Update_TimeIncrements()

        assert solver.params.dt == 0.05

    def test_check_step_validity_reduces_dt_when_measure_exceeds_target(self):
        params = _make_params()
        params.dt = 0.4
        params.TimeStepControl = {
            "Type": "Measure",
            "Measure": "CrackChange",
            "Target_Change": 0.1,
            "dt_min": 0.05,
        }
        params.Global_Measures = {"CrackChange": 0.3}
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)

        assert solver.Check_Step_Validity() is False
        assert solver.params.dt == 0.2

    def test_check_step_validity_uses_dt_cut_factor(self):
        params = _make_params()
        params.dt = 0.4
        params.TimeStepControl = {
            "Type": "Measure",
            "Measure": "CrackChange",
            "Target_Change": 0.1,
            "dt_min": 0.01,
            "dt_cut_factor": 10.0,
        }
        params.Global_Measures = {"CrackChange": 0.3}
        solver = TimeSolver(params, _PhysicsStub())

        assert solver.Check_Step_Validity() is False
        assert solver.params.dt == pytest.approx(0.04)

        params.TimeStepControl["dt_cut_factor"] = 1.0
        with pytest.raises(ValueError, match="dt_cut_factor"):
            solver.Check_Step_Validity()

    def test_check_step_validity_accepts_at_minimum_dt(self):
        params = _make_params()
        params.dt = 0.05
        params.TimeStepControl = {
            "Type": "Measure",
            "Measure": "CrackChange",
            "Target_Change": 0.1,
            "dt_min": 0.05,
        }
        params.Global_Measures = {"CrackChange": 1.0}
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)

        assert solver.Check_Step_Validity() is True

    def test_check_step_validity_defaults_to_true_without_policy(self):
        params = _make_params()
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)

        assert solver.Check_Step_Validity() is True

    def test_check_step_validity_accepts_measure_within_target(self, capsys):
        params = _make_params()
        params.TimeStepControl = {
            "Type": "Measure",
            "Measure": "CrackChange",
            "Target_Change": 0.5,
            "dt_min": 0.05,
        }
        params.Global_Measures = {"CrackChange": 0.1}
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)

        assert solver.Check_Step_Validity() is True
        assert "within target" in capsys.readouterr().out

    def test_run_advances_time_commits_and_writes_outputs(self):
        params = _make_params()
        params.start_time = 0.0
        params.end_time = 1.0
        params.dt = 0.5
        physics = _PhysicsStub()
        solver = _TrackingTimeSolver(params, physics)

        solver.run()

        assert solver.t == 1.0
        assert physics.commit_calls == 2
        assert physics.measure_calls == 2
        assert physics.mesh.outputs == [(1, 0.5), (2, 1.0)]

    @pytest.mark.parametrize("dt, end_time, expected_times", [
        (0.4, 1.0, [0.4, 0.8, 1.0]),                    # The last step is shortened to end at end_time
        (0.1, 0.3, [0.1, 0.2, 0.3]),                    # Round-off in t does not add a step of round-off length
    ])
    def test_run_lands_on_end_time(self, dt, end_time, expected_times):
        params = _make_params()
        params.start_time = 0.0
        params.end_time = end_time
        params.dt = dt
        params.TimeStepControl = {"Type": "Measure", "time_list": [0.0], "dt_list": [dt], "dt_min": 1.0e-6,
                                  "dt_max": dt, "Measure": "CrackChange", "Target_Change": 1.0}
        params.Global_Measures = {"CrackChange": 0.0}
        physics = _PhysicsStub()
        solver = _TrackingTimeSolver(params, physics)

        solver.run()

        assert solver.t == end_time
        assert [time for _, time in physics.mesh.outputs] == pytest.approx(expected_times)

    def test_run_retries_invalid_step_and_resets_physics(self):
        params = _make_params()
        params.start_time = 0.0
        params.end_time = 3.0
        params.dt = 1.0
        physics = _PhysicsStub()
        solver = _TrackingTimeSolver(params, physics, validity_sequence=[False, True])

        solver.run()

        assert physics.reset_calls == 1
        assert solver.solve_calls == 4

    def test_remesh_retries_same_increment_without_reset_or_dt_change(self):
        params = _make_params()
        params.end_time = 0.5
        params.dt = 0.5
        params.MeshRefinement = _refinement_config(max_levels_per_step=1)
        physics = _RefiningPhysicsStub([True])
        solver = _TrackingTimeSolver(params, physics)
        workspace_objects = [_Destroyable() for _ in range(3)]
        solver._solver_cache[0] = SimpleNamespace(
            du=workspace_objects[0],
            rhs=workspace_objects[1],
            ksp=workspace_objects[2],
            _fieldsplit_is=None,
        )

        solver.run()

        assert solver.solve_calls == 2
        assert physics.measure_calls == 2
        assert physics.reset_calls == 0
        assert physics.post_remesh_calls == 1
        assert solver.cleanup_calls == 1
        assert physics.commit_calls == 1
        assert physics.events == [
            "solve",
            "post_remesh",
            "cleanup",
            "solve",
            "commit",
        ]
        assert all(obj.destroy_calls == 1 for obj in workspace_objects)
        assert solver._solver_cache == {}
        assert solver.t == pytest.approx(0.5)
        assert params.dt == pytest.approx(0.5)
        assert solver.refinement_event_count == 1
        assert solver.last_retry_reason == "accept"

    def test_run_accumulates_refinement_timing_for_noop_attempt(self):
        params = _make_params()
        params.end_time = 0.5
        params.dt = 0.5
        params.MeshRefinement = _refinement_config()
        physics = _RefiningPhysicsStub([False])
        solver = _TrackingTimeSolver(params, physics)
        clock_values = iter((1.0, 1.4, 2.0, 2.1, 3.0, 3.2))
        solver._runtime_profiler = RuntimeProfiler(
            enabled=True,
            clock=lambda: next(clock_values),
        )

        solver.run()

        timings = solver.get_runtime_timings()
        assert timings["refinement"] == pytest.approx(0.4)
        assert timings["plotting"] == pytest.approx(0.1)
        assert timings["output"] == pytest.approx(0.2)

    @pytest.mark.parametrize("repartitioned", [False, True])
    def test_successful_remesh_reports_topology_size_balance_and_timing(
        self,
        capsys,
        repartitioned,
    ):
        params = _make_params()
        params.MeshRefinement = _refinement_config()
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)
        solver.refinement_event_count = 2
        result = SimpleNamespace(
            preparation_seconds=0.1,
            indicator_seconds=0.2,
            buffer_seconds=0.3,
            marker_transfer_seconds=0.4,
            standard_transfer_seconds=0.6,
            quadrature_transfer_seconds=0.7,
            install_seconds=0.25,
            visualization_seconds=0.05,
            initial_transfer_diagnostics={
                "u": SimpleNamespace(family="CG", space_creation_seconds=0.02, search_seconds=0.15, interpolation_seconds=0.25),
                "v": SimpleNamespace(family="CG", space_creation_seconds=0.03, synchronization_seconds=0.04, diagnostics_seconds=0.06),
                "history": SimpleNamespace(family="Quadrature", space_creation_seconds=0.05, search_seconds=0.1,
                                           reconstruction_seconds=0.2, interpolation_seconds=0.15,
                                           synchronization_seconds=0.05, diagnostics_seconds=0.05),
            },
            report=SimpleNamespace(
                global_marked_cells=4,
                global_cells_before=12,
                global_cells_after=21,
                old_min_h=0.4,
                old_max_h=0.8,
                new_min_h=0.2,
                new_max_h=0.7,
                imbalance_ratio_before_repartition=14.11,
                imbalance_ratio_after=1.02,
                repartitioned=repartitioned,
                repartition_seconds=0.75 if repartitioned else 0.0,
                topology_seconds=0.5,
            ),
        )

        solver._report_successful_refinement(result, refresh_seconds=0.8, total_seconds=5.0)

        output = capsys.readouterr().out
        assert "epoch 2" in output
        assert "indicator indicator (greater_equal 5.000e-01) marked 4 cells" in output
        assert "12 -> 21 (+9)" in output
        assert ("imbalance 14.11 -> 1.02" if repartitioned else "imbalance 1.02") in output
        assert "total 5.000 s" in output
        assert "timings (max across ranks)" in output
        for text in (
            "preparation 0.100 s", "indicator 0.200 s", "buffer 0.300 s",
            "topology 0.500 s", "markers 0.400 s", "CG/DG transfer 0.600 s",
            "quadrature transfer 0.700 s", "install 0.250 s",
            "including visualization 0.050 s", "refresh 0.800 s",
        ):
            assert text in output
        assert ("redistribution 0.750 s" if repartitioned else "redistribution 0.000 s") in output
        standard_line = next(line for line in output.splitlines() if "Initial CG/DG transfer:" in line)
        quadrature_line = next(line for line in output.splitlines() if "Initial quadrature transfer:" in line)
        for text in ("spaces 0.050 s", "search/setup 0.150 s", "interpolation 0.250 s",
                     "synchronization 0.040 s", "diagnostics 0.060 s", "other 0.050 s"):
            assert text in standard_line
        assert "reconstruction" not in standard_line
        for text in ("spaces 0.050 s", "search/setup 0.100 s", "reconstruction 0.200 s",
                     "interpolation 0.150 s", "synchronization 0.050 s", "diagnostics 0.050 s", "other 0.100 s"):
            assert text in quadrature_line
        assert "Re-solving the current time increment" in output

    def test_run_measures_model_refresh_and_total_refinement_time(self, monkeypatch):
        params = _make_params()
        params.end_time = 0.5
        params.MeshRefinement = _refinement_config()
        physics = _RefiningPhysicsStub([True])
        solver = _TrackingTimeSolver(params, physics)
        clock_values = iter((0.0, 1.0, 4.0, 5.0))
        monkeypatch.setattr(time_solver_module, "perf_counter", lambda: next(clock_values))
        reported = []
        monkeypatch.setattr(solver, "_report_successful_refinement", lambda result, **times: reported.append(times))

        solver.run()

        assert reported == [{"refresh_seconds": 3.0, "total_seconds": 5.0}]
        assert physics.post_remesh_calls == 1
        assert solver.solve_calls == 2

    def test_successful_remesh_reporting_accepts_minimal_custom_result(
        self,
        capsys,
    ):
        params = _make_params()
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)
        solver.refinement_event_count = 1

        solver._report_successful_refinement(SimpleNamespace())

        output = capsys.readouterr().out
        assert "epoch 1 installed" in output
        assert "re-solving the current time increment" in output

    def test_max_levels_per_step_terminates_successful_remesh_retries(self):
        params = _make_params()
        params.end_time = 0.5
        params.dt = 0.5
        params.MeshRefinement = _refinement_config(max_levels_per_step=2)
        physics = _RefiningPhysicsStub([True, True, True])
        solver = _TrackingTimeSolver(params, physics)

        solver.run()

        assert physics.mesh.refine_calls == 2
        assert physics.mesh.remesh_epoch == 2
        assert solver.refinement_event_count == 2
        assert solver.solve_calls == 3
        assert physics.post_remesh_calls == 2
        assert physics.commit_calls == 1
        assert physics.reset_calls == 0

    def test_max_events_disables_refinement_on_later_time_steps(self):
        params = _make_params()
        params.end_time = 1.0
        params.dt = 0.5
        params.MeshRefinement = _refinement_config(
            max_levels_per_step=1,
            max_events=1,
        )
        physics = _RefiningPhysicsStub([True, True])
        solver = _TrackingTimeSolver(params, physics)

        solver.run()

        assert physics.mesh.refine_calls == 1
        assert physics.mesh.remesh_epoch == 1
        assert solver.refinement_event_count == 1
        assert solver.solve_calls == 3
        assert physics.post_remesh_calls == 1
        assert physics.commit_calls == 2

    def test_refinement_start_and_interval_schedule_noop_checks(self):
        params = _make_params()
        params.end_time = 2.0
        params.dt = 0.5
        params.MeshRefinement = _refinement_config(
            start_step=1,
            check_interval=2,
        )
        physics = _RefiningPhysicsStub([False, False])
        solver = _TrackingTimeSolver(params, physics)

        solver.run()

        assert physics.mesh.refine_calls == 2
        assert solver.solve_calls == 4
        assert solver.refinement_event_count == 0
        assert physics.post_remesh_calls == 0
        assert physics.reset_calls == 0
        assert physics.commit_calls == 4

    def test_time_rejection_after_remesh_reduces_dt_and_resets_refined_state(self):
        params = _make_params()
        params.end_time = 3.0           # The rejected step (t = 2, dt = 1) is not shortened by the end time
        params.dt = 1.0
        params.MeshRefinement = _refinement_config(
            start_step=2,
            max_levels_per_step=1,
        )
        physics = _RefiningPhysicsStub([True])
        solver = _TrackingTimeSolver(
            params,
            physics,
            validity_sequence=[True, True, True, False, True],   # Every step is checked, also steps 0 and 1
            reduce_dt_on_invalid=True,
        )

        solver.run()

        assert solver.solve_calls == 6
        assert physics.post_remesh_calls == 1
        assert physics.reset_calls == 1
        assert physics.reset_epochs == [1]
        assert physics.commit_calls == 4
        assert params.dt == pytest.approx(0.5)
        assert solver.t == pytest.approx(3.0)

    def test_real_transient_refines_resolves_and_commits_on_new_mesh(
        self,
        linear_elastic_params,
    ):
        params = linear_elastic_params
        params.MeshRefinement = _refinement_config(
            target_h=0.75,
            max_levels_per_step=1,
            max_events=1,
        )
        mesh = Mesh(params)
        mesh.Add_Field("indicator", "Refinement indicator", "DG", 0)
        mesh.Constant_Field("indicator", 1.0)
        physics = Physics(mesh, params)
        physics.Initialize_Fields()
        solver = TimeSolver(params, physics)
        solver.warmup_jit = lambda: None
        solve_calls = []

        def solve_and_track():
            solve_calls.append(mesh.remesh_epoch)
            return StepSolver.Solve_Step(solver)

        solver.Solve_Step = solve_and_track
        solver.run()

        assert solve_calls == [0, 1]
        assert mesh.remesh_epoch == 1
        assert solver.refinement_event_count == 1
        assert solver.t == pytest.approx(params.end_time)
        for field_name, old_name in (
            ("u", "uOld"),
            ("v", "vOld"),
            ("a", "aOld"),
        ):
            assert np.all(np.isfinite(mesh.fields[field_name].x.array))
            np.testing.assert_allclose(
                mesh.fields[field_name].x.array,
                mesh.fields[old_name].x.array,
            )

    def test_real_time_retry_resets_transferred_history_after_remesh(
        self,
        linear_elastic_params,
        capsys,
    ):
        params = linear_elastic_params
        params.end_time = 0.125
        params.dt = 0.25
        params.TimeStepControl = {"Type": "None", "dt_min": 0.01}
        params.MeshRefinement = _refinement_config(
            target_h=0.75,
            max_levels_per_step=1,
            max_events=1,
        )
        mesh = Mesh(params)
        mesh.Add_Field("indicator", "Refinement indicator", "DG", 0)
        mesh.Constant_Field("indicator", 1.0)
        physics = Physics(mesh, params)
        physics.Initialize_Fields()
        original_reset = physics.Reset_Step
        reset_epochs = []

        def reset_and_track():
            reset_epochs.append(mesh.remesh_epoch)
            original_reset()

        physics.Reset_Step = reset_and_track
        solver = TimeSolver(params, physics)
        solver.warmup_jit = lambda: None
        solve_epochs = []
        reset_was_visible_to_retry = []

        def solve_with_one_post_remesh_failure():
            solve_epochs.append(mesh.remesh_epoch)
            if len(solve_epochs) == 3:
                reset_was_visible_to_retry.append(
                    np.allclose(
                        mesh.fields["u"].x.array,
                        mesh.fields["uOld"].x.array,
                    )
                )
            StepSolver.Solve_Step(solver)
            if len(solve_epochs) == 2:
                mesh.fields["u"].x.array[:] = 123.0
                raise RuntimeError("synthetic post-remesh time-step failure")

        solver.Solve_Step = solve_with_one_post_remesh_failure
        solver.run()

        assert solve_epochs[:3] == [0, 1, 1]
        assert all(epoch == 1 for epoch in solve_epochs[3:])
        assert reset_epochs == [1]
        assert reset_was_visible_to_retry == [True]
        assert 0.0 < params.dt < 0.25
        assert solver.t == pytest.approx(params.end_time)
        np.testing.assert_allclose(
            mesh.fields["u"].x.array,
            mesh.fields["uOld"].x.array,
        )
        assert "synthetic post-remesh time-step failure" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "options, expected_dts",
        [
            ({}, [0.25, 0.0625]),
            ({"reset_dt_factor": 4.0, "reset_dt_factor_repeated": 10.0}, [0.25, 0.025]),
        ],
    )
    def test_run_reduces_dt_by_repeated_factor_after_consecutive_failures(self, options, expected_dts, capsys):
        params = _make_params()
        params.start_time = 0.0
        params.end_time = 1.0
        params.dt = 1.0
        params.TimeStepControl = {"Type": "None", "dt_min": 1.0e-3}
        for name, value in options.items():
            setattr(params, name, value)
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)
        solver.warmup_jit = lambda: None
        attempts = []

        def _fail_twice_then_succeed():
            attempts.append(params.dt)
            if len(attempts) <= 2:
                raise RuntimeError("solve failed")

        solver.Solve_Step = _fail_twice_then_succeed

        solver.run()

        # Two failed attempts, then the reduced dt is kept until the end time
        assert attempts[:3] == [1.0] + expected_dts
        assert physics.reset_calls == 2
        assert physics.commit_calls == len(attempts) - 2
        assert solver.t == 1.0
        assert "Resetting step and reducing dt" in capsys.readouterr().out

    def test_rejects_invalid_reset_factors(self):
        params = _make_params()
        params.reset_dt_factor_repeated = 1.0

        with pytest.raises(ValueError, match="reset_dt_factor_repeated"):
            TimeSolver(params, _PhysicsStub())

    def test_run_aborts_when_solve_fails_at_minimum_dt(self, monkeypatch, capsys):
        params = _make_params()
        params.start_time = 0.0
        params.end_time = 1.0
        params.dt = 0.25
        params.TimeStepControl = {"Type": "None", "dt_min": 0.25}
        physics = _PhysicsStub()
        solver = TimeSolver(params, physics)
        solver.warmup_jit = lambda: None

        def _raise_solve_error():
            raise RuntimeError("solve failed")

        class _AbortComm:
            abort_code = None

            def allreduce(self, value, op=None):
                return value

            def Abort(self, code):
                self.abort_code = code
                raise RuntimeError(f"Abort({code})")

        abort_comm = _AbortComm()
        solver.Solve_Step = _raise_solve_error
        monkeypatch.setattr(time_solver_module, "_mpi_comm", abort_comm)

        with pytest.raises(RuntimeError, match=r"Abort\(1\)"):
            solver.run()

        output = capsys.readouterr().out
        assert abort_comm.abort_code == 1
        assert "Time step already at minimum" in output
        assert physics.reset_calls == 0
