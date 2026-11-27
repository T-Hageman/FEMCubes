"""Tests for the public ``StepSolver`` API."""

import builtins
import importlib
import io
import re
from types import SimpleNamespace

import numpy as np
import pytest
from petsc4py import PETSc

from femcubes.core.base_model import BaseModel
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.step_solver import StepSolver
from femcubes.core.solvers._lin_solvers import SolverWorkspace
from femcubes.core.utils.mpi_utils import comm

step_solver_module = importlib.import_module("femcubes.core.solvers.step_solver")
step_solver_base_module = importlib.import_module("femcubes.core.solvers._step_solver_base")


class _DummyVec:
    def __init__(self, value):
        self.value = float(value)

    def dot(self, other):
        return self.value * other.value

    def norm(self):
        return abs(self.value)

    def __mul__(self, scalar):
        return _DummyVec(self.value * scalar)

    __rmul__ = __mul__

    def copy(self):
        return _DummyVec(self.value)

    def duplicate(self):
        return _DummyVec(0.0)

    def scale(self, alpha):
        self.value *= alpha

    def axpy(self, alpha, other):
        self.value += alpha * other.value

    def destroy(self):
        pass


class _DummyMesh:
    def __init__(self):
        self.plot_calls = 0

    def Get_Residual_Row_Scale(self, step):
        return None

    def update_step_from_vector(self, du, step):
        value = None if du is None else du.value
        self.last_update = (value, step)

    def Plot_Fields(self):
        self.plot_calls += 1


class _DummyPhysics:
    def __init__(self):
        self.mesh = _DummyMesh()

    def Get_Kmat_Fvec(self, step):
        return _ScalarMatrix(), _DummyVec(-4.0)

    def Get_Fvec(self, step):
        return _DummyVec(-4.0)

    def get_fields_for_step(self, step):
        return [("u", object())]


class _Params:
    Solution_Steps = {"u": 0}
    n_passes = 1
    max_it = 1
    rel_tol = 1.0e-6
    abs_tol = 1.0e-8
    line_search = True
    line_search_lims = [0.25, 0.75]
    DEBUG_PRINT_TIMING = False


class _ResidualVec:
    def __init__(self, value):
        self.value = float(value)

    def duplicate(self):
        return _ResidualVec(0.0)

    def pointwiseMult(self, left, right):
        if right is None:
            self.value = abs(left.value)
        else:
            self.value = abs(left.value * right.value)

    def pointwiseDivide(self, left, right):
        self.value = left.value / right.value

    def abs(self):
        self.value = abs(self.value)

    def sum(self):
        return self.value

    def norm(self):
        return abs(self.value)

    def destroy(self):
        pass

    def copy(self):
        return _ResidualVec(self.value)

    def scale(self, alpha):
        self.value *= alpha

    def axpy(self, alpha, other):
        self.value += alpha * other.value

    def dot(self, other):
        return self.value * other.value

    def __mul__(self, scalar):
        return _ResidualVec(self.value * scalar)

    __rmul__ = __mul__

    def localForm(self):
        vector = self

        class _LocalForm:
            def __enter__(self):
                self.array = np.array([vector.value])
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                vector.value = self.array[0]

        return _LocalForm()


class _ScalarMatrix:
    def __init__(self, value=1.0):
        self.value = float(value)

    def getDiagonal(self, diagonal):
        diagonal.value = self.value

    def mult(self, vector, result):
        result.value = self.value * vector.value


class _SequencePhysics:
    def __init__(self, steps, residuals):
        self.mesh = _DummyMesh()
        self._steps = steps
        self._residuals = list(residuals)
        self.kf_assemblies = 0
        self.f_assemblies = 0

    def get_fields_for_step(self, step):
        return [("field", object())]

    def Get_Kmat_Fvec(self, step):
        self.kf_assemblies += 1
        value = self._residuals.pop(0)
        return _ScalarMatrix(), _ResidualVec(value)

    def Get_Fvec(self, step):
        self.f_assemblies += 1
        return _ResidualVec(self._residuals.pop(0))


class _SequenceParams:
    Solution_Steps = {"u": 0, "p": 1}
    n_passes = 2
    max_it = 1
    rel_tol = 1.0e-6
    abs_tol = 1.0e-8
    line_search = False
    line_search_lims = [0.25, 0.75]
    DEBUG_PRINT_TIMING = True


class TestStepSolverPublicApi:
    def test_defaults_each_step_to_newton_without_mutating_params(self):
        configured = {0: "LBFGS"}

        class _TwoStepParams(_SequenceParams):
            Step_Solvers = configured

        solver = StepSolver(_TwoStepParams(), _DummyPhysics())

        assert solver.SolverMethod == {0: "LBFGS", 1: "Newton"}
        assert configured == {0: "LBFGS"}

    @pytest.mark.parametrize(
        ("configured", "error", "message"),
        [
            (["Newton"], TypeError, "dictionary"),
            ({0: "BFGS"}, ValueError, "Invalid solver method"),
            ({1: "Newton"}, ValueError, "unknown step"),
        ],
    )
    def test_rejects_invalid_step_solver_configuration(
        self, configured, error, message
    ):
        class _ConfiguredParams(_Params):
            Step_Solvers = configured

        with pytest.raises(error, match=message):
            StepSolver(_ConfiguredParams(), _DummyPhysics())

    def test_solve_step_updates_linear_elastic_fields(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        solver = StepSolver(linear_elastic_params, physics)

        solver.Solve_Step()

        assert not np.allclose(mesh.fields["u"].x.array, 0.0)

    def test_solve_step_lbfgs_updates_linear_elastic_fields(
        self, linear_elastic_params
    ):
        linear_elastic_params.Step_Solvers = {0: "LBFGS"}
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        solver = StepSolver(linear_elastic_params, physics)

        solver.Solve_Step()

        assert not np.allclose(mesh.fields["u"].x.array, 0.0)

    def test_solve_step_updates_stokes_mixed_fields(self, stokes_params):
        mesh = Mesh(stokes_params)
        physics = Physics(mesh, stokes_params)
        physics.Initialize_Fields()
        solver = StepSolver(stokes_params, physics)

        solver.Solve_Step()

        assert not np.allclose(mesh.fields["p"].x.array, 0.0)

    def test_perform_line_search_backtracks_until_merit_decreases(self):
        physics = _SequencePhysics(steps=1, residuals=[4.0, 0.5])
        solver = StepSolver(_Params(), physics)
        updates = []
        solver._update_solution = lambda du, step: updates.append((du.value, step))

        alpha = solver.Perform_LineSearch(_ResidualVec(1.0), _ResidualVec(2.0), step=3)

        assert alpha == 0.5
        assert updates == [(2.0, 3), (-1.0, 3)]

    def test_perform_line_search_destroys_scaled_update_vectors(self):
        physics = _SequencePhysics(steps=1, residuals=[4.0, 0.5])
        solver = StepSolver(_Params(), physics)
        updates = []
        destroyed_values = []

        class _TrackedVec:
            def __init__(self, value):
                self.value = float(value)

            def copy(self):
                return _TrackedVec(self.value)

            def scale(self, scale):
                self.value *= scale

            def destroy(self):
                destroyed_values.append(self.value)

        solver._update_solution = lambda du, step: updates.append((du.value, step))

        alpha = solver.Perform_LineSearch(_ResidualVec(1.0), _TrackedVec(2.0), step=3)

        assert alpha == 0.5
        assert updates == [(2.0, 3), (-1.0, 3)]
        assert destroyed_values == [2.0, -1.0]

    def test_perform_line_search_accepts_relative_decrease_at_small_scale(self):
        physics = _SequencePhysics(steps=1, residuals=[1.0e-7])
        solver = StepSolver(_Params(), physics)
        updates = []
        solver._update_solution = lambda du, step: updates.append((du.value, step))

        alpha = solver.Perform_LineSearch(
            _ResidualVec(1.0e-6), _ResidualVec(2.0), step=0
        )

        assert alpha == 1.0
        assert updates == [(2.0, 0)]

    def test_lbfgs_line_search_uses_secant_root_with_positive_curvature(self):
        physics = _SequencePhysics(steps=1, residuals=[-2.0, 0.0])
        solver = StepSolver(_Params(), physics)
        solver.LineSearch_Lims = [0.25, 1.0]
        updates = []
        solver._update_solution = lambda du, step: updates.append(
            (du.value, step)
        )

        alpha = solver.Perform_LBFGS_LineSearch(
            _ResidualVec(2.0), _ResidualVec(-1.0), step=3
        )

        assert alpha == pytest.approx(0.5)
        assert updates == [(-1.0, 3), (0.5, 3)]

    def test_lbfgs_line_search_accepts_tighter_recovery_tolerance(self):
        physics = _SequencePhysics(steps=1, residuals=[0.5, 0.0])
        solver = StepSolver(_Params(), physics)
        solver.LineSearch_Lims = [0.25, 2.0]
        updates = []
        solver._update_solution = lambda du, step: updates.append(
            (du.value, step)
        )

        alpha = solver.Perform_LBFGS_LineSearch(
            _ResidualVec(1.0),
            _ResidualVec(-1.0),
            step=3,
            curvature_tolerance=0.1,
        )

        assert alpha == pytest.approx(2.0)
        assert updates == [(-1.0, 3), (-1.0, 3)]

    def test_lbfgs_recovery_line_search_retains_lowest_residual_curvature_trial(
        self,
    ):
        class _ControlledNormVec(_ResidualVec):
            def __init__(self, value, norm):
                super().__init__(value)
                self._norm = norm

            def norm(self):
                return self._norm

        class _ControlledNormPhysics(_SequencePhysics):
            def Get_Fvec(self, step):
                self.f_assemblies += 1
                return self._residuals.pop(0)

        physics = _ControlledNormPhysics(
            steps=1,
            residuals=[
                _ControlledNormVec(0.5, 0.25),
                _ControlledNormVec(-0.1, 0.8),
            ],
        )
        solver = StepSolver(_Params(), physics)
        solver.LineSearch_Lims = [0.25, 2.0]
        updates = []
        solver._update_solution = lambda du, step: updates.append(
            (du.value, step)
        )

        alpha = solver.Perform_LBFGS_LineSearch(
            _ControlledNormVec(2.0, 2.0),
            _ResidualVec(-1.0),
            step=3,
            curvature_tolerance=0.1,
            prefer_residual=True,
        )

        assert alpha == pytest.approx(1.0)
        assert updates == [(-1.0, 3), (-1.0, 3), (1.0, 3)]

    def test_lbfgs_line_search_forces_minimum_without_positive_curvature(
        self, capsys
    ):
        physics = _SequencePhysics(
            steps=1, residuals=[3.0, 2.5, 2.25]
        )
        solver = StepSolver(_Params(), physics)
        solver.LineSearch_Lims = [0.25, 1.0]
        updates = []
        solver._update_solution = lambda du, step: updates.append(
            (du.value, step)
        )

        alpha = solver.Perform_LBFGS_LineSearch(
            _ResidualVec(2.0), _ResidualVec(-1.0), step=3
        )

        assert alpha == pytest.approx(0.25)
        assert updates == [(-1.0, 3), (0.5, 3), (0.25, 3)]
        assert "Forced minimum alpha" in capsys.readouterr().out

    def test_lbfgs_line_search_backtracks_when_boundary_residual_increases(
        self, capsys
    ):
        class _ControlledNormVec(_ResidualVec):
            def __init__(self, value, norm):
                super().__init__(value)
                self._norm = norm

            def norm(self):
                return self._norm

        class _ControlledNormPhysics(_SequencePhysics):
            def Get_Fvec(self, step):
                self.f_assemblies += 1
                return self._residuals.pop(0)

        physics = _ControlledNormPhysics(
            steps=1,
            residuals=[
                _ControlledNormVec(1.0, 3.0),
                _ControlledNormVec(1.5, 1.5),
            ],
        )
        solver = StepSolver(_Params(), physics)
        solver.LineSearch_Lims = [0.25, 1.0]
        solver.LBFGS_LineSearch_MaxResidualGrowth = 1.0
        updates = []
        solver._update_solution = lambda du, step: updates.append(
            (du.value, step)
        )

        alpha = solver.Perform_LBFGS_LineSearch(
            _ControlledNormVec(2.0, 2.0),
            _ResidualVec(-1.0),
            step=3,
        )

        assert alpha == pytest.approx(0.5)
        assert updates == [(-1.0, 3), (0.5, 3)]
        output = capsys.readouterr().out
        assert "Accepted alpha = 0.500000" in output
        assert "Accepted boundary alpha" not in output

    def test_lbfgs_line_search_stops_when_unbracketed_trials_would_cycle(
        self, capsys
    ):
        physics = _SequencePhysics(steps=1, residuals=[0.75, 2.0])
        solver = StepSolver(_Params(), physics)
        solver.LineSearch_Lims = [0.25, 4.0]
        solver.LBFGS_LineSearch_Tolerance = 0.1
        updates = []
        solver._update_solution = lambda du, step: updates.append(
            (du.value, step)
        )

        alpha = solver.Perform_LBFGS_LineSearch(
            _ResidualVec(1.0), _ResidualVec(-1.0), step=3
        )

        assert alpha == pytest.approx(1.0)
        assert updates == [(-1.0, 3), (-1.0, 3), (1.0, 3)]
        assert physics.f_assemblies == 2
        assert (
            "Accepted lowest-residual curvature alpha = 1.000000"
            in capsys.readouterr().out
        )

    def test_solve_step_uses_line_search_when_enabled(self):
        physics = _SequencePhysics(steps=1, residuals=[0.0, 0.0, 0.0])
        solver = StepSolver(_Params(), physics)
        calls = []

        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(2.0),
            rhs=None,
        )
        def _line_search(F_old, du, step):
            calls.append((F_old.value, du.value, step))
            return 0.5

        solver.Perform_LineSearch = _line_search

        solver.Solve_Step()

        assert calls == [(0.0, 2.0, 0)]

    def test_solve_step_uses_directional_line_search_for_lbfgs(self):
        class _LBFGSParams(_Params):
            Step_Solvers = {0: "LBFGS"}

        physics = _SequencePhysics(steps=1, residuals=[1.0, 0.0, 0.0])
        solver = StepSolver(_LBFGSParams(), physics)
        calls = []
        solver._initialize_solver = lambda F, step: None
        solver._lbfgs_direction = (
            lambda K, F, step, history: _ResidualVec(-1.0)
        )
        solver._ensure_lbfgs_descent_direction = (
            lambda K, F, direction, step, history: (K, F, direction)
        )
        solver.Perform_LineSearch = lambda *args: pytest.fail(
            "Newton line search was used for l-BFGS"
        )

        def _directional_line_search(F, direction, step):
            calls.append((F.value, direction.value, step))
            return 1.0

        solver.Perform_LBFGS_LineSearch = _directional_line_search
        solver._update_lbfgs_history = lambda *args: True

        solver.Solve_Step()

        assert calls == [(1.0, -1.0, 0)]

    def test_solve_step_lbfgs_reuses_initial_tangent_and_force_only_assembly(self):
        class _LBFGSParams(_Params):
            Step_Solvers = {0: "LBFGS"}
            max_it = 2
            line_search = False

        physics = _SequencePhysics(steps=1, residuals=[2.0, 1.0, 0.0, 0.0])
        solver = StepSolver(_LBFGSParams(), physics)
        solve_inputs = []

        def _solve_linear_system(K, F, step):
            solve_inputs.append(F.value)
            return SolverWorkspace(
                signature=("LU", (1, 1), 1, 1, "aij"),
                ksp=None,
                du=_ResidualVec(-F.value),
                rhs=None,
            )

        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = _solve_linear_system

        solver.Solve_Step()

        assert physics.kf_assemblies == 1
        assert physics.f_assemblies == 3
        assert solve_inputs == pytest.approx([2.0, -1.0, 0.0])

    def test_solve_step_dispatches_nonlinear_method_per_staggered_step(self):
        class _MixedMethodParams(_SequenceParams):
            Step_Solvers = {0: "LBFGS", 1: "Newton"}
            n_passes = 1
            line_search = False

        physics = _SequencePhysics(
            steps=2, residuals=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        solver = StepSolver(_MixedMethodParams(), physics)
        lbfgs_steps = []
        newton_steps = []

        def _lbfgs_direction(K, F, step, history):
            lbfgs_steps.append(step)
            return _ResidualVec(1.0)

        def _solve_linear_system(K, F, step):
            newton_steps.append(step)
            return SolverWorkspace(
                signature=("LU", (1, 1), 1, 1, "aij"),
                ksp=None,
                du=_ResidualVec(1.0),
                rhs=None,
            )

        solver._initialize_solver = lambda F, step: None
        solver._lbfgs_direction = _lbfgs_direction
        solver._ensure_lbfgs_descent_direction = (
            lambda K, F, direction, step, history: (K, F, direction)
        )
        solver._solve_linear_system = _solve_linear_system

        solver.Solve_Step()

        assert lbfgs_steps == [0]
        assert newton_steps == [1]

    def test_solve_step_delegates_to_separate_innerstep_solvers(self):
        class _MixedMethodParams(_SequenceParams):
            Step_Solvers = {0: "LBFGS", 1: "Newton"}
            n_passes = 1

        physics = _SequencePhysics(steps=2, residuals=[0.0, 0.0])
        solver = StepSolver(_MixedMethodParams(), physics)
        calls = []

        solver.solve_innerstep_bfgs = (
            lambda step, residual_scale: calls.append(
                ("LBFGS", step, residual_scale)
            )
        )
        solver.solve_innerstep_newton = (
            lambda step, residual_scale: calls.append(
                ("Newton", step, residual_scale)
            )
        )

        solver.Solve_Step()

        assert [(method, step) for method, step, _ in calls] == [
            ("LBFGS", 0),
            ("Newton", 1),
        ]
        assert calls[0][2] is calls[1][2]

    @pytest.mark.parametrize("solver_method", ["Newton", "LBFGS"])
    def test_innerstep_solvers_share_disabled_line_search_update(
        self, solver_method
    ):
        solver = StepSolver(_Params(), _DummyPhysics())
        solver.LineSearch = False
        updates = []
        direction = _ResidualVec(2.0)
        solver._update_solution = (
            lambda du, step: updates.append((du, step))
        )

        alpha = solver._apply_innerstep_line_search(
            solver_method,
            _ResidualVec(3.0),
            direction,
            step=4,
        )

        assert alpha == pytest.approx(1.0)
        assert updates == [(direction, 4)]

    def test_innerstep_solvers_share_residual_convergence_criteria(self):
        class _ConvergenceParams(_Params):
            abs_tol = 2.0
            rel_tol = 0.25

        solver = StepSolver(_ConvergenceParams(), _DummyPhysics())
        residual_scale = np.asarray([8.0])

        residual = solver._residual_state(
            _ResidualVec(1.5), 0, residual_scale
        )

        assert residual.norm == pytest.approx(1.5)
        assert residual.relative == pytest.approx(0.1875)
        assert residual.converged is True

    def test_lbfgs_history_accepts_positive_curvature_at_small_scale(self):
        solver = StepSolver(_Params(), _DummyPhysics())
        history = []

        updated = solver._update_lbfgs_history(
            history,
            _ScalarMatrix(),
            _ResidualVec(1.0e-8),
            1.0,
            _ResidualVec(0.0),
            _ResidualVec(1.0e-8),
        )

        assert len(history) == 1
        assert history[0][2] == pytest.approx(1.0e16)
        assert updated is True
        solver._destroy_lbfgs_history(history)

    def test_lbfgs_powell_damps_negative_curvature_against_positive_base(self):
        solver = StepSolver(_Params(), _DummyPhysics())
        history = []

        updated = solver._update_lbfgs_history(
            history,
            _ScalarMatrix(4.0),
            _ResidualVec(1.0),
            1.0,
            _ResidualVec(1.0),
            _ResidualVec(0.0),
        )

        assert updated is True
        assert len(history) == 1
        assert history[0][1].value == pytest.approx(0.8)
        assert history[0][2] == pytest.approx(1.25)
        solver._destroy_lbfgs_history(history)

    def test_lbfgs_diagonal_fallback_preserves_variable_scale_and_descent(self):
        solver = StepSolver(_Params(), _DummyPhysics())

        direction = solver._diagonally_scaled_descent_direction(
            _ScalarMatrix(4.0), _ResidualVec(2.0)
        )

        assert direction.value == pytest.approx(-0.5)
        assert _ResidualVec(2.0).dot(direction) < 0.0

    def test_lbfgs_scales_preconditioner_to_latest_secant_curvature(self):
        solver = StepSolver(_Params(), _DummyPhysics())
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(-2.0 * F.value),
            rhs=None,
        )
        history = [
            (_ResidualVec(1.0), _ResidualVec(2.0), 0.5)
        ]

        scale = solver._lbfgs_initial_inverse_scale(
            None, 0, history
        )

        assert scale == pytest.approx(0.25)

    def test_lbfgs_uses_unscaled_preconditioner_for_indefinite_base(self):
        solver = StepSolver(_Params(), _DummyPhysics())
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(F.value),
            rhs=None,
        )
        history = [
            (_ResidualVec(1.0), _ResidualVec(2.0), 0.5)
        ]

        scale = solver._lbfgs_initial_inverse_scale(
            None, 0, history
        )

        assert scale == 1.0

    def test_lbfgs_keeps_existing_history_when_new_curvature_is_invalid(self):
        solver = StepSolver(_Params(), _DummyPhysics())
        existing_pair = (
            _ResidualVec(2.0),
            _ResidualVec(1.0),
            0.5,
        )
        history = [existing_pair]

        updated = solver._update_lbfgs_history(
            history,
            _ScalarMatrix(-1.0),
            _ResidualVec(1.0),
            1.0,
            _ResidualVec(1.0),
            _ResidualVec(0.0),
        )

        assert history == [existing_pair]
        assert updated is False
        solver._destroy_lbfgs_history(history)

    def test_lbfgs_refreshes_base_tangent_after_rejected_curvature(self, capsys):
        class _LBFGSParams(_Params):
            Step_Solvers = {0: "LBFGS"}
            abs_tol = 3.0
            line_search = False

        physics = _SequencePhysics(
            steps=1, residuals=[1.0, 2.0, 2.0, 2.0]
        )
        solver = StepSolver(_LBFGSParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(-F.value),
            rhs=None,
        )
        solver._update_lbfgs_history = lambda *args: False

        solver.Solve_Step()

        assert physics.kf_assemblies == 2
        assert physics.f_assemblies == 2
        assert "Refreshing l-BFGS base tangent" in capsys.readouterr().out

    def test_lbfgs_keeps_valid_history_active_after_rejected_curvature(self):
        class _LBFGSParams(_Params):
            Step_Solvers = {0: "LBFGS"}
            max_it = 3
            line_search = False

        physics = _SequencePhysics(
            steps=1,
            residuals=[3.0, 2.0, 3.0, 3.0, 0.0, 0.0],
        )
        solver = StepSolver(_LBFGSParams(), physics)
        active_history_lengths = []
        solver._initialize_solver = lambda F, step: None

        def _direction(K, F, step, history):
            active_history_lengths.append(len(history))
            return _ResidualVec(-1.0)

        solver._lbfgs_direction = _direction
        solver._ensure_lbfgs_descent_direction = (
            lambda K, F, direction, step, history: (K, F, direction)
        )
        update_calls = 0

        def _store_then_reject(history, K, direction, alpha, F_old, F_new):
            nonlocal update_calls
            update_calls += 1
            if update_calls == 2:
                return False
            history.append(
                (_ResidualVec(1.0), _ResidualVec(1.0), 1.0)
            )
            return True

        solver._update_lbfgs_history = _store_then_reject

        solver.Solve_Step()

        assert active_history_lengths == [0, 1, 1]

    def test_lbfgs_refreshes_tangent_without_rolling_back_or_clearing_history(
        self, capsys
    ):
        class _LBFGSParams(_Params):
            Step_Solvers = {0: "LBFGS"}
            LBFGS_StagnationIterations = 3
            max_it = 8
            abs_tol = 0.1
            line_search = False

        physics = _SequencePhysics(
            steps=1,
            residuals=[
                10.0,
                5.0,
                4.0,
                4.1,
                4.2,
                4.3,
                4.3,
                4.25,
                3.5,
                0.0,
                0.0,
            ],
        )
        solver = StepSolver(_LBFGSParams(), physics)
        active_history_lengths = []
        solution_updates = []
        solver._initialize_solver = lambda F, step: None

        def _direction(K, F, step, history):
            active_history_lengths.append(len(history))
            return _ResidualVec(-1.0)

        def _update_solution(direction, step):
            solution_updates.append(direction.value)
            physics.mesh.update_step_from_vector(direction, step)

        def _store_pair(history, K, direction, alpha, F_old, F_new):
            history.append(
                (_ResidualVec(1.0), _ResidualVec(1.0), 1.0)
            )
            return True

        solver._lbfgs_direction = _direction
        solver._ensure_lbfgs_descent_direction = (
            lambda K, F, direction, step, history: (K, F, direction)
        )
        solver._update_solution = _update_solution
        solver._update_lbfgs_history = _store_pair

        solver.Solve_Step()

        assert active_history_lengths == [0, 1, 2, 3, 4, 5, 6, 7]
        assert solution_updates == [-1.0] * 8
        assert physics.kf_assemblies == 2
        output = capsys.readouterr().out
        assert "current iterate while retaining valid correction pairs" in output
        assert "fresh active history improved" in output

    def test_lbfgs_tightens_only_first_line_search_after_stagnation(self):
        class _LBFGSParams(_Params):
            Step_Solvers = {0: "LBFGS"}
            LBFGS_StagnationIterations = 3
            max_it = 8
            abs_tol = 0.1

        physics = _SequencePhysics(
            steps=1,
            residuals=[
                10.0,
                5.0,
                4.0,
                4.1,
                4.2,
                4.3,
                4.3,
                4.25,
                0.0,
                0.0,
            ],
        )
        solver = StepSolver(_LBFGSParams(), physics)
        search_settings = []
        solver._initialize_solver = lambda F, step: None
        solver._lbfgs_direction = (
            lambda K, F, step, history: _ResidualVec(-1.0)
        )
        solver._ensure_lbfgs_descent_direction = (
            lambda K, F, direction, step, history: (K, F, direction)
        )
        solver._update_lbfgs_history = lambda *args: True

        def _line_search(
            F,
            direction,
            step,
            curvature_tolerance=None,
            prefer_residual=False,
        ):
            search_settings.append(
                (curvature_tolerance, prefer_residual)
            )
            return 1.0

        solver.Perform_LBFGS_LineSearch = _line_search

        solver.Solve_Step()

        assert search_settings == [
            (None, False),
            (None, False),
            (None, False),
            (None, False),
            (None, False),
            (
                pytest.approx(solver.LBFGS_RecoveryLineSearchTolerance),
                True,
            ),
            (None, False),
        ]

    def test_residual_norm_is_global_dof_normalized_f2(self):
        class _DistributedResidual(_ResidualVec):
            def getSize(self):
                return 9

        solver = StepSolver(_Params(), _DummyPhysics())

        assert solver._compute_residual_norm(
            _DistributedResidual(6.0)
        ) == pytest.approx(2.0)

    def test_solve_step_warns_and_returns_when_no_fields_are_registered(self, capsys):
        physics = _DummyPhysics()
        physics.get_fields_for_step = lambda step: []
        solver = StepSolver(_Params(), physics)

        solver.Solve_Step()

        assert "No fields to solve for this step" in capsys.readouterr().out

    def test_solve_step_multi_pass_prints_debug_and_stops_early(self, capsys):
        physics = _SequencePhysics(
            steps=2,
            residuals=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        solver = StepSolver(_SequenceParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(1.0),
            rhs=None,
        )

        solver.Solve_Step()

        output = capsys.readouterr().out
        assert "Pass 1/2" in output
        assert "Staggered Step 0" in output
        assert "Iteration duration" in output
        assert "Staggered Step duration" in output
        assert "All staggered steps converged after post-pass check" in output

    def test_solve_step_reports_nonconvergence_and_raises(self, capsys):
        physics = _SequencePhysics(steps=1, residuals=[1.0, 1.0, 1.0])

        class _ParamsNoConvergence:
            Solution_Steps = {"u": 0}
            n_passes = 1
            max_it = 1
            rel_tol = 1.0e-12
            abs_tol = 1.0e-12
            line_search = False
            line_search_lims = [0.25, 0.75]
            DEBUG_PRINT_TIMING = False

        solver = StepSolver(_ParamsNoConvergence(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=None,
            rhs=None,
        )
        solver._update_solution = lambda du, step: None

        with pytest.raises(RuntimeError, match="Staggered step 0 did not converge"):
            solver.Solve_Step()

        output = capsys.readouterr().out
        assert "WARNING: Did not converge" in output
        assert "Final RMS residual:" in output
        assert "Post-pass Step 0" not in output

    def test_solve_step_aborts_when_line_search_meets_nonfinite_residuals_without_descent(self, capsys):
        # Each Newton iteration: assembled residual 1.0, then trial residuals at
        # alpha = 1, 0.5, 0.25 of NaN, 10 and 10: no trial reduces the merit.
        residuals = [1.0, float("nan"), 10.0, 10.0] * 2
        physics = _SequencePhysics(steps=1, residuals=residuals)

        class _AbortParams(_Params):
            max_it = 3
            line_search_nonfinite_abort = 2

        solver = StepSolver(_AbortParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"), ksp=None, du=_ResidualVec(1.0), rhs=None,
        )
        solver._update_solution = lambda du, step: None

        with pytest.raises(RuntimeError, match="non-finite residuals without descent in 2 consecutive"):
            solver.Solve_Step()

        assert "aborting step 0" in capsys.readouterr().out
        assert solver.last_line_search_outcome.nonfinite_trials == 1
        assert solver.last_line_search_outcome.merit_decreased is False

    def test_solve_step_line_search_nonfinite_abort_is_off_by_default(self):
        residuals = [1.0, float("nan"), 10.0, 10.0, 1.0] * 2
        physics = _SequencePhysics(steps=1, residuals=residuals)

        class _DefaultParams(_Params):
            max_it = 2

        solver = StepSolver(_DefaultParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"), ksp=None, du=_ResidualVec(1.0), rhs=None,
        )
        solver._update_solution = lambda du, step: None

        # Without the option the solve only fails through the ordinary iteration limit.
        with pytest.raises(RuntimeError, match="did not converge after 2 iterations"):
            solver.Solve_Step()

    def test_line_search_outcome_reports_descent_and_finite_trials(self):
        physics = _SequencePhysics(steps=1, residuals=[4.0, 0.5])
        solver = StepSolver(_Params(), physics)
        solver._update_solution = lambda du, step: None

        solver.Perform_LineSearch(_ResidualVec(1.0), _ResidualVec(2.0), step=0)

        outcome = solver.last_line_search_outcome
        assert outcome.alpha == 0.5
        assert outcome.nonfinite_trials == 0
        assert outcome.merit_decreased is True

    def test_solve_step_aborts_when_staggered_passes_stagnate(self, capsys):
        # Per pass: each inner step assembles once before and once after its
        # update (the latter converged), then two post-pass residuals follow,
        # which shrink by less than the stagnation factor from pass 2 onwards.
        inner = [1.0, 1.0e-14, 1.0, 1.0e-14]
        residuals = (inner + [1.0, 1.0]) + (inner + [0.95, 0.95]) + (inner + [0.94, 0.94])
        physics = _SequencePhysics(steps=2, residuals=residuals)

        class _StagnationParams(_SequenceParams):
            n_passes = 4
            rel_tol = 1.0e-12
            abs_tol = 1.0e-12
            DEBUG_PRINT_TIMING = False
            staggered_stagnation_passes = 2
            staggered_stagnation_factor = 0.9

        solver = StepSolver(_StagnationParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"), ksp=None, du=_ResidualVec(1.0), rhs=None,
        )

        with pytest.raises(RuntimeError, match="stagnated over 2 passes"):
            solver.Solve_Step()

        assert "aborting the staggered iteration" in capsys.readouterr().out

    def test_solve_step_stagnation_abort_tolerates_slow_but_sufficient_decrease(self, capsys):
        inner = [1.0, 1.0e-14, 1.0, 1.0e-14]
        residuals = (inner + [1.0, 1.0]) + (inner + [0.5, 0.5]) + (inner + [0.45, 0.45]) + (inner + [0.4, 0.4])
        physics = _SequencePhysics(steps=2, residuals=residuals)

        class _StagnationParams(_SequenceParams):
            n_passes = 4
            rel_tol = 1.0e-12
            abs_tol = 1.0e-12
            DEBUG_PRINT_TIMING = False
            staggered_stagnation_passes = 2
            staggered_stagnation_factor = 0.9

        solver = StepSolver(_StagnationParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"), ksp=None, du=_ResidualVec(1.0), rhs=None,
        )

        solver.Solve_Step()

        output = capsys.readouterr().out
        assert "Reached 4 staggered passes" in output
        assert "aborting the staggered iteration" not in output

    @pytest.mark.parametrize(
        "name, value",
        [
            ("staggered_stagnation_passes", -1),
            ("staggered_stagnation_factor", 1.0),
            ("staggered_stagnation_factor", 0.0),
            ("line_search_nonfinite_abort", -2),
        ],
    )
    def test_rejects_invalid_early_abort_options(self, name, value):
        class _InvalidParams(_Params):
            pass

        setattr(_InvalidParams, name, value)

        with pytest.raises(ValueError, match=name):
            StepSolver(_InvalidParams(), _DummyPhysics())

    def test_solve_step_convergence_uses_force_l2_norm_not_update(self, capsys):
        physics = _SequencePhysics(steps=1, residuals=[10.0, 3.0, 3.0])

        class _ForceNormParams(_Params):
            max_it = 1
            abs_tol = 4.0
            line_search = False

        solver = StepSolver(_ForceNormParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(100.0),
            rhs=None,
        )

        solver.Solve_Step()

        output = capsys.readouterr().out
        assert "Iteration  0: RMS Residual = 3.000000e+00" in output

    def test_solve_step_relative_convergence_uses_first_updated_force_norm(self, capsys):
        physics = _SequencePhysics(
            steps=1, residuals=[10.0, 3.0, 1.0, 1.0]
        )

        class _RelativeNormParams(_Params):
            max_it = 2
            rel_tol = 0.4
            abs_tol = 1.0e-12
            line_search = False

        solver = StepSolver(_RelativeNormParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(1.0),
            rhs=None,
        )

        solver.Solve_Step()

        output = capsys.readouterr().out
        assert "Iteration  0: RMS Residual = 3.000000e+00, Relative = 1.000000e+00" in output
        assert "Iteration  1: RMS Residual = 1.000000e+00, Relative = 3.333333e-01" in output

    @pytest.mark.parametrize("max_residual, converges", [(float("inf"), True), (1.0e-3, False)])
    def test_solve_step_relative_convergence_after_overshoot_respects_max_residual(self, max_residual, converges, capsys):
        # The first update overshoots: relative to it, the next residual is tiny although it is still large.
        physics = _SequencePhysics(steps=1, residuals=[1.0, 1.0e30, 1.0, 1.0, 1.0])

        class _GuardParams(_Params):
            max_it = 2
            rel_tol = 1.0e-6
            abs_tol = 1.0e-12
            rel_tol_max_residual = max_residual
            line_search = False

        solver = StepSolver(_GuardParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(1.0),
            rhs=None,
        )
        solver._update_solution = lambda du, step: None

        if converges:
            solver.Solve_Step()
            assert "Converged in 2 iterations!" in capsys.readouterr().out
        else:
            with pytest.raises(RuntimeError, match="Staggered step 0 did not converge"):
                solver.Solve_Step()

    def test_solve_step_postpass_uses_residual_norm_without_update_vector(self, monkeypatch, capsys):
        physics = _SequencePhysics(steps=1, residuals=[0.0, 0.0, 0.25])
        solver = StepSolver(_Params(), physics)
        solver.LineSearch = False
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=None,
            rhs=None,
        )
        solver._update_solution = lambda du, step: None

        isnan_results = iter([False, True])
        monkeypatch.setattr(step_solver_module.np, "isnan", lambda value: next(isnan_results))

        solver.Solve_Step()

        output = capsys.readouterr().out
        assert "Post-pass Step 0: RMS Residual = 2.500000e-01" in output

    def test_solve_step_prints_memory_debug_summary_when_requested(self, monkeypatch, capsys):
        class _DebugMemoryParams(_Params):
            DEBUG_MEMORY = True

        physics = _SequencePhysics(steps=1, residuals=[0.0, 0.0, 0.0])
        solver = StepSolver(_DebugMemoryParams(), physics)
        solver.LineSearch = False
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=None,
            rhs=None,
        )
        solver._update_solution = lambda du, step: None
        monkeypatch.setattr(solver, "_resident_memory_mb", lambda: None)
        monkeypatch.setattr(
            solver,
            "_tracked_memory_object_counts",
            lambda: {"PythonObjects": 123, "PETSc.Vec": 4},
        )

        solver.Solve_Step()

        output = capsys.readouterr().out
        assert "Memory after completed solve step: RSS=unavailable; PythonObjects=123, PETSc.Vec=4" in output

    def test_resident_memory_mb_parses_proc_status(self, monkeypatch):
        solver = StepSolver(_Params(), _DummyPhysics())
        proc_status = io.StringIO("Name:\tpython\nVmRSS:\t2048 kB\n")
        monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: proc_status)

        assert solver._resident_memory_mb() == pytest.approx(2.0)

    def test_resident_memory_mb_returns_none_for_unavailable_or_incomplete_status(self, monkeypatch):
        solver = StepSolver(_Params(), _DummyPhysics())

        def _raise_oserror(*args, **kwargs):
            raise OSError("status unavailable")

        monkeypatch.setattr(builtins, "open", _raise_oserror)
        assert solver._resident_memory_mb() is None

        monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: io.StringIO("Name:\tpython\n"))
        assert solver._resident_memory_mb() is None

    def test_tracked_memory_object_counts_reports_expected_categories(self, monkeypatch):
        class _Vec:
            pass

        class _Mat:
            pass

        class _KSP:
            pass

        class _PC:
            pass

        class _IS:
            pass

        class _Function:
            pass

        class _Expression:
            pass

        solver = StepSolver(_Params(), _DummyPhysics())
        objects = [_Vec(), _Mat(), _KSP(), _PC(), _IS(), _Function(), _Expression(), object()]
        monkeypatch.setattr(
            step_solver_base_module,
            "PETSc",
            SimpleNamespace(Vec=_Vec, Mat=_Mat, KSP=_KSP, PC=_PC, IS=_IS),
        )
        monkeypatch.setattr(
            step_solver_base_module,
            "fem",
            SimpleNamespace(Function=_Function, Expression=_Expression),
        )
        monkeypatch.setattr(step_solver_base_module.gc, "get_objects", lambda: objects)

        counts = solver._tracked_memory_object_counts()

        assert counts == {
            "PythonObjects": len(objects),
            "PETSc.Vec": 1,
            "PETSc.Mat": 1,
            "PETSc.KSP": 1,
            "PETSc.PC": 1,
            "PETSc.IS": 1,
            "fem.Function": 1,
            "fem.Expression": 1,
        }

    def test_solve_step_debug_plotting_calls_mesh_plot_fields(self):
        class _DebugPlotParams:
            Solution_Steps = {"u": 0}
            n_passes = 1
            max_it = 1
            rel_tol = 1.0e-6
            abs_tol = 1.0e-8
            line_search = False
            line_search_lims = [0.25, 0.75]
            DEBUG_PRINT_TIMING = False
            _DEBUG_PLOTTING = True

        physics = _SequencePhysics(steps=1, residuals=[0.0, 0.0, 0.0])
        solver = StepSolver(_DebugPlotParams(), physics)
        solver._initialize_solver = lambda F, step: None
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=None,
            du=_ResidualVec(1.0),
            rhs=None,
        )

        solver.Solve_Step()

        assert physics.mesh.plot_calls == 1

    def test_perform_line_search_resolves_a_tension_compression_stiffness_jump(self):
        # A damaged tensile spring entering compression needs a correction well
        # below 0.125 before the residual-norm line search sees a decrease.
        physics = _DummyPhysics()
        displacement = 0.01
        residual = lambda: _DummyVec((1e-6 if displacement > 0 else 1.0)*displacement+1.0)
        physics.Get_Fvec = lambda step: residual()
        solver = StepSolver(_Params(), physics)
        solver.LineSearch_Lims = [1e-6, 1.0]

        def update(du, step):
            nonlocal displacement
            displacement += du.value

        solver._update_solution = update
        initial = residual()
        direction = _DummyVec(-initial.value/1e-6)
        alpha = solver.Perform_LineSearch(initial, direction, step=0)

        assert 1e-6 <= alpha < 0.1
        assert residual().norm() < initial.norm()
        assert displacement == pytest.approx(0.01+alpha*direction.value)

    @pytest.mark.parametrize("minimum, accepted", [(0.25, 0.25), (1e-6, 2**-19)])
    def test_perform_line_search_accepts_smallest_step_when_merit_does_not_decrease(self, minimum, accepted):
        physics = _DummyPhysics()
        physics.Get_Fvec = lambda step: _DummyVec(2.0)
        solver = StepSolver(_Params(), physics)
        solver.LineSearch_Lims = [minimum, 1.0]
        updates = []
        solver._update_solution = lambda du, step: updates.append((du.value, step))

        alpha = solver.Perform_LineSearch(_DummyVec(1.0), _DummyVec(-4.0), step=1)

        assert alpha == accepted
        assert updates[0] == (-4.0, 1)
        assert updates[-1][1] == 1
        assert sum(value for value, step in updates) == pytest.approx(-4.0*alpha)


class _ArrayField:
    def __init__(self, values):
        self.x = SimpleNamespace(array=np.asarray(values, dtype=float), scatter_forward=lambda: None)


class _TwoFieldMesh:
    """Monolithic step with the fields ``C`` (positive) and ``E`` (unrestricted), backed by NumPy arrays."""

    def __init__(self, C, E):
        self.fields = {"C": _ArrayField(C), "E": _ArrayField(E)}
        self.step_fields = {0: ["C", "E"]}

    def Get_Residual_Row_Scale(self, step):
        return None

    def get_step_block_offsets(self, step):
        n_C, n_E = len(self.fields["C"].x.array), len(self.fields["E"].x.array)
        return [0, n_C, n_C + n_E]

    def update_step_from_vector(self, du, step):
        values = du.getArray()
        offsets = self.get_step_block_offsets(step)
        for name, start, end in zip(self.step_fields[step], offsets, offsets[1:]):
            self.fields[name].x.array += values[start:end]


class TestPositiveFields:
    def _solve_once(self, positive_fields, C, direction_C, factor=0.1, fraction_fields=()):
        from petsc4py import PETSc

        C = np.asarray(C, dtype=float)
        mesh = _TwoFieldMesh(C=C.copy(), E=[0.5, 0.5])
        physics = _SequencePhysics(steps=1, residuals=[1.0, 0.0, 0.0])
        physics.mesh = mesh

        class _PositiveParams(_Params):
            line_search = False
            Positive_Fields = positive_fields
            Fraction_Fields = list(fraction_fields)
            Positive_Update_Factor = factor

        solver = StepSolver(_PositiveParams(), physics)
        solver._initialize_solver = lambda F, step: None
        direction = PETSc.Vec().createWithArray(np.concatenate([direction_C, [-2.0, -2.0]]))
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"), ksp=None, du=direction, rhs=None)
        solver.Solve_Step()
        return mesh.fields["C"].x.array, mesh.fields["E"].x.array

    def test_decreases_stay_positive_and_are_bounded_by_the_factor(self):
        C0 = np.array([1.0, 2.0e-3, 1.0, 1.0e-30])
        du = np.array([-0.5, -5.0e-3, -1.0e3, -1.0])            # the last three would go negative
        C, E = self._solve_once(["C"], C0, du, factor=0.1)
        assert np.all(C > 0.0)
        assert np.all(C >= 0.1 * C0 * (1.0 - 1e-12))
        assert np.all(C < C0)                                    # still a decrease
        assert C[2] == pytest.approx(0.1, rel=1e-6)              # a huge decrease approaches the cap smoothly
        np.testing.assert_allclose(E, [-1.5, -1.5])              # other fields take the full step

    def test_small_decreases_and_all_increases_are_the_newton_step(self):
        C0 = np.array([1.0, 1.0e-20, 2.0])
        du = np.array([-1.0e-6, -1.0e-26, 5.0])
        C, _E = self._solve_once(["C"], C0, du, factor=1.0e-4)
        np.testing.assert_allclose(C, C0 + du, rtol=1e-6)

    def test_zero_and_negative_entries_are_not_limited(self):
        C, _E = self._solve_once(["C"], [0.0, -1.0], [-1.0, -1.0])
        np.testing.assert_allclose(C, [-1.0, -2.0])

    def test_fraction_fields_stay_inside_the_unit_interval(self):
        theta0 = np.array([0.5, 1.0e-6, 0.999, 0.3])
        du = np.array([5.0, 14.0, 1.0, -2.0])                    # the first three would exceed 1, the last go negative
        theta, _E = self._solve_once([], theta0, du, factor=0.1, fraction_fields=["C"])
        assert np.all((theta > 0.0) & (theta < 1.0))
        assert np.all(1.0 - theta[:3] >= 0.1 * (1.0 - theta0[:3]) * (1.0 - 1e-12))
        assert theta[3] >= 0.1 * theta0[3] * (1.0 - 1e-12)

    def test_small_fraction_updates_are_the_newton_step(self):
        theta0 = np.array([0.5, 1.0e-17, 0.999999])
        du = np.array([1.0e-7, 1.0e-23, -1.0e-9])
        theta, _E = self._solve_once([], theta0, du, factor=1.0e-4, fraction_fields=["C"])
        np.testing.assert_allclose(theta, theta0 + du, rtol=1e-6)

    def test_line_search_follows_the_limited_path(self):
        # F = (C - c*, E - k C): the Newton step drives C from 1 to c* = 1e-6, which the factor 0.1 caps.
        # Scaling the capped step changes E by far more than C allows, so every point along it raises the
        # residual norm; the path u0 + limit(alpha du) follows the Newton direction for small alpha.
        from petsc4py import PETSc

        c_star, k = 1.0e-6, 100.0
        mesh = _TwoFieldMesh(C=[1.0], E=[k - 1.0])

        def vector(values):
            vec = PETSc.Vec().createSeq(len(values))
            vec.setArray(values)
            return vec

        def residual(step=0):
            C, E = mesh.fields["C"].x.array[0], mesh.fields["E"].x.array[0]
            return vector([C - c_star, E - k * C])

        physics = _DummyPhysics()
        physics.mesh = mesh
        physics.Get_Fvec = residual

        class _PositiveParams(_Params):
            line_search = True
            Positive_Fields = ["C"]
            Positive_Update_Factor = 0.1
            line_search_lims = [1.0e-3, 1.0]

        solver = StepSolver(_PositiveParams(), physics)
        F0 = residual()
        J = np.array([[1.0, 0.0], [-k, 1.0]])
        newton = -np.linalg.solve(J, F0.getArray())
        capped = vector(newton)
        solver._limit_positive_update(capped, 0)
        assert F0.getArray() @ (J @ capped.getArray()) > 0.0       # the capped step is an ascent direction

        alpha = solver.Perform_LineSearch(F0, vector(newton), 0)

        assert 0.0 < alpha < 1.0
        assert residual().norm() < F0.norm()
        assert mesh.fields["C"].x.array[0] > 0.0

    def test_a_field_cannot_be_positive_and_fraction(self):
        physics = _DummyPhysics()
        physics.mesh = _TwoFieldMesh(C=[0.5], E=[0.0])

        class _Both(_Params):
            Positive_Fields = ["C"]
            Fraction_Fields = ["C"]

        with pytest.raises(ValueError, match="both"):
            StepSolver(_Both(), physics)

    def _solve_capped(self, direction):
        from petsc4py import PETSc

        mesh = _TwoFieldMesh(C=[1.0, 1.0], E=[0.0, 0.0])
        physics = _SequencePhysics(steps=1, residuals=[1.0, 0.0, 0.0])
        physics.mesh = mesh

        class _CapParams(_Params):
            line_search = False
            Max_Field_Update = {"E": 0.2}

        solver = StepSolver(_CapParams(), physics)
        solver._initialize_solver = lambda F, step: None
        vec = PETSc.Vec().createWithArray(np.array(direction, dtype=float))
        solver._solve_linear_system = lambda K, F, step: SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"), ksp=None, du=vec, rhs=None)
        solver.Solve_Step()
        return mesh.fields["C"].x.array - 1.0, mesh.fields["E"].x.array

    def test_max_field_update_damps_the_whole_update_keeping_its_direction(self):
        direction = np.array([-0.5, 0.5, 5.0, 1.0e-4])
        dC, dE = self._solve_capped(direction)
        assert np.max(np.abs(dE)) <= 0.2 * (1.0 + 1e-12)          # a 5 V step is damped to the cap
        applied = np.concatenate([dC, dE])
        np.testing.assert_allclose(applied / direction, applied[2] / direction[2], rtol=1e-10)   # one common factor

    def test_max_field_update_leaves_small_updates_unchanged(self):
        direction = np.array([-0.5, 0.5, 1.0e-4, -2.0e-4])
        dC, dE = self._solve_capped(direction)
        np.testing.assert_allclose(np.concatenate([dC, dE]), direction, rtol=1e-6)

    @pytest.mark.parametrize("cap, message", [({"E": 0.0}, "positive"), ({"theta": 0.1}, "unknown")])
    def test_invalid_max_field_update_is_rejected(self, cap, message):
        physics = _DummyPhysics()
        physics.mesh = _TwoFieldMesh(C=[1.0], E=[0.0])

        class _BadCap(_Params):
            Max_Field_Update = cap

        with pytest.raises(ValueError, match=message):
            StepSolver(_BadCap(), physics)

    def test_without_positive_fields_the_full_update_is_applied(self):
        C, E = self._solve_once([], [1.0, 2.0e-3], [-0.5, -5.0e-3])
        np.testing.assert_allclose(C, [0.5, -3.0e-3])
        np.testing.assert_allclose(E, [-1.5, -1.5])

    @pytest.mark.parametrize("factor", [0.0, 1.0, -0.5])
    def test_invalid_factor_is_rejected(self, factor):
        class _BadFactor(_Params):
            Positive_Update_Factor = factor

        with pytest.raises(ValueError, match="Positive_Update_Factor"):
            StepSolver(_BadFactor(), _DummyPhysics())

    def test_unknown_positive_field_is_rejected(self):
        physics = _DummyPhysics()
        physics.mesh = _TwoFieldMesh(C=[1.0], E=[0.0])

        class _UnknownField(_Params):
            Positive_Fields = ["C", "theta"]

        with pytest.raises(ValueError, match="theta"):
            StepSolver(_UnknownField(), physics)


class _NewtonCycle(BaseModel):
    """Test-only: c = 2 on the mesh, and a global field g with the residual S (g^3 - 2 g + 2), on which Newton
    from g = 0 cycles between g = 1 (residual S) and g = 0 (residual 2 S) without converging."""

    MODEL_TYPE = "Test:NewtonCycle"
    S = 1.0e-6

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        mesh.Add_Field("c", "c", "CG", 1)
        mesh.Add_Global_Field("g", "g")
        for field in ("c", "g"):
            mesh.Add_TestFunction(field, step=0)

    def initialize_fields(self):
        self.mesh.Constant_Field("c", 0.0)
        self.mesh.Constant_Field("g", 0.0)

    def Get_Weak_Form(self, time_constant, dt_constant, step):
        dx = self.mesh.Get_Domain_Integrator()
        test = self.mesh.Get_TestFunction("c", step=0)
        return self.mesh.Get_Trial_Function("c", step=0) * test * dx, (self.mesh.Get_Field("c", step=0) - 2.0) * test * dx

    def Add_KF(self, K, F, step):
        dof = self.mesh.Get_Global_Field_Dof("g", step)
        g = self.mesh.Get_Global_Value("g")
        if comm.rank == 0:
            F.setValues([dof], [self.S * (g**3 - 2.0 * g + 2.0)], addv=PETSc.InsertMode.ADD_VALUES)
            if K is not None:
                K.setValues([dof], [dof], [[self.S * (3.0 * g**2 - 2.0)]], addv=PETSc.InsertMode.ADD)


class TestStagnation:
    @staticmethod
    def _solver(linear_elastic_params, max_it=40, **options):
        """Solver on the cycling model; abs_tol is set from the RMS residuals S/sqrt(N) at g = 1 and 2 S/sqrt(N) at g = 0."""
        params = linear_elastic_params
        params.Models = {"Cycle": {"Type": "Test:NewtonCycle"}}
        params.ModelNames = ["Cycle"]
        params.Solution_Steps = {"c": 0, "g": 0}
        params.max_it, params.rel_tol = max_it, 1.0e-14
        params.line_search = False
        for name, value in options.items():
            setattr(params, name, value)
        mesh = Mesh(params)
        physics = Physics(mesh, params)
        physics.Initialize_Fields()
        rms_at_one = _NewtonCycle.S / np.sqrt(physics.Get_Fvec(0).getSize())
        params.abs_tol = 0.6 * rms_at_one
        return mesh, StepSolver(params, physics)

    def test_cycling_newton_fails_after_max_it_by_default(self, linear_elastic_params):
        _mesh, solver = self._solver(linear_elastic_params, max_it=12)

        with pytest.raises(RuntimeError, match="after 12 iterations"):
            solver.Solve_Step()

    def test_stagnation_ends_the_solve_early(self, linear_elastic_params, capsys):
        _mesh, solver = self._solver(linear_elastic_params, Stagnation_Iterations=4)

        with pytest.raises(RuntimeError, match="after 5 iterations"):
            solver.Solve_Step()
        if comm.rank == 0:          # mprint reports on rank 0 only
            assert "Stagnated" in capsys.readouterr().out

    def test_best_iterate_within_the_factor_is_accepted(self, linear_elastic_params, capsys):
        # the best iterate g = 1 has 1/0.6 times abs_tol
        mesh, solver = self._solver(linear_elastic_params, Stagnation_Iterations=4, Stagnation_Accept_Factor=2.0)

        solver.Solve_Step()

        assert mesh.Get_Global_Value("g") == pytest.approx(1.0)
        np.testing.assert_allclose(mesh.fields["c"].x.array, 2.0)
        if comm.rank == 0:
            assert "Accepted the best iterate" in capsys.readouterr().out

    def test_best_iterate_above_the_factor_is_rejected(self, linear_elastic_params):
        _mesh, solver = self._solver(linear_elastic_params, Stagnation_Iterations=4, Stagnation_Accept_Factor=1.5)

        with pytest.raises(RuntimeError, match="did not converge"):
            solver.Solve_Step()

    def test_accept_factor_also_applies_at_max_it(self, linear_elastic_params):
        mesh, solver = self._solver(linear_elastic_params, max_it=6, Stagnation_Accept_Factor=2.0)

        solver.Solve_Step()

        assert mesh.Get_Global_Value("g") == pytest.approx(1.0)

    def test_trust_region_stagnation_accepts_the_best_iterate(self, linear_elastic_params):
        mesh, solver = self._solver(linear_elastic_params, Stagnation_Iterations=4, Stagnation_Accept_Factor=2.0,
                                    Step_Solvers={0: "NewtonTrustRegion"}, line_search=True)

        solver.Solve_Step()

        # the trust region stalls near the minimum of |F| (g = sqrt(2/3)); that best iterate is restored
        F = solver.physics.Get_Fvec(0)
        residual = F.norm() / np.sqrt(F.getSize())
        assert solver.absolute_tolerance < residual < 2.0 * solver.absolute_tolerance
        np.testing.assert_allclose(mesh.fields["c"].x.array, 2.0)

    @pytest.mark.parametrize("options, match", [({"Stagnation_Iterations": -1}, "must not be negative"),
                                                ({"Stagnation_Accept_Factor": 0.5}, "at least 1"),
                                                ({"Stagnation_Accept_Factor": float("inf")}, "at least 1")])
    def test_invalid_settings_are_rejected(self, linear_elastic_params, options, match):
        with pytest.raises(ValueError, match=match):
            self._solver(linear_elastic_params, **options)

    @pytest.mark.parametrize("order", [1, 2])
    def test_debug_location_of_bernstein_field(self, linear_elastic_params, order):
        # CG fields use the Bernstein variant, which has no DOF coordinates
        from femcubes.core.solvers._step_solver_base import _StepSolverBase

        mesh = Mesh(linear_elastic_params)
        mesh.Add_Field("b", "Bernstein field", "CG", order)
        space = mesh.fields["b"].function_space
        points = mesh._entity_dof_points(space)
        for node in (0, space.dofmap.index_map.size_local - 1):
            location = _StepSolverBase._dof_location(mesh, space, node)
            assert location == pytest.approx(tuple(points[node][: mesh.mesh.geometry.dim]))

    def test_debug_output_reports_every_field(self, linear_elastic_params, capsys):
        _mesh, solver = self._solver(linear_elastic_params, max_it=3, Stagnation_Accept_Factor=2.0, DEBUG_FIELD_CONVERGENCE=True)

        solver.Solve_Step()

        if comm.rank == 0:
            out = capsys.readouterr().out
            for name in ("c", "g"):
                assert f"{name}: RMS residual = " in out
            assert "last update max |du| = " in out
            assert re.search(r"\tc: .* at \([-0-9.e]+, [-0-9.e]+\)", out)        # the mesh field is located
            assert not re.search(r"\tg: .* at \(", out)                           # the global field has no location

