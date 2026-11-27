"""Targeted tests for the internal linear-solver helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
from petsc4py import PETSc

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers._lin_solvers import SolverWorkspace, _LinSolvers


class _Harness(_LinSolvers):
    pass


class _Destroyable:
    def __init__(self):
        self.destroy_calls = 0

    def destroy(self):
        self.destroy_calls += 1


class _FakeVec:
    def __init__(self, value=0.0):
        self.value = float(value)

    def copy(self, result=None):
        if result is None:
            return _FakeVec(self.value)
        result.value = self.value
        return result

    def scale(self, factor):
        self.value *= factor


class _FakePC:
    def __init__(self, sub_ksps=None):
        self.calls = []
        self.field_splits = []
        self.sub_ksps = list(sub_ksps or [])

    def setType(self, pc_type):
        self.calls.append(("setType", pc_type))

    def setGAMGType(self, gamg_type):
        self.calls.append(("setGAMGType", gamg_type))

    def setGAMGSmooths(self, smooths):
        self.calls.append(("setGAMGSmooths", smooths))

    def setFieldSplitType(self, split_type):
        self.calls.append(("setFieldSplitType", split_type))

    def setFieldSplitIS(self, split):
        self.field_splits.append(split)

    def setUp(self):
        self.calls.append(("setUp",))

    def getFieldSplitSubKSP(self):
        return self.sub_ksps


class _FakeKSP:
    def __init__(self, pc=None):
        self.calls = []
        self.pc = pc if pc is not None else _FakePC()

    def setType(self, ksp_type):
        self.calls.append(("setType", ksp_type))

    def setGMRESRestart(self, restart):
        self.calls.append(("setGMRESRestart", restart))

    def setTolerances(self, **kwargs):
        self.calls.append(("setTolerances", kwargs))

    def getPC(self):
        return self.pc

    def setOperators(self, K_mat):
        self.calls.append(("setOperators", K_mat))

    def solve(self, rhs, du):
        self.calls.append(("solve", rhs.value))
        du.value = rhs.value

    def getConvergedReason(self):
        return 1

    def getIterationNumber(self):
        return 0

    def getResidualNorm(self):
        return 0.0

    def destroy(self):
        self.calls.append(("destroy",))


class _NormObject:
    def __init__(self, norm=1.0):
        self._norm = norm

    def norm(self):
        return self._norm


def _build_params(solver=LinearSolver.LU):
    class Params(ParamsTemplate):
        Linear_Solver = solver
        DEBUG_PRINT_TIMING = False

    return Params()


def _build_seq_mat(value):
    mat = PETSc.Mat().createAIJ([1, 1], comm=PETSc.COMM_SELF)
    mat.setUp()
    mat[0, 0] = value
    mat.assemble()
    return mat


def _build_seq_vec(value):
    vec = PETSc.Vec().createSeq(1, comm=PETSc.COMM_SELF)
    vec.setValue(0, value)
    vec.assemble()
    return vec


def _build_diag_mat(values):
    size = len(values)
    mat = PETSc.Mat().createAIJ([size, size], comm=PETSc.COMM_SELF)
    mat.setUp()
    for i, value in enumerate(values):
        mat[i, i] = value
    mat.assemble()
    return mat


def _build_seq_vec_from_values(values):
    vec = PETSc.Vec().createSeq(len(values), comm=PETSc.COMM_SELF)
    for i, value in enumerate(values):
        vec.setValue(i, value)
    vec.assemble()
    return vec


class TestLinSolversInternalHelpers:
    def test_normalize_solver_choice_accepts_enum_and_string(self):
        solver_enum = _Harness(_build_params(LinearSolver.LU), object())
        solver_text = _Harness(_build_params("gmres"), object())

        assert solver_enum.linear_solver == "LU"
        assert solver_text.linear_solver == "GMRES"

    def test_cleanup_solver_destroys_allocated_objects(self):
        solver = _Harness(_build_params(), object())
        solver._solver_cache[0] = SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=_Destroyable(),
            du=_Destroyable(),
            rhs=_Destroyable(),
        )

        solver._cleanup_solver()

        assert solver._solver_cache == {}

    def test_destroy_solver_workspace_releases_fieldsplit_index_sets(self):
        solver = _Harness(_build_params(), object())
        first_is = _Destroyable()
        second_is = _Destroyable()
        workspace = SolverWorkspace(
            signature=("FIELDSPLIT", (2, 2), 2, 2, "aij"),
            ksp=None,
            du=None,
            rhs=None,
            _fieldsplit_is=[first_is, None, second_is],
        )

        solver._destroy_solver_workspace(workspace)

        assert first_is.destroy_calls == 1
        assert second_is.destroy_calls == 1
        assert workspace._fieldsplit_is is None

    def test_prepare_rhs_negates_force_vector(self):
        solver = _Harness(_build_params(), object())
        workspace = SolverWorkspace(
            signature=("LU", (1, 1), 1, 1, "aij"),
            ksp=PETSc.KSP().create(PETSc.COMM_SELF),
            du=_build_seq_vec(0.0),
            rhs=_build_seq_vec(0.0),
        )
        force = _build_seq_vec(3.5)

        solver._prepare_rhs(workspace, force)

        assert workspace.rhs.getArray()[0] == pytest.approx(-3.5)
        workspace.ksp.destroy()
        workspace.du.destroy()
        workspace.rhs.destroy()
        force.destroy()

    def test_solve_linear_system_solves_simple_lu_system(self):
        solver = _Harness(_build_params(LinearSolver.LU), object())
        K_mat = _build_seq_mat(2.0)
        F_vec = _build_seq_vec(4.0)

        workspace = solver._solve_linear_system(K_mat, F_vec, step=0)

        assert workspace.du.getArray()[0] == pytest.approx(-2.0)

        solver._cleanup_solver()
        K_mat.destroy()
        F_vec.destroy()

    def test_solve_linear_system_dispatches_to_gamg_backend(self, monkeypatch):
        solver = _Harness(_build_params(LinearSolver.GAMG), object())
        workspace = SolverWorkspace(
            signature=("GAMG", (1, 1), 1, 1, "aij"),
            ksp=_FakeKSP(),
            du=_FakeVec(),
            rhs=_FakeVec(),
        )
        K_mat = _NormObject()
        F_vec = _NormObject()
        calls = []

        monkeypatch.setattr(solver, "_get_or_create_solver_workspace", lambda step, K, F: workspace)
        monkeypatch.setattr(
            solver,
            "_solve_gamg",
            lambda workspace_arg, K_arg, F_arg, step=None: calls.append((workspace_arg, K_arg, F_arg)),
        )

        returned = solver._solve_linear_system(K_mat, F_vec, step=0)

        assert returned is workspace
        assert calls == [(workspace, K_mat, F_vec)]

    def test_solve_linear_system_dispatches_to_fieldsplit_backend(self, monkeypatch):
        solver = _Harness(_build_params(LinearSolver.FIELDSPLIT), object())
        workspace = SolverWorkspace(
            signature=("FIELDSPLIT", (1, 1), 1, 1, "aij"),
            ksp=_FakeKSP(),
            du=_FakeVec(),
            rhs=_FakeVec(),
        )
        K_mat = _NormObject()
        F_vec = _NormObject()
        calls = []

        monkeypatch.setattr(solver, "_get_or_create_solver_workspace", lambda step, K, F: workspace)
        monkeypatch.setattr(
            solver,
            "_solve_fieldsplit",
            lambda workspace_arg, K_arg, F_arg, step_arg: calls.append(
                (workspace_arg, K_arg, F_arg, step_arg)
            ),
        )

        returned = solver._solve_linear_system(K_mat, F_vec, step=2)

        assert returned is workspace
        assert calls == [(workspace, K_mat, F_vec, 2)]

    def test_solve_linear_system_reuses_workspace_for_same_step_and_size(self):
        solver = _Harness(_build_params(LinearSolver.LU), object())
        K_first = _build_seq_mat(2.0)
        F_first = _build_seq_vec(4.0)
        K_second = _build_seq_mat(3.0)
        F_second = _build_seq_vec(6.0)

        first_workspace = solver._solve_linear_system(K_first, F_first, step=0)

        second_workspace = solver._solve_linear_system(K_second, F_second, step=0)

        assert second_workspace is first_workspace
        assert second_workspace.du.getArray()[0] == pytest.approx(-2.0)

        solver._cleanup_solver()
        K_first.destroy()
        F_first.destroy()
        K_second.destroy()
        F_second.destroy()

    def test_solve_linear_system_rebuilds_workspace_when_size_changes(self):
        solver = _Harness(_build_params(LinearSolver.LU), object())
        K_first = _build_seq_mat(2.0)
        F_first = _build_seq_vec(4.0)
        K_second = _build_diag_mat([2.0, 5.0])
        F_second = _build_seq_vec_from_values([4.0, 10.0])

        first_workspace = solver._solve_linear_system(K_first, F_first, step=0)

        second_workspace = solver._solve_linear_system(K_second, F_second, step=0)

        assert second_workspace is not first_workspace
        assert second_workspace.ksp is not first_workspace.ksp
        assert second_workspace.du is not first_workspace.du
        assert second_workspace.rhs is not first_workspace.rhs
        np.testing.assert_allclose(second_workspace.du.getArray(), [-2.0, -2.0])

        solver._cleanup_solver()
        K_first.destroy()
        F_first.destroy()
        K_second.destroy()
        F_second.destroy()

    def test_solve_linear_system_rejects_invalid_force_vector(self):
        solver = _Harness(_build_params(), object())
        K_mat = _build_seq_mat(1.0)
        F_vec = _build_seq_vec(0.0)
        F_vec.setValue(0, np.nan)
        F_vec.assemble()

        with pytest.raises(RuntimeError, match="invalid force vector"):
            solver._solve_linear_system(K_mat, F_vec, step=0)

        solver._cleanup_solver()
        K_mat.destroy()
        F_vec.destroy()

    def test_solve_linear_system_rejects_invalid_matrix(self):
        solver = _Harness(_build_params(), object())

        class _Matrix:
            def norm(self):
                return np.nan

        F_vec = _build_seq_vec(1.0)

        with pytest.raises(RuntimeError, match="invalid matrix"):
            solver._solve_linear_system(_Matrix(), F_vec, step=0)

        solver._cleanup_solver()
        F_vec.destroy()

    def test_configure_mumps_factor_sets_expected_controls(self):
        solver = _Harness(_build_params(), object())
        calls = []

        class _Factor:
            def setMumpsIcntl(self, idx, value):
                calls.append(("icntl", idx, value))

            def setMumpsCntl(self, idx, value):
                calls.append(("cntl", idx, value))

        class _PC:
            def setFactorOrdering(self, **kwargs):
                calls.append(("ordering", kwargs))

            def setFactorSetUpSolverType(self):
                calls.append(("setup",))

            def getFactorMatrix(self):
                return _Factor()

        solver._configure_mumps_factor(_PC())

        assert ("ordering", {"reuse": True}) in calls
        assert ("setup",) in calls
        assert ("icntl", 28, 2) in calls
        assert ("icntl", 29, 2) in calls
        assert not any(call[:2] == ("icntl", 7) for call in calls)
        assert ("icntl", 24, 1) in calls
        assert ("cntl", 1, 0.01) in calls

    def test_configure_mumps_factor_uses_sequential_ordering_for_asm(self):
        solver = _Harness(_build_params(), object())
        calls = []

        class _Factor:
            def setMumpsIcntl(self, idx, value):
                calls.append(("icntl", idx, value))

            def setMumpsCntl(self, idx, value):
                calls.append(("cntl", idx, value))

        class _PC:
            def setFactorOrdering(self, **kwargs):
                calls.append(("ordering", kwargs))

            def setFactorSetUpSolverType(self):
                calls.append(("setup",))

            def getFactorMatrix(self):
                return _Factor()

        solver._configure_mumps_factor(_PC(), parallel_analysis=False)

        assert ("ordering", {"reuse": True}) in calls
        assert ("icntl", 7, 2) in calls
        assert not any(call[:2] == ("icntl", 28) for call in calls)
        assert not any(call[:2] == ("icntl", 29) for call in calls)

    def test_solve_linear_system_uses_asm_for_gmres(self):
        solver = _Harness(_build_params(LinearSolver.GMRES), object())
        K_mat = _build_seq_mat(2.0)
        F_vec = _build_seq_vec(4.0)

        workspace = solver._solve_linear_system(K_mat, F_vec, step=0)

        assert workspace.du.getArray()[0] == pytest.approx(-2.0)

        solver._cleanup_solver()
        K_mat.destroy()
        F_vec.destroy()

    def test_solve_linear_system_reports_nonconverged_reason(self, monkeypatch, capsys):
        solver = _Harness(_build_params(LinearSolver.LU), object())
        solver.DEBUG_PRINT_TIMING = True
        K_mat = _build_seq_mat(1.0)
        F_vec = _build_seq_vec(1.0)

        class _FakeKSP:
            def getConvergedReason(self):
                return -3

            def getIterationNumber(self):
                return 7

            def getResidualNorm(self):
                return 2.5

            def destroy(self):
                pass

        def _fake_solve_mumps(workspace, K, F):
            workspace.ksp = _FakeKSP()

        monkeypatch.setattr(solver, "_solve_mumps", _fake_solve_mumps)

        solver._solve_linear_system(K_mat, F_vec, step=0)

        output = capsys.readouterr().out
        assert "Warning: Solver did not converge" in output
        assert "Iterations: 7" in output
        assert "Solver step duration" in output

        solver._cleanup_solver()
        K_mat.destroy()
        F_vec.destroy()

    def test_solve_asm_configures_subsolvers_once(self):
        solver = _Harness(_build_params(LinearSolver.GMRES), object())
        workspace = SolverWorkspace(
            signature=("GMRES", (1, 1), 1, 1, "aij"),
            ksp=PETSc.KSP().create(PETSc.COMM_SELF),
            du=_build_seq_vec(0.0),
            rhs=_build_seq_vec(0.0),
        )
        K_mat = _build_seq_mat(2.0)
        F_vec = _build_seq_vec(4.0)

        solver._solve_asm(workspace, K_mat, F_vec)
        assert workspace.asm_configured is True
        assert workspace.du.getArray()[0] == pytest.approx(-2.0)

        solver._solve_asm(workspace, K_mat, F_vec)
        assert workspace.du.getArray()[0] == pytest.approx(-2.0)

        workspace.ksp.destroy()
        workspace.du.destroy()
        workspace.rhs.destroy()
        solver._cleanup_solver()
        K_mat.destroy()
        F_vec.destroy()

    def test_build_fieldsplit_is_returns_none_for_single_field_step(self):
        solver = _Harness(_build_params(LinearSolver.FIELDSPLIT), object())
        solver.physics = SimpleNamespace(mesh=SimpleNamespace(mixed_spaces={}))
        K_mat = _build_diag_mat([1.0, 1.0])

        assert solver._build_fieldsplit_is(step=0, K_mat=K_mat) is None

        K_mat.destroy()

    def test_build_fieldsplit_is_groups_owned_global_dofs_by_field(self):
        # Each field is one contiguous block of the owned matrix rows.
        mesh = SimpleNamespace(
            mixed_spaces={0: {"field_names": ["u", "p"]}},
            step_fields={0: ["u", "p"]},
            get_step_block_offsets=lambda step: [0, 4, 6],
        )
        solver = _Harness(_build_params(LinearSolver.FIELDSPLIT), object())
        solver.physics = SimpleNamespace(mesh=mesh)
        K_mat = _build_diag_mat([1.0] * 6)
        row_offset = K_mat.getOwnershipRange()[0]

        field_iss = solver._build_fieldsplit_is(step=0, K_mat=K_mat)

        try:
            assert [name for name, _ in field_iss] == ["u", "p"]
            np.testing.assert_array_equal(field_iss[0][1].getIndices(), row_offset + np.arange(0, 4))
            np.testing.assert_array_equal(field_iss[1][1].getIndices(), row_offset + np.arange(4, 6))
        finally:
            for _, is_obj in field_iss:
                is_obj.destroy()
            K_mat.destroy()

    def test_solve_gamg_configures_once_and_reuses_operator_updates(self):
        solver = _Harness(_build_params(LinearSolver.GAMG), object())
        fake_ksp = _FakeKSP()
        workspace = SolverWorkspace(
            signature=("GAMG", (1, 1), 1, 1, "aij"),
            ksp=fake_ksp,
            du=_FakeVec(),
            rhs=_FakeVec(),
        )
        K_first = object()
        K_second = object()

        solver._solve_gamg(workspace, K_first, _FakeVec(3.0))
        solver._solve_gamg(workspace, K_second, _FakeVec(5.0))

        assert workspace.gamg_configured is True
        assert ("setType", PETSc.KSP.Type.GMRES) in fake_ksp.calls
        assert ("setType", PETSc.PC.Type.GAMG) in fake_ksp.pc.calls
        assert fake_ksp.calls.count(("setOperators", K_first)) == 1
        assert fake_ksp.calls.count(("setOperators", K_second)) == 1
        assert workspace.du.value == pytest.approx(-5.0)

    def test_solve_fieldsplit_falls_back_to_gamg_for_single_field_steps(self, monkeypatch):
        solver = _Harness(_build_params(LinearSolver.FIELDSPLIT), object())
        workspace = SolverWorkspace(
            signature=("FIELDSPLIT", (1, 1), 1, 1, "aij"),
            ksp=_FakeKSP(),
            du=_FakeVec(),
            rhs=_FakeVec(),
        )
        calls = []

        monkeypatch.setattr(solver, "_build_fieldsplit_is", lambda step, K: None)
        monkeypatch.setattr(
            solver,
            "_solve_gamg",
            lambda workspace_arg, K_arg, F_arg, step=None: calls.append((workspace_arg, K_arg, F_arg)),
        )

        solver._solve_fieldsplit(workspace, object(), _FakeVec(1.0), step=0)
        solver._solve_fieldsplit(workspace, object(), _FakeVec(2.0), step=0)

        assert workspace.fieldsplit_configured is True
        assert workspace._fieldsplit_is is None
        assert len(calls) == 2

    def test_solve_fieldsplit_configures_field_blocks_once_and_reuses_them(self, monkeypatch):
        sub_ksps = [_FakeKSP(), _FakeKSP()]
        fake_pc = _FakePC(sub_ksps=sub_ksps)
        fake_ksp = _FakeKSP(pc=fake_pc)
        solver = _Harness(_build_params(LinearSolver.FIELDSPLIT), object())
        workspace = SolverWorkspace(
            signature=("FIELDSPLIT", (2, 2), 2, 2, "aij"),
            ksp=fake_ksp,
            du=_FakeVec(),
            rhs=_FakeVec(),
        )
        u_is = object()
        p_is = object()
        K_first = object()
        K_second = object()

        monkeypatch.setattr(solver, "_build_fieldsplit_is", lambda step, K: [("u", u_is), ("p", p_is)])

        solver._solve_fieldsplit(workspace, K_first, _FakeVec(4.0), step=0)
        solver._solve_fieldsplit(workspace, K_second, _FakeVec(6.0), step=0)

        assert workspace.fieldsplit_configured is True
        assert workspace._fieldsplit_is == [u_is, p_is]
        assert fake_pc.field_splits == [("u", u_is), ("p", p_is)]
        assert ("setType", PETSc.PC.Type.FIELDSPLIT) in fake_pc.calls
        assert ("setFieldSplitType", PETSc.PC.CompositeType.ADDITIVE) in fake_pc.calls
        for sub_ksp in sub_ksps:
            assert ("setType", PETSc.KSP.Type.PREONLY) in sub_ksp.calls
            assert ("setType", PETSc.PC.Type.GAMG) in sub_ksp.pc.calls
        assert fake_ksp.calls.count(("setOperators", K_first)) == 1
        assert fake_ksp.calls.count(("setOperators", K_second)) == 1
        assert workspace.du.value == pytest.approx(-6.0)
