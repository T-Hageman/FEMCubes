"""MPI integration tests for the solver layer."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.mesh.refinement import (
    FieldThresholdIndicatorConfig,
    RefinementConfig,
)
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.step_solver import StepSolver
from femcubes.core.solvers.time_solver import TimeSolver


def _build_linear_elastic_params(
    *,
    solver_choice=LinearSolver.LU,
    use_line_search=False,
    enable_jit_warmup=False,
):
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        dx = 0.5
        dim = 2
        Lx = 4.0
        Ly = 4.0

        Materials = {}
        Materials["Ice"] = {
            "Density": 917.0,
            "Youngs": 10.0e9,
            "Poisson": 0.3,
            "LE_Damping": 1.0e6,
        }
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = (
            Materials["Ice"]["Shear"]
            * (2.0 * Materials["Ice"]["Poisson"])
            / (1.0 - 2.0 * Materials["Ice"]["Poisson"])
        )

        gravity = -9.81
        Plot_Fields = []
        Plot_Measures = []

        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 2}

        Models = {}
        ModelNames = ["Solid", "BC_Left", "BC_Bottom", "BC_Right"]
        Models["Solid"] = {
            "Type": "Solid:LinearElastic",
            "Material": "Ice",
            "DamageModel": None,
        }
        Models["BC_Left"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "left",
            "field": "u_x",
            "value": 0.0,
        }
        Models["BC_Bottom"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "u_y",
            "value": 0.0,
        }
        Models["BC_Right"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "right",
            "field": "u_x",
            "value": 0.0,
        }

        Linear_Solver = solver_choice
        n_passes = 1
        max_it = 16
        line_search = use_line_search
        line_search_lims = [0.1, 1.0]
        rel_tol = 1.0e-7
        abs_tol = 1.0e-10
        GMRES_Max_Iter = 200
        GMRES_RelTol = 1.0e-8

        output_dir = None
        start_time = 0.0
        end_time = 0.5
        dt = 0.25
        Newmark_Beta = 0.25
        Newmark_Gamma = 0.5
        JIT_WARMUP = enable_jit_warmup

    return Params()


def _build_stokes_params():
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        dx = 0.5
        dim = 2
        Lx = 4.0
        Ly = 4.0

        Materials = {}
        Materials["Ice"] = {
            "Density": 917.0,
            "Bulk": 10.0e9,
            "Visc_Type": "Newtonian",
            "Viscosity": 1.0e13,
            "Glen_n": 3.0,
            "Glen_A": 1.0e-24,
            "visc_max": 1.0e20,
            "visc_min": 1.0e4,
        }

        gravity = -9.81
        Plot_Fields = []
        Plot_Measures = []

        Solution_Steps = {"v": 0, "p": 0}
        Field_Orders = {"v": 2, "p": 1}

        Models = {}
        ModelNames = ["Solid", "BC_Left", "BC_Bottom", "BC_Right"]
        Models["Solid"] = {
            "Type": "Solid:Stokes",
            "Material": "Ice",
            "DamageModel": None,
        }
        Models["BC_Left"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "left",
            "field": "v_x",
            "value": 0.0,
        }
        Models["BC_Bottom"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "v_y",
            "value": 0.0,
        }
        Models["BC_Right"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "right",
            "field": "v_x",
            "value": 0.0,
        }

        Linear_Solver = LinearSolver.LU
        n_passes = 1
        max_it = 16
        line_search = False
        line_search_lims = [0.1, 1.0]
        rel_tol = 1.0e-7
        abs_tol = 1.0e-10

        output_dir = None
        start_time = 0.0
        end_time = 0.25
        dt = 0.25
        JIT_WARMUP = False

    return Params()


def _owned_entries(field) -> int:
    value_shape = field.function_space.element.value_shape
    ncomp = 1 if len(value_shape) == 0 else int(np.prod(value_shape))
    return field.function_space.dofmap.index_map.size_local * ncomp


def _owned_abs_max(field) -> float:
    owned = _owned_entries(field)
    if owned == 0:
        return 0.0
    return float(np.max(np.abs(field.x.array[:owned])))


def _assert_field_owned_entries_finite(field) -> None:
    owned = _owned_entries(field)
    assert np.all(np.isfinite(field.x.array[:owned]))


def _run_mpi_linear_elastic_step_lu() -> None:
    from mpi4py import MPI

    from femcubes.core.utils.mpi_utils import comm, size, isolate_jit_cache

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("Solver MPI checks require at least 2 ranks.")

    params = _build_linear_elastic_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    _, initial_F = physics.Get_Kmat_Fvec(0)
    initial_norm = initial_F.norm()

    solver = StepSolver(params, physics)
    solver.Solve_Step()

    _, final_F = physics.Get_Kmat_Fvec(0)
    final_norm = final_F.norm()

    for field_name in ("u", "v", "a"):
        _assert_field_owned_entries_finite(mesh.fields[field_name])

    global_u_max = comm.allreduce(_owned_abs_max(mesh.fields["u"]), op=MPI.MAX)
    assert global_u_max > 0.0
    assert np.isfinite(initial_norm)
    assert np.isfinite(final_norm)
    assert final_norm < initial_norm


def _run_mpi_stokes_step_lu() -> None:
    from mpi4py import MPI

    from femcubes.core.utils.mpi_utils import comm, size, isolate_jit_cache

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("Solver MPI checks require at least 2 ranks.")

    params = _build_stokes_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    _, initial_F = physics.Get_Kmat_Fvec(0)
    initial_norm = initial_F.norm()

    solver = StepSolver(params, physics)
    solver.Solve_Step()

    _, final_F = physics.Get_Kmat_Fvec(0)
    final_norm = final_F.norm()

    assert mesh.mixed_spaces[0] is not None
    for field_name in ("v", "p"):
        _assert_field_owned_entries_finite(mesh.fields[field_name])

    global_pressure_max = comm.allreduce(_owned_abs_max(mesh.fields["p"]), op=MPI.MAX)
    assert global_pressure_max > 0.0
    assert np.isfinite(initial_norm)
    assert np.isfinite(final_norm)
    assert final_norm < initial_norm


def _run_mpi_linear_elastic_step_gmres_linesearch() -> None:
    from mpi4py import MPI

    from femcubes.core.utils.mpi_utils import comm, size, isolate_jit_cache

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("Solver MPI checks require at least 2 ranks.")

    params = _build_linear_elastic_params(solver_choice=LinearSolver.GMRES, use_line_search=True)
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    _, initial_F = physics.Get_Kmat_Fvec(0)
    initial_norm = initial_F.norm()

    solver = StepSolver(params, physics)
    solver.Solve_Step()

    _, final_F = physics.Get_Kmat_Fvec(0)
    final_norm = final_F.norm()

    _assert_field_owned_entries_finite(mesh.fields["u"])
    global_u_max = comm.allreduce(_owned_abs_max(mesh.fields["u"]), op=MPI.MAX)
    assert global_u_max > 0.0
    assert np.isfinite(initial_norm)
    assert np.isfinite(final_norm)
    assert final_norm < initial_norm


def _run_mpi_time_solver_linear_elastic() -> None:
    from mpi4py import MPI

    from femcubes.core.utils.mpi_utils import comm, size, isolate_jit_cache

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("Solver MPI checks require at least 2 ranks.")

    params = _build_linear_elastic_params(enable_jit_warmup=True)
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    solver = TimeSolver(params, physics)
    solver.run()

    assert params.t == pytest.approx(params.end_time)
    assert solver.t == pytest.approx(params.end_time)

    for field_name, old_name in (("u", "uOld"), ("v", "vOld"), ("a", "aOld")):
        owned = _owned_entries(mesh.fields[field_name])
        np.testing.assert_allclose(
            mesh.fields[field_name].x.array[:owned],
            mesh.fields[old_name].x.array[:owned],
        )

    global_uold_max = comm.allreduce(_owned_abs_max(mesh.fields["uOld"]), op=MPI.MAX)
    assert global_uold_max > 0.0

    gathered_times = comm.allgather(params.t)
    assert all(time_value == pytest.approx(params.end_time) for time_value in gathered_times)


def _run_mpi_time_solver_linear_elastic_refinement() -> None:
    from femcubes.core.utils.mpi_utils import comm, size, isolate_jit_cache

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("Solver MPI checks require at least 2 ranks.")

    params = _build_linear_elastic_params()
    params.end_time = 0.25
    params.dt = 0.25
    params.abs_tol = 2.0e-10
    params.MeshRefinement = RefinementConfig(
        enabled=True,
        indicator=FieldThresholdIndicatorConfig(
            field="indicator",
            threshold=0.5,
        ),
        target_h=0.4,
        buffer_layers=0,
        max_levels_per_step=1,
        max_events=1,
    )
    mesh = Mesh(params)
    mesh.Add_Field("indicator", "Refinement indicator", "DG", 0)
    mesh.Constant_Field("indicator", 1.0)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    class TrackingTimeSolver(TimeSolver):
        def __init__(self, solver_params, solver_physics):
            super().__init__(solver_params, solver_physics)
            self.solve_epochs = []

        def Solve_Step(self):
            self.solve_epochs.append(self.physics.mesh.remesh_epoch)
            return super().Solve_Step()

    solver = TrackingTimeSolver(params, physics)
    solver.run()

    assert solver.solve_epochs == [0, 1]
    assert mesh.remesh_epoch == 1
    assert solver.refinement_event_count == 1
    assert solver.last_retry_reason == "accept"
    assert solver.t == pytest.approx(params.end_time)
    for field_name, old_name in (("u", "uOld"), ("v", "vOld"), ("a", "aOld")):
        owned = _owned_entries(mesh.fields[field_name])
        np.testing.assert_allclose(
            mesh.fields[field_name].x.array[:owned],
            mesh.fields[old_name].x.array[:owned],
        )

    lifecycle = (
        tuple(solver.solve_epochs),
        solver.refinement_event_count,
        mesh.remesh_epoch,
        solver.last_retry_reason,
    )
    gathered = comm.allgather(lifecycle)
    assert all(candidate == gathered[0] for candidate in gathered)


def _run_mpi_case(case_name: str) -> None:
    cases = {
        "linear-elastic-step-lu": _run_mpi_linear_elastic_step_lu,
        "stokes-step-lu": _run_mpi_stokes_step_lu,
        "linear-elastic-step-gmres-linesearch": _run_mpi_linear_elastic_step_gmres_linesearch,
        "time-linear-elastic": _run_mpi_time_solver_linear_elastic,
        "time-linear-elastic-refinement": _run_mpi_time_solver_linear_elastic_refinement,
    }
    if case_name not in cases:
        raise ValueError(f"Unknown MPI solver case '{case_name}'.")
    cases[case_name]()


def _run_mpi_subprocess(case_name: str) -> None:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    try:
        import mpi4py  # noqa: F401
    except ImportError:
        pytest.skip("mpi4py not available")

    project_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) if not pythonpath else f"{project_root}{os.pathsep}{pythonpath}"

    cmd = [mpirun, "-n", "2", sys.executable, __file__, "--mpi-check", case_name]
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        output = "\n".join([result.stdout, result.stderr]).strip()
        raise AssertionError(f"MPI solver check '{case_name}' failed:\n{output}")


@pytest.mark.mpi
def test_mpi_step_solver_linear_elastic_lu():
    """Distributed LU solves should assemble and reduce the elastic residual."""
    _run_mpi_subprocess("linear-elastic-step-lu")


@pytest.mark.mpi
def test_mpi_step_solver_stokes_mixed_lu():
    """Distributed mixed Stokes solves should update the pressure field and reduce the residual."""
    _run_mpi_subprocess("stokes-step-lu")


@pytest.mark.mpi
def test_mpi_step_solver_linear_elastic_gmres_linesearch():
    """Distributed GMRES/ASM solves should work with the line-search path enabled."""
    _run_mpi_subprocess("linear-elastic-step-gmres-linesearch")


@pytest.mark.mpi
def test_mpi_time_solver_linear_elastic_run():
    """A real MPI time march should warm up JIT, advance time, and commit fields."""
    _run_mpi_subprocess("time-linear-elastic")


@pytest.mark.mpi
def test_mpi_time_solver_refinement_retry_agrees_across_ranks():
    """All ranks should install one epoch and re-solve the same increment."""
    _run_mpi_subprocess("time-linear-elastic-refinement")


if __name__ == "__main__" and "--mpi-check" in sys.argv:
    case_index = sys.argv.index("--mpi-check") + 1
    if case_index >= len(sys.argv):
        raise SystemExit("Missing MPI case name.")
    _run_mpi_case(sys.argv[case_index])
