"""
MPI integration tests for plotting and DOF utilities.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

import numpy as np


def _configure_offscreen_pyvista() -> None:
    import pyvista as pv

    pv.OFF_SCREEN = True
    original_plotter = pv.Plotter

    class TrackingPlotter(original_plotter):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs.setdefault("off_screen", True)
            super().__init__(*args, **kwargs)

        def show(self, *args, **kwargs):
            return None

        def update(self):
            return None

    pv.Plotter = TrackingPlotter


def _build_mesh():
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.params_template import ParamsTemplate

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 2.0
        Ly = 1.0
        dx = 1.0
        dim = 2

    return Mesh(TestParams())


def _run_mpi_checks() -> None:
    _configure_offscreen_pyvista()

    from femcubes.core.utils.mpi_utils import comm, rank, size

    if size < 2:
        raise RuntimeError("MPI checks require at least 2 ranks.")

    mesh = _build_mesh()
    mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
    mesh.Add_Vector_Field("v_copy", "Velocity Copy", "CG", 1, 2)
    mesh.Add_Field("p", "Pressure", "CG", 1)
    mesh.Add_Field("p_copy", "Pressure Copy", "CG", 1)

    mesh.fields["v"].x.array[:] = np.linspace(
        0.0, 1.0, mesh.fields["v"].x.array.size, dtype=float
    ) + rank
    mesh.fields["p"].x.array[:] = 2.0 + rank

    mesh.Copy_Field("v", "v_copy")
    mesh.Copy_Field("p", "p_copy")
    v_owned = mesh.fields["v"].function_space.dofmap.index_map.size_local
    p_owned = mesh.fields["p"].function_space.dofmap.index_map.size_local
    assert np.allclose(
        mesh.fields["v"].x.array[:v_owned], mesh.fields["v_copy"].x.array[:v_owned]
    )
    assert np.allclose(
        mesh.fields["p"].x.array[:p_owned], mesh.fields["p_copy"].x.array[:p_owned]
    )

    mesh.Add_Vector_TestFunction("v", nDim=2, step=0)
    mesh.Add_TestFunction("p", step=0)
    mesh.finalize_test_functions()

    mesh.Constant_Field("p", 4.0)
    mesh.Constant_Field("v", 4.0)
    mesh.Zero_Field("p")
    assert np.allclose(mesh.fields["p"].x.array, 0.0)
    assert np.allclose(mesh.fields["v"].x.array, 4.0)

    local_total_dofs = 0
    expected_global_total = 0
    for field_name in ("p", "v"):
        field = mesh.fields[field_name]
        local_base = field.function_space.dofmap.index_map.size_local
        global_base = field.function_space.dofmap.index_map.size_global
        value_shape = field.function_space.element.value_shape
        num_components = 1 if len(value_shape) == 0 else value_shape[0]
        local_total_dofs += local_base * num_components
        expected_global_total += global_base * num_components

    global_total_dofs = comm.allreduce(local_total_dofs)
    assert global_total_dofs == expected_global_total

    mesh.params.Plot_Fields = ["p", "v_x"]
    mesh.Plot_Fields()
    if rank == 0:
        assert "p" in mesh._field_plots
        assert "v_x" in mesh._field_plots
    else:
        assert mesh._field_plots == {}


@pytest.mark.mpi
def test_mpi_plotting_and_dofs():
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    try:
        import mpi4py  # noqa: F401
    except ImportError:
        pytest.skip("mpi4py not available")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.abspath(__file__)
    cmd = [mpirun, "-n", "2", sys.executable, test_path, "--mpi-check"]
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        output = "\n".join([result.stdout, result.stderr]).strip()
        raise AssertionError(f"MPI checks failed:\n{output}")


if __name__ == "__main__" and "--mpi-check" in sys.argv:
    _run_mpi_checks()
