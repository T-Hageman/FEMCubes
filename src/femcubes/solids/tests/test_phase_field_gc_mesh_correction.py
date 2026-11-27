"""Mesh-size correction of Gc in the phase-field model (``Gc_Mesh_Correction``)."""

import numpy as np
import pytest
from dolfinx import cpp as dcpp
from dolfinx.mesh import compute_midpoints
from mpi4py import MPI

import femcubes.boundary  # noqa: F401
import femcubes.solids  # noqa: F401
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver

ELL = 0.3
GC = 2.0
DX = 0.1


def _build_params(correction):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        MeshType: str = "Rectangle"
        dim: int = 2
        Lx: float = 4.0
        Ly: float = 1.0
        dx: float = DX

        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 917.0, "Youngs": 1.0e9, "Poisson": 0.3, "LE_Damping": 0.0,
            "ell": ELL, "Gc": GC, "PF_formulation": "AT2", "Inertia": False, "kappa": 1.0e-6,
            "f_t": 2.0e5, "f_s": 4.0e5, "CrackDrivingForce": "Stress-Based", "FailureType": "r1",
        }
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = Materials["Ice"]["Shear"] * (2.0 * Materials["Ice"]["Poisson"]) / (1.0 - 2.0 * Materials["Ice"]["Poisson"])
        gravity: float = 0.0
        Plot_Fields: list = []
        Plot_Measures: list = []
        Solution_Steps = {"u": 0, "phasefield": 1}
        Field_Orders = {"u": 1, "phasefield": 1}
        Models = {}
        ModelNames = ["Solid", "pf", "BC_Left"]
        Models["Solid"] = {"Type": "Solid:LinearElastic", "Material": "Ice", "DamageModel": "PhaseField"}
        Models["pf"] = {"Type": "Solid:PhaseField", "Material": "Ice", "CDF_field": "ElasticEnergy",
                        "Irreversible_Method": "Hist", "Gc_Mesh_Correction": correction}
        Models["BC_Left"] = {"Type": "Boundary:PrescribedValue", "boundary": "left", "field": "u_x", "value": 0.0}
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 5
        line_search: bool = False
        rel_tol: float = 1e-6
        abs_tol: float = 1e-6
        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0
        dt: float = 1.0
        Newmark_Beta: float = 0.25
        Newmark_Gamma: float = 0.5

    return Params()


def _setup(correction):
    params = _build_params(correction)
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    return params, mesh, physics


def _cell_geometry(mesh):
    msh = mesh.mesh
    tdim = msh.topology.dim
    imap = msh.topology.index_map(tdim)
    cells = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
    diameters = np.asarray(dcpp.mesh.h(msh._cpp_object, tdim, cells))
    midpoints = np.asarray(compute_midpoints(msh, tdim, cells))[:, :tdim]
    return cells, imap.size_local, diameters, midpoints


def _gc_eff_per_cell(mesh, cells):
    field = mesh.Get_Field("Gc_eff")
    return field.x.array[field.function_space.dofmap.list[cells, 0]]


def test_correction_is_off_by_default():
    params, mesh, physics = _setup(None)
    assert physics.models["pf"].gc_correction is None
    assert "Gc_eff" not in mesh.fields
    assert physics.models["pf"]._fracture_toughness_coefficient() == GC


@pytest.mark.parametrize("bad", [{"Coefficient": 0.5, "Radius": 1.0}, {"Coefficient": -1.0}, {"Source_Field": "x"}])
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        _setup(bad)


def test_prescribed_element_size_corrects_uniformly():
    c, h = 0.5, 0.05
    params, mesh, physics = _setup({"Coefficient": c, "Element_Size": h})
    pf = physics.models["pf"]
    cells, _, _, _ = _cell_geometry(mesh)
    assert np.allclose(_gc_eff_per_cell(mesh, cells), GC)  # nominal until the first assembly
    pf.Update_Weak_Constants(0.0, 1.0, pf.My_Step)
    assert np.allclose(_gc_eff_per_cell(mesh, cells), GC / (1.0 + c * h / ELL))


@pytest.mark.parametrize("reduction", ["mean", "min"])
def test_nonlocal_correction_follows_source_cells(reduction):
    c, radius = 0.4, 1.5
    params, mesh, physics = _setup({"Coefficient": c, "Source_Field": "ElasticEnergy", "Source_Threshold": 0.5,
                                    "Search_Radius": radius, "Reduction": reduction})
    pf = physics.models["pf"]
    cells, n_owned, diameters, midpoints = _cell_geometry(mesh)

    # Mark the crack core: owned cells with x < 0.4 carry a source value above the threshold.
    source = mesh.Get_Field("ElasticEnergy")
    dofs = source.function_space.dofmap.list
    core = np.flatnonzero(midpoints[:n_owned, 0] < 0.4)
    source.x.array[:] = 0.0
    source.x.array[dofs[core].ravel()] = 1.0
    source.x.scatter_forward()

    pf.Update_Weak_Constants(0.0, 1.0, pf.My_Step)
    gc_eff = _gc_eff_per_cell(mesh, cells)

    comm = MPI.COMM_WORLD
    core_points = np.concatenate(comm.allgather(midpoints[core]))
    core_sizes = np.concatenate(comm.allgather(diameters[core]))
    reduce = np.mean if reduction == "mean" else np.min
    for cell in cells:
        distance = np.linalg.norm(core_points - midpoints[cell], axis=1)
        nearby = distance <= radius * ELL
        if nearby.any():
            expected = GC / (1.0 + c * reduce(core_sizes[nearby]) / ELL)
        else:
            expected = GC
        assert np.isclose(gc_eff[cell], expected, rtol=1e-12), cell
    # Far from the core the nominal toughness is kept; near it the toughness is reduced.
    far = midpoints[:, 0] > 0.4 + radius * ELL + DX
    assert np.allclose(gc_eff[far], GC)
    assert np.all(gc_eff[core] < GC)

    # The same (time, dt, epoch) does not recompute; a new time does.
    key = pf._gc_correction_key
    pf.Update_Weak_Constants(0.0, 1.0, pf.My_Step)
    assert pf._gc_correction_key == key
    source.x.array[:] = 0.0
    pf.Update_Weak_Constants(1.0, 1.0, pf.My_Step)
    assert np.allclose(_gc_eff_per_cell(mesh, cells), GC)
