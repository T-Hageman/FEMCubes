"""Tests for the shallow-ice, shallow-ice BC and level-set models on a subdomain."""

import numpy as np
import pytest
import ufl
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc

import femcubes.glaciology  # noqa: F401 - ensure model registration
from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.two_squared import Sample_Parameters
from femcubes.core.physics.physics import Physics
from femcubes.glaciology.level_set import LevelSet
from femcubes.glaciology.shallow_ice import ShallowIce
from femcubes.glaciology.shallow_ice_bc import ShallowIceBC


def _build_left_ice_case(bc_boundary="left", level_set=False):
    """Return ``(mesh, models)`` with all shallow-ice models on ``domain:left``."""
    params = Sample_Parameters()
    params.dt = 0.1
    params.gravity = -9.81
    params.Materials = {
        "Ice": {"C_base": 1.0e-4, "Friction_Model": "Weertman", "Front_Ocean_Stress": False},
        "Water": {"Density": 1030.0, "WaterHeight": 0.0},
    }
    params.Solution_Steps = {"v": 0, "h": 1}
    params.Field_Orders = {"v": 1, "h": 0, "Temperature": 1}
    params.Models = {
        "Shallow": {"Type": "Glaciology:ShallowIce", "domain": "domain:left"},
        "Inlet": {"Type": "Glaciology:ShallowIceBC", "boundary": bc_boundary, "h": 100.0, "v_x": 1.0},
    }
    if level_set:
        params.Solution_Steps["LS"] = 2
        params.Field_Orders["LS"] = 1
        params.Level_Set = {"Transition_Width": 0.1}
        params.Models["Margin"] = {"Type": "Glaciology:LevelSet", "Velocity_Field": "v", "domain": "domain:left"}
    mesh = Mesh(params)
    models = {"Shallow": ShallowIce("Shallow", params, mesh), "Inlet": ShallowIceBC("Inlet", params, mesh)}
    if level_set:
        models["Margin"] = LevelSet("Margin", params, mesh)
    mesh.finalize_test_functions()
    return mesh, models


def _residual_sum(mesh, form) -> float:
    residual = assemble_vector(mesh.Compile_Form(form))
    residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return residual.sum()


def test_shallow_ice_fields_only_live_on_its_subdomain():
    mesh, models = _build_left_ice_case(level_set=True)

    for field_name in ("v", "vOld", "h", "hOld", "hOld2", "z", "T", "LS", "LSOld", "LS_phase"):
        assert mesh.Get_Field_Support(field_name).domain_names == ("domain:left",)
    shallow = models["Shallow"]
    for step in (shallow.step_V, shallow.step_H):
        K, F = shallow.assemble_KF(step)
        assert mesh.Compile_Form(K) is not None
        assert mesh.Compile_Form(F) is not None


def test_shallow_ice_thickness_flux_leaves_through_the_subdomain_interface():
    mesh, models = _build_left_ice_case()
    shallow = models["Shallow"]
    shallow.initialize_fields()
    mesh.Constant_Field("h", 2.0)
    mesh.Get_Field("v").x.array.reshape(-1, 2)[:, 0] = 1.0

    _K, F = shallow.assemble_KF(shallow.step_H)

    # Summing over all test functions leaves the net boundary flux of h*v:
    # inflow through "left" balances the outflow through the interface.
    assert _residual_sum(mesh, F) == pytest.approx(0.0, abs=1.0e-10)

    K_v, F_v = shallow.assemble_KF(shallow.step_V)
    assert mesh.Compile_Form(K_v) is not None
    assert mesh.Compile_Form(F_v) is not None


def test_shallow_ice_bc_acts_on_the_subdomain_thickness():
    mesh, models = _build_left_ice_case()
    models["Shallow"].initialize_fields()
    inlet = models["Inlet"]

    _K, F = inlet.assemble_KF(inlet.step_H)

    # h_penalty * (h - h_inlet) over the left face of unit length.
    assert _residual_sum(mesh, F) == pytest.approx(-100.0 * mesh.params.L, rel=1.0e-10)
    assert inlet.get_bcs_for_field("v")


def test_shallow_ice_bc_rejects_boundaries_outside_the_ice_subdomain():
    _mesh, models = _build_left_ice_case(bc_boundary="right")
    inlet = models["Inlet"]

    with pytest.raises(ValueError, match="cannot act on field 'h'"):
        inlet.assemble_KF(inlet.step_H)


def test_level_set_on_a_subdomain_assembles_and_reinitializes():
    mesh, models = _build_left_ice_case(level_set=True)
    margin = models["Margin"]
    x = ufl.SpatialCoordinate(mesh.mesh)
    mesh.Set_From_Expression("LS", 3.0 * (0.5 - x[0]))
    mesh.Constant_Field("h", 1.0)

    K, F = margin.assemble_KF(margin.step_LS)
    assert mesh.Compile_Form(K) is not None
    assert mesh.Compile_Form(F) is not None

    assert margin.reinitialize()
    LS = mesh.Get_Field("LS")
    coordinates = LS.function_space.tabulate_dof_coordinates()
    # Signed distance to the sampled zero contour x = 0.5 (no longer 3x steeper).
    np.testing.assert_allclose(LS.x.array, 0.5 - coordinates[:, 0], atol=2.0e-2)


def test_shallow_ice_lumped_thickness_terms_assemble_on_its_subdomain():
    mesh, _ = _build_left_ice_case()
    params = mesh.params
    params.ModelNames = ["Shallow", "Inlet"]
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    mesh.Constant_Field("hOld", 1.0)
    mesh.Constant_Field("h", 2.0)

    K, F = physics.Get_Kmat_Fvec(physics.models["Shallow"].step_H)

    # Lumped capacity (h - hOld) / dt over the unit-area left square plus the
    # inlet penalty (h - h_inlet) on the unit-length left face; v = 0.
    assert F.sum() == pytest.approx(1.0 / params.dt - 98.0, rel=1.0e-10)
    assert K.getSize()[0] == F.getSize()
