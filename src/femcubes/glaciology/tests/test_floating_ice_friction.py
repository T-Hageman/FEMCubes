"""Tests for the FloatingIceFriction glaciology model."""

import pytest
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc

import femcubes.glaciology  # noqa: F401 - ensure model registration
from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.two_squared import Sample_Parameters as TwoSquared_Sample_Parameters
from femcubes.glaciology.floating_ice_friction import FloatingIceFriction


def _build_floating_ice_friction_case(*, solution_steps=None):
    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2

    params = TestParams()
    params.dt = 0.1
    params.Solution_Steps = solution_steps if solution_steps is not None else {"u": 0}
    params.Field_Orders = {"u": 1, "v": 1}
    params.Models = {
        "Damping": {
            "h_ice": 1.0,
            "Friction": 1.0e-2,
        }
    }
    params.Materials = {"Water": {"Density": 1025.0}}

    mesh = Mesh(params)
    if "u" in params.Solution_Steps:
        mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 2)
        mesh.Add_Vector_Field("uOld", "Old Displacement", "CG", 1, 2)
        mesh.Add_Vector_TestFunction("u", 2, params.Solution_Steps["u"])
    if "v" in params.Solution_Steps:
        mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
        mesh.Add_Vector_TestFunction("v", 2, params.Solution_Steps["v"])

    mesh.finalize_test_functions()
    return params, mesh, FloatingIceFriction("Damping", params, mesh)


def test_floating_ice_friction_displacement_forms_assemble():
    _, _, model = _build_floating_ice_friction_case()

    K, F = model.assemble_KF(model.step)

    assert K is not None
    assert F is not None


def test_floating_ice_friction_velocity_forms_assemble():
    _, _, model = _build_floating_ice_friction_case(solution_steps={"v": 0})

    K, F = model.assemble_KF(model.step)

    assert K is not None
    assert F is not None


def test_floating_ice_friction_uses_public_model_type():
    assert BaseModel.get_model_class("Glaciology:FloatingIceFriction") is FloatingIceFriction


def test_floating_ice_friction_requires_displacement_or_velocity_step():
    with pytest.raises(ValueError, match="requires either 'u' or 'v'"):
        _build_floating_ice_friction_case(solution_steps={"phasefield": 0})


def test_floating_ice_friction_only_acts_on_its_subdomain():
    params = TwoSquared_Sample_Parameters()
    params.dt = 0.1
    params.Solution_Steps = {"u": 0}
    params.Field_Orders = {"u": 1}
    params.Models = {"Damping": {"h_ice": 1.0, "Friction": 1.0e-2, "domain": "domain:left"}}
    params.Materials = {"Water": {"Density": 1025.0}}
    mesh = Mesh(params)
    for name in ("u", "uOld"):
        mesh.Add_Vector_Field(name, name, "CG", 1, 2, domain="domain:left")
    mesh.Add_Vector_TestFunction("u", 2, 0)
    mesh.finalize_test_functions()
    model = FloatingIceFriction("Damping", params, mesh)
    mesh.Zero_Field("u")
    mesh.Zero_Field("uOld")
    mesh.Get_Field("u").x.array.reshape(-1, 2)[:, 0] = params.dt  # unit velocity in x

    K, F = model.assemble_KF(model.step)
    residual = assemble_vector(mesh.Compile_Form(F))
    residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # rho * C / h_ice * |v| * v integrated over the unit-area left square.
    assert residual.sum() == pytest.approx(1025.0 * 1.0e-2 * params.L**2, rel=1.0e-8)
    assert mesh.Compile_Form(K) is not None
