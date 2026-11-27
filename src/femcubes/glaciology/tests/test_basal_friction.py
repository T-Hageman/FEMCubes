"""Tests for the BasalFriction glaciology model."""

import pytest
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc

from femcubes.mesh_generators.two_squared import Sample_Parameters as TwoSquared_Sample_Parameters
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.glaciology.basal_friction import BasalFriction


def _build_basal_friction_case(
    *,
    friction_model="Linear",
    damaged=False,
    solution_steps=None,
):
    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2

    params = TestParams()
    params.dt = 0.1
    params.Solution_Steps = solution_steps if solution_steps is not None else {"u": 1}
    params.Models = {
        "Friction": {
            "boundary": "left",
            "FrictionModel": friction_model,
            "DamagedForces": damaged,
        }
    }
    params.Materials = {
        "Ice": {
            "Linear_Friction_C": 5.0,
            "Weertman_C": 7.0,
            "Weertman_m": 0.5,
        }
    }

    mesh = Mesh(params)
    if "u" in params.Solution_Steps:
        mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 2)
        mesh.Add_Vector_Field("uOld", "Old Displacement", "CG", 1, 2)
        mesh.Add_Vector_TestFunction("u", 2, params.Solution_Steps["u"])
        mesh.Constant_Field("u", 0.0)
        mesh.Constant_Field("uOld", 0.0)
    if "v" in params.Solution_Steps:
        mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
        mesh.Add_Vector_TestFunction("v", 2, params.Solution_Steps["v"])
        mesh.Constant_Field("v", 0.0)
    if damaged:
        params.Solution_Steps["phasefield"] = 2
        mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
        mesh.Constant_Field("phasefield", 0.25)

    mesh.finalize_test_functions()
    model = BasalFriction("Friction", params, mesh)
    return params, mesh, model


def test_basal_friction_uses_mesh_boundary_integrator(monkeypatch):
    _, mesh, model = _build_basal_friction_case()
    original = mesh.get_Boundary_Integrator
    calls = []

    def _wrapped_integrator():
        calls.append(True)
        return original()

    monkeypatch.setattr(mesh, "get_Boundary_Integrator", _wrapped_integrator)

    model.assemble_KF(model.step)

    assert len(calls) == 1


def test_basal_friction_initializes_weertman_parameters():
    _, _, model = _build_basal_friction_case(friction_model="Weertman")

    assert model.C_Weertman == 7.0
    assert model.m_Weertman == 0.5


def test_basal_friction_requires_u_or_v_solution_step():
    with pytest.raises(ValueError, match="requires either 'u' or 'v'"):
        _build_basal_friction_case(solution_steps={"phasefield": 2})


def test_basal_friction_tracks_damage_step_when_enabled():
    params, _, model = _build_basal_friction_case(damaged=True)

    assert model.step_dam == params.Solution_Steps["phasefield"]


def test_basal_friction_returns_none_for_inactive_step():
    _, _, model = _build_basal_friction_case()

    assert model.assemble_KF(model.step + 1) is None


def test_basal_friction_weertman_and_damage_paths_assemble():
    _, _, model = _build_basal_friction_case(
        friction_model="Weertman", damaged=True
    )

    K, F = model.assemble_KF(model.step)

    assert K is not None
    assert F is not None


def test_basal_friction_invalid_model_raises_during_assembly():
    _, _, model = _build_basal_friction_case(friction_model="Unknown")

    with pytest.raises(ValueError, match="Unknown FrictionModel type"):
        model.assemble_KF(model.step)


def _build_two_domain_case(boundary):
    """Return ``(mesh, model)`` with the ice displacement on ``domain:left`` only."""
    params = TwoSquared_Sample_Parameters()
    params.dt = 0.1
    params.Solution_Steps = {"u": 0}
    params.Models = {"Friction": {"boundary": boundary, "FrictionModel": "Linear"}}
    params.Materials = {"Ice": {"Linear_Friction_C": 5.0}}
    mesh = Mesh(params)
    for name in ("u", "uOld"):
        mesh.Add_Vector_Field(name, name, "CG", 1, 2, domain="domain:left")
    mesh.Add_Vector_TestFunction("u", 2, 0)
    mesh.finalize_test_functions()
    return mesh, BasalFriction("Friction", params, mesh)


def test_basal_friction_acts_on_ice_restricted_to_a_subdomain():
    mesh, model = _build_two_domain_case("bottom:left")
    penetration = 1.0e-3
    mesh.Zero_Field("u")
    mesh.Zero_Field("uOld")
    mesh.Get_Field("u").x.array.reshape(-1, 2)[:, 1] = -penetration

    K, F = model.assemble_KF(model.step)
    residual = assemble_vector(mesh.Compile_Form(F))
    residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Only the non-penetration penalty acts (no sliding): ky * u_y over bottom:left (length L).
    assert residual.sum() == pytest.approx(-1.0e12 * penetration * mesh.params.L, rel=1.0e-10)
    assert mesh.Compile_Form(K) is not None


def test_basal_friction_rejects_boundaries_outside_the_ice_subdomain():
    _, model = _build_two_domain_case("bottom")

    with pytest.raises(ValueError, match="cannot act on field 'u'"):
        model.assemble_KF(model.step)
