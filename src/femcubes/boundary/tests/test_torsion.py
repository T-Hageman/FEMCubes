"""Tests for the AppliedTorsion boundary model."""

import numpy as np
import pytest
from dolfinx.fem.petsc import assemble_matrix

from femcubes.boundary.torsion import AppliedTorsion


def test_torsion_uses_mesh_boundary_integrator(
    boundary_model_mesh, BoundaryModel_Test_Params, monkeypatch
):
    params = BoundaryModel_Test_Params
    params.t = 0.0
    params.dt = 0.1
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "Twist": {
            "boundary": "left",
            "plane": "xy",
            "rate": 0.5,
            "Dummy": 10.0,
        }
    }

    model = AppliedTorsion("Twist", params, boundary_model_mesh)
    original = boundary_model_mesh.get_Boundary_Integrator
    calls = []

    def _wrapped_integrator():
        calls.append(True)
        return original()

    monkeypatch.setattr(boundary_model_mesh, "get_Boundary_Integrator", _wrapped_integrator)

    model.assemble_KF(model.step)

    assert len(calls) == 1


def test_torsion_acts_on_a_subdomain_field(subdomain_boundary_mesh, SubdomainBoundary_Test_Params):
    params = SubdomainBoundary_Test_Params
    params.Models = {
        "Twist": {"boundary": "left", "plane": "xy", "rate": 0.5, "Dummy": 10.0, "domain": "domain:left"},
        "Outside": {"boundary": "top", "plane": "xy", "rate": 0.5, "Dummy": 10.0},
    }

    model = AppliedTorsion("Twist", params, subdomain_boundary_mesh)
    tangent, residual = model.assemble_KF(model.step)
    stiffness = assemble_matrix(subdomain_boundary_mesh.Compile_Form(tangent))
    stiffness.assemble()

    # The penalty stiffness integrates to Dummy * |left| per component.
    total = subdomain_boundary_mesh.mesh.comm.allreduce(stiffness.getValuesCSR()[2].sum())
    assert total == pytest.approx(2 * 10.0 * 1.0)
    subdomain_boundary_mesh.Compile_Form(residual)
    with pytest.raises(ValueError, match="cannot act on field 'u'"):
        outside = AppliedTorsion("Outside", params, subdomain_boundary_mesh)
        outside.assemble_KF(outside.step)


def test_torsion_measures_torque_of_penalty_traction(boundary_model_mesh, BoundaryModel_Test_Params):
    params = BoundaryModel_Test_Params
    params.t = 0.0
    params.dt = 0.1
    params.Global_Measures = {}
    params.Solution_Steps = {"u": 1}
    params.Models = {"Twist": {"boundary": "left", "plane": "xy", "rate": 0.5, "Dummy": 10.0}}

    model = AppliedTorsion("Twist", params, boundary_model_mesh)
    model.Update_Global_Measures()

    # u = 0 on x = 0, 0 <= y <= 2: the traction is D * u_target, its torque D sin(omega) * int y^2 dy.
    omega = 0.5 * 0.1
    assert params.Global_Measures["Twist_Angle"] == pytest.approx(omega)
    assert params.Global_Measures["Twist_Torque"] == pytest.approx(10.0 * np.sin(omega) * 8.0 / 3.0)
