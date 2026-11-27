"""Tests for the ExternalForce boundary model."""

import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from femcubes.boundary.external_force import ExternalForce


def test_external_force_uses_mesh_boundary_integrator(
    boundary_model_mesh, BoundaryModel_Test_Params, monkeypatch
):
    params = BoundaryModel_Test_Params
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "Load": {
            "boundary": "left",
            "field": "u_x",
            "value": 3.0,
        }
    }

    model = ExternalForce("Load", params, boundary_model_mesh)
    original = boundary_model_mesh.get_Boundary_Integrator
    calls = []

    def _wrapped_integrator():
        calls.append(True)
        return original()

    monkeypatch.setattr(boundary_model_mesh, "get_Boundary_Integrator", _wrapped_integrator)

    model.assemble_KF(model.step)

    assert len(calls) == 1


def test_external_force_parses_full_field_without_component(
    boundary_model_mesh, BoundaryModel_Test_Params
):
    params = BoundaryModel_Test_Params
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "Load": {
            "boundary": "left",
            "field": "u",
            "value": 3.0,
        }
    }

    model = ExternalForce("Load", params, boundary_model_mesh)

    assert model.field_name == "u"
    assert model.component is None


def test_external_force_returns_none_for_inactive_step(
    boundary_model_mesh, BoundaryModel_Test_Params
):
    params = BoundaryModel_Test_Params
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "Load": {
            "boundary": "left",
            "field": "u_x",
            "value": 3.0,
        }
    }

    model = ExternalForce("Load", params, boundary_model_mesh)

    assert model.assemble_KF(model.step + 1) is None
    
def test_external_force_returns_none_for_time_limits(
    boundary_model_mesh, BoundaryModel_Test_Params
):
    params = BoundaryModel_Test_Params
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "Load": {
            "boundary": "left",
            "field": "u_x",
            "value": 3.0,
            "tmin": 0.5,
            "tmax": 1.5
        }
    }

    model = ExternalForce("Load", params, boundary_model_mesh)
    
    # Test before tmin
    params.t = 0.0
    params.dt = 0.1
    assert model.assemble_KF(model.step) is None
    
    # Test after tmax
    params.t = 2.0
    params.dt = 0.1
    assert model.assemble_KF(model.step) is None


def test_external_force_requests_form_refresh_when_time_window_changes(
    boundary_model_mesh, BoundaryModel_Test_Params
):
    params = BoundaryModel_Test_Params
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "Load": {
            "boundary": "left",
            "field": "u_x",
            "value": 3.0,
            "tmin": 0.5,
        }
    }
    model = ExternalForce("Load", params, boundary_model_mesh)

    assert model.Update_Weak_Constants(0.0, 0.1, model.step) is False
    assert model.Get_Weak_Form(0.0, 0.1, model.step) is None
    assert model.Update_Weak_Constants(0.5, 0.1, model.step) is True
    assert model.Get_Weak_Form(0.5, 0.1, model.step) is not None
    assert model.Update_Weak_Constants(0.6, 0.1, model.step) is False
    


def _external_work(mesh, residual):
    """Return the work of a residual on a unit x-displacement of ``u``."""
    probe = fem.Function(mesh.Get_Field("u").function_space)
    probe.x.array[0::2] = 1.0
    local = fem.assemble_scalar(mesh.Compile_Form(ufl.action(residual, probe)))
    return mesh.mesh.comm.allreduce(local, op=MPI.SUM)


def test_external_force_acts_on_a_subdomain_field(
    subdomain_boundary_mesh, SubdomainBoundary_Test_Params
):
    params = SubdomainBoundary_Test_Params
    params.Models = {
        "Load": {"boundary": "left", "field": "u_x", "value": 3.0, "domain": "domain:left"},
        "Outside": {"boundary": "top", "field": "u_x", "value": 3.0},
    }

    model = ExternalForce("Load", params, subdomain_boundary_mesh)
    _, residual = model.assemble_KF(model.step)

    assert _external_work(subdomain_boundary_mesh, residual) == pytest.approx(-3.0)
    with pytest.raises(ValueError, match="cannot act on field 'u'"):
        outside = ExternalForce("Outside", params, subdomain_boundary_mesh)
        outside.assemble_KF(outside.step)
