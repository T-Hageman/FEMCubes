"""Tests for the DisplacementControl boundary model."""

import pytest

from femcubes.boundary.displacement_control import DisplacementControl


def test_displacement_control_uses_mesh_boundary_integrator(
    boundary_model_mesh, BoundaryModel_Test_Params, monkeypatch
):
    params = BoundaryModel_Test_Params
    params.t = 0.0
    params.dt = 0.1
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "BC_Left": {
            "boundary": "left",
            "field": "u_x",
            "rate": 0.5,
            "Dummy": 10.0,
            "F_Cutoff": 0.05
        }
    }
    params.Global_Measures = {}

    model = DisplacementControl("BC_Left", params, boundary_model_mesh)
    original = boundary_model_mesh.get_Boundary_Integrator
    calls = []

    def _wrapped_integrator():
        calls.append(True)
        return original()

    monkeypatch.setattr(boundary_model_mesh, "get_Boundary_Integrator", _wrapped_integrator)

    model.assemble_KF(model.step)
    model.Update_Global_Measures()

    assert len(calls) == 2


def test_displacement_control_parses_full_field_without_component(
    boundary_model_mesh, BoundaryModel_Test_Params
):
    params = BoundaryModel_Test_Params
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "BC_Left": {
            "boundary": "left",
            "field": "u",
            "rate": 0.5,
            "Dummy": 10.0,
            "F_Cutoff": 0.05
        }
    }

    model = DisplacementControl("BC_Left", params, boundary_model_mesh)

    assert model.field_name == "u"
    assert model.component is None


def test_displacement_control_clamps_negative_time_before_penalty_force(
    boundary_model_mesh, BoundaryModel_Test_Params
):
    params = BoundaryModel_Test_Params
    params.t = -0.2
    params.dt = 0.1
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "BC_Left": {
            "boundary": "left",
            "field": "u_x",
            "rate": 0.5,
            "Dummy": 10.0,
            "F_Cutoff": 0.05
        }
    }

    model = DisplacementControl("BC_Left", params, boundary_model_mesh)
    refresh = model.Update_Weak_Constants(
        params.t,
        params.dt,
        model.step,
    )

    assert refresh is False
    assert float(model.u_target.value) == 0.0


def test_displacement_control_commit_adjusts_end_time_around_force_cutoff(
    boundary_model_mesh, BoundaryModel_Test_Params
):
    params = BoundaryModel_Test_Params
    params.t = 100.0
    params.dt = 1.0
    params.end_time = 250.0
    params.Solution_Steps = {"u": 1}
    params.Models = {
        "BC_Left": {
            "boundary": "left",
            "field": "u_x",
            "rate": 0.5,
            "Dummy": 10.0,
            "F_Cutoff": 0.2,
        }
    }
    params.Global_Measures = {
        "BC_Left_F": 1.0,
        "BC_Left_u": 0.5,
    }

    model = DisplacementControl("BC_Left", params, boundary_model_mesh)
    model.F_max = 10.0

    model.commit()

    assert model.changed_tmax
    assert model.original_tmax == 250.0
    assert params.end_time == 151.0

    params.Global_Measures["BC_Left_F"] = 5.0
    params.Global_Measures["BC_Left_u"] = 1.0

    model.commit()

    assert not model.changed_tmax
    assert params.end_time == 250.0


def test_displacement_control_acts_on_a_subdomain_field(
    subdomain_boundary_mesh, SubdomainBoundary_Test_Params
):
    params = SubdomainBoundary_Test_Params
    params.Models = {
        "BC": {"boundary": "left", "field": "u_x", "rate": 0.5, "Dummy": 10.0, "domain": "domain:left"},
        "Outside": {"boundary": "bottom", "field": "u_x", "rate": 0.5, "Dummy": 10.0},
        "Wrong": {"boundary": "left", "field": "u_x", "rate": 0.5, "Dummy": 10.0, "domain": "domain:right"},
    }

    model = DisplacementControl("BC", params, subdomain_boundary_mesh)
    model.assemble_KF(model.step)
    model.Update_Global_Measures()

    # u = 0 against the target rate * (t + dt) on the unit-length left edge.
    assert params.Global_Measures["BC_F"] == pytest.approx(10.0 * 0.5 * 0.1)

    with pytest.raises(ValueError, match="cannot act on field 'u'"):
        DisplacementControl("Outside", params, subdomain_boundary_mesh).assemble_KF(model.step)
    with pytest.raises(ValueError, match="does not bound domain"):
        DisplacementControl("Wrong", params, subdomain_boundary_mesh)
