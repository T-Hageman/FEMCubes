"""Tests for the PrescribedValue boundary model."""

import numpy as np
import pytest

from femcubes.boundary.prescribed_value import PrescribedValue


def test_prescribed_value_uses_mesh_increment_bc_helper(
    boundary_model_mesh, BoundaryModel_Test_Params, monkeypatch
):
    params = BoundaryModel_Test_Params
    params.t = 0.0
    params.dt = 0.1
    params.Models = {
        "BC_Left": {
            "boundary": "left",
            "field": "u_y",
            "value": [1.0, 4.0],
        }
    }

    model = PrescribedValue("BC_Left", params, boundary_model_mesh)
    sentinel = [object()]
    calls = []

    def _fake_build(field_name, boundary_name, target_value, component=None):
        calls.append((field_name, boundary_name, target_value, component))
        return sentinel

    monkeypatch.setattr(boundary_model_mesh, "Build_Increment_Dirichlet_BCs", _fake_build)

    result = model.get_bcs_for_field("u")

    assert result is sentinel
    assert calls == [("u", "left", 4.0, 1)]


def test_prescribed_value_respects_t_cutoff(
    boundary_model_mesh, BoundaryModel_Test_Params
):
    params = BoundaryModel_Test_Params

    params.Models = {
        "BC_Left": {
            "boundary": "left",
            "field": "u_y",
            "value": 1.0,
            "tmax": 1.0,
            "tmin": 0.0,
        }
    }

    model = PrescribedValue("BC_Left", params, boundary_model_mesh)

    ## tmax
    params.t = 0.6
    params.dt = 0.5
    assert model.get_bcs_for_field("u") is None
    
    ## tmin
    params.t = -0.2
    params.dt = 0.1
    assert model.get_bcs_for_field("u") is None


def test_prescribed_value_constrains_a_subdomain_field_on_its_boundary(
    subdomain_boundary_mesh, SubdomainBoundary_Test_Params
):
    params = SubdomainBoundary_Test_Params
    params.Models = {"BC": {"boundary": "left", "field": "u_y", "value": 4.0, "domain": "domain:left"}}
    model = PrescribedValue("BC", params, subdomain_boundary_mesh)

    field = subdomain_boundary_mesh.Get_Field("u")
    correction = np.zeros_like(field.x.array)
    for bc in model.get_bcs_for_field("u"):
        bc.set(correction)
    updated = (field.x.array - correction).reshape(-1, 2)
    x = field.function_space.tabulate_dof_coordinates()[:, 0]

    on_left = np.isclose(x, 0.0)
    assert subdomain_boundary_mesh.mesh.comm.allreduce(np.count_nonzero(on_left)) > 0
    np.testing.assert_allclose(updated[on_left, 1], 4.0)
    np.testing.assert_allclose(updated[~on_left], 0.0)
    np.testing.assert_allclose(updated[:, 0], 0.0)


# Dirichlet BCs currently reject outside facets on the affected ranks only.
def test_prescribed_value_rejects_boundaries_outside_the_field_support(
    subdomain_boundary_mesh, SubdomainBoundary_Test_Params
):
    params = SubdomainBoundary_Test_Params
    params.Models = {"BC": {"boundary": "bottom", "field": "u_y", "value": 4.0}}
    model = PrescribedValue("BC", params, subdomain_boundary_mesh)

    with pytest.raises(ValueError, match="outside the support"):
        model.get_bcs_for_field("u")


def test_prescribed_value_domain_must_be_bounded_by_the_boundary(
    subdomain_boundary_mesh, SubdomainBoundary_Test_Params
):
    params = SubdomainBoundary_Test_Params
    params.Models = {
        "Interface": {"boundary": "interface", "field": "u", "value": 0.0, "domain": "domain:left"},
        "Wrong": {"boundary": "left", "field": "u", "value": 0.0, "domain": "domain:right"},
        "Unknown": {"boundary": "left", "field": "u", "value": 0.0, "domain": "domain:middle"},
    }

    PrescribedValue("Interface", params, subdomain_boundary_mesh)
    with pytest.raises(ValueError, match="does not bound domain 'domain:right'"):
        PrescribedValue("Wrong", params, subdomain_boundary_mesh)
    with pytest.raises(KeyError):
        PrescribedValue("Unknown", params, subdomain_boundary_mesh)
