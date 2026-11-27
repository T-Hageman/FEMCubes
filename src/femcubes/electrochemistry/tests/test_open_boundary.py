"""Tests for the OpenBoundary electrochemistry model."""

import numpy as np
import pytest

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.electrochemistry.open_boundary import OpenBoundary


def test_open_boundary_uses_mesh_increment_bc_helper(monkeypatch):
    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2

    params = TestParams()
    params.Models = {
        "Open": {
            "boundary": "left",
            "E": 0.25,
        }
    }
    params.Species = {
        "Na+": {"Charge": 1.0, "C0": 2.0},
    }

    mesh = Mesh(params)
    model = OpenBoundary("Open", params, mesh)
    sentinel = [object()]
    calls = []

    def _fake_build(field_name, boundary_name, target_value, component=None):
        calls.append((field_name, boundary_name, target_value, component))
        return sentinel

    monkeypatch.setattr(mesh, "Build_Increment_Dirichlet_BCs", _fake_build)

    result = model.get_bcs_for_field("E")

    assert result is sentinel
    assert calls == [("E", "left", 0.25, None)]


def _two_squared_open_boundary_mesh():
    """Return params and a two-squares mesh with the electrolyte fields on ``domain:right``."""
    from femcubes.mesh_generators.two_squared import Sample_Parameters

    params = Sample_Parameters()
    params.Species = {"Na+": {"Charge": 1.0, "C0": 2.0}}
    params.Models = {
        "Open": {"boundary": "right", "E": 0.25, "domain": "domain:right"},
        "Outside": {"boundary": "top", "E": 0.25},
        "Wrong": {"boundary": "right", "E": 0.25, "domain": "domain:left"},
    }
    mesh = Mesh(params)
    for name in ("Na+", "E"):
        mesh.Add_Field(name, name, "CG", 1, domain="domain:right")
        mesh.Constant_Field(name, 0.0)
    return params, mesh


def test_open_boundary_constrains_subdomain_fields_on_its_domain():
    params, mesh = _two_squared_open_boundary_mesh()

    potential = mesh.Get_Field("E")
    correction = np.zeros_like(potential.x.array)
    for bc in OpenBoundary("Open", params, mesh).get_bcs_for_field("E"):
        bc.set(correction)
    target = potential.x.array - correction
    on_right = np.isclose(potential.function_space.tabulate_dof_coordinates()[:, 0], 2.0 * params.L)
    np.testing.assert_allclose(target[on_right], 0.25)
    np.testing.assert_allclose(target[~on_right], 0.0)
    assert mesh.mesh.comm.allreduce(np.count_nonzero(on_right)) > 0

    with pytest.raises(ValueError, match="does not bound domain 'domain:left'"):
        OpenBoundary("Wrong", params, mesh)


# Dirichlet BCs currently reject outside facets on the affected ranks only.
def test_open_boundary_rejects_boundaries_outside_the_field_support():
    params, mesh = _two_squared_open_boundary_mesh()

    with pytest.raises(ValueError, match="outside the support"):
        OpenBoundary("Outside", params, mesh).get_bcs_for_field("Na+")
