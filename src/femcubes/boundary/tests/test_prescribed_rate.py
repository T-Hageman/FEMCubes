"""Absolute ramp targets use the existing incremental Dirichlet machinery."""

import numpy as np
import pytest
from dolfinx import fem

from femcubes.boundary.prescribed_rate import PrescribedRate
from femcubes.boundary.prescribed_value import PrescribedValue
from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.rectangle import Sample_Parameters


def _updated_field(mesh, model, field_name):
    field = mesh.Get_Field(field_name)
    correction = np.zeros_like(field.x.array)
    for bc in model.get_bcs_for_field(field_name):
        bc.set(correction)
    return (field.x.array-correction).reshape(-1, field.function_space.dofmap.bs)


@pytest.mark.parametrize("field, value, rate, expected", [
    ("velocity", 1.0, 2.0, [6.0]),
    ("u_x", 1.0, 2.0, [6.0, 0.25]),
    ("u_y", 1.0, -2.0, [0.25, -4.0]),
    ("u_y", [1.0, 4.0], [2.0, -1.0], [0.25, 1.5]),
    ("u_y", 1.0, [2.0, -1.0], [0.25, -1.5]),
    ("u_y", [1.0, 4.0], 2.0, [0.25, 9.0]),
    ("u", [1.0, 4.0], [2.0, -1.0], [6.0, 1.5]),
    ("u", 1.0, [2.0, -1.0], [6.0, -1.5]),
    ("u", [1.0, 4.0], 2.0, [6.0, 9.0]),
    ("u", 1.0, 2.0, [6.0, 6.0]),
])
def test_ramp_constrains_scalar_vector_and_component_fields(
    boundary_model_mesh, BoundaryModel_Test_Params, field, value, rate, expected,
):
    params = BoundaryModel_Test_Params
    params.t, params.dt = 2.0, 0.5
    params.Models = {"BC": {"boundary": "left", "field": field, "value": value, "rate": rate}}
    model = PrescribedRate("BC", params, boundary_model_mesh)
    field_name = model.field_name
    boundary_model_mesh.Constant_Field(field_name, 0.25)
    updated = _updated_field(boundary_model_mesh, model, field_name)
    space = boundary_model_mesh.Get_Field(field_name).function_space
    dofs = fem.locate_dofs_topological(space, 1, boundary_model_mesh.get_boundary_facets("left"))
    np.testing.assert_allclose(updated[dofs], np.broadcast_to(expected, updated[dofs].shape))
    other_dofs = np.setdiff1d(np.arange(updated.shape[0]), dofs)
    np.testing.assert_allclose(updated[other_dofs], 0.25)
    assert model.get_bcs_for_field("unrelated") is None
    assert model.Get_Weak_Form(params.t, params.dt, 1) is None


def test_repeated_iterations_and_retries_recompute_absolute_target(
    boundary_model_mesh, BoundaryModel_Test_Params,
):
    params = BoundaryModel_Test_Params
    params.Models = {"BC": {"boundary": "left", "field": "velocity", "value": 3.0, "rate": 2.0}}
    model = PrescribedRate("BC", params, boundary_model_mesh)
    space = boundary_model_mesh.Get_Field("velocity").function_space
    dofs = fem.locate_dofs_topological(space, 1, boundary_model_mesh.get_boundary_facets("left"))
    for time, dt, current in [(1.0, 1.0, 0.0), (1.0, 1.0, 7.0), (1.0, 0.25, 3.0), (-1.0, 0.25, 0.0)]:
        params.t, params.dt = time, dt
        boundary_model_mesh.Constant_Field("velocity", current)
        updated = _updated_field(boundary_model_mesh, model, "velocity")
        np.testing.assert_allclose(updated[dofs], 3.0+2.0*(time+dt))


def test_start_time_fallback_uses_absolute_time_origin(boundary_model_mesh, BoundaryModel_Test_Params):
    params = BoundaryModel_Test_Params
    params.start_time, params.dt = 2.0, 0.5
    params.Models = {"BC": {"boundary": "left", "field": "velocity", "value": 3.0, "rate": 2.0}}
    model = PrescribedRate("BC", params, boundary_model_mesh)
    updated = _updated_field(boundary_model_mesh, model, "velocity")
    space = boundary_model_mesh.Get_Field("velocity").function_space
    dofs = fem.locate_dofs_topological(space, 1, boundary_model_mesh.get_boundary_facets("left"))
    np.testing.assert_allclose(updated[dofs], 8.0)


@pytest.mark.parametrize("time, dt, active", [
    (0.5, 1.0, False), (1.0, 0.5, True), (2.5, 0.5, True), (2.5, 0.6, False),
])
def test_activation_window_matches_prescribed_value(
    boundary_model_mesh, BoundaryModel_Test_Params, time, dt, active,
):
    params = BoundaryModel_Test_Params
    params.t, params.dt = time, dt
    params.Models = {"BC": {"boundary": "left", "field": "velocity", "value": 3.0, "rate": 2.0,
                             "tmin": 1.0, "tmax": 3.0}}
    model = PrescribedRate("BC", params, boundary_model_mesh)
    assert bool(model.get_bcs_for_field("velocity")) == active
    assert model.Update_Weak_Constants(time, dt, 1) is False
    # Activation changes use the inherited weak-form refresh contract.
    assert model.Update_Weak_Constants(1.0 if not active else 0.0, dt, 1) is True


@pytest.mark.parametrize("value, rate", [
    (np.nan, 1.0), (0.0, np.inf), ([], 1.0), (0.0, []),
    ([[1.0]], 1.0), (0.0, [[1.0]]), ([0.0, 1.0], [1.0]),
])
def test_invalid_ramp_settings_are_rejected(boundary_model_mesh, BoundaryModel_Test_Params, value, rate):
    params = BoundaryModel_Test_Params
    params.Models = {"BC": {"boundary": "left", "field": "u", "value": value, "rate": rate}}
    with pytest.raises(ValueError, match="PrescribedRate"):
        PrescribedRate("BC", params, boundary_model_mesh)


@pytest.mark.parametrize("field, value, rate", [
    ("velocity", [1.0, 2.0], 1.0), ("u", [1.0], 1.0),
    ("u_y", 0.0, [1.0]),
])
def test_component_vectors_must_match_the_underlying_field(
    boundary_model_mesh, BoundaryModel_Test_Params, field, value, rate,
):
    params = BoundaryModel_Test_Params
    params.t, params.dt = 1.0, 0.5
    params.Models = {"BC": {"boundary": "left", "field": field, "value": value, "rate": rate}}
    model = PrescribedRate("BC", params, boundary_model_mesh)
    with pytest.raises(ValueError, match="component entries"):
        model.get_bcs_for_field(model.field_name)


@pytest.mark.parametrize("model_class", [PrescribedValue, PrescribedRate])
def test_three_dimensional_z_component(model_class):
    params = Sample_Parameters()
    params.dim, params.Lx, params.Ly, params.Lz, params.dx = 3, 1.0, 1.0, 1.0, 1.0
    params.t, params.dt = 1.0, 0.5
    params.Models = {"BC": {"boundary": "top", "field": "u_z", "value": [1.0, 2.0, 3.0],
                             "rate": [1.0, 2.0, 3.0]}}
    mesh = Mesh(params)
    mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 3)
    mesh.Constant_Field("u", 0.25)
    model = model_class("BC", params, mesh)
    updated = _updated_field(mesh, model, "u")
    dofs = fem.locate_dofs_topological(mesh.Get_Field("u").function_space, 2, mesh.get_boundary_facets("top"))
    expected = [0.25, 0.25, 7.5 if model_class is PrescribedRate else 3.0]
    np.testing.assert_allclose(updated[dofs], np.broadcast_to(expected, updated[dofs].shape))


@pytest.mark.parametrize("model_class", [PrescribedValue, PrescribedRate])
def test_component_like_text_inside_a_field_name_is_not_a_suffix(
    boundary_model_mesh, BoundaryModel_Test_Params, model_class,
):
    params = BoundaryModel_Test_Params
    params.t, params.dt = 1.0, 0.5
    params.Models = {"BC": {"boundary": "left", "field": "velocity_x_mean", "value": 1.0, "rate": 2.0}}
    boundary_model_mesh.Add_Field("velocity_x_mean", "Mean velocity", "CG", 1)
    model = model_class("BC", params, boundary_model_mesh)
    assert model.field_name == "velocity_x_mean"
    assert model.component is None
    assert model.get_bcs_for_field("velocity_x_mean")
