"""Tests for the rectangular no-phase-field strong constraint."""

import numpy as np
import pytest
from dolfinx import fem, mesh as dmesh

import femcubes.solids  # noqa: F401 - ensure model registration
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.two_squared import Sample_Parameters
from femcubes.solids.no_phase_field import No_PhaseField


def _params(model_settings=None):
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2
        Models = {"no_pf": model_settings or {}}
        Field_Orders = {"phasefield": 1}
        Solution_Steps = {"phasefield": 0}

    return Params()


def test_no_phasefield_registers_and_validates_coordinate_ranges():
    params = _params({"x_range": [1.0, 3.0], "y_range": [0.0, 2.0]})

    model = No_PhaseField("no_pf", params, object())

    assert model.MODEL_TYPE == "Solid:No_PhaseField"
    assert model.x_range == (1.0, 3.0)
    assert model.y_range == (0.0, 2.0)
    assert model.Get_Weak_Form(None, None, 0) is None


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"y_range": [0.0, 1.0]}, "requires 'x_range'"),
        ({"x_range": [0.0, 1.0]}, "requires 'y_range'"),
        ({"x_range": [1.0, 0.0], "y_range": [0.0, 1.0]}, "strictly increasing"),
        ({"x_range": [0.0, 1.0, 2.0], "y_range": [0.0, 1.0]}, "exactly two"),
    ],
)
def test_no_phasefield_rejects_invalid_ranges(settings, message):
    with pytest.raises(ValueError, match=message):
        No_PhaseField("no_pf", _params(settings), object())


def test_no_phasefield_returns_incremental_zero_bc_for_rectangle():
    params = _params({"x_range": [1.0, 3.0], "y_range": [0.0, 2.0]})
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
    mesh.Add_TestFunction("phasefield", 0)
    mesh.finalize_test_functions()
    mesh.Constant_Field("phasefield", 2.0)

    model = No_PhaseField("no_pf", params, mesh)
    bcs = model.get_bcs_for_field("phasefield")

    assert model.get_bcs_for_field("u") is None
    assert len(bcs) == 1
    assert model.mesh.Get_Field("phasefield").function_space is not None
    np.testing.assert_allclose(bcs[0].g.x.array, 2.0)

    dofs = bcs[0].dof_indices()[0]
    coordinates = mesh.fields["phasefield"].function_space.tabulate_dof_coordinates()
    selected = coordinates[dofs]
    assert selected.shape[0] > 0
    assert np.all((selected[:, 0] >= 1.0 - 1e-12) & (selected[:, 0] <= 3.0 + 1e-12))
    assert np.all((selected[:, 1] >= -1e-12) & (selected[:, 1] <= 2.0 + 1e-12))


def test_no_phasefield_supports_phasefield_in_a_mixed_solution_step():
    params = _params({"x_range": [1.0, 3.0], "y_range": [0.0, 2.0]})
    params.Field_Orders["auxiliary"] = 1
    params.Solution_Steps["auxiliary"] = 0
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
    mesh.Add_Field("auxiliary", "Auxiliary", "CG", 1)
    mesh.Add_TestFunction("phasefield", 0)
    mesh.Add_TestFunction("auxiliary", 0)
    mesh.finalize_test_functions()
    mesh.Constant_Field("phasefield", 2.0)

    model = No_PhaseField("no_pf", params, mesh)
    bcs = model.get_bcs_for_field("phasefield")

    assert len(bcs) == 1
    assert bcs[0].dof_indices()[0].size > 0
    np.testing.assert_allclose(bcs[0].g.x.array, 2.0)


def _two_squared_mesh(region, domain=None, phasefield_domain="domain"):
    """PRIVATE: Return a two-square mesh with a phase field and a No_PhaseField model."""
    params = Sample_Parameters()
    params.dx, params.dx_interface = 0.25, 0.25
    params.Models = {"no_pf": {**region, **({"domain": domain} if domain else {})}}
    params.Field_Orders = {"phasefield": 1}
    params.Solution_Steps = {"phasefield": 0}
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", 1, domain=phasefield_domain)
    mesh.Add_TestFunction("phasefield", 0)
    mesh.finalize_test_functions()
    return mesh, No_PhaseField("no_pf", params, mesh)


def _constrained_coordinates(mesh, bcs):
    """PRIVATE: Return the coordinates of the DOFs constrained by one BC."""
    coordinates = mesh.fields["phasefield"].function_space.tabulate_dof_coordinates()
    return coordinates[bcs[0].dof_indices()[0]]


def test_no_phasefield_domain_restricts_the_constrained_region():
    region = {"x_range": [0.5, 1.5], "y_range": [0.0, 1.0]}
    mesh, everywhere = _two_squared_mesh(region)
    _mesh, left_only = _two_squared_mesh(region, domain="domain:left")

    all_selected = _constrained_coordinates(mesh, everywhere.get_bcs_for_field("phasefield"))
    left_selected = _constrained_coordinates(_mesh, left_only.get_bcs_for_field("phasefield"))

    assert np.any(all_selected[:, 0] > 1.0 + 1.0e-12)
    assert left_selected.shape[0] > 0
    assert np.all((left_selected[:, 0] >= 0.5 - 1.0e-12) & (left_selected[:, 0] <= 1.0 + 1.0e-12))


def test_no_phasefield_constrains_a_phasefield_restricted_to_a_subdomain():
    mesh, model = _two_squared_mesh({"x_range": [0.5, 1.5], "y_range": [0.0, 1.0]}, phasefield_domain="domain:right")

    selected = _constrained_coordinates(mesh, model.get_bcs_for_field("phasefield"))

    assert mesh.Get_Field_Support("phasefield").domain_names == ("domain:right",)
    assert selected.shape[0] > 0
    assert np.all((selected[:, 0] >= 1.0 - 1.0e-12) & (selected[:, 0] <= 1.5))


def test_no_phasefield_rejects_a_domain_outside_the_phasefield():
    _mesh, model = _two_squared_mesh({"x_range": [0.5, 1.5], "y_range": [0.0, 1.0]}, domain="domain:left", phasefield_domain="domain:right")

    with pytest.raises(ValueError, match="not covered by the 'phasefield' field"):
        model.get_bcs_for_field("phasefield")


@pytest.mark.parametrize("mixed", [False, True])
def test_quadratic_bernstein_region_constraint_preserves_other_fields(mixed):
    """The constraint acts on P2 Bernstein coefficients alone and in a monolithic step with another field."""
    params = _params({"x_range": [1.0, 3.0], "y_range": [0.0, 2.0]})
    params.Field_Orders = {"phasefield": 2, "auxiliary": 1}
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", 2)
    mesh.Add_TestFunction("phasefield", 0)
    if mixed:
        params.Solution_Steps["auxiliary"] = 0
        mesh.Add_Field("auxiliary", "Auxiliary", "CG", 1)
        mesh.Add_TestFunction("auxiliary", 0)
    mesh.finalize_test_functions()
    mesh.Constant_Field("phasefield", 2.0)
    if mixed:
        mesh.Constant_Field("auxiliary", 3.0)

    model = No_PhaseField("no_pf", params, mesh)
    bcs = model.get_bcs_for_field("phasefield")
    assert len(bcs) == 1
    view = mesh.Get_Field_For_Output("phasefield")
    # Every field owns its storage, also when solved in one step with others.
    solution = view.field
    increment = np.zeros_like(solution.x.array)
    bcs[0].set(increment)
    solution.x.array[:] -= increment
    solution.x.scatter_forward()

    # Vertex values provide a basis-independent check of the selected region.
    probe = fem.Function(fem.functionspace(mesh.mesh, ("Lagrange", 1)))
    probe.interpolate(view.collapse_component(0))
    coordinates = probe.function_space.tabulate_dof_coordinates()
    inside = (coordinates[:, 0] >= 1.0) & (coordinates[:, 0] <= 3.0)
    np.testing.assert_allclose(probe.x.array[inside], 0.0, atol=1e-12)
    np.testing.assert_allclose(probe.x.array[~inside], 2.0, atol=1e-12)
    assert mesh.mesh.comm.allreduce(int(np.count_nonzero(inside))) > 0
    assert mesh.mesh.comm.allreduce(int(np.count_nonzero(~inside))) > 0

    # Check the field inside constrained cells, not just its vertex values:
    # zero vertex coefficients alone would leave a quadratic field nonzero.
    cells = dmesh.locate_entities(
        mesh.mesh, mesh.mesh.topology.dim,
        lambda x: (x[0] >= 1.0) & (x[0] <= 3.0) & (x[1] >= 0.0) & (x[1] <= 2.0),
    )
    midpoints = dmesh.compute_midpoints(mesh.mesh, mesh.mesh.topology.dim, cells)
    values = view.collapse_component(0).eval(midpoints, cells)
    np.testing.assert_allclose(values, 0.0, atol=1e-12)
    assert mesh.mesh.comm.allreduce(len(cells)) > 0
    if mixed:
        auxiliary = mesh.Get_Field_For_Output("auxiliary").collapse_component(0)
        np.testing.assert_allclose(auxiliary.x.array, 3.0)
