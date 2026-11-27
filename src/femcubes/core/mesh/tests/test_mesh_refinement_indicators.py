"""Tests for serial CG/DG mesh-refinement indicators."""

from __future__ import annotations

from dataclasses import replace

from dolfinx import fem
import numpy as np
import pytest
import ufl

from femcubes.core.mesh.refinement import (
    FieldThresholdIndicatorConfig,
    evaluate_field_indicator,
)


def _indicator(field: str, threshold: float, **overrides) -> FieldThresholdIndicatorConfig:
    values = {"field": field, "threshold": threshold}
    values.update(overrides)
    return FieldThresholdIndicatorConfig(**values)


def _owned_cells(mesh_wrapper) -> np.ndarray:
    topology = mesh_wrapper.mesh.topology
    return np.arange(topology.index_map(topology.dim).size_local, dtype=np.int32)


def _target_below_all_cells(mesh_wrapper) -> float:
    cells = _owned_cells(mesh_wrapper)
    diameters = mesh_wrapper.mesh.h(mesh_wrapper.mesh.topology.dim, cells)
    return 0.5 * float(np.min(diameters))


def _evaluate_scalar_expression(mesh_wrapper, expression, points) -> np.ndarray:
    cells = _owned_cells(mesh_wrapper)
    values = fem.Expression(expression, points).eval(mesh_wrapper.mesh, cells)
    return np.asarray(values).reshape(cells.size, len(points), -1)[:, :, 0]


def _set_dg0_cell_values(field, values: np.ndarray) -> None:
    for cell, value in enumerate(values):
        dofs = field.function_space.dofmap.cell_dofs(cell)
        assert dofs.size == 1
        field.x.array[dofs[0]] = value
    field.x.scatter_forward()


def test_cg1_indicator_marks_expected_geometric_region(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Field("phasefield", "Phase field", "CG", 1)
    field.interpolate(lambda x: x[0])
    field.x.scatter_forward()
    config = _indicator("phasefield", 1.5)

    result = evaluate_field_indicator(mesh, config, _target_below_all_cells(mesh))

    points = field.function_space.element.interpolation_points
    x = ufl.SpatialCoordinate(mesh.mesh)
    expected_values = np.max(_evaluate_scalar_expression(mesh, x[0], points), axis=1)
    expected_mask = expected_values >= 1.5
    np.testing.assert_allclose(result.values, expected_values)
    np.testing.assert_array_equal(result.marked_mask, expected_mask)
    assert 0 < result.marked_cells.size < result.cells.size


def test_cg2_bernstein_indicator_evaluates_values_not_raw_coefficients(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Field("phasefield", "Phase field", "CG", 2)
    topology = mesh.mesh.topology
    topology.create_entities(0)
    topology.create_connectivity(0, topology.dim)
    vertex_map = topology.index_map(0)
    vertices = np.arange(vertex_map.size_local + vertex_map.num_ghosts, dtype=np.int32)
    vertex_dofs = fem.locate_dofs_topological(field.function_space, 0, vertices)

    field.x.array[:] = 1.0
    field.x.array[vertex_dofs] = 0.0
    field.x.scatter_forward()

    result = evaluate_field_indicator(
        mesh,
        _indicator("phasefield", 0.75),
        _target_below_all_cells(mesh),
    )

    raw_cell_max = np.asarray(
        [np.max(field.x.array[field.function_space.dofmap.cell_dofs(cell)]) for cell in result.cells]
    )
    points = field.function_space.element.interpolation_points
    sampled_max = np.max(_evaluate_scalar_expression(mesh, field, points), axis=1)

    np.testing.assert_allclose(raw_cell_max, 1.0)
    np.testing.assert_allclose(result.values, sampled_max)
    assert np.max(sampled_max) < 0.75
    assert not np.any(result.marked_mask)


def test_dg0_indicator_preserves_cell_local_values(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Field("driving_force", "Driving force", "DG", 0)
    values = np.linspace(-1.0, 2.0, _owned_cells(mesh).size)
    _set_dg0_cell_values(field, values)

    result = evaluate_field_indicator(
        mesh,
        _indicator("driving_force", 0.5),
        _target_below_all_cells(mesh),
    )

    np.testing.assert_allclose(result.values, values)
    np.testing.assert_array_equal(result.marked_mask, values >= 0.5)


def test_dg1_indicator_is_evaluated_cell_locally(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Field("driving_force", "Driving force", "DG", 1)
    field.interpolate(lambda x: 2.0 * x[0] - x[1])
    field.x.scatter_forward()

    result = evaluate_field_indicator(
        mesh,
        _indicator("driving_force", 2.0),
        _target_below_all_cells(mesh),
    )

    points = field.function_space.element.interpolation_points
    x = ufl.SpatialCoordinate(mesh.mesh)
    expected = np.max(_evaluate_scalar_expression(mesh, 2.0 * x[0] - x[1], points), axis=1)
    np.testing.assert_allclose(result.values, expected)
    np.testing.assert_array_equal(result.marked_mask, expected >= 2.0)


@pytest.mark.parametrize(
    ("reduction", "reducer"),
    [
        ("cell_max", np.max),
        ("cell_min", np.min),
        ("cell_mean", np.mean),
    ],
)
def test_indicator_reductions_use_cell_samples(refinement_mesh, reduction, reducer):
    mesh = refinement_mesh
    field = mesh.Add_Field("sampled", "Sampled field", "DG", 1)
    field.interpolate(lambda x: x[0] + 2.0 * x[1])
    field.x.scatter_forward()
    base = _indicator("sampled", -1.0)

    result = evaluate_field_indicator(
        mesh,
        replace(base, reduction=reduction),
        _target_below_all_cells(mesh),
    )

    points = field.function_space.element.interpolation_points
    expected = reducer(_evaluate_scalar_expression(mesh, field, points), axis=1)
    np.testing.assert_allclose(result.values, expected)


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        ("greater_equal", np.greater_equal),
        ("greater", np.greater),
        ("less_equal", np.less_equal),
        ("less", np.less),
    ],
)
def test_indicator_comparison_modes(refinement_mesh, comparison, expected):
    mesh = refinement_mesh
    field = mesh.Add_Field("cell_values", "Cell values", "DG", 0)
    values = np.arange(_owned_cells(mesh).size, dtype=float) - 2.0
    _set_dg0_cell_values(field, values)

    result = evaluate_field_indicator(
        mesh,
        _indicator("cell_values", 1.0, comparison=comparison),
        _target_below_all_cells(mesh),
    )

    np.testing.assert_array_equal(result.threshold_mask, expected(values, 1.0))
    np.testing.assert_array_equal(result.marked_mask, expected(values, 1.0))


def test_indicator_can_reduce_absolute_values(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Field("signed", "Signed field", "DG", 0)
    values = np.linspace(-2.0, 1.0, _owned_cells(mesh).size)
    _set_dg0_cell_values(field, values)

    result = evaluate_field_indicator(
        mesh,
        _indicator("signed", 1.5, use_absolute_value=True),
        _target_below_all_cells(mesh),
    )

    np.testing.assert_allclose(result.values, np.abs(values))
    np.testing.assert_array_equal(result.marked_mask, np.abs(values) >= 1.5)


def test_indicator_rejects_unknown_vector_and_tensor_fields(refinement_mesh):
    mesh = refinement_mesh

    with pytest.raises(ValueError, match="missing.*not registered"):
        evaluate_field_indicator(mesh, _indicator("missing", 0.0), 0.1)

    mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 2)
    with pytest.raises(ValueError, match=r"u.*must be scalar.*\(2,\)"):
        evaluate_field_indicator(mesh, _indicator("u", 0.0), 0.1)

    mesh.Add_Quadrature_Field("sigma", "Stress", (6, 1))
    with pytest.raises(ValueError, match=r"sigma.*must be scalar.*\(6, 1\)"):
        evaluate_field_indicator(mesh, _indicator("sigma", 0.0), 0.1)

def test_target_h_excludes_cells_that_are_already_fine(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Field("all_marked", "All marked", "DG", 0)
    _set_dg0_cell_values(field, np.ones(_owned_cells(mesh).size))
    config = _indicator("all_marked", 0.5)

    initially_marked = evaluate_field_indicator(mesh, config, _target_below_all_cells(mesh))
    filtered = evaluate_field_indicator(mesh, config, float(np.max(initially_marked.diameters)))

    assert np.all(initially_marked.threshold_mask)
    assert np.all(initially_marked.size_mask)
    assert initially_marked.marked_cells.size == initially_marked.cells.size
    assert np.all(filtered.threshold_mask)
    assert not np.any(filtered.size_mask)
    assert filtered.marked_cells.size == 0


@pytest.mark.parametrize("target_h", [0.0, -1.0, float("nan"), float("inf"), True])
def test_indicator_rejects_invalid_target_h(refinement_mesh, target_h):
    mesh = refinement_mesh
    mesh.Add_Field("phasefield", "Phase field", "CG", 1)

    with pytest.raises(ValueError, match="target_h.*greater than zero"):
        evaluate_field_indicator(mesh, _indicator("phasefield", 0.0), target_h)


def test_indicator_rejects_nonfinite_samples(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Field("bad", "Nonfinite field", "DG", 0)
    values = np.ones(_owned_cells(mesh).size)
    values[0] = np.nan
    _set_dg0_cell_values(field, values)

    with pytest.raises(ValueError, match=r"bad.*non-finite.*\[0\]"):
        evaluate_field_indicator(mesh, _indicator("bad", 0.0), _target_below_all_cells(mesh))
