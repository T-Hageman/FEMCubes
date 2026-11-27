"""Tests for scalar quadrature mesh-refinement indicators."""

from __future__ import annotations

import numpy as np
import pytest

from femcubes.core.mesh.refinement import (
    FieldThresholdIndicatorConfig,
    evaluate_field_indicator,
)


def _owned_cells(mesh_wrapper) -> np.ndarray:
    topology = mesh_wrapper.mesh.topology
    return np.arange(topology.index_map(topology.dim).size_local, dtype=np.int32)


def _target_below_all_cells(mesh_wrapper) -> float:
    cells = _owned_cells(mesh_wrapper)
    diameters = mesh_wrapper.mesh.h(mesh_wrapper.mesh.topology.dim, cells)
    return 0.5 * float(np.min(diameters))


def _set_scalar_quadrature_samples(field, samples: np.ndarray) -> None:
    dofmap = field.function_space.dofmap
    assert dofmap.bs == 1
    assert dofmap.index_map_bs == 1
    assert samples.shape[1] == field.function_space.element.interpolation_points.shape[0]

    for cell in range(samples.shape[0]):
        cell_dofs = dofmap.cell_dofs(cell)
        assert cell_dofs.size == samples.shape[1]
        field.x.array[cell_dofs] = samples[cell]
    field.x.scatter_forward()


def _config(field: str, threshold: float, reduction: str = "cell_max"):
    return FieldThresholdIndicatorConfig(
        field=field,
        threshold=threshold,
        reduction=reduction,
    )


@pytest.mark.parametrize(
    ("reduction", "reducer"),
    [
        ("cell_max", np.max),
        ("cell_min", np.min),
        ("cell_mean", np.mean),
    ],
)
def test_scalar_quadrature_reductions_read_cell_dofs_directly(
    refinement_mesh,
    reduction,
    reducer,
):
    mesh = refinement_mesh
    field = mesh.Add_Quadrature_Field("driving_force", "Driving force", ())
    cells = _owned_cells(mesh)
    sample_count = field.function_space.element.interpolation_points.shape[0]
    samples = np.arange(cells.size * sample_count, dtype=float).reshape(cells.size, sample_count)
    samples = samples / max(1, sample_count) - 2.0
    _set_scalar_quadrature_samples(field, samples)

    result = evaluate_field_indicator(
        mesh,
        _config("driving_force", -1.0e20, reduction),
        _target_below_all_cells(mesh),
    )

    np.testing.assert_allclose(result.values, reducer(samples, axis=1))
    np.testing.assert_array_equal(result.cells, cells)
    assert np.all(result.marked_mask)


def test_quadrature_samples_varying_between_cells_produce_expected_marks(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Quadrature_Field("energy", "Energy", None)
    cells = _owned_cells(mesh)
    sample_count = field.function_space.element.interpolation_points.shape[0]
    samples = np.zeros((cells.size, sample_count), dtype=float)
    samples[:, :] = np.arange(cells.size, dtype=float)[:, None]
    samples[::2, -1] += 0.75
    _set_scalar_quadrature_samples(field, samples)

    result = evaluate_field_indicator(
        mesh,
        _config("energy", 3.5, "cell_max"),
        _target_below_all_cells(mesh),
    )

    expected_values = np.max(samples, axis=1)
    np.testing.assert_allclose(result.values, expected_values)
    np.testing.assert_array_equal(result.marked_mask, expected_values >= 3.5)
    assert 0 < result.marked_cells.size < cells.size


def test_psifield_quadrature_indicator_uses_owned_cell_order(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Quadrature_Field("PsiField", "Crack driving force", ())
    cells = _owned_cells(mesh)
    sample_count = field.function_space.element.interpolation_points.shape[0]
    samples = np.linspace(0.0, 2.0, cells.size * sample_count).reshape(cells.size, sample_count)
    _set_scalar_quadrature_samples(field, samples)

    result = evaluate_field_indicator(
        mesh,
        _config("PsiField", 1.0, "cell_mean"),
        _target_below_all_cells(mesh),
    )

    cell_map = mesh.mesh.topology.index_map(mesh.mesh.topology.dim)
    assert cell_map.num_ghosts == 0
    np.testing.assert_array_equal(result.cells, np.arange(cell_map.size_local, dtype=np.int32))
    np.testing.assert_allclose(result.values, np.mean(samples, axis=1))
    np.testing.assert_array_equal(result.marked_cells, result.cells[np.mean(samples, axis=1) >= 1.0])


@pytest.mark.parametrize("shape", [(2, 1), (6, 1), (6, 6)])
def test_tensor_quadrature_indicators_remain_explicitly_unsupported(refinement_mesh, shape):
    mesh = refinement_mesh
    field_name = f"tensor_{shape[0]}_{shape[1]}"
    mesh.Add_Quadrature_Field(field_name, "Tensor IP field", shape)

    with pytest.raises(ValueError, match=rf"{field_name}.*must be scalar.*{shape}"):
        evaluate_field_indicator(
            mesh,
            _config(field_name, 0.0),
            _target_below_all_cells(mesh),
        )


def test_quadrature_indicator_rejects_nonfinite_integration_point(refinement_mesh):
    mesh = refinement_mesh
    field = mesh.Add_Quadrature_Field("bad_ip", "Bad IP field", ())
    cells = _owned_cells(mesh)
    sample_count = field.function_space.element.interpolation_points.shape[0]
    samples = np.ones((cells.size, sample_count), dtype=float)
    samples[-1, -1] = np.inf
    _set_scalar_quadrature_samples(field, samples)

    with pytest.raises(ValueError, match=rf"bad_ip.*non-finite.*{int(cells[-1])}"):
        evaluate_field_indicator(
            mesh,
            _config("bad_ip", 0.0),
            _target_below_all_cells(mesh),
        )
