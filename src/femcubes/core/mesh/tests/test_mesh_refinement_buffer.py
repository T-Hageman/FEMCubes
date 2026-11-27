"""Physical refinement padding is independent of the current cell layers."""

from types import SimpleNamespace

from dolfinx import mesh as dmesh
from mpi4py import MPI
import numpy as np
import pytest

from femcubes.core.mesh.refinement import CellIndicatorResult, expand_refinement_buffer


def _indicator_result(mesh_wrapper, marked_mask, size_mask=None):
    topology = mesh_wrapper.mesh.topology
    cells = np.arange(topology.index_map(topology.dim).size_local, dtype=np.int32)
    marked_mask = np.asarray(marked_mask, dtype=bool)
    if size_mask is None:
        size_mask = np.ones(cells.size, dtype=bool)
    else:
        size_mask = np.asarray(size_mask, dtype=bool)
    diameters = mesh_wrapper.mesh.h(topology.dim, cells)
    return CellIndicatorResult(
        cells=cells,
        values=marked_mask.astype(float),
        diameters=diameters,
        threshold_mask=marked_mask.copy(),
        size_mask=size_mask,
        marked_mask=marked_mask & size_mask,
    )



def _rectangle(n):
    return SimpleNamespace(mesh=dmesh.create_unit_square(MPI.COMM_WORLD, n, n))


def _cell_points(wrapper):
    mesh = wrapper.mesh
    cells = np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32)
    return mesh.geometry.x[dmesh.entities_to_geometry(mesh, mesh.topology.dim, cells, False)]


@pytest.mark.parametrize("n", [4, 8, 16, 32])
@pytest.mark.parametrize("layers", [0, 1, 4, 8])
def test_padding_is_a_fixed_physical_distance_on_different_starting_meshes(n, layers):
    wrapper = _rectangle(n)
    points = _cell_points(wrapper)
    left, right = points[:, :, 0].min(axis=1), points[:, :, 0].max(axis=1)
    threshold = right <= 0.25+1e-12
    target_h = 0.02
    selection = expand_refinement_buffer(wrapper, _indicator_result(wrapper, threshold), layers, target_h=target_h)
    expected = threshold.copy() if layers == 0 else left <= 0.25+layers*target_h+1e-12
    np.testing.assert_array_equal(selection.marked_mask, expected)
    assert selection.global_marked_count == np.count_nonzero(expected)
    assert selection.local_initial_count == np.count_nonzero(threshold)
    assert np.max(right[selection.marked_mask]) <= 0.25+layers*target_h+1/n+1e-12


@pytest.mark.parametrize("target_h", [0.01, 0.02, 0.04])
def test_target_h_controls_the_padding_width(target_h):
    wrapper = _rectangle(16)
    points = _cell_points(wrapper)
    threshold = points[:, :, 0].max(axis=1) <= 0.25+1e-12
    selection = expand_refinement_buffer(wrapper, _indicator_result(wrapper, threshold), 4, target_h=target_h)
    expected = points[:, :, 0].min(axis=1) <= 0.25+4*target_h+1e-12
    np.testing.assert_array_equal(selection.marked_mask, expected)


def test_three_dimensional_padding_uses_the_same_physical_radius():
    wrapper = SimpleNamespace(mesh=dmesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4))
    points = _cell_points(wrapper)
    threshold = points[:, :, 0].max(axis=1) <= 0.25+1e-12
    selection = expand_refinement_buffer(wrapper, _indicator_result(wrapper, threshold), 2, target_h=0.04)
    expected = points[:, :, 0].min(axis=1) <= 0.33+1e-12
    np.testing.assert_array_equal(selection.marked_mask, expected)


def test_buffer_uses_euclidean_distance_rather_than_an_axis_aligned_box():
    wrapper = _rectangle(32)
    points = _cell_points(wrapper)[:, :, :2]
    lower, upper = points.min(axis=1), points.max(axis=1)
    threshold = np.all(upper <= 0.25+1e-12, axis=1)
    selection = expand_refinement_buffer(wrapper, _indicator_result(wrapper, threshold), 5, target_h=0.02)
    # For triangles containing their lower box corner, its distance to the
    # square criterion region is the exact simplex-to-region distance.
    has_lower_corner = np.any(np.all(np.abs(points-lower[:, None, :]) < 1e-12, axis=2), axis=1)
    distance = np.linalg.norm(np.maximum(lower-0.25, 0), axis=1)
    np.testing.assert_array_equal(selection.marked_mask[has_lower_corner], distance[has_lower_corner] <= 0.1+1e-12)
    bbox_only = has_lower_corner & np.all(lower <= 0.35, axis=1) & (distance > 0.1+1e-12)
    assert np.any(bbox_only)
    assert not np.any(selection.marked_mask[bbox_only])


def test_buffer_seeds_and_crosses_cells_already_at_target_size():
    wrapper = _rectangle(16)
    points = _cell_points(wrapper)
    lower, upper = points[:, :, 0].min(axis=1), points[:, :, 0].max(axis=1)
    threshold = upper <= 0.25+1e-12
    eligible = lower >= 0.375-1e-12
    selection = expand_refinement_buffer(wrapper, _indicator_result(wrapper, threshold, eligible), 8, target_h=0.02)
    expected = (lower <= 0.41+1e-12) & eligible
    np.testing.assert_array_equal(selection.marked_mask, expected)
    np.testing.assert_array_equal(selection.added_by_buffer_mask, expected)
    assert selection.local_initial_count == 0
    assert selection.any_marked


def test_zero_buffer_preserves_initial_selection_and_size_filter(refinement_mesh):
    count = refinement_mesh.mesh.topology.index_map(2).size_local
    threshold = np.arange(count) % 3 == 0
    eligible = np.arange(count) % 2 == 0
    selection = expand_refinement_buffer(refinement_mesh, _indicator_result(refinement_mesh, threshold, eligible),
                                         0, target_h=0.1)
    np.testing.assert_array_equal(selection.marked_mask, threshold & eligible)
    assert not np.any(selection.added_by_buffer_mask)


def test_empty_selection_produces_collective_noop_decision(refinement_mesh):
    mesh = refinement_mesh
    cell_count = mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_local
    result = _indicator_result(mesh, np.zeros(cell_count, dtype=bool))

    selection = expand_refinement_buffer(mesh, result, buffer_layers=2, target_h=0.1)

    assert selection.global_initial_count == 0
    assert selection.global_marked_count == 0
    assert selection.any_marked is False
    assert selection.marked_cells.size == 0


@pytest.mark.parametrize("buffer_layers", [-1, 1.5, True])
def test_buffer_rejects_invalid_layer_counts_collectively(refinement_mesh, buffer_layers):
    mesh = refinement_mesh
    cell_count = mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_local
    result = _indicator_result(mesh, np.zeros(cell_count, dtype=bool))

    with pytest.raises(ValueError, match="buffer_layers must be a nonnegative integer"):
        expand_refinement_buffer(mesh, result, buffer_layers, target_h=0.1)


def test_buffer_rejects_incomplete_owned_cell_ordering(refinement_mesh):
    mesh = refinement_mesh
    cell_count = mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_local
    result = _indicator_result(mesh, np.zeros(cell_count, dtype=bool))
    incomplete = CellIndicatorResult(
        cells=result.cells[:-1],
        values=result.values[:-1],
        diameters=result.diameters[:-1],
        threshold_mask=result.threshold_mask[:-1],
        size_mask=result.size_mask[:-1],
        marked_mask=result.marked_mask[:-1],
    )

    with pytest.raises(ValueError, match="complete owned local cell ordering"):
        expand_refinement_buffer(mesh, incomplete, buffer_layers=0, target_h=0.1)


def test_buffer_rejects_inconsistent_indicator_masks(refinement_mesh):
    mesh = refinement_mesh
    cell_count = mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_local
    result = _indicator_result(mesh, np.zeros(cell_count, dtype=bool))
    result.marked_mask[0] = True

    with pytest.raises(ValueError, match=r"marked_mask must equal threshold_mask & size_mask"):
        expand_refinement_buffer(mesh, result, buffer_layers=0, target_h=0.1)


@pytest.mark.parametrize("target_h", [0, -1, True, np.nan, np.inf, "0.1"])
def test_buffer_rejects_invalid_target_size_collectively(refinement_mesh, target_h):
    count = refinement_mesh.mesh.topology.index_map(2).size_local
    result = _indicator_result(refinement_mesh, np.zeros(count, dtype=bool))
    with pytest.raises(ValueError, match="target_h"):
        expand_refinement_buffer(refinement_mesh, result, 1, target_h=target_h)


def test_buffer_rejects_an_overflowing_physical_radius(refinement_mesh):
    count = refinement_mesh.mesh.topology.index_map(2).size_local
    result = _indicator_result(refinement_mesh, np.zeros(count, dtype=bool))
    with pytest.raises(ValueError, match="must be finite"):
        expand_refinement_buffer(refinement_mesh, result, 10, target_h=1e308)
