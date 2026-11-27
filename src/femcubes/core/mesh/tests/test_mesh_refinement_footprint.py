"""Measure the final fine-mesh footprint after repeated physical buffering."""

from types import SimpleNamespace

from dolfinx import mesh as dmesh
from mpi4py import MPI
import numpy as np
import pytest

from femcubes.core.mesh.refinement import (
    CellIndicatorResult, RefinementConfig, expand_refinement_buffer, refine_mesh_topology,
)


@pytest.mark.parametrize("dim, initial_n, layers", [
    (2, n, layers) for n in (4, 8) for layers in (1, 2, 4)
] + [(3, 4, layers) for layers in (0, 1, 2)])
def test_repeated_refinement_respects_the_padded_region(dim, initial_n, layers):
    """Resolve a fixed planar region without spreading fine cells indefinitely.

    These structured simplex meshes allow one target diameter of overhang in
    2-D and two in 3-D for intersected cells and conforming closure. Coarser
    transition cells may extend farther; they are measured separately from
    the target-sized region. Starting with coarse cells, no cell may become
    smaller than half the target diameter, allowing one stage's size reduction.
    """
    mesh = (dmesh.create_unit_square(MPI.COMM_SELF, initial_n, initial_n) if dim == 2
            else dmesh.create_unit_cube(MPI.COMM_SELF, initial_n, initial_n, initial_n))
    target_h = 1/32 if dim == 2 else 1/8
    criterion_edge = 0.25
    padded_edge = criterion_edge+layers*target_h
    config = RefinementConfig(target_h=target_h)
    stages = 0

    for _ in range(12):
        cells = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
        points = mesh.geometry.x[dmesh.entities_to_geometry(mesh, dim, cells, False)]
        left, right = points[:, :, 0].min(axis=1), points[:, :, 0].max(axis=1)
        sizes = mesh.h(dim, cells)
        criterion = right <= criterion_edge+1e-12
        eligible = sizes > target_h+1e-12
        indicator = CellIndicatorResult(cells, criterion.astype(float), sizes,
                                        criterion, eligible, criterion & eligible)
        selection = expand_refinement_buffer(SimpleNamespace(mesh=mesh), indicator, layers, target_h=target_h)
        if not selection.any_marked:
            break
        artifacts = refine_mesh_topology(mesh, selection, config)
        assert artifacts.report.occurred
        mesh = artifacts.refined_mesh
        stages += 1
    else:
        pytest.fail("A stationary criterion kept requesting further refinement.")

    assert stages >= 2
    inside_padding = right <= padded_edge+1e-12
    assert np.all(sizes[inside_padding] <= target_h+1e-12)
    assert np.min(sizes) >= 0.5*target_h-1e-12

    fine = sizes <= target_h+1e-12
    closure_allowance = (1 if dim == 2 else 2)*target_h
    assert np.max(right[fine]) <= padded_edge+closure_allowance+1e-12
    far_outside = left >= padded_edge+closure_allowance-1e-12
    assert np.any(far_outside)
    assert np.all(sizes[far_outside] > target_h)

    # The converged mesh must remain unchanged on another refinement attempt.
    no_op = refine_mesh_topology(mesh, selection, config)
    assert not no_op.report.occurred
    assert no_op.refined_mesh is mesh
