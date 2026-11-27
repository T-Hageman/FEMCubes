"""Baseline topology contracts used by later refinement tests."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI

from femcubes.core.mesh.refinement import (
    CellIndicatorResult,
    RefinementConfig,
    expand_refinement_buffer,
    refine_mesh_topology,
)


def _selection(mesh_wrapper, marked_mask, size_mask=None, buffer_layers=0):
    """Build a topology selection in complete owned-cell ordering."""

    mesh = mesh_wrapper.mesh if hasattr(mesh_wrapper, "mesh") else mesh_wrapper
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    marked_mask = np.asarray(marked_mask, dtype=bool)
    if size_mask is None:
        size_mask = np.ones(cells.size, dtype=bool)
    size_mask = np.asarray(size_mask, dtype=bool)
    result = CellIndicatorResult(
        cells=cells,
        values=marked_mask.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=marked_mask.copy(),
        size_mask=size_mask,
        marked_mask=marked_mask & size_mask,
    )
    wrapper = mesh_wrapper if hasattr(mesh_wrapper, "mesh") else type("Wrapper", (), {"mesh": mesh})()
    return expand_refinement_buffer(wrapper, result, buffer_layers, target_h=0.1)


def test_serial_refinement_fixture_has_named_simplex_topology(refinement_mesh):
    """The shared serial fixture should expose stable named mesh contracts."""
    mesh = refinement_mesh
    tdim = mesh.mesh.topology.dim
    cell_map = mesh.mesh.topology.index_map(tdim)

    assert tdim == 2
    assert mesh.mesh.topology.cell_type.name == "triangle"
    assert cell_map.size_local > 0
    assert cell_map.size_global == cell_map.size_local
    assert cell_map.num_ghosts == 0
    assert set(mesh.domain_names()) == {"domain"}
    assert set(mesh.boundary_names()) == {"bottom", "right", "top", "left"}

    area = fem.assemble_scalar(fem.form(fem.Constant(mesh.mesh, 1.0) * ufl.dx))
    assert area == pytest.approx(2.0)
    assert mesh.get_domain_cells("domain").size == cell_map.size_local
    assert all(mesh.get_boundary_facets(name).size > 0 for name in mesh.boundary_names())


def test_serial_refinement_fixture_records_representative_field_layouts(refinement_mesh):
    """CG, DG, vector, and quadrature layouts should exist before remeshing."""
    mesh = refinement_mesh

    scalar_cg = mesh.Add_Field("phasefield", "Phase field", "CG", 2)
    scalar_dg = mesh.Add_Field("driving_force", "Driving force", "DG", 1)
    vector_cg = mesh.Add_Vector_Field("u", "Displacement", "CG", 2, 2)
    tensor_q = mesh.Add_Quadrature_Field("sigma", "Stress", (6, 1))

    assert tuple(scalar_cg.function_space.element.value_shape) == ()
    assert tuple(scalar_dg.function_space.element.value_shape) == ()
    assert tuple(vector_cg.function_space.element.value_shape) == (2,)
    assert tuple(tensor_q.function_space.element.value_shape) == (6, 1)
    assert all(field.x.array.size > 0 for field in (scalar_cg, scalar_dg, vector_cg, tensor_q))
    assert list(mesh.field_definitions) == ["phasefield", "driving_force", "u", "sigma"]
    assert np.isfinite(sum(field.x.array.size for field in mesh.fields.values()))


def test_local_marked_region_creates_detached_refined_topology(refinement_mesh):
    """A local mark should increase cells without replacing the wrapper mesh."""

    old_mesh = refinement_mesh.mesh
    old_cell_count = old_mesh.topology.index_map(old_mesh.topology.dim).size_global
    marked = np.zeros(old_cell_count, dtype=bool)
    marked[old_cell_count // 2] = True

    artifacts = refine_mesh_topology(old_mesh, _selection(refinement_mesh, marked))

    assert artifacts.report.occurred
    assert artifacts.refined_mesh is not old_mesh
    assert refinement_mesh.mesh is old_mesh
    assert artifacts.report.global_cells_after > old_cell_count
    assert artifacts.report.global_cells_added > 0
    assert artifacts.report.levels_applied == 1
    assert artifacts.marked_edges.size > 0
    np.testing.assert_array_equal(artifacts.marked_edges, np.unique(artifacts.marked_edges))


def test_empty_marks_return_original_mesh_without_ancestry(refinement_mesh):
    """A collective no-op should not call DOLFINx refinement."""

    old_mesh = refinement_mesh.mesh
    cell_count = old_mesh.topology.index_map(old_mesh.topology.dim).size_local

    artifacts = refine_mesh_topology(
        old_mesh,
        _selection(refinement_mesh, np.zeros(cell_count, dtype=bool)),
    )

    assert not artifacts.report.occurred
    assert artifacts.refined_mesh is old_mesh
    assert artifacts.parent_cell.size == 0
    assert artifacts.parent_facet.size == 0
    assert artifacts.marked_edges.size == 0
    assert artifacts.report.global_cells_added == 0
    assert artifacts.report.old_min_h == pytest.approx(artifacts.report.new_min_h)
    assert artifacts.report.old_max_h == pytest.approx(artifacts.report.new_max_h)


def test_parent_arrays_have_valid_serial_ranges_and_allow_conforming_closure(refinement_mesh):
    """Ancestry should be valid without prescribing DOLFINx closure details."""

    old_mesh = refinement_mesh.mesh
    tdim = old_mesh.topology.dim
    old_cell_count = old_mesh.topology.index_map(tdim).size_local
    marked = np.zeros(old_cell_count, dtype=bool)
    marked[0] = True

    artifacts = refine_mesh_topology(old_mesh, _selection(refinement_mesh, marked))

    assert artifacts.parent_cell.size == artifacts.report.local_cells_after
    assert np.all(artifacts.parent_cell >= 0)
    assert np.all(artifacts.parent_cell < old_cell_count)
    facets_per_cell = tdim + 1
    assert artifacts.parent_facet.size == artifacts.parent_cell.size * facets_per_cell
    assert np.all(artifacts.parent_facet >= -1)
    assert np.all(artifacts.parent_facet < facets_per_cell)
    assert np.count_nonzero(artifacts.parent_cell == 0) > 1


def test_target_size_eligibility_prevents_repeated_topology_refinement(refinement_mesh):
    """Cells at the selected target diameter should produce a later no-op."""

    old_mesh = refinement_mesh.mesh
    old_count = old_mesh.topology.index_map(old_mesh.topology.dim).size_local
    first = refine_mesh_topology(old_mesh, _selection(refinement_mesh, np.ones(old_count, dtype=bool)))
    refined_mesh = first.refined_mesh
    tdim = refined_mesh.topology.dim
    refined_count = refined_mesh.topology.index_map(tdim).size_local
    diameters = refined_mesh.h(tdim, np.arange(refined_count, dtype=np.int32))
    target_h = float(np.max(diameters))
    eligible = diameters > target_h + 64.0 * np.finfo(float).eps * max(1.0, target_h)

    second = refine_mesh_topology(
        refined_mesh,
        _selection(refined_mesh, np.ones(refined_count, dtype=bool), size_mask=eligible),
    )

    assert not np.any(eligible)
    assert not second.report.occurred
    assert second.refined_mesh is refined_mesh


def test_topology_refinement_defers_repartition_until_after_field_transfer(refinement_mesh):
    """Raw topology refinement must retain identity-partition ancestry."""

    cell_count = refinement_mesh.mesh.topology.index_map(refinement_mesh.mesh.topology.dim).size_local
    marked = np.zeros(cell_count, dtype=bool)
    marked[0] = True

    artifacts = refine_mesh_topology(
        refinement_mesh.mesh,
        _selection(refinement_mesh, marked),
        config=RefinementConfig(repartition=True),
    )

    assert not artifacts.report.repartitioned
    assert artifacts.report.repartition_seconds == 0.0
    assert (
        artifacts.report.imbalance_ratio_before_repartition
        == artifacts.report.imbalance_ratio_after
    )


def test_non_simplex_mesh_is_rejected_before_topology_changes():
    """Quadrilateral refinement is outside the supported simplex contract."""

    mesh = dmesh.create_unit_square(
        MPI.COMM_WORLD,
        2,
        2,
        cell_type=dmesh.CellType.quadrilateral,
    )
    cell_count = mesh.topology.index_map(mesh.topology.dim).size_local
    selection = _selection(mesh, np.ones(cell_count, dtype=bool))

    with pytest.raises(ValueError, match="requires a simplex mesh"):
        refine_mesh_topology(mesh, selection)


@pytest.mark.parametrize("dim", [2, 3])
def test_only_edges_exceeding_target_size_are_requested(dim):
    if dim == 2:
        mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
        target = 1.0
    else:
        mesh = dmesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        target = np.sqrt(2.0)
    count = mesh.topology.index_map(dim).size_local
    artifacts = refine_mesh_topology(mesh, _selection(mesh, np.ones(count, dtype=bool)),
                                     RefinementConfig(target_h=target))
    assert artifacts.report.occurred
    assert np.all(mesh.h(1, artifacts.marked_edges) > target+1e-12)
    refined = artifacts.refined_mesh
    cells = np.arange(refined.topology.index_map(dim).size_local, dtype=np.int32)
    assert np.max(refined.h(dim, cells)) <= target+1e-12


def test_marked_cells_without_oversized_edges_leave_topology_unchanged(refinement_mesh):
    mesh = refinement_mesh.mesh
    cells = np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32)
    artifacts = refine_mesh_topology(mesh, _selection(mesh, np.ones(cells.size, dtype=bool)),
                                     RefinementConfig(target_h=float(np.max(mesh.h(2, cells)))))
    assert not artifacts.report.occurred
    assert artifacts.refined_mesh is mesh
    assert artifacts.marked_edges.size == 0
    assert artifacts.report.global_marked_cells == cells.size


def test_size_aware_refinement_reaches_target_without_uniformly_splitting_short_edges():
    target = 1/16
    counts = []
    for use_target in (False, True):
        mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
        for _ in range(8):
            cells = np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32)
            points = mesh.geometry.x[dmesh.entities_to_geometry(mesh, 2, cells, False)]
            criterion = points[:, :, 0].max(axis=1) <= 0.25+1e-12
            sizes = mesh.h(2, cells)
            marks = criterion & (sizes > target+1e-12)
            if not np.any(marks):
                break
            config = RefinementConfig(target_h=target) if use_target else None
            mesh = refine_mesh_topology(mesh, _selection(mesh, marks), config).refined_mesh
        assert not np.any(marks)
        counts.append(mesh.topology.index_map(2).size_global)
        if use_target:
            np.testing.assert_allclose(np.min(sizes[criterion]), target)
    assert counts[1] < 0.6*counts[0]


@pytest.mark.parametrize("config", ["invalid", {"target_h": -1}])
def test_invalid_topology_controls_are_rejected_collectively(refinement_mesh, config):
    mesh = refinement_mesh.mesh
    count = mesh.topology.index_map(2).size_local
    with pytest.raises(ValueError, match="Invalid topology-refinement input"):
        refine_mesh_topology(mesh, _selection(mesh, np.ones(count, dtype=bool)), config)
