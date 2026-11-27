"""Serial marker-transfer tests for detached refined meshes."""

from __future__ import annotations

from types import SimpleNamespace

from dolfinx import fem, mesh as dmesh
from mpi4py import MPI
import numpy as np
import pytest
import ufl

from femcubes.core.mesh.refinement import (
    CellIndicatorResult,
    MeshMarkerState,
    expand_refinement_buffer,
    refine_mesh_topology,
    resolve_owned_child_input_positions,
    transfer_mesh_markers,
)


def _selection(mesh, marked_mask, buffer_layers=0):
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    marked_mask = np.asarray(marked_mask, dtype=bool)
    result = CellIndicatorResult(
        cells=cells,
        values=marked_mask.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=marked_mask.copy(),
        size_mask=np.ones(cells.size, dtype=bool),
        marked_mask=marked_mask.copy(),
    )
    return expand_refinement_buffer(SimpleNamespace(mesh=mesh), result, buffer_layers, target_h=0.1)


def _refine_and_transfer(source, marked_mask):
    topology = refine_mesh_topology(source.mesh, _selection(source.mesh, marked_mask))
    return transfer_mesh_markers(source, topology)


def _tagged_measure(mesh, tags, tag, measure_name):
    measure = ufl.Measure(measure_name, domain=mesh, subdomain_data=tags)
    local_value = fem.assemble_scalar(fem.form(fem.Constant(mesh, 1.0) * measure(tag)))
    return float(mesh.comm.allreduce(local_value, op=MPI.SUM))


def _split_rectangle_domains(mesh_wrapper):
    mesh = mesh_wrapper.mesh
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    centers = dmesh.compute_midpoints(mesh, tdim, cells)
    values = np.where(centers[:, 0] < 1.0, 11, 22).astype(np.int32)
    mesh_wrapper.cell_tags = dmesh.meshtags(mesh, tdim, cells, values)
    mesh_wrapper.domain_markers.clear()
    mesh_wrapper.add_domain_marker("left_domain", 11)
    mesh_wrapper.add_domain_marker("right_domain", 22)


def test_owned_children_normalize_through_original_cell_indices(refinement_mesh):
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    cell_count = source.mesh.topology.index_map(source.mesh.topology.dim).size_local
    marked = np.zeros(cell_count, dtype=bool)
    marked[cell_count // 2] = True
    topology = refine_mesh_topology(source.mesh, _selection(source.mesh, marked))

    positions = resolve_owned_child_input_positions(topology)

    assert positions.size == topology.report.local_cells_after
    assert np.unique(positions).size == positions.size
    assert np.all(positions >= 0)
    assert np.all(positions < topology.parent_cell.size)


def test_multiple_named_cell_tags_inherit_from_refined_parent_cells(refinement_mesh):
    _split_rectangle_domains(refinement_mesh)
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    cell_count = source.mesh.topology.index_map(source.mesh.topology.dim).size_local
    transferred = _refine_and_transfer(source, np.ones(cell_count, dtype=bool))

    assert transferred.domain_names() == ["left_domain", "right_domain"]
    assert transferred.get_domain_cells("left_domain").size > 0
    assert transferred.get_domain_cells("right_domain").size > 0

    positions = resolve_owned_child_input_positions(transferred)
    parents = transferred.parent_cell[positions]
    old_values = dict(zip(source.cell_tags.indices.tolist(), source.cell_tags.values.tolist()))
    new_values = dict(
        zip(transferred.cell_tags.indices.tolist(), transferred.cell_tags.values.tolist())
    )
    assert all(new_values[child] == old_values[int(parent)] for child, parent in enumerate(parents))


def test_all_named_exterior_boundaries_survive_refinement(refinement_mesh):
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    cell_count = source.mesh.topology.index_map(source.mesh.topology.dim).size_local
    transferred = _refine_and_transfer(source, np.ones(cell_count, dtype=bool))

    assert transferred.boundary_names() == refinement_mesh.boundary_names()
    for name in refinement_mesh.boundary_names():
        assert transferred.get_boundary_facets(name).size > 0


def test_named_domain_and_boundary_measures_are_preserved(refinement_mesh):
    _split_rectangle_domains(refinement_mesh)
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    cell_count = source.mesh.topology.index_map(source.mesh.topology.dim).size_local
    transferred = _refine_and_transfer(source, np.ones(cell_count, dtype=bool))

    for name in transferred.domain_names():
        tag = transferred.get_domain_tag(name)
        old_measure = _tagged_measure(source.mesh, source.cell_tags, tag, "dx")
        new_measure = _tagged_measure(
            transferred.refined_mesh,
            transferred.cell_tags,
            tag,
            "dx",
        )
        assert new_measure == pytest.approx(old_measure)

    for name in transferred.boundary_names():
        tag = transferred.get_boundary_tag(name)
        old_measure = _tagged_measure(source.mesh, source.facet_tags, tag, "ds")
        new_measure = _tagged_measure(
            transferred.refined_mesh,
            transferred.facet_tags,
            tag,
            "ds",
        )
        assert new_measure == pytest.approx(old_measure)


def test_children_of_untagged_cells_remain_untagged(refinement_mesh):
    mesh = refinement_mesh.mesh
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    tagged_old_cells = cells[cells % 2 == 0]
    partial_tags = dmesh.meshtags(
        mesh,
        tdim,
        tagged_old_cells,
        np.full(tagged_old_cells.size, 7, dtype=np.int32),
    )
    source = MeshMarkerState(
        mesh=mesh,
        cell_tags=partial_tags,
        domain_markers={"tagged": 7},
    )
    transferred = _refine_and_transfer(source, np.ones(cells.size, dtype=bool))
    positions = resolve_owned_child_input_positions(transferred)
    parents = transferred.parent_cell[positions]
    expected_tagged_children = np.flatnonzero(np.isin(parents, tagged_old_cells)).astype(np.int32)

    np.testing.assert_array_equal(
        transferred.get_domain_cells("tagged"),
        expected_tagged_children,
    )
    assert transferred.cell_tags.indices.size < transferred.report.local_cells_after


def test_noop_transfer_preserves_original_tag_objects_and_names(refinement_mesh):
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    cell_count = source.mesh.topology.index_map(source.mesh.topology.dim).size_local
    topology = refine_mesh_topology(
        source.mesh,
        _selection(source.mesh, np.zeros(cell_count, dtype=bool)),
    )

    transferred = transfer_mesh_markers(source, topology)

    assert transferred.cell_tags is source.cell_tags
    assert transferred.facet_tags is source.facet_tags
    assert transferred.domain_names() == refinement_mesh.domain_names()
    assert transferred.boundary_names() == refinement_mesh.boundary_names()


def test_transferred_markers_can_source_a_later_refinement(refinement_mesh):
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    first_count = source.mesh.topology.index_map(source.mesh.topology.dim).size_local
    first_marked = np.zeros(first_count, dtype=bool)
    first_marked[first_count // 3] = True
    first = _refine_and_transfer(source, first_marked)
    second_source = MeshMarkerState.from_refined_artifacts(first)
    second_count = second_source.mesh.topology.index_map(second_source.mesh.topology.dim).size_local
    second_marked = np.zeros(second_count, dtype=bool)
    second_marked[2 * second_count // 3] = True

    second = _refine_and_transfer(second_source, second_marked)

    assert second.domain_names() == refinement_mesh.domain_names()
    assert second.boundary_names() == refinement_mesh.boundary_names()
    for name in second.boundary_names():
        assert second.get_boundary_facets(name).size > 0


def test_conflicting_duplicate_facet_assignments_are_rejected():
    import femcubes.core.mesh.refinement.markers as markers

    assignments = {}
    markers._record_tag_assignment(assignments, 4, 2)
    markers._record_tag_assignment(assignments, 4, 2)

    with pytest.raises(ValueError, match="Conflicting transferred tags"):
        markers._record_tag_assignment(assignments, 4, 3)


def test_interior_interface_tags_are_transferred_and_validated():
    # Interior facet tags (e.g. a subdomain interface) must survive refinement
    # next to them. The transfer validation compares tagged-facet measures;
    # for interior facets these are the interface length, not zero.
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.mesh_generators.two_squared import Sample_Parameters
    from femcubes.core.mesh.refinement.markers import _tagged_measure as validation_measure

    source = Mesh(Sample_Parameters())
    tag = source.get_boundary_tag("interface")
    length = source.params.L
    assert np.isclose(validation_measure(source.mesh, source.facet_tags, tag, "ds"), length)

    tdim = source.mesh.topology.dim
    owned_cells = np.arange(source.mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    near_interface = np.abs(dmesh.compute_midpoints(source.mesh, tdim, owned_cells)[:, 0] - length) < 0.1
    transferred = _refine_and_transfer(MeshMarkerState.from_mesh_wrapper(source), near_interface)

    refined = transferred.refined_mesh
    assert np.isclose(_tagged_measure(refined, transferred.facet_tags, tag, "dS"), length)
    assert transferred.get_boundary_facets("interface").size > source.get_boundary_facets("interface").size
