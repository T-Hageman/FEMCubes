"""Repartitioning must preserve stored finite-element coefficients exactly."""

from dataclasses import replace

from dolfinx import mesh as dmesh
from dolfinx.graph import adjacencylist
import numpy as np
import pytest

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.mesh.refinement import PartitionMigrationData, migrate_repartitioned_fields
from femcubes.core.mesh.tests.test_mesh_refinement_manager import _successful_detached_state


def _migration_case(params, dim, quadrature_degree):
    """Build a detached refined mesh with mixed, DG, and quadrature storage."""

    params.dim = dim
    params.Lz = 1.0
    params.Quadrature_Degree = quadrature_degree
    wrapper = Mesh(params)
    wrapper.Add_Field("cg", "Continuous scalar", "CG", 3)
    wrapper.Add_Vector_Field("vector", "Continuous vector", "CG", 3, dim)
    wrapper.Add_Field("dg", "Discontinuous polynomial", "DG", 2)
    wrapper.Add_Field("historyOld", "Previous cell history", "DG", 0)
    wrapper.Add_Quadrature_Field("samples", "Scalar samples", ())
    wrapper.Add_Quadrature_Field("stress", "Stress samples", (6, 1))
    wrapper.Add_Quadrature_Field("tangent", "Tangent samples", (6, 6))
    wrapper.Add_TestFunction("cg", step=0)
    wrapper.Add_Vector_TestFunction("vector", nDim=dim, step=0)
    wrapper.finalize_test_functions()
    artifacts, state = _successful_detached_state(wrapper)
    rng = np.random.default_rng(1234+state.target_mesh.comm.rank)
    for field in state.fields.values():
        field.x.array[:] = rng.normal(size=field.x.array.size)
        field.x.scatter_forward()
    return wrapper, artifacts, state


def _rotate_cell_ownership(source_mesh):
    """Force remote migration, retaining the old owner's copy as a ghost."""

    def rotate(comm, nparts, cell_types, cells):
        count = sum(len(rows)//(source_mesh.topology.dim+1) for rows in cells)
        owner = (comm.rank+1) % nparts
        destinations = np.full((count, 1 if nparts == 1 else 2), owner, dtype=np.int32)
        if nparts > 1:
            destinations[:, 1] = comm.rank
        return adjacencylist(destinations)._cpp_object

    source_mesh.topology.create_entities(1)
    return dmesh.refine(
        source_mesh,
        edges=np.empty(0, dtype=np.int32),
        partitioner=rotate,
        option=dmesh.RefinementOption.parent_cell_and_facet,
    )


def _check_exact_migration(artifacts, state, target_data=None):
    """Compare every coefficient with its original cell, including ghosts."""

    source = state.target_mesh
    cell_map = source.topology.index_map(source.topology.dim)
    owned = cell_map.size_local
    global_ids = cell_map.local_to_global(np.arange(owned, dtype=np.int32))
    originals = {name: field.x.array.copy() for name, field in state.fields.items()}
    target, parents, facets = _rotate_cell_ownership(source) if target_data is None else target_data
    target_artifacts = replace(
        artifacts,
        refined_mesh=target,
        cell_tags=None,
        facet_tags=None,
        migration_data=PartitionMigrationData(parents, facets),
    )

    migrated = migrate_repartitioned_fields(state, target_artifacts)

    for name, old in state.fields.items():
        new = migrated.fields[name]
        old_map, new_map = old.function_space.dofmap, new.function_space.dofmap
        bs = old_map.index_map_bs
        old_cells = old.x.array.reshape(-1, bs)[old_map.list[:owned]]
        records = source.comm.allgather((global_ids, old_cells))
        expected_by_cell = {int(gid): values for ids, cells in records for gid, values in zip(ids, cells)}
        ids = target.topology.original_cell_index
        expected = np.array([expected_by_cell[int(gid)] for gid in ids]).reshape((ids.size,)+old_cells.shape[1:])
        np.testing.assert_array_equal(new.x.array.reshape(-1, bs)[new_map.list], expected)
        np.testing.assert_array_equal(old.x.array, originals[name])
        assert new.function_space.mesh is target
        assert migrated.diagnostics[name].finite
        if state.field_definitions[name].family == "Quadrature":
            assert migrated.diagnostics[name].reconstruction_degree is None
    assert migrated.transaction_source_mesh is state.transaction_source_mesh
    assert migrated.field_definitions == state.field_definitions
    assert migrated.step_fields == state.step_fields
    assert not migrated.pending_field_names
    return target_artifacts, migrated


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("quadrature_degree", [3, 5])
def test_repartitioning_preserves_arbitrary_samples_and_coefficients(refinement_rectangle_params, dim, quadrature_degree):
    wrapper, artifacts, state = _migration_case(refinement_rectangle_params, dim, quadrature_degree)
    old_mesh = wrapper.mesh
    for _ in range(2):
        artifacts, state = _check_exact_migration(artifacts, state)
    assert wrapper.mesh is old_mesh
    assert wrapper.remesh_epoch == 0


def test_migration_requires_topology_preserving_ancestry(refinement_rectangle_params):
    _, artifacts, state = _migration_case(refinement_rectangle_params, 2, 3)
    with pytest.raises(ValueError, match="distinct target mesh"):
        migrate_repartitioned_fields(state, artifacts)
    target, parents, facets = _rotate_cell_ownership(state.target_mesh)
    target_artifacts = replace(artifacts, refined_mesh=target, cell_tags=None, facet_tags=None)
    with pytest.raises(ValueError, match="zero-edge refinement ancestry"):
        migrate_repartitioned_fields(state, target_artifacts)
    invalid_ancestry = PartitionMigrationData(parents[::-1].copy(), facets)
    with pytest.raises(ValueError, match="zero-edge refinement ancestry"):
        migrate_repartitioned_fields(state, replace(target_artifacts, migration_data=invalid_ancestry))


@pytest.mark.parametrize("argument", ["state", "artifacts"])
def test_migration_rejects_invalid_argument_types(refinement_rectangle_params, argument):
    _, artifacts, state = _migration_case(refinement_rectangle_params, 2, 3)
    with pytest.raises(TypeError, match="TransferredFieldState|RefinedMeshArtifacts"):
        if argument == "state":
            migrate_repartitioned_fields(None, artifacts)
        else:
            migrate_repartitioned_fields(state, None)
