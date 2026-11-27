"""Exact coefficient migration between equivalent mesh partitions."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from dolfinx import cpp, fem, la
from mpi4py import MPI
import numpy as np

from femcubes.core.mesh._mesh_dofs import _MeshDOFs
from femcubes.core.mesh.refinement.topology import RefinedMeshArtifacts
from femcubes.core.mesh.refinement.transfer import (
    FieldTransferDiagnostic,
    TransferredFieldState,
    _field_statistics,
    _global_scalar_integral,
    _raise_collective_transfer_error,
)


@dataclass(frozen=True)
class _CellMigration:
    """PRIVATE: Reusable source-cell ownership and destination lookup."""

    index_map: cpp.common.IndexMap
    target_to_source: np.ndarray


def migrate_repartitioned_fields(
    source_state: TransferredFieldState,
    artifacts: RefinedMeshArtifacts,
) -> TransferredFieldState:
    """Copy every registered field to a topology-preserving repartition.

    The target must come from zero-edge DOLFINx refinement of the source
    state's target mesh. Original input-cell IDs identify corresponding cells,
    and reference-cell ordering must be unchanged. A DOLFINx ``IndexMap`` and
    distributed vectors route cell coefficient blocks to their new ranks by
    forward scatter. Local cells are copied directly; shared CG coefficients
    are synchronized using the destination function's ordinary forward scatter.

    No quadrature reconstruction, interpolation, or point search is used.
    Scalar, blocked vector, and tensor CG/DG/quadrature fields retain their
    coefficients, including non-polynomial quadrature samples. Source fields
    and the live mesh wrapper are not replaced.
    """

    if not isinstance(source_state, TransferredFieldState):
        raise TypeError("source_state must be a TransferredFieldState.")
    if not isinstance(artifacts, RefinedMeshArtifacts):
        raise TypeError("artifacts must be a RefinedMeshArtifacts.")
    source_mesh = source_state.target_mesh
    target_mesh = artifacts.refined_mesh
    comm = source_mesh.comm
    if MPI.Comm.Compare(comm, target_mesh.comm) not in {MPI.IDENT, MPI.CONGRUENT}:
        raise ValueError("Source and repartitioned meshes must use congruent communicators.")

    migration = _create_cell_migration(source_state, artifacts)
    vectors = {}
    local_error = None
    try:
        source_geometry = source_mesh.geometry.x[
            source_mesh.geometry.dofmap[:migration.index_map.size_local]
        ]
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_transfer_error(comm, "cell geometry packing", local_error)
    migrated_geometry = _exchange_cell_data(migration, source_geometry, vectors)
    local_error = None
    if not np.array_equal(migrated_geometry, target_mesh.geometry.x[target_mesh.geometry.dofmap]):
        local_error = "Redistribution changed cell geometry or reference-cell ordering."
    _raise_collective_transfer_error(comm, "cell geometry validation", local_error)

    spaces = {}
    fields = {}
    diagnostics = {}
    for name, definition in source_state.field_definitions.items():
        source = source_state.fields[name]
        source_stats = _field_statistics(source)
        if not source_stats[3]:
            raise ValueError(f"Source field '{name}' contains non-finite owned coefficients.")
        start = perf_counter()
        local_error = None
        try:
            key = (definition.family, definition.degree, definition.quadrature_degree, definition.value_shape)
            if key not in spaces:
                destination = _MeshDOFs._create_function_from_definition(definition, target_mesh)
                spaces[key] = destination.function_space
            else:
                destination = fem.Function(spaces[key], name=definition.description)
            source_dofmap = source.function_space.dofmap
            target_dofmap = destination.function_space.dofmap
            block_size = source_dofmap.index_map_bs
            if (source_dofmap.bs != block_size
                    or target_dofmap.bs != block_size
                    or target_dofmap.index_map_bs != block_size
                    or source_dofmap.list.shape[1] != target_dofmap.list.shape[1]):
                raise ValueError(f"Field '{name}' changed its cell coefficient layout.")
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _raise_collective_transfer_error(comm, f"migration setup for field '{name}'", local_error)

        source.x.scatter_forward()
        cell_values = source.x.array.reshape(-1, block_size)[
            source_dofmap.list[:migration.index_map.size_local]
        ]
        migrated_values = _exchange_cell_data(migration, cell_values, vectors)
        destination.x.array.reshape(-1, block_size)[target_dofmap.list] = migrated_values
        destination.x.scatter_forward()
        destination_stats = _field_statistics(destination)
        if destination_stats != source_stats:
            raise ValueError(f"Redistribution changed coefficient statistics for field '{name}'.")

        standard = definition.family != "Quadrature"
        fields[name] = destination
        diagnostics[name] = FieldTransferDiagnostic(
            name=name,
            family=definition.family,
            degree=definition.degree,
            value_shape=definition.value_shape,
            global_source_dofs=source_stats[0],
            global_destination_dofs=destination_stats[0],
            global_source_min=source_stats[1],
            global_source_max=source_stats[2],
            global_destination_min=destination_stats[1],
            global_destination_max=destination_stats[2],
            global_source_integral=_global_scalar_integral(source) if standard else None,
            global_destination_integral=_global_scalar_integral(destination) if standard else None,
            transfer_seconds=perf_counter() - start,
            finite=True,
            quadrature_degree=definition.quadrature_degree,
        )

    return TransferredFieldState(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        fields=fields,
        field_definitions=source_state.field_definitions,
        step_fields=source_state.step_fields,
        diagnostics=diagnostics,
        transaction_source_mesh=source_state.transaction_source_mesh,
    )


def _create_cell_migration(source_state, artifacts) -> _CellMigration:
    """PRIVATE: Build one native exchange map from original input-cell IDs."""

    source_mesh = source_state.target_mesh
    target_mesh = artifacts.refined_mesh
    comm = source_mesh.comm
    local_error = None
    try:
        if source_mesh is target_mesh or source_state.pending_field_names:
            raise ValueError("Migration requires a complete state and a distinct target mesh.")
        tdim = source_mesh.topology.dim
        source_map = source_mesh.topology.index_map(tdim)
        target_map = target_mesh.topology.index_map(tdim)
        if source_map.size_global != target_map.size_global:
            raise ValueError("Migration requires unchanged global cell counts.")
        ancestry = artifacts.migration_data
        if ancestry is None or not np.array_equal(ancestry.parent_cell, np.arange(source_map.size_local)):
            raise ValueError("Migration requires zero-edge refinement ancestry in source-cell order.")
        cell_ids = np.asarray(target_mesh.topology.original_cell_index, dtype=np.int64)
        if cell_ids.size != target_map.size_local+target_map.num_ghosts:
            raise ValueError("Original cell IDs must cover all target owned and ghost cells.")
        if np.any(cell_ids < 0) or np.any(cell_ids >= source_map.size_global):
            raise ValueError("Original cell IDs fall outside the source mesh.")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_transfer_error(comm, "cell migration preflight", local_error)

    offsets = np.r_[0, np.cumsum(comm.allgather(source_map.size_local), dtype=np.int64)]
    local_start, local_end = source_map.local_range
    remote = (cell_ids < local_start) | (cell_ids >= local_end)
    ghost_ids = np.ascontiguousarray(cell_ids[remote])
    owners = np.searchsorted(offsets[1:], ghost_ids, side="right").astype(np.int32)
    index_map = cpp.common.IndexMap(comm, source_map.size_local, ghost_ids, owners, 0)
    return _CellMigration(index_map, index_map.global_to_local(cell_ids))


def _exchange_cell_data(migration, values, vectors):
    """PRIVATE: Scatter only requested remote cells and copy retained cells."""

    block_size = int(np.prod(values.shape[1:]))
    key = (block_size, values.dtype.str)
    if key not in vectors:
        vectors[key] = la.vector(migration.index_map, bs=block_size, dtype=values.dtype)
    vector = vectors[key]
    vector.array[:migration.index_map.size_local*block_size] = values.ravel()
    vector.scatter_forward()
    target_values = vector.array.reshape(-1, block_size)[migration.target_to_source]
    return target_values.reshape((migration.target_to_source.size,)+values.shape[1:])
