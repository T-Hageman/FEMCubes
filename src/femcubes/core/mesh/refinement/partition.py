"""Conditional post-transfer load balancing for adaptive refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import perf_counter
from typing import Any, TYPE_CHECKING

from dolfinx import mesh as dmesh
from dolfinx.graph import adjacencylist
from mpi4py import MPI
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from femcubes.core.mesh.refinement.config import RefinementConfig
    from femcubes.core.mesh.refinement.topology import RefinedMeshArtifacts
    from femcubes.core.mesh.refinement.transfer import TransferredFieldState


@dataclass(frozen=True)
class PartitionMigrationData:
    """DOLFINx ancestry returned by the topology-preserving redistribution."""

    parent_cell: npt.NDArray[np.int32]
    parent_facet: npt.NDArray[np.int8]


@dataclass(frozen=True)
class RepartitionResult:
    """Complete detached mesh/field state after a conditional rebalance."""

    artifacts: Any
    transferred_state: Any

    @property
    def repartitioned(self) -> bool:
        """Return whether cell ownership changed through graph partitioning."""

        return bool(self.artifacts.report.repartitioned)


def repartition_mesh(
    artifacts: RefinedMeshArtifacts,
    transferred_state: TransferredFieldState,
    config: RefinementConfig | None = None,
) -> RepartitionResult:
    """Conditionally rebalance a complete detached refined state.

    Refinement topology, markers, and all registered fields must already have
    been transferred to ``artifacts.refined_mesh``. If the configured strict
    imbalance trigger is exceeded, a zero-edge DOLFINx refinement recreates
    the same topology with the standard shared-facet graph partitioner. Its
    partition labels are assigned to ranks to maximize retained cell ownership.
    Tags and field coefficients are migrated exactly before the returned state
    can be installed atomically.
    """

    from femcubes.core.mesh.refinement.config import RefinementConfig
    from femcubes.core.mesh.refinement.markers import (
        MeshMarkerState,
        redistribute_mesh_markers,
    )
    from femcubes.core.mesh.refinement.topology import RefinedMeshArtifacts
    from femcubes.core.mesh.refinement.migration import migrate_repartitioned_fields
    from femcubes.core.mesh.refinement.transfer import (
        TransferredFieldState,
    )

    if not isinstance(artifacts, RefinedMeshArtifacts):
        raise TypeError("artifacts must be a RefinedMeshArtifacts.")
    if not isinstance(transferred_state, TransferredFieldState):
        raise TypeError("transferred_state must be a TransferredFieldState.")
    if config is None:
        config = RefinementConfig()
    if not isinstance(config, RefinementConfig):
        raise TypeError("config must be a RefinementConfig or None.")
    if transferred_state.target_mesh is not artifacts.refined_mesh:
        raise ValueError("Transferred fields and refinement artifacts use different meshes.")
    if transferred_state.pending_field_names:
        raise ValueError("Repartitioning requires every registered field to be transferred first.")

    source_mesh = artifacts.refined_mesh
    comm = source_mesh.comm
    if not _should_repartition(artifacts, config, comm.size):
        return RepartitionResult(artifacts, transferred_state)

    start = perf_counter()
    balanced_mesh, parent_cell, parent_facet = _create_balanced_mesh(source_mesh)
    ownership = _validate_balanced_mesh(source_mesh, balanced_mesh, artifacts)
    provisional_report = replace(
        artifacts.report,
        local_cells_after=ownership[0],
        min_owned_cells_after=ownership[1],
        max_owned_cells_after=ownership[2],
        mean_owned_cells_after=ownership[3],
        imbalance_ratio_after=ownership[4],
        new_min_h=ownership[5],
        new_max_h=ownership[6],
        repartitioned=True,
    )
    balanced_artifacts = replace(
        artifacts,
        refined_mesh=balanced_mesh,
        report=provisional_report,
        cell_tags=None,
        facet_tags=None,
        migration_data=PartitionMigrationData(parent_cell, parent_facet),
    )
    balanced_artifacts = redistribute_mesh_markers(
        MeshMarkerState.from_refined_artifacts(artifacts),
        balanced_artifacts,
    )

    balanced_state = migrate_repartitioned_fields(transferred_state, balanced_artifacts)
    final_report = replace(
        balanced_artifacts.report,
        repartition_seconds=perf_counter() - start,
    )
    balanced_artifacts = replace(balanced_artifacts, report=final_report)
    return RepartitionResult(balanced_artifacts, balanced_state)


def _should_repartition(artifacts, config, comm_size: int) -> bool:
    """PRIVATE: Apply the strict configured imbalance trigger."""

    return bool(
        artifacts.report.occurred
        and config.repartition
        and comm_size > 1
        and artifacts.report.imbalance_ratio_after
        > config.repartition_imbalance_threshold
    )


def _create_balanced_mesh(source_mesh):
    """PRIVATE: Recreate unchanged topology with shared-facet graph partitioning."""

    comm = source_mesh.comm
    local_error = None
    balanced_mesh = None
    parent_cell = None
    parent_facet = None
    try:
        source_mesh.topology.create_entities(1)
        source_mesh.topology.create_connectivity(source_mesh.topology.dim, 1)
    except Exception as exc:  # pragma: no cover - backend topology preparation failure
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_partition_error(comm, "topology preparation", local_error)

    local_error = None
    try:
        balanced_mesh, parent_cell, parent_facet = dmesh.refine(
            source_mesh,
            edges=np.empty(0, dtype=np.int32),
            partitioner=_ownership_preserving_partitioner(),
            option=dmesh.RefinementOption.parent_cell_and_facet,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_partition_error(comm, "graph redistribution", local_error)
    return (
        balanced_mesh,
        np.asarray(parent_cell, dtype=np.int32),
        np.asarray(parent_facet, dtype=np.int8),
    )


def _ownership_preserving_partitioner():
    """PRIVATE: Keep the native graph cut while minimizing rank-label swaps."""

    partitioner = dmesh.create_cell_partitioner(dmesh.GhostMode.shared_facet)

    def partition(comm, nparts, cell_types, cells):
        destinations = partitioner(comm, nparts, cell_types, cells)
        owners = destinations.array[destinations.offsets[:-1]]
        overlap = np.asarray(comm.allgather(np.bincount(owners, minlength=nparts)), dtype=np.int64)
        rank_for_part = _match_partition_ranks(overlap)
        return adjacencylist(
            rank_for_part[destinations.array],
            destinations.offsets.copy(),
        )._cpp_object

    return partition


def _match_partition_ranks(overlap: np.ndarray) -> np.ndarray:
    """PRIVATE: Maximize retained cells; overlap rows are ranks, columns parts."""

    ranks, parts = linear_sum_assignment(overlap, maximize=True)
    rank_for_part = np.empty(overlap.shape[1], dtype=np.int32)
    rank_for_part[parts] = ranks
    return rank_for_part


def _validate_balanced_mesh(source_mesh, target_mesh, artifacts):
    """PRIVATE: Validate topology preservation and return ownership statistics."""

    comm = source_mesh.comm
    local_error = None
    try:
        if source_mesh.topology.cell_type != target_mesh.topology.cell_type:  # pragma: no cover - zero-edge redistribution preserves cell type
            raise ValueError("Redistribution changed the mesh cell type.")
        if source_mesh.topology.dim != target_mesh.topology.dim:  # pragma: no cover - zero-edge redistribution preserves topology dimension
            raise ValueError("Redistribution changed the topological dimension.")
        if source_mesh.geometry.dim != target_mesh.geometry.dim:  # pragma: no cover - zero-edge redistribution preserves geometry dimension
            raise ValueError("Redistribution changed the geometric dimension.")
    except Exception as exc:  # pragma: no cover - backend topology metadata failure
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_partition_error(comm, "mesh validation", local_error)

    tdim = target_mesh.topology.dim
    target_map = target_mesh.topology.index_map(tdim)
    if target_map.size_global != artifacts.report.global_cells_after:
        raise ValueError(
            "Redistribution changed the global cell count from "
            f"{artifacts.report.global_cells_after} to {target_map.size_global}."
        )
    counts = comm.allgather(int(target_map.size_local))
    mean_owned = float(sum(counts) / len(counts))
    min_h, max_h = _global_diameter_bounds(target_mesh)
    tolerance = 1.0e-12 * max(
        1.0,
        abs(artifacts.report.new_min_h),
        abs(artifacts.report.new_max_h),
    )
    if not math.isclose(min_h, artifacts.report.new_min_h, rel_tol=1.0e-11, abs_tol=tolerance):
        raise ValueError("Redistribution changed the minimum cell diameter.")
    if not math.isclose(max_h, artifacts.report.new_max_h, rel_tol=1.0e-11, abs_tol=tolerance):
        raise ValueError("Redistribution changed the maximum cell diameter.")
    return (
        int(target_map.size_local),
        int(min(counts)),
        int(max(counts)),
        mean_owned,
        float(max(counts) / mean_owned),
        min_h,
        max_h,
    )


def _global_diameter_bounds(mesh) -> tuple[float, float]:
    """PRIVATE: Return global min/max owned-cell diameters."""

    tdim = mesh.topology.dim
    owned = mesh.topology.index_map(tdim).size_local
    diameters = np.asarray(mesh.h(tdim, np.arange(owned, dtype=np.int32)), dtype=float)
    local_min = float(np.min(diameters)) if diameters.size else math.inf
    local_max = float(np.max(diameters)) if diameters.size else -math.inf
    return (
        float(mesh.comm.allreduce(local_min, op=MPI.MIN)),
        float(mesh.comm.allreduce(local_max, op=MPI.MAX)),
    )


def _raise_collective_partition_error(comm, stage: str, local_error: str | None) -> None:
    """PRIVATE: Raise one rank-attributed redistribution error everywhere."""

    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        details = "; ".join(
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        )
        raise ValueError(f"Mesh redistribution {stage} failed: {details}")
