"""Cell/facet marker transfer for detached refined meshes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Integral
from types import MappingProxyType
from typing import Any

from dolfinx import fem, io, mesh as dmesh
from dolfinx.graph import adjacencylist
from mpi4py import MPI
import numpy as np
import numpy.typing as npt
import ufl

from femcubes.core.mesh.refinement.topology import RefinedMeshArtifacts


@dataclass(frozen=True)
class MeshMarkerState:
    """Explicit mesh tags and stable names before one refinement level."""

    mesh: Any
    cell_tags: Any = None
    facet_tags: Any = None
    domain_markers: Mapping[str, int] | None = None
    boundary_markers: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        topology = getattr(self.mesh, "topology", None)
        if topology is None:
            raise TypeError("Marker state mesh must expose a DOLFINx topology.")
        domain_markers = {} if self.domain_markers is None else dict(self.domain_markers)
        boundary_markers = {} if self.boundary_markers is None else dict(self.boundary_markers)
        _validate_marker_mapping("domain_markers", domain_markers)
        _validate_marker_mapping("boundary_markers", boundary_markers)
        object.__setattr__(self, "domain_markers", MappingProxyType(domain_markers))
        object.__setattr__(self, "boundary_markers", MappingProxyType(boundary_markers))

        tdim = topology.dim
        if self.cell_tags is not None and self.cell_tags.dim != tdim:
            raise ValueError(
                f"Cell tags must have dimension {tdim}; received {self.cell_tags.dim}."
            )
        if self.facet_tags is not None and self.facet_tags.dim != tdim - 1:
            raise ValueError(
                f"Facet tags must have dimension {tdim - 1}; received {self.facet_tags.dim}."
            )
        if domain_markers and self.cell_tags is None:
            raise ValueError("Named domains require source cell tags.")
        if boundary_markers and self.facet_tags is None:
            raise ValueError("Named boundaries require source facet tags.")

    @classmethod
    def from_mesh_wrapper(cls, mesh_wrapper) -> MeshMarkerState:
        """Snapshot marker artifacts from the public FEMCubes mesh wrapper."""

        return cls(
            mesh=mesh_wrapper.mesh,
            cell_tags=getattr(mesh_wrapper, "cell_tags", None),
            facet_tags=getattr(mesh_wrapper, "facet_tags", None),
            domain_markers=dict(getattr(mesh_wrapper, "domain_markers", {})),
            boundary_markers=dict(getattr(mesh_wrapper, "boundary_markers", {})),
        )

    @classmethod
    def from_refined_artifacts(cls, artifacts: RefinedMeshArtifacts) -> MeshMarkerState:
        """Create the source state for a subsequent detached refinement level."""

        return cls(
            mesh=artifacts.refined_mesh,
            cell_tags=artifacts.cell_tags,
            facet_tags=artifacts.facet_tags,
            domain_markers=artifacts.domain_markers,
            boundary_markers=artifacts.boundary_markers,
        )


def transfer_mesh_markers(
    source: MeshMarkerState,
    artifacts: RefinedMeshArtifacts,
) -> RefinedMeshArtifacts:
    """Transfer cell/facet tags through DOLFINx refinement ancestry.

    Owned refined cells are first mapped from final local ordering to the raw
    parent-array ordering through ``topology.original_cell_index``. Cell tags
    inherit directly from the old parent. Facet tags inherit only where
    ``parent_facet`` identifies an old parent facet; newly created interior
    facets remain untagged.
    """

    if not isinstance(source, MeshMarkerState):
        raise TypeError("source must be a MeshMarkerState.")
    if not isinstance(artifacts, RefinedMeshArtifacts):
        raise TypeError("artifacts must be a RefinedMeshArtifacts.")
    comm = source.mesh.comm
    if MPI.Comm.Compare(comm, artifacts.refined_mesh.comm) not in {
        MPI.IDENT,
        MPI.CONGRUENT,
    }:
        raise ValueError("Source and refined meshes must use congruent communicators.")

    marker_signature = (
        tuple(source.domain_markers.items()),
        tuple(source.boundary_markers.items()),
    )
    marker_signatures = comm.allgather(marker_signature)
    if any(signature != marker_signatures[0] for signature in marker_signatures):
        raise ValueError(
            "Marker name mappings must be identical on every mesh rank; "
            f"received {marker_signatures}."
        )

    if not artifacts.report.occurred:
        if artifacts.refined_mesh is not source.mesh:
            raise ValueError("No-op refinement artifacts must retain the source mesh.")
        return replace(
            artifacts,
            cell_tags=source.cell_tags,
            facet_tags=source.facet_tags,
            domain_markers=source.domain_markers,
            boundary_markers=source.boundary_markers,
        )

    child_input_positions = None
    parent_cells = None
    local_error = None
    try:
        child_input_positions = resolve_owned_child_input_positions(artifacts)
        parent_cells = artifacts.parent_cell[child_input_positions]
        old_owned_cells = source.mesh.topology.index_map(source.mesh.topology.dim).size_local
        if np.any(parent_cells < 0) or np.any(parent_cells >= old_owned_cells):
            raise ValueError(
                "Refinement parent-cell ancestry resolves outside owned source cells."
            )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_marker_error(comm, "ancestry normalization", local_error)

    cell_tags = None
    facet_tags = None
    local_error = None
    try:
        cell_tags = _transfer_cell_tags(source, artifacts, parent_cells)
        facet_tags = _transfer_facet_tags(
            source,
            artifacts,
            child_input_positions,
            parent_cells,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_marker_error(comm, "tag reconstruction", local_error)
    transferred = replace(
        artifacts,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        domain_markers=source.domain_markers,
        boundary_markers=source.boundary_markers,
    )
    _validate_named_marker_transfer(source, transferred)
    return transferred


def redistribute_mesh_markers(
    source: MeshMarkerState,
    artifacts: RefinedMeshArtifacts,
) -> RefinedMeshArtifacts:
    """Migrate exact cell/facet tags to a repartitioned equivalent mesh.

    Tagged entities are represented by the source geometry-node global IDs
    used as input to DOLFINx's zero-edge redistribution. DOLFINx then routes
    those entities and values to their new owning ranks before new
    ``MeshTags`` objects are constructed.
    """

    if not isinstance(source, MeshMarkerState):
        raise TypeError("source must be a MeshMarkerState.")
    if not isinstance(artifacts, RefinedMeshArtifacts):
        raise TypeError("artifacts must be a RefinedMeshArtifacts.")
    target_mesh = artifacts.refined_mesh
    comm = source.mesh.comm
    if source.mesh is target_mesh:
        raise ValueError("Marker redistribution requires a distinct target mesh.")
    if MPI.Comm.Compare(comm, target_mesh.comm) not in {MPI.IDENT, MPI.CONGRUENT}:
        raise ValueError("Source and repartitioned meshes must use congruent communicators.")

    cell_tags = _redistribute_meshtags(source.mesh, source.cell_tags, target_mesh)
    facet_tags = _redistribute_meshtags(source.mesh, source.facet_tags, target_mesh)
    transferred = replace(
        artifacts,
        cell_tags=cell_tags,
        facet_tags=facet_tags,
        domain_markers=source.domain_markers,
        boundary_markers=source.boundary_markers,
    )
    _validate_named_marker_transfer(source, transferred)
    return transferred


def resolve_owned_child_input_positions(
    artifacts: RefinedMeshArtifacts,
) -> npt.NDArray[np.int32]:
    """Map owned refined cells to raw DOLFINx parent-array positions."""

    if not isinstance(artifacts, RefinedMeshArtifacts):
        raise TypeError("artifacts must be a RefinedMeshArtifacts.")
    if not artifacts.report.occurred:
        return np.empty(0, dtype=np.int32)
    if artifacts.input_cell_global_offset is None:
        raise ValueError(
            "Refined child normalization requires an input-cell global offset."
        )

    mesh = artifacts.refined_mesh
    tdim = mesh.topology.dim
    owned_children = mesh.topology.index_map(tdim).size_local
    original_indices = np.asarray(mesh.topology.original_cell_index)
    if original_indices.ndim != 1 or original_indices.size < owned_children:  # pragma: no cover - backend original cell indices cover owned children
        raise ValueError(
            "Refined topology.original_cell_index does not cover all owned child cells."
        )
    positions = original_indices[:owned_children].astype(np.int64, copy=False)
    positions = positions - int(artifacts.input_cell_global_offset)
    if np.any(positions < 0) or np.any(positions >= artifacts.parent_cell.size):
        raise ValueError("Owned refined cells resolve outside the parent-cell array.")
    if np.unique(positions).size != positions.size:  # pragma: no cover - backend original cell indices are unique
        raise ValueError("Owned refined cells do not resolve to unique parent-array positions.")
    return positions.astype(np.int32, copy=False)


def _transfer_cell_tags(source, artifacts, parent_cells):
    """PRIVATE: Inherit source cell tags through resolved parent cells."""

    if source.cell_tags is None:
        return None
    source_tags = _tag_lookup(source.cell_tags)
    child_indices = []
    child_values = []
    for child_cell, parent_cell in enumerate(parent_cells):
        value = source_tags.get(int(parent_cell))
        if value is not None:
            child_indices.append(child_cell)
            child_values.append(value)
    return _create_meshtags_like(
        artifacts.refined_mesh,
        artifacts.refined_mesh.topology.dim,
        child_indices,
        child_values,
        source.cell_tags.values.dtype,
    )


def _transfer_facet_tags(source, artifacts, child_input_positions, parent_cells):
    """PRIVATE: Inherit tagged source facets through parent-facet ancestry."""

    if source.facet_tags is None:
        return None
    old_mesh = source.mesh
    new_mesh = artifacts.refined_mesh
    tdim = old_mesh.topology.dim
    fdim = tdim - 1
    old_mesh.topology.create_connectivity(tdim, fdim)
    new_mesh.topology.create_connectivity(tdim, fdim)
    old_cell_to_facet = old_mesh.topology.connectivity(tdim, fdim)
    new_cell_to_facet = new_mesh.topology.connectivity(tdim, fdim)
    source_tags = _tag_lookup(source.facet_tags)

    if artifacts.parent_cell.size == 0:  # pragma: no cover - successful artifacts already require parent-cell ancestry
        raise ValueError("Facet transfer requires nonempty parent-cell ancestry.")
    facets_per_child = artifacts.parent_facet.size // artifacts.parent_cell.size
    if facets_per_child * artifacts.parent_cell.size != artifacts.parent_facet.size:
        raise ValueError("Flattened parent-facet ancestry has an invalid size.")

    owned_facets = new_mesh.topology.index_map(fdim).size_local
    assignments: dict[int, Any] = {}
    for child_cell, (input_position, parent_cell) in enumerate(
        zip(child_input_positions, parent_cells)
    ):
        new_facets = new_cell_to_facet.links(child_cell)
        old_facets = old_cell_to_facet.links(int(parent_cell))
        if new_facets.size != facets_per_child:
            raise ValueError("New cell facet count does not match parent-facet ancestry width.")
        for child_local_facet, new_facet in enumerate(new_facets):
            parent_local_facet = int(
                artifacts.parent_facet[
                    int(input_position) * facets_per_child + child_local_facet
                ]
            )
            if parent_local_facet == -1:
                continue
            if parent_local_facet < 0 or parent_local_facet >= old_facets.size:
                raise ValueError("Parent-facet ancestry contains an invalid local facet number.")
            if new_facet >= owned_facets:
                continue
            old_facet = int(old_facets[parent_local_facet])
            value = source_tags.get(old_facet)
            if value is not None:
                _record_tag_assignment(assignments, int(new_facet), value)

    facet_indices = sorted(assignments)
    facet_values = [assignments[index] for index in facet_indices]
    return _create_meshtags_like(
        new_mesh,
        fdim,
        facet_indices,
        facet_values,
        source.facet_tags.values.dtype,
    )


def _record_tag_assignment(assignments: dict[int, Any], entity: int, value: Any) -> None:
    """PRIVATE: Add one entity tag and reject conflicting duplicate ancestry."""

    if entity in assignments and assignments[entity] != value:
        raise ValueError(
            f"Conflicting transferred tags for refined entity {entity}: "
            f"{assignments[entity]} and {value}."
        )
    assignments[entity] = value


def _tag_lookup(tags) -> dict[int, Any]:
    """PRIVATE: Return local entity-to-tag lookup with duplicate validation."""

    lookup: dict[int, Any] = {}
    for entity, value in zip(np.asarray(tags.indices), np.asarray(tags.values)):
        _record_tag_assignment(lookup, int(entity), value.item())
    return lookup


def _create_meshtags_like(mesh, dimension, indices, values, value_dtype):
    """PRIVATE: Create sorted MeshTags while preserving the source value dtype."""

    index_array = np.asarray(indices, dtype=np.int32)
    value_array = np.asarray(values, dtype=value_dtype)
    return dmesh.meshtags(mesh, dimension, index_array, value_array)


def _redistribute_meshtags(source_mesh, source_tags, target_mesh):
    """PRIVATE: Route owned tagged entities to their new owner ranks."""

    if source_tags is None:
        return None
    comm = source_mesh.comm
    dimension = int(source_tags.dim)
    entity_nodes = None
    values = None
    value_dtype = np.asarray(source_tags.values).dtype
    local_error = None
    try:
        source_topology = source_mesh.topology
        tdim = source_topology.dim
        source_topology.create_connectivity(dimension, tdim)
        source_topology.create_connectivity(tdim, dimension)
        indices = np.asarray(source_tags.indices, dtype=np.int32)
        owned_entities = source_topology.index_map(dimension).size_local
        if np.any(indices < 0) or np.any(indices >= owned_entities):
            raise ValueError("Redistribution requires source MeshTags on owned entities only.")
        geometry_dofs = np.asarray(
            dmesh.entities_to_geometry(
                source_mesh,
                dimension,
                indices,
                permute=False,
            ),
            dtype=np.int32,
        )
        flat_dofs = geometry_dofs.reshape(-1)
        global_nodes = source_mesh.geometry.index_map().local_to_global(flat_dofs)
        entity_nodes = np.asarray(global_nodes, dtype=np.int64).reshape(geometry_dofs.shape)
        values = np.asarray(source_tags.values, dtype=value_dtype).copy()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_marker_error(comm, "redistribution packing", local_error)

    local_entities = None
    local_values = None
    local_error = None
    try:
        local_entities, local_values = io.distribute_entity_data(
            target_mesh,
            dimension,
            entity_nodes,
            values,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_marker_error(comm, "redistribution routing", local_error)

    tags = None
    local_error = None
    try:
        target_mesh.topology.create_connectivity(dimension, target_mesh.topology.dim)
        distributed_tags = dmesh.meshtags_from_entities(
            target_mesh,
            dimension,
            adjacencylist(np.asarray(local_entities, dtype=np.int32)),
            np.asarray(local_values, dtype=value_dtype),
        )
        owned_entities = target_mesh.topology.index_map(dimension).size_local
        owned_mask = np.asarray(distributed_tags.indices) < owned_entities
        tags = dmesh.meshtags(
            target_mesh,
            dimension,
            np.asarray(distributed_tags.indices[owned_mask], dtype=np.int32),
            np.asarray(distributed_tags.values[owned_mask], dtype=value_dtype),
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_marker_error(comm, "redistribution reconstruction", local_error)
    return tags


def _validate_named_marker_transfer(source, transferred) -> None:
    """PRIVATE: Validate global name presence and domain/boundary measures."""

    for name, tag in source.domain_markers.items():
        old_count = source.mesh.comm.allreduce(
            source.cell_tags.find(tag).size,
            op=MPI.SUM,
        )
        new_count = source.mesh.comm.allreduce(
            transferred.get_domain_cells(name).size,
            op=MPI.SUM,
        )
        if old_count > 0 and new_count == 0:
            raise ValueError(f"Transferred named domain '{name}' contains no cells.")
        _validate_measure_preserved(
            "domain",
            name,
            _tagged_measure(source.mesh, source.cell_tags, tag, "dx"),
            _tagged_measure(transferred.refined_mesh, transferred.cell_tags, tag, "dx"),
        )

    for name, tag in source.boundary_markers.items():
        old_count = source.mesh.comm.allreduce(
            source.facet_tags.find(tag).size,
            op=MPI.SUM,
        )
        new_count = source.mesh.comm.allreduce(
            transferred.get_boundary_facets(name).size,
            op=MPI.SUM,
        )
        if old_count > 0 and new_count == 0:
            raise ValueError(f"Transferred named boundary '{name}' contains no facets.")
        _validate_measure_preserved(
            "boundary",
            name,
            _tagged_measure(source.mesh, source.facet_tags, tag, "ds"),
            _tagged_measure(transferred.refined_mesh, transferred.facet_tags, tag, "ds"),
        )


def _tagged_measure(mesh, tags, tag, measure_name: str) -> float:
    """PRIVATE: Assemble one globally reduced tagged geometric measure."""

    measure = ufl.Measure(measure_name, domain=mesh, subdomain_data=tags)
    local_value = fem.assemble_scalar(
        fem.form(fem.Constant(mesh, 1.0) * measure(int(tag)))
    )
    return float(mesh.comm.allreduce(local_value, op=MPI.SUM))


def _validate_measure_preserved(kind: str, name: str, old_value: float, new_value: float) -> None:
    """PRIVATE: Reject marker transfer that changes a named geometric measure."""

    tolerance = 1.0e-11 * max(1.0, abs(old_value), abs(new_value))
    if not math.isclose(old_value, new_value, rel_tol=1.0e-10, abs_tol=tolerance):
        raise ValueError(
            f"Transferred named {kind} '{name}' changed measure from "
            f"{old_value} to {new_value}."
        )


def _raise_collective_marker_error(comm, stage: str, local_error: str | None) -> None:
    """PRIVATE: Raise the same rank-attributed transfer error on every rank."""

    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        details = "; ".join(
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        )
        raise ValueError(f"Marker-transfer {stage} failed: {details}")


def _validate_marker_mapping(name: str, markers: Mapping[str, int]) -> None:
    """PRIVATE: Validate stable name-to-integer marker mappings."""

    for marker_name, tag in markers.items():
        if not isinstance(marker_name, str) or not marker_name:
            raise ValueError(f"{name} keys must be nonempty strings.")
        if isinstance(tag, bool) or not isinstance(tag, Integral):
            raise ValueError(f"{name} values must be integer tags.")
