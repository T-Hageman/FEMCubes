"""Detached topology refinement for marked simplex cells."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from time import perf_counter
from types import MappingProxyType
from typing import Any

from dolfinx import mesh as dmesh
from mpi4py import MPI
import numpy as np
import numpy.typing as npt

from femcubes.core.mesh._mesh_groups import _find_tagged_entities, _marker_names, _resolve_marker_tags, _single_marker_tag
from femcubes.core.mesh.refinement.config import normalize_refinement_config
from femcubes.core.mesh.refinement.indicators import BufferedCellSelection, _SIZE_TOLERANCE_FACTOR
_SIMPLEX_CELL_TYPES = frozenset({"interval", "triangle", "tetrahedron"})


@dataclass(frozen=True)
class RefinementReport:
    """Topology statistics for one detached refinement attempt.

    Timings are rank-local wall seconds. Topology timing includes validation
    and mesh statistics; repartition timing includes the second field transfer.
    """

    occurred: bool
    local_marked_cells: int
    global_marked_cells: int
    local_marked_edges: int
    global_marked_edges: int
    local_cells_before: int
    global_cells_before: int
    local_cells_after: int
    global_cells_after: int
    min_owned_cells_after: int
    max_owned_cells_after: int
    mean_owned_cells_after: float
    imbalance_ratio_after: float
    old_min_h: float
    old_max_h: float
    new_min_h: float
    new_max_h: float
    levels_applied: int
    imbalance_ratio_before_repartition: float
    repartitioned: bool
    repartition_seconds: float
    topology_seconds: float

    @property
    def global_cells_added(self) -> int:
        """Return the global cell-count increase caused by this attempt."""

        return self.global_cells_after - self.global_cells_before


@dataclass(frozen=True)
class RefinedMeshArtifacts:
    """Detached mesh and ancestry returned by topology refinement.

    ``refined_mesh`` is the original mesh when ``report.occurred`` is false.
    Parent arrays and marked edges are then empty. For a successful refinement,
    parent arrays retain DOLFINx's raw refinement ordering. The stored input
    offset and ``original_cell_index`` together normalize owned children on
    every rank before marker transfer.
    """

    refined_mesh: Any
    parent_cell: npt.NDArray[np.int32]
    parent_facet: npt.NDArray[np.int8]
    marked_edges: npt.NDArray[np.int32]
    report: RefinementReport
    input_cell_global_offset: int | None
    cell_tags: Any = None
    facet_tags: Any = None
    domain_markers: Mapping[str, int] = field(default_factory=dict)
    boundary_markers: Mapping[str, int] = field(default_factory=dict)
    migration_data: Any = None

    def __post_init__(self) -> None:
        for name, values in (
            ("parent_cell", self.parent_cell),
            ("parent_facet", self.parent_facet),
            ("marked_edges", self.marked_edges),
        ):
            if values.ndim != 1:
                raise ValueError(f"Refinement {name} must be one-dimensional.")
        if self.report.occurred:
            if self.parent_cell.size == 0:
                raise ValueError("Successful refinement must provide parent-cell ancestry.")
        elif self.parent_cell.size or self.parent_facet.size or self.marked_edges.size:
            raise ValueError("A no-op refinement must not provide topology ancestry or marked edges.")
        if not isinstance(self.domain_markers, Mapping):
            raise TypeError("Refinement domain_markers must be a mapping.")
        if not isinstance(self.boundary_markers, Mapping):
            raise TypeError("Refinement boundary_markers must be a mapping.")
        object.__setattr__(self, "domain_markers", MappingProxyType(dict(self.domain_markers)))
        object.__setattr__(self, "boundary_markers", MappingProxyType(dict(self.boundary_markers)))

    def domain_names(self) -> list[str]:
        """Return transferred domain group and part names in registration order."""

        return _marker_names(self.domain_markers)

    def boundary_names(self) -> list[str]:
        """Return transferred boundary group and part names in registration order."""

        return _marker_names(self.boundary_markers)

    def get_domain_tags(self, name: str) -> tuple[int, ...]:
        """Resolve a transferred domain part or group name to its tags."""

        return _resolve_marker_tags(self.domain_markers, name, "Domain")

    def get_boundary_tags(self, name: str) -> tuple[int, ...]:
        """Resolve a transferred boundary part or group name to its tags."""

        return _resolve_marker_tags(self.boundary_markers, name, "Boundary")

    def get_domain_tag(self, name: str) -> int:
        """Resolve a transferred domain name to its integer tag."""

        return _single_marker_tag(self.domain_markers, name, "Domain", "get_domain_tags")

    def get_boundary_tag(self, name: str) -> int:
        """Resolve a transferred boundary name to its integer tag."""

        return _single_marker_tag(self.boundary_markers, name, "Boundary", "get_boundary_tags")

    def get_domain_cells(self, name: str) -> npt.NDArray[np.int32]:
        """Return refined cells carrying a named transferred domain tag."""

        if self.cell_tags is None:
            raise ValueError("Refined artifacts do not contain transferred cell tags.")
        return _find_tagged_entities(self.cell_tags, self.get_domain_tags(name))

    def get_boundary_facets(self, name: str) -> npt.NDArray[np.int32]:
        """Return refined facets carrying a named transferred boundary tag."""

        if self.facet_tags is None:
            raise ValueError("Refined artifacts do not contain transferred facet tags.")
        return _find_tagged_entities(self.facet_tags, self.get_boundary_tags(name))


def refine_mesh_topology(
    mesh,
    selection: BufferedCellSelection,
    config=None,
) -> RefinedMeshArtifacts:
    """Refine buffered owned cells and return detached DOLFINx artifacts.

    The input mesh is never mutated or installed into an FEMCubes ``Mesh``
    wrapper. Only incident edges longer than ``config.target_h`` are marked
    when a target is supplied; otherwise all incident edges are marked.
    DOLFINx's identity partitioner and ``parent_cell_and_facet`` ancestry are
    used. Conforming closure may split additional edges and cells.

    Every rank in ``mesh.comm`` must call this function. The input-cell global
    offset is the exclusive prefix sum of the local raw parent-array lengths;
    marker transfer uses it to normalize final owned-child ordering under MPI.
    """

    start = perf_counter()
    _validate_simplex_mesh(mesh)
    comm = mesh.comm
    tdim = mesh.topology.dim
    cell_map = mesh.topology.index_map(tdim)
    owned_cells_before = cell_map.size_local

    local_error = _topology_input_error(selection, owned_cells_before)
    target_h = None
    try:
        target_h = normalize_refinement_config(config).target_h
    except (ValueError, TypeError) as exc:
        local_error = str(exc)
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        details = "; ".join(
            f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None
        )
        raise ValueError(f"Invalid topology-refinement input: {details}")

    targets = comm.allgather(target_h)
    if any(target != targets[0] for target in targets):
        raise ValueError("Topology refinement target_h must be identical on every rank.")

    local_marked_cells = int(np.count_nonzero(selection.marked_mask))
    global_marked_cells = int(comm.allreduce(local_marked_cells, op=MPI.SUM))
    reported_global_counts = comm.allgather(int(selection.global_marked_count))
    if any(count != global_marked_cells for count in reported_global_counts):
        raise ValueError(
            "Invalid topology-refinement input: selection.global_marked_count is "
            f"inconsistent with the mesh communicator; received {reported_global_counts}, "
            f"reduced {global_marked_cells}."
        )

    old_min_h, old_max_h = _global_diameter_bounds(mesh)

    marked_edges = np.empty(0, dtype=np.int32)
    if global_marked_cells:
        mesh.topology.create_entities(1)
        mesh.topology.create_connectivity(tdim, 1)
        marked_edges = np.unique(
            dmesh.compute_incident_entities(mesh.topology, selection.marked_cells, tdim, 1)
        ).astype(np.int32, copy=False)
        if target_h is not None:
            tolerance = _SIZE_TOLERANCE_FACTOR*np.finfo(float).eps*max(1.0, float(target_h))
            mesh.topology.create_connectivity(1, tdim)
            lengths = mesh.h(1, marked_edges)
            marked_edges = marked_edges[lengths > float(target_h)+tolerance]
    global_edge_requests = comm.allreduce(int(marked_edges.size), op=MPI.SUM)

    if global_edge_requests == 0:
        report = _build_report(
            mesh_before=mesh,
            mesh_after=mesh,
            local_marked_cells=local_marked_cells,
            global_marked_cells=global_marked_cells,
            local_marked_edges=0,
            global_marked_edges=0,
            old_min_h=old_min_h,
            old_max_h=old_max_h,
            occurred=False,
            topology_start=start,
        )
        return RefinedMeshArtifacts(
            refined_mesh=mesh,
            parent_cell=np.empty(0, dtype=np.int32),
            parent_facet=np.empty(0, dtype=np.int8),
            marked_edges=np.empty(0, dtype=np.int32),
            report=report,
            input_cell_global_offset=0 if comm.size == 1 else None,
        )

    edge_map = mesh.topology.index_map(1)
    local_marked_edges = int(np.count_nonzero(marked_edges < edge_map.size_local))
    global_marked_edges = int(comm.allreduce(local_marked_edges, op=MPI.SUM))

    refined_mesh, parent_cell, parent_facet = dmesh.refine(
        mesh,
        edges=marked_edges,
        partitioner=dmesh.IdentityPartitionerPlaceholder(),
        option=dmesh.RefinementOption.parent_cell_and_facet,
    )
    parent_cell_array = np.asarray(parent_cell, dtype=np.int32)
    parent_facet_array = np.asarray(parent_facet, dtype=np.int8)
    input_cell_global_offset = comm.exscan(parent_cell_array.size, op=MPI.SUM)
    if input_cell_global_offset is None:
        input_cell_global_offset = 0

    report = _build_report(
        mesh_before=mesh,
        mesh_after=refined_mesh,
        local_marked_cells=local_marked_cells,
        global_marked_cells=global_marked_cells,
        local_marked_edges=local_marked_edges,
        global_marked_edges=global_marked_edges,
        old_min_h=old_min_h,
        old_max_h=old_max_h,
        occurred=True,
        topology_start=start,
    )
    artifacts = RefinedMeshArtifacts(
        refined_mesh=refined_mesh,
        parent_cell=parent_cell_array,
        parent_facet=parent_facet_array,
        marked_edges=marked_edges,
        report=report,
        input_cell_global_offset=int(input_cell_global_offset),
    )
    return artifacts


def _topology_input_error(selection: Any, owned_cell_count: int) -> str | None:
    """PRIVATE: Return a rank-local input error before topology collectives."""

    if not isinstance(selection, BufferedCellSelection):
        return "selection must be a BufferedCellSelection"
    expected_cells = np.arange(owned_cell_count, dtype=np.int32)
    if not np.array_equal(selection.cells, expected_cells):
        return "selection cells must use complete owned local cell ordering"
    return None


def _validate_simplex_mesh(mesh) -> None:
    """PRIVATE: Reject topology types unsupported by the refinement path."""

    topology = getattr(mesh, "topology", None)
    if topology is None:
        raise TypeError("mesh must be a DOLFINx mesh with a topology.")
    cell_type_name = getattr(topology.cell_type, "name", str(topology.cell_type))
    if cell_type_name not in _SIMPLEX_CELL_TYPES:
        supported = ", ".join(sorted(_SIMPLEX_CELL_TYPES))
        raise ValueError(
            f"Adaptive refinement requires a simplex mesh ({supported}); "
            f"received {cell_type_name!r}."
        )


def _global_diameter_bounds(mesh) -> tuple[float, float]:
    """PRIVATE: Return global min/max owned-cell diameters."""

    tdim = mesh.topology.dim
    owned_count = mesh.topology.index_map(tdim).size_local
    cells = np.arange(owned_count, dtype=np.int32)
    diameters = np.asarray(mesh.h(tdim, cells), dtype=float)
    local_min = float(np.min(diameters)) if diameters.size else math.inf
    local_max = float(np.max(diameters)) if diameters.size else -math.inf
    global_min = float(mesh.comm.allreduce(local_min, op=MPI.MIN))
    global_max = float(mesh.comm.allreduce(local_max, op=MPI.MAX))
    if not math.isfinite(global_min) or not math.isfinite(global_max):  # pragma: no cover - valid backend meshes contain cells globally
        raise ValueError("Cannot refine a mesh with no owned cells globally.")
    return global_min, global_max


def _build_report(
    *,
    mesh_before,
    mesh_after,
    local_marked_cells: int,
    global_marked_cells: int,
    local_marked_edges: int,
    global_marked_edges: int,
    old_min_h: float,
    old_max_h: float,
    occurred: bool,
    topology_start: float,
) -> RefinementReport:
    """PRIVATE: Build rank-consistent topology and ownership statistics."""

    comm = mesh_after.comm
    tdim = mesh_after.topology.dim
    before_map = mesh_before.topology.index_map(mesh_before.topology.dim)
    after_map = mesh_after.topology.index_map(tdim)
    local_after = int(after_map.size_local)
    owned_counts = comm.allgather(local_after)
    mean_owned = float(sum(owned_counts) / len(owned_counts))
    new_min_h, new_max_h = _global_diameter_bounds(mesh_after)
    imbalance_ratio = float(max(owned_counts) / mean_owned)
    return RefinementReport(
        occurred=occurred,
        local_marked_cells=local_marked_cells,
        global_marked_cells=global_marked_cells,
        local_marked_edges=local_marked_edges,
        global_marked_edges=global_marked_edges,
        local_cells_before=int(before_map.size_local),
        global_cells_before=int(before_map.size_global),
        local_cells_after=local_after,
        global_cells_after=int(after_map.size_global),
        min_owned_cells_after=int(min(owned_counts)),
        max_owned_cells_after=int(max(owned_counts)),
        mean_owned_cells_after=mean_owned,
        imbalance_ratio_after=imbalance_ratio,
        old_min_h=old_min_h,
        old_max_h=old_max_h,
        new_min_h=new_min_h,
        new_max_h=new_max_h,
        levels_applied=int(occurred),
        imbalance_ratio_before_repartition=imbalance_ratio,
        repartitioned=False,
        repartition_seconds=0.0,
        topology_seconds=perf_counter() - topology_start,
    )
