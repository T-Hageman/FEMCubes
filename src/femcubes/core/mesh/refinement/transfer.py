"""Detached transfer of registered finite-element fields between meshes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from time import perf_counter
from types import MappingProxyType
from typing import Any

from dolfinx import cpp, fem, la
from mpi4py import MPI
import numpy as np
import ufl

from femcubes.core.mesh._mesh_dofs import FieldDefinition, _MeshDOFs
from femcubes.core.mesh.refinement.config import RefinementConfig
from femcubes.core.mesh.refinement.quadrature import (
    QuadratureReconstruction,
    get_quadrature_reconstruction,
    reconstruct_scalar_quadrature_samples,
)
from femcubes.core.mesh.refinement.topology import RefinedMeshArtifacts


@dataclass(frozen=True)
class FieldTransferSnapshot:
    """Immutable registry snapshot taken before one detached remesh transfer.

    Function objects intentionally remain references to the source-mesh
    storage. The registry mappings and step membership are copied so later
    live-wrapper mutation cannot change the transaction contract.
    """

    source_mesh: Any
    fields: Mapping[str, fem.Function]
    field_definitions: Mapping[str, FieldDefinition]
    step_fields: Mapping[int, tuple[str, ...]]

    def __post_init__(self) -> None:
        fields = dict(self.fields)
        definitions = dict(self.field_definitions)
        steps = {step: tuple(names) for step, names in self.step_fields.items()}
        _validate_snapshot_contract(self.source_mesh, fields, definitions, steps)
        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "field_definitions", MappingProxyType(definitions))
        object.__setattr__(self, "step_fields", MappingProxyType(steps))

    @property
    def standard_field_names(self) -> tuple[str, ...]:
        """Return registered CG/DG field names in stable registration order."""

        return tuple(
            name
            for name, definition in self.field_definitions.items()
            if definition.family in {"CG", "DG"}
        )

    @property
    def quadrature_field_names(self) -> tuple[str, ...]:
        """Return registered quadrature field names in stable order."""

        return tuple(
            name
            for name, definition in self.field_definitions.items()
            if definition.family == "Quadrature"
        )


@dataclass(frozen=True)
class FieldTransferDiagnostic:
    """Global field statistics and rank-local transfer timings.

    Detailed timings are populated by the initial CG/DG and quadrature
    interpolation routines. Space creation includes function allocation.
    ``search_seconds`` is zero for transfers using refinement ancestry. Direct
    copies are timed separately from interpolation. Exact
    repartitioning migration reports only its overall ``transfer_seconds``.
    """

    name: str
    family: str
    degree: int
    value_shape: tuple[int, ...]
    global_source_dofs: int
    global_destination_dofs: int
    global_source_min: float
    global_source_max: float
    global_destination_min: float
    global_destination_max: float
    global_source_integral: float | None
    global_destination_integral: float | None
    transfer_seconds: float
    finite: bool
    quadrature_degree: int | None = None
    reconstruction_degree: int | None = None
    space_creation_seconds: float = 0.0
    search_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    interpolation_seconds: float = 0.0
    synchronization_seconds: float = 0.0
    diagnostics_seconds: float = 0.0
    copy_seconds: float = 0.0


@dataclass(frozen=True)
class TransferredFieldState:
    """Detached recreated fields and metadata for a future mesh installation.

    Standard transfer populates every CG/DG field first. Quadrature transfer
    subsequently extends the same immutable state. ``pending_field_names``
    exposes any definitions not yet represented by target-mesh functions.
    ``source_mesh`` is the immediate interpolation source, while
    ``transaction_source_mesh`` retains the original live mesh across an
    optional second transfer through a redistributed mesh.
    """

    source_mesh: Any
    target_mesh: Any
    fields: Mapping[str, fem.Function]
    field_definitions: Mapping[str, FieldDefinition]
    step_fields: Mapping[int, tuple[str, ...]]
    diagnostics: Mapping[str, FieldTransferDiagnostic]
    transaction_source_mesh: Any = None

    def __post_init__(self) -> None:
        fields = dict(self.fields)
        definitions = dict(self.field_definitions)
        steps = {step: tuple(names) for step, names in self.step_fields.items()}
        diagnostics = dict(self.diagnostics)
        transaction_source_mesh = (
            self.source_mesh
            if self.transaction_source_mesh is None
            else self.transaction_source_mesh
        )

        definition_names = set(definitions)
        field_names = set(fields)
        standard_names = {
            name
            for name, definition in definitions.items()
            if definition.family in {"CG", "DG"}
        }
        if not standard_names.issubset(field_names) or not field_names.issubset(definition_names):
            missing = sorted(standard_names - field_names)
            unexpected = sorted(field_names - definition_names)
            raise ValueError(
                "Transferred field set does not match its definitions: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if set(diagnostics) != field_names:
            raise ValueError("Every transferred field requires exactly one diagnostic.")
        for name, field in fields.items():
            if field.function_space.mesh is not self.target_mesh:
                raise ValueError(f"Transferred field '{name}' is not defined on the target mesh.")
            if tuple(field.function_space.element.value_shape) != definitions[name].value_shape:
                raise ValueError(f"Transferred field '{name}' value shape disagrees with its definition.")

        object.__setattr__(self, "fields", MappingProxyType(fields))
        object.__setattr__(self, "field_definitions", MappingProxyType(definitions))
        object.__setattr__(self, "step_fields", MappingProxyType(steps))
        object.__setattr__(self, "diagnostics", MappingProxyType(diagnostics))
        object.__setattr__(self, "transaction_source_mesh", transaction_source_mesh)

    @property
    def pending_field_names(self) -> tuple[str, ...]:
        """Return registered fields not yet present in this detached state."""

        return tuple(name for name in self.field_definitions if name not in self.fields)


def snapshot_registered_fields(mesh_wrapper) -> FieldTransferSnapshot:
    """Snapshot all registered field storage.

    Every field owns its storage, so the current function references and
    mesh-independent definitions are captured directly. Fields restricted
    to subdomains or interfaces are captured as parent-mesh proxies, and
    global fields (not tied to the mesh) are left out. The live registries
    themselves are not replaced or reordered.
    """

    required = ("mesh", "fields", "field_definitions", "step_fields")
    missing = [name for name in required if not hasattr(mesh_wrapper, name)]
    if missing:
        raise TypeError(f"Mesh field snapshot requires attributes: {', '.join(missing)}.")

    comm = mesh_wrapper.mesh.comm
    local_error = None
    try:
        # Restricted fields take part as parent-mesh proxies; global fields are not transferred.
        fields, definitions, step_fields = mesh_wrapper._refinement_field_registry()
        snapshot = FieldTransferSnapshot(
            source_mesh=mesh_wrapper.mesh,
            fields=fields,
            field_definitions=definitions,
            step_fields=step_fields,
        )
    except Exception as exc:
        snapshot = None
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_transfer_error(comm, "field snapshot", local_error)

    signature = (
        tuple(snapshot.field_definitions.items()),
        tuple(snapshot.step_fields.items()),
    )
    signatures = comm.allgather(signature)
    if any(candidate != signatures[0] for candidate in signatures):
        raise ValueError(
            "Field definitions and step membership must be identical on every mesh rank."
        )
    return snapshot


def transfer_standard_fields(
    snapshot: FieldTransferSnapshot,
    artifacts: RefinedMeshArtifacts,
    config: RefinementConfig | None = None,
) -> TransferredFieldState:
    """Transfer every registered CG/DG field to a detached refined mesh.

    Unchanged cells retain their coefficients directly. Only children of
    split parents, including conformity closure, require interpolation.
    DOLFINx evaluates the coarse function in each child's known parent cell
    and applies the destination interpolation operator without point searches.
    Source functions and live mesh-wrapper registries remain intact.
    """

    if not isinstance(snapshot, FieldTransferSnapshot):
        raise TypeError("snapshot must be a FieldTransferSnapshot.")
    if not isinstance(artifacts, RefinedMeshArtifacts):
        raise TypeError("artifacts must be a RefinedMeshArtifacts.")
    if config is None:
        config = RefinementConfig()
    if not isinstance(config, RefinementConfig):
        raise TypeError("config must be a RefinementConfig or None.")

    source_mesh = snapshot.source_mesh
    target_mesh = artifacts.refined_mesh
    comm = source_mesh.comm
    if MPI.Comm.Compare(comm, target_mesh.comm) not in {MPI.IDENT, MPI.CONGRUENT}:
        raise ValueError("Source and target meshes must use congruent communicators.")

    paddings = comm.allgather(float(config.interpolation_padding))
    if any(padding != paddings[0] for padding in paddings):
        raise ValueError(
            "Refinement interpolation_padding must be identical on every rank; "
            f"received {paddings}."
        )

    plan = _create_refinement_transfer_plan(source_mesh, artifacts) if snapshot.standard_field_names else None
    destination_spaces: dict[tuple[str, int, tuple[int, ...]], Any] = {}
    transferred_fields: dict[str, fem.Function] = {}
    diagnostics: dict[str, FieldTransferDiagnostic] = {}

    for name in snapshot.standard_field_names:
        start = perf_counter()
        source_field = snapshot.fields[name]
        definition = snapshot.field_definitions[name]
        stage_start = perf_counter()
        source_stats = _field_statistics(source_field)
        diagnostics_seconds = perf_counter() - stage_start
        if not source_stats[3]:
            bad_ranks = comm.allgather(
                comm.rank if not _local_owned_coefficients_finite(source_field) else None
            )
            bad_ranks = [rank for rank in bad_ranks if rank is not None]
            raise ValueError(
                f"Source field '{name}' contains non-finite owned coefficients "
                f"on rank(s) {bad_ranks}."
            )

        search_seconds = 0.0
        interpolation_seconds = 0.0
        copy_seconds = 0.0
        synchronization_seconds = 0.0
        if artifacts.report.occurred:
            stage_start = perf_counter()
            source_field.x.scatter_forward()
            synchronization_seconds += perf_counter() - stage_start
        local_error = None
        destination = None
        try:
            stage_start = perf_counter()
            space_key = (definition.family, definition.degree, definition.value_shape)
            destination_space = destination_spaces.get(space_key)
            if destination_space is None:
                destination = _MeshDOFs._create_function_from_definition(
                    definition,
                    target_mesh,
                )
                destination_space = destination.function_space
                destination_spaces[space_key] = destination_space
            else:
                destination = fem.Function(
                    destination_space,
                    name=definition.description,
                )
            space_creation_seconds = perf_counter() - stage_start

            if artifacts.report.occurred:
                stage_start = perf_counter()
                _interpolate_from_parent_cells(source_field, destination, plan)
                interpolation_seconds = perf_counter() - stage_start
                stage_start = perf_counter()
                _copy_unchanged_cell_coefficients(source_field, destination, plan)
                copy_seconds = perf_counter() - stage_start
            else:
                if target_mesh is not source_mesh:
                    raise ValueError("No-op refinement artifacts must retain the source mesh.")
                stage_start = perf_counter()
                destination.x.array[:] = source_field.x.array
                copy_seconds = perf_counter() - stage_start
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _raise_collective_transfer_error(
            comm,
            f"CG/DG interpolation for field '{name}'",
            local_error,
        )
        stage_start = perf_counter()
        destination.x.scatter_forward()
        synchronization_seconds += perf_counter() - stage_start

        stage_start = perf_counter()
        destination_stats = _field_statistics(destination)
        diagnostics_seconds += perf_counter() - stage_start
        if not destination_stats[3]:
            bad_ranks = comm.allgather(
                comm.rank if not _local_owned_coefficients_finite(destination) else None
            )
            bad_ranks = [rank for rank in bad_ranks if rank is not None]
            raise ValueError(
                f"Transferred field '{name}' contains non-finite owned coefficients "
                f"on rank(s) {bad_ranks}."
            )

        stage_start = perf_counter()
        source_integral = _global_scalar_integral(source_field)
        destination_integral = _global_scalar_integral(destination)
        diagnostics_seconds += perf_counter() - stage_start
        transferred_fields[name] = destination
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
            global_source_integral=source_integral,
            global_destination_integral=destination_integral,
            transfer_seconds=perf_counter() - start,
            finite=True,
            space_creation_seconds=space_creation_seconds,
            search_seconds=search_seconds,
            interpolation_seconds=interpolation_seconds,
            copy_seconds=copy_seconds,
            synchronization_seconds=synchronization_seconds,
            diagnostics_seconds=diagnostics_seconds,
        )

    transferred_fields = {
        name: transferred_fields[name]
        for name in snapshot.field_definitions
        if name in transferred_fields
    }
    diagnostics = {
        name: diagnostics[name]
        for name in snapshot.field_definitions
        if name in diagnostics
    }
    return TransferredFieldState(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        fields=transferred_fields,
        field_definitions=snapshot.field_definitions,
        step_fields=snapshot.step_fields,
        diagnostics=diagnostics,
    )


def transfer_quadrature_fields(
    snapshot: FieldTransferSnapshot,
    artifacts: RefinedMeshArtifacts,
    transferred_state: TransferredFieldState,
    config: RefinementConfig | None = None,
) -> TransferredFieldState:
    """Reconstruct and transfer every registered quadrature field.

    Unchanged cells retain their original samples without reconstruction.
    Only parents of split children are reconstructed into a same-shaped DG
    function, then evaluated directly in those parents. Components are flattened
    in C order, and reconstruction runs in bounded batches. Owned source
    coefficients and the input detached state remain intact.
    """

    if not isinstance(snapshot, FieldTransferSnapshot):
        raise TypeError("snapshot must be a FieldTransferSnapshot.")
    if not isinstance(artifacts, RefinedMeshArtifacts):
        raise TypeError("artifacts must be a RefinedMeshArtifacts.")
    if not isinstance(transferred_state, TransferredFieldState):
        raise TypeError("transferred_state must be a TransferredFieldState.")
    if config is None:
        config = RefinementConfig()
    if not isinstance(config, RefinementConfig):
        raise TypeError("config must be a RefinementConfig or None.")

    source_mesh = snapshot.source_mesh
    target_mesh = artifacts.refined_mesh
    comm = source_mesh.comm
    if transferred_state.source_mesh is not source_mesh:
        raise ValueError("Transferred field state belongs to a different source mesh.")
    if transferred_state.target_mesh is not target_mesh:
        raise ValueError("Transferred field state belongs to a different target mesh.")
    if tuple(transferred_state.field_definitions.items()) != tuple(snapshot.field_definitions.items()):
        raise ValueError("Transferred field definitions differ from the source snapshot.")
    if tuple(transferred_state.step_fields.items()) != tuple(snapshot.step_fields.items()):
        raise ValueError("Transferred step membership differs from the source snapshot.")
    if MPI.Comm.Compare(comm, target_mesh.comm) not in {MPI.IDENT, MPI.CONGRUENT}:
        raise ValueError("Source and target meshes must use congruent communicators.")

    paddings = comm.allgather(float(config.interpolation_padding))
    if any(padding != paddings[0] for padding in paddings):
        raise ValueError(
            "Refinement interpolation_padding must be identical on every rank; "
            f"received {paddings}."
        )
    already_transferred = [
        name for name in snapshot.quadrature_field_names if name in transferred_state.fields
    ]
    if already_transferred:
        raise ValueError(
            "Quadrature fields are already present in the transferred state: "
            f"{already_transferred}."
        )

    plan = _create_refinement_transfer_plan(source_mesh, artifacts) if snapshot.quadrature_field_names else None
    reconstruction_spaces: dict[tuple[int, tuple[int, ...]], Any] = {}
    destination_spaces: dict[tuple[int, tuple[int, ...]], Any] = {}
    evaluation_cells = np.unique(plan.parent_cells[plan.split_cells]) if plan is not None else None
    transferred_fields = dict(transferred_state.fields)
    diagnostics = dict(transferred_state.diagnostics)

    for name in snapshot.quadrature_field_names:
        start = perf_counter()
        source_field = snapshot.fields[name]
        definition = snapshot.field_definitions[name]
        stage_start = perf_counter()
        source_stats = _field_statistics(source_field)
        diagnostics_seconds = perf_counter() - stage_start
        if not source_stats[3]:
            bad_ranks = comm.allgather(
                comm.rank if not _local_owned_coefficients_finite(source_field) else None
            )
            bad_ranks = [rank for rank in bad_ranks if rank is not None]
            raise ValueError(
                f"Source quadrature field '{name}' contains non-finite owned coefficients "
                f"on rank(s) {bad_ranks}."
            )

        reconstruction_seconds = 0.0
        search_seconds = 0.0
        interpolation_seconds = 0.0
        copy_seconds = 0.0
        synchronization_seconds = 0.0
        if artifacts.report.occurred:
            stage_start = perf_counter()
            source_field.x.scatter_forward()
            synchronization_seconds += perf_counter() - stage_start
        reconstruction = None
        destination = None
        reconstructed = None
        destination_space = None
        reconstruction_space = None
        local_error = None
        try:
            stage_start = perf_counter()
            destination_key = (definition.quadrature_degree, definition.value_shape)
            destination_space = destination_spaces.get(destination_key)
            if destination_space is None:
                destination = _MeshDOFs._create_function_from_definition(
                    definition,
                    target_mesh,
                )
                destination_space = destination.function_space
                destination_spaces[destination_key] = destination_space
            else:
                destination = fem.Function(destination_space, name=definition.description)
            space_creation_seconds = perf_counter() - stage_start

            if artifacts.report.occurred:
                stage_start = perf_counter()
                source_points = np.asarray(
                    source_field.function_space.element.interpolation_points,
                    dtype=np.float64,
                )
                reconstruction = get_quadrature_reconstruction(
                    source_mesh.basix_cell(),
                    definition.quadrature_degree,
                    points=source_points,
                )
                reconstruction_seconds += perf_counter() - stage_start
                stage_start = perf_counter()
                reconstruction_key = (reconstruction.degree, definition.value_shape)
                reconstruction_space = reconstruction_spaces.get(reconstruction_key)
                if reconstruction_space is None:
                    dg_definition = FieldDefinition(
                        name=f"{name}__remesh_reconstruction",
                        description=f"{definition.description} reconstruction",
                        family="DG",
                        degree=reconstruction.degree,
                        value_shape=definition.value_shape,
                    )
                    reconstructed = _MeshDOFs._create_function_from_definition(
                        dg_definition,
                        source_mesh,
                    )
                    reconstruction_space = reconstructed.function_space
                    reconstruction_spaces[reconstruction_key] = reconstruction_space
                else:
                    reconstructed = fem.Function(
                        reconstruction_space,
                        name=f"{definition.description} reconstruction",
                    )
                space_creation_seconds += perf_counter() - stage_start
                stage_start = perf_counter()
                _populate_reconstructed_dg_field(
                    source_field,
                    reconstructed,
                    reconstruction,
                    definition.value_shape,
                    evaluation_cells,
                )
                reconstruction_seconds += perf_counter() - stage_start
            else:
                if target_mesh is not source_mesh:
                    raise ValueError("No-op refinement artifacts must retain the source mesh.")
                stage_start = perf_counter()
                destination.x.array[:] = source_field.x.array
                copy_seconds = perf_counter() - stage_start
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        _raise_collective_transfer_error(
            comm,
            f"quadrature reconstruction for field '{name}'",
            local_error,
        )

        if artifacts.report.occurred:
            local_error = None
            try:
                stage_start = perf_counter()
                _interpolate_from_parent_cells(reconstructed, destination, plan)
                interpolation_seconds = perf_counter() - stage_start
                stage_start = perf_counter()
                _copy_unchanged_cell_coefficients(source_field, destination, plan)
                copy_seconds = perf_counter() - stage_start
            except Exception as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            _raise_collective_transfer_error(
                comm,
                f"quadrature interpolation for field '{name}'",
                local_error,
            )
        stage_start = perf_counter()
        destination.x.scatter_forward()
        synchronization_seconds += perf_counter() - stage_start

        stage_start = perf_counter()
        destination_stats = _field_statistics(destination)
        diagnostics_seconds += perf_counter() - stage_start
        if not destination_stats[3]:
            bad_ranks = comm.allgather(
                comm.rank if not _local_owned_coefficients_finite(destination) else None
            )
            bad_ranks = [rank for rank in bad_ranks if rank is not None]
            raise ValueError(
                f"Transferred quadrature field '{name}' contains non-finite owned coefficients "
                f"on rank(s) {bad_ranks}."
            )

        transferred_fields[name] = destination
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
            global_source_integral=None,
            global_destination_integral=None,
            transfer_seconds=perf_counter() - start,
            finite=True,
            quadrature_degree=definition.quadrature_degree,
            reconstruction_degree=(
                reconstruction.degree if reconstruction is not None else None
            ),
            space_creation_seconds=space_creation_seconds,
            search_seconds=search_seconds,
            reconstruction_seconds=reconstruction_seconds,
            interpolation_seconds=interpolation_seconds,
            copy_seconds=copy_seconds,
            synchronization_seconds=synchronization_seconds,
            diagnostics_seconds=diagnostics_seconds,
        )

    transferred_fields = {
        field_name: transferred_fields[field_name]
        for field_name in snapshot.field_definitions
        if field_name in transferred_fields
    }
    diagnostics = {
        field_name: diagnostics[field_name]
        for field_name in snapshot.field_definitions
        if field_name in diagnostics
    }
    return TransferredFieldState(
        source_mesh=source_mesh,
        target_mesh=target_mesh,
        fields=transferred_fields,
        field_definitions=snapshot.field_definitions,
        step_fields=snapshot.step_fields,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class _RefinementTransferPlan:
    """PRIVATE: Local parent lookup and actual split/unchanged target cells."""

    parent_cells: np.ndarray
    split_cells: np.ndarray
    unchanged_cells: np.ndarray


def _interpolate_from_parent_cells(source, destination, plan):
    """PRIVATE: Evaluate split children in their parents using native operators.

    Callable interpolation supplies physical points in cell-major order and
    applies the destination element's interpolation matrix, including for
    Bernstein elements whose coefficients are not point values. Explicit
    source cell indices bypass geometric search and select the parent-side
    trace of discontinuous fields at coarse-cell faces.
    """

    point_count = destination.function_space.element.interpolation_points.shape[0]
    value_size = source.function_space.value_size
    for start in range(0, plan.split_cells.size, 512):
        cells = plan.split_cells[start:start + 512]
        evaluation_cells = np.repeat(plan.parent_cells[cells], point_count)

        def evaluate(points):
            values = source.eval(np.ascontiguousarray(points.T), evaluation_cells)
            return np.ascontiguousarray(values.reshape(-1, value_size).T)

        destination.interpolate(evaluate, cells0=cells)


def _create_refinement_transfer_plan(source_mesh, artifacts):
    """PRIVATE: Resolve ancestry on owned and ghost children before transfer."""

    if not artifacts.report.occurred:
        return None
    comm = source_mesh.comm
    source_map = source_mesh.topology.index_map(source_mesh.topology.dim)
    target_mesh = artifacts.refined_mesh
    target_map = target_mesh.topology.index_map(target_mesh.topology.dim)
    parents = artifacts.parent_cell
    counts = comm.allgather(parents.size)
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    input_ids = np.asarray(target_mesh.topology.original_cell_index, dtype=np.int64)
    local_error = None
    if np.any(parents < 0) or np.any(parents >= source_map.size_local):
        local_error = "Refinement parents must index owned source cells."
    elif (offsets[-1] != target_map.size_global
            or artifacts.input_cell_global_offset != offsets[comm.rank]
            or input_ids.size != target_map.size_local + target_map.num_ghosts
            or np.any(input_ids < 0) or np.any(input_ids >= offsets[-1])):
        local_error = "Refinement input-cell IDs do not match the parent ancestry."
    _raise_collective_transfer_error(comm, "refinement transfer ancestry", local_error)

    # DOLFINx reorders children and creates ghosts, whereas parent_cell is
    # in raw refinement-input order. Exchange only this integer metadata.
    remote = (input_ids < offsets[comm.rank]) | (input_ids >= offsets[comm.rank + 1])
    remote_ids = np.ascontiguousarray(input_ids[remote])
    owners = np.searchsorted(offsets[1:], remote_ids, side="right").astype(np.int32)
    index_map = cpp.common.IndexMap(comm, parents.size, remote_ids, owners, 0)
    ancestry = la.vector(index_map, bs=2, dtype=np.int64)
    owned = ancestry.array[:2 * parents.size].reshape(-1, 2)
    owned[:, 0] = source_map.local_to_global(parents)
    multiplicities = np.bincount(parents, minlength=source_map.size_local)
    owned[:, 1] = multiplicities[parents] > 1
    ancestry.scatter_forward()
    metadata = ancestry.array.reshape(-1, 2)[index_map.global_to_local(input_ids)]
    parent_cells = source_map.global_to_local(np.ascontiguousarray(metadata[:, 0]))
    split_cells = np.flatnonzero(metadata[:, 1]).astype(np.int32)
    unchanged_cells = np.flatnonzero(metadata[:, 1] == 0).astype(np.int32)

    local_error = None
    if np.any(parent_cells < 0):
        local_error = "A target cell has no local owned/ghost source parent."
    else:
        for start in range(0, unchanged_cells.size, 512):
            cells = unchanged_cells[start:start + 512]
            old_geometry = source_mesh.geometry.x[source_mesh.geometry.dofmap[parent_cells[cells]]]
            new_geometry = target_mesh.geometry.x[target_mesh.geometry.dofmap[cells]]
            if not np.array_equal(old_geometry, new_geometry):
                local_error = "An unchanged cell changed geometry or reference-cell ordering."
                break
    _raise_collective_transfer_error(comm, "unchanged-cell transfer geometry", local_error)
    return _RefinementTransferPlan(parent_cells, split_cells, unchanged_cells)


def _copy_unchanged_cell_coefficients(source, destination, plan):
    """PRIVATE: Copy matching cell-local coefficients in bounded batches."""

    source_dofmap = source.function_space.dofmap
    target_dofmap = destination.function_space.dofmap
    block_size = source_dofmap.index_map_bs
    if (source_dofmap.bs != block_size
            or target_dofmap.bs != block_size
            or target_dofmap.index_map_bs != block_size
            or source_dofmap.list.shape[1] != target_dofmap.list.shape[1]):
        raise ValueError("Unchanged-cell transfer requires matching blocked coefficient layouts.")
    source_values = source.x.array.reshape(-1, block_size)
    target_values = destination.x.array.reshape(-1, block_size)
    for start in range(0, plan.unchanged_cells.size, 512):
        cells = plan.unchanged_cells[start:start + 512]
        target_values[target_dofmap.list[cells]] = source_values[
            source_dofmap.list[plan.parent_cells[cells]]
        ]


def _populate_reconstructed_dg_field(
    source_field: fem.Function,
    reconstructed: fem.Function,
    reconstruction: QuadratureReconstruction,
    value_shape: tuple[int, ...],
    cells: np.ndarray,
) -> None:
    """PRIVATE: Reconstruct only required parent cells, in bounded batches."""

    source_space = source_field.function_space
    reconstructed_space = reconstructed.function_space
    source_mesh = source_space.mesh
    if reconstructed_space.mesh is not source_mesh:  # pragma: no cover - internal reconstruction is created on the source mesh
        raise ValueError("Quadrature reconstruction fields must share the source mesh.")
    value_size = int(math.prod(value_shape)) if value_shape else 1
    source_dofmap = source_space.dofmap
    reconstructed_dofmap = reconstructed_space.dofmap
    for label, dofmap in (
        ("quadrature", source_dofmap),
        ("DG reconstruction", reconstructed_dofmap),
    ):
        if dofmap.bs != value_size or dofmap.index_map_bs != value_size:  # pragma: no cover - backend block sizes match the constructed value shape
            raise ValueError(
                f"{label} field block sizes bs={dofmap.bs}, "
                f"index_map_bs={dofmap.index_map_bs}; expected flattened "
                f"value size {value_size}."
            )

    if (source_dofmap.list.shape[1] != reconstruction.sample_count
            or reconstructed_dofmap.list.shape[1] != reconstruction.coefficient_count):  # pragma: no cover - backend dof counts match the constructed elements
        raise ValueError("Quadrature/DG cell dof counts disagree with the reconstruction.")
    source_values = source_field.x.array.reshape(-1, value_size)
    destination_values = reconstructed.x.array.reshape(-1, value_size)
    for start in range(0, cells.size, 512):
        batch = cells[start:start + 512]
        samples_by_component = source_values[source_dofmap.list[batch]].swapaxes(1, 2)
        coefficients_by_component = reconstruct_scalar_quadrature_samples(
            reconstruction,
            samples_by_component,
        )
        destination_values[reconstructed_dofmap.list[batch]] = coefficients_by_component.swapaxes(1, 2)


def _validate_snapshot_contract(source_mesh, fields, definitions, steps) -> None:
    """PRIVATE: Validate one copied field registry before collective use."""

    if getattr(source_mesh, "topology", None) is None:
        raise TypeError("Field snapshot source_mesh must expose a DOLFINx topology.")
    if set(fields) != set(definitions):
        missing = sorted(set(fields) - set(definitions))
        stale = sorted(set(definitions) - set(fields))
        raise ValueError(
            "Every registered field requires exactly one FieldDefinition: "
            f"missing={missing}, stale={stale}."
        )
    for name, definition in definitions.items():
        if not isinstance(definition, FieldDefinition) or definition.name != name:
            raise ValueError(f"Invalid FieldDefinition stored for field '{name}'.")
        field = fields[name]
        if not isinstance(field, fem.Function):
            raise TypeError(f"Registered field '{name}' must be a dolfinx.fem.Function.")
        if field.function_space.mesh is not source_mesh:
            raise ValueError(f"Registered field '{name}' is not defined on the source mesh.")
        if tuple(field.function_space.element.value_shape) != definition.value_shape:
            raise ValueError(f"Registered field '{name}' value shape disagrees with its definition.")
    for step, names in steps.items():
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("Field snapshot step identifiers must be integers.")
        if len(set(names)) != len(names):
            raise ValueError(f"Field snapshot step {step} contains duplicate field names.")
        unknown = [name for name in names if name not in fields]
        if unknown:
            raise ValueError(f"Field snapshot step {step} references unknown fields: {unknown}.")


def _owned_coefficients(field: fem.Function) -> np.ndarray:
    """PRIVATE: Return the owned scalar coefficient prefix for one function."""

    dofmap = field.function_space.dofmap
    owned_size = dofmap.index_map.size_local * dofmap.index_map_bs
    if owned_size > field.x.array.size:  # pragma: no cover - backend vector covers its owned degrees of freedom
        raise ValueError("Function vector does not cover its owned degrees of freedom.")
    return field.x.array[:owned_size]


def _local_owned_coefficients_finite(field: fem.Function) -> bool:
    """PRIVATE: Return whether this rank's owned field coefficients are finite."""

    return bool(np.all(np.isfinite(_owned_coefficients(field))))


def _field_statistics(field: fem.Function) -> tuple[int, float, float, bool]:
    """PRIVATE: Return global scalar-DOF count, extrema, and finiteness."""

    comm = field.function_space.mesh.comm
    owned = _owned_coefficients(field)
    local_finite = bool(np.all(np.isfinite(owned)))
    finite = bool(comm.allreduce(local_finite, op=MPI.LAND))
    local_min = float(np.min(owned)) if owned.size and local_finite else math.inf
    local_max = float(np.max(owned)) if owned.size and local_finite else -math.inf
    global_min = float(comm.allreduce(local_min, op=MPI.MIN))
    global_max = float(comm.allreduce(local_max, op=MPI.MAX))
    dofmap = field.function_space.dofmap
    global_dofs = int(dofmap.index_map.size_global * dofmap.index_map_bs)
    return global_dofs, global_min, global_max, finite


def _global_scalar_integral(field: fem.Function) -> float | None:
    """PRIVATE: Assemble a diagnostic global integral for scalar fields."""

    if tuple(field.function_space.element.value_shape):
        return None
    mesh = field.function_space.mesh
    local_value = fem.assemble_scalar(fem.form(field * ufl.dx))
    return float(mesh.comm.allreduce(local_value, op=MPI.SUM))


def _raise_collective_transfer_error(comm, stage: str, local_error: str | None) -> None:
    """PRIVATE: Raise the same rank-attributed field-transfer error everywhere."""

    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        details = "; ".join(
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        )
        raise ValueError(f"Field-transfer {stage} failed: {details}")
