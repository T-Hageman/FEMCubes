"""Live installation transactions for detached mesh-refinement states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from numbers import Integral
import os
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

from mpi4py import MPI

from femcubes.core.mesh._mesh_dofs import _MeshDOFs
from femcubes.core.mesh.refinement.config import (
    RefinementConfig,
    normalize_refinement_config,
)
from femcubes.core.mesh.refinement.indicators import (
    evaluate_field_indicator,
    expand_refinement_buffer,
)
from femcubes.core.mesh.refinement.markers import (
    MeshMarkerState,
    transfer_mesh_markers,
)
from femcubes.core.mesh.refinement.partition import repartition_mesh
from femcubes.core.mesh.refinement.topology import (
    RefinedMeshArtifacts,
    RefinementReport,
    refine_mesh_topology,
)
from femcubes.core.mesh.refinement.transfer import (
    FieldTransferDiagnostic,
    TransferredFieldState,
    snapshot_registered_fields,
    transfer_quadrature_fields,
    transfer_standard_fields,
)


@dataclass(frozen=True)
class RefinementInstallResult:
    """Outcome of installing one detached refinement state.

    Timings are rank-local wall seconds. ``total_seconds`` covers the entire
    indicator transaction, or installation alone for a direct install call.
    ``install_seconds`` includes cache invalidation and visualization.
    ``initial_transfer_diagnostics`` retains per-field interpolation timings
    before optional exact repartitioning replaces the detached diagnostics.
    """

    installed: bool
    report: RefinementReport
    epoch_before: int
    epoch_after: int
    visible_plot_refreshed: bool = False
    initial_animation_frame_written: bool = False
    animation_frame_written: bool = False
    install_seconds: float = 0.0
    cache_invalidation_seconds: float = 0.0
    visualization_seconds: float = 0.0
    preparation_seconds: float = 0.0
    indicator_seconds: float = 0.0
    buffer_seconds: float = 0.0
    marker_transfer_seconds: float = 0.0
    standard_transfer_seconds: float = 0.0
    quadrature_transfer_seconds: float = 0.0
    total_seconds: float = 0.0
    initial_transfer_diagnostics: Mapping[str, FieldTransferDiagnostic] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_transfer_diagnostics", MappingProxyType(dict(self.initial_transfer_diagnostics)))


@dataclass(frozen=True)
class _VisualizationRequest:
    """PRIVATE: Resolved collective visualization controls."""

    show_window: bool
    movie_path: str | None
    movie_fps: int | None


@dataclass(frozen=True)
class _LiveMeshState:
    """PRIVATE: Live references retained for best-effort rollback."""

    mesh: Any
    cell_tags: Any
    facet_tags: Any
    fields: Any
    field_definitions: Any
    field_order: Any
    field_type: Any
    test_functions: Any
    trial_functions: Any
    mixed_spaces: Any
    step_fields: Any
    dg_spaces: Any
    cg_spaces: Any
    quadrature_spaces: Any
    domain_markers: Mapping[str, int]
    boundary_markers: Mapping[str, int]
    submesh_state: Mapping[str, Any]


def refine_from_indicator(
    mesh_wrapper,
    config: RefinementConfig | Mapping[str, Any] | None = None,
) -> RefinementInstallResult:
    """Run one complete field-driven refinement transaction.

    Mixed solution storage is synchronized before the configured indicator is
    evaluated. Topology, marker, CG/DG, and quadrature transfers are built as
    detached artifacts. An optional imbalance-triggered redistribution runs
    only after that first complete transfer and produces another complete
    detached state. The live mesh wrapper is changed only by the final
    installation transaction. An empty collective mark set returns a no-op
    result and leaves the remesh epoch unchanged.

    Parameters
    ----------
    mesh_wrapper : femcubes.core.mesh.mesh.Mesh
        Live mesh wrapper containing registered fields and markers.
    config : RefinementConfig or mapping, optional
        Explicit refinement controls. If omitted, ``params.MeshRefinement``
        is used. The resolved configuration must be enabled.

    Returns
    -------
    RefinementInstallResult
        Installed/no-op status, topology report, and epoch transition.

    Notes
    -----
    This mesh transaction cannot refresh caches owned by a ``Physics`` or
    solver instance. Direct callers must run their post-remesh lifecycle after
    a successful result; ``TimeSolver`` performs that sequence automatically.
    """

    start = perf_counter()
    if not hasattr(mesh_wrapper, "mesh") or getattr(mesh_wrapper.mesh, "comm", None) is None:
        raise TypeError("mesh_wrapper must expose a live DOLFINx mesh.")
    comm = mesh_wrapper.mesh.comm
    resolved_config = None
    visualization = None
    local_error = None
    try:
        resolved_config = _resolve_config(mesh_wrapper, config)
        visualization = _resolve_visualization_request(mesh_wrapper, resolved_config)
        if not resolved_config.enabled:
            raise ValueError("A refinement attempt requires MeshRefinement.enabled=True.")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_preflight_error(comm, local_error)
    _validate_collective_config(comm, resolved_config, visualization)

    snapshot = snapshot_registered_fields(mesh_wrapper)
    marker_state = MeshMarkerState.from_mesh_wrapper(mesh_wrapper)
    preparation_seconds = perf_counter() - start
    stage_start = perf_counter()
    indicator_result = evaluate_field_indicator(
        mesh_wrapper,
        resolved_config.indicator,
        resolved_config.target_h,
    )
    indicator_seconds = perf_counter() - stage_start
    stage_start = perf_counter()
    selection = expand_refinement_buffer(
        mesh_wrapper,
        indicator_result,
        resolved_config.buffer_layers,
        target_h=resolved_config.target_h,
    )
    buffer_seconds = perf_counter() - stage_start
    artifacts = refine_mesh_topology(
        mesh_wrapper.mesh,
        selection,
        resolved_config,
    )

    marker_transfer_seconds = 0.0
    standard_transfer_seconds = 0.0
    quadrature_transfer_seconds = 0.0
    initial_transfer_diagnostics = {}
    transferred_state = None
    if artifacts.report.occurred:
        stage_start = perf_counter()
        artifacts = transfer_mesh_markers(marker_state, artifacts)
        marker_transfer_seconds = perf_counter() - stage_start
        stage_start = perf_counter()
        transferred_state = transfer_standard_fields(
            snapshot,
            artifacts,
            resolved_config,
        )
        standard_transfer_seconds = perf_counter() - stage_start
        stage_start = perf_counter()
        transferred_state = transfer_quadrature_fields(
            snapshot,
            artifacts,
            transferred_state,
            resolved_config,
        )
        quadrature_transfer_seconds = perf_counter() - stage_start
        initial_transfer_diagnostics = transferred_state.diagnostics
        partition_result = repartition_mesh(
            artifacts,
            transferred_state,
            resolved_config,
        )
        artifacts = partition_result.artifacts
        transferred_state = partition_result.transferred_state

    result = install_refined_mesh_state(
        mesh_wrapper,
        artifacts,
        transferred_state,
        resolved_config,
    )
    return replace(
        result,
        preparation_seconds=preparation_seconds,
        indicator_seconds=indicator_seconds,
        buffer_seconds=buffer_seconds,
        marker_transfer_seconds=marker_transfer_seconds,
        standard_transfer_seconds=standard_transfer_seconds,
        quadrature_transfer_seconds=quadrature_transfer_seconds,
        total_seconds=perf_counter() - start,
        initial_transfer_diagnostics=initial_transfer_diagnostics,
    )


def install_refined_mesh_state(
    mesh_wrapper,
    artifacts: RefinedMeshArtifacts,
    transferred_state: TransferredFieldState | None = None,
    config: RefinementConfig | Mapping[str, Any] | None = None,
) -> RefinementInstallResult:
    """Collectively install one complete detached refinement state.

    The topology, tags, and transferred standalone functions must already be
    built off to the side. All rank-local preflight checks complete before the
    live wrapper is changed. Mixed/test/trial registries are then reconstructed
    on the target mesh, followed by one collective success check and one epoch
    advancement.

    Visualization is deliberately last. Every rank enters the existing
    :meth:`Mesh.plot_mesh` gather when requested, while only rank zero renders
    or records. Plot backend failures therefore cannot roll back a numerically
    successful installation.
    """

    start = perf_counter()
    if not hasattr(mesh_wrapper, "mesh") or getattr(mesh_wrapper.mesh, "comm", None) is None:
        raise TypeError("mesh_wrapper must expose a live DOLFINx mesh.")
    comm = mesh_wrapper.mesh.comm
    resolved_config = None
    visualization = None
    local_error = None
    try:
        resolved_config = _resolve_config(mesh_wrapper, config)
        visualization = _resolve_visualization_request(mesh_wrapper, resolved_config)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_preflight_error(comm, local_error)
    _validate_collective_config(comm, resolved_config, visualization)
    _validate_collective_topology_attempt(mesh_wrapper, artifacts)

    epoch_before = mesh_wrapper.remesh_epoch
    if not artifacts.report.occurred:
        install_seconds = perf_counter() - start
        return RefinementInstallResult(
            installed=False,
            report=artifacts.report,
            epoch_before=epoch_before,
            epoch_after=epoch_before,
            install_seconds=install_seconds,
            total_seconds=install_seconds,
        )

    _validate_collective_install_state(mesh_wrapper, artifacts, transferred_state)
    old_state = _capture_live_state(mesh_wrapper)

    local_error = None
    cache_invalidation_seconds = 0.0
    try:
        invalidation_start = perf_counter()
        _invalidate_mesh_owned_caches(mesh_wrapper)
        cache_invalidation_seconds = perf_counter() - invalidation_start
        _install_local_state(mesh_wrapper, artifacts, transferred_state)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"

    install_errors = comm.allgather(local_error)
    if any(error is not None for error in install_errors):
        _restore_live_state(mesh_wrapper, old_state)
        details = "; ".join(
            f"rank {rank}: {error}"
            for rank, error in enumerate(install_errors)
            if error is not None
        )
        raise RuntimeError(f"Live mesh installation failed: {details}")

    root_had_visible_mesh_plotter = None
    if comm.rank == 0:
        root_had_visible_mesh_plotter = bool(
            getattr(mesh_wrapper, "_mesh_plotter", None) is not None
            and getattr(mesh_wrapper, "_mesh_plot_visible", False)
        )
    had_visible_mesh_plotter = comm.bcast(
        root_had_visible_mesh_plotter,
        root=0,
    )
    mesh_wrapper.remesh_epoch = epoch_before + 1
    mesh_wrapper._prepare_plots_for_remesh()
    if had_visible_mesh_plotter and not visualization.show_window:
        visualization = _VisualizationRequest(
            show_window=True,
            movie_path=visualization.movie_path,
            movie_fps=visualization.movie_fps,
        )
    visualization_start = perf_counter()
    initial_frame, refined_frame = _visualize_successful_install(
        mesh_wrapper,
        old_state,
        artifacts,
        visualization,
    )
    visualization_seconds = perf_counter() - visualization_start
    mesh_wrapper.last_refinement_report = artifacts.report

    install_seconds = perf_counter() - start
    return RefinementInstallResult(
        installed=True,
        report=artifacts.report,
        epoch_before=epoch_before,
        epoch_after=mesh_wrapper.remesh_epoch,
        visible_plot_refreshed=bool(
            getattr(refined_frame, "visible_updated", False)
        ),
        initial_animation_frame_written=bool(
            getattr(initial_frame, "movie_frame_written", False)
        ),
        animation_frame_written=bool(
            getattr(refined_frame, "movie_frame_written", False)
        ),
        install_seconds=install_seconds,
        cache_invalidation_seconds=cache_invalidation_seconds,
        visualization_seconds=visualization_seconds,
        total_seconds=install_seconds,
    )


def _resolve_config(mesh_wrapper, config) -> RefinementConfig:
    """PRIVATE: Normalize explicit or parameter-owned refinement controls."""

    if config is None:
        config = getattr(mesh_wrapper.params, "MeshRefinement", None)
    return normalize_refinement_config(config)


def _resolve_visualization_request(
    mesh_wrapper,
    config: RefinementConfig,
) -> _VisualizationRequest:
    """PRIVATE: Resolve the ordinary plotting fallbacks used for remeshing."""

    movie_path = None
    movie_fps = None

    if config.save_remesh_animations:
        movie_fps = getattr(mesh_wrapper.params, "movie_fps", 10)
        if isinstance(movie_fps, bool) or not isinstance(movie_fps, Integral) or movie_fps < 1:
            raise ValueError("params.movie_fps must be an integer greater than or equal to one.")
        movie_folder = getattr(mesh_wrapper.params, "movie_folder", None)
        if movie_folder is None:
            output_dir = getattr(mesh_wrapper.params, "output_dir", "Results")
            if output_dir is None:
                output_dir = "Results"
            movie_folder = Path(output_dir) / "movies"
        try:
            movie_path = os.fspath(Path(movie_folder) / "mesh_refinement.mp4")
        except TypeError as exc:
            raise TypeError("params.movie_folder must be path-like or None.") from exc

    return _VisualizationRequest(
        show_window=config.replot_on_refinement,
        movie_path=movie_path,
        movie_fps=None if movie_fps is None else int(movie_fps),
    )


def _validate_collective_config(comm, config, visualization) -> None:
    """PRIVATE: Require identical manager controls on every mesh rank."""

    signatures = comm.allgather((config, visualization))
    if any(signature != signatures[0] for signature in signatures):
        raise ValueError(
            "Refinement installation configuration must be identical on every mesh rank."
        )


def _validate_collective_topology_attempt(mesh_wrapper, artifacts) -> None:
    """PRIVATE: Validate attempt-level invariants before the no-op branch."""

    local_error = None
    try:
        if not isinstance(artifacts, RefinedMeshArtifacts):
            raise TypeError("artifacts must be a RefinedMeshArtifacts.")
        if not isinstance(mesh_wrapper.remesh_epoch, int) or mesh_wrapper.remesh_epoch < 0:
            raise ValueError("mesh_wrapper.remesh_epoch must be a nonnegative integer.")
        if artifacts.report.occurred:
            if artifacts.refined_mesh is mesh_wrapper.mesh:
                raise ValueError("Successful refinement must provide a new target mesh.")
        elif artifacts.refined_mesh is not mesh_wrapper.mesh:
            raise ValueError("No-op refinement artifacts must retain the live mesh.")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_preflight_error(mesh_wrapper.mesh.comm, local_error)

    occurred = mesh_wrapper.mesh.comm.allgather(artifacts.report.occurred)
    if any(value != occurred[0] for value in occurred):
        raise ValueError("Refinement occurrence must be identical on every mesh rank.")


def _validate_collective_install_state(
    mesh_wrapper,
    artifacts: RefinedMeshArtifacts,
    transferred_state: TransferredFieldState | None,
) -> None:
    """PRIVATE: Validate a complete successful state without live mutation."""

    local_error = None
    try:
        if not isinstance(transferred_state, TransferredFieldState):
            raise TypeError(
                "Successful refinement requires a TransferredFieldState."
            )
        if transferred_state.transaction_source_mesh is not mesh_wrapper.mesh:
            raise ValueError("Transferred fields do not originate from the live mesh.")
        if transferred_state.target_mesh is not artifacts.refined_mesh:
            raise ValueError("Transferred fields and topology use different target meshes.")
        if transferred_state.pending_field_names:
            raise ValueError(
                "Transferred field state is incomplete; pending fields: "
                f"{list(transferred_state.pending_field_names)}."
            )
        _fields, live_definitions, live_steps = mesh_wrapper._refinement_field_registry(lift_fields=False)
        if tuple(live_definitions.items()) != tuple(
            transferred_state.field_definitions.items()
        ):
            raise ValueError("Transferred field definitions differ from the live registry.")
        current_steps = tuple(
            (step, tuple(names)) for step, names in live_steps.items()
        )
        if current_steps != tuple(transferred_state.step_fields.items()):
            raise ValueError("Transferred step membership differs from the live registry.")
        if tuple(transferred_state.fields) != tuple(
            transferred_state.field_definitions
        ):
            raise ValueError("Transferred fields must preserve field registration order.")
        for name, diagnostic in transferred_state.diagnostics.items():
            if not diagnostic.finite:
                raise ValueError(f"Transferred field '{name}' is not finite.")
        _validate_step_fields(transferred_state)
        _validate_marker_artifacts(artifacts)
        if MPI.Comm.Compare(
            mesh_wrapper.mesh.comm,
            artifacts.refined_mesh.comm,
        ) not in {MPI.IDENT, MPI.CONGRUENT}:
            raise ValueError("Live and refined meshes must use congruent communicators.")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_collective_preflight_error(mesh_wrapper.mesh.comm, local_error)

    signature = (
        tuple(transferred_state.field_definitions.items()),
        tuple(transferred_state.step_fields.items()),
        tuple(artifacts.domain_markers.items()),
        tuple(artifacts.boundary_markers.items()),
    )
    signatures = mesh_wrapper.mesh.comm.allgather(signature)
    if any(candidate != signatures[0] for candidate in signatures):
        raise ValueError(
            "Transferred field and marker metadata must be identical on every mesh rank."
        )


def _validate_step_fields(transferred_state: TransferredFieldState) -> None:
    """PRIVATE: Reject memberships the current mixed-space builder cannot represent."""

    for step, names in transferred_state.step_fields.items():
        for name in names:
            definition = transferred_state.field_definitions[name]
            if definition.family not in {"CG", "DG"}:
                raise ValueError(
                    f"Solution step {step} field '{name}' must use CG or DG storage."
                )
            if len(definition.value_shape) > 1:
                raise ValueError(
                    f"Solution step {step} field '{name}' has unsupported tensor shape "
                    f"{definition.value_shape}."
                )


def _validate_marker_artifacts(artifacts: RefinedMeshArtifacts) -> None:
    """PRIVATE: Validate tag dimensions and named-tag availability."""

    tdim = artifacts.refined_mesh.topology.dim
    if artifacts.cell_tags is not None and artifacts.cell_tags.dim != tdim:
        raise ValueError(f"Refined cell tags must have dimension {tdim}.")
    if artifacts.facet_tags is not None and artifacts.facet_tags.dim != tdim - 1:
        raise ValueError(f"Refined facet tags must have dimension {tdim - 1}.")
    if artifacts.domain_markers and artifacts.cell_tags is None:
        raise ValueError("Named refined domains require cell tags.")
    if artifacts.boundary_markers and artifacts.facet_tags is None:
        raise ValueError("Named refined boundaries require facet tags.")


def _capture_live_state(mesh_wrapper) -> _LiveMeshState:
    """PRIVATE: Retain live references needed for best-effort rollback."""

    return _LiveMeshState(
        mesh=mesh_wrapper.mesh,
        cell_tags=mesh_wrapper.cell_tags,
        facet_tags=mesh_wrapper.facet_tags,
        fields=mesh_wrapper.fields,
        field_definitions=mesh_wrapper.field_definitions,
        field_order=mesh_wrapper.field_order,
        field_type=mesh_wrapper.field_type,
        test_functions=mesh_wrapper.test_functions,
        trial_functions=mesh_wrapper.trial_functions,
        mixed_spaces=mesh_wrapper.mixed_spaces,
        step_fields=mesh_wrapper.step_fields,
        dg_spaces=mesh_wrapper._dg_spaces,
        cg_spaces=mesh_wrapper._cg_spaces,
        quadrature_spaces=mesh_wrapper._quadrature_spaces,
        domain_markers=dict(mesh_wrapper.domain_markers),
        boundary_markers=dict(mesh_wrapper.boundary_markers),
        submesh_state=mesh_wrapper._capture_submesh_state(),
    )


def _install_local_state(
    mesh_wrapper,
    artifacts: RefinedMeshArtifacts,
    transferred_state: TransferredFieldState,
) -> None:
    """PRIVATE: Replace live mesh storage and reconstruct solution registries."""

    live_fields = dict(mesh_wrapper.fields)
    live_definitions = dict(mesh_wrapper.field_definitions)
    live_steps = {step: tuple(names) for step, names in mesh_wrapper.step_fields.items()}
    mesh_wrapper.mesh = artifacts.refined_mesh
    mesh_wrapper.cell_tags = artifacts.cell_tags
    mesh_wrapper.facet_tags = artifacts.facet_tags
    mesh_wrapper.fields = dict(transferred_state.fields)
    mesh_wrapper.field_definitions = dict(transferred_state.field_definitions)
    mesh_wrapper.field_order = {
        name: definition.degree
        for name, definition in mesh_wrapper.field_definitions.items()
    }
    mesh_wrapper.field_type = {
        name: _MeshDOFs._legacy_field_type(definition)
        for name, definition in mesh_wrapper.field_definitions.items()
    }

    mesh_wrapper.domain_markers.clear()
    for name, tag in artifacts.domain_markers.items():
        mesh_wrapper.add_domain_marker(name, tag)
    mesh_wrapper.boundary_markers.clear()
    for name, tag in artifacts.boundary_markers.items():
        mesh_wrapper.add_boundary_marker(name, tag)

    mesh_wrapper._finish_refined_field_install(live_fields, live_definitions)
    rebuild_solution_registries(mesh_wrapper, live_steps)


def rebuild_solution_registries(mesh_wrapper, step_fields: Mapping[int, Any]) -> None:
    """Recreate the test/trial functions and block layouts of all steps.

    Uses the wrapper's current fields, e.g. after they were replaced by
    transferred functions on a refined mesh.
    """

    step_fields = {step: tuple(names) for step, names in step_fields.items()}
    mesh_wrapper.test_functions = {}
    mesh_wrapper.trial_functions = {}
    mesh_wrapper.mixed_spaces = {}
    mesh_wrapper.step_fields = {}

    for step, names in step_fields.items():
        if not names:
            mesh_wrapper.step_fields[step] = []
            continue
        for name in names:
            definition = mesh_wrapper.field_definitions[name]
            if definition.value_shape:
                mesh_wrapper.Add_Vector_TestFunction(
                    name,
                    nDim=definition.value_shape[0],
                    step=step,
                )
            else:
                mesh_wrapper.Add_TestFunction(name, step=step)

    mesh_wrapper.finalize_test_functions()
    for step, names in step_fields.items():
        for name in names:
            mesh_wrapper.Get_Trial_Function(name, step=step)


def _invalidate_mesh_owned_caches(mesh_wrapper) -> None:
    """PRIVATE: Invalidate every cache owned by the live mesh wrapper."""

    mesh_wrapper.close_outputs()
    mesh_wrapper._invalidate_dof_caches()
    mesh_wrapper._invalidate_boundary_cache()
    mesh_wrapper._invalidate_lumped_cache()
    mesh_wrapper._invalidate_plot_mesh_cache()


def _restore_live_state(mesh_wrapper, state: _LiveMeshState) -> None:
    """PRIVATE: Restore pre-install references after a local rebuild failure."""

    mesh_wrapper.mesh = state.mesh
    mesh_wrapper.cell_tags = state.cell_tags
    mesh_wrapper.facet_tags = state.facet_tags
    mesh_wrapper.fields = state.fields
    mesh_wrapper.field_definitions = state.field_definitions
    mesh_wrapper.field_order = state.field_order
    mesh_wrapper.field_type = state.field_type
    mesh_wrapper.test_functions = state.test_functions
    mesh_wrapper.trial_functions = state.trial_functions
    mesh_wrapper.mixed_spaces = state.mixed_spaces
    mesh_wrapper.step_fields = state.step_fields
    mesh_wrapper._dg_spaces = state.dg_spaces
    mesh_wrapper._cg_spaces = state.cg_spaces
    mesh_wrapper._quadrature_spaces = state.quadrature_spaces
    mesh_wrapper._restore_submesh_state(state.submesh_state)

    mesh_wrapper.domain_markers.clear()
    for name, tag in state.domain_markers.items():
        mesh_wrapper.add_domain_marker(name, tag)
    mesh_wrapper.boundary_markers.clear()
    for name, tag in state.boundary_markers.items():
        mesh_wrapper.add_boundary_marker(name, tag)


def _visualize_successful_install(
    mesh_wrapper,
    old_state: _LiveMeshState,
    artifacts: RefinedMeshArtifacts,
    request: _VisualizationRequest,
):
    """PRIVATE: Append initial/final frames and refresh the persistent plot."""

    if not request.show_window and request.movie_path is None:
        return None, None

    comm = artifacts.refined_mesh.comm
    needs_initial_frame = None
    if comm.rank == 0:
        needs_initial_frame = bool(
            request.movie_path is not None
            and getattr(mesh_wrapper, "_mesh_plot_movie_path", None) is None
            and not getattr(mesh_wrapper, "_mesh_plot_movie_disabled", False)
        )
    needs_initial_frame = comm.bcast(needs_initial_frame, root=0)

    initial_result = None
    if needs_initial_frame:
        initial_result = mesh_wrapper.plot_mesh(
            old_state.mesh,
            facet_tags=old_state.facet_tags,
            boundary_markers=old_state.boundary_markers,
            domain_markers=old_state.domain_markers,
            annotation=_mesh_epoch_annotation(
                mesh_wrapper.remesh_epoch - 1,
                artifacts.report.global_cells_before,
                _MeshDOFs._count_global_solution_dofs(
                    old_state.fields,
                    old_state.step_fields,
                ),
            ),
            show_window=request.show_window,
            movie_path=request.movie_path,
            movie_fps=request.movie_fps,
        )

    refined_result = mesh_wrapper.plot_mesh(
        artifacts.refined_mesh,
        facet_tags=artifacts.facet_tags,
        boundary_markers=artifacts.boundary_markers,
        domain_markers=artifacts.domain_markers,
        annotation=_mesh_epoch_annotation(
            mesh_wrapper.remesh_epoch,
            artifacts.report.global_cells_after,
            mesh_wrapper.get_global_solution_dof_count(),
        ),
        show_window=request.show_window,
        movie_path=request.movie_path,
        movie_fps=request.movie_fps,
    )
    return initial_result, refined_result


def _mesh_epoch_annotation(
    epoch: int,
    global_cells: int,
    global_dofs: int,
) -> str:
    """PRIVATE: Return the stable label used by live remesh visualization."""

    return (
        f"Mesh epoch {epoch:03d}\n"
        f"Global cells: {global_cells}\n"
        f"Global DOFs: {global_dofs}"
    )


def _raise_collective_preflight_error(comm, local_error: str | None) -> None:
    """PRIVATE: Raise the same rank-attributed preflight error everywhere."""

    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        details = "; ".join(
            f"rank {rank}: {error}"
            for rank, error in enumerate(errors)
            if error is not None
        )
        raise ValueError(f"Refinement installation preflight failed: {details}")
