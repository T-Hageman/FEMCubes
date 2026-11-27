"""Cell-indicator evaluation for adaptive mesh refinement."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any

from dolfinx import fem, geometry, mesh as dmesh
from mpi4py import MPI
import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from femcubes.core.mesh.refinement.config import FieldThresholdIndicatorConfig


_SIZE_TOLERANCE_FACTOR = 64.0


@dataclass(frozen=True)
class CellIndicatorResult:
    """Cell-local values and masks produced by one indicator evaluation.

    Arrays use owned-cell ordering. ``threshold_mask`` records the field-only
    decision, ``size_mask`` records whether a cell is still coarser than the
    requested target, and ``marked_mask`` is their conjunction.
    """

    cells: npt.NDArray[np.int32]
    values: npt.NDArray[np.floating]
    diameters: npt.NDArray[np.floating]
    threshold_mask: npt.NDArray[np.bool_]
    size_mask: npt.NDArray[np.bool_]
    marked_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        expected_shape = self.cells.shape
        arrays = {
            "values": self.values,
            "diameters": self.diameters,
            "threshold_mask": self.threshold_mask,
            "size_mask": self.size_mask,
            "marked_mask": self.marked_mask,
        }
        if self.cells.ndim != 1:
            raise ValueError("Indicator cell indices must be one-dimensional.")
        for name, values in arrays.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"Indicator {name} shape {values.shape} does not match cell shape {expected_shape}."
                )

    @property
    def marked_cells(self) -> npt.NDArray[np.int32]:
        """Return owned local cell indices selected for refinement."""

        return self.cells[self.marked_mask]


@dataclass(frozen=True)
class BufferedCellSelection:
    """Collective owned-cell selection after physical-distance buffering."""

    cells: npt.NDArray[np.int32]
    initial_marked_mask: npt.NDArray[np.bool_]
    marked_mask: npt.NDArray[np.bool_]
    buffer_layers: int
    local_initial_count: int
    global_initial_count: int
    local_marked_count: int
    global_marked_count: int
    any_marked: bool

    def __post_init__(self) -> None:
        if self.cells.ndim != 1:
            raise ValueError("Buffered cell indices must be one-dimensional.")
        if self.initial_marked_mask.shape != self.cells.shape:
            raise ValueError("Initial buffered-cell mask must match the owned-cell shape.")
        if self.marked_mask.shape != self.cells.shape:
            raise ValueError("Final buffered-cell mask must match the owned-cell shape.")
        if np.any(self.initial_marked_mask & ~self.marked_mask):
            raise ValueError("Buffer expansion cannot remove initially marked cells.")
        if self.local_initial_count != int(np.count_nonzero(self.initial_marked_mask)):
            raise ValueError("local_initial_count does not match initial_marked_mask.")
        if self.local_marked_count != int(np.count_nonzero(self.marked_mask)):
            raise ValueError("local_marked_count does not match marked_mask.")
        if self.any_marked != (self.global_marked_count > 0):
            raise ValueError("any_marked does not match global_marked_count.")

    @property
    def marked_cells(self) -> npt.NDArray[np.int32]:
        """Return owned local cells selected after buffering."""

        return self.cells[self.marked_mask]

    @property
    def added_by_buffer_mask(self) -> npt.NDArray[np.bool_]:
        """Return cells introduced beyond direct threshold-and-size selection."""

        return self.marked_mask & ~self.initial_marked_mask


def evaluate_field_indicator(
    mesh_wrapper: Any,
    indicator: FieldThresholdIndicatorConfig,
    target_h: float,
) -> CellIndicatorResult:
    """Evaluate a scalar CG/DG/quadrature refinement indicator on owned cells.

    Parameters
    ----------
    mesh_wrapper : femcubes.core.mesh.mesh.Mesh
        Mesh wrapper containing registered standalone fields and structured
        field definitions.
    indicator : FieldThresholdIndicatorConfig
        Field, reduction, comparison, and threshold selection.
    target_h : float
        Strictly positive target cell diameter. Cells already at or below the
        target within floating-point tolerance are excluded.

    Returns
    -------
    CellIndicatorResult
        Reduced values, diameters, intermediate masks, and selected owned
        local cells.

    Notes
    -----
    The caller is responsible for synchronizing mixed solution storage back
    into standalone fields before evaluation. Values are sampled through a
    cell-local DOLFINx expression, never by reducing raw coefficient arrays.
    """

    if not isinstance(indicator, FieldThresholdIndicatorConfig):
        raise TypeError("indicator must be a FieldThresholdIndicatorConfig.")
    _validate_target_h(target_h)

    field, family, support_mask = _resolve_scalar_field(mesh_wrapper, indicator.field)
    cells = _owned_cells(mesh_wrapper.mesh)
    diameters = np.asarray(
        mesh_wrapper.mesh.h(mesh_wrapper.mesh.topology.dim, cells),
        dtype=float,
    )

    # Expression compilation is collective even when this rank owns no cells.
    samples = _evaluate_cell_samples(mesh_wrapper.mesh, field, cells, family)
    if cells.size == 0:
        empty_values = np.empty(0, dtype=float)
        empty_mask = np.empty(0, dtype=bool)
        return CellIndicatorResult(
            cells=cells,
            values=empty_values,
            diameters=diameters,
            threshold_mask=empty_mask.copy(),
            size_mask=empty_mask.copy(),
            marked_mask=empty_mask.copy(),
        )

    if indicator.use_absolute_value:
        samples = np.abs(samples)
    if not np.all(np.isfinite(samples)):
        bad_cells = cells[np.any(~np.isfinite(samples), axis=1)]
        raise ValueError(
            f"Refinement indicator field '{indicator.field}' contains non-finite samples "
            f"on owned cell(s) {bad_cells.tolist()}."
        )

    values = _reduce_samples(samples, indicator.reduction)
    threshold_mask = _compare_values(values, indicator.threshold, indicator.comparison)
    # Fields restricted to a subdomain only mark cells of their subdomain.
    if support_mask is not None:
        threshold_mask &= support_mask[cells]
    size_tolerance = _SIZE_TOLERANCE_FACTOR * np.finfo(float).eps * max(1.0, abs(float(target_h)))
    size_mask = diameters > float(target_h) + size_tolerance
    marked_mask = threshold_mask & size_mask

    return CellIndicatorResult(
        cells=cells,
        values=values,
        diameters=diameters,
        threshold_mask=threshold_mask,
        size_mask=size_mask,
        marked_mask=marked_mask,
    )


def expand_refinement_buffer(
    mesh_wrapper: Any,
    indicator_result: CellIndicatorResult,
    buffer_layers: int,
    *,
    target_h: float,
) -> BufferedCellSelection:
    """Pad the criterion region by ``buffer_layers * target_h`` in physical space.

    All mesh ranks must participate. A size-eligible cell is selected when
    its closed simplex is within the padding distance of any threshold cell,
    including threshold cells already at target size. Zero layers selects
    only the original threshold-and-size mask. Distances are Euclidean and
    independent of the current cell sizes, facet paths, and partition layout.

    Whole cells intersecting the padding are selected, so a coarse cell may
    extend beyond the requested distance. Conforming topology refinement can
    further enlarge the transition region. Neither changes the padding radius.
    """

    mesh = getattr(mesh_wrapper, "mesh", None)
    if mesh is None:
        raise TypeError("mesh_wrapper must expose a DOLFINx mesh through '.mesh'.")
    comm = mesh.comm

    local_error = _buffer_input_error(mesh, indicator_result, buffer_layers, target_h)
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        details = "; ".join(
            f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None
        )
        raise ValueError(f"Invalid refinement buffer input: {details}")

    layers = int(buffer_layers)
    layers_by_rank = comm.allgather(layers)
    if any(rank_layers != layers_by_rank[0] for rank_layers in layers_by_rank):
        raise ValueError(
            "Invalid refinement buffer input: buffer_layers must be identical on every rank; "
            f"received {layers_by_rank}."
        )
    targets_by_rank = comm.allgather(float(target_h))
    if any(target != targets_by_rank[0] for target in targets_by_rank):
        raise ValueError("Invalid refinement buffer input: target_h must be identical on every rank.")

    initial_mask = np.asarray(indicator_result.marked_mask, dtype=bool).copy()
    criterion_mask = np.asarray(indicator_result.threshold_mask, dtype=bool)
    eligible_mask = np.asarray(indicator_result.size_mask, dtype=bool)
    if layers == 0:
        marked_mask = initial_mask.copy()
    else:
        influence = _physical_buffer_mask(mesh, criterion_mask, eligible_mask, layers*float(target_h))
        marked_mask = influence & eligible_mask
    local_initial_count = int(np.count_nonzero(initial_mask))
    local_marked_count = int(np.count_nonzero(marked_mask))
    global_initial_count = int(comm.allreduce(local_initial_count, op=MPI.SUM))
    global_marked_count = int(comm.allreduce(local_marked_count, op=MPI.SUM))

    return BufferedCellSelection(
        cells=np.asarray(indicator_result.cells, dtype=np.int32).copy(),
        initial_marked_mask=initial_mask,
        marked_mask=marked_mask,
        buffer_layers=layers,
        local_initial_count=local_initial_count,
        global_initial_count=global_initial_count,
        local_marked_count=local_marked_count,
        global_marked_count=global_marked_count,
        any_marked=global_marked_count > 0,
    )


def _physical_buffer_mask(mesh, criterion_mask, eligible_mask, radius):
    """PRIVATE: Route candidate simplices to nearby criterion-cell owners."""
    comm = mesh.comm
    cells = np.arange(criterion_mask.size, dtype=np.int32)
    geometry_dofs = dmesh.entities_to_geometry(mesh, mesh.topology.dim, cells, False)
    points = mesh.geometry.x[geometry_dofs]
    seeds = points[criterion_mask]
    candidates = cells[eligible_mask & ~criterion_mask]
    candidate_points = points[candidates]
    if seeds.size:
        bounds = (seeds.min(axis=(0, 1)), seeds.max(axis=(0, 1)))
    else:
        bounds = None
    rank_bounds = comm.allgather(bounds)
    influence = criterion_mask.copy()
    if all(bounds is None for bounds in rank_bounds):
        return influence
    if comm.allreduce(candidates.size, op=MPI.SUM) == 0:
        return influence

    tolerance = _SIZE_TOLERANCE_FACTOR*np.finfo(float).eps*max(1.0, radius)
    search_radius = radius+tolerance
    lower = candidate_points.min(axis=1)
    upper = candidate_points.max(axis=1)
    requests = []
    for bounds in rank_bounds:
        if bounds is None:
            near = np.zeros(candidates.size, dtype=bool)
        else:
            separation = np.maximum(np.maximum(lower-bounds[1], bounds[0]-upper), 0.0)
            near = np.linalg.norm(separation, axis=1) <= search_radius
        requests.append((candidates[near], candidate_points[near]))
    incoming = comm.alltoall(requests)

    replies = []
    local_error = None
    try:
        if seeds.size:
            centers = seeds.mean(axis=1)
            seed_radii = np.linalg.norm(seeds-centers[:, None, :], axis=2).max(axis=1)
            tree = cKDTree(centers)
            seed_lower, seed_upper = seeds.min(axis=1), seeds.max(axis=1)
        for indices, queries in incoming:
            selected = []
            if seeds.size and indices.size:
                centers = queries.mean(axis=1)
                radii = np.linalg.norm(queries-centers[:, None, :], axis=2).max(axis=1)
                neighbors = tree.query_ball_point(centers, search_radius+radii+seed_radii.max())
                for index, query, nearby in zip(indices, queries, neighbors):
                    nearby = np.asarray(nearby, dtype=np.int32)
                    separation = np.maximum(
                        np.maximum(query.min(axis=0)-seed_upper[nearby], seed_lower[nearby]-query.max(axis=0)),
                        0.0,
                    )
                    nearby = nearby[np.linalg.norm(separation, axis=1) <= search_radius]
                    for seed in nearby:
                        distance = geometry.compute_distance_gjk(query, seeds[seed])
                        if np.linalg.norm(distance) <= search_radius:
                            selected.append(index)
                            break
            replies.append(np.asarray(selected, dtype=np.int32))
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    errors = comm.allgather(local_error)
    if any(error is not None for error in errors):
        details = "; ".join(f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None)
        raise ValueError(f"Physical refinement buffer evaluation failed: {details}")
    for selected in comm.alltoall(replies):
        influence[selected] = True
    return influence


def _buffer_input_error(mesh, indicator_result: Any, buffer_layers: Any, target_h: Any) -> str | None:
    """PRIVATE: Return a local validation error without entering ghost collectives."""

    if not isinstance(indicator_result, CellIndicatorResult):
        return "indicator_result must be a CellIndicatorResult"
    if isinstance(buffer_layers, bool) or not isinstance(buffer_layers, Integral) or buffer_layers < 0:
        return "buffer_layers must be a nonnegative integer"

    try:
        _validate_target_h(target_h)
        radius = int(buffer_layers)*float(target_h)
        if not math.isfinite(radius):
            return "buffer_layers * target_h must be finite"
    except (ValueError, OverflowError) as exc:
        return str(exc)

    cell_map = mesh.topology.index_map(mesh.topology.dim)
    expected_cells = np.arange(cell_map.size_local, dtype=np.int32)
    if not np.array_equal(indicator_result.cells, expected_cells):
        return "indicator_result cells must use complete owned local cell ordering"
    expected_marked_mask = np.asarray(indicator_result.threshold_mask, dtype=bool) & np.asarray(
        indicator_result.size_mask,
        dtype=bool,
    )
    if not np.array_equal(indicator_result.marked_mask, expected_marked_mask):
        return "indicator_result marked_mask must equal threshold_mask & size_mask"
    return None


def _resolve_scalar_field(mesh_wrapper: Any, field_name: str):
    """PRIVATE: Return ``(field, family, support mask or None)`` of a scalar indicator field."""

    fields = getattr(mesh_wrapper, "fields", None)
    if fields is None or field_name not in fields:
        raise ValueError(f"Refinement indicator field '{field_name}' is not registered on the mesh.")

    definitions = getattr(mesh_wrapper, "field_definitions", None)
    if definitions is None or field_name not in definitions:
        raise ValueError(
            f"Refinement indicator field '{field_name}' has no structured FieldDefinition."
        )

    definition = definitions[field_name]
    # Mesh wrappers with restricted fields provide their parent-mesh view.
    indicator_view = getattr(mesh_wrapper, "_refinement_indicator_view", None)
    field, support_mask = indicator_view(field_name) if indicator_view is not None else (fields[field_name], None)
    if definition.value_shape:
        raise ValueError(
            f"Refinement indicator field '{field_name}' must be scalar; "
            f"received value shape {definition.value_shape}."
        )
    if definition.family not in {"CG", "DG", "Quadrature"}:  # pragma: no cover - FieldDefinition already restricts supported families
        raise ValueError(
            f"Refinement indicator field '{field_name}' uses unsupported family "
            f"{definition.family!r}."
        )
    if field.function_space.mesh is not mesh_wrapper.mesh:
        raise ValueError(
            f"Refinement indicator field '{field_name}' belongs to a stale or different mesh."
        )
    return field, definition.family, support_mask


def _owned_cells(mesh) -> npt.NDArray[np.int32]:
    """PRIVATE: Return owned cells in local index order."""

    cell_map = mesh.topology.index_map(mesh.topology.dim)
    return np.arange(cell_map.size_local, dtype=np.int32)


def _evaluate_cell_samples(mesh, field, cells: npt.NDArray[np.int32], family: str) -> np.ndarray:
    """PRIVATE: Return scalar samples in owned-cell order."""

    if family == "Quadrature":
        return _read_quadrature_cell_samples(field, cells)

    points = np.asarray(field.function_space.element.interpolation_points)
    if points.ndim != 2 or points.shape[0] == 0:  # pragma: no cover - supported CG/DG elements expose interpolation points
        raise ValueError(
            f"Cannot sample {family} field '{field.name}': its element exposes no interpolation points."
        )

    expression = fem.Expression(field, points)
    if cells.size == 0:
        return np.empty((0, points.shape[0]), dtype=field.x.array.dtype)
    evaluated = np.asarray(expression.eval(mesh, cells))
    evaluated = evaluated.reshape(cells.size, points.shape[0], -1)
    if evaluated.shape[2] != 1:  # pragma: no cover - validated scalar fields have scalar backend evaluations
        raise ValueError(
            f"Refinement indicator field '{field.name}' did not evaluate as a scalar field."
        )
    return evaluated[:, :, 0]


def _read_quadrature_cell_samples(field, cells: npt.NDArray[np.int32]) -> np.ndarray:
    """PRIVATE: Read scalar integration-point values through the cell dofmap."""

    function_space = field.function_space
    dofmap = function_space.dofmap
    if dofmap.bs != 1 or dofmap.index_map_bs != 1:  # pragma: no cover - scalar quadrature elements have unit block sizes
        raise ValueError(
            f"Scalar quadrature indicator field '{field.name}' has unsupported dof block "
            f"sizes bs={dofmap.bs}, index_map_bs={dofmap.index_map_bs}; expected 1."
        )

    points = np.asarray(function_space.element.interpolation_points)
    if points.ndim != 2 or points.shape[0] == 0:  # pragma: no cover - valid quadrature elements expose interpolation points
        raise ValueError(
            f"Cannot sample quadrature field '{field.name}': its element exposes no "
            "interpolation points."
        )

    samples_per_cell = points.shape[0]
    samples = np.empty((cells.size, samples_per_cell), dtype=field.x.array.dtype)
    for row, cell in enumerate(cells):
        cell_dofs = np.asarray(dofmap.cell_dofs(int(cell)), dtype=np.int64)
        if cell_dofs.size != samples_per_cell:  # pragma: no cover - backend quadrature dof counts match the rule
            raise ValueError(
                f"Quadrature indicator field '{field.name}' exposes {cell_dofs.size} cell "
                f"dofs but {samples_per_cell} interpolation points on cell {int(cell)}."
            )
        if np.any(cell_dofs < 0) or np.any(cell_dofs >= field.x.array.size):  # pragma: no cover - backend dofmaps index their own function vector
            raise ValueError(
                f"Quadrature indicator field '{field.name}' has out-of-range cell dofs "
                f"on cell {int(cell)}."
            )
        samples[row, :] = field.x.array[cell_dofs]

    return samples


def _reduce_samples(samples: np.ndarray, reduction: str) -> np.ndarray:
    """PRIVATE: Reduce sample values into one value per cell."""

    if reduction == "cell_max":
        return np.max(samples, axis=1)
    if reduction == "cell_min":
        return np.min(samples, axis=1)
    if reduction == "cell_mean":
        return np.mean(samples, axis=1)
    raise ValueError(f"Unsupported refinement reduction {reduction!r}.")  # pragma: no cover


def _compare_values(values: np.ndarray, threshold: float, comparison: str) -> npt.NDArray[np.bool_]:
    """PRIVATE: Apply a configured threshold comparison."""

    if comparison == "greater_equal":
        return values >= threshold
    if comparison == "greater":
        return values > threshold
    if comparison == "less_equal":
        return values <= threshold
    if comparison == "less":
        return values < threshold
    raise ValueError(f"Unsupported refinement comparison {comparison!r}.")  # pragma: no cover


def _validate_target_h(target_h: Any) -> None:
    """PRIVATE: Validate target diameter for standalone indicator use."""

    if (
        isinstance(target_h, bool)
        or not isinstance(target_h, Real)
        or not math.isfinite(float(target_h))
        or target_h <= 0
    ):
        raise ValueError("target_h must be a finite real number greater than zero.")
