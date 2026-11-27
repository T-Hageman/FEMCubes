"""Cell-local quadrature-to-DG reconstruction using Basix and NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import basix
from basix.ufl import element
import numpy as np
import numpy.typing as npt


_SIMPLEX_CELL_TYPES = frozenset(
    {
        basix.CellType.interval,
        basix.CellType.triangle,
        basix.CellType.tetrahedron,
    }
)
_RANK_TOLERANCE_FACTOR = 64.0
_REPRODUCTION_TOLERANCE = 1.0e-10
_RECONSTRUCTION_CACHE: dict[tuple[Any, ...], "QuadratureReconstruction"] = {}


@dataclass(frozen=True)
class QuadratureReconstruction:
    """Weighted map from one cell's quadrature samples to Basix DG DOFs.

    ``basis_values`` uses quadrature-point rows and Basix local DG-basis
    columns. ``sample_to_coefficients`` therefore maps the final sample axis
    into Basix's local DG coefficient/DOF ordering. For vector or tensor data,
    callers reconstruct each scalar component independently in a stable
    flattened component order.

    Signed quadrature rules cannot define a least-squares norm. The metric
    consequently uses normalized absolute quadrature-weight magnitudes, stored
    in ``least_squares_weights``. This choice changes only the approximation of
    nonrepresentable samples; every representable DG polynomial is reproduced
    independently of the positive weighting.
    """

    cell_type: basix.CellType
    quadrature_degree: int
    points: npt.NDArray[np.float64]
    quadrature_weights: npt.NDArray[np.float64]
    least_squares_weights: npt.NDArray[np.float64]
    degree: int
    basis_values: npt.NDArray[np.float64]
    sample_to_coefficients: npt.NDArray[np.float64]
    rank: int
    condition_number: float
    constant_reproduction_error: float
    basis_reproduction_error: float

    def __post_init__(self) -> None:
        _validate_cell_type(self.cell_type)
        _validate_quadrature_degree(self.quadrature_degree)
        points = _readonly_float_array(self.points)
        quadrature_weights = _readonly_float_array(self.quadrature_weights)
        least_squares_weights = _readonly_float_array(self.least_squares_weights)
        basis_values = _readonly_float_array(self.basis_values)
        operator = _readonly_float_array(self.sample_to_coefficients)

        if points.ndim != 2 or points.shape[0] == 0:
            raise ValueError("Quadrature reconstruction points must be a nonempty matrix.")
        sample_count = points.shape[0]
        if basis_values.ndim != 2 or basis_values.shape[1] == 0:
            raise ValueError("Quadrature reconstruction DG basis must be a nonempty matrix.")
        coefficient_count = basis_values.shape[1]
        if operator.ndim != 2:
            raise ValueError("Quadrature reconstruction operator must be a matrix.")
        if quadrature_weights.shape != (sample_count,):
            raise ValueError("Quadrature reconstruction weights must match the sample count.")
        if least_squares_weights.shape != (sample_count,):
            raise ValueError("Least-squares weights must match the sample count.")
        if basis_values.shape[0] != sample_count:
            raise ValueError("DG basis rows must match the quadrature sample count.")
        if operator.shape != (coefficient_count, sample_count):
            raise ValueError("Reconstruction operator shape is inconsistent with the DG basis.")
        if self.rank != coefficient_count:
            raise ValueError("A quadrature reconstruction must have full DG column rank.")
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(quadrature_weights)):
            raise ValueError("Quadrature reconstruction points and weights must be finite.")
        if (
            not np.all(np.isfinite(least_squares_weights))
            or np.any(least_squares_weights < 0.0)
            or not np.isclose(np.sum(least_squares_weights), 1.0)
        ):
            raise ValueError("Least-squares weights must be finite, nonnegative, and normalized.")
        if not np.all(np.isfinite(basis_values)) or not np.all(np.isfinite(operator)):
            raise ValueError("Quadrature reconstruction matrices must be finite.")

        object.__setattr__(self, "points", points)
        object.__setattr__(self, "quadrature_weights", quadrature_weights)
        object.__setattr__(self, "least_squares_weights", least_squares_weights)
        object.__setattr__(self, "basis_values", basis_values)
        object.__setattr__(self, "sample_to_coefficients", operator)

    @property
    def sample_count(self) -> int:
        """Return the number of scalar quadrature samples per cell."""

        return int(self.points.shape[0])

    @property
    def coefficient_count(self) -> int:
        """Return the number of scalar DG coefficients produced per cell."""

        return int(self.basis_values.shape[1])


def get_quadrature_reconstruction(
    cell_type: basix.CellType,
    quadrature_degree: int,
    points: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
) -> QuadratureReconstruction:
    """Return a cached reconstruction for one simplex quadrature contract.

    When no points are supplied, Basix's default rule for
    ``quadrature_degree`` provides points and weights. Supplying points without
    weights is supported when they are a permutation of that default rule;
    default weights are then reordered to match the supplied sample order.
    Explicit points and weights allow equivalent custom rules and focused rank
    tests while retaining the same reconstruction algorithm.
    """

    cell_type = _validate_cell_type(cell_type)
    degree = _validate_quadrature_degree(quadrature_degree)
    if points is None:
        if weights is not None:
            raise ValueError("Quadrature weights cannot be supplied without points.")
        point_array, weight_array = basix.make_quadrature(cell_type, degree)
    else:
        point_array = _validate_points(cell_type, points)
        if weights is None:
            default_points, default_weights = basix.make_quadrature(cell_type, degree)
            weight_array = _match_default_weights(
                point_array,
                np.asarray(default_points, dtype=np.float64),
                np.asarray(default_weights, dtype=np.float64),
            )
        else:
            weight_array = np.asarray(weights, dtype=np.float64)

    point_array = _validate_points(cell_type, point_array)
    weight_array = _validate_weights(weight_array, point_array.shape[0])
    cache_key = (
        int(cell_type),
        degree,
        point_array.shape,
        point_array.tobytes(),
        weight_array.tobytes(),
    )
    reconstruction = _RECONSTRUCTION_CACHE.get(cache_key)
    if reconstruction is None:
        reconstruction = _build_quadrature_reconstruction(
            cell_type,
            degree,
            point_array,
            weight_array,
        )
        _RECONSTRUCTION_CACHE[cache_key] = reconstruction
    return reconstruction


def reconstruct_scalar_quadrature_samples(
    reconstruction: QuadratureReconstruction,
    samples: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Reconstruct scalar quadrature samples into local Basix DG coefficients.

    The final input axis must follow ``reconstruction.points`` ordering. Any
    leading axes are treated as independent cells or batches, and the returned
    final axis follows the Basix DG local coefficient ordering documented by
    ``reconstruction.basis_values``.
    """

    if not isinstance(reconstruction, QuadratureReconstruction):
        raise TypeError("reconstruction must be a QuadratureReconstruction.")
    if np.iscomplexobj(samples):
        raise ValueError("Quadrature reconstruction currently requires real samples.")
    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.ndim == 0:
        raise ValueError("Quadrature samples must have a final sample axis.")
    if sample_array.shape[-1] != reconstruction.sample_count:
        raise ValueError(
            "Quadrature sample axis has length "
            f"{sample_array.shape[-1]}; expected {reconstruction.sample_count}."
        )
    if not np.all(np.isfinite(sample_array)):
        raise ValueError("Quadrature samples must contain only finite values.")

    coefficients = sample_array @ reconstruction.sample_to_coefficients.T
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Quadrature reconstruction produced non-finite coefficients.")
    return np.asarray(coefficients, dtype=np.float64)


def _build_quadrature_reconstruction(
    cell_type: basix.CellType,
    quadrature_degree: int,
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> QuadratureReconstruction:
    """PRIVATE: Select the largest stable complete DG space and build its map."""

    metric_weights = np.abs(weights)
    metric_sum = float(np.sum(metric_weights))
    if not np.isfinite(metric_sum) or metric_sum <= 0.0:
        raise ValueError("Quadrature weights must contain nonzero finite magnitudes.")
    metric_weights = metric_weights / metric_sum
    sqrt_weights = np.sqrt(metric_weights)
    sample_count = points.shape[0]

    candidates: list[tuple[int, Any]] = []
    for degree in range(sample_count + 1):
        dg_element = element("DG", cell_type, degree).basix_element
        if dg_element is None:  # pragma: no cover - defensive Basix/UFL guard
            raise RuntimeError("Could not create a Basix DG element for reconstruction.")
        if dg_element.dim > sample_count:
            break
        candidates.append((degree, dg_element))

    for degree, dg_element in reversed(candidates):
        table = np.asarray(dg_element.tabulate(0, points), dtype=np.float64)
        if table.ndim != 4 or table.shape[0] != 1 or table.shape[1] != sample_count:  # pragma: no cover - Basix scalar DG tabulation layout invariant
            raise ValueError("Unexpected Basix scalar DG tabulation layout.")
        if table.shape[3] != 1:  # pragma: no cover - internally constructed DG basis is scalar
            raise ValueError("Quadrature reconstruction requires a scalar DG basis.")
        basis_values = table[0, :, :, 0]
        weighted_basis = sqrt_weights[:, None] * basis_values
        singular_values = np.linalg.svd(weighted_basis, compute_uv=False)
        if singular_values.size == 0:  # pragma: no cover - nonempty samples and constant basis imply nonempty SVD
            continue
        relative_tolerance = (
            _RANK_TOLERANCE_FACTOR
            * np.finfo(np.float64).eps
            * max(weighted_basis.shape)
        )
        rank = int(np.count_nonzero(singular_values > relative_tolerance * singular_values[0]))
        if rank != dg_element.dim:
            continue

        weighted_pseudoinverse = np.linalg.pinv(
            weighted_basis,
            rcond=relative_tolerance,
        )
        sample_to_coefficients = weighted_pseudoinverse * sqrt_weights[None, :]
        coefficient_identity = sample_to_coefficients @ basis_values
        basis_error = float(
            np.max(np.abs(coefficient_identity - np.eye(dg_element.dim)))
        )
        constant_samples = np.ones(sample_count, dtype=np.float64)
        constant_coefficients = sample_to_coefficients @ constant_samples
        constant_error = float(
            np.max(np.abs(basis_values @ constant_coefficients - constant_samples))
        )
        if (
            basis_error > _REPRODUCTION_TOLERANCE
            or constant_error > _REPRODUCTION_TOLERANCE
        ):
            continue

        return QuadratureReconstruction(
            cell_type=cell_type,
            quadrature_degree=quadrature_degree,
            points=points,
            quadrature_weights=weights,
            least_squares_weights=metric_weights,
            degree=degree,
            basis_values=basis_values,
            sample_to_coefficients=sample_to_coefficients,
            rank=rank,
            condition_number=float(singular_values[0] / singular_values[-1]),
            constant_reproduction_error=constant_error,
            basis_reproduction_error=basis_error,
        )

    raise ValueError(  # pragma: no cover - valid rules always admit the constant DG fallback
        "Quadrature samples do not support even a stable constant DG reconstruction."
    )


def _validate_cell_type(cell_type: Any) -> basix.CellType:
    """PRIVATE: Require an explicitly supported Basix simplex cell type."""

    if not isinstance(cell_type, basix.CellType):
        raise TypeError("cell_type must be a basix.CellType.")
    if cell_type not in _SIMPLEX_CELL_TYPES:
        names = ", ".join(cell.name for cell in sorted(_SIMPLEX_CELL_TYPES, key=int))
        raise ValueError(f"Quadrature reconstruction supports simplex cells only: {names}.")
    return cell_type


def _validate_quadrature_degree(value: Any) -> int:
    """PRIVATE: Require a nonnegative integer Basix quadrature degree."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("quadrature_degree must be a nonnegative integer.")
    return int(value)


def _validate_points(
    cell_type: basix.CellType,
    points: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """PRIVATE: Validate reference-cell point layout and finite coordinates."""

    point_array = np.asarray(points, dtype=np.float64)
    topological_dimension = {
        basix.CellType.interval: 1,
        basix.CellType.triangle: 2,
        basix.CellType.tetrahedron: 3,
    }[cell_type]
    if point_array.ndim != 2 or point_array.shape[1] != topological_dimension:
        raise ValueError(
            "Quadrature points must have shape "
            f"(sample_count, {topological_dimension}) for {cell_type.name}."
        )
    if point_array.shape[0] == 0:
        raise ValueError("Quadrature reconstruction requires at least one sample point.")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("Quadrature points must contain only finite coordinates.")
    return np.ascontiguousarray(point_array)


def _validate_weights(
    weights: npt.ArrayLike,
    sample_count: int,
) -> npt.NDArray[np.float64]:
    """PRIVATE: Validate signed quadrature weights and sample alignment."""

    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (sample_count,):
        raise ValueError(
            f"Quadrature weights must have shape ({sample_count},); received {weight_array.shape}."
        )
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("Quadrature weights must contain only finite values.")
    if not np.any(np.abs(weight_array) > 0.0):
        raise ValueError("Quadrature weights must contain at least one nonzero value.")
    return np.ascontiguousarray(weight_array)


def _match_default_weights(
    points: npt.NDArray[np.float64],
    default_points: npt.NDArray[np.float64],
    default_weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """PRIVATE: Reorder default-rule weights to an explicit point ordering."""

    if points.shape != default_points.shape:
        raise ValueError(
            "Explicit quadrature points require explicit weights unless they match "
            "the Basix default rule."
        )
    tolerance = 128.0 * np.finfo(np.float64).eps
    available = set(range(default_points.shape[0]))
    order = []
    for point in points:
        matches = [
            index
            for index in available
            if np.allclose(point, default_points[index], rtol=0.0, atol=tolerance)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Explicit quadrature points require explicit weights unless they are a "
                "permutation of the Basix default rule."
            )
        match = matches[0]
        available.remove(match)
        order.append(match)
    return np.ascontiguousarray(default_weights[np.asarray(order, dtype=np.int32)])


def _readonly_float_array(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """PRIVATE: Copy one numeric array and prevent cache mutation by callers."""

    array = np.array(values, dtype=np.float64, copy=True, order="C")
    array.setflags(write=False)
    return array
