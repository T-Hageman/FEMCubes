"""Pure Basix/NumPy tests for cell-local quadrature reconstruction."""

from __future__ import annotations

import basix
from dolfinx import fem
import numpy as np
import pytest

from femcubes.core.mesh.refinement import (
    CellIndicatorResult,
    QuadratureReconstruction,
    expand_refinement_buffer,
    get_quadrature_reconstruction,
    refine_mesh_topology,
    reconstruct_scalar_quadrature_samples,
    snapshot_registered_fields,
    transfer_quadrature_fields,
    transfer_standard_fields,
)


def _refine_locally(mesh_wrapper, buffer_layers=1):
    mesh = mesh_wrapper.mesh
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    marked = np.zeros(cells.size, dtype=bool)
    marked[cells.size // 2] = True
    indicator = CellIndicatorResult(
        cells=cells,
        values=marked.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=marked.copy(),
        size_mask=np.ones(cells.size, dtype=bool),
        marked_mask=marked.copy(),
    )
    selection = expand_refinement_buffer(mesh_wrapper, indicator, buffer_layers, target_h=0.1)
    return refine_mesh_topology(mesh, selection)


def _transfer_registered_fields(mesh_wrapper):
    snapshot = snapshot_registered_fields(mesh_wrapper)
    artifacts = _refine_locally(mesh_wrapper)
    standard = transfer_standard_fields(snapshot, artifacts)
    complete = transfer_quadrature_fields(snapshot, artifacts, standard)
    return snapshot, artifacts, standard, complete


def _set_blocked_constant(field, values):
    value_size = field.function_space.dofmap.index_map_bs
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    assert flat_values.size == value_size
    field.x.array.reshape(-1, value_size)[:] = flat_values[None, :]
    field.x.scatter_forward()


def test_degree_three_triangle_rule_selects_largest_full_rank_dg_space():
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 3)

    assert reconstruction.sample_count == 6
    assert reconstruction.degree == 1
    assert reconstruction.coefficient_count == 3
    assert reconstruction.rank == reconstruction.coefficient_count
    quadratic = basix.ufl.element("DG", basix.CellType.triangle, 2).basix_element
    quadratic_basis = quadratic.tabulate(0, reconstruction.points)[0, :, :, 0]
    assert quadratic.dim == reconstruction.sample_count
    assert np.linalg.matrix_rank(quadratic_basis) == 5


def test_registered_degree_three_triangle_field_matches_reconstruction_rule(
    refinement_mesh,
):
    field = refinement_mesh.Add_Quadrature_Field("ip", "Integration points")
    field_points = field.function_space.element.interpolation_points

    reconstruction = get_quadrature_reconstruction(
        refinement_mesh.mesh.basix_cell(),
        refinement_mesh.params.Quadrature_Degree,
        points=field_points,
    )

    assert reconstruction.sample_count == field_points.shape[0]
    np.testing.assert_allclose(reconstruction.points, field_points)
    assert field.function_space.dofmap.cell_dofs(0).size == reconstruction.sample_count


@pytest.mark.parametrize("value_shape", [(), (2, 1), (6, 1), (6, 6)])
def test_constant_registered_quadrature_shapes_transfer_exactly(
    refinement_mesh,
    value_shape,
):
    field = refinement_mesh.Add_Quadrature_Field(
        f"constant_{len(value_shape)}_{int(np.prod(value_shape)) if value_shape else 1}",
        "Constant quadrature field",
        value_shape,
    )
    value_size = int(np.prod(value_shape)) if value_shape else 1
    values = np.linspace(-2.0, 3.0, value_size)
    _set_blocked_constant(field, values)
    source_values = field.x.array.copy()

    snapshot, artifacts, standard, complete = _transfer_registered_fields(refinement_mesh)

    name = next(iter(snapshot.fields))
    destination = complete.fields[name]
    expected = np.broadcast_to(values, destination.x.array.reshape(-1, value_size).shape)
    np.testing.assert_allclose(
        destination.x.array.reshape(-1, value_size),
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_array_equal(field.x.array, source_values)
    assert standard.pending_field_names == (name,)
    assert complete.pending_field_names == ()
    assert complete.field_definitions == snapshot.field_definitions
    diagnostic = complete.diagnostics[name]
    assert diagnostic.family == "Quadrature"
    assert diagnostic.quadrature_degree == 3
    assert diagnostic.reconstruction_degree == 1
    assert diagnostic.value_shape == value_shape
    assert diagnostic.finite
    stages = (
        diagnostic.space_creation_seconds,
        diagnostic.reconstruction_seconds, diagnostic.interpolation_seconds,
        diagnostic.copy_seconds, diagnostic.synchronization_seconds, diagnostic.diagnostics_seconds,
    )
    assert all(seconds > 0.0 for seconds in stages)
    assert diagnostic.search_seconds == 0.0
    assert diagnostic.transfer_seconds >= sum(stages)
    assert diagnostic.global_destination_dofs > diagnostic.global_source_dofs
    assert artifacts.report.occurred


def test_manufactured_linear_scalar_quadrature_field_transfers_exactly(refinement_mesh):
    expression = lambda x: 1.5 + 2.0 * x[0] - 0.75 * x[1]
    field = refinement_mesh.Add_Quadrature_Field("linear_ip", "Linear IP")
    field.interpolate(expression)
    field.x.scatter_forward()

    _, _, _, complete = _transfer_registered_fields(refinement_mesh)

    destination = complete.fields["linear_ip"]
    expected = fem.Function(destination.function_space)
    expected.interpolate(expression)
    expected.x.scatter_forward()
    np.testing.assert_allclose(
        destination.x.array,
        expected.x.array,
        rtol=2.0e-11,
        atol=2.0e-11,
    )


def test_nonrepresentable_quadratic_samples_remain_finite_and_bounded(refinement_mesh):
    expression = lambda x: x[0] ** 2 - 0.5 * x[0] * x[1] + 0.25 * x[1] ** 2
    field = refinement_mesh.Add_Quadrature_Field("quadratic_ip", "Quadratic IP")
    field.interpolate(expression)
    field.x.scatter_forward()

    _, _, _, complete = _transfer_registered_fields(refinement_mesh)

    destination = complete.fields["quadratic_ip"]
    expected = fem.Function(destination.function_space)
    expected.interpolate(expression)
    expected.x.scatter_forward()
    error = np.max(np.abs(destination.x.array - expected.x.array))
    source_scale = max(1.0, float(np.max(np.abs(field.x.array))))
    assert np.all(np.isfinite(destination.x.array))
    assert error / source_scale < 0.2
    assert complete.diagnostics["quadratic_ip"].reconstruction_degree == 1


@pytest.mark.parametrize("second_shape", [(), (6, 1), (6, 6)])
def test_multiple_quadrature_fields_reuse_reconstruction_without_point_search(
    refinement_mesh,
    monkeypatch,
    second_shape,
):
    import femcubes.core.mesh.refinement.transfer as transfer_module

    first = refinement_mesh.Add_Quadrature_Field("first_ip", "First IP")
    second = refinement_mesh.Add_Quadrature_Field("second_ip", "Second IP", second_shape)
    _set_blocked_constant(first, [1.0])
    _set_blocked_constant(second, np.full(int(np.prod(second_shape)), 2.0))
    original_reconstruction = transfer_module.get_quadrature_reconstruction
    original_interpolation_data = transfer_module.fem.create_interpolation_data
    reconstruction_ids = []
    interpolation_calls = []

    def tracked_reconstruction(*args, **kwargs):
        result = original_reconstruction(*args, **kwargs)
        reconstruction_ids.append(id(result))
        return result

    def tracked_interpolation_data(*args, **kwargs):
        interpolation_calls.append((args, kwargs))
        return original_interpolation_data(*args, **kwargs)

    monkeypatch.setattr(
        transfer_module,
        "get_quadrature_reconstruction",
        tracked_reconstruction,
    )
    monkeypatch.setattr(
        transfer_module.fem,
        "create_interpolation_data",
        tracked_interpolation_data,
    )

    _, _, _, complete = _transfer_registered_fields(refinement_mesh)

    assert len(reconstruction_ids) == 2
    assert len(set(reconstruction_ids)) == 1
    assert len(interpolation_calls) == 0
    assert complete.diagnostics["first_ip"].search_seconds == 0.0
    assert complete.diagnostics["second_ip"].search_seconds == 0.0
    assert complete.diagnostics["second_ip"].interpolation_seconds > 0.0
    np.testing.assert_allclose(complete.fields["first_ip"].x.array, 1.0)
    np.testing.assert_allclose(complete.fields["second_ip"].x.array, 2.0)


def test_complete_state_restores_interleaved_registration_order(refinement_mesh):
    first_ip = refinement_mesh.Add_Quadrature_Field("first_ip", "First IP")
    nodal = refinement_mesh.Add_Field("nodal", "Nodal", "CG", 1)
    second_ip = refinement_mesh.Add_Quadrature_Field("second_ip", "Second IP")
    _set_blocked_constant(first_ip, [1.0])
    nodal.x.array[:] = 2.0
    nodal.x.scatter_forward()
    _set_blocked_constant(second_ip, [3.0])

    snapshot, _, standard, complete = _transfer_registered_fields(refinement_mesh)

    assert list(snapshot.fields) == ["first_ip", "nodal", "second_ip"]
    assert list(standard.fields) == ["nodal"]
    assert list(complete.fields) == list(snapshot.fields)
    assert list(complete.diagnostics) == list(snapshot.fields)


def test_nonfinite_quadrature_source_is_rejected_before_reconstruction(refinement_mesh):
    field = refinement_mesh.Add_Quadrature_Field("broken_ip", "Broken IP")
    field.x.array[:] = 1.0
    field.x.array[0] = np.nan
    snapshot = snapshot_registered_fields(refinement_mesh)
    artifacts = _refine_locally(refinement_mesh)
    standard = transfer_standard_fields(snapshot, artifacts)

    with pytest.raises(ValueError, match="Source quadrature field 'broken_ip'.*non-finite"):
        transfer_quadrature_fields(snapshot, artifacts, standard)


def test_noop_topology_recreates_quadrature_storage_without_reconstruction(refinement_mesh):
    field = refinement_mesh.Add_Quadrature_Field("same_ip", "Same IP", (2, 1))
    _set_blocked_constant(field, [1.25, -3.5])
    snapshot = snapshot_registered_fields(refinement_mesh)
    mesh = refinement_mesh.mesh
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    marked = np.zeros(cells.size, dtype=bool)
    indicator = CellIndicatorResult(
        cells=cells,
        values=np.zeros(cells.size),
        diameters=mesh.h(tdim, cells),
        threshold_mask=marked.copy(),
        size_mask=np.ones(cells.size, dtype=bool),
        marked_mask=marked.copy(),
    )
    selection = expand_refinement_buffer(refinement_mesh, indicator, buffer_layers=0, target_h=0.1)
    artifacts = refine_mesh_topology(mesh, selection)
    standard = transfer_standard_fields(snapshot, artifacts)

    complete = transfer_quadrature_fields(snapshot, artifacts, standard)

    assert not artifacts.report.occurred
    assert complete.fields["same_ip"] is not field
    np.testing.assert_array_equal(complete.fields["same_ip"].x.array, field.x.array)
    assert complete.diagnostics["same_ip"].reconstruction_degree is None
    assert complete.diagnostics["same_ip"].reconstruction_seconds == 0.0
    assert complete.diagnostics["same_ip"].search_seconds == 0.0
    assert complete.diagnostics["same_ip"].interpolation_seconds == 0.0
    assert complete.diagnostics["same_ip"].copy_seconds > 0.0
    assert complete.pending_field_names == ()


def test_degree_three_tetrahedron_rule_handles_signed_weights():
    reconstruction = get_quadrature_reconstruction(basix.CellType.tetrahedron, 3)

    assert reconstruction.sample_count == 5
    assert reconstruction.degree == 1
    assert reconstruction.coefficient_count == 4
    assert np.any(reconstruction.quadrature_weights < 0.0)
    assert np.all(reconstruction.least_squares_weights >= 0.0)
    assert np.sum(reconstruction.least_squares_weights) == pytest.approx(1.0)
    np.testing.assert_allclose(
        reconstruction.least_squares_weights,
        np.abs(reconstruction.quadrature_weights)
        / np.sum(np.abs(reconstruction.quadrature_weights)),
    )


@pytest.mark.parametrize(
    "cell_type",
    [basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron],
)
def test_constant_samples_reproduce_exactly(cell_type):
    reconstruction = get_quadrature_reconstruction(cell_type, 3)
    samples = np.full(reconstruction.sample_count, 4.25)

    coefficients = reconstruct_scalar_quadrature_samples(reconstruction, samples)

    np.testing.assert_allclose(
        reconstruction.basis_values @ coefficients,
        samples,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert reconstruction.constant_reproduction_error < 1.0e-12


@pytest.mark.parametrize(
    ("cell_type", "linear_samples"),
    [
        (basix.CellType.interval, lambda points: 1.0 + 2.0 * points[:, 0]),
        (
            basix.CellType.triangle,
            lambda points: 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1],
        ),
        (
            basix.CellType.tetrahedron,
            lambda points: (
                1.0
                + 2.0 * points[:, 0]
                - 3.0 * points[:, 1]
                + 0.5 * points[:, 2]
            ),
        ),
    ],
)
def test_linear_samples_reproduce_when_selected_space_permits(
    cell_type,
    linear_samples,
):
    reconstruction = get_quadrature_reconstruction(cell_type, 3)
    assert reconstruction.degree >= 1
    samples = linear_samples(reconstruction.points)

    coefficients = reconstruct_scalar_quadrature_samples(reconstruction, samples)

    np.testing.assert_allclose(
        reconstruction.basis_values @ coefficients,
        samples,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_highest_selected_polynomial_space_reproduces_all_basis_coefficients():
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 6)
    assert reconstruction.degree == 3
    coefficients = np.linspace(-1.5, 2.0, reconstruction.coefficient_count)
    samples = reconstruction.basis_values @ coefficients

    recovered = reconstruct_scalar_quadrature_samples(reconstruction, samples)

    np.testing.assert_allclose(recovered, coefficients, rtol=2.0e-12, atol=2.0e-12)
    assert reconstruction.basis_reproduction_error < 1.0e-11


def test_weighted_pseudoinverse_matches_explicit_least_squares_construction():
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 5)
    sqrt_weights = np.sqrt(reconstruction.least_squares_weights)
    weighted_basis = sqrt_weights[:, None] * reconstruction.basis_values
    relative_tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(weighted_basis.shape)
    )
    expected = np.linalg.pinv(weighted_basis, rcond=relative_tolerance)
    expected = expected * sqrt_weights[None, :]

    np.testing.assert_allclose(
        reconstruction.sample_to_coefficients,
        expected,
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_rank_deficient_linear_candidate_falls_back_to_constant():
    points = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.2],
            [0.6, 0.2],
        ]
    )
    weights = np.array([0.2, 0.3, 0.5])

    reconstruction = get_quadrature_reconstruction(
        basix.CellType.triangle,
        3,
        points=points,
        weights=weights,
    )

    assert reconstruction.degree == 0
    assert reconstruction.coefficient_count == 1
    assert reconstruction.rank == 1


def test_reconstruction_cache_reuses_identical_element_contract():
    points, weights = basix.make_quadrature(basix.CellType.triangle, 4)

    first = get_quadrature_reconstruction(
        basix.CellType.triangle,
        4,
        points=np.array(points, copy=True),
        weights=np.array(weights, copy=True),
    )
    second = get_quadrature_reconstruction(
        basix.CellType.triangle,
        4,
        points=np.array(points, copy=True),
        weights=np.array(weights, copy=True),
    )

    assert second is first
    assert not first.points.flags.writeable
    assert not first.sample_to_coefficients.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        first.points[0, 0] = 0.0


def test_explicit_default_point_permutation_reorders_weights():
    points, weights = basix.make_quadrature(basix.CellType.triangle, 3)
    order = np.array([4, 2, 0, 5, 3, 1])

    reconstruction = get_quadrature_reconstruction(
        basix.CellType.triangle,
        3,
        points=points[order],
    )

    np.testing.assert_allclose(reconstruction.points, points[order])
    np.testing.assert_allclose(reconstruction.quadrature_weights, weights[order])


def test_scalar_reconstruction_supports_cell_batches():
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 3)
    coefficients = np.array(
        [
            [1.0, 2.0, 3.0],
            [-2.0, 0.5, 4.0],
        ]
    )
    samples = coefficients @ reconstruction.basis_values.T

    recovered = reconstruct_scalar_quadrature_samples(reconstruction, samples)

    assert recovered.shape == coefficients.shape
    np.testing.assert_allclose(recovered, coefficients, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize(
    ("cell_type", "degree", "points", "weights", "message"),
    [
        (
            basix.CellType.triangle,
            3,
            np.empty((0, 2)),
            np.empty(0),
            "at least one sample point",
        ),
        (
            basix.CellType.triangle,
            3,
            np.ones((2, 3)),
            np.ones(2),
            "must have shape",
        ),
        (
            basix.CellType.triangle,
            3,
            np.ones((2, 2)),
            np.ones(3),
            "weights must have shape",
        ),
        (
            basix.CellType.triangle,
            3,
            np.ones((2, 2)),
            np.zeros(2),
            "at least one nonzero",
        ),
        (
            basix.CellType.quadrilateral,
            3,
            None,
            None,
            "simplex cells only",
        ),
    ],
)
def test_invalid_quadrature_contracts_raise_clear_errors(
    cell_type,
    degree,
    points,
    weights,
    message,
):
    with pytest.raises(ValueError, match=message):
        get_quadrature_reconstruction(
            cell_type,
            degree,
            points=points,
            weights=weights,
        )


def test_invalid_scalar_sample_arrays_raise_clear_errors():
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 3)

    with pytest.raises(ValueError, match="final sample axis"):
        reconstruct_scalar_quadrature_samples(reconstruction, 1.0)
    with pytest.raises(ValueError, match="expected 6"):
        reconstruct_scalar_quadrature_samples(reconstruction, np.ones(5))
    with pytest.raises(ValueError, match="finite"):
        reconstruct_scalar_quadrature_samples(
            reconstruction,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]),
        )
    with pytest.raises(ValueError, match="real samples"):
        reconstruct_scalar_quadrature_samples(
            reconstruction,
            np.ones(6, dtype=np.complex128),
        )


def test_reconstruction_type_is_required():
    with pytest.raises(TypeError, match="QuadratureReconstruction"):
        reconstruct_scalar_quadrature_samples(object(), np.ones(1))


def test_public_reconstruction_contract_is_frozen():
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 3)

    assert isinstance(reconstruction, QuadratureReconstruction)
    with pytest.raises(AttributeError):
        reconstruction.degree = 2
