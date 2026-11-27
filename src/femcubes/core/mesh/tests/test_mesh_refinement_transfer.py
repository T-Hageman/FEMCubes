"""Detached CG/DG field-transfer tests for adaptive mesh refinement."""

from __future__ import annotations

from dolfinx import fem
import numpy as np
import pytest

from femcubes.core.mesh.refinement import (
    CellIndicatorResult,
    RefinementConfig,
    expand_refinement_buffer,
    refine_mesh_topology,
    snapshot_registered_fields,
    transfer_standard_fields,
)


def _refine_locally(mesh_wrapper, buffer_layers=1):
    """Create one detached local topology refinement for transfer tests."""

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


def _exact_on_space(function_space, expression):
    """Interpolate a manufactured expression on one destination space."""

    expected = fem.Function(function_space)
    expected.interpolate(expression)
    expected.x.scatter_forward()
    return expected


def _assert_matches_expression(field, expression, tolerance=2.0e-11):
    expected = _exact_on_space(field.function_space, expression)
    np.testing.assert_allclose(field.x.array, expected.x.array, rtol=tolerance, atol=tolerance)


def test_constant_scalar_cg_dg_fields_transfer_with_diagnostics(refinement_mesh):
    cg = refinement_mesh.Add_Field("constant_cg", "Constant CG", "CG", 1)
    dg = refinement_mesh.Add_Field("constant_dg", "Constant DG", "DG", 0)
    refinement_mesh.Add_Quadrature_Field("pending_ip", "Pending IP")
    cg.x.array[:] = 2.5
    dg.x.array[:] = -4.0
    cg.x.scatter_forward()
    dg.x.scatter_forward()
    snapshot = snapshot_registered_fields(refinement_mesh)

    transferred = transfer_standard_fields(
        snapshot,
        _refine_locally(refinement_mesh),
        RefinementConfig(interpolation_padding=1.0e-11),
    )

    np.testing.assert_allclose(transferred.fields["constant_cg"].x.array, 2.5)
    np.testing.assert_allclose(transferred.fields["constant_dg"].x.array, -4.0)
    assert transferred.pending_field_names == ("pending_ip",)
    assert list(transferred.diagnostics) == ["constant_cg", "constant_dg"]
    for diagnostic in transferred.diagnostics.values():
        assert diagnostic.finite
        assert diagnostic.global_source_dofs > 0
        assert diagnostic.global_destination_dofs > diagnostic.global_source_dofs
        assert diagnostic.transfer_seconds >= 0.0
        stages = (
            diagnostic.space_creation_seconds,
            diagnostic.interpolation_seconds, diagnostic.synchronization_seconds,
            diagnostic.copy_seconds, diagnostic.diagnostics_seconds,
        )
        assert all(seconds > 0.0 for seconds in stages)
        assert diagnostic.transfer_seconds >= sum(stages)
        assert diagnostic.reconstruction_seconds == 0.0
        assert diagnostic.search_seconds == 0.0
    assert transferred.diagnostics["constant_cg"].global_source_integral == pytest.approx(5.0)
    assert transferred.diagnostics["constant_cg"].global_destination_integral == pytest.approx(5.0)
    assert transferred.diagnostics["constant_dg"].global_source_integral == pytest.approx(-8.0)
    assert transferred.diagnostics["constant_dg"].global_destination_integral == pytest.approx(-8.0)


def test_parent_cell_transfer_does_not_use_geometric_search(refinement_mesh, monkeypatch):
    import femcubes.core.mesh.refinement.transfer as transfer_module

    field = refinement_mesh.Add_Field("padded", "Padded", "CG", 1)
    field.x.array[:] = 1.0
    snapshot = snapshot_registered_fields(refinement_mesh)
    def unexpected_search(*args, **kwargs):
        raise AssertionError("Refinement ancestry should bypass geometric searches.")

    monkeypatch.setattr(
        transfer_module.fem,
        "create_interpolation_data",
        unexpected_search,
    )
    monkeypatch.setattr(transfer_module.fem.Function, "interpolate_nonmatching", unexpected_search)

    transferred = transfer_standard_fields(
        snapshot,
        _refine_locally(refinement_mesh),
        RefinementConfig(interpolation_padding=3.5e-9),
    )

    np.testing.assert_allclose(transferred.fields["padded"].x.array, 1.0)
    assert transferred.diagnostics["padded"].search_seconds == 0.0


def test_parent_cell_transfer_handles_different_component_and_point_sets(refinement_mesh):
    scalar = refinement_mesh.Add_Field("scalar", "Scalar", "CG", 2)
    vector = refinement_mesh.Add_Vector_Field("vector", "Vector", "CG", 2, 2)
    linear = refinement_mesh.Add_Field("linear", "Linear", "CG", 1)
    expressions = (
        lambda x: 1.0 + x[0] ** 2,
        lambda x: np.vstack((x[0] ** 2 + x[1], x[0] - x[1] ** 2)),
        lambda x: 2.0 - x[0],
    )
    for field, expression in zip((scalar, vector, linear), expressions):
        field.interpolate(expression)
        field.x.scatter_forward()
    transferred = transfer_standard_fields(snapshot_registered_fields(refinement_mesh), _refine_locally(refinement_mesh))
    assert all(diagnostic.search_seconds == 0.0 for diagnostic in transferred.diagnostics.values())
    for field, expression in zip(transferred.fields.values(), expressions):
        _assert_matches_expression(field, expression)


def test_noop_topology_recreates_standard_fields_without_nonmatching_search(refinement_mesh):
    field = refinement_mesh.Add_Field("same_mesh", "Same mesh", "DG", 1)
    field.interpolate(lambda x: 1.0 + x[0] - 2.0 * x[1])
    field.x.scatter_forward()
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

    transferred = transfer_standard_fields(snapshot, artifacts)

    assert not artifacts.report.occurred
    assert transferred.fields["same_mesh"] is not field
    assert transferred.fields["same_mesh"].function_space.mesh is mesh
    np.testing.assert_array_equal(transferred.fields["same_mesh"].x.array, field.x.array)
    diagnostic = transferred.diagnostics["same_mesh"]
    assert diagnostic.global_destination_dofs == diagnostic.global_source_dofs
    assert diagnostic.search_seconds == 0.0
    assert diagnostic.reconstruction_seconds == 0.0
    assert diagnostic.interpolation_seconds == 0.0
    assert diagnostic.copy_seconds > 0.0


@pytest.mark.parametrize(
    ("family", "degree", "expression"),
    [
        ("CG", 1, lambda x: 1.0 + 2.0 * x[0] - 0.5 * x[1]),
        ("CG", 2, lambda x: x[0] ** 2 + x[0] * x[1] - 0.25 * x[1] ** 2),
        ("DG", 1, lambda x: -2.0 + 0.75 * x[0] + 3.0 * x[1]),
        ("DG", 2, lambda x: 0.5 * x[0] ** 2 - x[0] * x[1] + x[1] ** 2),
    ],
)
def test_manufactured_scalar_polynomials_transfer_exactly(
    refinement_mesh,
    family,
    degree,
    expression,
):
    field = refinement_mesh.Add_Field("manufactured", "Manufactured", family, degree)
    field.interpolate(expression)
    field.x.scatter_forward()
    source_values = field.x.array.copy()
    snapshot = snapshot_registered_fields(refinement_mesh)

    transferred = transfer_standard_fields(snapshot, _refine_locally(refinement_mesh))

    _assert_matches_expression(transferred.fields["manufactured"], expression)
    np.testing.assert_array_equal(field.x.array, source_values)
    diagnostic = transferred.diagnostics["manufactured"]
    assert diagnostic.family == family
    assert diagnostic.degree == degree
    assert diagnostic.value_shape == ()
    assert diagnostic.global_destination_integral == pytest.approx(
        diagnostic.global_source_integral,
        rel=1.0e-11,
        abs=1.0e-11,
    )


@pytest.mark.parametrize("family", ["CG", "DG"])
def test_manufactured_vector_fields_transfer_exactly(refinement_mesh, family):
    expression = lambda x: np.vstack(
        (x[0] ** 2 + 0.5 * x[1], -x[0] * x[1] + 2.0 * x[1] ** 2)
    )
    field = refinement_mesh.Add_Vector_Field("vector", "Vector", family, 2, 2)
    field.interpolate(expression)
    field.x.scatter_forward()
    snapshot = snapshot_registered_fields(refinement_mesh)

    transferred = transfer_standard_fields(snapshot, _refine_locally(refinement_mesh))

    _assert_matches_expression(transferred.fields["vector"], expression)
    diagnostic = transferred.diagnostics["vector"]
    assert diagnostic.value_shape == (2,)
    assert diagnostic.global_source_integral is None
    assert diagnostic.global_destination_integral is None


def test_different_orders_and_current_old_pairs_remain_independent(refinement_mesh):
    current_expression = lambda x: 1.0 + x[0] + 2.0 * x[1]
    old_expression = lambda x: -3.0 + x[0] ** 2 - x[1] ** 2
    low_expression = lambda x: 0.25 - 0.5 * x[0]
    current = refinement_mesh.Add_Field("phase", "Current phase", "CG", 1)
    old = refinement_mesh.Add_Field("phase_old", "Old phase", "CG", 2)
    low = refinement_mesh.Add_Field("history", "History", "DG", 1)
    current.interpolate(current_expression)
    old.interpolate(old_expression)
    low.interpolate(low_expression)
    for field in (current, old, low):
        field.x.scatter_forward()
    source_values = {name: field.x.array.copy() for name, field in refinement_mesh.fields.items()}
    refinement_mesh.Add_TestFunction("phase", step=0)
    refinement_mesh.Add_TestFunction("phase_old", step=1)
    snapshot = snapshot_registered_fields(refinement_mesh)

    transferred = transfer_standard_fields(snapshot, _refine_locally(refinement_mesh))

    _assert_matches_expression(transferred.fields["phase"], current_expression)
    _assert_matches_expression(transferred.fields["phase_old"], old_expression)
    _assert_matches_expression(transferred.fields["history"], low_expression)
    assert transferred.fields["phase"].function_space is not transferred.fields["phase_old"].function_space
    assert transferred.step_fields == {0: ("phase",), 1: ("phase_old",)}
    for name, expected_values in source_values.items():
        np.testing.assert_array_equal(refinement_mesh.fields[name].x.array, expected_values)


def test_snapshot_captures_fields_of_monolithic_steps(refinement_mesh):
    # Fields solved together in one block step keep their own storage, which
    # is what the snapshot and transfer operate on.
    first_expression = lambda x: 2.0 + x[0] - x[1]
    second_expression = lambda x: -1.0 + 0.5 * x[0] + 3.0 * x[1]
    first = refinement_mesh.Add_Field("first", "First", "CG", 1)
    second = refinement_mesh.Add_Field("second", "Second", "CG", 1)
    refinement_mesh.Add_TestFunction("first", step=0)
    refinement_mesh.Add_TestFunction("second", step=0)
    refinement_mesh.finalize_test_functions()
    first.interpolate(first_expression)
    second.interpolate(second_expression)

    snapshot = snapshot_registered_fields(refinement_mesh)

    _assert_matches_expression(snapshot.fields["first"], first_expression)
    _assert_matches_expression(snapshot.fields["second"], second_expression)
    transferred = transfer_standard_fields(snapshot, _refine_locally(refinement_mesh))
    _assert_matches_expression(transferred.fields["first"], first_expression)
    _assert_matches_expression(transferred.fields["second"], second_expression)
    assert transferred.step_fields == {0: ("first", "second")}


def test_nonfinite_source_field_is_rejected_before_interpolation(refinement_mesh):
    field = refinement_mesh.Add_Field("broken", "Broken", "CG", 1)
    field.x.array[:] = 1.0
    field.x.array[0] = np.nan
    snapshot = snapshot_registered_fields(refinement_mesh)

    with pytest.raises(ValueError, match="Source field 'broken'.*non-finite"):
        transfer_standard_fields(snapshot, _refine_locally(refinement_mesh))


def test_nonfinite_destination_field_is_rejected_after_interpolation(
    refinement_mesh,
    monkeypatch,
):
    import femcubes.core.mesh.refinement.transfer as transfer_module

    field = refinement_mesh.Add_Field("target_broken", "Target broken", "CG", 1)
    field.x.array[:] = 1.0
    snapshot = snapshot_registered_fields(refinement_mesh)
    original = transfer_module.fem.Function.interpolate

    def interpolate_with_nonfinite_value(destination, *args, **kwargs):
        original(destination, *args, **kwargs)
        destination.x.array[:] = np.inf

    monkeypatch.setattr(
        transfer_module.fem.Function,
        "interpolate",
        interpolate_with_nonfinite_value,
    )

    with pytest.raises(ValueError, match="Transferred field 'target_broken'.*non-finite"):
        transfer_standard_fields(snapshot, _refine_locally(refinement_mesh))


def test_snapshot_rejects_fields_without_structured_definitions(refinement_mesh):
    field = refinement_mesh.Add_Field("defined", "Defined", "CG", 1)
    refinement_mesh.fields["manual"] = fem.Function(field.function_space)

    with pytest.raises(ValueError, match="Every registered field requires exactly one"):
        snapshot_registered_fields(refinement_mesh)
