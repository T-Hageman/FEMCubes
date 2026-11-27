"""Public refinement objects reject incomplete or inconsistent detached data."""

from dataclasses import replace
from types import SimpleNamespace

import basix
from dolfinx import mesh as dmesh
import numpy as np
import pytest

from femcubes.core.mesh.refinement import (
    BufferedCellSelection, CellIndicatorResult, FieldThresholdIndicatorConfig,
    MeshMarkerState, RefinementConfig,
    evaluate_field_indicator, expand_refinement_buffer,
    get_quadrature_reconstruction, reconstruct_scalar_quadrature_samples,
    repartition_mesh, resolve_owned_child_input_positions, redistribute_mesh_markers,
    snapshot_registered_fields, transfer_mesh_markers, transfer_standard_fields,
    transfer_quadrature_fields, refine_mesh_topology,
)
from femcubes.core.mesh.tests.test_mesh_refinement_manager import _successful_detached_state
import femcubes.core.mesh.refinement as refinement_api


def _indicator(mesh):
    cells = np.arange(mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_local, dtype=np.int32)
    eligible = np.ones(cells.size, dtype=bool)
    return CellIndicatorResult(cells, cells.astype(float), mesh.mesh.h(mesh.mesh.topology.dim, cells),
                               eligible, eligible.copy(), eligible.copy())


@pytest.mark.parametrize("attribute", ["cells", "values", "diameters", "threshold_mask", "size_mask", "marked_mask"])
def test_indicator_arrays_must_share_owned_cell_shape(refinement_mesh, attribute):
    original = _indicator(refinement_mesh)
    value = original.cells[:, None] if attribute == "cells" else getattr(original, attribute)[:-1]
    with pytest.raises(ValueError, match="one-dimensional|shape"):
        replace(original, **{attribute: value})


@pytest.mark.parametrize("change, message", [
    ({"cells": np.array([[0]])}, "one-dimensional"),
    ({"initial_marked_mask": np.array([True, False])}, "Initial"),
    ({"marked_mask": np.array([True, False])}, "Final"),
    ({"marked_mask": np.array([False]), "local_marked_count": 0}, "remove"),
    ({"local_initial_count": 0}, "local_initial_count"),
    ({"local_marked_count": 0}, "local_marked_count"),
    ({"any_marked": False}, "any_marked"),
])
def test_buffered_selection_rejects_inconsistent_counts(change, message):
    original = BufferedCellSelection(np.array([0]), np.array([True]), np.array([True]),
                                     0, 1, 1, 1, 1, True)
    with pytest.raises(ValueError, match=message):
        replace(original, **change)


@pytest.mark.parametrize("layers", [True, -1, 0.5])
def test_buffer_rejects_invalid_layer_count(refinement_mesh, layers):
    with pytest.raises(ValueError, match="buffer_layers"):
        expand_refinement_buffer(refinement_mesh, _indicator(refinement_mesh), layers, target_h=0.1)


def test_buffer_rejects_wrong_order_and_inconsistent_mask(refinement_mesh):
    result = _indicator(refinement_mesh)
    with pytest.raises(ValueError, match="ordering"):
        expand_refinement_buffer(refinement_mesh, replace(result, cells=result.cells[::-1]), 0, target_h=0.1)
    with pytest.raises(ValueError, match="marked_mask"):
        expand_refinement_buffer(refinement_mesh, replace(result, marked_mask=~result.marked_mask), 0, target_h=0.1)
    with pytest.raises(ValueError, match="CellIndicatorResult"):
        expand_refinement_buffer(refinement_mesh, None, 0, target_h=0.1)
    with pytest.raises(TypeError, match="mesh_wrapper"):
        expand_refinement_buffer(None, result, 0, target_h=0.1)


@pytest.mark.parametrize("target", [True, 0.0, -1.0, np.nan, np.inf, "0.1"])
def test_indicator_rejects_invalid_target_size(refinement_mesh, target):
    refinement_mesh.Add_Field("indicator", "Indicator", "DG", 0)
    with pytest.raises(ValueError, match="target_h"):
        evaluate_field_indicator(refinement_mesh, FieldThresholdIndicatorConfig("indicator", 0.5), target)


def test_indicator_rejects_wrong_config_and_stale_field(refinement_mesh):
    field = refinement_mesh.Add_Field("indicator", "Indicator", "DG", 0)
    with pytest.raises(TypeError, match="FieldThresholdIndicatorConfig"):
        evaluate_field_indicator(refinement_mesh, None, 0.1)
    artifacts, _ = _successful_detached_state(refinement_mesh)
    stale_wrapper = SimpleNamespace(mesh=artifacts.refined_mesh, fields={"indicator": field},
                                    field_definitions=refinement_mesh.field_definitions)
    with pytest.raises(ValueError, match="stale or different"):
        evaluate_field_indicator(stale_wrapper, FieldThresholdIndicatorConfig("indicator", 0.5), 0.1)
    incomplete = SimpleNamespace(mesh=refinement_mesh.mesh, fields={"indicator": field})
    with pytest.raises(ValueError, match="FieldDefinition"):
        evaluate_field_indicator(incomplete, FieldThresholdIndicatorConfig("indicator", 0.5), 0.1)


def test_reconstruction_requires_an_explicit_basix_cell_type():
    with pytest.raises(TypeError, match="cell_type must be a basix.CellType"):
        get_quadrature_reconstruction("triangle", 3)


@pytest.mark.parametrize("degree", [True, -1, 1.5, "3"])
def test_reconstruction_rejects_invalid_quadrature_degrees(degree):
    with pytest.raises(ValueError, match="quadrature_degree must be a nonnegative integer"):
        get_quadrature_reconstruction(basix.CellType.triangle, degree)


@pytest.mark.parametrize("change, message", [
    ({"points": np.empty((0, 2))}, "nonempty matrix"),
    ({"basis_values": np.empty((4, 0))}, "nonempty matrix"),
    ({"sample_to_coefficients": np.zeros(4)}, "operator must be a matrix"),
    ({"quadrature_weights": np.ones(1)}, "weights must match"),
    ({"least_squares_weights": np.ones(1)}, "weights must match"),
    ({"basis_values": np.ones((1, 3))}, "rows must match"),
    ({"sample_to_coefficients": np.ones((1, 1))}, "operator shape"),
    ({"rank": 0}, "full DG column rank"),
    ({"points": lambda r: np.full_like(r.points, np.nan)}, "points and weights must be finite"),
    ({"least_squares_weights": lambda r: np.full(r.sample_count, -1.0/r.sample_count)}, "nonnegative"),
    ({"basis_values": lambda r: np.full_like(r.basis_values, np.nan)}, "matrices must be finite"),
])
def test_reconstruction_contract_rejects_invalid_arrays(change, message):
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 3)
    change = {name: value(reconstruction) if callable(value) else value for name, value in change.items()}
    with pytest.raises(ValueError, match=message):
        replace(reconstruction, **change)


@pytest.mark.parametrize("points, weights, message", [
    (None, np.ones(1), "without points"),
    (np.empty((0, 2)), np.empty(0), "at least one"),
    (np.array([[np.nan, 0.0]]), np.ones(1), "finite coordinates"),
    (np.array([[0.2, 0.2]]), np.ones(2), "weights must have shape"),
    (np.array([[0.2, 0.2]]), np.array([np.inf]), "finite values"),
    (np.array([[0.2, 0.2]]), None, "explicit weights"),
    (np.full_like(basix.make_quadrature(basix.CellType.triangle, 3)[0], 0.2), None, "permutation"),
])
def test_custom_quadrature_rule_rejects_ambiguous_samples(points, weights, message):
    with pytest.raises(ValueError, match=message):
        get_quadrature_reconstruction(basix.CellType.triangle, 3, points=points, weights=weights)


def test_reconstruction_requires_a_reconstruction_object():
    with pytest.raises(TypeError, match="QuadratureReconstruction"):
        reconstruct_scalar_quadrature_samples(None, [1.0])


def test_nearly_degenerate_quadrature_rule_falls_back_to_stable_reconstruction():
    points = np.array([[0.2, 0.2], [0.2+1e-10, 0.2], [0.2, 0.2+1e-10]])
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 2,
                                                   points=points, weights=np.ones(3))
    coefficients = reconstruct_scalar_quadrature_samples(reconstruction, np.full((2, 3), 5.0))
    np.testing.assert_allclose(coefficients @ reconstruction.basis_values.T, 5.0, atol=1e-11)
    assert reconstruction.basis_reproduction_error <= 1e-10
    assert reconstruction.constant_reproduction_error <= 1e-10


def test_finite_quadrature_input_overflow_is_reported():
    points = np.array([[0.2, 0.2], [0.6, 0.2], [0.2, 0.6]])
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="nonzero finite magnitudes"):
            get_quadrature_reconstruction(basix.CellType.triangle, 2, points=points,
                                          weights=np.full(3, np.finfo(float).max))
    reconstruction = get_quadrature_reconstruction(basix.CellType.triangle, 2)
    samples = np.finfo(float).max * np.sign(reconstruction.sample_to_coefficients[0])
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="non-finite coefficients"):
            reconstruct_scalar_quadrature_samples(reconstruction, samples)


def test_unknown_refinement_export_is_an_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(refinement_api, "not_a_refinement_export")


def test_topology_refinement_requires_a_mesh_and_buffered_selection(refinement_mesh):
    with pytest.raises(TypeError, match="DOLFINx mesh"):
        refine_mesh_topology(None, None)
    with pytest.raises(ValueError, match="BufferedCellSelection"):
        refine_mesh_topology(refinement_mesh.mesh, None)


@pytest.mark.parametrize("attribute", ["parent_cell", "parent_facet", "marked_edges"])
def test_topology_ancestry_arrays_must_be_one_dimensional(refinement_mesh, attribute):
    artifacts, _ = _successful_detached_state(refinement_mesh)
    with pytest.raises(ValueError, match="one-dimensional"):
        replace(artifacts, **{attribute: getattr(artifacts, attribute)[:, None]})


def test_topology_ancestry_and_marker_mapping_contracts(refinement_mesh):
    artifacts, _ = _successful_detached_state(refinement_mesh)
    with pytest.raises(ValueError, match="parent-cell ancestry"):
        replace(artifacts, parent_cell=np.empty(0, dtype=np.int32))
    with pytest.raises(ValueError, match="no-op refinement"):
        replace(artifacts, report=replace(artifacts.report, occurred=False))
    for attribute in ("domain_markers", "boundary_markers"):
        with pytest.raises(TypeError, match="mapping"):
            replace(artifacts, **{attribute: []})


@pytest.mark.parametrize("mapping", [{"": 1}, {None: 1}, {"ice": True}, {"ice": 1.5}])
def test_marker_names_and_tags_are_validated(refinement_mesh, mapping):
    with pytest.raises(ValueError, match="keys|values"):
        MeshMarkerState(refinement_mesh.mesh, cell_tags=refinement_mesh.cell_tags,
                        domain_markers=mapping)


@pytest.mark.parametrize("change, message", [
    ({"cell_tags": None}, "Named domains"),
    ({"facet_tags": None}, "Named boundaries"),
])
def test_named_marker_state_requires_tags(refinement_mesh, change, message):
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    with pytest.raises(ValueError, match=message):
        replace(source, **change)


def test_marker_state_rejects_wrong_tag_dimensions(refinement_mesh):
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    with pytest.raises(ValueError, match="Cell tags must have dimension"):
        replace(source, cell_tags=source.facet_tags)
    with pytest.raises(ValueError, match="Facet tags must have dimension"):
        replace(source, facet_tags=source.cell_tags)
    with pytest.raises(TypeError, match="topology"):
        MeshMarkerState(None)


def test_detached_marker_names_and_missing_tag_queries(refinement_mesh):
    artifacts, _ = _successful_detached_state(refinement_mesh)
    assert artifacts.domain_names() == refinement_mesh.domain_names()
    assert artifacts.boundary_names() == refinement_mesh.boundary_names()
    for query in (artifacts.get_domain_tag, artifacts.get_boundary_tag):
        with pytest.raises(KeyError, match="Available"):
            query("missing")
    with pytest.raises(ValueError, match="cell tags"):
        replace(artifacts, cell_tags=None).get_domain_cells(artifacts.domain_names()[0])
    with pytest.raises(ValueError, match="facet tags"):
        replace(artifacts, facet_tags=None).get_boundary_facets(artifacts.boundary_names()[0])


def test_detached_marker_queries_resolve_group_names(refinement_mesh):
    artifacts, _ = _successful_detached_state(refinement_mesh)
    domain_tag = artifacts.get_domain_tag("domain")
    boundary_tags = [artifacts.get_boundary_tag(name) for name in artifacts.boundary_names()[:2]]
    grouped = replace(
        artifacts,
        domain_markers={"domain:all": domain_tag},
        boundary_markers={"edge:a": boundary_tags[0], "edge:b": boundary_tags[1]},
    )

    assert grouped.domain_names() == ["domain", "domain:all"]
    assert grouped.get_domain_tags("domain") == (domain_tag,)
    np.testing.assert_array_equal(grouped.get_domain_cells("domain"), artifacts.get_domain_cells("domain"))
    assert grouped.get_boundary_tags("edge") == tuple(boundary_tags)
    with pytest.raises(ValueError, match="use get_boundary_tags"):
        grouped.get_boundary_tag("edge")
    expected_facets = np.unique(np.concatenate([artifacts.facet_tags.find(tag) for tag in boundary_tags]))
    np.testing.assert_array_equal(grouped.get_boundary_facets("edge"), expected_facets)


def test_marker_transfer_checks_ancestry_and_public_input_types(refinement_mesh):
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    artifacts, _ = _successful_detached_state(refinement_mesh)
    for bad_source, bad_artifacts in ((None, artifacts), (source, None)):
        with pytest.raises(TypeError):
            transfer_mesh_markers(bad_source, bad_artifacts)
    with pytest.raises(TypeError):
        resolve_owned_child_input_positions(None)
    with pytest.raises(ValueError, match="offset"):
        resolve_owned_child_input_positions(replace(artifacts, input_cell_global_offset=None))
    with pytest.raises(ValueError, match="outside owned source"):
        transfer_mesh_markers(source, replace(artifacts, parent_cell=np.full_like(artifacts.parent_cell, -1)))
    with pytest.raises(ValueError, match="invalid size"):
        transfer_mesh_markers(source, replace(artifacts, parent_facet=artifacts.parent_facet[:-1]))
    with pytest.raises(ValueError, match="invalid local facet"):
        transfer_mesh_markers(source, replace(artifacts, parent_facet=np.full_like(artifacts.parent_facet, -2)))
    with pytest.raises(ValueError, match="ancestry width"):
        transfer_mesh_markers(source, replace(artifacts, parent_facet=np.full(2*artifacts.parent_cell.size, -1, dtype=np.int8)))
    with pytest.raises(ValueError, match="outside the parent-cell"):
        resolve_owned_child_input_positions(replace(artifacts, input_cell_global_offset=10**12))
    for bad_source, bad_artifacts in ((None, artifacts), (source, None)):
        with pytest.raises(TypeError):
            redistribute_mesh_markers(bad_source, bad_artifacts)
    with pytest.raises(ValueError, match="distinct target mesh"):
        redistribute_mesh_markers(source, replace(artifacts, refined_mesh=source.mesh))


def test_refinement_can_transfer_an_unnamed_mesh_without_tags(refinement_mesh):
    artifacts, _ = _successful_detached_state(refinement_mesh)
    source = MeshMarkerState(refinement_mesh.mesh)
    transferred = transfer_mesh_markers(source, artifacts)
    assert transferred.cell_tags is None and transferred.facet_tags is None
    assert transferred.domain_names() == [] and transferred.boundary_names() == []


@pytest.mark.parametrize("failure", ["missing_domain", "changed_domain_measure", "missing_boundary"])
def test_corrupted_ancestry_cannot_drop_or_resize_named_regions(refinement_mesh, failure):
    artifacts, _ = _successful_detached_state(refinement_mesh)
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    if failure == "missing_boundary":
        invalid = replace(artifacts, parent_facet=np.full_like(artifacts.parent_facet, -1))
        message = "named boundary.*contains no facets"
    else:
        tag = refinement_mesh.get_domain_tag(refinement_mesh.domain_names()[0])
        tags = dmesh.meshtags(refinement_mesh.mesh, refinement_mesh.mesh.topology.dim,
                             np.array([0], dtype=np.int32), np.array([tag], dtype=np.int32))
        source = MeshMarkerState(refinement_mesh.mesh, cell_tags=tags, domain_markers={"single_cell": tag})
        parent = 1 if failure == "missing_domain" else 0
        invalid = replace(artifacts, parent_cell=np.full_like(artifacts.parent_cell, parent))
        message = "named domain.*contains no cells" if failure == "missing_domain" else "named domain.*changed measure"
    with pytest.raises(ValueError, match=message):
        transfer_mesh_markers(source, invalid)


def test_transfer_snapshot_rejects_wrong_fields_and_step_membership(refinement_mesh):
    refinement_mesh.Add_Field("scalar", "Scalar", "CG", 1)
    snapshot = snapshot_registered_fields(refinement_mesh)
    definition = snapshot.field_definitions["scalar"]
    for change, message in [
        ({"source_mesh": None}, "topology"),
        ({"field_definitions": {}}, "FieldDefinition"),
        ({"field_definitions": {"scalar": None}}, "Invalid FieldDefinition"),
        ({"fields": {"scalar": None}}, "dolfinx.fem.Function"),
        ({"field_definitions": {"scalar": replace(definition, value_shape=(2,))}}, "value shape"),
        ({"step_fields": {True: ("scalar",)}}, "identifiers"),
        ({"step_fields": {0: ("scalar", "scalar")}}, "duplicate"),
        ({"step_fields": {0: ("unknown",)}}, "unknown"),
    ]:
        with pytest.raises((ValueError, TypeError), match=message):
            replace(snapshot, **change)
    artifacts, _ = _successful_detached_state(refinement_mesh)
    with pytest.raises(ValueError, match="not defined on the source mesh"):
        replace(snapshot, source_mesh=artifacts.refined_mesh)


def test_transferred_state_requires_correct_target_fields_and_diagnostics(refinement_mesh):
    refinement_mesh.Add_Field("scalar", "Scalar", "CG", 1)
    artifacts, state = _successful_detached_state(refinement_mesh)
    for change, message in [
        ({"fields": {}}, "field set"),
        ({"diagnostics": {}}, "diagnostic"),
        ({"target_mesh": refinement_mesh.mesh}, "target mesh"),
        ({"field_definitions": {"scalar": replace(state.field_definitions["scalar"], value_shape=(2,))}}, "value shape"),
    ]:
        with pytest.raises(ValueError, match=message):
            replace(state, **change)
    with pytest.raises(TypeError, match="requires attributes"):
        snapshot_registered_fields(SimpleNamespace(mesh=refinement_mesh.mesh))


def test_partition_and_transfer_reject_mismatched_detached_state(refinement_mesh):
    refinement_mesh.Add_Field("scalar", "Scalar", "CG", 1)
    refinement_mesh.Add_Quadrature_Field("pending", "Pending")
    snapshot = snapshot_registered_fields(refinement_mesh)
    artifacts, complete = _successful_detached_state(refinement_mesh)
    standard = transfer_standard_fields(snapshot, artifacts)
    with pytest.raises(ValueError, match="every registered field"):
        repartition_mesh(artifacts, standard)
    with pytest.raises(ValueError, match="different meshes"):
        repartition_mesh(replace(artifacts, refined_mesh=refinement_mesh.mesh), complete)
    for call, arguments in [
        (repartition_mesh, (None, complete)), (repartition_mesh, (artifacts, None)),
        (transfer_standard_fields, (None, artifacts)), (transfer_standard_fields, (snapshot, None)),
        (transfer_quadrature_fields, (None, artifacts, standard)),
        (transfer_quadrature_fields, (snapshot, None, standard)),
        (transfer_quadrature_fields, (snapshot, artifacts, None)),
    ]:
        with pytest.raises(TypeError):
            call(*arguments)
    for call, arguments in [
        (repartition_mesh, (artifacts, complete)),
        (transfer_standard_fields, (snapshot, artifacts)),
        (transfer_quadrature_fields, (snapshot, artifacts, standard)),
    ]:
        with pytest.raises(TypeError, match="RefinementConfig"):
            call(*arguments, config={})
    assert not repartition_mesh(artifacts, complete, RefinementConfig()).repartitioned
