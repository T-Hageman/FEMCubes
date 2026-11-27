"""Atomic refinement installation, recovery and public preflight contracts."""

from dataclasses import replace

import numpy as np
import pytest
from basix.ufl import element
from dolfinx import fem

from femcubes.core.mesh.refinement import (
    RefinementConfig, FieldThresholdIndicatorConfig, install_refined_mesh_state,
    refine_from_indicator, transfer_quadrature_fields,
    snapshot_registered_fields, transfer_standard_fields, MeshMarkerState,
    transfer_mesh_markers, resolve_owned_child_input_positions,
    FieldTransferSnapshot,
)
from femcubes.core.mesh._mesh_dofs import FieldDefinition
from femcubes.core.mesh.refinement.manager import _validate_step_fields
from femcubes.core.mesh.tests.test_mesh_refinement_manager import (
    _register_manager_fields, _successful_detached_state,
)


def test_failed_rebuild_restores_live_state_and_allows_retry(refinement_mesh, monkeypatch):
    wrapper = refinement_mesh
    _register_manager_fields(wrapper)
    artifacts, fields = _successful_detached_state(wrapper)
    old_mesh, old_fields = wrapper.mesh, wrapper.fields
    old_values = {name: field.x.array.copy() for name, field in wrapper.fields.items()}
    old_markers = (wrapper.domain_names(), wrapper.boundary_names())
    old_steps = {step: list(names) for step, names in wrapper.step_fields.items()}
    original_finalize = wrapper.finalize_test_functions
    plot_calls = []
    monkeypatch.setattr(wrapper, "plot_mesh", lambda **kwargs: plot_calls.append(kwargs))

    def fail_after_storage_replacement():
        assert wrapper.mesh is artifacts.refined_mesh
        raise RuntimeError("synthetic mixed-space rebuild failure")

    monkeypatch.setattr(wrapper, "finalize_test_functions", fail_after_storage_replacement)
    with pytest.raises(RuntimeError, match="Live mesh installation failed.*rebuild failure"):
        wrapper.Install_Refined_State(artifacts, fields)
    assert wrapper.mesh is old_mesh and wrapper.fields is old_fields
    assert wrapper.remesh_epoch == 0
    assert wrapper.step_fields == old_steps
    assert (wrapper.domain_names(), wrapper.boundary_names()) == old_markers
    assert not plot_calls
    for name, values in old_values.items():
        np.testing.assert_array_equal(wrapper.fields[name].x.array, values)

    monkeypatch.setattr(wrapper, "finalize_test_functions", original_finalize)
    result = wrapper.Install_Refined_State(artifacts, fields)
    assert result.installed and wrapper.remesh_epoch == 1
    np.testing.assert_allclose(wrapper.Get_Field("history").x.array, 5.5)


def test_empty_step_membership_survives_installation(refinement_mesh):
    refinement_mesh.Add_Field("temperature", "Temperature", "CG", 1)
    artifacts, fields = _successful_detached_state(refinement_mesh)
    fields = replace(fields, step_fields={4: ()})
    refinement_mesh.step_fields[4] = []
    result = refinement_mesh.Install_Refined_State(artifacts, fields)
    assert result.installed
    assert refinement_mesh.step_fields[4] == []
    np.testing.assert_allclose(refinement_mesh.Get_Field("temperature").x.array, 0.0)


def test_quadrature_history_cannot_be_installed_as_a_solution_unknown(refinement_mesh):
    refinement_mesh.Add_Quadrature_Field("history", "History", ())
    refinement_mesh.Add_TestFunction("history", step=0)
    refinement_mesh.finalize_test_functions()
    artifacts, state = _successful_detached_state(refinement_mesh)
    with pytest.raises(ValueError, match="must use CG or DG"):
        refinement_mesh.Install_Refined_State(artifacts, state)


def test_tensor_coefficients_transfer_but_cannot_be_solution_unknowns(refinement_mesh):
    mesh = refinement_mesh.mesh
    space = fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1, shape=(2, 2)))
    tensor = fem.Function(space)
    tensor.x.array[:] = np.tile([1.0, 2.0, 3.0, 4.0], tensor.x.array.size//4)
    tensor.x.scatter_forward()
    definition = FieldDefinition("tensor", "Tensor coefficient", "CG", 1, (2, 2))
    snapshot = FieldTransferSnapshot(mesh, {"tensor": tensor}, {"tensor": definition}, {0: ("tensor",)})
    artifacts, _ = _successful_detached_state(refinement_mesh)
    transferred = transfer_standard_fields(snapshot, artifacts)
    np.testing.assert_allclose(transferred.fields["tensor"].x.array.reshape(-1, 4),
                               np.tile([1.0, 2.0, 3.0, 4.0], (transferred.fields["tensor"].x.array.size//4, 1)))
    # The current mesh registry exposes no public CG tensor registration; protect
    # the installer's rejection of tensor solution unknowns at its validation boundary.
    with pytest.raises(ValueError, match="unsupported tensor shape"):
        _validate_step_fields(transferred)


def test_noop_artifacts_must_retain_the_live_mesh(refinement_mesh):
    artifacts, state = _successful_detached_state(refinement_mesh)
    invalid = replace(artifacts, report=replace(artifacts.report, occurred=False),
                      parent_cell=np.empty(0, dtype=np.int32), parent_facet=np.empty(0, dtype=np.int8),
                      marked_edges=np.empty(0, dtype=np.int32))
    with pytest.raises(ValueError, match="No-op refinement artifacts must retain"):
        refinement_mesh.Install_Refined_State(invalid)
    with pytest.raises(ValueError, match="No-op refinement artifacts must retain"):
        transfer_mesh_markers(MeshMarkerState.from_mesh_wrapper(refinement_mesh), invalid)
    valid = replace(invalid, refined_mesh=refinement_mesh.mesh)
    assert resolve_owned_child_input_positions(valid).size == 0


def test_installation_requires_the_transferred_target_to_match_topology(refinement_mesh):
    artifacts, state = _successful_detached_state(refinement_mesh)
    with pytest.raises(ValueError, match="different target meshes"):
        refinement_mesh.Install_Refined_State(artifacts, replace(state, target_mesh=refinement_mesh.mesh))


@pytest.mark.parametrize("family", ["CG", "Quadrature"])
def test_noop_transfer_cannot_copy_coefficients_to_a_different_mesh(refinement_mesh, family):
    if family == "CG":
        refinement_mesh.Add_Field("field", "Field", "CG", 1)
    else:
        refinement_mesh.Add_Quadrature_Field("field", "Field", ())
    snapshot = snapshot_registered_fields(refinement_mesh)
    artifacts, state = _successful_detached_state(refinement_mesh)
    invalid = replace(artifacts, report=replace(artifacts.report, occurred=False),
                      parent_cell=np.empty(0, dtype=np.int32), parent_facet=np.empty(0, dtype=np.int8),
                      marked_edges=np.empty(0, dtype=np.int32))
    with pytest.raises(ValueError, match="No-op refinement artifacts must retain the source mesh"):
        standard = transfer_standard_fields(snapshot, invalid)
        if family == "Quadrature":
            transfer_quadrature_fields(snapshot, invalid, standard)


@pytest.mark.parametrize("change, message", [
    ({"transaction_source_mesh": None}, "do not originate"),
    ({"step_fields": {7: ("p",)}}, "step membership"),
    ({"fields": "reversed"}, "registration order"),
    ({"diagnostics": "nonfinite"}, "not finite"),
    ({"field_definitions": "description"}, "field definitions"),
])
def test_installation_rejects_inconsistent_metadata_without_mutation(refinement_mesh, change, message):
    wrapper = refinement_mesh
    _register_manager_fields(wrapper)
    artifacts, state = _successful_detached_state(wrapper)
    values = dict(change)
    if "transaction_source_mesh" in values:
        values["transaction_source_mesh"] = artifacts.refined_mesh
    if values.get("fields") == "reversed":
        values["fields"] = dict(reversed(list(state.fields.items())))
    if values.get("diagnostics") == "nonfinite":
        diagnostics = dict(state.diagnostics)
        diagnostics["p"] = replace(diagnostics["p"], finite=False)
        values["diagnostics"] = diagnostics
    if values.get("field_definitions") == "description":
        definitions = dict(state.field_definitions)
        definitions["p"] = replace(definitions["p"], description="Different metadata")
        values["field_definitions"] = definitions
    invalid = replace(state, **values)
    old_mesh, old_fields = wrapper.mesh, wrapper.fields
    with pytest.raises(ValueError, match=message):
        wrapper.Install_Refined_State(artifacts, invalid)
    assert wrapper.mesh is old_mesh and wrapper.fields is old_fields
    assert wrapper.remesh_epoch == 0


@pytest.mark.parametrize("case, message", [
    ("no_state", "TransferredFieldState"), ("bad_artifacts", "RefinedMeshArtifacts"),
    ("same_mesh", "new target mesh"), ("negative_epoch", "nonnegative"),
    ("cell_dimension", "cell tags"), ("facet_dimension", "facet tags"),
    ("missing_cells", "domains require cell tags"), ("missing_facets", "boundaries require facet tags"),
])
def test_installation_rejects_inconsistent_topology(refinement_mesh, case, message):
    wrapper = refinement_mesh
    artifacts, state = _successful_detached_state(wrapper)
    if case == "no_state":
        state = None
    elif case == "bad_artifacts":
        artifacts = None
    elif case == "same_mesh":
        artifacts = replace(artifacts, refined_mesh=wrapper.mesh)
    elif case == "negative_epoch":
        wrapper.remesh_epoch = -1
    elif case == "cell_dimension":
        artifacts = replace(artifacts, cell_tags=artifacts.facet_tags)
    elif case == "facet_dimension":
        artifacts = replace(artifacts, facet_tags=artifacts.cell_tags)
    elif case == "missing_cells":
        artifacts = replace(artifacts, cell_tags=None)
    elif case == "missing_facets":
        artifacts = replace(artifacts, facet_tags=None)
    old_mesh = wrapper.mesh
    with pytest.raises(ValueError, match=message):
        wrapper.Install_Refined_State(artifacts, state)
    assert wrapper.mesh is old_mesh


def test_refinement_preflight_reports_invalid_controls_collectively(refinement_mesh):
    with pytest.raises(TypeError, match="live DOLFINx mesh"):
        refine_from_indicator(None)
    with pytest.raises(TypeError, match="live DOLFINx mesh"):
        install_refined_mesh_state(None, None)
    with pytest.raises(ValueError, match="enabled=True"):
        refine_from_indicator(refinement_mesh, RefinementConfig())


@pytest.mark.parametrize("attribute, value, message", [
    ("movie_fps", False, "movie_fps"), ("movie_fps", 0, "movie_fps"),
    ("movie_folder", object(), "path-like"),
])
def test_invalid_animation_controls_leave_live_mesh_intact(refinement_mesh, attribute, value, message):
    artifacts, state = _successful_detached_state(refinement_mesh)
    setattr(refinement_mesh.params, attribute, value)
    old_mesh = refinement_mesh.mesh
    with pytest.raises(ValueError, match=message):
        refinement_mesh.Install_Refined_State(artifacts, state, RefinementConfig(save_remesh_animations=True))
    assert refinement_mesh.mesh is old_mesh and refinement_mesh.remesh_epoch == 0


def test_refinement_resolves_default_controls_and_disabled_movie_path(refinement_mesh, monkeypatch):
    wrapper = refinement_mesh
    wrapper.Add_Field("indicator", "Indicator", "DG", 0)
    wrapper.Constant_Field("indicator", 1.0)
    wrapper.params.MeshRefinement = RefinementConfig(
        enabled=True, target_h=0.1, indicator=FieldThresholdIndicatorConfig("indicator", 0.5),
        save_remesh_animations=True,
    )
    wrapper.params.output_dir = None
    wrapper.params.movie_folder = None
    calls = []
    monkeypatch.setattr(wrapper, "plot_mesh", lambda *args, **kwargs: calls.append(kwargs))
    assert wrapper.Refine_From_Indicator().installed
    assert calls and all(call["movie_path"].endswith("Results/movies/mesh_refinement.mp4") for call in calls)


def test_quadrature_transfer_cannot_be_replayed_or_mixed_with_other_transaction(refinement_mesh):
    wrapper = refinement_mesh
    wrapper.Add_Field("scalar", "Scalar", "CG", 1)
    wrapper.Add_Quadrature_Field("ip", "IP", ())
    snapshot = snapshot_registered_fields(wrapper)
    artifacts, complete = _successful_detached_state(wrapper)
    standard = transfer_standard_fields(snapshot, artifacts)
    with pytest.raises(ValueError, match="already present"):
        transfer_quadrature_fields(snapshot, artifacts, complete)
    with pytest.raises(ValueError, match="different source"):
        transfer_quadrature_fields(replace(snapshot, source_mesh=artifacts.refined_mesh,
                                          fields=complete.fields), artifacts, standard)
    with pytest.raises(ValueError, match="different target"):
        transfer_quadrature_fields(snapshot, replace(artifacts, refined_mesh=wrapper.mesh), standard)
    with pytest.raises(ValueError, match="definitions"):
        transfer_quadrature_fields(snapshot, artifacts, replace(standard, field_definitions={
            name: replace(definition, description="Different")
            for name, definition in standard.field_definitions.items()
        }))
    with pytest.raises(ValueError, match="step membership"):
        transfer_quadrature_fields(snapshot, artifacts, replace(standard, step_fields={1: ("scalar",)}))


@pytest.mark.parametrize("family", ["CG", "Quadrature"])
@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_interpolation_failure_preserves_live_source(refinement_mesh, monkeypatch, family, failure):
    wrapper = refinement_mesh
    if family == "CG":
        field = wrapper.Add_Field("field", "Field", "CG", 1)
    else:
        field = wrapper.Add_Quadrature_Field("field", "Field", ())
    wrapper.Constant_Field("field", 3.25)
    snapshot = snapshot_registered_fields(wrapper)
    artifacts, _ = _successful_detached_state(wrapper)
    old_mesh, old_values = wrapper.mesh, field.x.array.copy()
    original = fem.Function.interpolate

    def fail_interpolation(destination, *args, **kwargs):
        if failure == "exception":
            raise RuntimeError("synthetic interpolation failure")
        original(destination, *args, **kwargs)
        destination.x.array[:] = np.nan

    monkeypatch.setattr(fem.Function, "interpolate", fail_interpolation)
    with pytest.raises(ValueError, match="interpolation failure|non-finite owned coefficients"):
        standard = transfer_standard_fields(snapshot, artifacts)
        if family == "Quadrature":
            transfer_quadrature_fields(snapshot, artifacts, standard)
    assert wrapper.mesh is old_mesh and wrapper.remesh_epoch == 0
    np.testing.assert_array_equal(field.x.array, old_values)
