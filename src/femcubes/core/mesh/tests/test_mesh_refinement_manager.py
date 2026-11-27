"""Live installation tests for detached mesh-refinement transactions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from ufl.domain import extract_unique_domain

from femcubes.core.mesh.refinement import (
    CellIndicatorResult,
    FieldThresholdIndicatorConfig,
    MeshMarkerState,
    RepartitionResult,
    RefinementConfig,
    expand_refinement_buffer,
    refine_mesh_topology,
    snapshot_registered_fields,
    transfer_mesh_markers,
    transfer_quadrature_fields,
    transfer_standard_fields,
)


def _selection(mesh, marked_mask):
    """Return one complete owned-cell selection for topology refinement."""

    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    marked_mask = np.asarray(marked_mask, dtype=bool)
    indicator = CellIndicatorResult(
        cells=cells,
        values=marked_mask.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=marked_mask.copy(),
        size_mask=np.ones(cells.size, dtype=bool),
        marked_mask=marked_mask.copy(),
    )
    return expand_refinement_buffer(SimpleNamespace(mesh=mesh), indicator, 0, target_h=0.1)


def _register_manager_fields(mesh_wrapper):
    """Register representative plain, mixed, and quadrature storage."""

    mesh_wrapper.Add_Field("p", "Pressure", "CG", 1)
    mesh_wrapper.Add_Vector_Field("u", "Displacement", "CG", 1, 2)
    mesh_wrapper.Add_Field("history", "History", "DG", 0)
    mesh_wrapper.Add_Quadrature_Field("PsiField", "Driving force")
    mesh_wrapper.Add_TestFunction("p", step=0)
    mesh_wrapper.Add_Vector_TestFunction("u", nDim=2, step=0)
    mesh_wrapper.Add_TestFunction("history", step=1)
    mesh_wrapper.finalize_test_functions()

    mesh_wrapper.Constant_Field("p", 2.25)
    mesh_wrapper.Constant_Field("u", -0.75)
    mesh_wrapper.Constant_Field("history", 5.5)
    mesh_wrapper.Constant_Field("PsiField", 9.0)


def _successful_detached_state(mesh_wrapper):
    """Build complete topology, marker, and field artifacts off to the side."""

    marker_state = MeshMarkerState.from_mesh_wrapper(mesh_wrapper)
    snapshot = snapshot_registered_fields(mesh_wrapper)
    cell_count = mesh_wrapper.mesh.topology.index_map(
        mesh_wrapper.mesh.topology.dim
    ).size_local
    marked = np.zeros(cell_count, dtype=bool)
    marked[cell_count // 2] = True
    artifacts = refine_mesh_topology(
        mesh_wrapper.mesh,
        _selection(mesh_wrapper.mesh, marked),
    )
    artifacts = transfer_mesh_markers(marker_state, artifacts)
    transferred = transfer_standard_fields(snapshot, artifacts)
    transferred = transfer_quadrature_fields(
        snapshot,
        artifacts,
        transferred,
    )
    return artifacts, transferred


def _empty_successful_detached_state(mesh_wrapper):
    """Build a complete successful state for a wrapper with no fields."""

    marker_state = MeshMarkerState.from_mesh_wrapper(mesh_wrapper)
    snapshot = snapshot_registered_fields(mesh_wrapper)
    cell_count = mesh_wrapper.mesh.topology.index_map(
        mesh_wrapper.mesh.topology.dim
    ).size_local
    marked = np.zeros(cell_count, dtype=bool)
    marked[cell_count // 2] = True
    artifacts = refine_mesh_topology(
        mesh_wrapper.mesh,
        _selection(mesh_wrapper.mesh, marked),
    )
    artifacts = transfer_mesh_markers(marker_state, artifacts)
    transferred = transfer_standard_fields(snapshot, artifacts)
    return artifacts, transferred


def test_initial_epoch_and_noop_install_preserve_every_live_identity(refinement_mesh):
    mesh = refinement_mesh.mesh
    cell_count = mesh.topology.index_map(mesh.topology.dim).size_local
    no_op = refine_mesh_topology(
        mesh,
        _selection(mesh, np.zeros(cell_count, dtype=bool)),
    )
    identities = {
        name: id(getattr(refinement_mesh, name))
        for name in (
            "mesh",
            "fields",
            "field_definitions",
            "test_functions",
            "trial_functions",
            "mixed_spaces",
            "step_fields",
            "cell_tags",
            "facet_tags",
        )
    }

    result = refinement_mesh.Install_Refined_State(no_op)

    assert not result.installed
    assert result.epoch_before == result.epoch_after == 0
    assert refinement_mesh.remesh_epoch == 0
    assert refinement_mesh.last_refinement_report is None
    assert identities == {
        name: id(getattr(refinement_mesh, name)) for name in identities
    }


@pytest.mark.parametrize(
    ("indicator_field", "threshold"),
    (("p", 2.0), ("PsiField", 8.0)),
)
def test_public_indicator_transaction_refines_and_transfers_all_registered_state(
    refinement_mesh,
    indicator_field,
    threshold,
):
    _register_manager_fields(refinement_mesh)
    old_mesh = refinement_mesh.mesh
    old_cells = old_mesh.topology.index_map(old_mesh.topology.dim).size_global
    config = RefinementConfig(
        enabled=True,
        indicator=FieldThresholdIndicatorConfig(
            field=indicator_field,
            threshold=threshold,
        ),
        target_h=0.4,
        buffer_layers=0,
    )

    result = refinement_mesh.Refine_From_Indicator(config)

    new_mesh = refinement_mesh.mesh
    new_cells = new_mesh.topology.index_map(new_mesh.topology.dim).size_global
    assert result.installed
    assert result.epoch_before == 0
    assert result.epoch_after == 1
    assert new_mesh is not old_mesh
    assert new_cells > old_cells
    stage_seconds = (
        result.preparation_seconds,
        result.indicator_seconds,
        result.buffer_seconds,
        result.report.topology_seconds,
        result.marker_transfer_seconds,
        result.standard_transfer_seconds,
        result.quadrature_transfer_seconds,
        result.install_seconds,
    )
    assert all(seconds > 0.0 for seconds in stage_seconds)
    assert result.total_seconds >= sum(stage_seconds)
    assert result.install_seconds >= result.visualization_seconds
    assert set(result.initial_transfer_diagnostics) == set(refinement_mesh.fields)
    assert result.initial_transfer_diagnostics["p"].search_seconds == 0.0
    assert result.initial_transfer_diagnostics["PsiField"].reconstruction_seconds > 0.0
    np.testing.assert_allclose(refinement_mesh.fields["p"].x.array, 2.25)
    np.testing.assert_allclose(refinement_mesh.fields["u"].x.array, -0.75)
    np.testing.assert_allclose(refinement_mesh.fields["history"].x.array, 5.5)
    np.testing.assert_allclose(refinement_mesh.fields["PsiField"].x.array, 9.0)
    assert set(refinement_mesh.boundary_names()) == {"left", "right", "bottom", "top"}
    assert all(
        refinement_mesh.get_boundary_facets(name).size > 0
        for name in refinement_mesh.boundary_names()
    )


def test_public_indicator_transaction_noop_preserves_mesh_and_epoch(refinement_mesh):
    refinement_mesh.Add_Field("indicator", "Indicator", "DG", 0)
    refinement_mesh.Constant_Field("indicator", 0.0)
    old_mesh = refinement_mesh.mesh
    config = RefinementConfig(
        enabled=True,
        indicator=FieldThresholdIndicatorConfig(
            field="indicator",
            threshold=1.0,
        ),
        target_h=0.4,
    )

    result = refinement_mesh.Refine_From_Indicator(config)

    assert not result.installed
    assert result.epoch_before == result.epoch_after == 0
    assert refinement_mesh.mesh is old_mesh
    assert refinement_mesh.remesh_epoch == 0
    assert result.preparation_seconds > 0.0
    assert result.indicator_seconds > 0.0
    assert result.buffer_seconds > 0.0
    assert result.report.topology_seconds > 0.0
    assert result.marker_transfer_seconds == 0.0
    assert result.standard_transfer_seconds == 0.0
    assert result.quadrature_transfer_seconds == 0.0
    assert not result.initial_transfer_diagnostics
    assert result.total_seconds >= result.install_seconds > 0.0


def test_serial_repartition_option_preserves_single_rank_ownership(refinement_mesh):
    refinement_mesh.Add_Field("indicator", "Indicator", "DG", 0)
    refinement_mesh.Constant_Field("indicator", 1.0)
    config = RefinementConfig(
        enabled=True,
        indicator=FieldThresholdIndicatorConfig(
            field="indicator",
            threshold=0.5,
        ),
        target_h=0.4,
        buffer_layers=0,
        repartition=True,
        repartition_imbalance_threshold=1.0,
    )

    result = refinement_mesh.Refine_From_Indicator(config)

    assert result.installed
    assert not result.report.repartitioned
    assert result.report.imbalance_ratio_before_repartition == pytest.approx(1.0)
    assert result.report.imbalance_ratio_after == pytest.approx(1.0)
    assert result.report.repartition_seconds == 0.0


def test_repartition_stage_receives_complete_transferred_state(
    refinement_mesh,
    monkeypatch,
):
    import femcubes.core.mesh.refinement.manager as manager

    _register_manager_fields(refinement_mesh)
    observed = []

    def track_repartition(artifacts, transferred_state, config):
        observed.append(
            (
                artifacts.cell_tags is not None,
                artifacts.facet_tags is not None,
                transferred_state.pending_field_names,
                tuple(transferred_state.fields),
            )
        )
        return RepartitionResult(artifacts, transferred_state)

    monkeypatch.setattr(manager, "repartition_mesh", track_repartition)
    config = RefinementConfig(
        enabled=True,
        indicator=FieldThresholdIndicatorConfig(field="p", threshold=2.0),
        target_h=0.4,
        buffer_layers=0,
    )

    result = refinement_mesh.Refine_From_Indicator(config)

    assert result.installed
    assert observed == [
        (
            True,
            True,
            (),
            ("p", "u", "history", "PsiField"),
        )
    ]


def test_detached_repartition_failure_preserves_live_state(
    refinement_mesh,
    monkeypatch,
):
    import femcubes.core.mesh.refinement.manager as manager

    _register_manager_fields(refinement_mesh)
    old_mesh = refinement_mesh.mesh
    old_fields = refinement_mesh.fields
    old_cell_tags = refinement_mesh.cell_tags
    old_facet_tags = refinement_mesh.facet_tags
    def fail_repartition(*args, **kwargs):
        raise RuntimeError("injected redistribution failure")

    monkeypatch.setattr(manager, "repartition_mesh", fail_repartition)
    config = RefinementConfig(
        enabled=True,
        indicator=FieldThresholdIndicatorConfig(field="p", threshold=2.0),
        target_h=0.4,
        buffer_layers=0,
        repartition=True,
        repartition_imbalance_threshold=1.0,
    )

    with pytest.raises(RuntimeError, match="injected redistribution failure"):
        refinement_mesh.Refine_From_Indicator(config)

    assert refinement_mesh.mesh is old_mesh
    assert refinement_mesh.fields is old_fields
    assert refinement_mesh.cell_tags is old_cell_tags
    assert refinement_mesh.facet_tags is old_facet_tags
    assert refinement_mesh.remesh_epoch == 0


def test_successful_install_preserves_wrapper_and_rebuilds_complete_live_state(
    refinement_mesh,
):
    _register_manager_fields(refinement_mesh)
    artifacts, transferred = _successful_detached_state(refinement_mesh)
    wrapper_identity = id(refinement_mesh)
    old_mesh = refinement_mesh.mesh
    definitions = dict(refinement_mesh.field_definitions)
    boundary_names = refinement_mesh.boundary_names()

    result = refinement_mesh.Install_Refined_State(artifacts, transferred)

    assert result.installed
    assert id(refinement_mesh) == wrapper_identity
    assert refinement_mesh.mesh is artifacts.refined_mesh
    assert refinement_mesh.mesh is not old_mesh
    assert refinement_mesh.remesh_epoch == 1
    assert result.epoch_before == 0
    assert result.epoch_after == 1
    assert refinement_mesh.last_refinement_report is artifacts.report
    assert refinement_mesh.field_definitions == definitions
    assert list(refinement_mesh.fields) == list(definitions)
    assert all(
        field.function_space.mesh is refinement_mesh.mesh
        for field in refinement_mesh.fields.values()
    )
    for boundary_name in boundary_names:
        assert refinement_mesh.get_boundary_facets(boundary_name).size > 0


def test_mixed_and_plain_fields_retain_transferred_values(refinement_mesh):
    _register_manager_fields(refinement_mesh)
    artifacts, transferred = _successful_detached_state(refinement_mesh)

    refinement_mesh.Install_Refined_State(artifacts, transferred)

    np.testing.assert_allclose(refinement_mesh.fields["p"].x.array, 2.25)
    np.testing.assert_allclose(refinement_mesh.fields["u"].x.array, -0.75)
    np.testing.assert_allclose(refinement_mesh.fields["history"].x.array, 5.5)
    np.testing.assert_allclose(refinement_mesh.fields["PsiField"].x.array, 9.0)
    # The rebuilt block layout of the monolithic step uses the new fields' spaces.
    block_layout = refinement_mesh.mixed_spaces[0]
    assert block_layout["field_names"] == ["p", "u"]
    assert all(
        space is refinement_mesh.fields[name].function_space
        for name, space in zip(block_layout["field_names"], block_layout["function_spaces"])
    )


@pytest.mark.filterwarnings("error:Expr.ufl_domain:DeprecationWarning")
def test_test_and_trial_functions_are_rebuilt_on_the_new_domain(refinement_mesh):
    _register_manager_fields(refinement_mesh)
    old_test = refinement_mesh.Get_TestFunction("p", step=0)
    old_trial = refinement_mesh.Get_Trial_Function("u", step=0)
    artifacts, transferred = _successful_detached_state(refinement_mesh)

    refinement_mesh.Install_Refined_State(artifacts, transferred)

    new_test = refinement_mesh.Get_TestFunction("p", step=0)
    new_trial = refinement_mesh.Get_Trial_Function("u", step=0)
    assert new_test is not old_test
    assert new_trial is not old_trial
    assert extract_unique_domain(new_test).ufl_cargo() is refinement_mesh.mesh._cpp_object
    assert extract_unique_domain(new_trial).ufl_cargo() is refinement_mesh.mesh._cpp_object
    assert set(refinement_mesh.trial_functions) == {
        "p_trial_step0",
        "u_trial_step0",
        "history_trial_step1",
    }


def test_incomplete_transfer_fails_preflight_without_live_mutation_or_plot(
    refinement_mesh,
    monkeypatch,
):
    _register_manager_fields(refinement_mesh)
    marker_state = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    snapshot = snapshot_registered_fields(refinement_mesh)
    cell_count = refinement_mesh.mesh.topology.index_map(
        refinement_mesh.mesh.topology.dim
    ).size_local
    marked = np.zeros(cell_count, dtype=bool)
    marked[0] = True
    artifacts = refine_mesh_topology(
        refinement_mesh.mesh,
        _selection(refinement_mesh.mesh, marked),
    )
    artifacts = transfer_mesh_markers(marker_state, artifacts)
    incomplete = transfer_standard_fields(snapshot, artifacts)
    old_mesh = refinement_mesh.mesh
    old_fields = refinement_mesh.fields
    plot_calls = []
    monkeypatch.setattr(
        refinement_mesh,
        "plot_mesh",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )

    with pytest.raises(ValueError, match="pending fields.*PsiField"):
        refinement_mesh.Install_Refined_State(
            artifacts,
            incomplete,
            RefinementConfig(
                replot_on_refinement=True,
                save_remesh_animations=True,
            ),
        )

    assert refinement_mesh.mesh is old_mesh
    assert refinement_mesh.fields is old_fields
    assert refinement_mesh.remesh_epoch == 0
    assert refinement_mesh.last_refinement_report is None
    assert plot_calls == []


def test_first_animated_install_records_epoch_zero_then_epoch_one(
    refinement_mesh,
    tmp_path,
    monkeypatch,
):
    artifacts, transferred = _empty_successful_detached_state(refinement_mesh)
    initial_global_dofs = refinement_mesh.get_global_solution_dof_count()
    refinement_mesh.params.output_dir = str(tmp_path)
    calls = []

    def track_plot(mesh=None, **kwargs):
        calls.append((mesh, kwargs))
        return SimpleNamespace(
            visible_updated=kwargs["show_window"],
            movie_frame_written=kwargs["movie_path"] is not None,
        )

    monkeypatch.setattr(refinement_mesh, "plot_mesh", track_plot)
    result = refinement_mesh.Install_Refined_State(
        artifacts,
        transferred,
        RefinementConfig(save_remesh_animations=True),
    )

    assert len(calls) == 2
    assert "Mesh epoch 000" in calls[0][1]["annotation"]
    assert "Mesh epoch 001" in calls[1][1]["annotation"]
    assert (
        f"Global DOFs: {initial_global_dofs}"
        in calls[0][1]["annotation"]
    )
    assert (
        f"Global DOFs: {refinement_mesh.get_global_solution_dof_count()}"
        in calls[1][1]["annotation"]
    )
    assert calls[0][0] is not artifacts.refined_mesh
    assert calls[1][0] is artifacts.refined_mesh
    expected_movie = tmp_path / "movies" / "mesh_refinement.mp4"
    assert calls[0][1]["movie_path"] == str(expected_movie)
    assert calls[1][1]["movie_path"] == str(expected_movie)
    assert not calls[0][1]["show_window"]
    assert not calls[1][1]["show_window"]
    assert result.initial_animation_frame_written
    assert result.animation_frame_written
    assert not result.visible_plot_refreshed
    assert result.visualization_seconds >= 0.0


@pytest.mark.parametrize("replot", [False, True])
def test_successful_install_refreshes_visible_plot_only_when_requested(
    refinement_mesh,
    monkeypatch,
    replot,
):
    artifacts, transferred = _empty_successful_detached_state(refinement_mesh)
    calls = []

    def track_plot(mesh=None, **kwargs):
        calls.append((mesh, kwargs))
        return SimpleNamespace(
            visible_updated=kwargs["show_window"],
            movie_frame_written=False,
        )

    monkeypatch.setattr(refinement_mesh, "plot_mesh", track_plot)
    result = refinement_mesh.Install_Refined_State(
        artifacts,
        transferred,
        RefinementConfig(replot_on_refinement=replot),
    )

    assert len(calls) == int(replot)
    assert result.visible_plot_refreshed is replot


def test_successful_install_refreshes_an_already_open_mesh_window(
    refinement_mesh,
    monkeypatch,
):
    """An existing user-positioned mesh window should override replot=False."""

    artifacts, transferred = _empty_successful_detached_state(refinement_mesh)
    retained_plotter = object()
    refinement_mesh._mesh_plotter = retained_plotter
    refinement_mesh._mesh_plot_visible = True
    calls = []

    def track_plot(mesh=None, **kwargs):
        calls.append((mesh, kwargs))
        return SimpleNamespace(
            visible_updated=kwargs["show_window"],
            movie_frame_written=False,
        )

    monkeypatch.setattr(refinement_mesh, "plot_mesh", track_plot)
    result = refinement_mesh.Install_Refined_State(
        artifacts,
        transferred,
        RefinementConfig(replot_on_refinement=False),
    )

    assert refinement_mesh._mesh_plotter is retained_plotter
    assert len(calls) == 1
    assert calls[0][0] is artifacts.refined_mesh
    assert calls[0][1]["show_window"] is True
    assert result.visible_plot_refreshed


def test_live_install_centrally_invalidates_every_mesh_owned_cache(refinement_mesh):
    _register_manager_fields(refinement_mesh)

    class TrackingNativeObject:
        def __init__(self):
            self.destroy_calls = 0

        def destroy(self):
            self.destroy_calls += 1

    class TrackingWriter:
        def __init__(self, wrapper):
            self.close_calls = 0
            self.wrapper = wrapper
            self.mesh_at_close = None

        def close(self):
            self.close_calls += 1
            self.mesh_at_close = self.wrapper.mesh

    old_space = refinement_mesh._get_dg_space(2)
    refinement_mesh.get_Boundary_Integrator()
    refinement_mesh.Get_Weights_Dofs(["p"], step=0)
    refinement_mesh._build_plot_mesh_data()
    old_projection = object()
    old_expression = object()
    old_matrix = TrackingNativeObject()
    old_ksp = TrackingNativeObject()
    old_writer = TrackingWriter(refinement_mesh)
    refinement_mesh._output_projection_cache[("synthetic", id(old_space))] = old_projection
    refinement_mesh._output_expression_cache[("synthetic", id(old_space))] = old_expression
    refinement_mesh._output_mass_projection_cache[id(old_space)] = (
        None,
        old_matrix,
        old_ksp,
    )
    refinement_mesh._vtk_file = old_writer
    artifacts, transferred = _successful_detached_state(refinement_mesh)

    result = refinement_mesh.Install_Refined_State(artifacts, transferred)

    assert result.cache_invalidation_seconds >= 0.0
    assert old_writer.close_calls == 1
    assert old_writer.mesh_at_close is old_space.mesh
    assert old_matrix.destroy_calls == 1
    assert old_ksp.destroy_calls == 1
    assert refinement_mesh._dg_spaces == {}
    assert refinement_mesh._cg_spaces == {}
    assert refinement_mesh._quadrature_spaces == {}
    assert refinement_mesh._output_projection_cache == {}
    assert refinement_mesh._output_expression_cache == {}
    assert refinement_mesh._output_mass_projection_cache == {}
    assert refinement_mesh._boundary_integrator is None
    assert refinement_mesh._boundary_integrator_cache_key is None
    assert refinement_mesh._lumped_weights_cache == {}
    assert refinement_mesh._plot_mesh_data is None
    assert refinement_mesh._vtk_file is None


def test_boundary_integrator_rebuilds_once_on_the_new_epoch(refinement_mesh):
    old_integrator = refinement_mesh.get_Boundary_Integrator()
    old_cache_key = refinement_mesh._boundary_integrator_cache_key
    artifacts, transferred = _empty_successful_detached_state(refinement_mesh)

    refinement_mesh.Install_Refined_State(artifacts, transferred)
    new_integrator = refinement_mesh.get_Boundary_Integrator()
    repeated = refinement_mesh.get_Boundary_Integrator()

    assert new_integrator is repeated
    assert new_integrator is not old_integrator
    assert refinement_mesh._boundary_integrator_cache_key != old_cache_key
    assert refinement_mesh._boundary_integrator_cache_key[0] == 1
    assert new_integrator.ufl_domain().ufl_cargo() is refinement_mesh.mesh._cpp_object


def test_lumped_weights_rebuild_with_the_refined_layout_and_reuse_within_epoch(
    refinement_mesh,
):
    _register_manager_fields(refinement_mesh)
    old_weights, _, _ = refinement_mesh.Get_Weights_Dofs(["p"], step=0)
    artifacts, transferred = _successful_detached_state(refinement_mesh)

    refinement_mesh.Install_Refined_State(artifacts, transferred)
    new_weights, new_dofs, _ = refinement_mesh.Get_Weights_Dofs(["p"], step=0)
    repeated_weights, repeated_dofs, _ = refinement_mesh.Get_Weights_Dofs(
        ["p"],
        step=0,
    )

    assert new_weights is repeated_weights
    assert new_dofs["p"] is repeated_dofs["p"]
    assert new_weights.size > old_weights.size
    assert np.sum(new_weights) == pytest.approx(2.0)
    assert next(iter(refinement_mesh._lumped_weights_cache))[0] == 1


def test_output_projection_native_objects_are_destroyed_and_rebuilt(refinement_mesh):
    _register_manager_fields(refinement_mesh)
    assert refinement_mesh.Prepare_Field_For_Output("PsiField") is not None
    old_projection = next(iter(refinement_mesh._output_projection_cache.values()))
    _old_test, old_matrix, old_ksp = next(
        iter(refinement_mesh._output_mass_projection_cache.values())
    )
    assert old_matrix.handle != 0
    assert old_ksp.handle != 0
    artifacts, transferred = _successful_detached_state(refinement_mesh)

    refinement_mesh.Install_Refined_State(artifacts, transferred)

    assert old_matrix.handle == 0
    assert old_ksp.handle == 0
    assert refinement_mesh._output_projection_cache == {}
    assert refinement_mesh._output_mass_projection_cache == {}

    assert refinement_mesh.Prepare_Field_For_Output("PsiField") is not None
    new_projection = next(iter(refinement_mesh._output_projection_cache.values()))
    new_mass_entry = next(iter(refinement_mesh._output_mass_projection_cache.values()))
    assert new_projection is not old_projection
    assert new_mass_entry[1] is not old_matrix
    assert new_mass_entry[2] is not old_ksp

    refinement_mesh.Prepare_Field_For_Output("PsiField")
    assert next(iter(refinement_mesh._output_projection_cache.values())) is new_projection
    assert next(iter(refinement_mesh._output_mass_projection_cache.values())) is new_mass_entry
    refinement_mesh._clear_output_projection_caches()


def test_plot_mesh_data_rebuilds_for_new_cells_and_reuses_within_epoch(refinement_mesh):
    old_data = refinement_mesh._build_plot_mesh_data()
    old_cell_types = old_data[1]
    old_space = old_data[3]
    artifacts, transferred = _empty_successful_detached_state(refinement_mesh)

    refinement_mesh.Install_Refined_State(artifacts, transferred)

    assert refinement_mesh._plot_mesh_data is None
    new_data = refinement_mesh._build_plot_mesh_data()
    repeated = refinement_mesh._build_plot_mesh_data()
    assert new_data is repeated
    assert new_data is not old_data
    assert new_data[3] is not old_space
    assert new_data[3].mesh is refinement_mesh.mesh
    assert new_data[1].size > old_cell_types.size
    assert refinement_mesh._plot_mesh_data_epoch == 1
