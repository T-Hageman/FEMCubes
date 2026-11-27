"""Collective refinement failures must leave every rank in the same state.

Run with mpiexec -n 2 python -m pytest on this file. These checks exercise
real shared-facet meshes and detached transfers, with rank-local bad inputs
or a failure injected after a completed installation to test rollback.
"""

from dataclasses import replace
from types import SimpleNamespace

from dolfinx import fem, mesh as dmesh
from dolfinx.graph import adjacencylist
from mpi4py import MPI
import numpy as np
import pytest

from femcubes.core.mesh.refinement import (
    CellIndicatorResult, FieldThresholdIndicatorConfig, MeshMarkerState,
    RefinementConfig, evaluate_field_indicator, expand_refinement_buffer,
    refine_mesh_topology, snapshot_registered_fields, transfer_mesh_markers, migrate_repartitioned_fields,
    transfer_standard_fields, transfer_quadrature_fields, repartition_mesh, redistribute_mesh_markers,
)
from femcubes.core.mesh._mesh_dofs import FieldDefinition
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.mesh.tests.test_mesh_refinement_manager import _selection, _successful_detached_state
from femcubes.core.mesh.tests.test_mesh_refinement_migration import _migration_case, _check_exact_migration, _rotate_cell_ownership
import femcubes.core.mesh.refinement.manager as manager_module
import femcubes.core.mesh.refinement.markers as markers_module
import femcubes.core.mesh.refinement.partition as partition_module
import femcubes.core.mesh.refinement.indicators as indicators_module

pytestmark = [pytest.mark.mpi, pytest.mark.skipif(MPI.COMM_WORLD.size != 2, reason="requires two MPI ranks")]


def _physical_buffer_case(dim=2):
    mesh = (dmesh.create_unit_square(MPI.COMM_WORLD, 8, 8, ghost_mode=dmesh.GhostMode.none)
            if dim == 2 else dmesh.create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, ghost_mode=dmesh.GhostMode.none))
    cells = np.arange(mesh.topology.index_map(dim).size_local, dtype=np.int32)
    points = mesh.geometry.x[dmesh.entities_to_geometry(mesh, dim, cells, False)]
    threshold = points[:, :, 0].max(axis=1) <= 0.25+1e-12
    eligible = np.ones(cells.size, dtype=bool)
    result = CellIndicatorResult(cells, threshold.astype(float), mesh.h(dim, cells), threshold, eligible, threshold.copy())
    return SimpleNamespace(mesh=mesh), result, points


@pytest.mark.parametrize("dim", [2, 3])
def test_physical_padding_is_independent_of_partition_boundaries_and_ghost_cells(dim):
    wrapper, indicator, points = _physical_buffer_case(dim)
    selection = expand_refinement_buffer(wrapper, indicator, 4, target_h=0.08)
    expected = points[:, :, 0].min(axis=1) <= 0.57+1e-12
    np.testing.assert_array_equal(selection.marked_mask, expected)
    assert selection.global_marked_count == MPI.COMM_WORLD.allreduce(int(expected.sum()), op=MPI.SUM)


@pytest.mark.parametrize("stage", ["buffer", "topology"])
def test_rank_dependent_target_sizes_are_rejected_collectively(stage):
    wrapper, indicator, _ = _physical_buffer_case()
    target = 0.1+0.01*MPI.COMM_WORLD.rank
    if stage == "buffer":
        with pytest.raises(ValueError, match="target_h.*identical"):
            expand_refinement_buffer(wrapper, indicator, 1, target_h=target)
    else:
        selection = expand_refinement_buffer(wrapper, indicator, 0, target_h=0.1)
        with pytest.raises(ValueError, match="target_h.*identical"):
            refine_mesh_topology(wrapper.mesh, selection, RefinementConfig(target_h=target))


def test_rank_local_distance_failure_is_reported_on_all_ranks(monkeypatch):
    wrapper, indicator, _ = _physical_buffer_case()
    owns_seeds = bool(np.any(indicator.threshold_mask))
    owners = MPI.COMM_WORLD.allgather(owns_seeds)
    if MPI.COMM_WORLD.rank == owners.index(True):
        def fail_distance(*args):
            raise RuntimeError("injected distance failure")
        monkeypatch.setattr(indicators_module.geometry, "compute_distance_gjk", fail_distance)
    with pytest.raises(ValueError, match="injected distance failure"):
        expand_refinement_buffer(wrapper, indicator, 1, target_h=1.0)


@pytest.mark.parametrize("family", ["CG", "Quadrature"])
def test_parent_evaluation_failure_is_collective_and_preserves_source(refinement_mesh, monkeypatch, family):
    wrapper = refinement_mesh
    field = (wrapper.Add_Field("field", "Field", "CG", 2) if family == "CG"
             else wrapper.Add_Quadrature_Field("field", "Field", ()))
    field.x.array[:] = 3.25
    field.x.scatter_forward()
    original = field.x.array.copy()
    snapshot = snapshot_registered_fields(wrapper)
    cells = wrapper.mesh.topology.index_map(2).size_local
    artifacts = refine_mesh_topology(wrapper.mesh, _selection(wrapper.mesh, np.ones(cells, dtype=bool)))
    if MPI.COMM_WORLD.rank == 1:
        def fail_evaluation(*args, **kwargs):
            raise RuntimeError("synthetic parent evaluation failure")
        monkeypatch.setattr(fem.Function, "eval", fail_evaluation)
    with pytest.raises(ValueError, match="rank 1.*parent evaluation failure"):
        standard = transfer_standard_fields(snapshot, artifacts)
        if family == "Quadrature":
            transfer_quadrature_fields(snapshot, artifacts, standard)
    np.testing.assert_array_equal(field.x.array, original)
    assert wrapper.remesh_epoch == 0


def test_size_aware_refinement_reaches_the_same_target_on_every_rank():
    mesh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4, ghost_mode=dmesh.GhostMode.shared_facet)
    target = 0.25
    cells = np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32)
    selection = _selection(mesh, np.ones(cells.size, dtype=bool))
    artifacts = refine_mesh_topology(mesh, selection, RefinementConfig(target_h=target))
    assert np.all(mesh.h(1, artifacts.marked_edges) > target+1e-12)
    refined = artifacts.refined_mesh
    cells = np.arange(refined.topology.index_map(2).size_local, dtype=np.int32)
    maximum = MPI.COMM_WORLD.allreduce(float(np.max(refined.h(2, cells))), op=MPI.MAX)
    assert maximum <= target+1e-12


def test_refinement_report_uses_the_slowest_rank_for_each_timing(refinement_mesh, capsys):
    refinement_mesh.Add_Field("indicator", "Indicator", "DG", 0)
    refinement_mesh.Add_Field("second", "Second field", "CG", 1)
    refinement_mesh.Add_Quadrature_Field("ip", "Integration points")
    refinement_mesh.Constant_Field("indicator", 1.0)
    config = RefinementConfig(
        enabled=True,
        indicator=FieldThresholdIndicatorConfig("indicator", 0.5),
        target_h=0.4,
        buffer_layers=0,
    )
    result = refinement_mesh.Refine_From_Indicator(config)
    rank = MPI.COMM_WORLD.rank
    diagnostics = dict(result.initial_transfer_diagnostics)
    diagnostics["indicator"] = replace(diagnostics["indicator"], space_creation_seconds=0.1+0.2*rank)
    diagnostics["second"] = replace(diagnostics["second"], space_creation_seconds=0.3-0.2*rank)
    diagnostics["indicator"] = replace(diagnostics["indicator"], copy_seconds=0.2+0.1*rank)
    diagnostics["second"] = replace(diagnostics["second"], copy_seconds=0.3-0.1*rank)
    diagnostics["ip"] = replace(diagnostics["ip"], reconstruction_seconds=0.2+0.5*rank, search_seconds=0.8-0.3*rank)
    result = replace(
        result,
        preparation_seconds=0.2+0.3*rank,
        quadrature_transfer_seconds=0.8-0.3*rank,
        total_seconds=2.0+rank,
        initial_transfer_diagnostics=diagnostics,
    )
    params = refinement_mesh.params
    params.start_time = 0.0
    params.Solution_Steps = {"indicator": 0}
    params.MeshRefinement = config
    solver = TimeSolver(params, SimpleNamespace(mesh=refinement_mesh))

    solver._report_successful_refinement(result, refresh_seconds=0.1+0.1*rank)

    output = capsys.readouterr().out
    if rank == 0:
        assert "total 3.000 s" in output
        assert "preparation 0.500 s" in output
        assert "quadrature transfer 0.800 s" in output
        assert "refresh 0.200 s" in output
        standard_line = next(line for line in output.splitlines() if "Initial CG/DG transfer:" in line)
        quadrature_line = next(line for line in output.splitlines() if "Initial quadrature transfer:" in line)
        assert "copy 0.500 s" in standard_line
        assert "spaces 0.400 s" in standard_line
        assert "reconstruction 0.700 s" in quadrature_line
        assert "search/setup 0.800 s" in quadrature_line
    else:
        assert "Refinement timings" not in output
    assert result.total_seconds == 2.0+rank


def test_rank_without_cells_has_an_empty_indicator():
    def assign_cells_to_root(comm, parts, tdim, cells):
        cell_count = sum(cell.size for cell in cells) // 2
        return adjacencylist(np.zeros((cell_count, 1), dtype=np.int32))._cpp_object

    mesh = dmesh.create_unit_interval(MPI.COMM_WORLD, 1, partitioner=assign_cells_to_root)
    field = fem.Function(fem.functionspace(mesh, ("DG", 0)))
    field.x.array[:] = 1.0
    wrapper = SimpleNamespace(mesh=mesh, fields={"indicator": field}, field_definitions={
        "indicator": FieldDefinition("indicator", "Indicator", "DG", 0, ()),
    })
    result = evaluate_field_indicator(wrapper, FieldThresholdIndicatorConfig("indicator", 0.5), 0.1)
    owned = mesh.topology.index_map(mesh.topology.dim).size_local
    assert result.cells.size == owned
    assert MPI.COMM_WORLD.allreduce(int(owned == 0), op=MPI.SUM) == 1
    assert result.marked_cells.size == owned
    selection = expand_refinement_buffer(wrapper, result, 1, target_h=0.1)
    assert selection.global_marked_count == 1


@pytest.mark.parametrize("case", ["layers", "ordering", "global_count"])
def test_invalid_one_rank_selection_is_reported_everywhere(refinement_mesh, case):
    mesh = refinement_mesh.mesh
    owned = mesh.topology.index_map(mesh.topology.dim).size_local
    selection = _selection(mesh, np.ones(owned, dtype=bool))
    if case == "global_count":
        if mesh.comm.rank == 0:
            selection = replace(selection, global_marked_count=selection.global_marked_count+1)
        with pytest.raises(ValueError, match="global_marked_count"):
            refine_mesh_topology(mesh, selection)
    elif case == "ordering":
        if mesh.comm.rank == 0:
            selection = replace(selection, cells=selection.cells[::-1])
        with pytest.raises(ValueError, match="rank 0.*ordering"):
            refine_mesh_topology(mesh, selection)
    else:
        mask = np.ones(owned, dtype=bool)
        result = CellIndicatorResult(selection.cells, np.ones(owned), mesh.h(mesh.topology.dim, selection.cells),
                                     mask, mask.copy(), mask.copy())
        with pytest.raises(ValueError, match="identical on every rank"):
            expand_refinement_buffer(refinement_mesh, result, mesh.comm.rank, target_h=0.1)


def test_snapshot_rejects_rank_inconsistent_field_metadata(refinement_mesh):
    refinement_mesh.Add_Field("temperature", f"Temperature rank {MPI.COMM_WORLD.rank}", "CG", 1)
    with pytest.raises(ValueError, match="identical on every mesh rank"):
        snapshot_registered_fields(refinement_mesh)


def test_transfer_rejects_rank_inconsistent_padding(refinement_mesh):
    refinement_mesh.Add_Field("temperature", "Temperature", "CG", 1)
    artifacts, _ = _successful_detached_state(refinement_mesh)
    snapshot = snapshot_registered_fields(refinement_mesh)
    config = RefinementConfig(interpolation_padding=1e-10*(MPI.COMM_WORLD.rank+1))
    with pytest.raises(ValueError, match="interpolation_padding.*identical"):
        transfer_standard_fields(snapshot, artifacts, config)
    standard = transfer_standard_fields(snapshot, artifacts)
    with pytest.raises(ValueError, match="interpolation_padding.*identical"):
        transfer_quadrature_fields(snapshot, artifacts, standard, config)


def test_marker_transfer_rejects_rank_inconsistent_aliases(refinement_mesh):
    artifacts, _ = _successful_detached_state(refinement_mesh)
    source = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    if MPI.COMM_WORLD.rank == 0:
        source = replace(source, boundary_markers={**source.boundary_markers,
                                                  "extra_alias": refinement_mesh.get_boundary_tag("bottom")})
    with pytest.raises(ValueError, match="Marker name mappings must be identical"):
        transfer_mesh_markers(source, artifacts)


def test_install_rejects_rank_inconsistent_controls_before_mutation(refinement_mesh):
    artifacts, fields = _successful_detached_state(refinement_mesh)
    old_mesh = refinement_mesh.mesh
    config = RefinementConfig(replot_on_refinement=MPI.COMM_WORLD.rank == 0)
    with pytest.raises(ValueError, match="configuration must be identical"):
        refinement_mesh.Install_Refined_State(artifacts, fields, config)
    assert refinement_mesh.mesh is old_mesh and refinement_mesh.remesh_epoch == 0


def test_one_rank_install_failure_rolls_back_all_ranks(refinement_mesh, monkeypatch):
    refinement_mesh.Add_Field("temperature", "Temperature", "CG", 1)
    refinement_mesh.Constant_Field("temperature", 12.0)
    artifacts, fields = _successful_detached_state(refinement_mesh)
    old_mesh, old_fields = refinement_mesh.mesh, refinement_mesh.fields
    original = manager_module._install_local_state

    def install_then_fail_on_root(*args):
        original(*args)
        if MPI.COMM_WORLD.rank == 0:
            raise RuntimeError("synthetic rank-local installation failure")

    monkeypatch.setattr(manager_module, "_install_local_state", install_then_fail_on_root)
    with pytest.raises(RuntimeError, match="rank 0.*rank-local installation failure"):
        refinement_mesh.Install_Refined_State(artifacts, fields)
    assert refinement_mesh.mesh is old_mesh and refinement_mesh.fields is old_fields
    assert refinement_mesh.remesh_epoch == 0
    np.testing.assert_allclose(refinement_mesh.Get_Field("temperature").x.array, 12.0)


def test_install_occurrence_and_metadata_must_agree_on_all_ranks(refinement_mesh):
    artifacts, fields = _successful_detached_state(refinement_mesh)
    owned = refinement_mesh.mesh.topology.index_map(refinement_mesh.mesh.topology.dim).size_local
    no_op = refine_mesh_topology(refinement_mesh.mesh, _selection(refinement_mesh.mesh, np.zeros(owned, dtype=bool)))
    attempt = artifacts if MPI.COMM_WORLD.rank == 0 else no_op
    with pytest.raises(ValueError, match="occurrence must be identical"):
        refinement_mesh.Install_Refined_State(attempt, fields)

    if MPI.COMM_WORLD.rank == 0:
        artifacts = replace(artifacts, boundary_markers={**artifacts.boundary_markers,
                                                        "alias": artifacts.get_boundary_tag("bottom")})
    with pytest.raises(ValueError, match="metadata must be identical"):
        refinement_mesh.Install_Refined_State(artifacts, fields)
    assert refinement_mesh.remesh_epoch == 0


def test_mesh_transactions_require_congruent_communicators(refinement_mesh):
    artifacts, state = _successful_detached_state(refinement_mesh)
    target = dmesh.create_unit_square(MPI.COMM_SELF, 1, 1)
    invalid_artifacts = replace(artifacts, refined_mesh=target, cell_tags=None, facet_tags=None,
                                domain_markers={}, boundary_markers={})
    state = replace(state, target_mesh=target)
    snapshot = snapshot_registered_fields(refinement_mesh)
    with pytest.raises(ValueError, match="congruent communicators"):
        transfer_standard_fields(snapshot, invalid_artifacts)
    with pytest.raises(ValueError, match="congruent communicators"):
        transfer_quadrature_fields(snapshot, invalid_artifacts, state)
    with pytest.raises(ValueError, match="congruent communicators"):
        transfer_mesh_markers(MeshMarkerState.from_mesh_wrapper(refinement_mesh), invalid_artifacts)
    with pytest.raises(ValueError, match="congruent communicators"):
        redistribute_mesh_markers(MeshMarkerState.from_mesh_wrapper(refinement_mesh), invalid_artifacts)
    with pytest.raises(ValueError, match="congruent communicators"):
        refinement_mesh.Install_Refined_State(invalid_artifacts, state)


def _imbalanced_detached_state(wrapper):
    field = wrapper.Add_Field("indicator", "One-sided indicator", "DG", 0)
    field.x.array[:] = float(MPI.COMM_WORLD.rank == 0)
    field.x.scatter_forward()
    indicator = evaluate_field_indicator(wrapper, FieldThresholdIndicatorConfig("indicator", 0.5), 0.1)
    selection = expand_refinement_buffer(wrapper, indicator, 0, target_h=0.1)
    artifacts = refine_mesh_topology(wrapper.mesh, selection)
    artifacts = transfer_mesh_markers(MeshMarkerState.from_mesh_wrapper(wrapper), artifacts)
    state = transfer_standard_fields(snapshot_registered_fields(wrapper), artifacts)
    assert artifacts.report.imbalance_ratio_after > 1.0
    return artifacts, state, RefinementConfig(repartition=True, repartition_imbalance_threshold=1.0)


@pytest.mark.parametrize("statistic, message", [
    ("global_cells_after", "global cell count"),
    ("new_min_h", "minimum cell diameter"),
    ("new_max_h", "maximum cell diameter"),
])
def test_repartition_rejects_inconsistent_geometry_report(refinement_mesh, statistic, message):
    artifacts, fields, config = _imbalanced_detached_state(refinement_mesh)
    invalid = replace(artifacts, report=replace(artifacts.report, **{
        statistic: getattr(artifacts.report, statistic)+1,
    }))
    old_mesh, old_fields = refinement_mesh.mesh, refinement_mesh.fields
    with pytest.raises(ValueError, match=message):
        repartition_mesh(invalid, fields, config)
    assert refinement_mesh.mesh is old_mesh and refinement_mesh.fields is old_fields


@pytest.mark.parametrize("stage", ["graph", "routing", "tags"])
def test_repartition_backend_failure_is_collective_and_detached(refinement_mesh, monkeypatch, stage):
    artifacts, fields, config = _imbalanced_detached_state(refinement_mesh)
    old_mesh, old_fields = refinement_mesh.mesh, refinement_mesh.fields

    def fail_backend(*args, **kwargs):
        raise RuntimeError("synthetic redistribution backend failure")

    if stage == "graph":
        monkeypatch.setattr(partition_module.dmesh, "refine", fail_backend)
    elif stage == "routing":
        monkeypatch.setattr(markers_module.io, "distribute_entity_data", fail_backend)
    else:
        monkeypatch.setattr(markers_module.dmesh, "meshtags_from_entities", fail_backend)
    with pytest.raises(ValueError, match="rank 0.*backend failure"):
        repartition_mesh(artifacts, fields, config)
    assert refinement_mesh.mesh is old_mesh and refinement_mesh.fields is old_fields
    assert refinement_mesh.remesh_epoch == 0


def test_repartition_preserves_marker_free_meshes(refinement_mesh):
    artifacts, fields, config = _imbalanced_detached_state(refinement_mesh)
    artifacts = replace(artifacts, cell_tags=None, facet_tags=None, domain_markers={}, boundary_markers={})
    result = repartition_mesh(artifacts, fields, config)
    assert result.repartitioned
    assert result.artifacts.cell_tags is None and result.artifacts.facet_tags is None
    assert result.artifacts.domain_names() == [] and result.artifacts.boundary_names() == []
    diagnostic = result.transferred_state.diagnostics["indicator"]
    assert diagnostic.global_destination_integral == pytest.approx(diagnostic.global_source_integral)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("quadrature_degree", [3, 5])
def test_exact_migration_preserves_arbitrary_samples_across_ranks(refinement_rectangle_params, dim, quadrature_degree):
    wrapper, artifacts, state = _migration_case(refinement_rectangle_params, dim, quadrature_degree)
    old_mesh = wrapper.mesh
    for _ in range(2):
        artifacts, state = _check_exact_migration(artifacts, state)
    assert wrapper.mesh is old_mesh
    assert wrapper.remesh_epoch == 0


def test_exact_migration_handles_ranks_gaining_or_losing_all_owned_cells(refinement_rectangle_params):
    _, artifacts, state = _migration_case(refinement_rectangle_params, 2, 3)

    def collect_on_root(comm, nparts, cell_types, cells):
        count = sum(len(rows)//3 for rows in cells)
        return adjacencylist(np.zeros((count, 1), dtype=np.int32))._cpp_object

    target_data = dmesh.refine(
        state.target_mesh,
        edges=np.empty(0, dtype=np.int32),
        partitioner=collect_on_root,
        option=dmesh.RefinementOption.parent_cell_and_facet,
    )
    artifacts, state = _check_exact_migration(artifacts, state, target_data)
    owned = state.target_mesh.topology.index_map(2).size_local
    assert (owned == 0) == (MPI.COMM_WORLD.rank != 0)
    _check_exact_migration(artifacts, state)


def test_repartition_rank_matching_retains_the_maximum_possible_cells(refinement_mesh, monkeypatch):
    artifacts, state, config = _imbalanced_detached_state(refinement_mesh)
    source = artifacts.refined_mesh
    source_range = source.topology.index_map(source.topology.dim).local_range
    create_partitioner = partition_module.dmesh.create_cell_partitioner
    overlap = []

    def record_partitioner(*args, **kwargs):
        native = create_partitioner(*args, **kwargs)

        def partition(comm, nparts, cell_types, cells):
            destinations = native(comm, nparts, cell_types, cells)
            owners = destinations.array[destinations.offsets[:-1]]
            overlap.append(np.asarray(comm.allgather(np.bincount(owners, minlength=nparts))))
            return destinations

        return partition

    monkeypatch.setattr(partition_module.dmesh, "create_cell_partitioner", record_partitioner)
    result = repartition_mesh(artifacts, state, config)

    target = result.artifacts.refined_mesh
    ids = target.topology.original_cell_index[:target.topology.index_map(target.topology.dim).size_local]
    retained = MPI.COMM_WORLD.allreduce(int(np.count_nonzero((ids >= source_range[0]) & (ids < source_range[1]))), op=MPI.SUM)
    matrix = overlap[0]
    assert retained == max(matrix[0, 0]+matrix[1, 1], matrix[0, 1]+matrix[1, 0])
    assert result.artifacts.report.imbalance_ratio_after < artifacts.report.imbalance_ratio_after


def test_nonfinite_migration_input_is_rejected_on_every_rank(refinement_rectangle_params):
    _, artifacts, state = _migration_case(refinement_rectangle_params, 2, 3)
    target, parents, facets = _rotate_cell_ownership(state.target_mesh)
    target_artifacts = replace(artifacts, refined_mesh=target, cell_tags=None, facet_tags=None,
                               migration_data=partition_module.PartitionMigrationData(parents, facets))
    if MPI.COMM_WORLD.rank == 0:
        state.fields["samples"].x.array[0] = np.nan
    with pytest.raises(ValueError, match="samples.*non-finite"):
        migrate_repartitioned_fields(state, target_artifacts)


def test_repartition_rejects_tags_outside_owned_entities(refinement_mesh):
    artifacts, fields, config = _imbalanced_detached_state(refinement_mesh)
    topology = artifacts.refined_mesh.topology
    first_unowned = topology.index_map(topology.dim).size_local
    tag = artifacts.get_domain_tag(artifacts.domain_names()[0])
    tags = dmesh.meshtags(artifacts.refined_mesh, topology.dim,
                         np.array([first_unowned], dtype=np.int32), np.array([tag], dtype=np.int32))
    with pytest.raises(ValueError, match="packing failed.*owned entities only"):
        repartition_mesh(replace(artifacts, cell_tags=tags), fields, config)
    assert refinement_mesh.remesh_epoch == 0
