"""MPI baseline and integration tests for adaptive mesh refinement."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def _mpi_python_invocation(mpirun, arguments):
    """Include MPI child measurements when the parent is running coverage."""
    environment = os.environ.copy()
    python_command = [sys.executable]
    coverage_module = sys.modules.get("coverage")
    active = coverage_module.Coverage.current() if coverage_module is not None else None
    if active is not None:
        python_command += ["-m", "coverage", "run", "--parallel-mode"]
        if active.config.branch:
            python_command.append("--branch")
        if active.config.source:
            python_command.append("--source=" + ",".join(active.config.source))
        environment["COVERAGE_FILE"] = active.config.data_file
    return [mpirun, "-n", "2", *python_command, *arguments], environment


def _run_shared_facet_baseline() -> None:
    """Run the shared-facet checks inside a two-rank subprocess."""

    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.utils.mpi_utils import comm, size

    if size != 2:
        raise RuntimeError("The refinement MPI baseline requires exactly two ranks.")

    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 2.0
        Ly = 1.0
        dx = 0.5
        dim = 2

    mesh = Mesh(Params())
    topology = mesh.mesh.topology
    tdim = topology.dim
    fdim = tdim - 1
    cell_map = topology.index_map(tdim)

    ghost_count = cell_map.num_ghosts
    assert comm.allreduce(ghost_count, op=MPI.SUM) > 0
    assert cell_map.size_global == comm.bcast(cell_map.size_global, root=0)

    topology.create_connectivity(tdim, fdim)
    topology.create_connectivity(fdim, tdim)
    cell_to_facet = topology.connectivity(tdim, fdim)
    facet_to_cell = topology.connectivity(fdim, tdim)
    owned_cells = cell_map.size_local

    local_interface_ghosts = 0
    for ghost_cell in range(owned_cells, owned_cells + ghost_count):
        has_owned_neighbor = any(
            any(adjacent_cell < owned_cells for adjacent_cell in facet_to_cell.links(facet))
            for facet in cell_to_facet.links(ghost_cell)
        )
        local_interface_ghosts += int(has_owned_neighbor)

    assert comm.allreduce(local_interface_ghosts, op=MPI.SUM) > 0

    local_area = fem.assemble_scalar(fem.form(fem.Constant(mesh.mesh, 1.0) * ufl.dx))
    assert comm.allreduce(local_area, op=MPI.SUM) == pytest.approx(2.0)
    assert set(mesh.boundary_names()) == {"bottom", "right", "top", "left"}
    for name in mesh.boundary_names():
        assert comm.allreduce(mesh.get_boundary_facets(name).size, op=MPI.SUM) > 0


def _run_buffer_expansion_checks() -> None:
    """Run cross-partition buffering and collective decision checks."""

    import numpy as np
    from mpi4py import MPI

    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.mesh.refinement import CellIndicatorResult, expand_refinement_buffer
    from femcubes.core.utils.mpi_utils import comm, rank, size

    if size != 2:
        raise RuntimeError("The refinement buffer checks require exactly two ranks.")

    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 1.0
        dx = 0.35
        dim = 2

    mesh = Mesh(Params())
    topology = mesh.mesh.topology
    tdim = topology.dim
    fdim = tdim - 1
    cell_map = topology.index_map(tdim)
    owned_count = cell_map.size_local
    topology.create_connectivity(tdim, fdim)
    topology.create_connectivity(fdim, tdim)
    cell_to_facet = topology.connectivity(tdim, fdim)
    facet_to_cell = topology.connectivity(fdim, tdim)

    interface_owned = []
    for cell in range(owned_count):
        if any(
            np.any(facet_to_cell.links(facet) >= owned_count)
            for facet in cell_to_facet.links(cell)
        ):
            interface_owned.append(cell)
    assert interface_owned

    initial = np.zeros(owned_count, dtype=bool)
    if rank == 1:
        initial[interface_owned[0]] = True
    cells = np.arange(owned_count, dtype=np.int32)
    result = CellIndicatorResult(
        cells=cells,
        values=initial.astype(float),
        diameters=mesh.mesh.h(tdim, cells),
        threshold_mask=initial.copy(),
        size_mask=np.ones(owned_count, dtype=bool),
        marked_mask=initial.copy(),
    )

    zero = expand_refinement_buffer(mesh, result, buffer_layers=0, target_h=0.1)
    one = expand_refinement_buffer(mesh, result, buffer_layers=1, target_h=0.1)
    two = expand_refinement_buffer(mesh, result, buffer_layers=2, target_h=0.1)
    two_repeated = expand_refinement_buffer(mesh, result, buffer_layers=2, target_h=0.1)

    moving_size_mask = ~initial
    moving_result = CellIndicatorResult(
        cells=cells,
        values=initial.astype(float),
        diameters=result.diameters.copy(),
        threshold_mask=initial.copy(),
        size_mask=moving_size_mask,
        marked_mask=initial & moving_size_mask,
    )
    moving = expand_refinement_buffer(mesh, moving_result, buffer_layers=1, target_h=0.1)

    assert zero.global_initial_count == 1
    assert zero.global_marked_count == 1
    assert zero.any_marked
    assert zero.local_marked_count == (1 if rank == 1 else 0)

    assert one.global_marked_count > zero.global_marked_count
    if rank == 0:
        assert one.local_marked_count > 0
    assert two.global_marked_count >= one.global_marked_count
    assert two.global_marked_count == two_repeated.global_marked_count
    np.testing.assert_array_equal(two.marked_mask, two_repeated.marked_mask)

    assert moving.global_initial_count == 0
    assert moving.global_marked_count > 0
    assert moving.any_marked
    if rank == 0:
        assert moving.local_marked_count > 0
    else:
        assert not moving.marked_mask[interface_owned[0]]

    for selection in (zero, one, two):
        gathered_counts = comm.allgather(selection.global_marked_count)
        assert len(set(gathered_counts)) == 1
        assert selection.global_marked_count == comm.allreduce(
            selection.local_marked_count,
            op=MPI.SUM,
        )
        assert selection.any_marked is True

    try:
        expand_refinement_buffer(mesh, result, buffer_layers=rank, target_h=0.1)
    except ValueError as exc:
        mismatch_messages = comm.allgather(str(exc))
        assert len(set(mismatch_messages)) == 1
        assert "buffer_layers must be identical on every rank" in str(exc)
    else:
        raise AssertionError("Rank-dependent buffer layer counts were accepted.")

    rank_local_result = result
    if rank == 1:
        rank_local_result = CellIndicatorResult(
            cells=result.cells[:-1],
            values=result.values[:-1],
            diameters=result.diameters[:-1],
            threshold_mask=result.threshold_mask[:-1],
            size_mask=result.size_mask[:-1],
            marked_mask=result.marked_mask[:-1],
        )
    try:
        expand_refinement_buffer(mesh, rank_local_result, buffer_layers=0, target_h=0.1)
    except ValueError as exc:
        invalid_input_messages = comm.allgather(str(exc))
        assert len(set(invalid_input_messages)) == 1
        assert "rank 1: indicator_result cells must use complete owned local cell ordering" in str(exc)
    else:
        raise AssertionError("An indicator input invalid only on rank one was accepted.")


def _run_topology_marker_checks() -> None:
    """Run distributed topology ancestry and marker-transfer checks."""

    import numpy as np
    import ufl
    from dolfinx import fem, mesh as dmesh
    from mpi4py import MPI

    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.mesh.refinement import (
        CellIndicatorResult,
        MeshMarkerState,
        expand_refinement_buffer,
        refine_mesh_topology,
        resolve_owned_child_input_positions,
        transfer_mesh_markers,
    )
    from femcubes.core.utils.mpi_utils import comm, rank, size

    if size != 2:
        raise RuntimeError("The topology/marker checks require exactly two ranks.")

    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 1.0
        dx = 0.35
        dim = 2

    mesh_wrapper = Mesh(Params())
    source = MeshMarkerState.from_mesh_wrapper(mesh_wrapper)
    mesh = source.mesh
    topology = mesh.topology
    tdim = topology.dim
    fdim = tdim - 1
    cell_map = topology.index_map(tdim)
    owned_count = cell_map.size_local
    topology.create_connectivity(tdim, fdim)
    topology.create_connectivity(fdim, tdim)
    cell_to_facet = topology.connectivity(tdim, fdim)
    facet_to_cell = topology.connectivity(fdim, tdim)

    interface_owned = [
        cell
        for cell in range(owned_count)
        if any(
            np.any(facet_to_cell.links(facet) >= owned_count)
            for facet in cell_to_facet.links(cell)
        )
    ]
    assert interface_owned

    initial = np.zeros(owned_count, dtype=bool)
    if rank == 1:
        initial[interface_owned[0]] = True
    cells = np.arange(owned_count, dtype=np.int32)
    indicator = CellIndicatorResult(
        cells=cells,
        values=initial.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=initial.copy(),
        size_mask=np.ones(owned_count, dtype=bool),
        marked_mask=initial.copy(),
    )
    selection = expand_refinement_buffer(mesh_wrapper, indicator, buffer_layers=1, target_h=0.1)
    assert selection.global_initial_count == 1

    topology_artifacts = refine_mesh_topology(mesh, selection)
    assert topology_artifacts.report.occurred
    assert topology_artifacts.report.global_cells_after > cell_map.size_global

    expected_offset = comm.exscan(topology_artifacts.parent_cell.size, op=MPI.SUM)
    if expected_offset is None:
        expected_offset = 0
    assert topology_artifacts.input_cell_global_offset == expected_offset

    positions = resolve_owned_child_input_positions(topology_artifacts)
    assert positions.size == topology_artifacts.report.local_cells_after
    assert np.unique(positions).size == positions.size
    parents = topology_artifacts.parent_cell[positions]
    assert np.all(parents >= 0)
    assert np.all(parents < owned_count)

    refined = topology_artifacts.refined_mesh
    refined_cell_map = refined.topology.index_map(tdim)
    assert comm.allreduce(refined_cell_map.num_ghosts, op=MPI.SUM) > 0

    transferred = transfer_mesh_markers(source, topology_artifacts)
    assert transferred.domain_names() == mesh_wrapper.domain_names()
    assert transferred.boundary_names() == mesh_wrapper.boundary_names()
    assert np.all(transferred.cell_tags.indices < refined_cell_map.size_local)
    global_tagged_cells = comm.allreduce(transferred.cell_tags.indices.size, op=MPI.SUM)
    assert global_tagged_cells == refined_cell_map.size_global

    refined.topology.create_connectivity(fdim, tdim)
    exterior_facets = dmesh.exterior_facet_indices(refined.topology)
    refined_facet_map = refined.topology.index_map(fdim)
    assert np.all(transferred.facet_tags.indices < refined_facet_map.size_local)
    global_exterior_facets = comm.allreduce(exterior_facets.size, op=MPI.SUM)
    global_tagged_facets = comm.allreduce(transferred.facet_tags.indices.size, op=MPI.SUM)
    assert global_tagged_facets == global_exterior_facets

    def tagged_measure(target_mesh, tags, tag, measure_name):
        measure = ufl.Measure(measure_name, domain=target_mesh, subdomain_data=tags)
        local_value = fem.assemble_scalar(
            fem.form(fem.Constant(target_mesh, 1.0) * measure(tag))
        )
        return comm.allreduce(local_value, op=MPI.SUM)

    domain_tag = transferred.get_domain_tag("domain")
    source_area = tagged_measure(mesh, source.cell_tags, domain_tag, "dx")
    refined_area = tagged_measure(refined, transferred.cell_tags, domain_tag, "dx")
    assert source_area == pytest.approx(4.0)
    assert refined_area == pytest.approx(source_area)

    expected_boundary_measures = {
        "bottom": 4.0,
        "right": 1.0,
        "top": 4.0,
        "left": 1.0,
    }
    for name, expected_measure in expected_boundary_measures.items():
        assert comm.allreduce(transferred.get_boundary_facets(name).size, op=MPI.SUM) > 0
        boundary_measure = tagged_measure(
            refined,
            transferred.facet_tags,
            transferred.get_boundary_tag(name),
            "ds",
        )
        assert boundary_measure == pytest.approx(expected_measure)


def _run_field_transfer_checks() -> None:
    """Run two-rank manufactured CG/DG nonmatching interpolation checks."""

    import numpy as np
    from dolfinx import fem
    from mpi4py import MPI

    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.mesh.refinement import (
        CellIndicatorResult,
        expand_refinement_buffer,
        refine_mesh_topology,
        snapshot_registered_fields,
        transfer_standard_fields,
    )
    from femcubes.core.utils.mpi_utils import comm, rank, size

    if size != 2:
        raise RuntimeError("The field-transfer checks require exactly two ranks.")

    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 1.0
        dx = 0.35
        dim = 2
        Quadrature_Degree = 3

    mesh_wrapper = Mesh(Params())
    scalar_expression = lambda x: 1.0 + x[0] ** 2 - 0.5 * x[0] * x[1]
    old_expression = lambda x: -2.0 + 0.25 * x[0] + x[1]
    vector_expression = lambda x: np.vstack(
        (0.5 + x[0] - 2.0 * x[1], -x[0] + 0.75 * x[1])
    )
    scalar = mesh_wrapper.Add_Field("phase", "Phase", "CG", 2)
    old = mesh_wrapper.Add_Field("phase_old", "Old phase", "CG", 1)
    vector = mesh_wrapper.Add_Vector_Field("driving", "Driving", "DG", 1, 2)
    mesh_wrapper.Add_Quadrature_Field("pending_ip", "Pending IP")
    scalar.interpolate(scalar_expression)
    old.interpolate(old_expression)
    vector.interpolate(vector_expression)
    for field in (scalar, old, vector):
        field.x.scatter_forward()
    source_values = {
        name: field.x.array.copy()
        for name, field in mesh_wrapper.fields.items()
        if name != "pending_ip"
    }
    mesh_wrapper.Add_TestFunction("phase", step=0)
    mesh_wrapper.Add_TestFunction("phase_old", step=1)
    snapshot = snapshot_registered_fields(mesh_wrapper)

    mesh = mesh_wrapper.mesh
    topology = mesh.topology
    tdim = topology.dim
    fdim = tdim - 1
    cell_map = topology.index_map(tdim)
    owned_count = cell_map.size_local
    topology.create_connectivity(tdim, fdim)
    topology.create_connectivity(fdim, tdim)
    cell_to_facet = topology.connectivity(tdim, fdim)
    facet_to_cell = topology.connectivity(fdim, tdim)
    interface_owned = [
        cell
        for cell in range(owned_count)
        if any(
            np.any(facet_to_cell.links(facet) >= owned_count)
            for facet in cell_to_facet.links(cell)
        )
    ]
    assert interface_owned

    initial = np.zeros(owned_count, dtype=bool)
    if rank == 1:
        initial[interface_owned[0]] = True
    cells = np.arange(owned_count, dtype=np.int32)
    indicator = CellIndicatorResult(
        cells=cells,
        values=initial.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=initial.copy(),
        size_mask=np.ones(owned_count, dtype=bool),
        marked_mask=initial.copy(),
    )
    selection = expand_refinement_buffer(mesh_wrapper, indicator, buffer_layers=1, target_h=0.1)
    artifacts = refine_mesh_topology(mesh, selection)
    transferred = transfer_standard_fields(snapshot, artifacts)

    def assert_exact(name, expression):
        target = transferred.fields[name]
        expected = fem.Function(target.function_space)
        expected.interpolate(expression)
        expected.x.scatter_forward()
        assert np.allclose(target.x.array, expected.x.array, rtol=2.0e-11, atol=2.0e-11)

    assert_exact("phase", scalar_expression)
    assert_exact("phase_old", old_expression)
    assert_exact("driving", vector_expression)
    assert transferred.pending_field_names == ("pending_ip",)
    assert transferred.step_fields == {0: ("phase",), 1: ("phase_old",)}
    for name, expected_values in source_values.items():
        np.testing.assert_array_equal(mesh_wrapper.fields[name].x.array, expected_values)
    for diagnostic in transferred.diagnostics.values():
        assert diagnostic.finite
        assert diagnostic.global_destination_dofs > diagnostic.global_source_dofs
        assert comm.allreduce(int(diagnostic.finite), op=MPI.SUM) == size


def _run_quadrature_transfer_checks() -> None:
    """Run two-rank scalar/tensor quadrature reconstruction checks."""

    import numpy as np
    from dolfinx import fem
    from mpi4py import MPI

    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.mesh.refinement import (
        CellIndicatorResult,
        expand_refinement_buffer,
        refine_mesh_topology,
        snapshot_registered_fields,
        transfer_quadrature_fields,
        transfer_standard_fields,
    )
    from femcubes.core.utils.mpi_utils import comm, rank, size

    if size != 2:
        raise RuntimeError("The quadrature-transfer checks require exactly two ranks.")

    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 1.0
        dx = 0.35
        dim = 2
        Quadrature_Degree = 3

    mesh_wrapper = Mesh(Params())
    nodal = mesh_wrapper.Add_Field("nodal", "Nodal", "CG", 1)
    scalar_ip = mesh_wrapper.Add_Quadrature_Field("scalar_ip", "Scalar IP")
    tensor_ip = mesh_wrapper.Add_Quadrature_Field("tensor_ip", "Tensor IP", (2, 1))
    nodal_expression = lambda x: 0.5 + x[0] - x[1]
    scalar_expression = lambda x: 2.0 - 0.25 * x[0] + 1.5 * x[1]
    nodal.interpolate(nodal_expression)
    scalar_ip.interpolate(scalar_expression)
    nodal.x.scatter_forward()
    scalar_ip.x.scatter_forward()
    tensor_values = np.array([3.25, -1.75])
    tensor_ip.x.array.reshape(-1, 2)[:] = tensor_values[None, :]
    tensor_ip.x.scatter_forward()
    source_values = {
        name: field.x.array.copy()
        for name, field in mesh_wrapper.fields.items()
    }
    snapshot = snapshot_registered_fields(mesh_wrapper)

    mesh = mesh_wrapper.mesh
    topology = mesh.topology
    tdim = topology.dim
    fdim = tdim - 1
    cell_map = topology.index_map(tdim)
    owned_count = cell_map.size_local
    topology.create_connectivity(tdim, fdim)
    topology.create_connectivity(fdim, tdim)
    cell_to_facet = topology.connectivity(tdim, fdim)
    facet_to_cell = topology.connectivity(fdim, tdim)
    interface_owned = [
        cell
        for cell in range(owned_count)
        if any(
            np.any(facet_to_cell.links(facet) >= owned_count)
            for facet in cell_to_facet.links(cell)
        )
    ]
    assert interface_owned

    initial = np.zeros(owned_count, dtype=bool)
    if rank == 1:
        initial[interface_owned[0]] = True
    cells = np.arange(owned_count, dtype=np.int32)
    indicator = CellIndicatorResult(
        cells=cells,
        values=initial.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=initial.copy(),
        size_mask=np.ones(owned_count, dtype=bool),
        marked_mask=initial.copy(),
    )
    selection = expand_refinement_buffer(mesh_wrapper, indicator, buffer_layers=1, target_h=0.1)
    artifacts = refine_mesh_topology(mesh, selection)
    standard = transfer_standard_fields(snapshot, artifacts)
    complete = transfer_quadrature_fields(snapshot, artifacts, standard)

    expected_nodal = fem.Function(complete.fields["nodal"].function_space)
    expected_nodal.interpolate(nodal_expression)
    expected_nodal.x.scatter_forward()
    expected_scalar = fem.Function(complete.fields["scalar_ip"].function_space)
    expected_scalar.interpolate(scalar_expression)
    expected_scalar.x.scatter_forward()
    assert np.allclose(
        complete.fields["nodal"].x.array,
        expected_nodal.x.array,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert np.allclose(
        complete.fields["scalar_ip"].x.array,
        expected_scalar.x.array,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    tensor_target = complete.fields["tensor_ip"].x.array.reshape(-1, 2)
    assert np.allclose(tensor_target, tensor_values[None, :], rtol=2.0e-12, atol=2.0e-12)
    assert standard.pending_field_names == ("scalar_ip", "tensor_ip")
    assert complete.pending_field_names == ()
    assert list(complete.fields) == list(snapshot.fields)
    for name, expected_values in source_values.items():
        np.testing.assert_array_equal(mesh_wrapper.fields[name].x.array, expected_values)
    for name in ("scalar_ip", "tensor_ip"):
        diagnostic = complete.diagnostics[name]
        assert diagnostic.finite
        assert diagnostic.reconstruction_degree == 1
        assert diagnostic.global_destination_dofs > diagnostic.global_source_dofs
        assert comm.allreduce(int(diagnostic.finite), op=MPI.SUM) == size

    import femcubes.core.mesh.refinement.transfer as transfer_module

    original_populate = transfer_module._populate_reconstructed_dg_field
    if rank == 1:
        def fail_rank_one_reconstruction(*args, **kwargs):
            raise RuntimeError("synthetic local reconstruction failure")

        transfer_module._populate_reconstructed_dg_field = fail_rank_one_reconstruction
    try:
        transfer_quadrature_fields(snapshot, artifacts, standard)
    except ValueError as exc:
        messages = comm.allgather(str(exc))
        assert len(set(messages)) == 1
        assert "quadrature reconstruction for field 'scalar_ip'" in str(exc)
        assert "rank 1: RuntimeError: synthetic local reconstruction failure" in str(exc)
    else:
        raise AssertionError("A rank-local quadrature reconstruction failure was accepted.")
    finally:
        transfer_module._populate_reconstructed_dg_field = original_populate


def _run_live_install_checks() -> None:
    """Run two-rank live installation and collective visualization checks."""

    from types import SimpleNamespace

    import numpy as np
    from mpi4py import MPI

    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.mesh.refinement import (
        CellIndicatorResult,
        MeshMarkerState,
        RefinementConfig,
        expand_refinement_buffer,
        refine_mesh_topology,
        snapshot_registered_fields,
        transfer_mesh_markers,
        transfer_standard_fields,
    )
    from femcubes.core.utils.mpi_utils import comm, rank, size

    if size != 2:
        raise RuntimeError("The live-install checks require exactly two ranks.")

    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 1.0
        dx = 0.35
        dim = 2

    mesh_wrapper = Mesh(Params())
    mesh_wrapper.Add_Field("p", "Pressure", "CG", 1)
    mesh_wrapper.Add_Vector_Field("u", "Displacement", "CG", 1, 2)
    mesh_wrapper.Add_TestFunction("p", step=0)
    mesh_wrapper.Add_Vector_TestFunction("u", nDim=2, step=0)
    mesh_wrapper.finalize_test_functions()
    mesh_wrapper.Constant_Field("p", 2.5)
    mesh_wrapper.Constant_Field("u", -1.25)
    marker_state = MeshMarkerState.from_mesh_wrapper(mesh_wrapper)
    snapshot = snapshot_registered_fields(mesh_wrapper)
    old_mesh = mesh_wrapper.mesh

    tdim = old_mesh.topology.dim
    owned_count = old_mesh.topology.index_map(tdim).size_local
    cells = np.arange(owned_count, dtype=np.int32)
    initial = np.zeros(owned_count, dtype=bool)
    if rank == 1:
        initial[0] = True
    indicator = CellIndicatorResult(
        cells=cells,
        values=initial.astype(float),
        diameters=old_mesh.h(tdim, cells),
        threshold_mask=initial.copy(),
        size_mask=np.ones(owned_count, dtype=bool),
        marked_mask=initial.copy(),
    )
    selection = expand_refinement_buffer(mesh_wrapper, indicator, buffer_layers=1, target_h=0.1)
    artifacts = refine_mesh_topology(old_mesh, selection)
    artifacts = transfer_mesh_markers(marker_state, artifacts)
    transferred = transfer_standard_fields(snapshot, artifacts)

    plot_calls = []

    def track_collective_plot(mesh=None, **kwargs):
        plot_calls.append((mesh, kwargs))
        return SimpleNamespace(
            visible_updated=kwargs["show_window"],
            movie_frame_written=kwargs["movie_path"] is not None,
        )

    mesh_wrapper.plot_mesh = track_collective_plot
    if rank == 0:
        mesh_wrapper._mesh_plotter = object()
        mesh_wrapper._mesh_plot_visible = True
    result = mesh_wrapper.Install_Refined_State(
        artifacts,
        transferred,
        RefinementConfig(
            replot_on_refinement=False,
            save_remesh_animations=True,
        ),
    )

    assert result.installed
    assert mesh_wrapper.mesh is artifacts.refined_mesh
    assert mesh_wrapper.mesh is not old_mesh
    assert mesh_wrapper.remesh_epoch == 1
    assert mesh_wrapper.last_refinement_report is artifacts.report
    assert comm.allreduce(mesh_wrapper.remesh_epoch, op=MPI.SUM) == size
    assert comm.allgather(len(plot_calls)) == [2, 2]


    assert "Mesh epoch 000" in plot_calls[0][1]["annotation"]
    assert "Mesh epoch 001" in plot_calls[1][1]["annotation"]
    assert all(call[1]["show_window"] for call in plot_calls)
    assert result.visible_plot_refreshed
    assert result.initial_animation_frame_written
    assert result.animation_frame_written

    np.testing.assert_allclose(mesh_wrapper.fields["p"].x.array, 2.5)
    np.testing.assert_allclose(mesh_wrapper.fields["u"].x.array, -1.25)
    for boundary_name in mesh_wrapper.boundary_names():
        assert comm.allreduce(
            mesh_wrapper.get_boundary_facets(boundary_name).size,
            op=MPI.SUM,
        ) > 0

    new_mesh = mesh_wrapper.mesh
    new_owned_count = new_mesh.topology.index_map(new_mesh.topology.dim).size_local
    empty_cells = np.arange(new_owned_count, dtype=np.int32)
    empty = np.zeros(new_owned_count, dtype=bool)
    empty_indicator = CellIndicatorResult(
        cells=empty_cells,
        values=empty.astype(float),
        diameters=new_mesh.h(new_mesh.topology.dim, empty_cells),
        threshold_mask=empty.copy(),
        size_mask=np.ones(new_owned_count, dtype=bool),
        marked_mask=empty.copy(),
    )
    no_op_selection = expand_refinement_buffer(
        mesh_wrapper,
        empty_indicator,
        buffer_layers=0, target_h=0.1,
    )
    no_op = refine_mesh_topology(new_mesh, no_op_selection)
    no_op_result = mesh_wrapper.Install_Refined_State(
        no_op,
        config=RefinementConfig(
            replot_on_refinement=True,
            save_remesh_animations=True,
        ),
    )
    assert not no_op_result.installed
    assert mesh_wrapper.remesh_epoch == 1
    assert comm.allgather(len(plot_calls)) == [2, 2]

    try:
        mesh_wrapper.Install_Refined_State(
            no_op,
            config=RefinementConfig(replot_on_refinement=rank == 1),
        )
    except ValueError as exc:
        messages = comm.allgather(str(exc))
        assert len(set(messages)) == 1
        assert "configuration must be identical" in str(exc)
    else:
        raise AssertionError("Rank-dependent live-install configuration was accepted.")
    assert mesh_wrapper.remesh_epoch == 1
    assert comm.allgather(len(plot_calls)) == [2, 2]


def _run_repartition_checks() -> None:
    """Run a complete imbalanced-refine and post-transfer redistribution."""

    import numpy as np
    import ufl
    from dolfinx import fem
    from mpi4py import MPI

    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.mesh.refinement import (
        FieldThresholdIndicatorConfig,
        RefinementConfig,
    )
    from femcubes.core.utils.mpi_utils import comm, rank, size

    if size != 2:
        raise RuntimeError("The repartition checks require exactly two ranks.")

    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 1.0
        dx = 0.25
        dim = 2
        Quadrature_Degree = 3

    mesh = Mesh(Params())
    indicator = mesh.Add_Field("indicator", "One-sided indicator", "DG", 0)
    cg = mesh.Add_Field("cg", "Quadratic scalar", "CG", 2)
    dg = mesh.Add_Field("dg", "Linear discontinuous scalar", "DG", 1)
    vector = mesh.Add_Vector_Field("vector", "Linear vector", "CG", 1, 2)
    mesh.Add_Quadrature_Field("quadrature", "Quadrature history", ())

    owned_cells = mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_local
    for cell in range(owned_cells):
        dofs = indicator.function_space.dofmap.cell_dofs(cell)
        indicator.x.array[dofs] = float(rank == 1)
    indicator.x.scatter_forward()
    cg.interpolate(lambda x: 1.0 + x[0] + 2.0 * x[1] + 0.25 * x[0] ** 2)
    dg.interpolate(lambda x: -0.5 + 2.0 * x[0] - x[1])
    vector.interpolate(lambda x: np.vstack((x[0] + x[1], 2.0 * x[0] - x[1])))
    mesh.Constant_Field("quadrature", 7.25)
    for field in (cg, dg, vector):
        field.x.scatter_forward()

    initial_global_cells = mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_global
    local_h = mesh.mesh.h(
        mesh.mesh.topology.dim,
        np.arange(owned_cells, dtype=np.int32),
    )
    global_min_h = comm.allreduce(float(np.min(local_h)), op=MPI.MIN)
    config = RefinementConfig(
        enabled=True,
        indicator=FieldThresholdIndicatorConfig(field="indicator", threshold=0.5),
        target_h=0.4 * global_min_h,
        buffer_layers=0,
        repartition=True,
        repartition_imbalance_threshold=1.1,
    )

    result = mesh.Refine_From_Indicator(config)
    report = result.report
    topology = mesh.mesh.topology
    cell_map = topology.index_map(topology.dim)
    counts = comm.allgather(cell_map.size_local)
    mean_owned = sum(counts) / len(counts)

    assert result.installed
    assert report.repartitioned
    assert report.imbalance_ratio_before_repartition > config.repartition_imbalance_threshold
    assert report.imbalance_ratio_after < report.imbalance_ratio_before_repartition
    assert report.repartition_seconds > 0.0
    assert result.initial_transfer_diagnostics["quadrature"].reconstruction_seconds > 0.0
    assert report.global_cells_after > initial_global_cells
    assert report.global_cells_after == cell_map.size_global == sum(counts)
    assert report.min_owned_cells_after == min(counts)
    assert report.max_owned_cells_after == max(counts)
    assert report.mean_owned_cells_after == pytest.approx(mean_owned)
    assert report.imbalance_ratio_after == pytest.approx(max(counts) / mean_owned)
    assert comm.allreduce(cell_map.num_ghosts, op=MPI.SUM) > 0

    expected_expressions = {
        "cg": lambda x: 1.0 + x[0] + 2.0 * x[1] + 0.25 * x[0] ** 2,
        "dg": lambda x: -0.5 + 2.0 * x[0] - x[1],
        "vector": lambda x: np.vstack((x[0] + x[1], 2.0 * x[0] - x[1])),
    }
    for name, expression in expected_expressions.items():
        field = mesh.fields[name]
        expected = fem.Function(field.function_space)
        expected.interpolate(expression)
        expected.x.scatter_forward()
        owned_dofs = (
            field.function_space.dofmap.index_map.size_local
            * field.function_space.dofmap.index_map_bs
        )
        np.testing.assert_allclose(
            field.x.array[:owned_dofs],
            expected.x.array[:owned_dofs],
            rtol=1.0e-11,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(mesh.fields["quadrature"].x.array, 7.25)
    assert all(field.function_space.mesh is mesh.mesh for field in mesh.fields.values())

    assert set(mesh.domain_names()) == {"domain"}
    assert set(mesh.boundary_names()) == {"bottom", "right", "top", "left"}
    domain_count = comm.allreduce(mesh.get_domain_cells("domain").size, op=MPI.SUM)
    assert domain_count == cell_map.size_global
    local_area = fem.assemble_scalar(fem.form(fem.Constant(mesh.mesh, 1.0) * ufl.dx))
    assert comm.allreduce(local_area, op=MPI.SUM) == pytest.approx(4.0)
    expected_lengths = {"bottom": 4.0, "right": 1.0, "top": 4.0, "left": 1.0}
    measure = ufl.Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
    for name, expected_length in expected_lengths.items():
        local_length = fem.assemble_scalar(
            fem.form(
                fem.Constant(mesh.mesh, 1.0)
                * measure(mesh.get_boundary_tag(name))
            )
        )
        assert comm.allreduce(local_length, op=MPI.SUM) == pytest.approx(expected_length)

@pytest.mark.mpi
def test_two_rank_refinement_baseline_retains_shared_facet_ghosts():
    """The mesh generator must provide ghosts needed by future refinement."""

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    test_path = Path(__file__).resolve()
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, [str(test_path), "--shared-facet-baseline"])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if result.returncode != 0:
        output = "\n".join((result.stdout, result.stderr)).strip()
        raise AssertionError(f"MPI refinement baseline failed:\n{output}")


@pytest.mark.mpi
def test_two_rank_buffer_expansion_and_collective_decision():
    """Marks must cross shared facets and produce rank-consistent decisions."""

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    test_path = Path(__file__).resolve()
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, [str(test_path), "--buffer-check"])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if result.returncode != 0:
        output = "\n".join((result.stdout, result.stderr)).strip()
        raise AssertionError(f"MPI refinement buffer checks failed:\n{output}")


@pytest.mark.mpi
def test_two_rank_topology_and_marker_transfer():
    """Distributed refinement must preserve ancestry, ghosts, tags, and measures."""

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    test_path = Path(__file__).resolve()
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, [str(test_path), "--topology-marker-check"])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if result.returncode != 0:
        output = "\n".join((result.stdout, result.stderr)).strip()
        raise AssertionError(f"MPI topology/marker checks failed:\n{output}")


@pytest.mark.mpi
def test_two_rank_manufactured_field_transfer():
    """CG/DG transfer must remain exact across a refined partition interface."""

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    test_path = Path(__file__).resolve()
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, [str(test_path), "--field-transfer-check"])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if result.returncode != 0:
        output = "\n".join((result.stdout, result.stderr)).strip()
        raise AssertionError(f"MPI field-transfer checks failed:\n{output}")


@pytest.mark.mpi
def test_two_rank_quadrature_field_transfer():
    """Quadrature reconstruction must remain exact across a partition interface."""

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    test_path = Path(__file__).resolve()
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, [str(test_path), "--quadrature-transfer-check"])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if result.returncode != 0:
        output = "\n".join((result.stdout, result.stderr)).strip()
        raise AssertionError(f"MPI quadrature-transfer checks failed:\n{output}")


@pytest.mark.mpi
def test_two_rank_live_install_and_collective_visualization():
    """Every rank must install and enter requested remesh visualization calls."""

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    test_path = Path(__file__).resolve()
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, [str(test_path), "--live-install-check"])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    if result.returncode != 0:
        output = "\n".join((result.stdout, result.stderr)).strip()
        raise AssertionError(f"MPI live-install checks failed:\n{output}")


@pytest.mark.mpi
def test_two_rank_post_transfer_repartition_preserves_complete_state():
    """A one-sided refinement must be rebalanced only after full transfer."""

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    test_path = Path(__file__).resolve()
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, [str(test_path), "--repartition-check"])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    if result.returncode != 0:
        output = "\n".join((result.stdout, result.stderr)).strip()
        raise AssertionError(f"MPI post-transfer repartition checks failed:\n{output}")


@pytest.mark.mpi
def test_two_rank_collective_refinement_contracts():
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")
    test_path = Path(__file__).with_name("test_mesh_refinement_distributed_contracts.py").resolve()
    selective_path = test_path.with_name("test_mesh_refinement_selective_transfer.py")
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, ["-m", "pytest", "-q", str(test_path), str(selective_path)])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True,
                            env=environment, timeout=120)
    if result.returncode != 0:
        raise AssertionError("MPI collective contract checks failed:\n" + result.stdout + result.stderr)


@pytest.mark.mpi
def test_two_rank_refinement_of_subdomain_and_interface_fields():
    """Restricted fields must survive distributed refinement and repartitioning exactly."""
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")
    test_path = Path(__file__).with_name("test_mesh_subdomain_fields.py").resolve()
    project_root = test_path.parents[4]
    cmd, environment = _mpi_python_invocation(mpirun, ["-m", "pytest", "-q", "-p", "no:cacheprovider", str(test_path),
                                                       "-k", "TestRefinementOfRestrictedFields or TestInterfaceIntegrals"])
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True,
                            env=environment, timeout=300)
    if result.returncode != 0:
        raise AssertionError("MPI restricted-field refinement checks failed:\n" + result.stdout + result.stderr)


if __name__ == "__main__":
    if "--shared-facet-baseline" in sys.argv:
        _run_shared_facet_baseline()
    elif "--buffer-check" in sys.argv:
        _run_buffer_expansion_checks()
    elif "--topology-marker-check" in sys.argv:
        _run_topology_marker_checks()
    elif "--field-transfer-check" in sys.argv:
        _run_field_transfer_checks()
    elif "--quadrature-transfer-check" in sys.argv:
        _run_quadrature_transfer_checks()
    elif "--live-install-check" in sys.argv:
        _run_live_install_checks()
    elif "--repartition-check" in sys.argv:
        _run_repartition_checks()
