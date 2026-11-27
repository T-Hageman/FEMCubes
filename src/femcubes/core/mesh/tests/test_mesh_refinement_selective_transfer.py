"""Selective refinement transfer must retain untouched history samples."""

from dataclasses import replace

import basix.ufl
from dolfinx import fem
from mpi4py import MPI
import numpy as np
import pytest

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.mesh.refinement import (
    CellIndicatorResult, expand_refinement_buffer, get_quadrature_reconstruction,
    refine_mesh_topology, snapshot_registered_fields, transfer_quadrature_fields,
    transfer_standard_fields,
)
import femcubes.core.mesh.refinement.transfer as transfer_module


def _local_refinement(wrapper, uniform=False):
    """Refine one corner cell, allowing conformity closure to add neighbours."""

    mesh = wrapper.mesh
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    cells = np.arange(cell_map.size_local, dtype=np.int32)
    centers = mesh.geometry.x[mesh.geometry.dofmap[cells]].mean(axis=1)
    local_nearest = (float(np.min(np.linalg.norm(centers, axis=1))), mesh.comm.rank)
    seed_rank = min(mesh.comm.allgather(local_nearest))[1]
    marked = np.full(cells.size, uniform, dtype=bool)
    if mesh.comm.rank == seed_rank:
        marked[np.argmin(np.linalg.norm(centers, axis=1))] = True
    indicator = CellIndicatorResult(cells, marked.astype(float), mesh.h(mesh.topology.dim, cells),
                                    marked.copy(), np.ones(cells.size, dtype=bool), marked.copy())
    selection = expand_refinement_buffer(wrapper, indicator, 0, target_h=0.01)
    return refine_mesh_topology(mesh, selection)


def _child_classification(source, artifacts):
    """Resolve test ancestry independently using gathered global parent IDs."""

    cell_map = source.topology.index_map(source.topology.dim)
    parents = artifacts.parent_cell
    global_parents = np.concatenate(source.comm.allgather(cell_map.local_to_global(parents)))
    multiplicities = np.bincount(parents, minlength=cell_map.size_local)
    split_parent = np.concatenate(source.comm.allgather(multiplicities[parents] > 1))
    child_ids = artifacts.refined_mesh.topology.original_cell_index
    parent_local = cell_map.global_to_local(np.ascontiguousarray(global_parents[child_ids]))
    return parent_local, np.flatnonzero(split_parent[child_ids]), np.flatnonzero(~split_parent[child_ids])


def _parent_quadrature_reference(source, destination, quadrature_degree, parents, cells):
    """Evaluate the fitted parent polynomial independently using Basix."""

    mesh = source.function_space.mesh
    reconstruction = get_quadrature_reconstruction(
        mesh.basix_cell(), quadrature_degree,
        points=source.function_space.element.interpolation_points,
    )
    basis = basix.ufl.element("DG", mesh.basix_cell(), reconstruction.degree).basix_element
    bs = source.function_space.dofmap.index_map_bs
    expected = fem.Function(destination.function_space)
    target = destination.function_space.mesh
    points = destination.function_space.element.interpolation_points
    for cell in cells:
        parent = parents[cell]
        physical = target.geometry.cmap.push_forward(points, target.geometry.x[target.geometry.dofmap[cell]])
        reference = mesh.geometry.cmap.pull_back(physical, mesh.geometry.x[mesh.geometry.dofmap[parent]])
        samples = source.x.array.reshape(-1, bs)[source.function_space.dofmap.cell_dofs(parent)]
        coefficients = reconstruction.sample_to_coefficients @ samples
        values = basis.tabulate(0, reference)[0, :, :, 0] @ coefficients
        expected.x.array.reshape(-1, bs)[expected.function_space.dofmap.cell_dofs(cell)] = values
    return expected


@pytest.mark.parametrize(("dim", "quadrature_degree"), [(2, 3), (3, 3), (3, 5)])
def test_selective_transfer_copies_untouched_cells_and_interpolates_closure(
    refinement_rectangle_params, dim, quadrature_degree, monkeypatch,
):
    params = refinement_rectangle_params
    params.dim, params.Lz, params.dx = dim, 1.0, 0.25
    params.Quadrature_Degree = quadrature_degree
    wrapper = Mesh(params)
    expressions = {
        "cg": lambda x: 1.0 + x[0] ** 3 - 0.5 * x[1],
        "vector": lambda x: np.vstack([x[i] + x[0] ** 3 for i in range(dim)]),
        "dg": lambda x: 2.0 - x[0] + x[1] ** 2,
    }
    wrapper.Add_Field("cg", "Continuous scalar", "CG", 3)
    wrapper.Add_Vector_Field("vector", "Continuous vector", "CG", 3, dim)
    wrapper.Add_Field("dg", "Discontinuous scalar", "DG", 2)
    for name, expression in expressions.items():
        wrapper.fields[name].interpolate(expression)
    rng = np.random.default_rng(321 + wrapper.mesh.comm.rank)
    for name, shape in (("samples", ()), ("stress", (6, 1)), ("tangent", (6, 6))):
        field = wrapper.Add_Quadrature_Field(name, name, shape)
        field.x.array[:] = rng.normal(size=field.x.array.size)
    for field in wrapper.fields.values():
        field.x.scatter_forward()
    originals = {name: field.x.array.copy() for name, field in wrapper.fields.items()}
    snapshot = snapshot_registered_fields(wrapper)
    artifacts = _local_refinement(wrapper)
    parents, split, unchanged = _child_classification(wrapper.mesh, artifacts)
    assert wrapper.mesh.comm.allreduce(unchanged.size, op=MPI.SUM) > 0
    assert wrapper.mesh.comm.allreduce(split.size, op=MPI.SUM) > 1
    # Count actual split coarse parents: closure must extend beyond the seed.
    assert wrapper.mesh.comm.allreduce(np.count_nonzero(np.bincount(artifacts.parent_cell) > 1), op=MPI.SUM) > 1

    reconstructed_counts = []
    original_populate = transfer_module._populate_reconstructed_dg_field

    def track_reconstruction(source, reconstructed, reconstruction, shape, cells):
        reconstructed_counts.append(cells.size)
        return original_populate(source, reconstructed, reconstruction, shape, cells)

    monkeypatch.setattr(transfer_module, "_populate_reconstructed_dg_field", track_reconstruction)
    standard = transfer_standard_fields(snapshot, artifacts)
    complete = transfer_quadrature_fields(snapshot, artifacts, standard)
    old_owned = wrapper.mesh.topology.index_map(dim).size_local
    for count in reconstructed_counts:
        assert wrapper.mesh.comm.allreduce(count, op=MPI.SUM) < wrapper.mesh.comm.allreduce(old_owned, op=MPI.SUM)

    for name, old in wrapper.fields.items():
        new = complete.fields[name]
        old_map, new_map = old.function_space.dofmap, new.function_space.dofmap
        bs = old_map.index_map_bs
        copied = new.x.array.reshape(-1, bs)[new_map.list[unchanged]]
        original = old.x.array.reshape(-1, bs)[old_map.list[parents[unchanged]]]
        if name in ("cg", "vector"):
            np.testing.assert_allclose(copied, original, rtol=1e-13, atol=1e-13)
        else:
            np.testing.assert_array_equal(copied, original)
        if name in expressions:
            expected = fem.Function(new.function_space)
            expected.interpolate(expressions[name])
            expected.x.scatter_forward()
        else:
            expected = _parent_quadrature_reference(old, new, quadrature_degree, parents, split)
        np.testing.assert_allclose(new.x.array.reshape(-1, bs)[new_map.list[split]],
                                   expected.x.array.reshape(-1, bs)[new_map.list[split]], rtol=2e-11, atol=2e-11)
        np.testing.assert_array_equal(old.x.array, originals[name])
        assert complete.diagnostics[name].copy_seconds > 0.0


def test_uniform_refinement_reconstructs_multiple_batches(refinement_rectangle_params):
    params = refinement_rectangle_params
    params.dx = 0.07
    wrapper = Mesh(params)
    old_owned = wrapper.mesh.topology.index_map(2).size_local
    assert wrapper.mesh.comm.allreduce(old_owned, op=MPI.SUM) > 512
    field = wrapper.Add_Quadrature_Field("history", "Vector history", (6, 1))
    values = np.linspace(-2.0, 3.0, 6)
    field.x.array.reshape(-1, 6)[:] = values
    field.x.scatter_forward()
    snapshot = snapshot_registered_fields(wrapper)
    artifacts = _local_refinement(wrapper, uniform=True)
    standard = transfer_standard_fields(snapshot, artifacts)
    complete = transfer_quadrature_fields(snapshot, artifacts, standard)
    np.testing.assert_allclose(complete.fields["history"].x.array.reshape(-1, 6),
                               np.broadcast_to(values, complete.fields["history"].x.array.reshape(-1, 6).shape), atol=1e-12)


@pytest.mark.parametrize(("dim", "quadrature_degree"), [(2, 3), (3, 5)])
def test_discontinuous_fields_keep_the_parent_trace_at_cell_faces(refinement_rectangle_params, dim, quadrature_degree):
    params = refinement_rectangle_params
    params.dim, params.Lz, params.dx = dim, 1.0, 0.4
    params.Quadrature_Degree = quadrature_degree
    wrapper = Mesh(params)
    dg = wrapper.Add_Field("dg", "Discontinuous polynomial", "DG", 2)
    history = wrapper.Add_Quadrature_Field("history", "Discontinuous history", ())
    source = wrapper.mesh
    source_map = source.topology.index_map(dim)
    source_ids = source_map.local_to_global(np.arange(source_map.size_local + source_map.num_ghosts, dtype=np.int32))
    expression = lambda x: 2.0 + x[0] - 0.7 * x[1]
    for field in (dg, history):
        field.interpolate(expression)
        for cell in range(source_map.size_local):
            field.x.array[field.function_space.dofmap.cell_dofs(cell)] += 10.0 * source_ids[cell]
        field.x.scatter_forward()
    if dim == 3:
        points = history.function_space.element.interpolation_points
        assert np.min(np.column_stack((points, 1.0 - points.sum(axis=1)))) == pytest.approx(0.0)
    snapshot = snapshot_registered_fields(wrapper)
    artifacts = _local_refinement(wrapper)
    parents, _, _ = _child_classification(source, artifacts)
    standard = transfer_standard_fields(snapshot, artifacts)
    complete = transfer_quadrature_fields(snapshot, artifacts, standard)
    for field in complete.fields.values():
        expected = fem.Function(field.function_space)
        expected.interpolate(expression)
        for cell, parent in enumerate(parents):
            expected.x.array[expected.function_space.dofmap.cell_dofs(cell)] += 10.0 * source_ids[parent]
        np.testing.assert_allclose(field.x.array, expected.x.array, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(("invalid", "message"), [
    ("parent", "parents must index owned source cells"),
    ("offset", "input-cell IDs do not match"),
    ("geometry", "changed geometry or reference-cell ordering"),
])
def test_invalid_copy_ancestry_is_rejected_collectively(refinement_mesh, invalid, message):
    field = refinement_mesh.Add_Field("history", "History", "DG", 0)
    field.x.array[:] = 3.0
    snapshot = snapshot_registered_fields(refinement_mesh)
    artifacts = _local_refinement(refinement_mesh)
    if refinement_mesh.mesh.comm.rank == 0:
        if invalid == "parent":
            parents = artifacts.parent_cell.copy()
            parents[0] = -1
            artifacts = replace(artifacts, parent_cell=parents)
        elif invalid == "offset":
            artifacts = replace(artifacts, input_cell_global_offset=artifacts.input_cell_global_offset + 1)
        else:
            # A rigid translation leaves topology valid, but prevents direct
            # coefficient copying between nominally unchanged cells.
            artifacts.refined_mesh.geometry.x[:, 0] += 0.01
    with pytest.raises(ValueError, match=message):
        transfer_standard_fields(snapshot, artifacts)
    np.testing.assert_array_equal(field.x.array, 3.0)
