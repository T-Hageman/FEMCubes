"""Geometry, symmetry and distributed contracts for the half-plate generator.

The same tests can be run under mpirun; geometric measures are global sums.
"""
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.half_plate_with_hole import Sample_Parameters
from femcubes.core.mesh.generator_base import MeshGeneratorBase
from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig


def _measure(mesh, name=None):
    """PRIVATE: Integrate the domain or a named boundary across all ranks."""
    measure = ufl.dx if name is None else mesh.get_Boundary_Integrator()(mesh.get_boundary_tag(name))
    value = fem.assemble_scalar(fem.form(fem.Constant(mesh.mesh, 1.0) * measure))
    return mesh.mesh.comm.allreduce(value, op=MPI.SUM)


@pytest.mark.parametrize('ellipticity,angle', [(1.0, 0.0), (2.0, 0.0), (2.0, 90.0)])
def test_half_plate_geometry_and_complete_boundary_partition(ellipticity, angle):
    params = Sample_Parameters()
    params.ellipticity = ellipticity
    params.rotation_angle = angle
    mesh = Mesh(params)
    domain = mesh.mesh
    expected = {'bottom', 'top', 'right', 'hole', 'symmetry'}
    assert set(mesh.boundary_names()) == expected
    assert mesh.domain_names() == ['domain']
    assert np.isclose(_measure(mesh), (params.Lx * params.Ly - np.pi * params.R**2 / ellipticity) / 2, rtol=2e-3)
    assert np.all(domain.geometry.x[:, 0] >= params.Lx / 2 - 1e-12)
    assert np.all(domain.geometry.x[:, 0] <= params.Lx + 1e-12)
    assert np.all(domain.geometry.x[:, 1] >= -1e-12)
    assert np.all(domain.geometry.x[:, 1] <= params.Ly + 1e-12)
    assert np.isclose(_measure(mesh, 'top'), params.Lx / 2)
    assert np.isclose(_measure(mesh, 'bottom'), params.Lx / 2)
    assert np.isclose(_measure(mesh, 'right'), params.Ly)
    ry = params.R / ellipticity if angle == 0 else params.R
    assert np.isclose(_measure(mesh, 'symmetry'), params.Ly - 2 * ry)
    facet_dim = domain.topology.dim - 1
    domain.topology.create_connectivity(facet_dim, domain.topology.dim)
    owned = domain.topology.index_map(facet_dim).size_local
    tagged = np.concatenate([mesh.get_boundary_facets(n) for n in expected])
    assert np.unique(tagged).size == tagged.size
    exterior = dmesh.exterior_facet_indices(domain.topology)
    np.testing.assert_array_equal(np.sort(tagged[tagged < owned]), exterior)
    symmetry_facets = mesh.get_boundary_facets('symmetry')
    centers = dmesh.compute_midpoints(domain, facet_dim, symmetry_facets)
    np.testing.assert_allclose(centers[:, 0], params.Lx / 2, atol=1e-12)
    assert np.all(np.abs(centers[:, 1] - params.Ly / 2) >= ry - 1e-12)
    counts = domain.comm.allreduce(np.array([np.count_nonzero(centers[:,1] < params.Ly / 2),
                                           np.count_nonzero(centers[:,1] > params.Ly / 2)]), op=MPI.SUM)
    assert np.all(counts > 0)


@pytest.mark.parametrize('name,value,message', [
    ('dim', 3, 'only dim=2'), ('R', 2.0, 'fit strictly'),
    ('Lx', -1, 'positive finite Lx'), ('dx', 0.0, 'positive finite dx'),
    ('dxmin', np.nan, 'positive finite dxmin'),
    ('R_Refine', -1, 'positive finite R_Refine'),
    ('ellipticity', 0, 'positive finite ellipticity'),
    ('rotation_angle', np.inf, 'finite rotation_angle'),
])
def test_half_plate_rejects_invalid_geometry(name, value, message):
    params = Sample_Parameters()
    setattr(params, name, value)
    with pytest.raises((ValueError, NotImplementedError), match=message):
        MeshGeneratorBase.get_generator(params.MeshType).generate(params)


def test_half_plate_rejects_rotation_that_breaks_reflection_symmetry():
    params = Sample_Parameters()
    params.ellipticity = 2.0
    params.rotation_angle = 30.0
    with pytest.raises(ValueError, match='multiple of 90 degrees'):
        MeshGeneratorBase.get_generator(params.MeshType).generate(params)


def test_quadratic_bernstein_symmetry_bc_constrains_only_horizontal_motion():
    params = Sample_Parameters()
    mesh = Mesh(params)
    mesh.Add_Vector_Field('u', 'Displacement', 'CG', 2, 2)
    mesh.Add_Vector_TestFunction('u', 2, 0)
    mesh.finalize_test_functions()
    mesh.Constant_Field('u', 2.0)
    bc = mesh.Build_Increment_Dirichlet_BCs('u', 'symmetry', 0.0, component=0)[0]
    field = mesh.Get_Field_For_Output('u').field
    increment = np.zeros_like(field.x.array)
    bc.set(increment)
    field.x.array[:] -= increment
    field.x.scatter_forward()
    view = mesh.Get_Field_For_Output('u')
    ux, uy = view.collapse_component(0), view.collapse_component(1)
    dofs = fem.locate_dofs_topological(ux.function_space, 1, mesh.get_boundary_facets('symmetry'))
    assert mesh.mesh.comm.allreduce(len(dofs)) > 0
    np.testing.assert_allclose(ux.x.array[dofs], 0.0, atol=1e-12)
    np.testing.assert_allclose(uy.x.array, 2.0, atol=1e-12)


def test_half_plate_symmetry_tags_survive_adaptive_refinement():
    mesh = Mesh(Sample_Parameters())
    before = {name: _measure(mesh, name) for name in mesh.boundary_names()}
    mesh.Add_Field('indicator', 'Refinement indicator', 'CG', 1)
    mesh.Constant_Field('indicator', 1.0)
    config = RefinementConfig(enabled=True,
                             indicator=FieldThresholdIndicatorConfig(field='indicator', threshold=0.5),
                             target_h=0.1, buffer_layers=0)
    result = mesh.Refine_From_Indicator(config)
    assert result.installed
    for name, length in before.items():
        assert np.isclose(_measure(mesh, name), length, rtol=1e-12, atol=1e-12)


def _elastic_reaction(mesh_type):
    """PRIVATE: Solve a small clamped tensile test and integrate top traction."""
    params = Sample_Parameters()
    params.MeshType = mesh_type
    params.Lx = params.Ly = 1.0
    params.R = 0.1
    params.dx = 0.08
    params.dxmin = 0.025
    params.R_Refine = 0.05
    mesh = Mesh(params)
    domain = mesh.mesh
    V = fem.functionspace(domain, ('Lagrange', 2, (2,)))
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    def stress(u):
        return 2.0 * ufl.sym(ufl.grad(u)) + ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(2)
    a = ufl.inner(stress(trial), ufl.sym(ufl.grad(test))) * ufl.dx
    L = ufl.inner(fem.Constant(domain, PETSc.ScalarType((0, 0))), test) * ufl.dx
    bcs = []
    for boundary, component, value in [('bottom',0,0.0),('bottom',1,0.0),
                                        ('top',0,0.0),('top',1,0.001)]:
        dofs = fem.locate_dofs_topological(V.sub(component), 1, mesh.get_boundary_facets(boundary))
        bcs.append(fem.dirichletbc(PETSc.ScalarType(value), dofs, V.sub(component)))
    if mesh_type == 'HalfPlateWithHole':
        dofs = fem.locate_dofs_topological(V.sub(0), 1, mesh.get_boundary_facets('symmetry'))
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0), dofs, V.sub(0)))
    problem = LinearProblem(a, L, bcs=bcs, petsc_options_prefix='half_plate_test_',
                            petsc_options={'ksp_type':'preonly', 'pc_type':'lu',
                                           'pc_factor_mat_solver_type':'mumps'})
    solution = problem.solve()
    solution.x.scatter_forward()
    assert problem.solver.getConvergedReason() > 0
    traction = ufl.dot(stress(solution), ufl.FacetNormal(domain))[1]
    ds = mesh.get_Boundary_Integrator()
    reaction = fem.assemble_scalar(fem.form(traction * ds(mesh.get_boundary_tag('top'))))
    return domain.comm.allreduce(reaction, op=MPI.SUM)


def test_doubled_half_plate_elastic_reaction_matches_full_plate():
    full = _elastic_reaction('FullPlateWithHole')
    half = _elastic_reaction('HalfPlateWithHole')
    assert full > 0 and half > 0
    assert np.isclose(2 * half, full, rtol=5e-3)
