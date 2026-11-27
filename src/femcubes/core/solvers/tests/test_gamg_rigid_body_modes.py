"""Tests of the rigid-body near-null space that the GAMG solver attaches for displacement steps."""

import numpy as np
import pytest
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration
from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.solvers._lin_solvers import SolverWorkspace, _LinSolvers
from femcubes.core.utils.maths_utils import ft_to_Gc


class _Harness(_LinSolvers):
    pass


def _build_params(dim=2, steps=None, linear_solver=LinearSolver.GAMG):
    """PRIVATE: a steel plate in tension, with phase-field damage, on a coarse mesh."""
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        MeshType: str = "Rectangle"
        Lx: float = 1.0
        Ly: float = 0.5
        Lz: float = 0.5
        dx: float = 0.125
        Quadrature_Degree: int = 3

        Materials: dict = {}
        Materials["Steel"] = {"Density": 7.85e3, "Youngs": 210.0e9, "Poisson": 0.3, "Inertia": False, "ell": 0.05,
                              "f_t": 500.0e6, "kappa": 1.0e-6, "PF_formulation": "AT2", "pf_damping": 0.0}
        steel = Materials["Steel"]
        steel["Shear"] = steel["Youngs"] / (2.0 * (1.0 + steel["Poisson"]))
        steel["Lame"] = steel["Shear"] * (2.0 * steel["Poisson"]) / (1.0 - 2.0 * steel["Poisson"])
        steel["Gc"] = ft_to_Gc(steel["f_t"], steel["Youngs"], steel["ell"], steel["PF_formulation"])
        gravity: float = 0.0

        Field_Orders = {"u": 2, "phasefield": 1}
        Models: dict = {
            "Solid": {"Type": "Solid:LinearElastic", "Material": "Steel", "DamageModel": "PhaseField"},
            "PF": {"Type": "Solid:PhaseField", "Material": "Steel", "CDF_field": "ElasticEnergy",
                   "Irreversible_Method": "Hist"},
            "BC_Left": {"Type": "Boundary:PrescribedValue", "boundary": "left", "field": "u_x", "value": 0.0},
            "BC_Bottom": {"Type": "Boundary:PrescribedValue", "boundary": "bottom", "field": "u_y", "value": 0.0},
            "BC_Right": {"Type": "Boundary:DisplacementControl", "boundary": "right", "field": "u_x",
                         "rate": 1.0e-7, "Dummy": 1.0e16},
        }
        if dim == 3:
            Models["BC_Back"] = {"Type": "Boundary:PrescribedValue", "boundary": "back", "field": "u_z", "value": 0.0}
        ModelNames = list(Models)

        Linear_Solver: LinearSolver = LinearSolver.LU
        GMRES_RelTol: float = 1e-10
        n_passes: int = 2
        max_it: int = 20
        rel_tol: float = 1e-8
        abs_tol: float = 1e-6
        Newmark_Beta: float = 0.25
        Newmark_Gamma: float = 0.5
        start_time: float = 0.0
        end_time: float = 10.0
        dt: float = 5.0
        TimeStepControl = {"Type": "None"}

        output_dir: str = None
        write_vtk: bool = False
        write_hdf5: bool = False
        Plot_Fields: list = []
        Plot_Measures: list = []

    params = Params()
    params.dim = dim
    params.Solution_Steps = steps if steps is not None else {"u": 0, "phasefield": 1}
    params.Linear_Solver = linear_solver
    return params


def _physics(params):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    return mesh, physics


def _step_matrix(mesh, step):
    """PRIVATE: an empty matrix with the owned rows of ``step``'s solution vector."""
    n_owned = mesh.get_step_block_offsets(step)[-1]
    matrix = PETSc.Mat().createAIJ(((n_owned, PETSc.DETERMINE), (n_owned, PETSc.DETERMINE)), nnz=1,
                                   comm=MPI.COMM_WORLD)
    matrix.setUp()
    matrix.assemble()
    return matrix


@pytest.mark.parametrize("dim, n_modes", [(2, 3), (3, 6)])
def test_displacement_step_gets_orthonormal_rigid_body_modes(dim, n_modes):
    params = _build_params(dim)
    mesh, physics = _physics(params)
    K_mat = _step_matrix(mesh, 0)
    nullspace = _Harness(params, physics)._rigid_body_near_nullspace(0, K_mat)
    try:
        vectors = nullspace.getVecs()
        assert len(vectors) == n_modes
        gram = np.array([[a.dot(b) for b in vectors] for a in vectors])
        np.testing.assert_allclose(gram, np.eye(n_modes), atol=1e-12)

        # The modes span exactly the rigid motions u = t + W x (rotations here about the origin).
        space = mesh.fields["u"].function_space
        n_owned = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
        motions = [lambda x, k=k: np.outer(np.eye(dim)[k], np.ones(x.shape[1])) for k in range(dim)]
        for a, b in ([(0, 1)] if dim == 2 else [(0, 1), (1, 2), (2, 0)]):
            def rotation(x, a=a, b=b):
                values = np.zeros((dim, x.shape[1]))
                values[a], values[b] = -x[b], x[a]
                return values
            motions.append(rotation)
        function = fem.Function(space)
        basis = np.array([v.getArray(readonly=True) for v in vectors]).T
        for motion in motions:
            function.interpolate(motion)
            local = function.x.array[:n_owned]
            coefficients = MPI.COMM_WORLD.allreduce(basis.T @ local, op=MPI.SUM)
            residual = local - basis @ coefficients
            assert MPI.COMM_WORLD.allreduce(float(residual @ residual), op=MPI.SUM) < 1e-20 * max(
                MPI.COMM_WORLD.allreduce(float(local @ local), op=MPI.SUM), 1.0)
    finally:
        nullspace.destroy()
        K_mat.destroy()


@pytest.mark.parametrize("steps, step", [
    ({"u": 0, "phasefield": 1}, 1),     # a step without u
    ({"u": 0, "phasefield": 0}, 0),     # u together with another field
])
def test_steps_other_than_displacement_alone_get_no_modes(steps, step):
    params = _build_params(steps=steps)
    mesh, physics = _physics(params)
    K_mat = _step_matrix(mesh, step)
    try:
        assert _Harness(params, physics)._rigid_body_near_nullspace(step, K_mat) is None
    finally:
        K_mat.destroy()


def test_modes_are_attached_to_each_operator_and_rebuilt_after_a_remesh():
    params = _build_params()
    mesh, physics = _physics(params)
    solver = _Harness(params, physics)
    workspace = SolverWorkspace(signature=None, ksp=None, du=None, rhs=None)
    first, second = _step_matrix(mesh, 0), _step_matrix(mesh, 0)
    try:
        solver._attach_rigid_body_modes(workspace, first, 0)
        built = workspace.near_nullspace
        solver._attach_rigid_body_modes(workspace, second, 0)
        assert workspace.near_nullspace is built  # reused for the same mesh
        assert len(first.getNearNullSpace().getVecs()) == len(second.getNearNullSpace().getVecs()) == 3

        mesh.remesh_epoch = getattr(mesh, "remesh_epoch", 0) + 1
        solver._attach_rigid_body_modes(workspace, second, 0)
        assert workspace.near_nullspace is not built
    finally:
        solver._destroy_solver_workspace(workspace)
        first.destroy()
        second.destroy()


def test_no_mesh_or_no_step_attaches_nothing():
    solver = _Harness(_build_params(), object())
    workspace = SolverWorkspace(signature=None, ksp=None, du=None, rhs=None)
    solver._attach_rigid_body_modes(workspace, object(), 0)
    solver._attach_rigid_body_modes(workspace, object(), None)
    assert workspace.near_nullspace is None


@pytest.mark.full_model
def test_gamg_with_rigid_body_modes_reproduces_the_direct_solution():
    solutions = {}
    for linear_solver in (LinearSolver.LU, LinearSolver.GAMG):
        params = _build_params(linear_solver=linear_solver)
        mesh, physics = _physics(params)
        TimeSolver(params, physics).run()
        solutions[linear_solver] = mesh.Get_Field("u").x.array.copy()
    reference = solutions[LinearSolver.LU]
    assert np.abs(reference).max() > 0.0
    np.testing.assert_allclose(solutions[LinearSolver.GAMG], reference, rtol=0.0, atol=1e-8 * np.abs(reference).max())
