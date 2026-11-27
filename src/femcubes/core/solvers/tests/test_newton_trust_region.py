"""Public nonlinear-solver checks with distributed residuals and exact tangents."""

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.step_solver import StepSolver
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import SplitLinearElasticKernels
from femcubes.solids.tests.test_linear_elastic import _build_params


class _Params(ParamsTemplate):
    Solution_Steps = {"u": 0}
    n_passes = 1
    max_it = 50
    abs_tol = 1e-8
    rel_tol = 0.0
    line_search = True
    line_search_lims = [1e-3, 1.0]
    Step_Solvers = {0: "NewtonTrustRegion"}


class _Physics:
    """Three unknowns distributed over MPI ranks; buffers are deliberately reused."""

    def __init__(self, response, start, *, ghosted=False):
        self.response = response
        self.x = np.array(start, dtype=float)
        self.mesh = self
        self.K = PETSc.Mat().createAIJ((3, 3), nnz=3, comm=MPI.COMM_WORLD)
        self.K.setUp()
        self.F = (PETSc.Vec().createGhost([], size=3, comm=MPI.COMM_WORLD) if ghosted
                  else PETSc.Vec().createMPI(3, comm=MPI.COMM_WORLD))
        self.trials = 0
        self.updates = []

    def Get_Residual_Row_Scale(self, step):
        return None

    def Get_Kmat_Fvec(self, step):
        _, tangent = self.response(self.x)
        self.K.zeroEntries()
        begin, end = self.K.getOwnershipRange()
        self.K.setValues(range(begin, end), range(3), tangent[begin:end])
        self.K.assemble()
        return self.K, self.Get_Fvec(step)

    def Get_Fvec(self, step):
        residual, _ = self.response(self.x)
        begin, end = self.F.getOwnershipRange()
        self.F.array[:] = residual[begin:end]
        self.trials += 1
        return self.F

    def get_fields_for_step(self, step):
        return [("u", self.x)]

    def update_step_from_vector(self, direction, step):
        self.x += np.concatenate(MPI.COMM_WORLD.allgather(direction.getArray(readonly=True).copy()))
        self.updates.append(self.x.copy())

    def close(self):
        self.K.destroy()
        self.F.destroy()


def _stress_problem():
    # Four fixed-damage integration points coupled through three displacement
    # modes. Repeated stress-sign changes make backtracking alone stall.
    gradients = np.array([[[-0.17629885193383213, -0.5366100732132443, -1.0759821083114613],
      [0.16504232119700166, -1.118485849993954, 0.777889801389606],
      [1.5299268989806853, -1.051349383939167, -1.9655413282242522]],
     [[-1.0932452990743708, -0.1271906145711214, -1.1590460049676603],
      [0.9442284015768878, -1.577185318492185, -0.20497080951955327],
      [-2.3615590368263235, -0.5446558732044111, 0.22338748659141686]],
     [[1.1750024842024056, -1.3689312042388813, 1.9117977144248355],
      [-0.8765592910473323, 0.3446405164383298, -0.34097088896778],
      [-0.4327242554447549, -0.5304920682656402, 1.137552362205795]],
     [[-0.1601704763533892, 0.031149437519432315, -0.03328611453800047],
      [1.1766040293449396, -1.7385729562147005, -0.7073390884191151],
      [0.6694109906141685, 1.4588934487637872, -0.8379052226434885]]])
    offsets = np.array([[-0.03169058508376998, 0.01761802491237979, 0.0317973244495371],
     [0.0950719410293039, -0.24331097302361743, -0.12547078481921667],
     [-0.06689283127643418, 0.07062691041840687, -0.006607364020579318],
     [0.07953576694566253, 0.037727383954529646, 0.13185604279142876]])
    damage = np.array([6.33549686970418e-06, 0.00014971735730300457, 0.00016466523150594553, 0.0015839055685753086])
    solution = np.array([-0.10594252807779495, 0.563108187153122, -0.11898232808682711])
    start = np.array([0.010802909877968787, -0.0705928604043668, -0.28994557235997237])

    B = np.zeros((4, 6, 3))
    B[:, [0, 1, 5], :] = gradients
    base = np.zeros((4, 6))
    base[:, [0, 1, 5]] = offsets
    kernel = SplitLinearElasticKernels(4.0, 3.0, "Stress-Based", True)

    def internal(x):
        result = kernel(base+np.einsum("ijk,k->ij", B, x), damage)
        return (np.einsum("ijk,ij->k", B, result["stress"]),
                np.einsum("ijk,ijl,ilm->km", B, result["tangent"], B))

    target, _ = internal(solution)

    def response(x):
        force, tangent = internal(x)
        return force-target, tangent

    return response, start


def test_trust_region_recovers_a_stalled_stress_spectral_solve():
    response, start = _stress_problem()
    physics = _Physics(response, start)
    params = _Params()
    solver = StepSolver(params, physics)
    try:
        solver.Solve_Step()
        assert np.linalg.norm(response(physics.x)[0])/np.sqrt(3) < params.abs_tol
    finally:
        solver._cleanup_solver()
        physics.close()


@pytest.mark.parametrize("line_search", [False, True])
@pytest.mark.parametrize("at_solution", [False, True])
@pytest.mark.parametrize("method", ["NewtonTrustRegion", "NewtonContinuation"])
def test_trust_region_solves_nonsymmetric_linear_system(line_search, at_solution, method):
    matrix = np.array([[4., 1., 0.], [-2., 3., 1.], [0., -1., 2.]])
    exact = np.array([0.3, -0.2, 0.7])
    physics = _Physics(lambda x: (matrix@(x-exact), matrix), exact if at_solution else np.zeros(3))
    params = _Params()
    params.Step_Solvers = {0: method}
    params.line_search = line_search
    solver = StepSolver(params, physics)
    try:
        solver.Solve_Step()
        np.testing.assert_allclose(physics.x, exact, atol=1e-12)
    finally:
        solver._cleanup_solver()
        physics.close()


def test_trust_region_keeps_a_small_correction_when_no_trial_improves(capsys):
    # An inconsistent tangent deliberately makes every positive correction worse.
    physics = _Physics(lambda x: (np.ones(3)+x, -np.eye(3)), np.zeros(3))
    params = _Params()
    params.max_it = 1
    solver = StepSolver(params, physics)
    try:
        with pytest.raises(RuntimeError, match="did not converge after 1 iterations"):
            solver.Solve_Step()
        assert np.all(physics.x > 0)
        assert np.linalg.norm(physics.x) < 1e-8
        if MPI.COMM_WORLD.rank == 0:
            assert "accepting the smallest tried correction" in capsys.readouterr().out
    finally:
        solver._cleanup_solver()
        physics.close()


def test_pass_limit_warns_but_keeps_the_current_iterate(capsys):
    physics = _Physics(lambda x: (np.ones(3), np.eye(3)), np.zeros(3))
    params = _Params()
    params.Solution_Steps = {"u": 0, "phasefield": 1}
    params.Step_Solvers = {}
    params.n_passes = 2
    solver = StepSolver(params, physics)
    solver.solve_innerstep_newton = lambda step, residual_scale: None
    try:
        solver.Solve_Step()
        np.testing.assert_array_equal(physics.x, np.zeros(3))
        if MPI.COMM_WORLD.rank == 0:
            assert "staggered passes with residuals above tolerance" in capsys.readouterr().out
    finally:
        solver._cleanup_solver()
        physics.close()


@pytest.mark.parametrize("damage", [None, 1.0])
@pytest.mark.parametrize("split", ["NoTension", "Stress-Based"])
@pytest.mark.parametrize("method", ["NewtonTrustRegion", "NewtonContinuation"])
def test_solver_runs_the_split_elastic_finite_element_model(damage, split, method):
    params = _build_params()
    params.Models["Solid"]["Type"] = "Solid:SplitLinearElastic"
    params.Materials["Ice"]["CrackDrivingForce"] = split
    params.Step_Solvers = {0: method}
    params.line_search = True
    if damage is not None:
        params.Models["Solid"]["DamageModel"] = "PhaseField"
        params.Materials["Ice"].update(Gc=1.0, ell=1.0, f_t=1.0, kappa=1e-6)
    mesh = Mesh(params)
    if damage is not None:
        mesh.Add_Field("phasefield", "Prescribed damage", "CG", 1)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    if damage is not None:
        mesh.Constant_Field("phasefield", damage)
    solver = TimeSolver(params, physics)
    solver.run()
    material = params.Materials["Ice"]
    expected = material["Density"]*params.gravity*params.Ly**2/(2*(material["Lame"]+2*material["Shear"]))
    actual = MPI.COMM_WORLD.allreduce(mesh.Get_Field("u").sub(1).collapse().x.array.min(), op=MPI.MIN)
    np.testing.assert_allclose(actual, expected, rtol=1e-2)
