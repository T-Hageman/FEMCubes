"""Finite-element integration of the split viscoelastic model."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.solids.split_viscoelastic import SplitViscoElastic
from femcubes.solids.tests.test_split_viscoelastic_kernels import SPLITS, COUPLINGS
import femcubes.boundary


def _build_params(split="NoTension", coupling="EffectiveStress", *, true_2d=False):
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        dim = 2
        dx = 0.5
        Lx = 1.0
        Ly = 1.0
        Quadrature_Degree = 2
        Global_Measures = {}
        Materials = {"Ice": {
            "Lame": 4.0, "Shear": 3.0, "Bulk": 6.0, "Density": 1.0,
            "Visc_Type": "Newtonian", "Viscosity": 5.0,
            "Gc": 1.0, "ell": 0.25, "f_t": 2.0, "kappa": 1e-3,
            "PF_formulation": "AT2", "LE_Damping": 0.0, "Inertia": False,
            "CrackDrivingForce": split,
        }}
        gravity = 0.0
        Plot_Fields = []
        Plot_Measures = []
        Solution_Steps = {"u": 0, "phasefield": 1}
        Field_Orders = {"u": 1, "phasefield": 1}
        ModelNames = ["Solid", "pf"]
        Models = {
            "Solid": {"Type": "Solid:SplitViscoElastic", "Material": "Ice",
                      "DamageModel": "PhaseField",
                      "CreepCoupling": coupling, "true_2d": true_2d},
            "pf": {"Type": "Solid:PhaseField", "Material": "Ice",
                   "CDF_field": "ElasticEnergy", "Irreversible_Method": "Hist"},
        }
        dt = 0.7
        t = 0.0
        Newmark_Beta = 0.25
        Newmark_Gamma = 0.5
        output_dir = None
    return Params()


def _build(params):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    mesh.Constant_Field("phasefield", 0.6)
    return mesh, physics, physics.models["Solid"]


def _integral(form):
    return MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("coupling", COUPLINGS)
@pytest.mark.parametrize("n", [1, 3])
def test_assembled_mechanics_tangent_matches_residual_derivative(split, coupling, n):
    params = _build_params(split, coupling)
    params.Materials["Ice"].update(Inertia=True, LE_Damping=0.4, kappa=1e-6)
    if n == 3:
        params.Materials["Ice"].update(Visc_Type="Glen", Glen_n=3, Glen_A=9/(2*5**3))
    mesh, physics, model = _build(params)
    mesh.Constant_Field("phasefield", 1.0)
    assert isinstance(model, SplitViscoElastic)
    u = mesh.Get_Field("u")
    if coupling == "PartialExplicit":
        mesh.Get_Field("uOld").interpolate(lambda x: np.vstack((0.13*x[0], -0.17*x[1])))
        mesh.Get_Field("uOld").x.scatter_forward()
    probe = fem.Function(u.function_space)
    probe.interpolate(lambda x: np.vstack((0.3*x[0]+0.2*x[1], -0.1*x[1]+0.1*x[0])))
    probe.x.scatter_forward()
    # Distinct test/trial directions also detect a transposed nonsymmetric tangent.
    test = fem.Function(u.function_space)
    test.interpolate(lambda x: np.vstack((-0.2*x[0]+0.3*x[1], 0.4*x[1]+0.1*x[0])))
    test.x.scatter_forward()

    def set_displacement(delta):
        u.interpolate(lambda x: np.vstack((0.17*x[0]+0.04*x[1], -0.23*x[1]+0.04*x[0])))
        u.x.array[:] += delta*probe.x.array
        u.x.scatter_forward()
        return model.assemble_KF(0)

    K, F = set_displacement(0)
    analytic = _integral(ufl.action(ufl.action(K, probe), test))
    h = 1e-6
    _, F = set_displacement(h)
    plus = _integral(ufl.action(F, test))
    _, F = set_displacement(-h)
    minus = _integral(ufl.action(F, test))
    np.testing.assert_allclose(analytic, (plus-minus)/(2*h), rtol=1e-6, atol=1e-9)
    assert np.isfinite(mesh.Get_Field("ElasticEnergy").x.array).all()
    Kpf, Fpf = physics.models["pf"].assemble_KF(1)
    assert Kpf is not None and Fpf is not None


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_weak_form_degrades_only_the_selected_stress_once(split, sign):
    params = _build_params(split, true_2d=True)
    mesh, _, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: sign*0.2*x[:2])
    u.x.scatter_forward()
    _, F = model.assemble_KF(0)
    factor = 1e-3+(1-1e-3)*0.4**2 if sign > 0 or split == "FullEnergy" else 1.0
    np.testing.assert_allclose(_integral(ufl.action(F, u)), 1.12*factor, rtol=1e-10)


@pytest.mark.parametrize("inertia", [False, True])
def test_density_body_force_and_inertia_are_retained_at_all_damage_levels(inertia):
    params = _build_params()
    params.gravity = -2.0
    params.Materials["Ice"].update(Density=3.0, Inertia=inertia)
    mesh, _, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.full(x.shape[1], 0.2))))
    u.x.scatter_forward()
    probe = fem.Function(u.function_space)
    probe.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.ones(x.shape[1]))))
    probe.x.scatter_forward()
    mesh.Constant_Field("phasefield", 0.0)
    K, F = model.assemble_KF(0)
    mass_tangent = 3.0 / (params.Newmark_Beta * params.dt**2) if inertia else 0.0
    for damage in (0.0, 0.8, 1.0):
        mesh.Constant_Field("phasefield", damage)
        model.Update_Weak_Constants(0.0, params.dt, 0)
        assert model.d_rho == 1.0
        # Rigid translation eliminates internal work, isolating gravity and mass.
        np.testing.assert_allclose(_integral(ufl.action(F, probe)),
                                   6.0 + 0.2*mass_tangent, rtol=1e-10)
        np.testing.assert_allclose(_integral(ufl.action(ufl.action(K, probe), probe)),
                                   mass_tangent, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("pf_form", ["AT1", "AT2", "HO-AT1", "HO-AT2"])
def test_strength_driver_uses_current_elastic_stress(pf_form):
    params = _build_params("Stress-Based", "DegradedStress", true_2d=True)
    # The solid's driver can be checked independently of higher-order PF fields.
    params.ModelNames = ["Solid"]
    params.Materials["Ice"]["PF_formulation"] = pf_form
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Damage", "CG", 1)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]
    mesh.Constant_Field("phasefield", 0.5)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((0.3*x[0], 0.1*x[1])))
    u.x.scatter_forward()
    model.assemble_KF(0)
    positive = mesh.Get_Field("sigma_plus").x.array.reshape(-1, 6)
    strength = np.sum(positive**2, axis=1)/4.0
    expected = strength-1 if pf_form in ("AT2", "HO-AT2") else 0.75*strength
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, expected)
    assert np.linalg.norm(mesh.Get_Field("Plastic_Strain").x.array) > 0


@pytest.mark.parametrize("coupling", COUPLINGS)
def test_committed_history_trial_updates_and_reset(coupling):
    params = _build_params(coupling=coupling)
    mesh, physics, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((0.2*x[0], -0.1*x[1])))
    u.x.scatter_forward()
    model.assemble_KF(0)
    first = mesh.Get_Field("sigma").x.array.copy()
    model.assemble_KF(0)
    np.testing.assert_allclose(mesh.Get_Field("sigma").x.array, first)
    physics.commit()
    committed = mesh.Get_Field("Plastic_Strain").x.array.copy()
    model.assemble_KF(0)
    assert not np.allclose(mesh.Get_Field("Plastic_Strain").x.array, committed)
    physics.Reset_Step()
    np.testing.assert_allclose(mesh.Get_Field("Plastic_Strain").x.array, committed)
    model.assemble_KF(0)
    before = mesh.Get_Field("sigma").x.array.copy()
    # Cache rebuilding must bind to live coefficient objects without changing state.
    physics.Post_Remesh()
    model.assemble_KF(0)
    np.testing.assert_allclose(mesh.Get_Field("sigma").x.array, before)


def test_partial_explicit_creep_is_fixed_during_mechanics_and_recomputed_for_retry():
    params = _build_params(coupling="PartialExplicit")
    mesh, physics, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((0.2*x[0], -0.1*x[1])))
    u.x.scatter_forward()
    model.assemble_KF(0)
    np.testing.assert_array_equal(mesh.Get_Field("Plastic_Strain").x.array, 0.0)
    physics.commit()
    committed_u = mesh.Get_Field("uOld").x.array.copy()

    def update(dt):
        model.Update_Weak_Constants(0.0, dt, 0)
        trial = np.array([0.2, -0.1, 0, 0, 0, 0])
        dev = trial.copy()
        dev[:3] -= dev[:3].mean()
        relaxation = params.Materials["Ice"]["Shear"]*dt/params.Materials["Ice"]["Viscosity"]
        expected = relaxation/(1+relaxation)*dev
        creep = mesh.Get_Field("Plastic_Strain").x.array.reshape(-1, 6)
        np.testing.assert_allclose(creep, np.broadcast_to(expected, creep.shape), rtol=1e-11, atol=1e-13)
        np.testing.assert_array_equal(mesh.Get_Field("Plastic_Strain_Old").x.array, 0.0)
        np.testing.assert_array_equal(mesh.Get_Field("uOld").x.array, committed_u)
        return creep.copy()

    first = update(params.dt)
    u.interpolate(lambda x: np.vstack((-0.3*x[0]+0.1*x[1], 0.4*x[1])))
    u.x.scatter_forward()
    mesh.Constant_Field("phasefield", 1.0)
    np.testing.assert_array_equal(update(params.dt), first)
    physics.Reset_Step()
    np.testing.assert_array_equal(mesh.Get_Field("Plastic_Strain").x.array, 0.0)
    retried = update(params.dt/2)
    assert np.linalg.norm(retried) < np.linalg.norm(first)


@pytest.mark.full_model
def test_partial_explicit_solved_displacement_hold_matches_backward_euler_relaxation():
    params = _build_params(coupling="PartialExplicit")
    params.Models["Solid"]["DamageModel"] = None
    params.ModelNames = ["Solid"]
    params.Solution_Steps = {"u": 0}
    params.start_time, params.end_time, params.dt = 0.0, 0.3, 0.1
    params.Linear_Solver = LinearSolver.LU
    params.Step_Solvers = {0: "Newton"}
    params.n_passes, params.max_it = 1, 10
    params.rel_tol, params.abs_tol = 1e-10, 1e-12
    for name, boundary, field, value in (
        ("Left", "left", "u_x", 0.0), ("Right", "right", "u_x", 0.1),
        ("Bottom", "bottom", "u_y", 0.0), ("Top", "top", "u_y", -0.1),
    ):
        params.ModelNames.append(name)
        params.Models[name] = {"Type": "Boundary:PrescribedValue", "boundary": boundary,
                               "field": field, "value": value}
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solver = TimeSolver(params, physics)
    solver.run()
    strain = np.array([0.1, -0.1, 0, 0, 0, 0])
    # The first loading increment is elastic; subsequent increments relax at
    # the held strain using the Newtonian backward-Euler factor.
    shear, viscosity = params.Materials["Ice"]["Shear"], params.Materials["Ice"]["Viscosity"]
    factor = (1+shear*params.dt/viscosity)**-2
    stress = mesh.Get_Field("sigma").x.array.reshape(-1, 6)
    creep = mesh.Get_Field("Plastic_Strain").x.array.reshape(-1, 6)
    np.testing.assert_allclose(stress, np.broadcast_to(2*shear*factor*strain, stress.shape), atol=1e-11)
    np.testing.assert_allclose(creep, np.broadcast_to((1-factor)*strain, creep.shape), atol=1e-12)


@pytest.mark.parametrize("coupling", COUPLINGS)
def test_fields_and_assembly_survive_live_refinement(coupling):
    from femcubes.core.physics.tests.test_physics_remesh import _refine_live_mesh
    params = _build_params(coupling=coupling)
    mesh, physics, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((0.2*x[0], -0.1*x[1])))
    u.x.scatter_forward()
    physics.Get_Kmat_Fvec(0)
    physics.commit()
    physics.Get_Kmat_Fvec(0)
    expected_stress = mesh.Get_Field("sigma").x.array.reshape(-1, 6)[0].copy()
    expected_driver = mesh.Get_Field("ElasticEnergy").x.array[0]
    assert _refine_live_mesh(mesh).installed
    physics.Post_Remesh()
    K, F = physics.Get_Kmat_Fvec(0)
    assert np.isfinite(K.norm()) and np.isfinite(F.norm())
    stresses = mesh.Get_Field("sigma").x.array.reshape(-1, 6)
    np.testing.assert_allclose(stresses, np.broadcast_to(expected_stress, stresses.shape), atol=1e-10)
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, expected_driver, atol=1e-10)


def test_three_dimensional_model_assembles_with_all_shear_components():
    params = _build_params("Stress-Based", "DegradedStress")
    params.dim, params.Lz, params.dx = 3, 1.0, 1.0
    mesh, physics, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((0.2*x[0]+0.04*x[1], -0.1*x[1]+0.03*x[2],
                                       0.02*x[2]+0.05*x[0])))
    u.x.scatter_forward()
    K, F = physics.Get_Kmat_Fvec(0)
    assert np.isfinite(K.norm()) and np.isfinite(F.norm())
    stress = mesh.Get_Field("sigma").x.array.reshape(-1, 6)
    assert np.all(np.abs(stress[:, 3:]) > 0)


def test_no_damage_model_and_live_timestep_are_supported():
    params = _build_params()
    params.ModelNames = ["Solid"]
    params.Models["Solid"]["DamageModel"] = None
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((0.2*x[0], -0.1*x[1])))
    u.x.scatter_forward()
    model.Update_Weak_Constants(0.0, 0.7, 0)
    first = mesh.Get_Field("sigma").x.array.copy()
    model.Update_Weak_Constants(0.0, 1.4, 0)
    assert not np.allclose(mesh.Get_Field("sigma").x.array, first)
    np.testing.assert_allclose(mesh.Get_Field("dam_elastic").x.array, 1)


@pytest.mark.full_model
@pytest.mark.parametrize("coupling", COUPLINGS)
@pytest.mark.parametrize("split", ["NoTension", "Stress-Based"])
@pytest.mark.parametrize("method", ["Newton", "NewtonContinuation"])
def test_coupled_damage_and_creep_time_solve(split, coupling, method):
    params = _build_params(split, coupling)
    params.Materials["Ice"]["f_t"] = 0.2
    params.Materials["Ice"]["PF_formulation"] = "AT1"
    params.Models["pf"]["Irreversible_Method"] = "Penalty"  # AT1 needs a bound on the phase field
    params.start_time, params.end_time, params.dt = 0.0, 0.2, 0.1
    params.Linear_Solver = LinearSolver.LU
    params.Step_Solvers = {0: method, 1: "Newton"}
    params.n_passes, params.max_it = 30, 30
    params.line_search = True
    params.rel_tol, params.abs_tol = 1e-7, 1e-9
    for name, boundary, field, value in (
        ("Left", "left", "u_x", 0.0), ("Bottom", "bottom", "u_y", 0.0),
        ("Right", "right", "u_x", 0.1),
    ):
        params.ModelNames.append(name)
        params.Models[name] = {"Type": "Boundary:PrescribedValue", "boundary": boundary,
                               "field": field, "value": value}
    # Energy-based AT2 has no threshold; strength-based AT1 uses f_t above.
    if split == "NoTension":
        params.Materials["Ice"]["PF_formulation"] = "AT2"
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solver = TimeSolver(params, physics)
    solver.run()
    assert solver.t >= params.end_time
    damage = mesh.Get_Field("phasefield").x.array
    assert np.all(damage >= -1e-8) and np.all(damage <= 1.0+1e-8)
    assert np.max(damage) > 0
    assert np.linalg.norm(mesh.Get_Field("Plastic_Strain").x.array) > 0
    assert np.isfinite(mesh.Get_Field("sigma").x.array).all()


@pytest.mark.parametrize("key, value", [
    ("CrackDrivingForce", "typo"), ("CreepCoupling", "typo"), ("DamageModel", "poros"),
])
def test_invalid_settings_are_rejected(key, value):
    params = _build_params()
    settings = params.Materials["Ice"] if key == "CrackDrivingForce" else params.Models["Solid"]
    settings[key] = value
    with pytest.raises(ValueError):
        _build(params)
