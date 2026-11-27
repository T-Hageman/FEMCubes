"""Constitutive and finite-element contracts for split linear elasticity."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.solids.linear_elastic import LinearElastic
from femcubes.solids.split_linear_elastic import SplitLinearElastic
from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import SplitLinearElasticKernels
from femcubes.solids.tests.test_split_viscoelastic import _build_params as _visco_params, _integral
from femcubes.solids.tests.test_split_viscoelastic_kernels import SPLITS, _stiffness


def _build_params(split="NoTension", dim=2, true_2d=False):
    params = _visco_params(split, true_2d=true_2d)
    params.Models["Solid"]["Type"] = "Solid:SplitLinearElastic"
    del params.Models["Solid"]["CreepCoupling"]
    for key in ("Visc_Type", "Viscosity", "Bulk"):
        del params.Materials["Ice"][key]
    params.dim = dim
    if dim == 3:
        params.Lz, params.dx = 1.0, 1.0
    return params


def _build(params):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    if params.Models["Solid"].get("DamageModel") == "PhaseField":
        mesh.Constant_Field("phasefield", 0.6)
    return mesh, physics, physics.models["Solid"]


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("true_2d", [False, True])
def test_elastic_kernel_stress_tangent_and_energy_derivatives(split, true_2d):
    kernel = SplitLinearElasticKernels(4.0, 3.0, split, true_2d)
    strain = np.array([[0.17, -0.23, 0.04, 0.08, -0.03, 0.06]])
    damage = np.array([0.2])
    result = kernel(strain, damage)
    gradient, tangent = np.zeros(6), np.zeros((6, 6))
    h = 1e-6
    for i in range(6):
        direction = h*np.eye(6)[i]
        plus, minus = kernel(strain+direction, damage), kernel(strain-direction, damage)
        tangent[:, i] = (plus["stress"][0]-minus["stress"][0])/(2*h)
        gradient[i] = (0.2*(plus["driver"][0]-minus["driver"][0])
                       + plus["energy_minus"][0]-minus["energy_minus"][0])/(2*h)
    np.testing.assert_allclose(result["tangent"][0], tangent, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result["stress"], 0.2*result["stress_plus"]+result["stress_minus"])
    if split != "Stress-Based":
        np.testing.assert_allclose(result["stress"][0], gradient, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(result["driver"][0]+result["energy_minus"][0],
                                   0.5*strain[0]@_stiffness(true_2d)@strain[0])


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("dim, true_2d", [(2, False), (2, True), (3, False)])
def test_assembled_tangent_including_inertia_and_damping(split, dim, true_2d):
    params = _build_params(split, dim, true_2d)
    params.Materials["Ice"].update(Inertia=True, LE_Damping=0.4)
    mesh, _, model = _build(params)
    assert isinstance(model, LinearElastic) and isinstance(model, SplitLinearElastic)
    with pytest.raises(ValueError, match="not found"):
        mesh.Get_Field("Plastic_Strain")
    u = mesh.Get_Field("u")
    base = fem.Function(u.function_space)
    base.interpolate(lambda x: np.vstack((0.17*x[0]+0.04*x[1], -0.23*x[1]+0.04*x[0],
                                         0.05*x[2]+0.03*x[0]))[:dim])
    probe = fem.Function(u.function_space)
    probe.interpolate(lambda x: np.vstack((0.3*x[0]+0.2*x[1], -0.1*x[1]+0.1*x[0],
                                          0.2*x[2]-0.03*x[1]))[:dim])
    probe.x.scatter_forward()

    def evaluate(delta):
        u.x.array[:] = base.x.array + delta*probe.x.array
        u.x.scatter_forward()
        return model.assemble_KF(0)

    K, F = evaluate(0)
    analytic = _integral(ufl.action(ufl.action(K, probe), probe))
    h = 1e-6
    _, F = evaluate(h)
    plus = _integral(ufl.action(F, probe))
    _, F = evaluate(-h)
    minus = _integral(ufl.action(F, probe))
    np.testing.assert_allclose(analytic, (plus-minus)/(2*h), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("split", SPLITS)
def test_fully_damaged_compression_retention_and_tension_degradation(split):
    params = _build_params(split, true_2d=True)
    mesh, _, model = _build(params)
    mesh.Constant_Field("phasefield", 1.0)
    u = mesh.Get_Field("u")
    for sign in (-1.0, 1.0):
        u.interpolate(lambda x: sign*0.2*x[:2])
        u.x.scatter_forward()
        _, F = model.assemble_KF(0)
        factor = 1e-3 if sign > 0 or split == "FullEnergy" else 1.0
        np.testing.assert_allclose(_integral(ufl.action(F, u)), 1.12*factor, rtol=1e-10)


def test_density_body_forces_and_inertia_remain_active_after_damage_changes():
    params = _build_params()
    params.gravity = -2.0
    params.Materials["Ice"].update(Density=3.0, Inertia=True)
    mesh, _, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.full(x.shape[1], 0.2))))
    u.x.scatter_forward()
    probe = fem.Function(u.function_space)
    probe.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.ones(x.shape[1]))))
    probe.x.scatter_forward()
    K, F = model.assemble_KF(0)
    mass = 3.0/(params.Newmark_Beta*params.dt**2)
    for damage in (0, 0.8, 1):
        mesh.Constant_Field("phasefield", damage)
        model.Update_Weak_Constants(0, params.dt, 0)
        assert model.d_rho == 1.0
        np.testing.assert_allclose(_integral(ufl.action(F, probe)), 6.0+0.2*mass)
        np.testing.assert_allclose(_integral(ufl.action(ufl.action(K, probe), probe)), mass)


@pytest.mark.parametrize("formulation", ["AT1", "AT2", "HO-AT1", "HO-AT2"])
@pytest.mark.parametrize("strain", [0.01, 0.3])
def test_strength_driver_retains_linear_elastic_threshold(formulation, strain):
    params = _build_params("Stress-Based", true_2d=True)
    params.Models["pf"]["Type"] = "Solid:PhaseField"
    params.Materials["Ice"]["PF_formulation"] = formulation
    if formulation in ("AT1", "HO-AT1"):  # history irreversibility needs AT2/HO-AT2
        params.Models["pf"]["Irreversible_Method"] = "Penalty"
    mesh, _, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: strain*x[:2])
    u.x.scatter_forward()
    model.assemble_KF(0)
    strength = 2*(14*strain)**2 / params.Materials["Ice"]["f_t"]**2
    expected = max(strength-1, 0) if formulation in ("AT2", "HO-AT2") else 0.75*strength
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, expected)


@pytest.mark.parametrize("true_2d", [False, True])
@pytest.mark.parametrize("damage_model", [None, "PhaseField"])
def test_undamaged_response_matches_original_elasticity(true_2d, damage_model):
    params = _build_params(true_2d=true_2d)
    params.Models["Solid"]["DamageModel"] = damage_model
    if damage_model is None:
        params.ModelNames = ["Solid"]
    mesh, _, model = _build(params)
    if damage_model:
        mesh.Constant_Field("phasefield", 0.0)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((0.17*x[0]+0.04*x[1], -0.23*x[1]+0.04*x[0])))
    u.x.scatter_forward()
    K, F = model.assemble_KF(0)
    strain = np.array([0.17, -0.23, 0, 0, 0, np.sqrt(2)*0.04])
    expected = _stiffness(true_2d)@strain
    sigma = mesh.Get_Field("sigma").x.array.reshape(-1, 6)
    np.testing.assert_allclose(sigma, np.broadcast_to(expected, sigma.shape), atol=1e-12)
    np.testing.assert_allclose(_integral(ufl.action(F, u)), strain@expected)
    np.testing.assert_allclose(_integral(ufl.action(ufl.action(K, u), u)), strain@expected)


def test_commit_reset_and_live_remesh_preserve_elastic_response():
    from femcubes.core.physics.tests.test_physics_remesh import _refine_live_mesh
    params = _build_params()
    mesh, physics, model = _build(params)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((0.2*x[0], -0.1*x[1])))
    u.x.scatter_forward()
    physics.Get_Kmat_Fvec(0)
    # Commit must refresh stress if the last phase-field solve changed damage.
    mesh.Constant_Field("phasefield", 0.9)
    physics.commit()
    expected_stress = mesh.Get_Field("sigma").x.array.reshape(-1, 6)[0].copy()
    expected_driver = mesh.Get_Field("ElasticEnergy").x.array[0]
    np.testing.assert_allclose(mesh.Get_Field("p").x.array, -0.4)
    mesh.Constant_Field("u", 0)
    physics.Reset_Step()
    physics.Get_Kmat_Fvec(0)
    sigma = mesh.Get_Field("sigma").x.array.reshape(-1, 6)
    np.testing.assert_allclose(sigma, np.broadcast_to(expected_stress, sigma.shape), atol=1e-12)
    assert _refine_live_mesh(mesh).installed
    physics.Post_Remesh()
    K, F = physics.Get_Kmat_Fvec(0)
    assert np.isfinite(K.norm()) and np.isfinite(F.norm())
    sigma = mesh.Get_Field("sigma").x.array.reshape(-1, 6)
    np.testing.assert_allclose(sigma, np.broadcast_to(expected_stress, sigma.shape), atol=1e-10)
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, expected_driver, atol=1e-10)


@pytest.mark.full_model
@pytest.mark.parametrize("damage", [None, 1.0])
def test_gravity_time_solve_matches_linear_elastic_solution(damage):
    from femcubes.solids.tests.test_linear_elastic import _build_params as gravity_params
    params = gravity_params()
    params.Models["Solid"]["Type"] = "Solid:SplitLinearElastic"
    params.Materials["Ice"]["CrackDrivingForce"] = "NoTension"
    del params.Materials["Ice"]["Bulk"]
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
    TimeSolver(params, physics).run()
    material = params.Materials["Ice"]
    expected = material["Density"]*params.gravity*params.Ly**2/(2*(material["Lame"]+2*material["Shear"]))
    local_displacement = mesh.Get_Field("u").sub(1).collapse().x.array.min(initial=np.inf)
    displacement = MPI.COMM_WORLD.allreduce(local_displacement, op=MPI.MIN)
    np.testing.assert_allclose(displacement, expected, rtol=1e-2)


@pytest.mark.parametrize("key, value", [
    ("CrackDrivingForce", "typo"), ("DamageModel", "poros"), ("CreepCoupling", "EffectiveStress"),
])
def test_invalid_settings_are_rejected(key, value):
    params = _build_params()
    settings = params.Materials["Ice"] if key == "CrackDrivingForce" else params.Models["Solid"]
    settings[key] = value
    with pytest.raises(ValueError):
        _build(params)


@pytest.mark.parametrize("lame, shear", [(-1, 3), (4, 0), (np.nan, 3)])
def test_invalid_kernel_moduli_are_rejected(lame, shear):
    with pytest.raises(ValueError):
        SplitLinearElasticKernels(lame, shear, "NoTension")


@pytest.mark.parametrize("strain, damage", [
    (np.zeros(6), np.ones(1)), (np.zeros((1, 6)), np.ones(2)),
    (np.full((1, 6), np.nan), np.ones(1)), (np.zeros((1, 6)), np.array([-0.1])),
])
def test_invalid_kernel_arrays_are_rejected(strain, damage):
    with pytest.raises(ValueError):
        SplitLinearElasticKernels(4, 3, "NoTension")(strain, damage)
