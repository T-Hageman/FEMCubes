"""Energy-driver equivalence without changing the original mechanical laws."""

import numpy as np
import pytest
import ufl

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.solids.tests.test_split_viscoelastic import _build_params, _integral


ENERGY_SPLITS = ("FullEnergy", "VolDev", "VolOnly", "Spectral", "NoTension")


@pytest.mark.parametrize("kind", ["LinearElastic", "ViscoElastic", "SplitLinearElastic", "SplitViscoElastic"])
@pytest.mark.parametrize("split", [None, "Spectral"])
def test_driver_selection_uses_referenced_material_and_preserves_default(kind, split):
    params = _build_params("NoTension")
    params.Materials["OtherIce"] = dict(params.Materials["Ice"])
    if split is None:
        del params.Materials["OtherIce"]["CrackDrivingForce"]
    else:
        params.Materials["OtherIce"]["CrackDrivingForce"] = split
    params.Models["Solid"].update(Type="Solid:"+kind, Material="OtherIce")
    params.Models["Solid"].pop("CreepCoupling")
    params.Models["pf"]["Material"] = "OtherIce"
    physics = Physics(Mesh(params), params)
    assert physics.models["Solid"].DrivingForce_Type == (split or "Stress-Based")


def _build(kind, split, dim, true_2d, *, consistent=False, order=1, formulation="AT2"):
    params = _build_params(split, true_2d=true_2d)
    params.Models["Solid"]["Type"] = "Solid:" + ("Split" if consistent else "") + kind
    params.Models["Solid"].pop("CreepCoupling")
    params.Materials["Ice"].update(Viscosity=1e50, Inertia=False, PF_formulation=formulation)
    if formulation in ("AT1", "HO-AT1"):  # history irreversibility needs AT2/HO-AT2
        params.Models["pf"]["Irreversible_Method"] = "Penalty"
    params.Field_Orders["u"] = order
    params.dim = dim
    if dim == 3:
        params.Lz, params.dx = 1.0, 1.0
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    mesh.Constant_Field("phasefield", 0.6)
    return params, mesh, physics, physics.models["Solid"]


def _set_strain(mesh, tensor, dim):
    displacement = mesh.Get_Field("u")
    displacement.interpolate(lambda x: tensor[:dim, :dim]@x[:dim])
    displacement.x.scatter_forward()


@pytest.mark.parametrize("kind", ["LinearElastic", "ViscoElastic"])
@pytest.mark.parametrize("split", (*ENERGY_SPLITS, "Stress-Based"))
@pytest.mark.parametrize("dim, true_2d", [(2, False), (2, True), (3, False)])
def test_energy_drivers_match_consistent_models_at_same_elastic_strain(kind, split, dim, true_2d):
    _, mesh, _, original = _build(kind, split, dim, true_2d)
    _, split_mesh, _, consistent = _build(kind, split, dim, true_2d, consistent=True)
    strains = [np.diag([sign*0.2]*3) for sign in (-1, 1)]
    strains.append(np.array([[0.17, 0.06, -0.03], [0.06, -0.23, 0.04], [-0.03, 0.04, 0.08]]))
    for total in strains:
        elastic = total.copy()
        if dim == 2:
            elastic[2, :] = elastic[:, 2] = 0
        for current_mesh in (mesh, split_mesh):
            _set_strain(current_mesh, total, dim)
            if kind == "ViscoElastic":
                # Include an out-of-plane viscous strain in plane strain.
                old = np.array([0.01, -0.02, 0.01, 0, 0, 0.005])
                if true_2d:
                    old[2] = 0
                for field in ("Plastic_Strain", "Plastic_Strain_Old"):
                    values = current_mesh.Get_Field(field)
                    values.x.array.reshape(-1, 6)[:] = old
                    values.x.scatter_forward()
        if kind == "ViscoElastic":
            elastic -= np.array([[old[0], old[5]/np.sqrt(2), 0],
                                 [old[5]/np.sqrt(2), old[1], 0], [0, 0, old[2]]])
        original.Update_Weak_Constants(0.0, 0.7, 0)
        consistent.Update_Weak_Constants(0.0, 0.7, 0)
        driver = mesh.Get_Field("ElasticEnergy").x.array
        np.testing.assert_allclose(driver, split_mesh.Get_Field("ElasticEnergy").x.array, atol=1e-12)
        # Independent analytic checks catch regressions in the shared definitions.
        d = 2 if true_2d else 3
        e = elastic[:d, :d]
        volume, bulk = np.trace(e), 4+6/d
        full = 2*volume**2+3*np.sum(e*e)
        if split == "FullEnergy":
            expected = full
        elif split == "VolDev":
            expected = 0.5*bulk*max(volume, 0)**2+3*np.sum((e-volume*np.eye(d)/d)**2)
        elif split == "VolOnly":
            expected = 0.5*bulk*max(volume, 0)**2
        elif split == "Spectral":
            expected = 2*max(volume, 0)**2+3*np.sum(np.maximum(np.linalg.eigvalsh(e), 0)**2)
        elif split == "Stress-Based":
            effective = 4*volume*np.eye(d)+6*e
            expected = np.sum(np.maximum(np.linalg.eigvalsh(effective), 0)**2)/4-1
            if kind == "LinearElastic":
                expected = max(expected, 0)
        else:
            continue  # NoTension's piecewise formula is checked in the kernel tests.
        np.testing.assert_allclose(driver, expected, atol=1e-12)


@pytest.mark.parametrize("kind", ["LinearElastic", "ViscoElastic"])
@pytest.mark.parametrize("split", ENERGY_SPLITS)
def test_original_models_keep_full_stress_degradation_in_compression(kind, split):
    params, mesh, _, model = _build(kind, split, 2, True)
    _set_strain(mesh, -0.2*np.eye(3), 2)
    _, residual = model.assemble_KF(0)
    g = params.Materials["Ice"]["kappa"]+(1-params.Materials["Ice"]["kappa"])*0.4**2
    np.testing.assert_allclose(_integral(ufl.action(residual, mesh.Get_Field("u"))), 1.12*g)
    energy = mesh.Get_Field("ElasticEnergy").x.array
    np.testing.assert_allclose(energy, 0.56 if split == "FullEnergy" else 0.0, atol=1e-12)


@pytest.mark.parametrize("split, expected", [
    ("FullEnergy", 0.17), ("VolDev", 0.17), ("VolOnly", 0.035),
    ("Spectral", 0.14), ("NoTension", 0.128),
])
def test_viscoelastic_driver_accepts_explicit_elastic_strain_without_updating_stress_or_history(split, expected):
    _, mesh, _, model = _build("ViscoElastic", split, 2, True)
    before = {name: mesh.Get_Field(name).x.array.copy()
              for name in ("sigma", "Plastic_Strain", "Plastic_Strain_Old")}
    model.Update_PhaseField_Fields(ufl.as_vector([0.2, -0.1, 0, 0, 0, 0]))
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, expected, atol=1e-12)
    for name, values in before.items():
        np.testing.assert_array_equal(mesh.Get_Field(name).x.array, values)


@pytest.mark.parametrize("kind", ["LinearElastic", "ViscoElastic"])
@pytest.mark.parametrize("split", ["NoTension", "Stress-Based"])
def test_spatially_varying_energy_matches_at_each_quadrature_point(kind, split):
    drivers = []
    for consistent in (False, True):
        _, mesh, _, model = _build(kind, split, 2, False, consistent=consistent, order=2)
        displacement = mesh.Get_Field("u")
        displacement.interpolate(lambda x: np.vstack((0.2*x[0]**2+0.1*x[0]*x[1], -0.1*x[1]**2)))
        displacement.x.scatter_forward()
        model.Update_Weak_Constants(0.0, 0.7, 0)
        drivers.append(mesh.Get_Field("ElasticEnergy").x.array.copy())
    assert np.ptp(drivers[0]) > 0
    np.testing.assert_allclose(*drivers, atol=1e-12)


@pytest.mark.parametrize("kind", ["LinearElastic", "ViscoElastic"])
@pytest.mark.parametrize("split", ["NoTension", "Stress-Based"])
def test_energy_driver_interpolation_survives_remeshing(kind, split):
    from femcubes.core.physics.tests.test_physics_remesh import _refine_live_mesh
    _, mesh, physics, model = _build(kind, split, 2, False)
    _set_strain(mesh, np.diag([0.2, -0.1, 0]), 2)
    model.assemble_KF(0)
    expected = mesh.Get_Field("ElasticEnergy").x.array[0]
    assert _refine_live_mesh(mesh).installed
    physics.Post_Remesh()
    model.assemble_KF(0)
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, expected, atol=1e-12)


@pytest.mark.parametrize("kind", ["LinearElastic", "ViscoElastic"])
@pytest.mark.parametrize("formulation", ["AT1", "AT2", "HO-AT1", "HO-AT2"])
@pytest.mark.parametrize("strain", [0.01, 0.3])
def test_shared_stress_driver_preserves_threshold_and_normalization(kind, formulation, strain):
    strength = 2*(14*strain)**2/4
    if formulation in ("AT2", "HO-AT2"):
        expected = max(strength-1, 0) if kind == "LinearElastic" else strength-1
    else:
        expected = 0.75*strength
    for consistent in (False, True):
        _, mesh, _, model = _build(kind, "Stress-Based", 2, True,
                                   consistent=consistent, formulation=formulation)
        _set_strain(mesh, strain*np.eye(3), 2)
        model.Update_Weak_Constants(0.0, 0.7, 0)
        np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, expected, atol=1e-12)


@pytest.mark.parametrize("kind", ["SplitLinearElastic", "SplitViscoElastic"])
def test_stress_driver_without_phasefield_reports_total_elastic_energy(kind):
    params = _build_params("Stress-Based", true_2d=True)
    params.Models["Solid"].update(Type="Solid:"+kind, DamageModel=None)
    params.Models["Solid"].pop("CreepCoupling")
    params.ModelNames = ["Solid"]
    params.Materials["Ice"]["Viscosity"] = 1e50
    for key in ("Gc", "ell", "f_t"):
        del params.Materials["Ice"][key]
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    _set_strain(mesh, np.diag([0.2, -0.1, 0]), 2)
    physics.models["Solid"].Update_Weak_Constants(0.0, 0.7, 0)
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, 0.17, atol=1e-12)
