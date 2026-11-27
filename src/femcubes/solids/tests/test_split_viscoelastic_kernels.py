"""Constitutive contracts for selectively degraded viscoelasticity."""

import numpy as np
import pytest

from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import (
    SplitLinearElasticKernels, SplitViscoElasticKernels, SplitViscoElasticKernelsVariableProps,
    ViscoElasticKernels,
)


SPLITS = ("FullEnergy", "VolDev", "VolOnly", "Spectral", "NoTension", "Stress-Based")
IMPLICIT_COUPLINGS = ("EffectiveStress", "DegradedStress", "FullDamagedStress")
COUPLINGS = (*IMPLICIT_COUPLINGS, "PartialExplicit")


def _evaluate(strain, split, coupling="EffectiveStress", *, damage=0.2,
              viscosity=1.0e50, n=1.0, true_2d=False, old=None, dt=0.7, old_total=None):
    strain = np.asarray(strain, dtype=float).reshape(1, 6)
    old = np.zeros_like(strain) if old is None else np.asarray(old).reshape(1, 6)
    old_total = None if old_total is None else np.asarray(old_total).reshape(1, 6)
    return SplitViscoElasticKernels(viscosity, n, 4.0, 3.0, split, coupling, true_2d)(
        strain, old, np.array([damage]), dt, old_total_strains=old_total,
    )


def _stiffness(true_2d):
    active = [0, 1, 5] if true_2d else list(range(6))
    identity = np.array([1, 1, 0 if true_2d else 1, 0, 0, 0])
    return 4.0*np.outer(identity, identity) + 6.0*np.diag(np.isin(np.arange(6), active))


@pytest.mark.parametrize("split", SPLITS[:-1])
@pytest.mark.parametrize("true_2d", [False, True])
def test_stress_is_derivative_of_degraded_energy(split, true_2d):
    strain = np.array([0.17, -0.23, 0.04, 0.08, -0.03, 0.06])
    result = _evaluate(strain, split, true_2d=true_2d)
    h = 1.0e-6
    gradient = np.empty(6)
    for i in range(6):
        direction = np.eye(6)[i]*h
        plus = _evaluate(strain+direction, split, true_2d=true_2d)
        minus = _evaluate(strain-direction, split, true_2d=true_2d)
        gradient[i] = (0.2*(plus["driver"][0]-minus["driver"][0])
                       + plus["energy_minus"][0]-minus["energy_minus"][0])/(2*h)
    np.testing.assert_allclose(result["stress"][0], gradient, rtol=2e-8, atol=1e-9)
    np.testing.assert_allclose(result["driver"] + result["energy_minus"],
                               0.5*strain@_stiffness(true_2d)@strain)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("coupling", COUPLINGS)
@pytest.mark.parametrize("true_2d", [False, True])
@pytest.mark.parametrize("n", [1.0, 3.0])
def test_algorithmic_tangent_includes_creep_update(split, coupling, true_2d, n):
    strain = np.array([0.17, -0.23, 0.04, 0.08, -0.03, 0.06])
    old = np.array([-0.01, 0.006, 0.004, 0.002, 0.001, -0.005])
    kwargs = dict(viscosity=5.0, n=n, true_2d=true_2d, old=old)
    if coupling == "PartialExplicit":
        kwargs["old_total"] = np.array([0.13, -0.17, 0.03, 0.04, 0.02, -0.06])
    result = _evaluate(strain, split, coupling, **kwargs)
    h = 1e-6
    numerical = np.empty((6, 6))
    for i in range(6):
        direction = np.eye(6)[i]*h
        plus = _evaluate(strain+direction, split, coupling, **kwargs)["stress"][0]
        minus = _evaluate(strain-direction, split, coupling, **kwargs)["stress"][0]
        numerical[:, i] = (plus-minus)/(2*h)
    np.testing.assert_allclose(result["tangent"][0], numerical, rtol=1e-6, atol=2e-8)
    np.testing.assert_allclose(result["stress"], 0.2*result["stress_plus"]+result["stress_minus"])


@pytest.mark.parametrize("split", ["NoTension", "Spectral", "Stress-Based"])
@pytest.mark.parametrize("coupling", COUPLINGS)
@pytest.mark.parametrize("true_2d", [False, True])
@pytest.mark.parametrize("side", [-1, 1])
@pytest.mark.parametrize("dt", [0.5, 3600.0])
def test_glen_tangent_near_split_transition_at_ice_cliff_scales(split, coupling, true_2d, side, dt):
    youngs, nu, n, rate_factor = 9.5e9, 0.35, 3, 1e-24
    lame = youngs*nu/((1+nu)*(1-2*nu))
    shear = youngs/(2*(1+nu))
    d = 2 if true_2d else 3
    viscosity = (d**((n+1)/2)/(2*rate_factor))**(1/n)
    identity = np.array([1, 1, 0 if true_2d else 1, 0, 0, 0])
    projection = np.diag([1, 1, 0, 0, 0, 1]) if true_2d else np.eye(6)
    dev = projection-np.outer(identity, identity)/d
    stiffness = lame*np.outer(identity, identity)+2*shear*projection
    principal = np.array([-2e-4, side*2e-7, 0 if true_2d else 4e-4])
    if split == "NoTension":
        # The second compression direction activates at this relaxed-stress boundary.
        principal[1] += -lame*principal[0]/(2*shear+(d-1)*lame)
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    tensor = rotation@np.diag(principal)@rotation.T
    elastic = np.array([*np.diag(tensor), np.sqrt(2)*tensor[1, 2],
                        np.sqrt(2)*tensor[0, 2], np.sqrt(2)*tensor[0, 1]])
    if split == "Stress-Based":
        effective_stress = 2*shear*elastic
        active = [0, 1, 5] if true_2d else list(range(6))
        elastic[active] = np.linalg.solve(stiffness[np.ix_(active, active)], effective_stress[active])
    damage = np.array([1e-6])
    response = SplitLinearElasticKernels(lame, shear, split, true_2d)(elastic[None, :], damage)
    if coupling == "DegradedStress":
        creep_stress = response["stress"][0]
    else:
        creep_stress = stiffness@elastic
        if coupling == "FullDamagedStress":
            creep_stress *= damage[0]
    s = dev@creep_stress
    q = np.sqrt(d/2)*np.linalg.norm(s)
    flow = (d/2)*(q/viscosity)**n*s/q
    old = np.array([1e-5, -1e-5, 0, 0, 0, 2e-5])
    total = elastic+old+dt*flow
    kernel = SplitViscoElasticKernels(viscosity, n, lame, shear, split, coupling, true_2d)

    def evaluate(strain):
        return kernel(strain[None, :], old[None, :], damage, dt,
                      old_total_strains=total[None, :] if coupling == "PartialExplicit" else None)

    result = evaluate(total)
    np.testing.assert_allclose(total-result["strain_p"][0], elastic, atol=1e-14, rtol=1e-8)
    h = 1e-10
    numerical = np.column_stack([
        (evaluate(total+h*direction)["stress"][0]-evaluate(total-h*direction)["stress"][0])/(2*h)
        for direction in np.eye(6)
    ])
    relative_error = np.linalg.norm(result["tangent"][0]-numerical)/np.linalg.norm(numerical)
    assert relative_error < 2e-6


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("coupling", IMPLICIT_COUPLINGS)
@pytest.mark.parametrize("true_2d", [False, True])
def test_undamaged_response_recovers_original_j2_law(split, coupling, true_2d):
    strain = np.array([0.17, -0.23, 0.04, 0.08, -0.03, 0.06])
    stress, plastic, tangent, multiplier = np.zeros(6), np.zeros(6), np.zeros(36), np.zeros(1)
    ViscoElasticKernels(5.0, 3.0, 4.0, 3.0, true_2d)(
        strain, 0.7, False, np.ones(1), stress, plastic, np.zeros(6),
        tangent, multiplier, np.zeros(1),
    )
    result = _evaluate(strain, split, coupling, damage=1, viscosity=5.0, n=3, true_2d=true_2d)
    np.testing.assert_allclose(result["stress"][0], stress, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(result["strain_p"][0], plastic, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(result["tangent"][0], tangent.reshape(6, 6), rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(result["lambda"], multiplier, rtol=2e-7)


@pytest.mark.parametrize("split", SPLITS[1:])
@pytest.mark.parametrize("true_2d", [False, True])
def test_hydrostatic_compression_is_retained(split, true_2d):
    strain = np.array([-0.2, -0.2, -0.2, 0, 0, 0])
    if true_2d:
        strain[2] = 0
    result = _evaluate(strain, split, damage=1e-6, true_2d=true_2d)
    np.testing.assert_allclose(result["stress"][0], _stiffness(true_2d)@strain)
    np.testing.assert_allclose(result["driver"], 0, atol=1e-14)
    assert np.isfinite(result["tangent"]).all()


@pytest.mark.parametrize("split", ["Spectral", "NoTension", "Stress-Based"])
@pytest.mark.parametrize("value", [-0.2, 0.0, 0.2])
def test_repeated_eigenvalues_and_zero_state_have_finite_tangents(split, value):
    strain = np.array([value, value, value, 0, 0, 0])
    result = _evaluate(strain, split)
    assert all(np.isfinite(value).all() for value in result.values())
    if value != 0:
        h = 1e-6
        direction = np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1])
        numerical = (_evaluate(strain+h*direction, split)["stress"][0]
                     - _evaluate(strain-h*direction, split)["stress"][0])/(2*h)
        np.testing.assert_allclose(result["tangent"][0]@direction, numerical, atol=1e-8)


def test_stress_based_split_uses_principal_stress_signs():
    strain = np.array([0.15, -0.3, 0.06, 0.0, 0.0, 0.09])
    effective = _stiffness(False)@strain
    tensor = np.diag(effective[:3])
    tensor[0, 1] = tensor[1, 0] = effective[5]/np.sqrt(2)
    values, vectors = np.linalg.eigh(tensor)
    expected = vectors@np.diag(np.maximum(values, 0))@vectors.T
    result = _evaluate(strain, "Stress-Based")
    np.testing.assert_allclose(result["stress_plus"][0],
                               [expected[0,0], expected[1,1], expected[2,2], 0, 0, np.sqrt(2)*expected[0,1]])
    np.testing.assert_allclose(result["driver"], np.maximum(values, 0)@np.maximum(values, 0))


@pytest.mark.parametrize("strain", [
    [0.1, 0.2, 0.3, 0, 0, 0],
    [-0.2, 0.1, 0.3, 0, 0, 0],
    [-0.3, -0.1, 0.3, 0, 0, 0],
    [-0.3, -0.2, -0.1, 0, 0, 0],
])
def test_no_tension_energy_matches_piecewise_3d_formula(strain):
    e1, e2, e3 = strain[:3]
    lame, shear = 4.0, 3.0
    nu = lame/(2*(lame+shear))
    youngs = 2*shear*(1+nu)
    full = 0.5*np.array(strain)@_stiffness(False)@strain
    if e1 >= 0:
        expected = full
    elif e2+nu*e1 > 0:
        expected = full - 0.5*youngs*e1**2
    elif (1-nu)*e3+nu*(e2+e1) > 0:
        expected = youngs*((1-nu)*e3+nu*(e2+e1))**2/(2*(1+nu)*(1-2*nu)*(1-nu))
    else:
        expected = 0.0
    np.testing.assert_allclose(_evaluate(strain, "NoTension")["driver"], expected, atol=1e-14)


@pytest.mark.parametrize("true_2d", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.37])
@pytest.mark.parametrize("gap", [1e-12, 1e-14, 1e-16, 1e-17])
def test_no_tension_nearly_hydrostatic_compression_retains_exact_elastic_response(true_2d, angle, gap):
    youngs, nu = 9.5e9, 0.35
    lame = youngs*nu/((1+nu)*(1-2*nu))
    shear = youngs/(2*(1+nu))
    identity = np.array([1, 1, 0 if true_2d else 1, 0, 0, 0])
    projection = np.diag([1, 1, 0, 0, 0, 1]) if true_2d else np.eye(6)
    stiffness = lame*np.outer(identity, identity)+2*shear*projection
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    tensor = rotation@np.diag([-1e-3, -1e-3+gap, 0 if true_2d else -1e-3-gap])@rotation.T
    strain = np.array([*np.diag(tensor), 0, 0, np.sqrt(2)*tensor[0, 1]])
    result = SplitLinearElasticKernels(lame, shear, "NoTension", true_2d)(
        strain[None, :], np.array([1e-6]),
    )
    np.testing.assert_allclose(result["tangent"][0], stiffness, rtol=2e-14, atol=2e-14*shear)
    np.testing.assert_allclose(result["stress"][0], stiffness@strain, rtol=2e-14, atol=1e-12)
    np.testing.assert_allclose(result["driver"], 0.0, atol=1e-14)


@pytest.mark.parametrize("angle", [0.0, 0.37])
@pytest.mark.parametrize("gap", [1e-12, 1e-14, 1e-16, 1e-17])
def test_no_tension_nearly_equal_compression_directions_in_mixed_loading(angle, gap):
    youngs, nu = 9.5e9, 0.35
    lame, shear = youngs*nu/((1+nu)*(1-2*nu)), youngs/(2*(1+nu))
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    tensor = rotation@np.diag([-1e-3, -1e-3+gap, 2e-3])@rotation.T
    strain = np.array([*np.diag(tensor), 0, 0, np.sqrt(2)*tensor[0, 1]])
    g = 1e-6
    result = SplitLinearElasticKernels(lame, shear, "NoTension")(
        strain[None, :], np.array([g]),
    )
    # The compressed x/y plane is an isotropic 2-D block with relaxed lambda.
    identity = np.array([1, 1, 0])
    full = lame*np.outer(identity, identity)+2*shear*np.eye(3)
    retained = (2*shear*lame/(2*shear+lame))*np.outer(identity, identity)+2*shear*np.eye(3)
    expected = g*full+(1-g)*retained
    actual = result["tangent"][0][np.ix_([0, 1, 5], [0, 1, 5])]
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14*shear)


@pytest.mark.parametrize("true_2d", [False, True])
@pytest.mark.parametrize("angle", [0.0, 0.37])
@pytest.mark.parametrize("gap", [1e-12, 1e-17])
def test_spectral_nearly_hydrostatic_tension_has_only_degraded_stiffness(true_2d, angle, gap):
    youngs, nu = 9.5e9, 0.35
    lame, shear = youngs*nu/((1+nu)*(1-2*nu)), youngs/(2*(1+nu))
    identity = np.array([1, 1, 0 if true_2d else 1, 0, 0, 0])
    projection = np.diag([1, 1, 0, 0, 0, 1]) if true_2d else np.eye(6)
    stiffness = lame*np.outer(identity, identity)+2*shear*projection
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    tensor = rotation@np.diag([1e-3, 1e-3+gap, 0 if true_2d else 1e-3-gap])@rotation.T
    strain = np.array([*np.diag(tensor), 0, 0, np.sqrt(2)*tensor[0, 1]])
    g = 1e-6
    result = SplitLinearElasticKernels(lame, shear, "Spectral", true_2d)(
        strain[None, :], np.array([g]),
    )
    np.testing.assert_allclose(result["tangent"][0], g*stiffness, rtol=1e-8, atol=1e-12*shear)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("coupling", IMPLICIT_COUPLINGS)
def test_creep_obeys_selected_stress_law_even_at_low_stiffness(split, coupling):
    strain = np.array([0.17, -0.23, 0.04, 0.08, -0.03, 0.06])
    result = _evaluate(strain, split, coupling, damage=1e-6, viscosity=5.0, n=3)
    elastic = strain-result["strain_p"][0]
    stress = result["stress"][0] if coupling == "DegradedStress" else _stiffness(False)@elastic
    if coupling == "FullDamagedStress":
        stress *= 1e-6
    dev = stress.copy()
    dev[:3] -= dev[:3].mean()
    q = np.sqrt(1.5*(dev@dev))
    expected = 1.5*(q/5.0)**3/q*dev
    np.testing.assert_allclose(result["strain_p"][0]/0.7, expected, rtol=2e-7, atol=1e-12)
    np.testing.assert_allclose(result["strain_p"][0, :3].sum(), 0, atol=1e-12)


def test_creep_coupling_changes_damaged_relaxation():
    strain = [0.2, -0.2, 0, 0, 0, 0.1]
    effective = _evaluate(strain, "Spectral", viscosity=5.0)
    degraded = _evaluate(strain, "Spectral", "DegradedStress", viscosity=5.0)
    assert not np.allclose(effective["strain_p"], degraded["strain_p"])


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("true_2d", [False, True])
@pytest.mark.parametrize("n", [1.0, 3.0])
@pytest.mark.parametrize("damage", [0.0, 0.2])
def test_full_damage_scales_creep_without_rotating_flow_or_degrading_retained_stress(split, true_2d, n, damage):
    d = 2 if true_2d else 3
    identity = np.array([1, 1, 0 if true_2d else 1, 0, 0, 0])
    projection = np.diag([1, 1, 0, 0, 0, 1]) if true_2d else np.eye(6)
    dev = projection-np.outer(identity, identity)/d
    elastic = projection@np.array([0.17, -0.23, 0.04, 0.08, -0.03, 0.06])
    old = np.array([-0.01, 0.01, 0, 0, 0, -0.005])
    effective_dev = dev@(_stiffness(true_2d)@elastic)
    q = np.sqrt(d/2)*np.linalg.norm(effective_dev)
    effective_rate = (d/2)*(q/5.0)**n/q*effective_dev
    increment = 0.7*damage**n*effective_rate
    # Prescribe the end-step elastic strain and construct its backward-Euler
    # solution independently of the selected mechanical split.
    total = elastic+old+increment
    result = _evaluate(total, split, "FullDamagedStress", damage=damage,
                       viscosity=5.0, n=n, true_2d=true_2d, old=old)
    np.testing.assert_allclose(result["strain_p"][0], old+increment, rtol=1e-10, atol=1e-13)
    mechanical = SplitLinearElasticKernels(4.0, 3.0, split, true_2d)(elastic[None, :], np.array([damage]))
    np.testing.assert_allclose(result["stress"], mechanical["stress"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("true_2d", [False, True])
@pytest.mark.parametrize("n", [1.0, 3.0])
@pytest.mark.parametrize("damage", [1e-6, 0.2, 1.0])
def test_partial_explicit_relaxes_old_strain_implicitly_and_uses_current_split(split, true_2d, n, damage):
    d = 2 if true_2d else 3
    identity = np.array([1, 1, 0 if true_2d else 1, 0, 0, 0])
    projection = np.diag([1, 1, 0, 0, 0, 1]) if true_2d else np.eye(6)
    dev = projection-np.outer(identity, identity)/d
    relaxed_old = projection@np.array([0.12, -0.18, 0.06, 0.04, -0.02, 0.05])
    old_creep = np.array([-0.01, 0.01, 0, 0, 0, -0.005])
    s = dev@(_stiffness(true_2d)@relaxed_old)
    q = np.sqrt(d/2)*np.linalg.norm(s)
    rate = (d/2)*(q/5.0)**n/q*s
    updated_creep = old_creep+0.7*rate
    old_total = relaxed_old+updated_creep
    current_total = projection@np.array([0.21, -0.11, 0.03, 0.08, 0.02, 0.1])
    result = _evaluate(current_total, split, "PartialExplicit", damage=damage,
                       viscosity=5.0, n=n, true_2d=true_2d, old=old_creep, old_total=old_total)
    np.testing.assert_allclose(result["strain_p"][0], updated_creep, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result["lambda"], 0.7*np.sqrt(2/d)*np.linalg.norm(rate), rtol=1e-10)
    mechanical = SplitLinearElasticKernels(4.0, 3.0, split, true_2d)(
        (current_total-updated_creep)[None, :], np.array([damage]),
    )
    for field in ("stress", "stress_plus", "stress_minus", "driver", "energy_minus", "tangent"):
        np.testing.assert_allclose(result[field], mechanical[field], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("previous", [None, np.zeros(6), np.zeros((2, 6)), np.zeros((1, 5)), np.full((1, 6), np.nan)])
def test_partial_explicit_requires_valid_committed_total_strain(previous):
    kernel = SplitViscoElasticKernels(5, 3, 4, 3, "NoTension", "PartialExplicit")
    with pytest.raises(ValueError, match="old_total_strains|Strains must be finite"):
        kernel(np.zeros((1, 6)), np.zeros((1, 6)), np.ones(1), 0.7, old_total_strains=previous)


@pytest.mark.parametrize("kwargs", [
    {"split": "typo"}, {"creep_coupling": "typo"}, {"viscosity": 0},
    {"n_exp": 0.5}, {"lame": -1}, {"shear": 0}, {"viscosity": np.inf},
])
def test_invalid_kernel_settings_are_rejected(kwargs):
    arguments = dict(viscosity=5, n_exp=3, lame=4, shear=3, split="NoTension")
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        SplitViscoElasticKernels(**arguments)


@pytest.mark.parametrize("damage, dt", [(-0.1, 1), (1.1, 1), (np.nan, 1), (0.2, 0), (0.2, np.inf)])
def test_invalid_state_inputs_are_rejected(damage, dt):
    with pytest.raises(ValueError):
        _evaluate(np.zeros(6), "NoTension", damage=damage, dt=dt)


@pytest.mark.parametrize("coupling", COUPLINGS)
def test_variable_split_kernel_accepts_a_rank_without_integration_points(coupling):
    kernel = SplitViscoElasticKernelsVariableProps(3, "NoTension", coupling)
    result = kernel(np.empty((0, 6)), np.empty((0, 6)), np.empty(0), 1.0,
                    np.empty(0), np.empty(0), np.empty(0),
                    old_total_strains=np.empty((0, 6)) if coupling == "PartialExplicit" else None)
    assert result["stress"].shape == (0, 6)
    assert result["tangent"].shape == (0, 6, 6)
    assert result["driver"].shape == (0,)


@pytest.mark.parametrize("name, value", [
    ("lame", np.ones(2)), ("shear", np.ones((1, 1))),
    ("viscosity", np.array(5.0)), ("lame", np.array([-1.0])),
    ("shear", np.array([0.0])), ("viscosity", np.array([np.nan])),
])
def test_variable_split_kernel_rejects_invalid_material_arrays(name, value):
    material = {"lame": np.array([4.0]), "shear": np.array([3.0]), "viscosity": np.array([5.0])}
    material[name] = value
    kernel = SplitViscoElasticKernelsVariableProps(3, "NoTension")
    with pytest.raises(ValueError, match="Material properties|requires finite"):
        kernel(np.zeros((1, 6)), np.zeros((1, 6)), np.ones(1), 1.0, **material)
