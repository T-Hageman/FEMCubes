"""Tests for the opening-only r2 criterion ("r2_2L_PSD") in the viscoelastic cohesive kernel.

The principal-space update of ``femcubes.solids._principal_cve`` solves the same point
problem in the principal strains and serves as the independent reference.
"""

import math

import numpy as np
import pytest

from femcubes.solids._principal_cve import PrincipalCVEUpdate
from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    CohesiveViscoElasticKernels,
    cohesive_multiplier_slots,
    get_cohesive_num_lambdas,
)


YOUNGS = 9.5e9
POISSON = 0.35
SHEAR = YOUNGS / (2.0 * (1.0 + POISSON))
BULK = YOUNGS / (3.0 * (1.0 - 2.0 * POISSON))
LAME = 2.0 * SHEAR * POISSON / (1.0 - 2.0 * POISSON)
F_T = 0.4e6
F_S = 1.5e6
RESIDUAL_DAMAGE = 1.0e-3
GLEN_A = 1.0e-24
GLEN_VISCOSITY = (9.0 / (2.0 * GLEN_A)) ** (1.0 / 3.0)
CREEP_DT = 2.0e3
ONSET_ETA = 1.0e-9
SQRT2 = math.sqrt(2.0)


def _kernel(failure_type="r2_2L_PSD", viscosity=0.0, plane_stress=False, f_t=F_T, f_s=F_S, true_2d=False):
    return CohesiveViscoElasticKernels(
        BULK, LAME, SHEAR, viscosity, 3.0,
        f_t, f_s, 1.0, 0.0, F_S,
        RESIDUAL_DAMAGE, failure_type, plane_stress, true_2d,
    )


def _reference(viscosity=0.0, plane_stress=False):
    return PrincipalCVEUpdate(
        lame=LAME, shear=SHEAR, f_t=F_T, f_s=F_S, kappa=RESIDUAL_DAMAGE,
        viscosity=viscosity, n_exp=3.0 if viscosity > 0.0 else 1.0, plane_stress=plane_stress,
    )


def _update(update, strain, damage, slots, creep_strain_old=None, dt=CREEP_DT):
    """Return the state of ``update`` for every row of ``strain``."""
    n = strain.shape[0]
    widths = (("stress", 6), ("eta", 6), ("creep_strain", 6), ("tangent", 36), ("multipliers", slots))
    state = {name: np.zeros(n * width) for name, width in widths}
    state["strain"] = strain.copy().ravel()     # plane stress writes e_zz back
    old = np.zeros(n * 6) if creep_strain_old is None else creep_strain_old.copy().ravel()
    update(
        state["strain"], dt, damage.copy(),
        state["stress"], state["eta"], np.zeros(n * 6), state["creep_strain"], old,
        state["tangent"], np.zeros(n), np.zeros(n), np.zeros(n), state["multipliers"], np.zeros(n * slots),
    )
    state = {name: value.reshape(n, -1) for name, value in state.items()}
    state["tangent"] = state["tangent"].reshape(n, 6, 6)
    return state


def _mandel(tensor):
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                     SQRT2 * tensor[1, 2], SQRT2 * tensor[0, 2], SQRT2 * tensor[0, 1]])


def _tensor(vector):
    return np.array([[vector[0], vector[5] / SQRT2, vector[4] / SQRT2],
                     [vector[5] / SQRT2, vector[1], vector[3] / SQRT2],
                     [vector[4] / SQRT2, vector[3] / SQRT2, vector[2]]])


def _random_strains(rng, n, amplitude):
    strains = np.empty((n, 6))
    for i in range(n):
        sample = rng.normal(size=(3, 3))
        strains[i] = _mandel(0.5 * (sample + sample.T)) * amplitude
    return strains


def _strain_of(principal_stress):
    """Elastic strain of a stress given by its three principal values."""
    stiffness = LAME * np.ones((3, 3)) + 2.0 * SHEAR * np.eye(3)
    strain = np.zeros(6)
    strain[:3] = np.linalg.solve(stiffness, np.asarray(principal_stress, dtype=float))
    return strain


def _states(rng, n, in_plane=False):
    """Strains around the strength surface, old creep strains and damage values."""
    strain = _random_strains(rng, n, 1.5e-4)
    strain[: n // 2, :3] += rng.uniform(0.0, 3.0e-4, size=(n // 2, 1))
    creep_old = _random_strains(rng, n, 2.0e-5)
    creep_old[:, :3] -= creep_old[:, :3].mean(axis=1, keepdims=True)
    if in_plane:
        strain[:, 2:5] = 0.0
        creep_old[:, 3:5] = 0.0
    return strain, creep_old, 10.0 ** rng.uniform(-2.0, 0.0, size=n)


def _stress_scale(state, damage):
    return np.maximum(np.abs(state["stress"]).max(axis=1), np.clip(damage, RESIDUAL_DAMAGE, 1.0) * F_T)


def test_three_multipliers_per_point():
    assert get_cohesive_num_lambdas("r2_2L_PSD") == 3
    assert cohesive_multiplier_slots("r2_2L_PSD") == 3
    # the other types keep their two stored values
    for failure_type in ("r1", "r1_DP", "r2", "r2_DP", "r2_2L", "r2_2L_DP"):
        assert cohesive_multiplier_slots(failure_type) == 2


@pytest.mark.parametrize("viscosity", [0.0, GLEN_VISCOSITY])
def test_state_equals_principal_space_solution(viscosity):
    rng = np.random.default_rng(11)
    n = 600
    strain, creep_old, damage = _states(rng, n)
    if viscosity == 0.0:
        creep_old[:] = 0.0
    ours = _update(_kernel(viscosity=viscosity), strain, damage, 3, creep_old)
    ref = _update(_reference(viscosity), strain, damage, 2, creep_old)

    opening = np.linalg.norm(ref["eta"], axis=1) > ONSET_ETA
    assert 100 < np.count_nonzero(opening) < n - 100
    assert np.array_equal(np.linalg.norm(ours["eta"], axis=1) > ONSET_ETA, opening)
    scale = _stress_scale(ref, damage)
    assert (np.abs(ours["stress"] - ref["stress"]).max(axis=1) <= 1.0e-5 * scale).all()
    assert np.abs(ours["eta"] - ref["eta"]).max() <= 1.0e-5 * np.abs(ref["eta"]).max()
    assert np.abs(ours["creep_strain"] - ref["creep_strain"]).max() <= 1.0e-6 * np.abs(strain).max()
    assert np.abs(ours["tangent"] - ref["tangent"].reshape(n, 6, 6)).max() <= 1.0e-3 * YOUNGS


def test_cohesive_strain_is_positive_semidefinite_and_coaxial_with_the_strain():
    rng = np.random.default_rng(12)
    strain, _, damage = _states(rng, 300)
    state = _update(_kernel(), strain, damage, 3)
    opening = np.linalg.norm(state["eta"], axis=1) > ONSET_ETA
    assert np.count_nonzero(opening) > 100
    for i in np.flatnonzero(opening):
        eta, total = _tensor(state["eta"][i]), _tensor(strain[i])
        assert np.linalg.eigvalsh(eta).min() >= -1.0e-9 * np.abs(eta).max()
        assert np.abs(eta @ total - total @ eta).max() <= 1.0e-9 * np.abs(eta).max() * np.abs(total).max()
    assert (state["multipliers"] >= 0.0).all()


@pytest.mark.parametrize("viscosity", [0.0, GLEN_VISCOSITY])
def test_tangent_matches_finite_differences(viscosity):
    rng = np.random.default_rng(13)
    n = 200
    strain, creep_old, damage = _states(rng, n)
    strain[:40, 2:5] = 0.0                 # plane strain
    creep_old[:40, 3:5] = 0.0
    if viscosity == 0.0:
        creep_old[:] = 0.0
    kernel = _kernel(viscosity=viscosity)
    tangent = _update(kernel, strain, damage, 3, creep_old)["tangent"]
    step = 1.0e-9
    difference = np.empty((n, 6, 6))
    for k in range(6):
        up, down = strain.copy(), strain.copy()
        up[:, k] += step
        down[:, k] -= step
        difference[:, :, k] = (_update(kernel, up, damage, 3, creep_old)["stress"]
                               - _update(kernel, down, damage, 3, creep_old)["stress"]) / (2.0 * step)
    error = np.abs(tangent - difference).reshape(n, -1).max(axis=1)
    # a finite difference across the strength surface or a change of the open directions is not a derivative
    assert np.quantile(error, 0.95) <= 1.0e-5 * YOUNGS
    # symmetric up to the convergence tolerance of the point iteration
    assert np.abs(tangent - tangent.transpose(0, 2, 1)).max() <= 1.0e-5 * YOUNGS


@pytest.mark.parametrize("viscosity", [0.0, GLEN_VISCOSITY])
@pytest.mark.parametrize("relative_gap", [0.0, 1.0e-12, 1.0e-9, 1.0e-6])
def test_tangent_at_repeated_principal_strains_is_the_limit_of_the_isotropic_response(viscosity, relative_gap):
    rng = np.random.default_rng(14)
    n = 120
    principal = rng.normal(size=(n, 3)) * 1.5e-4 + rng.uniform(0.0, 3.0e-4, size=(n, 1))
    principal[:, 1] = principal[:, 0] * (1.0 + relative_gap)
    strain = np.empty((n, 6))
    for i in range(n):
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0] if i >= 40 else np.eye(3)
        strain[i] = _mandel(rotation @ np.diag(principal[i]) @ rotation.T)
    damage = 10.0 ** rng.uniform(-2.0, 0.0, size=n)
    ours = _update(_kernel(viscosity=viscosity), strain, damage, 3)
    ref = _update(_reference(viscosity), strain, damage, 2)
    assert np.count_nonzero(np.linalg.norm(ref["eta"], axis=1) > ONSET_ETA) > 60
    assert (np.abs(ours["stress"] - ref["stress"]).max(axis=1) <= 1.0e-5 * _stress_scale(ref, damage)).all()
    assert np.abs(ours["tangent"] - ref["tangent"].reshape(n, 6, 6)).max() <= 1.0e-3 * YOUNGS


@pytest.mark.parametrize("viscosity", [0.0, GLEN_VISCOSITY])
def test_plane_stress_state_equals_principal_space_solution(viscosity):
    rng = np.random.default_rng(15)
    n = 400
    strain, creep_old, damage = _states(rng, n, in_plane=True)
    if viscosity == 0.0:
        creep_old[:] = 0.0
    ours = _update(_kernel(viscosity=viscosity, plane_stress=True), strain, damage, 3, creep_old)
    ref = _update(_reference(viscosity, plane_stress=True), strain, damage, 2, creep_old)
    opening = np.linalg.norm(ref["eta"], axis=1) > ONSET_ETA
    assert 60 < np.count_nonzero(opening) < n - 60
    scale = _stress_scale(ref, damage)
    assert (np.abs(ours["stress"][:, 2]) <= 1.0e-8 * scale).all()
    assert (np.abs(ours["stress"] - ref["stress"]).max(axis=1) <= 1.0e-5 * scale).all()
    assert np.abs(ours["strain"][:, 2] - ref["strain"][:, 2]).max() <= 1.0e-6 * np.abs(strain).max()
    assert np.abs(ours["tangent"] - ref["tangent"].reshape(n, 6, 6)).max() <= 1.0e-3 * YOUNGS


@pytest.mark.parametrize("damage", [1.0, 0.3, RESIDUAL_DAMAGE])
def test_uniaxial_tensile_strength_is_degraded_f_t(damage):
    kernel = _kernel()
    low, high = 0.0, 5.0e6
    for _ in range(60):
        stress = 0.5 * (low + high)
        state = _update(kernel, _strain_of([stress, 0.0, 0.0])[None], np.array([damage]), 3)
        low, high = (low, stress) if np.linalg.norm(state["eta"]) > 1.0e-13 else (stress, high)
    assert high == pytest.approx(damage * F_T, rel=1.0e-6)


def test_compressive_states_stay_elastic():
    rng = np.random.default_rng(16)
    n = 200
    principal_stress = -10.0 ** rng.uniform(3.0, 8.0, size=(n, 3))
    strain = np.empty((n, 6))
    for i in range(n):
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        strain[i] = _mandel(rotation @ _tensor(_strain_of(principal_stress[i])) @ rotation.T)
    state = _update(_kernel(), strain, np.full(n, RESIDUAL_DAMAGE), 3)
    assert not state["eta"].any()
    values = np.array([np.linalg.eigvalsh(_tensor(stress)) for stress in state["stress"]])
    assert np.allclose(values, np.sort(principal_stress, axis=1), rtol=1.0e-9)


def test_equals_r2_2L_where_its_cohesive_strain_is_positive_definite():
    rng = np.random.default_rng(17)
    n = 200
    strain = _random_strains(rng, n, 1.0e-4)
    strain[:, :3] += rng.uniform(0.5e-4, 4.0e-4, size=(n, 1))
    damage = 10.0 ** rng.uniform(-2.0, 0.0, size=n)
    ours = _update(_kernel(), strain, damage, 3)
    ref = _update(_kernel("r2_2L"), strain, damage, 2)
    free = np.array([np.linalg.eigvalsh(_tensor(eta)).min() for eta in ref["eta"]]) > 1.0e-12
    assert np.count_nonzero(free) > 50
    scale = np.maximum(np.abs(ref["stress"]).max(axis=1), 1.0)
    assert (np.abs(ours["stress"] - ref["stress"]).max(axis=1)[free] <= 1.0e-5 * scale[free]).all()


def test_result_does_not_depend_on_the_stored_multipliers():
    """The open directions are identified from the stored cohesive strain, not from the
    order of the stored multipliers, which changes with the order of the principal strains."""
    rng = np.random.default_rng(18)
    n = 200
    strain, _, damage = _states(rng, n)
    kernel = _kernel()
    first = _update(kernel, strain, damage, 3)
    for multipliers in (first["multipliers"], first["multipliers"][:, ::-1]):
        stress, eta, tangent = np.zeros(n * 6), first["eta"].copy().ravel(), np.zeros(n * 36)
        kernel(
            strain.copy().ravel(), CREEP_DT, damage.copy(),
            stress, eta, np.zeros(n * 6), np.zeros(n * 6), np.zeros(n * 6),
            tangent, np.zeros(n), np.zeros(n), np.zeros(n), multipliers.copy().ravel(), np.zeros(n * 3),
        )
        assert np.abs(stress.reshape(n, 6) - first["stress"]).max() <= 1.0e-6 * F_T


@pytest.mark.parametrize("viscosity", [0.0, GLEN_VISCOSITY])
def test_states_just_beyond_onset_converge(viscosity):
    """A closed point that opens in three directions starts from its onset direction.

    Started from zero cohesive strain, the local iteration stayed within the regularisation
    of the potential and returned NaN for about one in a hundred of these states. The first
    state is the one that stopped a cliff simulation at its first crack growth.
    """
    recorded = np.array([1.70139e-05, -1.16182e-05, 0.0, 0.0, 0.0, 2.2859e-05])
    rng = np.random.default_rng(0)
    n = 3000
    strain = recorded * (1.0 + 0.3 * rng.normal(size=(n, 1)))
    strain += 3.0e-6 * rng.normal(size=(n, 6)) * np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    strain[0] = recorded
    damage = np.clip(0.44769 * (1.0 + 0.5 * rng.normal(size=n)), RESIDUAL_DAMAGE, 1.0)
    damage[0] = 0.44769
    ours = _update(_kernel(viscosity=viscosity), strain, damage, 3, dt=30.0)
    ref = _update(_reference(viscosity), strain, damage, 2, dt=30.0)

    assert np.isfinite(ours["stress"]).all()
    assert np.isfinite(ours["tangent"]).all()
    opening = np.linalg.norm(ref["eta"], axis=1) > ONSET_ETA
    assert 1000 < np.count_nonzero(opening) < n - 1000
    # The regularised square root of the kernel potential shifts the stress of a point
    # that has only just opened; beyond that layer the two solutions coincide.
    error = np.abs(ours["stress"] - ref["stress"]).max(axis=1) / _stress_scale(ref, damage)
    assert error.max() <= 2.0e-4
    assert np.quantile(error, 0.99) <= 1.0e-6


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="intrinsic 2-D"):
        _kernel(true_2d=True)
    with pytest.raises(ValueError, match="uniaxial tensile strength"):
        _kernel(f_t=1.3e6)
    _kernel(f_t=math.sqrt(2.0 / 3.0) * F_S)      # the limit itself is valid
    with pytest.raises(ValueError, match="cohesive_multiplier_slots"):
        _update(_kernel(), _strain_of([1.0e5, 0.0, 0.0])[None], np.array([1.0]), 2)
