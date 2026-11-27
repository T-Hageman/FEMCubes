"""Opening-only r2 criterion ("r2_2L_PSD"): principal-space integration-point update."""

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from femcubes.solids._principal_cve import PrincipalCVEUpdate, opening_strength_scales, principal_state
from femcubes.solids._reduced_cve import ReducedCVEMaterial, ReducedCVEUpdate

E, NU = 9.5e9, 0.35
MU = E / (2.0 * (1.0 + NU))
LAME = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
BULK = LAME + 2.0 * MU / 3.0
F_T, F_S, KAPPA = 0.4e6, 1.5e6, 1.0e-3
A, B = opening_strength_scales(F_T, F_S)
SQRT2 = math.sqrt(2.0)


def mandel(T):
    return np.array([T[0, 0], T[1, 1], T[2, 2], SQRT2 * T[1, 2], SQRT2 * T[0, 2], SQRT2 * T[0, 1]])


def tensor(v):
    return np.array([[v[0], v[5] / SQRT2, v[4] / SQRT2], [v[5] / SQRT2, v[1], v[3] / SQRT2],
                     [v[4] / SQRT2, v[3] / SQRT2, v[2]]])


def random_strains(rng, n, amplitude):
    out = np.empty((n, 6))
    for i in range(n):
        X = rng.normal(size=(3, 3))
        out[i] = mandel(0.5 * (X + X.T)) * amplitude
    return out


def run(update, eps, dam, dt=1.0, strain_v_old=None):
    n = eps.shape[0]
    out = {name: np.zeros(n * k) for name, k in (("stress", 6), ("eta", 6), ("strain_v", 6), ("tangent", 36),
                                                 ("energy", 1), ("psi", 1), ("lam_v", 1), ("lam_c", 2))}
    old = np.zeros(n * 6) if strain_v_old is None else strain_v_old.copy().ravel()
    out["strain"] = eps.copy().ravel()      # plane stress writes e_zz back
    update(out["strain"], dt, dam.copy(), out["stress"], out["eta"], np.zeros(n * 6), out["strain_v"],
           old, out["tangent"], out["energy"], out["psi"], out["lam_v"], out["lam_c"], np.zeros(n * 2))
    return {name: value.reshape(n, -1) for name, value in out.items()}


def energy(eta, e, d):
    el = e - eta
    t = eta.sum()
    return 0.5 * (LAME * el.sum() ** 2 + 2.0 * MU * (el ** 2).sum()) \
        + d * math.sqrt(max(A * t * t + B * ((eta - t / 3.0) ** 2).sum(), 0.0))


@pytest.fixture(scope="module")
def update():
    return PrincipalCVEUpdate(lame=LAME, shear=MU, f_t=F_T, f_s=F_S, kappa=KAPPA)


def test_principal_solution_is_the_constrained_minimiser():
    rng = np.random.default_rng(1)
    m = 120
    e = rng.normal(size=(m, 3)) * 10.0 ** rng.uniform(-5.5, -3.0, size=(m, 1))
    d = 10.0 ** rng.uniform(-3.0, 0.0, size=m)
    sigma, eta, _, pot = principal_state(e, d, LAME, MU, A, B)
    assert eta.min() >= 0.0
    assert 20 < np.count_nonzero(pot > 0.0) < m - 20
    for i in range(m):
        scale = np.abs(e[i]).max()
        best = min(minimize(lambda x: energy(x * scale, e[i], d[i]) / (E * scale ** 2), x0, bounds=[(0, None)] * 3,
                            method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12}).fun
                   for x0 in (np.full(3, 1e-3), np.maximum(e[i], 0.0) / scale, np.ones(3)))
        assert energy(eta[i], e[i], d[i]) / (E * scale ** 2) <= best + 1e-12
        if pot[i] > 0.0:
            t = eta[i].sum()
            grad = d[i] * (A * t + B * (eta[i] - t / 3.0)) / pot[i] - sigma[i]
            violation = np.where(eta[i] > 0.0, np.abs(grad), np.maximum(-grad, 0.0)).max()
            assert violation <= 1e-9 * max(np.abs(sigma[i]).max(), d[i] * F_T)


def test_cohesive_strain_is_coaxial_minimiser_over_all_psd_tensors(update):
    rng = np.random.default_rng(2)
    n = 12
    eps = random_strains(rng, n, 2e-4)
    dam = 10.0 ** rng.uniform(-2.0, 0.0, size=n)
    res = run(update, eps, dam)
    assert np.count_nonzero(res["psi"]) >= 4

    def full_energy(x, T, dd, scale):
        L = np.zeros((3, 3))
        L[np.tril_indices(3)] = x
        H = L @ L.T * scale
        el = T - H
        dv = H - np.trace(H) / 3.0 * np.eye(3)
        return (0.5 * (LAME * np.trace(el) ** 2 + 2.0 * MU * (el * el).sum())
                + dd * math.sqrt(A * np.trace(H) ** 2 + B * (dv * dv).sum() + 1e-300)) / (E * scale ** 2)

    for i in range(n):
        T, scale = tensor(eps[i]), np.abs(eps[i]).max()
        best = min((minimize(full_energy, x0, args=(T, dam[i], scale), method="BFGS",
                             options={"gtol": 1e-11, "maxiter": 2000}) for x0 in rng.normal(size=(4, 6))),
                   key=lambda r: r.fun)
        L = np.zeros((3, 3))
        L[np.tril_indices(3)] = best.x
        H = tensor(res["eta"][i])
        assert np.linalg.eigvalsh(H).min() >= -1e-12 * scale
        assert np.abs(L @ L.T * scale - H).max() <= 1e-5 * scale


def test_tangent_matches_finite_differences(update):
    rng = np.random.default_rng(3)
    n = 120
    eps = random_strains(rng, n, 2e-4)
    eps[:20, 3:5] = 0.0
    eps[:20, 2] = 0.0                      # plane strain
    eps[20:30, 3:] = 0.0
    eps[20:30, 1] = eps[20:30, 0]          # repeated principal strains
    dam = 10.0 ** rng.uniform(-2.0, 0.0, size=n)
    tan = run(update, eps, dam)["tangent"].reshape(n, 6, 6)
    h = 1e-9
    fd = np.empty((n, 6, 6))
    for k in range(6):
        up, down = eps.copy(), eps.copy()
        up[:, k] += h
        down[:, k] -= h
        fd[:, :, k] = (run(update, up, dam)["stress"] - run(update, down, dam)["stress"]) / (2.0 * h)
    assert np.abs(tan - fd).max() <= 1e-7 * E
    assert np.abs(tan - tan.transpose(0, 2, 1)).max() <= 1e-12 * E


def test_states_without_tensile_principal_stress_stay_elastic(update):
    rng = np.random.default_rng(4)
    n = 500
    sp = -10.0 ** rng.uniform(3.0, 8.0, size=(n, 3))
    ep = (sp - LAME / (3.0 * LAME + 2.0 * MU) * sp.sum(axis=1, keepdims=True)) / (2.0 * MU)
    eps = np.empty((n, 6))
    for i in range(n):
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        eps[i] = mandel(Q @ np.diag(ep[i]) @ Q.T)
    res = run(update, eps, np.full(n, KAPPA))
    assert not res["eta"].any() and not res["psi"].any()
    assert np.allclose(np.sort(np.array([np.linalg.eigvalsh(tensor(s)) for s in res["stress"]]), axis=1),
                       np.sort(sp, axis=1), rtol=1e-9)


def test_equals_r2_2L_where_its_cohesive_strain_is_positive_definite(update):
    rng = np.random.default_rng(5)
    n = 200
    eps = random_strains(rng, n, 1e-4)
    eps[:, :3] += rng.uniform(0.5e-4, 4e-4, size=(n, 1))
    dam = 10.0 ** rng.uniform(-2.0, 0.0, size=n)
    material = ReducedCVEMaterial.from_model_constants(
        true_2d=False, lame=LAME, shear=MU, bulk=BULK, viscosity=0.0, n_exp=1.0, f_t=F_T, f_s=F_S, kappa=KAPPA,
        failure_type="r2_2L")
    ours, ref = run(update, eps, dam), run(ReducedCVEUpdate(material), eps, dam)
    free = np.array([np.linalg.eigvalsh(tensor(v)).min() for v in ref["eta"]]) > 1e-12
    assert np.count_nonzero(free) > 50
    scale = np.maximum(np.abs(ref["stress"]).max(axis=1), 1.0)
    assert (np.abs(ours["stress"] - ref["stress"]).max(axis=1)[free] <= 1e-6 * scale[free]).all()


def test_uniaxial_tensile_strength_and_lateral_compression():
    def onset(s2):
        lo, hi = 0.0, 5.0e6
        for _ in range(50):
            s1 = 0.5 * (lo + hi)
            sp = np.array([[s1, s2, 0.0]])
            ep = (sp - LAME / (3.0 * LAME + 2.0 * MU) * sp.sum()) / (2.0 * MU)
            lo, hi = (lo, s1) if principal_state(ep, np.ones(1), LAME, MU, A, B)[3][0] > 0.0 else (s1, hi)
        return hi

    assert onset(0.0) == pytest.approx(F_T, rel=1e-6)
    # opening against a compressed lateral direction: support without that direction
    assert onset(-5.0e6) > onset(-0.25e6) > F_T
    assert onset(-5.0e6) < math.sqrt(A + 2.0 * B / 3.0) + 1.0


GLEN_A, CREEP_DT = 1.0e-24, 2.0e3


def creep_update(n_exp):
    """Update with creep and its J2 stress scale: Glen's law for n = 3, a viscosity with a
    comparable creep strain per step for n = 1."""
    viscosity = (3.0 ** ((n_exp + 1.0) / 2.0) / (2.0 * GLEN_A)) ** (1.0 / n_exp) if n_exp > 1.0 else 3.0e13
    return PrincipalCVEUpdate(lame=LAME, shear=MU, f_t=F_T, f_s=F_S, kappa=KAPPA, viscosity=viscosity,
                              n_exp=n_exp), viscosity


def creep_states(rng, n):
    eps = random_strains(rng, n, 1.5e-4)
    eps[: n // 2, :3] += rng.uniform(0.0, 3e-4, size=(n // 2, 1))
    old = random_strains(rng, n, 2e-5)
    old[:, :3] -= old[:, :3].mean(axis=1, keepdims=True)
    return eps, old, 10.0 ** rng.uniform(-2.0, 0.0, size=n)


@pytest.mark.parametrize("n_exp", [1.0, 3.0])
def test_creep_update_satisfies_creep_law_and_opening_conditions(update, n_exp):
    rng = np.random.default_rng(7)
    n = 300
    eps, old, dam = creep_states(rng, n)
    visco, viscosity = creep_update(n_exp)
    res = run(visco, eps, dam, dt=CREEP_DT, strain_v_old=old)
    opening = res["psi"][:, 0] > 0.0
    assert 60 < np.count_nonzero(opening) < n - 60
    # creep increment: dt gamma^(n + 1) (|s| / viscosity)^n s / |s| of the stress deviator
    dev = res["stress"].copy()
    dev[:, :3] -= dev[:, :3].mean(axis=1, keepdims=True)
    norm = np.linalg.norm(dev, axis=1, keepdims=True)
    expected = CREEP_DT * 1.5 ** ((n_exp + 1.0) / 2.0) * (norm / viscosity) ** n_exp * dev / norm
    increment = res["strain_v"] - old
    assert np.median(np.abs(increment).max(axis=1)) > 1e-6
    assert np.abs(increment - expected).max() <= 1e-9 * np.abs(increment).max()
    assert np.allclose(res["lam_v"][:, 0], np.linalg.norm(increment, axis=1) / math.sqrt(1.5), rtol=1e-9)
    # stress and opening: the update without creep at the strain less the creep strain
    ref = run(update, eps - res["strain_v"], dam)
    assert np.abs(res["stress"] - ref["stress"]).max() <= 1e-8 * np.abs(ref["stress"]).max()
    assert np.abs(res["eta"] - ref["eta"]).max() <= 1e-8 * np.abs(ref["eta"]).max()
    assert np.allclose(res["psi"], ref["psi"], rtol=1e-8, atol=1e-12)
    assert np.allclose(res["energy"], ref["energy"], rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("n_exp", [1.0, 3.0])
def test_creep_update_equals_r2_2L_where_its_cohesive_strain_is_positive_definite(n_exp):
    rng = np.random.default_rng(8)
    n = 200
    eps, old, dam = creep_states(rng, n)
    eps[:, :3] += rng.uniform(0.5e-4, 3e-4, size=(n, 1))
    visco, viscosity = creep_update(n_exp)
    material = ReducedCVEMaterial.from_model_constants(
        true_2d=False, lame=LAME, shear=MU, bulk=BULK, viscosity=viscosity, n_exp=n_exp, f_t=F_T, f_s=F_S,
        kappa=KAPPA, failure_type="r2_2L")
    ours = run(visco, eps, dam, dt=CREEP_DT, strain_v_old=old)
    ref = run(ReducedCVEUpdate(material), eps, dam, dt=CREEP_DT, strain_v_old=old)
    free = np.array([np.linalg.eigvalsh(tensor(v)).min() for v in ref["eta"]]) > 1e-12
    assert np.count_nonzero(free) > 50
    scale = np.maximum(np.abs(ref["stress"]).max(axis=1), 1.0)
    assert (np.abs(ours["stress"] - ref["stress"]).max(axis=1)[free] <= 1e-6 * scale[free]).all()
    assert np.abs(ours["strain_v"] - ref["strain_v"])[free].max() <= 1e-6 * np.abs(ref["strain_v"]).max()


@pytest.mark.parametrize("n_exp", [1.0, 3.0])
def test_creep_tangent_matches_finite_differences(n_exp):
    rng = np.random.default_rng(9)
    n = 120
    eps, old, dam = creep_states(rng, n)
    eps[:20, 3:5] = old[:20, 3:5] = 0.0
    eps[:20, 2] = 0.0                      # plane strain
    eps[20:30, 3:] = old[20:30, 3:] = 0.0
    eps[20:30, 1], old[20:30, 1] = eps[20:30, 0], old[20:30, 0]          # repeated principal strains
    visco, _ = creep_update(n_exp)
    tan = run(visco, eps, dam, dt=CREEP_DT, strain_v_old=old)["tangent"].reshape(n, 6, 6)
    h = 1e-9
    fd = np.empty((n, 6, 6))
    for k in range(6):
        up, down = eps.copy(), eps.copy()
        up[:, k] += h
        down[:, k] -= h
        fd[:, :, k] = (run(visco, up, dam, dt=CREEP_DT, strain_v_old=old)["stress"]
                       - run(visco, down, dam, dt=CREEP_DT, strain_v_old=old)["stress"]) / (2.0 * h)
    assert np.abs(tan - fd).max() <= 1e-6 * E
    assert np.abs(tan - tan.transpose(0, 2, 1)).max() <= 1e-10 * E


def test_invalid_creep_constants_are_rejected():
    with pytest.raises(ValueError):
        PrincipalCVEUpdate(lame=LAME, shear=MU, f_t=F_T, f_s=F_S, kappa=KAPPA, viscosity=-1.0)
    with pytest.raises(ValueError):
        PrincipalCVEUpdate(lame=LAME, shear=MU, f_t=F_T, f_s=F_S, kappa=KAPPA, viscosity=1.0e8, n_exp=0.5)


def test_true_2d_is_rejected():
    with pytest.raises(NotImplementedError):
        PrincipalCVEUpdate(lame=LAME, shear=MU, f_t=F_T, f_s=F_S, kappa=KAPPA, true_2d=True)


def plane_updates(n_exp):
    """(plane-stress update, 3-D update, J2 stress scale); ``n_exp`` None: no creep."""
    if n_exp is None:
        viscosity, exponent = 0.0, 1.0
    else:
        viscosity, exponent = creep_update(n_exp)[1], n_exp
    kwargs = dict(lame=LAME, shear=MU, f_t=F_T, f_s=F_S, kappa=KAPPA, viscosity=viscosity, n_exp=exponent)
    return PrincipalCVEUpdate(plane_stress=True, **kwargs), PrincipalCVEUpdate(**kwargs), viscosity


def plane_states(rng, n):
    """In-plane strains (half of them with added tension), an arbitrary input e_zz, a deviatoric
    committed creep strain without out-of-plane shear, and the damage."""
    eps, old, dam = creep_states(rng, n)
    eps[:, 3:5] = old[:, 3:5] = 0.0
    eps[:, 2] = rng.normal(size=n) * 1e-3
    return eps, old, dam


@pytest.mark.parametrize("n_exp", [None, 1.0, 3.0])
def test_plane_stress_state_is_the_3d_state_with_zero_out_of_plane_stress(n_exp):
    rng = np.random.default_rng(21)
    n = 300
    eps, old, dam = plane_states(rng, n)
    if n_exp is None:
        old[:] = 0.0
    plane, full, _ = plane_updates(n_exp)
    res = run(plane, eps, dam, dt=CREEP_DT, strain_v_old=old)
    opening = res["psi"][:, 0] > 0.0
    assert 30 < np.count_nonzero(opening) < n - 30
    assert np.array_equal(res["strain"][:, (0, 1, 3, 4, 5)], eps[:, (0, 1, 3, 4, 5)])
    in_plane = np.abs(res["stress"][:, 0]) + np.abs(res["stress"][:, 1])
    assert (np.abs(res["stress"][:, 2]) <= 1e-8 * np.maximum(in_plane, dam * F_T)).all()
    assert np.abs(res["stress"][:, 3:5]).max() <= 1e-9 * np.abs(res["stress"]).max()
    ref = run(full, res["strain"], dam, dt=CREEP_DT, strain_v_old=old)
    for name in ("stress", "eta", "strain_v", "energy", "psi", "lam_v"):
        assert np.abs(res[name] - ref[name]).max() <= 1e-7 * np.abs(ref[name]).max(), name
    # independent of the input e_zz
    other = eps.copy()
    other[:, 2] = 0.0
    assert np.abs(run(plane, other, dam, dt=CREEP_DT, strain_v_old=old)["stress"] - res["stress"]).max() \
        <= 1e-7 * np.abs(res["stress"]).max()


@pytest.mark.parametrize("n_exp", [None, 3.0])
def test_plane_stress_tangent_is_condensed_and_matches_finite_differences(n_exp):
    rng = np.random.default_rng(22)
    n = 120
    eps, old, dam = plane_states(rng, n)
    if n_exp is None:
        old[:] = 0.0
    plane, _, _ = plane_updates(n_exp)
    res = run(plane, eps, dam, dt=CREEP_DT, strain_v_old=old)
    assert np.count_nonzero(res["psi"][:, 0] > 0.0) > 30
    tan = res["tangent"].reshape(n, 6, 6)
    assert not tan[:, 2, :].any() and not tan[:, :, 2].any()
    h = 1e-9
    for k in (0, 1, 5):
        up, down = eps.copy(), eps.copy()
        up[:, k] += h
        down[:, k] -= h
        fd = (run(plane, up, dam, dt=CREEP_DT, strain_v_old=old)["stress"]
              - run(plane, down, dam, dt=CREEP_DT, strain_v_old=old)["stress"]) / (2.0 * h)
        assert np.abs(tan[:, (0, 1, 5), k] - fd[:, (0, 1, 5)]).max() <= 1e-6 * E, k
    assert np.abs(tan - tan.transpose(0, 2, 1)).max() <= 1e-10 * E


@pytest.mark.parametrize("n_exp", [None, 3.0])
def test_plane_stress_equals_r2_2L_where_its_cohesive_strain_is_positive_definite(n_exp):
    rng = np.random.default_rng(23)
    n = 300
    eps, old, dam = plane_states(rng, n)
    eps[:, :2] += rng.uniform(0.5e-4, 3e-4, size=(n, 1))
    if n_exp is None:
        old[:] = 0.0
    plane, _, viscosity = plane_updates(n_exp)
    material = ReducedCVEMaterial.from_model_constants(
        true_2d=False, lame=LAME, shear=MU, bulk=BULK, viscosity=viscosity, n_exp=n_exp or 1.0, f_t=F_T, f_s=F_S,
        kappa=KAPPA, failure_type="r2_2L")
    ours = run(plane, eps, dam, dt=CREEP_DT, strain_v_old=old)
    ref = run(ReducedCVEUpdate(material, plane_stress=True), eps, dam, dt=CREEP_DT, strain_v_old=old)
    free = np.array([np.linalg.eigvalsh(tensor(v)).min() for v in ref["eta"]]) > 1e-12
    assert np.count_nonzero(free) > 30
    scale = np.maximum(np.abs(ref["stress"]).max(axis=1), 1.0)
    assert (np.abs(ours["stress"] - ref["stress"]).max(axis=1)[free] <= 1e-6 * scale[free]).all()
    assert np.abs(ours["strain"][:, 2] - ref["strain"][:, 2])[free].max() <= 1e-6 * np.abs(ref["strain"][:, 2]).max()


def _random_tensors(n, seed):
    rng = np.random.default_rng(seed)
    mandel = rng.normal(size=(n, 6)) * 10.0 ** rng.uniform(-3, 7, size=(n, 1))
    mandel[:50, 3:] = 0.0                                   # principal axes along the coordinates
    mandel[50:60] = [[2.0, 2.0, 2.0, 0.0, 0.0, 0.0]]        # hydrostatic
    mandel[60:70] = [[1.0, 1.0, -3.0, 0.0, 0.0, 0.0]]       # double largest eigenvalue
    mandel[70:80] = 0.0
    tensor = np.empty((n, 3, 3))
    for i in range(3):
        tensor[:, i, i] = mandel[:, i]
    for k, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))):
        tensor[:, i, j] = tensor[:, j, i] = mandel[:, 3 + k] / math.sqrt(2.0)
    return mandel, tensor


def test_principal_values_are_the_eigenvalues_largest_first():
    from femcubes.solids._principal_cve import principal_values

    mandel, tensor = _random_tensors(2000, 3)
    expected = np.linalg.eigvalsh(tensor)[:, ::-1]
    scale = np.maximum(np.abs(mandel).max(axis=1), 1.0e-300)[:, None]
    assert (np.abs(principal_values(mandel) - expected) / scale).max() < 1e-7


def test_strength_ratio_is_one_on_the_strength_surface_and_scales_with_the_stress():
    from femcubes.solids._principal_cve import strength_ratio

    f_t, f_s = 0.4e6, 1.5e6
    A, B = opening_strength_scales(f_t, f_s)
    one = np.ones(1)
    for opening_only in (False, True):
        assert strength_ratio([[f_t, 0, 0, 0, 0, 0]], one, A, B, opening_only)[0] == pytest.approx(1.0)
        assert strength_ratio([[0.5 * f_t, 0, 0, 0, 0, 0]], 0.25 * one, A, B, opening_only)[0] == pytest.approx(2.0)
        # biaxial tension reaches the surface below f_t
        assert strength_ratio([[f_t, 0, 0.5 * f_t, 0, 0, 0]], one, A, B, opening_only)[0] > 1.2
    # uniaxial compression: shear failure at f_s for "r2_2L", none for the opening-only set
    assert strength_ratio([[0, -f_s, 0, 0, 0, 0]], one, A, B, False)[0] == pytest.approx(1.0)
    assert strength_ratio([[0, -f_s, 0, 0, 0, 0]], one, A, B, True)[0] < 1e-9


def test_strength_ratio_of_opening_states_is_at_least_one():
    """On the opening-only strength surface the bound is >= 1, and = 1 where no principal stress is compressive."""
    from femcubes.solids._principal_cve import principal_values, strength_ratio

    lame, shear, f_t, f_s = 8.2e9, 3.5e9, 0.4e6, 1.5e6
    update = PrincipalCVEUpdate(lame=lame, shear=shear, f_t=f_t, f_s=f_s, kappa=1.0e-3)
    rng = np.random.default_rng(5)
    n = 4000
    strain = rng.normal(size=(n, 6)) * 2.0e-4
    strain[n // 2:] *= 0.02                                   # second half: all-round tension, near the strength
    strain[n // 2:, :3] += rng.uniform(0.0, 4.0e-5, size=(n - n // 2, 1))
    damage = rng.uniform(0.05, 1.0, size=n)
    stress, eta, strain_v, tangent = np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 36))
    scalars = [np.zeros(n) for _ in range(3)]
    update(strain.copy(), 1.0, damage, stress, eta, np.zeros((n, 6)), strain_v, np.zeros((n, 6)), tangent,
           scalars[0], scalars[1], scalars[2], np.zeros((n, 2)), np.zeros((n, 2)))
    ratio = strength_ratio(stress, damage, update.A, update.B, True)
    active = np.linalg.norm(eta, axis=1) > 0.0
    assert active.sum() > 500 and (~active).sum() > 500
    assert np.all(ratio[active] >= 1.0 - 1e-6)
    tensile = principal_values(stress).min(axis=1) >= 0.0
    assert (tensile & active).sum() > 20 and (tensile & ~active).sum() > 20
    np.testing.assert_allclose(ratio[tensile & active], 1.0, rtol=1e-6)
    assert np.all(ratio[tensile & ~active] <= 1.0 + 1e-6)


def floe_band_states(rng, n):
    """States of a line-search trial inside a fresh crack band: large in-plane strains at almost
    broken points, where the stress is the small difference of large strains."""
    amplitude = 10.0 ** rng.uniform(-2.0, 0.0, size=n)
    eps = np.zeros((n, 6))
    eps[:, 0], eps[:, 1], eps[:, 5] = (rng.normal(size=n) * amplitude for _ in range(3))
    old = np.zeros((n, 6))
    old[:, :2] = rng.normal(size=(n, 2)) * 1e-5
    old[:, 2] = -old[:, :2].sum(axis=1)
    dam = np.where(rng.random(n) < 0.5, 1.0, 10.0 ** rng.uniform(-3.0, 0.0, size=n))
    return eps, old, dam


@pytest.mark.parametrize("n_exp", [None, 3.0])
def test_plane_stress_converges_at_large_strains_of_almost_broken_points(n_exp):
    # bug fix: a tolerance relative to the degraded strength alone is below the round-off of the stress here
    update, _, _ = plane_updates(n_exp)
    eps, old, dam = floe_band_states(np.random.default_rng(3), 2000)
    out = run(update, eps, dam, dt=30.0, strain_v_old=old)
    assert np.isfinite(out["stress"]).all() and np.isfinite(out["tangent"]).all()
    sigma = out["stress"]
    relative = np.maximum(np.clip(dam, KAPPA, 1.0) * F_S, np.abs(sigma[:, 0]) + np.abs(sigma[:, 1]))
    roundoff = (LAME + 2.0 * MU) * np.abs(out["strain"].reshape(-1, 6) - old).max(axis=1)
    assert (np.abs(sigma[:, 2]) <= 1e-9 * relative + 1e-12 * roundoff).all()


@pytest.mark.parametrize("n_exp", [None, 3.0])
def test_point_without_converged_local_iteration_gets_nonfinite_stress_and_no_exception(n_exp, monkeypatch):
    # an exception raised on one rank stalls a parallel run; a non-finite residual is handled by the solver
    update, update_3d, _ = plane_updates(n_exp)
    eps, old, dam = plane_states(np.random.default_rng(22), 200)
    reference = run(update, eps, dam, dt=30.0, strain_v_old=old)
    monkeypatch.setattr(update, "PLANE_STRESS_MAX_ITER", 1)
    out = run(update, eps, dam, dt=30.0, strain_v_old=old)
    failed = ~np.isfinite(out["stress"]).all(axis=1)
    assert failed.any() and (n_exp is not None or not failed.all())    # with creep no point is done after one pass
    assert np.isfinite(out["tangent"]).all()
    assert np.array_equal(out["stress"][~failed], reference["stress"][~failed])
    if n_exp is not None:
        import femcubes.solids._principal_cve as module
        original = module.coupled_state

        def never_converged(*args, **kwargs):
            result = original(*args, **kwargs)
            return (*result[:5], np.zeros_like(result[5]))

        monkeypatch.setattr(module, "coupled_state", never_converged)
        out = run(update_3d, eps, dam, dt=30.0, strain_v_old=old)
        failed = ~np.isfinite(out["stress"]).all(axis=1)
        assert 0 < failed.sum() < 200 and np.isfinite(out["tangent"]).all()


def test_nonfinite_strain_gives_nonfinite_stress_and_no_exception():
    update, update_3d, _ = plane_updates(3.0)
    eps, old, dam = plane_states(np.random.default_rng(23), 20)
    eps[3, 0], eps[7, 5] = np.nan, np.inf
    for candidate in (update, update_3d):
        out = run(candidate, eps, dam, dt=30.0, strain_v_old=old)
        finite = np.isfinite(out["stress"]).all(axis=1)
        assert not finite[3] and not finite[7] and finite[[0, 1, 2, 4, 5, 6]].all()


def test_principal_stress_is_continuous_where_a_direction_starts_to_open():
    # bug fix: the support was chosen by energies that differ by less than round-off at the switch
    rng = np.random.default_rng(31)
    m, worst = 400, 0.0
    e = rng.normal(size=(m, 3)) * 10.0 ** rng.uniform(-5.0, -3.0, size=(m, 1))
    d = 10.0 ** rng.uniform(-3.0, 0.0, size=m)
    direction = rng.normal(size=(m, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    span = np.abs(e).max(axis=1)

    def opened(t):
        return (principal_state(e + (t * span)[:, None] * direction, d, LAME, MU, A, B)[1] > 0.0).sum(axis=1)

    # bisect each line e + t span direction, t in [0, 1], to a change of the number of opening directions
    lo, hi = np.zeros(m), np.ones(m)
    n_lo = opened(lo)
    switching = opened(hi) != n_lo
    assert switching.sum() > 100
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        same = opened(mid) == n_lo
        lo, hi = np.where(same, mid, lo), np.where(same, hi, mid)
    sigma_lo = principal_state(e + (lo * span)[:, None] * direction, d, LAME, MU, A, B)[0]
    sigma_hi = principal_state(e + (hi * span)[:, None] * direction, d, LAME, MU, A, B)[0]
    jump = np.abs(sigma_hi - sigma_lo).max(axis=1)[switching]
    assert jump.max() <= 1e-12 * (LAME + 2.0 * MU) * span[switching].max()
