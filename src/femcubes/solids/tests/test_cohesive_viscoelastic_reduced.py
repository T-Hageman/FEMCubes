"""Solid:Cohesive_ViscoElastic and its reduced update (femcubes.solids._reduced_cve).

Contract: for every FailureType, intrinsic 2-D, 3-D and plane stress, the reduced update
reproduces the C++ kernel of Solid:Cohesive_ViscoElastic_Kernel (to the kernel's tolerance; the
kernel's smoothed split-r2 potential may return tiny "regularised" multipliers inside the
failure surface, and its accepted states are less accurate at large openings, so the stress
may differ by the elastic effect of the kernel's multipliers there); the tangent is the
derivative of the update; plane stress is the 3-D update at sigma_zz = 0; points the reduced
Newton method does not converge at are recomputed by the fallback; the model uses the update
for every failure type. Numba bodies are also run as Python (``py_func``) for coverage.
"""

import math

import numpy as np
import pytest

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.solids import _reduced_cve as reduced
from femcubes.solids._reduced_cve import FAILURE_TYPES, ReducedCVEMaterial, ReducedCVEUpdate, creep_only_table
from femcubes.solids.cohesive_linear_elastic import Cohesive_LinearElastic
from femcubes.solids.cohesive_viscoelastic import Cohesive_ViscoElastic
from femcubes.solids.cohesive_viscoelastic_kernel import Cohesive_ViscoElastic_Kernel
from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    CohesiveViscoElasticKernels,
)

YOUNGS, POISSON = 9.5e9, 0.35
SHEAR = YOUNGS / (2.0 * (1.0 + POISSON))
GLEN_A, GLEN_N = 1.0e-24, 3.0
F_T, F_S, KAPPA = 0.4e6, 0.6e6, 1.0e-3
DT = 100.0
TYPES = sorted(FAILURE_TYPES)
# (failure type, nd) and (failure type, nd, plane stress)
TYPES_ND = [(t, nd) for t in TYPES for nd in (2, 3)]
TYPES_ND_PLANE_STRESS = [(t, nd, False) for t, nd in TYPES_ND] + [(t, 3, True) for t in TYPES]


def _material(nd, failure_type, viscous=True):
    """PRIVATE: ice-like constants; DP types with a 30 degree friction angle and a cap."""
    if nd == 2:
        lame = YOUNGS * POISSON / (1.0 - POISSON**2)
        bulk = lame + SHEAR
    else:
        lame = YOUNGS * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
        bulk = lame + 2.0 * SHEAR / 3.0
    viscosity = (nd ** ((GLEN_N + 1.0) / 2.0) / (2.0 * GLEN_A)) ** (1.0 / GLEN_N) if viscous else 0.0
    dp = failure_type in reduced.DP_TYPES
    return ReducedCVEMaterial(nd=nd, lame=lame, shear=SHEAR, bulk=bulk, viscosity=viscosity, n_exp=GLEN_N, f_t=F_T,
                              f_s=F_S, kappa=KAPPA, failure_type=failure_type, DP_phi=30.0 if dp else 0.0,
                              DP_q=5.0e6 if dp else F_S)


def _kernel(material, plane_stress=False):
    """PRIVATE: the C++ kernel of Cohesive_ViscoElastic_Kernel for the same material."""
    m = material
    return CohesiveViscoElasticKernels(m.bulk, m.lame, m.shear, m.viscosity, m.n_exp, m.f_t, m.f_s, m.G2_over_G1,
                                       m.DP_phi, m.DP_q, m.kappa, m.failure_type, plane_stress,
                                       m.nd == 2 and not plane_stress)


def _states(material, n, seed, plane_stress=False):
    """PRIVATE: random strains (0.1-30 times the strength strain), committed viscous strains and damage."""
    rng = np.random.default_rng(seed)
    s0 = material.f_t / material.youngs
    strain = rng.normal(size=(n, 6)) * s0 * 10.0 ** rng.uniform(-1.0, 1.5, size=(n, 1))
    strain_v_old = rng.normal(size=(n, 6)) * s0 * 0.3
    if material.nd == 2:
        strain[:, 2:5] = 0.0
        strain_v_old[:, 2:5] = 0.0
    if plane_stress:
        strain[:, 2:5] = 0.0
        strain_v_old[:, 3:5] = 0.0
    damage = np.where(rng.uniform(size=n) < 0.4, 1.0, 10.0 ** rng.uniform(-3.0, 0.0, size=n))
    return strain, strain_v_old, damage


def _call(update, strain, strain_v_old, damage, dt=DT):
    """PRIVATE: evaluate a kernel-signature update; outputs reshaped per point, the strain as
    updated (plane-stress e_zz) and the update's return value."""
    n = damage.size
    out = {"stress": np.zeros(6 * n), "eta": np.zeros(6 * n), "strain_v": np.zeros(6 * n), "tangent": np.zeros(36 * n),
           "energy": np.zeros(n), "psi": np.zeros(n), "lambda_v": np.zeros(n), "lambda_c": np.zeros(2 * n)}
    strain_io = np.ascontiguousarray(strain, dtype=np.float64).ravel().copy()
    result = update(strain_io, dt, damage.copy(), out["stress"], out["eta"], np.zeros(6 * n), out["strain_v"],
                    np.ascontiguousarray(strain_v_old, dtype=np.float64).ravel(), out["tangent"], out["energy"],
                    out["psi"], out["lambda_v"], out["lambda_c"], np.zeros(2 * n))
    shapes = {"tangent": (n, 6, 6), "energy": (n,), "psi": (n,), "lambda_v": (n,), "lambda_c": (n, 2)}
    out = {k: v.reshape(shapes.get(k, (n, 6))) for k, v in out.items()}
    out["strain"] = strain_io.reshape(n, 6)
    out["result"] = result
    return out


def _stress_error(material, ours, reference, damage):
    """PRIVATE: stress error relative to max(d f, |sigma|), and the allowance for the kernel's
    multipliers (elastic effect of its cohesive strain, relative to the same scale)."""
    scale = np.maximum(np.clip(damage, KAPPA, 1.0) * max(F_T, F_S), np.abs(reference["stress"]).max(axis=1))
    allowance = (material.k_vol * np.abs(reference["lambda_c"][:, 0])
                 + material.k_dev * np.abs(reference["lambda_c"][:, 1])) / scale
    return np.abs(ours["stress"] - reference["stress"]).max(axis=1) / scale, allowance


# ----------------------------------------------------------------------
# Agreement with the C++ kernel
# ----------------------------------------------------------------------

@pytest.mark.parametrize("failure_type, nd", TYPES_ND)
def test_reduced_update_matches_the_kernel(failure_type, nd):
    material = _material(nd, failure_type)
    strain, strain_v_old, damage = _states(material, 2000, seed=nd)
    ours = _call(ReducedCVEUpdate(material), strain, strain_v_old, damage)
    reference = _call(_kernel(material), strain, strain_v_old, damage)
    finite = np.all(np.isfinite(reference["stress"]), axis=1)
    error, allowance = _stress_error(material, ours, reference, damage)
    active = np.any(ours["lambda_c"] > 0.0, axis=1)
    assert ours["result"] == 0
    assert active.sum() > 200 and finite.sum() > 1900
    assert np.all(error[finite] < 1.0e-6 + 2.0 * allowance[finite])
    assert np.quantile(error[finite & active], 0.99) < 1.0e-5


@pytest.mark.parametrize("failure_type", ["r2_2L", "r1", "r2_DP"])
def test_reduced_update_without_creep_matches_the_kernel(failure_type):
    material = _material(3, failure_type, viscous=False)
    strain, strain_v_old, damage = _states(material, 1000, seed=5)
    ours = _call(ReducedCVEUpdate(material), strain, strain_v_old, damage)
    reference = _call(_kernel(material), strain, strain_v_old, damage)
    error, allowance = _stress_error(material, ours, reference, damage)
    assert np.all(np.abs(ours["lambda_v"]) < 1.0e-20)   # round-off of the coupled Newton step
    assert np.all(error < 1.0e-6 + 2.0 * allowance)


@pytest.mark.parametrize("failure_type, nd, plane_stress", TYPES_ND_PLANE_STRESS)
def test_reduced_tangent_is_the_derivative_of_the_update(failure_type, nd, plane_stress):
    material = _material(nd, failure_type)
    update = ReducedCVEUpdate(material, plane_stress=plane_stress, fallback=_kernel(material, plane_stress))
    strain, strain_v_old, damage = _states(material, 200, seed=10 + nd, plane_stress=plane_stress)
    base = _call(update, strain, strain_v_old, damage)
    active = base["lambda_c"] > 0.0
    components = (0, 1, 5) if (nd == 2 or plane_stress) else tuple(range(6))
    h = 1.0e-7 * material.f_t / material.youngs
    errors = []
    for k in components:
        step = np.zeros(6)
        step[k] = h
        plus = _call(update, strain + step, strain_v_old, damage)
        minus = _call(update, strain - step, strain_v_old, damage)
        same_branch = (np.all((plus["lambda_c"] > 0.0) == active, axis=1)
                       & np.all((minus["lambda_c"] > 0.0) == active, axis=1))
        fd = (plus["stress"] - minus["stress"]) / (2.0 * h)
        rows = list(components)
        errors.append(np.where(same_branch, np.abs(fd[:, rows] - base["tangent"][:, rows, k]).max(axis=1), 0.0))
    assert np.any(active, axis=1).sum() > 50
    assert np.max(errors) < 1.0e-5 * (material.lame + 2.0 * material.shear)


@pytest.mark.parametrize("failure_type", TYPES)
def test_reduced_plane_stress_is_the_3d_kernel_at_sigma_zz_zero(failure_type):
    # The kernel's own plane-stress states are less accurate at strongly damaged points, so the
    # reference is the 3-D kernel at the solved e_zz; the e_zz of both agree closely.
    material = _material(3, failure_type)
    strain, strain_v_old, damage = _states(material, 1000, seed=21, plane_stress=True)
    ours = _call(ReducedCVEUpdate(material, plane_stress=True, fallback=_kernel(material, True)), strain,
                 strain_v_old, damage)
    reference = _call(_kernel(material), ours["strain"], strain_v_old, damage)
    plane_stress_kernel = _call(_kernel(material, True), strain, strain_v_old, damage)
    error, allowance = _stress_error(material, ours, reference, damage)
    strain_scale = np.maximum(np.abs(strain[:, :2]).max(axis=1), material.f_t / material.youngs)
    assert np.all(np.abs(ours["stress"][:, 2]) < 1.0e-8 * np.maximum(np.abs(ours["stress"]).max(axis=1), 1.0))
    assert np.all(error < 1.0e-6 + 2.0 * allowance)
    assert np.quantile(np.abs(ours["strain"][:, 2] - plane_stress_kernel["strain"][:, 2]) / strain_scale, 0.99) < 1.0e-4
    np.testing.assert_array_equal(ours["tangent"][:, 2, :], 0.0)
    np.testing.assert_array_equal(ours["tangent"][:, :, 2], 0.0)


@pytest.mark.parametrize("failure_type", TYPES)
def test_reduced_update_is_finite_for_pure_volumetric_and_pure_shear_strain(failure_type):
    # e_q = 0 (the kernel has no converged solution for some types) and e_v = 0
    material = _material(3, failure_type)
    s0 = material.f_t / material.youngs
    strain = np.zeros((4, 6))
    strain[0, :3] = 5.0 * s0
    strain[1, :3] = -5.0 * s0
    strain[2, 5] = 50.0 * s0
    strain[3, 3] = 0.5 * s0
    out = _call(ReducedCVEUpdate(material, fallback=_kernel(material)), strain, np.zeros((4, 6)), np.ones(4))
    for key in ("stress", "tangent", "eta", "strain_v", "energy", "psi"):
        assert np.all(np.isfinite(out[key]))
    np.testing.assert_allclose(out["stress"][1, :3], -5.0 * s0 * material.k_vol, rtol=1.0e-10)   # elastic compression


# ----------------------------------------------------------------------
# Fallback, validation, tables
# ----------------------------------------------------------------------

class _NoIterations(ReducedCVEUpdate):
    """PRIVATE: a reduced update that never converges at active points."""
    MAX_ITER = 0


def test_points_the_reduced_newton_does_not_converge_at_are_recomputed_by_the_fallback():
    material = _material(2, "r2_2L")
    strain, strain_v_old, damage = _states(material, 400, seed=31)
    reference = _call(_kernel(material), strain, strain_v_old, damage)
    exact = _call(ReducedCVEUpdate(material), strain, strain_v_old, damage)
    ours = _call(_NoIterations(material, fallback=_kernel(material)), strain, strain_v_old, damage)
    active = np.any(exact["lambda_c"] > 0.0, axis=1)
    assert ours["result"] == active.sum() > 0
    np.testing.assert_array_equal(ours["stress"][active], reference["stress"][active])
    np.testing.assert_array_equal(ours["stress"][~active], exact["stress"][~active])
    with pytest.raises(RuntimeError, match="did not converge"):
        _call(_NoIterations(material), strain, strain_v_old, damage)


def test_kernel_writes_nan_at_a_point_it_cannot_update_instead_of_raising():
    # An exception would be raised on one MPI rank only and leave the others waiting; NaN
    # outputs reach the global residual on every rank instead.
    material = _material(3, "r2_2L")
    strain, strain_v_old, damage = _states(material, 20, seed=5)
    reference = _call(_kernel(material), strain, strain_v_old, damage)
    strain[7, 1] = np.nan
    result = _call(_kernel(material), strain, strain_v_old, damage)
    assert np.all(np.isnan(result["stress"][7])) and np.all(np.isnan(result["tangent"][7]))
    assert np.isnan(result["energy"][7]) and np.isnan(result["psi"][7])
    others = np.arange(20) != 7
    np.testing.assert_array_equal(result["stress"][others], reference["stress"][others])


@pytest.mark.parametrize("change, message", [
    ({"failure_type": "r3"}, "Unknown FailureType"),
    ({"nd": 4}, "nd must be"),
    ({"f_s": 0.3e6}, "f_s > f_t"),
    ({"kappa": 0.0}, "kappa"),
    ({"shear": -1.0}, "positive shear"),
    ({"f_t": 0.0}, "positive strengths"),
    ({"viscosity": -1.0}, "viscosity"),
    ({"G2_over_G1": 0.0}, "G2_over_G1"),
    ({"failure_type": "r2_DP", "DP_q": 0.1e6}, "DP criteria"),
    ({"failure_type": "r2_2L_DPa"}, "Unknown FailureType"),
])
def test_reduced_material_rejects_invalid_constants(change, message):
    values = dict(nd=3, lame=7.0e9, shear=3.5e9, bulk=9.0e9, viscosity=1.0e8, n_exp=3.0, f_t=F_T, f_s=F_S,
                  kappa=KAPPA, failure_type="r2_2L")
    values.update(change)
    with pytest.raises(ValueError, match=message):
        ReducedCVEMaterial(**values)


def test_reduced_material_uses_the_kernels_bulk_modulus():
    common = dict(lame=7.0e9, shear=3.5e9, bulk=1.0, viscosity=0.0, n_exp=1.0, f_t=F_T, f_s=F_S, kappa=KAPPA,
                  failure_type="r2_DP", DP_phi=30.0)
    intrinsic = ReducedCVEMaterial.from_model_constants(true_2d=True, **common)
    three_d = ReducedCVEMaterial.from_model_constants(true_2d=False, **common)
    assert intrinsic.nd == 2 and intrinsic.bulk == 7.0e9 + 3.5e9
    assert three_d.nd == 3 and three_d.bulk == 1.0 and three_d.DP_q == F_S
    assert intrinsic.num_multipliers == 1 and _material(3, "r2_2L").num_multipliers == 2
    with pytest.raises(ValueError, match="3-D"):
        ReducedCVEUpdate(intrinsic, plane_stress=True)


def test_creep_only_table_solves_the_scalar_creep_equation():
    axis, s_v, ds_v = creep_only_table(GLEN_N)
    log_c = axis[0] + axis[1] * np.arange(s_v.size)
    x = -np.expm1(-s_v)
    c = 10.0 ** log_c
    inside = (log_c > -20.0) & (log_c < 10.0)
    np.testing.assert_allclose(x[inside], (c * (1.0 - x) ** GLEN_N)[inside], rtol=1.0e-12)
    fd = np.gradient(s_v, axis[1])
    np.testing.assert_allclose(ds_v[inside][1:-1], fd[inside][1:-1], rtol=1.0e-5, atol=1.0e-12)


# ----------------------------------------------------------------------
# Compiled bodies as Python (coverage of the numba code)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("failure_type", TYPES)
def test_compiled_helpers_match_their_python_bodies(failure_type):
    material = _material(3, failure_type)
    c = material.constants()
    s0 = material.f_t / material.youngs
    for eta_v, eta_q, e_v, d in ((2.0 * s0, 3.0 * s0, 4.0 * s0, 0.5), (-1.0e-3 * s0, 2.0 * s0, -20.0 * s0, 1.0),
                                 (0.0, 0.0, -1.0e-2, 0.2), (1.0e-9, 1.0e-9, 1.0e-6, 1.0)):
        np.testing.assert_allclose(reduced._surface.py_func(eta_v, eta_q, e_v, d, c),
                                   reduced._surface(eta_v, eta_q, e_v, d, c), rtol=1e-13, atol=1e-300)
    for nl in (1, 2):
        for i in range(nl):
            for e_v in (3.0 * s0, -3.0 * s0):
                np.testing.assert_allclose(reduced._direction.py_func(nl, i, e_v, 2.0 * s0),
                                           reduced._direction(nl, i, e_v, 2.0 * s0), rtol=1e-14)
    strain, strain_v_old, damage = _states(material, 30, seed=41)
    update = ReducedCVEUpdate(material)
    creep_coef = update._creep_coefficient(DT)
    creep_rate = DT * (material.gamma / material.viscosity) ** material.n_exp
    nvec = np.zeros(6)
    for i in range(30):
        e_v, e_q = reduced._point_invariants.py_func(strain[i] - strain_v_old[i], 3, nvec)
        s_v, ds_v = reduced._creep_only.py_func(e_q, creep_coef, GLEN_N, *update.table)
        d = min(max(damage[i], KAPPA), 1.0)
        args = (e_v, e_q, d, s_v, ds_v, DT, creep_rate, c, update.TOL, update.MAX_ITER)
        np.testing.assert_allclose(reduced._point_state.py_func(*args), reduced._point_state(*args), rtol=1e-9,
                                   atol=1e-12 * max(F_T, F_S))
    n = damage.size
    outputs = [np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 6)), np.zeros((n, 36)), np.zeros(n), np.zeros(n),
               np.zeros(n), np.zeros((n, 2)), np.zeros(n, dtype=np.bool_)]
    reduced._update_points.py_func(strain, strain_v_old, damage, DT, c[None, :], *update.table,
                                   np.atleast_1d(creep_coef), update.TOL, update.MAX_ITER, *outputs)
    compiled = _call(update, strain, strain_v_old, damage)
    np.testing.assert_allclose(outputs[0], compiled["stress"], rtol=1e-9, atol=1e-6)
    planar = strain.copy()
    planar[:, 2:5] = 0.0
    args = (planar, strain_v_old, damage, DT, c[None, :], *update.table, np.atleast_1d(creep_coef), update.TOL,
            update.MAX_ITER, 1e-9, 60)
    np.testing.assert_allclose(reduced._plane_stress_e_zz.py_func(*args, np.zeros(n, dtype=np.bool_)),
                               reduced._plane_stress_e_zz(*args, np.zeros(n, dtype=np.bool_)), rtol=1e-9, atol=1e-15)


def test_small_helpers_match_their_python_bodies():
    axis, s_tab, ds_tab = creep_only_table(GLEN_N)
    for cnum in (0.0, 1.0e-35, 1.0e-3, 1.0, 1.0e20):
        np.testing.assert_allclose(reduced._point_creep.py_func(cnum, 2.0e-4, GLEN_N, axis, s_tab, ds_tab),
                                   reduced._point_creep(cnum, 2.0e-4, GLEN_N, axis, s_tab, ds_tab), rtol=1e-14)
    for x, y in ((1.0, 2.0), (0.0, 3.0), (-1.0, -2.0), (0.0, 0.0)):
        np.testing.assert_allclose(reduced._fischer_burmeister.py_func(x, y, 1.0), reduced._fischer_burmeister(x, y, 1.0))
    J = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]])
    rhs, rhs_py = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
    assert reduced._solve_small(J, rhs, 3) and reduced._solve_small.py_func(J, rhs_py, 3)
    np.testing.assert_allclose(J @ rhs, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rhs_py, rhs)
    assert not reduced._solve_small.py_func(np.zeros((3, 3)), np.ones(3), 2)
    for nd in (2, 3):
        e = np.array([1.0, 1.0, 1.0 if nd == 3 else 0.0, 0.0, 0.0, 0.0])
        nvec, nvec_py = np.zeros(6), np.zeros(6)
        assert reduced._point_invariants(e, nd, nvec) == reduced._point_invariants.py_func(e, nd, nvec_py)
        np.testing.assert_array_equal(nvec, nvec_py)
    tangent = np.random.default_rng(1).normal(size=(3, 36)) + 10.0 * np.eye(6).ravel()
    compiled, python = tangent.copy(), tangent.copy()
    reduced._condense_zz(compiled)
    reduced._condense_zz.py_func(python)
    np.testing.assert_allclose(python, compiled, rtol=1e-14)
    assert reduced._r2_dev_scale.py_func(3.0, 2) == 3.0 and reduced._merit.py_func(np.ones(3), 3, 2.0) == 6.0


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

STRAIN = 4.0e-6   # ~0.1 f_t / E


def _params(*, failure_type="r2_2L", true_2d=True, plane_stress=False, model_type="Solid:Cohesive_ViscoElastic"):
    class Params(ParamsTemplate):
        pass

    params = Params()
    params.MeshType = "Rectangle"
    params.dx = 1.0
    params.dim = 2
    params.Lx = 2.0
    params.Ly = 1.0
    params.ModelNames = []
    params.Field_Orders = {"u": 1}
    params.Solution_Steps = {"u": 0}
    params.gravity = 0.0
    params.dt = 0.1
    params.t = 0.0
    params.Newmark_Beta = 0.25
    params.Newmark_Gamma = 0.5
    params.Quadrature_Degree = 2
    params.Global_Measures = {}
    lame = YOUNGS * POISSON / (1.0 - POISSON**2) if true_2d else YOUNGS * POISSON / ((1 + POISSON) * (1 - 2 * POISSON))
    params.Materials = {"Ice": {"Lame": lame, "Shear": SHEAR, "Bulk": lame + SHEAR, "Density": 910.0,
                                "Visc_Type": "Glen", "Glen_A": GLEN_A, "Glen_n": GLEN_N, "Gc": 1.0, "ell": 0.25,
                                "f_t": F_T, "f_s": F_S, "kappa": KAPPA, "FailureType": failure_type,
                                "DP_phi": 30.0 if failure_type in reduced.DP_TYPES else 0.0,
                                "DP_q": 5.0e6 if failure_type in reduced.DP_TYPES else F_S}}
    params.Models = {"Solid": {"Type": model_type, "Material": "Ice", "DamageModel": "PhaseField",
                               "plane_stress": plane_stress, "true_2d": true_2d}}
    return params


def _build(model_class, params):
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
    model = model_class("Solid", params, mesh)
    mesh.finalize_test_functions()
    model.initialize_fields()
    mesh.Zero_Field("phasefield")
    return mesh, model


def _ip_fields(model_class, scale, **settings):
    """PRIVATE: IP fields of one update at a biaxial strain (``scale`` x STRAIN)."""
    mesh, model = _build(model_class, _params(model_type=model_class.MODEL_TYPE, **settings))
    mesh.Get_Field("u").interpolate(lambda x: np.vstack((scale * STRAIN * x[0] + 0.3 * scale * STRAIN * x[1],
                                                         0.6 * scale * STRAIN * x[1])))
    model.Update_Weak_Constants(None, model.params.dt, model.Step_u)
    return model, {name: mesh.Get_Field(name).x.array.copy()
                   for name in ("sigma", "dsigma_dStrainMandel", "Plastic_Strain", "LambdaField_coh", "etaNorm")}


@pytest.mark.parametrize("failure_type", TYPES)
def test_reduced_model_uses_the_reduced_update_for_every_failure_type(failure_type):
    _, model = _build(Cohesive_ViscoElastic, _params(failure_type=failure_type, true_2d=True))
    assert model.type == "Solid:Cohesive_ViscoElastic"
    assert isinstance(model._update_cve_state, ReducedCVEUpdate)
    assert model._update_cve_state.fallback is not None
    assert model.reduced_material.failure_type == failure_type
    assert model.reduced_material.nd == 2


@pytest.mark.parametrize("failure_type, true_2d", [("r2_2L", True), ("r2_2L", False), ("r1", True), ("r2_DP", False),
                                                   ("r2_2L_DP", False)])
def test_reduced_model_ip_update_matches_the_kernel_model_while_cracking(failure_type, true_2d):
    model, ours = _ip_fields(Cohesive_ViscoElastic, 200.0, failure_type=failure_type, true_2d=true_2d)
    _, kernel = _ip_fields(Cohesive_ViscoElastic_Kernel, 200.0, failure_type=failure_type, true_2d=true_2d)
    assert np.all(np.abs(ours["LambdaField_coh"]).reshape(-1, 2).max(axis=1) > 0.0)   # every point cracks
    for name in ("sigma", "Plastic_Strain", "LambdaField_coh", "etaNorm"):
        np.testing.assert_allclose(ours[name], kernel[name], rtol=1e-6, atol=1e-7 * np.abs(kernel[name]).max())
    np.testing.assert_allclose(ours["dsigma_dStrainMandel"], kernel["dsigma_dStrainMandel"], rtol=1e-4,
                               atol=1e-6 * np.abs(kernel["dsigma_dStrainMandel"]).max())


def test_reduced_model_plane_stress_has_zero_sigma_zz():
    model, ours = _ip_fields(Cohesive_ViscoElastic, 200.0, true_2d=False, plane_stress=True)
    sigma = ours["sigma"].reshape(-1, 6)
    assert model._update_cve_state.plane_stress
    assert np.abs(sigma[:, 2]).max() < 1e-8 * np.abs(sigma).max()


# ---------------------------------------------------------------------------------------------------
# Optional field "StrengthRatio"
# ---------------------------------------------------------------------------------------------------

def _strength_ratio_fields(scale, failure_type="r2_2L"):
    params = _params(failure_type=failure_type, true_2d=False)
    params.Models["Solid"]["Strength_Ratio_Field"] = True
    mesh, model = _build(Cohesive_ViscoElastic, params)
    mesh.Get_Field("u").interpolate(lambda x: np.vstack((scale * STRAIN * x[0] + 0.3 * scale * STRAIN * x[1],
                                                         0.6 * scale * STRAIN * x[1])))
    model.Update_Weak_Constants(None, model.params.dt, model.Step_u)
    return {name: mesh.Get_Field(name).x.array.copy() for name in ("sigma", "etaNorm", "StrengthRatio")}


@pytest.mark.parametrize("failure_type", ["r2_2L", "r2_2L_PSD"])
def test_strength_ratio_is_below_one_and_proportional_to_the_stress_inside_the_strength_surface(failure_type):
    one, half = _strength_ratio_fields(1.0, failure_type), _strength_ratio_fields(0.5, failure_type)
    assert np.all(one["etaNorm"] == 0.0)
    assert np.all((one["StrengthRatio"] > 0.0) & (one["StrengthRatio"] < 1.0))
    # creep is negligible at this stress, so half the strain gives half the ratio
    np.testing.assert_allclose(half["StrengthRatio"], 0.5 * one["StrengthRatio"], rtol=1e-6)


def test_strength_ratio_is_one_where_the_r2_2L_cohesive_strain_is_active():
    fields = _strength_ratio_fields(200.0, "r2_2L")
    assert np.all(fields["etaNorm"] > 0.0)
    np.testing.assert_allclose(fields["StrengthRatio"], 1.0, rtol=1e-6)


def test_strength_ratio_is_at_least_one_where_the_opening_only_cohesive_strain_is_active():
    fields = _strength_ratio_fields(200.0, "r2_2L_PSD")
    assert np.all(fields["etaNorm"] > 0.0)
    assert np.all(fields["StrengthRatio"] >= 1.0 - 1e-6)


def test_strength_ratio_field_is_optional_and_its_setting_is_validated():
    mesh, _ = _build(Cohesive_ViscoElastic, _params(true_2d=False))
    assert "StrengthRatio" not in mesh.fields
    params = _params(true_2d=False)
    params.Models["Solid"]["Strength_Ratio_Field"] = "yes"
    with pytest.raises(ValueError, match="Strength_Ratio_Field"):
        _build(Cohesive_ViscoElastic, params)
    params = _params(failure_type="r1", true_2d=False)
    params.Models["Solid"]["Strength_Ratio_Field"] = True
    with pytest.raises(ValueError, match="r2_2L"):
        _build(Cohesive_ViscoElastic, params)


# ---------------------------------------------------------------------------------------------------
# Failure types, the kernel model and the creep-free forward
# ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("model_class", [Cohesive_ViscoElastic, Cohesive_ViscoElastic_Kernel, Cohesive_LinearElastic])
@pytest.mark.parametrize("failure_type", ["r1TO", "r1_1L", "r1_DP_1L", "ShearOnly", "r2_2L_DPi", "r2_2L_DPa", "r3"])
def test_cohesive_models_reject_unsupported_failure_types(model_class, failure_type):
    params = _params(model_type=model_class.MODEL_TYPE)
    params.Materials["Ice"]["FailureType"] = failure_type
    with pytest.raises(ValueError, match="supported: r1, r1_DP, r2, r2_DP, r2_2L, r2_2L_DP, r2_2L_PSD"):
        _build(model_class, params)


def test_kernel_model_uses_the_cpp_kernel_and_three_multipliers_for_the_opening_only_type():
    _, model = _build(Cohesive_ViscoElastic_Kernel, _params(failure_type="r2_2L_PSD", true_2d=False))
    assert model.type == "Solid:Cohesive_ViscoElastic_Kernel"
    assert isinstance(model._update_cve_state, CohesiveViscoElasticKernels)
    assert model.mesh.Get_Field("LambdaField_coh").x.array.size == 3 * model.mesh.Get_Field("dam_elastic").x.array.size


def test_linear_elastic_forward_is_the_model_without_creep():
    elastic, ours = _ip_fields(Cohesive_LinearElastic, 200.0, true_2d=False)   # material asks for Glen creep
    params = _params(true_2d=False)
    params.Materials["Ice"]["Visc_Type"] = "None"
    mesh, model = _build(Cohesive_ViscoElastic, params)
    mesh.Get_Field("u").interpolate(lambda x: np.vstack((200.0 * STRAIN * x[0] + 0.3 * 200.0 * STRAIN * x[1],
                                                         0.6 * 200.0 * STRAIN * x[1])))
    model.Update_Weak_Constants(None, model.params.dt, model.Step_u)
    assert elastic.type == "Solid:Cohesive_LinearElastic"
    assert elastic.viscosity_type == "None" and elastic.viscosity == 0.0
    assert isinstance(elastic._update_cve_state, ReducedCVEUpdate)
    np.testing.assert_allclose(ours["Plastic_Strain"], 0.0, atol=1.0e-20)   # round-off of the creep-free update
    assert np.abs(ours["LambdaField_coh"]).max() > 0.0
    for name in ("sigma", "LambdaField_coh", "etaNorm", "dsigma_dStrainMandel"):
        np.testing.assert_array_equal(ours[name], mesh.Get_Field(name).x.array)
