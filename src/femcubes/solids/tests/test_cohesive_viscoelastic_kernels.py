"""Direct tests for cohesive visco-elastic C++ kernels."""

import numpy as np
import pytest

from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    CohesiveViscoElasticKernels,
    CohesiveViscoElasticKernelsVariableProps,
)
from femcubes.core.utils.maths_utils import DevMat, StiffnessMatrixMandel


ICE_YOUNGS = 9.5e9
ICE_POISSON = 0.35
ICE_SHEAR = ICE_YOUNGS / (2.0 * (1.0 + ICE_POISSON))
ICE_BULK = ICE_YOUNGS / (3.0 * (1.0 - 2.0 * ICE_POISSON))
ICE_LAME = 2.0 * ICE_SHEAR * ICE_POISSON / (1.0 - 2.0 * ICE_POISSON)
ICE_VISCOSITY = (1.0 / (2.0e-24)) ** (1.0 / 3.0)
ICE_FT = 0.4e6
ICE_FS = 1.0e6
ICE_DP_PHI = np.degrees(np.arctan(3.0 / (1.0 + ICE_POISSON)))
STRICT_RETURN_MAP_TOL = 1.0e-6


def test_standard_j2_cohesive_kernel_reproduces_glen_flow_without_fracture():
    """Its viscous branch retains Glen creep when cohesive modes are inactive."""
    glen_a = 1.0e-24
    n_exp = 3.0
    eta_vp = (3.0 ** ((n_exp + 1.0) / 2.0) / (2.0 * glen_a)) ** (1.0 / n_exp)
    strain_v = np.zeros(6)
    stress = np.zeros(6)
    CohesiveViscoElasticKernels(
        1.0e9, 1.0e9, 5.0e8, eta_vp, n_exp,
        1.0e12, 1.0e12, 1.0, 0.0, 1.0e12, 1.0e-3, "r2", False,
    )(
        np.array([1.0e-3, -4.0e-4, 0.0, 2.0e-4, 0.0, 0.0]),
        10.0,
        np.ones(1),
        stress,
        np.zeros(6),
        np.zeros(6),
        strain_v,
        np.zeros(6),
        np.zeros(36),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(2),
        np.zeros(2),
    )

    deviatoric_stress = stress.copy()
    deviatoric_stress[:3] -= np.mean(stress[:3])
    tau_e = np.sqrt(0.5 * np.dot(deviatoric_stress, deviatoric_stress))
    expected_rate = glen_a * tau_e ** (n_exp - 1.0) * deviatoric_stress
    np.testing.assert_allclose(strain_v / 10.0, expected_rate, rtol=2.0e-4, atol=1.0e-14)


def _make_kernel(
    plane_stress=False, *, true_2d=False, viscosity=1.0e50, f_t=1.0e12, f_s=1.0e12
):
    return CohesiveViscoElasticKernels(
        10.0,    # bulk
        4.0,     # lame
        3.0,     # shear
        viscosity,
        1.0,
        f_t,
        f_s,
        1.0,
        0.0,
        f_s,
        1.0e-3,
        "r2",
        plane_stress,
        true_2d,
    )


def _run_single_ip(
    plane_stress=False,
    *,
    true_2d=False,
    strain=None,
    dam=1.0,
    viscosity=1.0e50,
    f_t=1.0e12,
    f_s=1.0e12,
):
    strains = np.zeros(6) if strain is None else np.asarray(strain, dtype=float).copy()
    if strain is None:
        strains[0] = 1.0e-3

    stress = np.zeros(6)
    eta_c = np.zeros(6)
    eta_c_old = np.zeros(6)
    strain_v = np.zeros(6)
    ep_old = np.zeros(6)
    tangent = np.zeros(36)
    elastic_energy = np.zeros(1)
    psi = np.zeros(1)
    lambda_v = np.zeros(1)
    lambda_c = np.zeros(2)
    lambda_c_old = np.zeros(2)

    _make_kernel(
        plane_stress, true_2d=true_2d, viscosity=viscosity, f_t=f_t, f_s=f_s
    )(
        strains,
        1.0e-3,
        np.asarray([dam]),
        stress,
        eta_c,
        eta_c_old,
        strain_v,
        ep_old,
        tangent,
        elastic_energy,
        psi,
        lambda_v,
        lambda_c,
        lambda_c_old,
    )
    return strains, stress, tangent.reshape(6, 6), lambda_v, lambda_c, eta_c


def _run_ice_case(
    failure_type, strain, dt, damage, strain_v_old, *, plane_stress=False, true_2d=False
):
    kernel = CohesiveViscoElasticKernels(
        ICE_BULK, ICE_LAME, ICE_SHEAR, ICE_VISCOSITY, 3.0,
        ICE_FT, ICE_FS, 1.0, ICE_DP_PHI, 9.0e6,
        1.0e-4, failure_type, plane_stress, true_2d,
    )

    strain = np.asarray(strain, dtype=float).copy()
    stress = np.zeros(6)
    eta_c = np.zeros(6)
    strain_v = np.zeros(6)
    tangent = np.zeros(36)
    lambda_v = np.zeros(1)
    lambda_c = np.zeros(2)
    kernel(
        strain, dt, np.asarray([damage]), stress,
        eta_c, np.zeros(6), strain_v, np.asarray(strain_v_old, dtype=float), tangent,
        np.zeros(1), np.zeros(1), lambda_v, lambda_c, np.zeros(2),
    )
    return stress, tangent, eta_c, strain_v, lambda_v, lambda_c


def _assert_strict_ice_residual(
    failure_type, strain, dt, damage, strain_v_old, *, plane_stress=False, true_2d=False
):
    """Assert the scaled residual contract used by every CVE solve route."""
    kernel = CohesiveViscoElasticKernels(
        ICE_BULK, ICE_LAME, ICE_SHEAR, ICE_VISCOSITY, 3.0,
        ICE_FT, ICE_FS, 1.0, ICE_DP_PHI, 9.0e6,
        1.0e-4, failure_type, plane_stress, true_2d,
    )
    strain = np.asarray(strain, dtype=float)
    strain_v_old = np.asarray(strain_v_old, dtype=float)
    nl_c = 2 if failure_type in {"r1", "r2_2L", "r2_2L_DP", "r1_DP"} else 1
    kf = kernel.inspect_kf(
        strain,
        np.zeros(6),
        np.zeros(6),
        strain_v_old,
        np.zeros(nl_c),
        np.zeros(nl_c),
        0.0,
        dt,
        damage,
    )
    residual = np.asarray(kf["F"])

    elastic_tangent = StiffnessMatrixMandel(ICE_LAME, ICE_SHEAR)
    elastic_modulus = np.max(np.abs(elastic_tangent))
    strain_scale = max(np.linalg.norm(strain), np.linalg.norm(strain_v_old))
    stress_scale = max(
        elastic_modulus * strain_scale,
        elastic_modulus * 1.0e-12 / STRICT_RETURN_MAP_TOL,
    )

    trial_strain = strain.copy()
    if plane_stress:
        trial_strain[2] = (
            strain_v_old[2]
            - ICE_LAME / (ICE_LAME + 2.0 * ICE_SHEAR)
            * (
                strain[0] - strain_v_old[0]
                + strain[1] - strain_v_old[1]
            )
        )
    trial_stress = elastic_tangent @ (trial_strain - strain_v_old)
    trial_j2 = max(0.5 * trial_stress @ DevMat() @ trial_stress, 1.0e-20)
    lambda_v_trial = dt * (np.sqrt(trial_j2) / ICE_VISCOSITY) ** 3.0
    lambda_v_scale = max(
        strain_scale,
        lambda_v_trial,
        1.0e-8 / STRICT_RETURN_MAP_TOL,
    )
    damaged_strength_scale = max(
        damage * max(ICE_FT, ICE_FS),
        1.0e-12 * max(ICE_FT, ICE_FS),
    )

    scaled_components = [
        np.linalg.norm(residual[:6]) / stress_scale,
        abs(residual[6]) / lambda_v_scale,
        np.linalg.norm(residual[7:7 + nl_c]) / damaged_strength_scale,
    ]
    if plane_stress:
        scaled_components.append(abs(residual[7 + nl_c]) / stress_scale)

    assert max(scaled_components) < STRICT_RETURN_MAP_TOL


def test_cve_plane_stress_solves_and_writes_ezz():
    strains, stress, tangent, _, _, _ = _run_single_ip(plane_stress=True)

    elastic_tangent = StiffnessMatrixMandel(4.0, 3.0)
    expected_tangent = elastic_tangent - np.outer(elastic_tangent[:, 2], elastic_tangent[2, :]) / elastic_tangent[2, 2]

    np.testing.assert_allclose(strains[2], -4.0e-4, rtol=1.0e-10, atol=1.0e-14)
    np.testing.assert_allclose(stress[2], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1.0e-10, atol=1.0e-12)


def test_cve_plane_strain_keeps_ezz_fixed():
    strains, stress, _, _, _, _ = _run_single_ip(plane_stress=False)

    np.testing.assert_allclose(strains[2], 0.0, atol=0.0)
    assert stress[2] > 0.0


def test_cve_true_2d_zero_fills_out_of_plane_components_and_uses_intrinsic_stiffness():
    strains, stress, tangent, _, _, eta_c = _run_single_ip(
        true_2d=True,
        strain=np.asarray([1.0e-3, -2.0e-4, 3.0e-3, 4.0e-3, -5.0e-3, 6.0e-4]),
    )

    expected_tangent = np.zeros((6, 6))
    expected_tangent[0, 0] = expected_tangent[1, 1] = 10.0
    expected_tangent[0, 1] = expected_tangent[1, 0] = 4.0
    expected_tangent[5, 5] = 6.0

    np.testing.assert_allclose(strains[[2, 3, 4]], 0.0, atol=0.0)
    np.testing.assert_allclose(stress[[2, 3, 4]], 0.0, atol=0.0)
    np.testing.assert_allclose(eta_c[[2, 3, 4]], 0.0, atol=0.0)
    np.testing.assert_allclose(tangent, expected_tangent, rtol=1.0e-10, atol=1.0e-12)


def test_cve_true_2d_has_no_out_of_plane_newton_unknown():
    kernel = _make_kernel(true_2d=True)
    kf = kernel.inspect_kf(
        np.asarray([1.0e-3, 0.0, 2.0e-3, 4.0e-3, -5.0e-3, 1.0e-4]),
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        np.zeros(1),
        np.zeros(1),
        0.0,
        1.0,
        1.0,
    )

    K = kf["K"]
    C = kf["C"]
    assert K.shape == (8, 8)
    assert C.shape == (8, 6)
    np.testing.assert_allclose(K[[2, 3, 4]], np.eye(8)[[2, 3, 4]], atol=0.0)
    np.testing.assert_allclose(C[:, [2, 3, 4]], 0.0, atol=0.0)


def test_cve_true_2d_rejects_plane_stress_combination():
    with np.testing.assert_raises_regex(ValueError, "cannot both be enabled"):
        _make_kernel(plane_stress=True, true_2d=True)


def test_cve_variable_properties_true_2d_zero_fills_out_of_plane_components():
    kernel = CohesiveViscoElasticKernelsVariableProps(
        1.0, 1.0e12, 1.0, 0.0, 1.0e12, 1.0e-3, "r2", False, True
    )
    strain = np.asarray([1.0e-3, -2.0e-4, 3.0e-3, 4.0e-3, -5.0e-3, 6.0e-4])
    stress = np.zeros(6)
    eta_c = np.zeros(6)
    strain_v = np.zeros(6)
    tangent = np.zeros(36)

    kernel(
        strain,
        1.0e-3,
        np.ones(1),
        np.asarray([4.0]),
        np.asarray([3.0]),
        np.asarray([1.0e50]),
        np.asarray([1.0e12]),
        stress,
        eta_c,
        np.zeros(6),
        strain_v,
        np.zeros(6),
        tangent,
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(2),
        np.zeros(2),
    )

    np.testing.assert_allclose(strain[[2, 3, 4]], 0.0, atol=0.0)
    np.testing.assert_allclose(stress[[2, 3, 4]], 0.0, atol=0.0)
    np.testing.assert_allclose(eta_c[[2, 3, 4]], 0.0, atol=0.0)
    np.testing.assert_allclose(strain_v[[2, 3, 4]], 0.0, atol=0.0)
    np.testing.assert_allclose(tangent.reshape(6, 6)[[2, 3, 4], :], 0.0, atol=0.0)
    np.testing.assert_allclose(tangent.reshape(6, 6)[:, [2, 3, 4]], 0.0, atol=0.0)


def test_cve_plane_strain_r1_active_set_stall_recovers():
    strain = np.asarray([-7.071067811865475e-4, 7.071067811865475e-4, 0.0, 0.0, 0.0, 0.0])
    strain_v_old = 0.5 * strain
    outputs = _run_ice_case("r1", strain, 1.0e-10, 1.0, strain_v_old)

    for output in outputs:
        assert np.isfinite(output).all()
    _assert_strict_ice_residual("r1", strain, 1.0e-10, 1.0, strain_v_old)


def test_cve_plane_strain_r2_dp_residual_damage_stall_recovers():
    strain = np.asarray([-2.182178902359924e-4, 5.45544725589981e-5, 0.0, 0.0, 0.0, -1.091089451179962e-4])
    outputs = _run_ice_case("r2_DP", strain, 1.0e-10, 1.0e-4, np.zeros(6))

    for output in outputs:
        assert np.isfinite(output).all()
    _assert_strict_ice_residual(
        "r2_DP", strain, 1.0e-10, 1.0e-4, np.zeros(6)
    )


def test_cve_plane_strain_r2_large_dt_continuation_recovers():
    strain = np.asarray([7.071067811865475e-2, 7.071067811865475e-2, 0.0, 0.0, 0.0, 0.0])
    strain_v_old = -0.5 * strain
    strain_v_old[2] = -(strain_v_old[0] + strain_v_old[1])
    outputs = _run_ice_case("r2", strain, 1.0e-4, 1.0, strain_v_old)

    for output in outputs:
        assert np.isfinite(output).all()
    _assert_strict_ice_residual(
        "r2", strain, 1.0e-4, 1.0, strain_v_old
    )


@pytest.mark.parametrize(
    ("plane_stress", "true_2d"),
    [(False, False), (True, False), (False, True)],
)
def test_cve_r2_2l_active_volumetric_and_deviatoric_modes_remain_finite(
    plane_stress, true_2d
):
    """The split r2 criterion updates both cohesive modes without NaN output."""
    strain = np.asarray([2.0e-3, 4.0e-4, 0.0, 0.0, 0.0, 7.0e-4])
    outputs = _run_ice_case(
        "r2_2L", strain, 1.0, 1.0, np.zeros(6),
        plane_stress=plane_stress, true_2d=true_2d,
    )

    for output in outputs:
        assert np.isfinite(output).all()

    lambda_c = outputs[-1]
    assert lambda_c[0] > 0.0
    assert lambda_c[1] > 0.0


@pytest.mark.parametrize(
    ("plane_stress", "true_2d"),
    [(False, False), (True, False), (False, True)],
)
def test_cve_r2_2l_active_modes_satisfy_strict_local_residual(plane_stress, true_2d):
    """The split r2 criterion should satisfy the normal local residual contract."""
    strain = np.asarray([2.0e-3, 4.0e-4, 0.0, 0.0, 0.0, 7.0e-4])
    _assert_strict_ice_residual(
        "r2_2L", strain, 1.0, 1.0, np.zeros(6),
        plane_stress=plane_stress, true_2d=true_2d,
    )


def test_cve_r2_2l_high_damage_compressive_penalty_retains_deviatoric_precision():
    """A trace penalty must not contaminate the orthogonal cohesive mode."""
    f_t = 0.2e6
    f_s = 5.0e6
    kappa = 1.0e-6
    damage = 1.65089e-6
    dt = 60.0
    strain = np.asarray([
        0.00273354, -0.00273565, 0.0, 0.0, 0.0, -0.00108401,
    ])
    strain_v_old = np.asarray([
        1.24302e-5, -2.29897e-5, 1.05595e-5,
        0.0, 0.0, -1.37642e-6,
    ])
    kernel = CohesiveViscoElasticKernels(
        ICE_BULK, ICE_LAME, ICE_SHEAR, ICE_VISCOSITY, 3.0,
        f_t, f_s, 1.0, ICE_DP_PHI, f_s,
        kappa, "r2_2L", False,
    )

    stress = np.zeros(6)
    eta_c = np.zeros(6)
    strain_v = np.zeros(6)
    tangent = np.zeros(36)
    lambda_v = np.zeros(1)
    lambda_c = np.zeros(2)
    kernel(
        strain.copy(), dt, np.asarray([damage]), stress,
        eta_c, np.zeros(6), strain_v, strain_v_old, tangent,
        np.zeros(1), np.zeros(1), lambda_v, lambda_c, np.zeros(2),
    )

    assert np.isfinite(stress).all()
    assert np.isfinite(tangent).all()
    np.testing.assert_allclose(lambda_c[0], 0.0, atol=1.0e-14)
    assert lambda_c[1] > 0.0

    kf = kernel.inspect_kf(
        strain,
        np.zeros(6),
        np.zeros(6),
        strain_v_old,
        np.zeros(2),
        np.zeros(2),
        0.0,
        dt,
        damage,
    )
    residual = np.asarray(kf["F"])
    cohesive_scale = damage * f_s
    assert np.linalg.norm(residual[7:9]) / cohesive_scale < 1.0e-8
    assert np.isfinite(np.asarray(kf["C"])).all()


@pytest.mark.parametrize("plane_stress", [False, True])
def test_cve_previous_converged_state_restart_matches_cold_solution(plane_stress):
    """Warm and cold guesses must produce equivalent converged local states."""
    damage = 1.65089e-6
    dt = 60.0
    initial_strain = np.asarray([
        0.00273354, -0.00273565, 0.0, 0.0, 0.0, -0.00108401,
    ])
    target_strain = initial_strain + np.asarray([
        2.0e-6, -1.0e-6, 0.0, 0.0, 0.0, 5.0e-7,
    ])
    strain_v_old = np.asarray([
        1.24302e-5, -2.29897e-5, 1.05595e-5,
        0.0, 0.0, -1.37642e-6,
    ])
    kernel = CohesiveViscoElasticKernels(
        ICE_BULK, ICE_LAME, ICE_SHEAR, ICE_VISCOSITY, 3.0,
        0.2e6, 5.0e6, 1.0, ICE_DP_PHI, 5.0e6,
        1.0e-6, "r2_2L", plane_stress,
    )

    def evaluate(strain, state=None):
        if state is None:
            state = {
                "stress": np.zeros(6),
                "eta": np.zeros(6),
                "strain_v": np.zeros(6),
                "tangent": np.zeros(36),
                "energy": np.zeros(1),
                "psi": np.zeros(1),
                "lambda_v": np.zeros(1),
                "lambda_c": np.zeros(2),
            }
        kernel(
            strain.copy(), dt, np.asarray([damage]), state["stress"],
            state["eta"], np.zeros(6), state["strain_v"], strain_v_old,
            state["tangent"], state["energy"], state["psi"],
            state["lambda_v"], state["lambda_c"], np.zeros(2),
        )
        return state

    warm_state = evaluate(initial_strain)
    assert warm_state["lambda_c"][1] > 0.0
    warm_state = evaluate(target_strain, warm_state)
    cold_state = evaluate(target_strain)

    # At this tiny damage the plane-stress e_zz/lambda_c direction is nearly
    # singular, so warm and cold starts may converge to slightly different
    # points within the residual tolerance. The tangent is therefore compared
    # relative to its own magnitude rather than entry by entry.
    tangent_scale = np.abs(cold_state["tangent"]).max()
    for field in ("stress", "eta", "strain_v", "tangent", "energy", "psi", "lambda_v", "lambda_c"):
        absolute_tolerance = {
            "eta": 2.0e-7,
            "tangent": 5.0e-3 * tangent_scale,
            "lambda_c": 1.0e-7,
        }.get(field, 1.0e-11)
        relative_tolerance = 5.0e-4 if field == "tangent" else 2.0e-7
        np.testing.assert_allclose(
            warm_state[field], cold_state[field],
            rtol=relative_tolerance, atol=absolute_tolerance,
            err_msg=field,
        )


def test_cve_plane_strain_r2_dp_coupled_continuation_recovers():
    strain = np.asarray([0.0, -1.0e-5, 0.0, 0.0, 0.0, 0.0])
    strain_v_old = np.asarray([0.0, 2.5e-6, -2.5e-6, 0.0, 0.0, 0.0])
    outputs = _run_ice_case("r2_DP", strain, 1.0, 1.0e-4, strain_v_old)

    for output in outputs:
        assert np.isfinite(output).all()
    _assert_strict_ice_residual(
        "r2_DP", strain, 1.0, 1.0e-4, strain_v_old
    )


def test_cve_plane_stress_r2_dp_large_dt_coupled_continuation_recovers():
    strain = np.asarray([-1.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0])
    strain_v_old = np.asarray([-5.0e-3, 0.0, 5.0e-3, 0.0, 0.0, 0.0])
    outputs = _run_ice_case(
        "r2_DP", strain, 3600.0, 1.0e-4, strain_v_old, plane_stress=True,
    )

    for output in outputs:
        assert np.isfinite(output).all()
    _assert_strict_ice_residual(
        "r2_DP",
        strain,
        3600.0,
        1.0e-4,
        strain_v_old,
        plane_stress=True,
    )

def test_cve_plane_stress_inspection_includes_ezz_unknown():
    kernel = _make_kernel(plane_stress=True)
    kf = kernel.inspect_kf(
        np.asarray([1.0e-3, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.zeros(6),
        np.zeros(6),
        np.zeros(6),
        np.zeros(1),
        np.zeros(1),
        0.0,
        1.0,
        1.0,
    )

    assert kf["K"].shape == (9, 9)
    assert kf["C"].shape == (9, 6)


def test_cve_plane_stress_viscous_tangent_matches_finite_difference_at_high_damage():
    base_strain = np.asarray([1.0e-2, 2.0e-3, 0.0, 0.0, 0.0, 1.0e-3])

    def evaluate(strain):
        _, stress, tangent, _, _, _ = _run_single_ip(
            plane_stress=True,
            strain=strain,
            dam=1.0e-3,
            viscosity=5.0,
        )
        return stress, tangent

    stress, tangent = evaluate(base_strain)
    assert np.isfinite(stress).all()
    assert np.isfinite(tangent).all()

    h = 1.0e-7
    for col in (0, 1, 5):
        direction = np.zeros(6)
        direction[col] = h
        stress_plus, _ = evaluate(base_strain + direction)
        stress_minus, _ = evaluate(base_strain - direction)
        fd_col = (stress_plus - stress_minus) / (2.0 * h)
        np.testing.assert_allclose(tangent[:, col], fd_col, rtol=1.0e-5, atol=5.0e-7)


def test_cve_plane_strain_viscous_tangent_matches_finite_difference_at_high_damage():
    base_strain = np.asarray([1.0e-2, 2.0e-3, 0.0, 0.0, 0.0, 1.0e-3])

    def evaluate(strain):
        _, stress, tangent, _, _, _ = _run_single_ip(
            plane_stress=False,
            strain=strain,
            dam=1.0e-3,
            viscosity=5.0,
        )
        return stress, tangent

    stress, tangent = evaluate(base_strain)
    assert np.isfinite(stress).all()
    assert np.isfinite(tangent).all()

    h = 1.0e-7
    for col in (0, 1, 5):
        direction = np.zeros(6)
        direction[col] = h
        stress_plus, _ = evaluate(base_strain + direction)
        stress_minus, _ = evaluate(base_strain - direction)
        fd_col = (stress_plus - stress_minus) / (2.0 * h)
        np.testing.assert_allclose(tangent[:, col], fd_col, rtol=1.0e-5, atol=5.0e-7)


def test_cve_plane_stress_active_high_damage_update_stays_finite():
    strains, stress, tangent, lambda_v, lambda_c, eta_c = _run_single_ip(
        plane_stress=True,
        strain=np.asarray([1.0e-2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        dam=1.0e-3,
        viscosity=5.0,
        f_t=2.0e-3,
        f_s=1.5e-3,
    )

    assert np.isfinite(strains).all()
    assert np.isfinite(stress).all()
    assert np.isfinite(tangent).all()
    assert np.isfinite(eta_c).all()
    np.testing.assert_allclose(stress[2], 0.0, atol=1.0e-12)
    assert lambda_v[0] >= 0.0
    assert lambda_c[0] >= 0.0


def test_cve_plane_stress_below_strength_r1_dp_remains_cohesively_elastic():
    youngs = 9.5e9
    poisson = 0.35
    shear = youngs / (2.0 * (1.0 + poisson))
    bulk = youngs / (3.0 * (1.0 - 2.0 * poisson))
    lame = 2.0 * shear * poisson / (1.0 - 2.0 * poisson)
    viscosity = (1.0 / (2.0e-24)) ** (1.0 / 3.0)
    dp_phi = np.degrees(np.arctan(3.0 / (1.0 + poisson)))

    kernel = CohesiveViscoElasticKernels(
        bulk, lame, shear, viscosity, 3.0,
        0.4e6, 1.0e6, 1.0, dp_phi, 9.0e6,
        1.0e-4, "r1_DP", True,
    )
    strain = np.asarray([8.48031e-6, -8.99414e-7, 0.0, 0.0, 0.0, 1.35610e-6])
    strain_v_old = np.asarray([1.42780e-11, -4.03352e-12, -1.02445e-11, 0.0, 0.0, 2.64770e-12])
    stress = np.zeros(6)
    eta_c = np.zeros(6)
    strain_v = np.zeros(6)
    tangent = np.zeros(36)
    elastic_energy = np.zeros(1)
    psi = np.zeros(1)
    lambda_v = np.zeros(1)
    lambda_c = np.zeros(2)

    kernel(
        strain, 0.1, np.ones(1), stress,
        eta_c, np.zeros(6), strain_v, strain_v_old, tangent,
        elastic_energy, psi, lambda_v, lambda_c, np.zeros(2),
    )

    assert np.isfinite(stress).all()
    assert np.isfinite(tangent).all()
    assert np.isfinite(strain).all()
    np.testing.assert_allclose(eta_c, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(lambda_c, 0.0, atol=1.0e-12)

    near_zero_strain = np.asarray([1.84851e-8, -1.84589e-8, 0.0, 0.0, 0.0, -5.55812e-8])
    near_zero_stress = np.zeros(6)
    near_zero_tangent = np.zeros(36)
    near_zero_lambda_c = np.zeros(2)
    kernel(
        near_zero_strain, 0.1, np.ones(1), near_zero_stress,
        np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), near_zero_tangent,
        np.zeros(1), np.zeros(1), np.zeros(1), near_zero_lambda_c, np.zeros(2),
    )

    assert np.isfinite(near_zero_stress).all()
    assert np.isfinite(near_zero_tangent).all()
    np.testing.assert_allclose(near_zero_stress[2], 0.0, atol=1.0e-8)
    np.testing.assert_allclose(near_zero_lambda_c, 0.0, atol=1.0e-12)


def test_cve_plane_stress_r2_dp_tiny_dt_active_set_stall_stays_finite():
    youngs = 9.5e9
    poisson = 0.35
    shear = youngs / (2.0 * (1.0 + poisson))
    bulk = youngs / (3.0 * (1.0 - 2.0 * poisson))
    lame = 2.0 * shear * poisson / (1.0 - 2.0 * poisson)
    viscosity = (1.0 / (2.0e-24)) ** (1.0 / 3.0)
    dp_phi = np.degrees(np.arctan(3.0 / (1.0 + poisson)))
    kernel = CohesiveViscoElasticKernels(
        bulk, lame, shear, viscosity, 3.0,
        0.4e6, 1.0e6, 1.0, dp_phi, 9.0e6,
        1.0e-4, "r2_DP", True,
    )

    strain = np.asarray([-4.73981e-6, -6.51825e-6, 0.0, 0.0, 0.0, -2.95645e-6])
    stress = np.zeros(6)
    tangent = np.zeros(36)
    lambda_c = np.zeros(2)
    kernel(
        strain, 1.0e-10, np.asarray([1.0e-3]), stress,
        np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), tangent,
        np.zeros(1), np.zeros(1), np.zeros(1), lambda_c, np.zeros(2),
    )

    assert np.isfinite(stress).all()
    assert np.isfinite(tangent).all()
    assert lambda_c[0] >= 0.0


def test_cve_variable_properties_match_independent_constant_property_updates():
    n_ip = 2
    strains = np.asarray(
        [
            [1.0e-3, 2.0e-4, 0.0, 1.0e-4, 0.0, 0.0],
            [4.0e-4, 1.2e-3, 0.0, 0.0, 0.0, 2.0e-4],
        ]
    )
    lame = np.asarray([4.0, 7.0])
    shear = np.asarray([3.0, 5.0])
    viscosity = np.asarray([5.0, 9.0])
    tensile_strength = np.asarray([1.0e12, 2.0e12])
    damage = np.ones(n_ip)

    outputs = {
        "stress": np.zeros((n_ip, 6)),
        "eta": np.zeros((n_ip, 6)),
        "eta_old": np.zeros((n_ip, 6)),
        "strain_v": np.zeros((n_ip, 6)),
        "ep_old": np.zeros((n_ip, 6)),
        "tangent": np.zeros((n_ip, 36)),
        "elastic_energy": np.zeros(n_ip),
        "psi": np.zeros(n_ip),
        "lambda_v": np.zeros(n_ip),
        "lambda_c": np.zeros((n_ip, 2)),
        "lambda_c_old": np.zeros((n_ip, 2)),
    }

    variable_kernel = CohesiveViscoElasticKernelsVariableProps(
        1.0, 1.0e12, 1.0, 0.0, 1.0e12, 1.0e-3, "r2", False
    )
    variable_kernel(
        strains.ravel(),
        1.0e-3,
        damage,
        lame,
        shear,
        viscosity,
        tensile_strength,
        outputs["stress"].ravel(),
        outputs["eta"].ravel(),
        outputs["eta_old"].ravel(),
        outputs["strain_v"].ravel(),
        outputs["ep_old"].ravel(),
        outputs["tangent"].ravel(),
        outputs["elastic_energy"],
        outputs["psi"],
        outputs["lambda_v"],
        outputs["lambda_c"].ravel(),
        outputs["lambda_c_old"].ravel(),
    )

    for ip in range(n_ip):
        expected = {key: np.zeros_like(value[ip]) for key, value in outputs.items()}
        constant_kernel = CohesiveViscoElasticKernels(
            lame[ip] + 2.0 * shear[ip] / 3.0,
            lame[ip],
            shear[ip],
            viscosity[ip],
            1.0,
            tensile_strength[ip],
            1.0e12,
            1.0,
            0.0,
            1.0e12,
            1.0e-3,
            "r2",
            False,
        )
        strain_ip = np.asarray(
            [1.0e-3, 2.0e-4, 0.0, 1.0e-4, 0.0, 0.0]
            if ip == 0
            else [4.0e-4, 1.2e-3, 0.0, 0.0, 0.0, 2.0e-4]
        )
        constant_kernel(
            strain_ip,
            1.0e-3,
            np.ones(1),
            expected["stress"],
            expected["eta"],
            expected["eta_old"],
            expected["strain_v"],
            expected["ep_old"],
            expected["tangent"],
            np.atleast_1d(expected["elastic_energy"]),
            np.atleast_1d(expected["psi"]),
            np.atleast_1d(expected["lambda_v"]),
            expected["lambda_c"],
            expected["lambda_c_old"],
        )

        for key in (
            "stress",
            "eta",
            "strain_v",
            "tangent",
            "elastic_energy",
            "psi",
            "lambda_v",
            "lambda_c",
        ):
            np.testing.assert_allclose(outputs[key][ip], expected[key])


def test_cve_variable_properties_require_one_value_per_ip():
    kernel = CohesiveViscoElasticKernelsVariableProps(
        1.0, 1.0e12, 1.0, 0.0, 1.0e12, 1.0e-3, "r2", False
    )

    with np.testing.assert_raises_regex(ValueError, "one value per integration point"):
        kernel(
            np.zeros(12),
            1.0,
            np.ones(2),
            np.ones(1),
            np.ones(2),
            np.ones(2),
            np.ones(2),
            np.zeros(12),
            np.zeros(12),
            np.zeros(12),
            np.zeros(12),
            np.zeros(12),
            np.zeros(72),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(4),
            np.zeros(4),
        )
