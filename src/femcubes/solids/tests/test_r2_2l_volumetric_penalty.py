"""Regression tests for the split-r2 unilateral volumetric treatment."""

import numpy as np
import pytest

from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    CohesiveViscoElasticKernels,
    compute_failure_surface_point,
    compute_strength_surface_point,
    get_cohesive_num_lambdas,
)


YOUNGS = 9.5e9
POISSON = 0.35
SHEAR = YOUNGS / (2.0 * (1.0 + POISSON))
BULK = YOUNGS / (3.0 * (1.0 - 2.0 * POISSON))
LAME = 2.0 * SHEAR * POISSON / (1.0 - 2.0 * POISSON)
VISCOSITY = (1.0 / (2.0e-24)) ** (1.0 / 3.0)
F_T = 0.4e6
F_S = 1.0e6
CALIBRATED_F_T = 0.4e6
CALIBRATED_F_S = 0.6e6
DP_PHI = np.degrees(np.arctan(3.0 / (1.0 + POISSON)))


def _run_r2_2l_return_map(
    volumetric_strain, shear_strain=2.0e-3, failure_type="r2_2L"
):
    """Return cohesive multipliers for a sheared state with hydrostatic bias."""
    kernel = CohesiveViscoElasticKernels(
        BULK, LAME, SHEAR, VISCOSITY, 3.0,
        F_T, F_S, 1.0, DP_PHI, 9.0e6,
        1.0e-4, failure_type, False,
    )
    strain = np.asarray([
        volumetric_strain, volumetric_strain, volumetric_strain,
        0.0, 0.0, shear_strain,
    ])
    lambda_c = np.zeros(2)
    kernel(
        strain, 1.0, np.ones(1),
        np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6),
        np.zeros(36), np.zeros(1), np.zeros(1), np.zeros(1), lambda_c, np.zeros(2),
    )
    return lambda_c


def _run_elastic_return_map(strain, damage, f_t, f_s, dp_q, failure_type, kappa):
    """Return (stress, eta, tangent, multipliers) of the kernel without creep (the cohesive
    linear-elastic solid)."""
    kernel = CohesiveViscoElasticKernels(
        BULK, LAME, SHEAR, 0.0, 1.0,
        f_t, f_s, 1.0, DP_PHI, dp_q,
        kappa, failure_type, False,
    )
    stress = np.zeros(6)
    eta = np.zeros(6)
    tangent = np.zeros(36)
    lambda_c = np.zeros(2)
    kernel(
        np.asarray(strain, dtype=float), 1.0, np.asarray([damage]), stress, eta, np.zeros(6),
        np.zeros(6), np.zeros(6), tangent, np.zeros(1), np.zeros(1), np.zeros(1), lambda_c, np.zeros(2),
    )
    return stress, eta, tangent, lambda_c


def _run_true_2d_uniaxial_tension(target_stress):
    """Return the true-2D R2_2L stress and multipliers for one load state."""
    kernel = CohesiveViscoElasticKernels(
        BULK, LAME, SHEAR, VISCOSITY, 3.0,
        CALIBRATED_F_T, CALIBRATED_F_S, 1.0, DP_PHI, 9.0e6,
        1.0e-3, "r2_2L", False, True,
    )
    stiffness = np.asarray([
        [LAME + 2.0 * SHEAR, LAME],
        [LAME, LAME + 2.0 * SHEAR],
    ])
    strain = np.zeros(6)
    strain[:2] = np.linalg.solve(stiffness, [target_stress, 0.0])
    stress = np.zeros(6)
    eta = np.zeros(6)
    eta_old = np.zeros(6)
    strain_v = np.zeros(6)
    strain_v_old = np.zeros(6)
    tangent = np.zeros(36)
    elastic_energy = np.zeros(1)
    psi = np.zeros(1)
    lambda_v = np.zeros(1)
    lambda_c = np.zeros(2)
    lambda_c_old = np.zeros(2)
    kernel(
        strain, 1.0, np.ones(1), stress, eta, eta_old, strain_v,
        strain_v_old, tangent, elastic_energy, psi, lambda_v, lambda_c,
        lambda_c_old,
    )
    return stress, lambda_c


def test_r2_2l_compressive_volumetric_mode_is_inactive_and_shear_limit_is_continuous():
    """A vanishing hydrostatic bias must not change the active shear response."""
    volumetric_bias = 1.0e-10
    lambda_c_tension = _run_r2_2l_return_map(volumetric_bias)
    lambda_c_compression = _run_r2_2l_return_map(-volumetric_bias)

    assert lambda_c_tension[1] > 0.0
    assert lambda_c_compression[1] > 0.0
    np.testing.assert_allclose(lambda_c_compression[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        lambda_c_tension[1], lambda_c_compression[1], rtol=1.0e-6, atol=1.0e-14,
    )


@pytest.mark.parametrize("failure_type", ["r2_2L", "r2_2L_DP"])
def test_cohesive_elastic_r2_2l_keeps_compressive_penalty_out_of_deviatoric_mode(
    failure_type,
):
    """The shared penalty contraction also applies to the return map without creep."""
    stress, eta, tangent, lambda_c = _run_elastic_return_map(
        [0.00273354, -0.00273565, 0.0, 0.0, 0.0, -0.00108401],
        1.65089e-6, 0.2e6, 5.0e6, 5.0e6, failure_type, 1.0e-6,
    )

    assert np.isfinite(stress).all()
    assert np.isfinite(tangent).all()
    np.testing.assert_allclose(lambda_c[0], 0.0, atol=1.1e-12)
    assert lambda_c[1] > 0.0


def test_r2_2l_dp_uses_two_modes_and_hardens_only_in_compression():
    """Pressure hardening increases resistance without coupling the modes."""
    assert get_cohesive_num_lambdas("r2_2L_DP") == 2

    lambda_base = _run_r2_2l_return_map(-1.0e-5)
    lambda_dp = _run_r2_2l_return_map(
        -1.0e-5, failure_type="r2_2L_DP",
    )

    np.testing.assert_allclose(lambda_dp[0], 0.0, atol=1.0e-14)
    assert 0.0 < lambda_dp[1] < lambda_base[1]


def test_cohesive_elastic_r2_2l_dp_converges_with_elastic_volume_hardening():
    """The return map without creep uses the shared elastic-volume DP potential."""
    stress, eta, tangent, lambda_c = _run_elastic_return_map(
        [-1.0e-5, -1.0e-5, -1.0e-5, 0.0, 0.0, 2.0e-3],
        1.0, F_T, F_S, 9.0e6, "r2_2L_DP", 1.0e-4,
    )

    assert np.isfinite(stress).all()
    assert np.isfinite(eta).all()
    assert np.isfinite(tangent).all()
    np.testing.assert_allclose(lambda_c[0], 0.0, atol=1.1e-12)
    assert lambda_c[1] > 0.0


def test_r2_2l_dp_tensile_potential_matches_r2_2l():
    """The DP extension leaves the calibrated tensile branch unchanged."""
    eta = np.asarray([2.0e-4, -1.0e-4, 5.0e-5, 0.0, 0.0, 3.0e-4])
    strain = np.asarray([3.0e-4, 1.0e-4, 5.0e-5, 0.0, 0.0, 2.0e-4])

    base = compute_failure_surface_point(
        F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L",
        eta, 0.3, strain,
    )
    dp = compute_failure_surface_point(
        F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
        eta, 0.3, strain,
    )

    for base_value, dp_value in zip(base, dp):
        np.testing.assert_allclose(dp_value, base_value)


def test_r2_2l_dp_compressive_potential_uses_calibrated_confined_strength():
    """DP confinement uses elastic volume and exposes consistent derivatives."""
    damage = 0.3
    eta_deviatoric = 2.0e-3 * np.asarray(
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    ) / np.sqrt(2.0)
    eta = eta_deviatoric - 2.0e-6 * np.asarray(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    strain = np.asarray([-1.0e-5, -1.0e-5, -1.0e-5, 0.0, 0.0, 0.0])
    result = compute_failure_surface_point(
        F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
        eta, damage, strain,
    )

    confined_strength = min(
        F_S - BULK * np.tan(np.radians(DP_PHI))
        * np.sum(strain[:3] - eta[:3]),
        9.0e6,
    )
    deviatoric_strength = 2.0 * confined_strength / np.sqrt(3.0)
    regularized_norm = np.sqrt(
        0.5 * deviatoric_strength**2
        * np.dot(eta_deviatoric, eta_deviatoric) + 1.0e-12,
    ) - 1.0e-6
    volumetric_strength = F_T / np.sqrt(1.0 - (F_T / F_S) ** 2)
    penalty = -1.0e6 * volumetric_strength / 3.0 * np.sum(eta[:3])
    np.testing.assert_allclose(result[0], damage * regularized_norm + penalty)
    np.testing.assert_allclose(result[4], regularized_norm)

    h = 1.0e-9
    mixed_derivative_fd = np.empty((6, 6))
    for column in range(6):
        perturbation = np.zeros(6)
        perturbation[column] = h
        plus = compute_failure_surface_point(
            F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
            eta, damage, strain + perturbation,
        )[1]
        minus = compute_failure_surface_point(
            F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
            eta, damage, strain - perturbation,
        )[1]
        mixed_derivative_fd[:, column] = (
            np.ravel(plus) - np.ravel(minus)
        ) / (2.0 * h)

    np.testing.assert_allclose(
        result[3], mixed_derivative_fd, rtol=1.0e-6, atol=1.0e-3,
    )

    hessian_step = 1.0e-7
    hessian_fd = np.empty((6, 6))
    for column in range(6):
        perturbation = np.zeros(6)
        perturbation[column] = hessian_step
        plus = compute_failure_surface_point(
            F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
            eta + perturbation, damage, strain,
        )[1]
        minus = compute_failure_surface_point(
            F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
            eta - perturbation, damage, strain,
        )[1]
        hessian_fd[:, column] = (
            np.ravel(plus) - np.ravel(minus)
        ) / (2.0 * hessian_step)

    np.testing.assert_allclose(
        result[2], hessian_fd, rtol=1.0e-6, atol=1.0e3,
    )

    unrelaxed = compute_failure_surface_point(
        F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
        eta_deviatoric, damage, strain,
    )
    assert result[4] < unrelaxed[4]

    fully_relaxed_eta = eta_deviatoric - 2.0e-5 * np.asarray(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    fully_relaxed = compute_failure_surface_point(
        F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
        fully_relaxed_eta, damage, strain,
    )
    unconfined_deviatoric_strength = 2.0 * F_S / np.sqrt(3.0)
    unconfined_norm = np.sqrt(
        0.5 * unconfined_deviatoric_strength**2
        * np.dot(eta_deviatoric, eta_deviatoric) + 1.0e-12,
    ) - 1.0e-6
    np.testing.assert_allclose(fully_relaxed[4], unconfined_norm)

    capped_strain = np.asarray([-1.0e-2, -1.0e-2, -1.0e-2, 0.0, 0.0, 0.0])
    capped = compute_failure_surface_point(
        F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
        eta, damage, capped_strain,
    )
    more_confined = compute_failure_surface_point(
        F_T, F_S, BULK, DP_PHI, 9.0e6, "r2_2L_DP",
        eta, damage, 2.0 * capped_strain,
    )
    np.testing.assert_allclose(more_confined[0], capped[0])
    np.testing.assert_allclose(more_confined[1], capped[1])
    np.testing.assert_allclose(capped[3], 0.0, atol=0.0)


def test_r2_2l_true_2d_tension_activates_both_modes_at_uniaxial_limit():
    """A partially opened tensile mode must not overshoot the coupled surface."""
    stress, lambda_c = _run_true_2d_uniaxial_tension(4.47e5)

    assert lambda_c[0] > 0.0
    assert lambda_c[1] > 0.0
    assert _strength_indicator("r2_2L", stress, true_2d=True) <= 1.0e-5
    assert stress[0] < 4.47e5


@pytest.mark.parametrize("shear_strain", [5.0e-4, 1.0e-3, 2.0e-3, 5.0e-3])
@pytest.mark.parametrize("volumetric_bias", [1.0e-6, 1.0e-5, 1.0e-4])
def test_r2_2l_compressive_hydrostatic_bias_does_not_change_shear_return_map(
    shear_strain, volumetric_bias
):
    """The penalty keeps the volumetric mode inactive over practical load ranges."""
    lambda_c_zero = _run_r2_2l_return_map(0.0, shear_strain)
    lambda_c_compression = _run_r2_2l_return_map(-volumetric_bias, shear_strain)

    np.testing.assert_allclose(lambda_c_compression[0], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        lambda_c_compression[1], lambda_c_zero[1], rtol=1.0e-10, atol=1.0e-14,
    )


def _strength_indicator(failure_type, stress, true_2d=False):
    """Evaluate the plotted strength indicator for one elastic stress state."""
    strain = np.zeros(6)
    if true_2d:
        stiffness = np.array(
            [[LAME + 2.0 * SHEAR, LAME], [LAME, LAME + 2.0 * SHEAR]],
        )
        strain[:2] = np.linalg.solve(stiffness, stress[:2])
        failure_bulk = LAME + SHEAR
    else:
        stiffness = np.full((3, 3), LAME)
        stiffness[np.diag_indices(3)] += 2.0 * SHEAR
        strain[:3] = np.linalg.solve(stiffness, stress[:3])
        failure_bulk = BULK
    return compute_strength_surface_point(
        CALIBRATED_F_T, CALIBRATED_F_S, failure_bulk, DP_PHI, 9.0e6,
        failure_type, stress, strain, true_2d=true_2d,
        num_direction_samples=10001,
    )


@pytest.mark.parametrize("failure_type", ["r2", "r2_2L"])
@pytest.mark.parametrize("true_2d", [False, True])
def test_r2_criteria_match_uniaxial_compressive_strength(failure_type, true_2d):
    """The r2 deviatoric scale matches the requested true-2D/3-D compression."""
    stress = np.zeros(6)
    stress[0] = -CALIBRATED_F_S

    np.testing.assert_allclose(
        _strength_indicator(failure_type, stress, true_2d=true_2d),
        0.0,
        atol=1.0e-6,
    )


@pytest.mark.parametrize("true_2d", [False, True])
def test_r2_2l_matches_uniaxial_tensile_and_compressive_strengths(true_2d):
    """The split r2 surface matches both requested uniaxial strengths."""
    tensile_stress = np.zeros(6)
    tensile_stress[0] = CALIBRATED_F_T
    compressive_stress = np.zeros(6)
    compressive_stress[0] = -CALIBRATED_F_S

    np.testing.assert_allclose(
        _strength_indicator("r2_2L", tensile_stress, true_2d=true_2d),
        0.0,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        _strength_indicator("r2_2L", compressive_stress, true_2d=true_2d),
        0.0,
        atol=1.0e-6,
    )


def test_r2_2l_plane_strain_strengths_follow_the_true_3d_calibration():
    """Plane strain is a prediction, not the reference state for the 3-D scale."""
    tensile_strength = 1.0 / np.sqrt(
        (1.0 + POISSON) ** 2 / CALIBRATED_F_T ** 2
        - 3.0 * POISSON / CALIBRATED_F_S ** 2,
    )
    compressive_strength = CALIBRATED_F_S / np.sqrt(
        1.0 - POISSON + POISSON ** 2,
    )

    tensile_stress = np.zeros(6)
    tensile_stress[:3] = [tensile_strength, 0.0, POISSON * tensile_strength]
    compressive_stress = np.zeros(6)
    compressive_stress[:3] = [
        -compressive_strength, 0.0, -POISSON * compressive_strength,
    ]

    np.testing.assert_allclose(
        _strength_indicator("r2_2L", tensile_stress), 0.0, atol=1.0e-5,
    )
    np.testing.assert_allclose(
        _strength_indicator("r2_2L", compressive_stress), 0.0, atol=1.0e-6,
    )


@pytest.mark.parametrize("failure_type", ["r2_2L", "r2_2L_DP"])
def test_r2_2l_rejects_strengths_that_cannot_be_calibrated(failure_type):
    """A finite split-r2 tensile calibration requires f_s to exceed f_t."""
    stress = np.zeros(6)
    strain = np.zeros(6)

    with pytest.raises(ValueError, match="f_s is greater than f_t"):
        compute_strength_surface_point(
            CALIBRATED_F_S, CALIBRATED_F_S, BULK, DP_PHI, 9.0e6, failure_type,
            stress, strain,
        )


@pytest.mark.parametrize(
    ("dp_phi", "dp_q", "message"),
    [
        (-1.0, F_S, "DP_phi must be non-negative"),
        (DP_PHI, 0.5 * F_S, "DP_q must be >= f_s"),
        (0.0, 2.0 * F_S, "DP_phi must be positive"),
    ],
)
def test_r2_2l_dp_rejects_invalid_hardening_parameters(
    dp_phi, dp_q, message,
):
    """The kernel enforces the DP hardening parameter contract."""
    with pytest.raises(ValueError, match=message):
        CohesiveViscoElasticKernels(
            BULK, LAME, SHEAR, VISCOSITY, 3.0,
            F_T, F_S, 1.0, dp_phi, dp_q,
            1.0e-4, "r2_2L_DP", False,
        )


@pytest.mark.parametrize("true_2d", [False, True])
def test_r2_2l_strength_surface_is_finite_near_hydrostatic_loading(true_2d):
    """Pure and nearly hydrostatic stresses must not create a negative dev norm."""
    if true_2d:
        stiffness = np.array(
            [[LAME + 2.0 * SHEAR, LAME], [LAME, LAME + 2.0 * SHEAR]],
        )
        failure_bulk = LAME + SHEAR
        stress_states = [
            np.asarray([pressure, pressure, 0.0, 0.0, 0.0, 0.0])
            for pressure in (-1.0e6, -1.0e-3, 0.0, 1.0e-3, 1.0e6)
        ]
        stress_states += [
            np.asarray([1.0e6, 1.0e6 + perturbation, 0.0, 0.0, 0.0, 0.0])
            for perturbation in (-1.0, 1.0)
        ]
    else:
        stiffness = np.full((3, 3), LAME)
        stiffness[np.diag_indices(3)] += 2.0 * SHEAR
        failure_bulk = BULK
        stress_states = [
            np.asarray([pressure, pressure, pressure, 0.0, 0.0, 0.0])
            for pressure in (-1.0e6, -1.0e-3, 0.0, 1.0e-3, 1.0e6)
        ]
        stress_states += [
            np.asarray([1.0e6, 1.0e6 + perturbation, 1.0e6, 0.0, 0.0, 0.0])
            for perturbation in (-1.0, 1.0)
        ]

    for stress in stress_states:
        strain = np.zeros(6)
        dimension = 2 if true_2d else 3
        strain[:dimension] = np.linalg.solve(stiffness, stress[:dimension])
        indicator = compute_strength_surface_point(
            F_T, F_S, failure_bulk, DP_PHI, 9.0e6, "r2_2L", stress, strain,
            true_2d=true_2d,
        )
        assert np.isfinite(indicator)


@pytest.mark.parametrize(
    "failure_type",
    ["r1", "r1_DP", "r2", "r2_DP", "r2_2L", "r2_2L_DP", "r2_2L_PSD"],
)
def test_generic_strength_surface_evaluator_handles_all_supported_criteria(failure_type):
    """Every supported criterion uses the same polar strength-surface evaluator."""
    stress = np.asarray([5.0e5, 1.0e5, -1.0e5, 0.0, 0.0, 2.0e5])
    stiffness = np.full((3, 3), LAME)
    stiffness[np.diag_indices(3)] += 2.0 * SHEAR
    strain = np.zeros(6)
    strain[:3] = np.linalg.solve(stiffness, stress[:3])

    value = compute_strength_surface_point(
        F_T, F_S, BULK, DP_PHI, 9.0e6, failure_type, stress, strain,
        num_direction_samples=9,
    )
    assert np.isfinite(value)


@pytest.mark.parametrize("failure_type", ["r1TO", "r1_1L", "r1_DP_1L", "ShearOnly", "r2_2L_DPi", "r2_2L_DPa", "unknown"])
def test_kernel_rejects_unsupported_failure_types(failure_type):
    """Only the seven supported failure types are accepted; the message lists them."""
    with pytest.raises(ValueError, match="r1, r1_DP, r2, r2_DP, r2_2L, r2_2L_DP, r2_2L_PSD"):
        CohesiveViscoElasticKernels(
            BULK, LAME, SHEAR, VISCOSITY, 3.0,
            F_T, F_S, 1.0, DP_PHI, 9.0e6,
            1.0e-4, failure_type, False,
        )


def test_opening_only_strength_indicator_is_the_exact_onset_gauge():
    """r2_2L_PSD: zero at the uniaxial tensile strength, no yield in uniaxial compression."""
    stiffness = np.full((3, 3), LAME)
    stiffness[np.diag_indices(3)] += 2.0 * SHEAR
    for stress_xx, expected in ((CALIBRATED_F_T, 0.0), (-10.0 * CALIBRATED_F_S, -1.0)):
        stress = np.zeros(6)
        stress[0] = stress_xx
        strain = np.zeros(6)
        strain[:3] = np.linalg.solve(stiffness, stress[:3])
        value = compute_strength_surface_point(
            CALIBRATED_F_T, CALIBRATED_F_S, BULK, 0.0, CALIBRATED_F_S, "r2_2L_PSD", stress, strain,
        )
        np.testing.assert_allclose(value, expected, atol=1.0e-10)
