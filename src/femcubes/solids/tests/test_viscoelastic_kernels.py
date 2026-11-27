"""Direct tests for visco-elastic C++ return-mapping kernels."""

import numpy as np
import pytest

from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import (
    ViscoElasticKernels,
    ViscoElasticKernelsVariableProps,
)


def _supports_true_2d_kernel():
    try:
        ViscoElasticKernels(1.0, 1.0, 1.0, 1.0, True)
    except TypeError:
        return False
    return True


def test_standard_j2_kernel_reproduces_glen_tensor_flow_law():
    """The von-Mises parameterisation must retain the Glen tensor law."""
    glen_a = 1.0e-24
    n_exp = 3.0
    eta_vp = (3.0 ** ((n_exp + 1.0) / 2.0) / (2.0 * glen_a)) ** (1.0 / n_exp)
    dt = 10.0
    stress = np.zeros(6)
    strain_p = np.zeros(6)
    tangent = np.zeros(36)
    lambda_field = np.zeros(1)

    ViscoElasticKernels(eta_vp, n_exp, 1.0e9, 5.0e8)(
        np.array([1.0e-3, -4.0e-4, 0.0, 2.0e-4, 0.0, 0.0]),
        dt,
        False,
        np.ones(1),
        stress,
        strain_p,
        np.zeros(6),
        tangent,
        lambda_field,
        np.zeros(1),
    )

    deviatoric_stress = stress.copy()
    deviatoric_stress[:3] -= np.mean(stress[:3])
    tau_e = np.sqrt(0.5 * np.dot(deviatoric_stress, deviatoric_stress))
    expected_rate = glen_a * tau_e ** (n_exp - 1.0) * deviatoric_stress

    np.testing.assert_allclose(strain_p / dt, expected_rate, rtol=1.0e-7, atol=1.0e-14)
    np.testing.assert_allclose(
        lambda_field[0] / dt,
        np.sqrt(2.0 / 3.0 * np.dot(expected_rate, expected_rate)),
        rtol=1.0e-7,
    )


def test_standard_j2_tangent_matches_finite_difference():
    n_exp = 3.0
    glen_a = 1.0e-24
    eta_vp = (3.0 ** ((n_exp + 1.0) / 2.0) / (2.0 * glen_a)) ** (1.0 / n_exp)
    kernel = ViscoElasticKernels(eta_vp, n_exp, 1.0e9, 5.0e8)
    base_strain = np.array([1.0e-3, -4.0e-4, 0.0, 2.0e-4, 0.0, 0.0])

    def evaluate(strain):
        stress = np.zeros(6)
        tangent = np.zeros(36)
        kernel(
            strain, 10.0, False, np.ones(1), stress, np.zeros(6),
            np.zeros(6), tangent, np.zeros(1), np.zeros(1),
        )
        return stress, tangent.reshape(6, 6)

    _, tangent = evaluate(base_strain)
    h = 1.0e-7
    for column in range(6):
        direction = np.zeros(6)
        direction[column] = h
        stress_plus, _ = evaluate(base_strain + direction)
        stress_minus, _ = evaluate(base_strain - direction)
        np.testing.assert_allclose(
            tangent[:, column], (stress_plus - stress_minus) / (2.0 * h),
            rtol=2.0e-5, atol=2.0e2,
        )


def test_damaged_elastic_return_uses_the_damage_scaled_tangent():
    lame = 4.0
    shear = 3.0
    damage = 0.5
    stress = np.zeros(6)
    tangent = np.zeros(36)
    ViscoElasticKernels(1.0e50, 1.0, lame, shear)(
        np.full(6, 1.0e-18), 1.0, True, np.array([damage]), stress,
        np.zeros(6), np.zeros(6), tangent, np.zeros(1), np.zeros(1),
    )

    elastic_tangent = np.zeros((6, 6))
    elastic_tangent[:3, :3] = lame
    elastic_tangent[np.diag_indices(6)] += 2.0 * shear
    np.testing.assert_allclose(tangent.reshape(6, 6), damage * elastic_tangent)


def test_true_2d_kernel_uses_intrinsic_stiffness_and_zero_fills_inactive_components():
    if not _supports_true_2d_kernel():
        pytest.skip("viscoelastic extension must be rebuilt for true_2d support")
    lame = 4.0
    shear = 3.0
    stress = np.zeros(6)
    strain_p = np.zeros(6)
    tangent = np.zeros(36)
    ViscoElasticKernels(1.0e50, 1.0, lame, shear, True)(
        np.array([1.0e-3, 2.0e-3, 9.0e-3, 4.0e-4, 5.0e-4, 6.0e-4]),
        1.0,
        False,
        np.ones(1),
        stress,
        strain_p,
        np.zeros(6),
        tangent,
        np.zeros(1),
        np.zeros(1),
    )

    np.testing.assert_allclose(stress, [0.018, 0.024, 0.0, 0.0, 0.0, 0.0036])
    np.testing.assert_allclose(strain_p[2:5], 0.0)
    tangent_matrix = tangent.reshape(6, 6)
    np.testing.assert_allclose(tangent_matrix[2:5, :], 0.0)
    np.testing.assert_allclose(tangent_matrix[:, 2:5], 0.0)


def test_variable_property_kernel_matches_constant_property_kernel():
    n_ip = 2
    dt = 100.0
    viscosity = 1.0e8
    n_exp = 3.0
    lame = 1.0e9
    shear = 5.0e8

    strains = np.zeros(n_ip * 6)
    strains[1] = 1.0e-3
    strains[6 + 1] = 2.0e-3
    damage = np.ones(n_ip)
    ep_old = np.zeros(n_ip * 6)
    lambda_old = np.zeros(n_ip)

    stress_constant = np.zeros(n_ip * 6)
    strain_p_constant = np.zeros(n_ip * 6)
    tangent_constant = np.zeros(n_ip * 36)
    lambda_constant = np.zeros(n_ip)

    constant_kernel = ViscoElasticKernels(viscosity, n_exp, lame, shear)
    constant_kernel(
        strains,
        dt,
        True,
        damage,
        stress_constant,
        strain_p_constant,
        ep_old,
        tangent_constant,
        lambda_constant,
        lambda_old,
    )

    stress_variable = np.zeros(n_ip * 6)
    strain_p_variable = np.zeros(n_ip * 6)
    tangent_variable = np.zeros(n_ip * 36)
    lambda_variable = np.zeros(n_ip)

    variable_kernel = ViscoElasticKernelsVariableProps(n_exp, False)
    variable_kernel(
        strains,
        dt,
        damage,
        np.full(n_ip, lame),
        np.full(n_ip, shear),
        np.full(n_ip, viscosity),
        stress_variable,
        strain_p_variable,
        ep_old,
        tangent_variable,
        lambda_variable,
        lambda_old,
    )

    np.testing.assert_allclose(stress_variable, stress_constant)
    np.testing.assert_allclose(strain_p_variable, strain_p_constant)
    np.testing.assert_allclose(tangent_variable, tangent_constant)
    np.testing.assert_allclose(lambda_variable, lambda_constant)
