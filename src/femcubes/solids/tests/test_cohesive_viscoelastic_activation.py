"""Damaged split-r2 onset must not depend on the previous trial's local guess."""

import numpy as np
import pytest

from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    CohesiveViscoElasticKernels,
)


def _captured_point(damage, multipliers):
    """PRIVATE: Evaluate the integration point captured at the first plate stall."""
    youngs, poisson = 9.5e9, 0.35
    shear = youngs / (2 * (1 + poisson))
    lame = youngs * poisson / (1 - poisson**2)
    kernel = CohesiveViscoElasticKernels(
        lame + shear, lame, shear, 0.0, 1.0,
        0.4e6, 0.6e6, 1.0, 0.0, 0.6e6, 1e-3, "r2_2L", False, True,
    )
    strain = np.array([-4.924764016066874e-6, 3.6693092844970836e-5,
                       0.0, 0.0, 0.0, -5.55814569949434e-6])
    stress, tangent = np.zeros(6), np.zeros(36)
    kernel(strain, 120.0, np.array([damage]), stress,
           np.zeros(6), np.zeros(6), np.zeros(6), np.zeros(6), tangent,
           np.zeros(1), np.zeros(1), np.zeros(1), np.array(multipliers), np.zeros(2))
    return stress


@pytest.mark.parametrize("damage", [1.0, 0.9787133976526249])
def test_split_r2_stress_respects_the_current_damaged_strength_surface(damage):
    stress = _captured_point(damage, [0.0, 0.0])
    trace = stress[0] + stress[1]
    deviator = stress - trace / 2 * np.array([1, 1, 0, 0, 0, 0])
    volumetric_strength = 0.4e6 / np.sqrt(1 - (0.4e6 / 0.6e6)**2)
    strength_ratio = np.hypot(trace / volumetric_strength,
                              np.sqrt(2) * np.linalg.norm(deviator) / 0.6e6)
    assert strength_ratio <= damage * (1 + 1e-7)


def test_split_r2_damaged_onset_is_independent_of_previous_trial_multipliers():
    damage = 0.9787133976526249
    cold = _captured_point(damage, [0.0, 0.0])
    warm = _captured_point(damage, [5.02593280758852e-7, 2.616966979597312e-7])
    np.testing.assert_allclose(cold, warm, rtol=1e-7, atol=0.05)
