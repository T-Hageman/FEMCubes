"""Hydrostatic cohesive states need a finite, normalized fallback direction."""

import numpy as np
import pytest

from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import compute_eta_point


@pytest.mark.parametrize("true_2d", [False, True])
@pytest.mark.parametrize("volumetric_strain", [-1.0e-3, 0.0, 1.0e-3])
def test_hydrostatic_deviatoric_mode_uses_the_normalized_fallback(true_2d, volumetric_strain):
    dimension = 2 if true_2d else 3
    strain = np.zeros(6)
    strain[:dimension] = volumetric_strain
    eta, deta_de, deta_dl, d2eta_dlde = compute_eta_point(
        "r1", strain, np.asarray([0.0, 1.0]), true_2d=true_2d,
    )

    for result in (eta, deta_de, deta_dl, d2eta_dlde):
        assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(eta), 1.0, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.sum(eta[:dimension]), 0.0, atol=1e-14)
    if true_2d:
        np.testing.assert_allclose(eta[[2, 3, 4]], 0.0, atol=1e-14)
    np.testing.assert_allclose(deta_dl[1].ravel(), eta.ravel(), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(deta_de, 0.0, atol=1e-14)
    np.testing.assert_allclose(d2eta_dlde, 0.0, atol=1e-14)
