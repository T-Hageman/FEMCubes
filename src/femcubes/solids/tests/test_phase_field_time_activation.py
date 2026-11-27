"""Regression tests for crack driving activation through cached physics forms."""

import numpy as np
import pytest

from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.rectangle import Sample_Parameters
from femcubes.core.physics.physics import Physics
from femcubes.solids.phase_field import PhaseField


@pytest.mark.parametrize(
    ("pf_formulation", "irreversible_method"),
    [(form, method) for form in ("AT1", "AT2", "HO-AT1", "HO-AT2") for method in ("AL", "Hist", "Penalty")
     if not (method == "Hist" and form in ("AT1", "HO-AT1"))],  # Hist is rejected for AT1 and HO-AT1
)
@pytest.mark.parametrize("negative_time_damage", [None, False, True], ids=["default", "disabled", "enabled"])
def test_crack_driving_activation_updates_cached_residual_and_tangent(
    pf_formulation, irreversible_method, negative_time_damage
):
    params = Sample_Parameters()
    params.Lx = params.Ly = 1.0
    params.dx = 0.5
    params.Quadrature_Degree = 2
    params.Global_Measures = {}
    params.Materials = {
        "Ice": {"Gc": 1.0, "ell": 0.25, "kappa": 1.0e-3, "PF_formulation": pf_formulation}
    }
    params.Field_Orders = {"phasefield": 2}
    params.Solution_Steps = {"phasefield": 1}
    params.ModelNames = ["pf"]
    params.Models = {
        "pf": {
            "Type": PhaseField.MODEL_TYPE,
            "Material": "Ice",
            "Irreversible_Method": irreversible_method,
            "CDF_field": "DrivingEnergy",
        }
    }
    if negative_time_damage is not None:
        params.Models["pf"]["Negative_Time_Damage"] = negative_time_damage
    params.t = params.dt = 0.25

    mesh = Mesh(params)
    mesh.Add_Quadrature_Field("DrivingEnergy", "Crack driving energy", ())
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    mesh.Constant_Field("phasefield", 0.25)
    mesh.Constant_Field("phasefieldOld", 0.4)
    if irreversible_method == "AL":
        mesh.Constant_Field("LMultiplier_PF", -0.2)

    step = params.Solution_Steps["phasefield"]
    mesh.Zero_Field("DrivingEnergy")
    K, F = physics.Get_Kmat_Fvec(step)
    K_without_driving = K.copy()
    F_without_driving = F.array.copy()
    K_with_driving = None
    try:
        mesh.Constant_Field("DrivingEnergy", 5.0)
        if irreversible_method == "Hist":
            mesh.Constant_Field("phasefield_HistOld", 3.0)
        K, F = physics.Get_Kmat_Fvec(step)
        K_with_driving = K.copy()
        F_with_driving = F.array.copy()
        assert not np.allclose(F_with_driving, F_without_driving)
        K_difference = K.copy()
        try:
            K_difference.axpy(-1.0, K_without_driving)
            assert K_difference.norm() > 1.0e-6
        finally:
            K_difference.destroy()

        # Reuse one Physics instance through activation, dt changes, and rollback.
        for time, dt, positive_end_time in [
            (-1.0, 0.25, False),
            (-0.25, 0.25, False),
            (0.0, 5.0e-10, False),
            (0.0, 1.0e-9, False),
            (0.0, 2.0e-9, True),
            (-0.5, 0.25, False),
            (-0.5, 0.75, True),
            (0.25, 0.25, True),
            (-1.0, 0.25, False),
        ]:
            params.t, params.dt = time, dt
            driving_active = negative_time_damage is True or positive_end_time
            K, F = physics.Get_Kmat_Fvec(step)
            K_expected = K_with_driving if driving_active else K_without_driving
            F_expected = F_with_driving if driving_active else F_without_driving
            K_difference = K.copy()
            try:
                K_difference.axpy(-1.0, K_expected)
                assert K_difference.norm() == pytest.approx(0.0, abs=1.0e-12)
            finally:
                K_difference.destroy()
            np.testing.assert_allclose(F.array, F_expected, atol=1.0e-12)
            np.testing.assert_allclose(physics.Get_Fvec(step).array, F_expected, atol=1.0e-12)
            if irreversible_method == "Hist":
                expected_history = 5.0 if driving_active else 3.0
                np.testing.assert_allclose(mesh.Get_Field("phasefield_Hist").x.array, expected_history)
                np.testing.assert_allclose(mesh.Get_Field("phasefield_HistOld").x.array, 3.0)
    finally:
        K_without_driving.destroy()
        if K_with_driving is not None:
            K_with_driving.destroy()
