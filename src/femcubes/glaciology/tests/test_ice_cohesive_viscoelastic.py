"""Tests for the ice-specific cohesive visco-elastic model.

Contract: the model is Solid:Cohesive_ViscoElastic with per-point ice properties (equal to the
parent at uniform porosity and temperature for every failure type and 2-D mode); it runs
through TimeSolver with the parent's measures and etaNorm; it assembles after a refinement;
invalid porosity, temperature or tensile strength raise the same error on every rank.
"""

import os
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest
import ufl

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.physics.physics import Physics
from femcubes.core.physics.tests.test_model_remesh import (
    _build_ice_fracture_params,
    _contains_ufl_terminal,
    _install_one_refinement,
)
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.glaciology.ice_cohesive_viscoelastic import IceCohesiveViscoElastic
from femcubes.glaciology.ice_material import IceMaterial
from femcubes.solids.cohesive_viscoelastic import Cohesive_ViscoElastic

import femcubes.boundary  # noqa: F401 - model registration
import femcubes.glaciology  # noqa: F401
import femcubes.solids  # noqa: F401

POROSITY, TEMPERATURE = 0.1, -20.0
F_S = 1.0e6


def _build_params(failure_type="r2_2L", mode="plane strain", parent=False):
    """PRIVATE: one 2-D solid model on a small rectangle (a phasefield field, no phase-field model)."""
    material = {"ell": 2.0, "f_s": F_S, "FailureType": failure_type, "kappa": 1.0e-6, "Inertia": False}
    if failure_type in ("r1_DP", "r2_DP", "r2_2L_DP"):
        material.update(DP_phi=30.0, DP_q=5.0e6)
    if parent:
        ice = IceMaterial()
        material.update(
            Lame=ice.Get_Lame_Parameter(POROSITY, TEMPERATURE), Shear=ice.Get_Shear_Modulus(POROSITY, TEMPERATURE),
            Bulk=ice.Get_Bulk_Modulus(POROSITY, TEMPERATURE), Density=ice.Get_Density(POROSITY, TEMPERATURE),
            Visc_Type="Glen", Glen_A=float(ice.Get_Creep_A(POROSITY, TEMPERATURE)), Glen_n=3.0,
            f_t=ice.Get_Tensile_Strength(POROSITY, TEMPERATURE), Gc=1.0,
        )

    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        MeshType = "Rectangle"
        dx = 1.0
        dim = 2
        Lx = 2.0
        Ly = 2.0
        Quadrature_Degree = 2
        Materials = {"Ice": material}
        gravity = 0.0
        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 1, "Porosity": 0, "Temperature": 1}
        Models = {
            "Solid": {
                "Type": "Solid:Cohesive_ViscoElastic" if parent else "Glaciology:Cohesive_ViscoElastic",
                "Material": "Ice",
                "plane_stress": mode == "plane stress",
                "true_2d": mode == "true_2d",
            },
        }
        ModelNames = ["Solid"]
        Newmark_Beta = 0.25
        Newmark_Gamma = 0.5
        Plot_Fields: list = []
        output_dir = None
        t = 1.0e7
        dt = 1.0e7

    return Params()


def _model(params, porosity=POROSITY, temperature=TEMPERATURE):
    """PRIVATE: mesh and initialized solid model of ``params``, with uniform ice fields if given."""
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phase field [-]", "CG", 1)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    if "poros" in mesh.fields:
        if porosity is not None:
            mesh.Constant_Field("poros", porosity)
        if temperature is not None:
            mesh.Constant_Field("T", temperature)
    return mesh, physics.models["Solid"]


def _assemble_state(params, strain_scale):
    """PRIVATE: the point state after one assembly at a uniform strain and phase field 0.3."""
    mesh, model = _model(params)
    mesh.Constant_Field("phasefield", 0.3)
    x = ufl.SpatialCoordinate(mesh.mesh)
    mesh.Set_From_Expression("u", ufl.as_vector([strain_scale * (x[0] + 3.0 * x[1]), -0.4 * strain_scale * x[1]]))
    model.assemble_KF(model.Step_u)
    names = ("sigma", "eta", "etaNorm", "Plastic_Strain", "dsigma_dStrainMandel", "PsiField", "ElasticEnergy",
             "LambdaField_coh", "StrengthRatio")
    return {name: mesh.Get_Field(name).x.array.copy() for name in names if name in mesh.fields}


# ----------------------------------------------------------------------
# Equality with Solid:Cohesive_ViscoElastic
# ----------------------------------------------------------------------

@pytest.mark.parametrize("failure_type", ["r2_2L", "r2_2L_PSD", "r1", "r1_DP", "r2", "r2_DP", "r2_2L_DP"])
@pytest.mark.parametrize("mode", ["plane strain", "true_2d", "plane stress"])
def test_equals_the_parent_at_uniform_porosity_and_temperature(failure_type, mode):
    if failure_type == "r2_2L_PSD" and mode == "true_2d":
        for parent in (False, True):
            params = _build_params(failure_type, mode, parent)
            with pytest.raises(NotImplementedError, match="true_2d"):
                Physics(Mesh(params), params)
        return
    # below and beyond the strength, with creep; beyond the strength with little creep
    for strain_scale, dt in ((1.0e-5, 1.0e7), (2.0e-4, 1.0e7), (2.0e-4, 1.0)):
        states = []
        for parent in (False, True):
            params = _build_params(failure_type, mode, parent)
            params.t = params.dt = dt
            params.Models["Solid"]["Strength_Ratio_Field"] = failure_type in ("r2_2L", "r2_2L_PSD")
            states.append(_assemble_state(params, strain_scale))
        ice, reference = states
        for name, expected in reference.items():
            scale = max(np.abs(expected).max(), 1e-300)
            np.testing.assert_allclose(ice[name], expected, rtol=1e-9, atol=1e-9 * scale, err_msg=name)
        assert np.abs(reference["Plastic_Strain"]).max() > 0.0
        assert ("StrengthRatio" in ice) == (failure_type in ("r2_2L", "r2_2L_PSD"))
    assert reference["etaNorm"].max() > 0.0


def test_uses_local_ice_properties():
    params = _build_params()
    mesh, model = _model(params, porosity=None, temperature=None)
    assert isinstance(model, IceCohesiveViscoElastic)
    assert isinstance(model, Cohesive_ViscoElastic)
    assert model.LumpedMass is False

    x = ufl.SpatialCoordinate(mesh.mesh)
    mesh.Set_From_Expression("poros", 0.05 + 0.05 * x[0])
    mesh.Set_From_Expression("T", -30.0 + 5.0 * x[1])
    model.assemble_KF(model.Step_u)

    material = IceMaterial()
    porosity = mesh.Evaluate_At_Field_Points("Lame", mesh.Get_Field("poros")).reshape(-1)
    temperature = mesh.Evaluate_At_Field_Points("Lame", mesh.Get_Field("T")).reshape(-1)
    assert np.ptp(porosity) > 0 and np.ptp(temperature) > 0
    expected = {
        "Lame": material.Get_Lame_Parameter(porosity, temperature),
        "Shear": material.Get_Shear_Modulus(porosity, temperature),
        "Creep_Stress_Scale": material.Get_ViscoPlastic_viscosity(porosity, temperature),
        "TensileStrength": material.Get_Tensile_Strength(porosity, temperature),
    }
    for field_name, values in expected.items():
        np.testing.assert_allclose(mesh.Get_Field(field_name).x.array, values, rtol=1e-12, err_msg=field_name)


def test_true_2d_uses_the_two_dimensional_creep_scale_and_bulk_modulus():
    mesh, model = _model(_build_params(mode="true_2d"))
    model.assemble_KF(model.Step_u)
    np.testing.assert_allclose(mesh.Get_Field("Creep_Stress_Scale").x.array,
                               IceMaterial().Get_ViscoPlastic_viscosity(POROSITY, TEMPERATURE, dim=2), rtol=1e-12)
    bulk = mesh.Evaluate_At_Field_Points("Lame", model.bulk).reshape(-1)
    np.testing.assert_allclose(bulk, mesh.Get_Field("Lame").x.array + mesh.Get_Field("Shear").x.array, rtol=1e-12)


@pytest.mark.parametrize("mode", ["plane strain", "plane stress", "true_2d"])
def test_fixed_stiffness_is_the_elastic_stiffness_of_the_mode(mode):
    params = _build_params(mode=mode)
    params.Models["Solid"]["consistent_stiffness"] = False
    mesh, model = _model(params)
    K, F = model.assemble_KF(model.Step_u)
    assert mesh.Compile_Form(K) is not None
    D = mesh.Evaluate_At_Field_Points("dsigma_dStrainMandel", model.D)[0].reshape(6, 6)
    material = IceMaterial()
    lame, shear = material.Get_Lame_Parameter(POROSITY, TEMPERATURE), material.Get_Shear_Modulus(POROSITY, TEMPERATURE)
    if mode == "plane stress":
        lame = 2.0 * lame * shear / (lame + 2.0 * shear)
    np.testing.assert_allclose(D[0, :2], [lame + 2.0 * shear, lame], rtol=1e-12)
    np.testing.assert_allclose(D[5, 5], 2.0 * shear, rtol=1e-12)
    assert (D[2, 2] == 0.0) == (mode != "plane strain")


def test_rejects_the_removed_energy_ratio_input():
    params = _build_params()
    params.Materials["Ice"]["GcII_over_Gc"] = 2.0
    with pytest.raises(ValueError, match="GcII_over_Gc is not supported"):
        Physics(Mesh(params), params)


def test_reads_the_mode_ii_ratio_as_the_parent():
    params = _build_params("r1_DP")
    params.Materials["Ice"].update(Gc=100.0, GcII=250.0)
    _, model = _model(params)
    assert model.GcII / model.Gc == 2.5
    params.Materials["Ice"].pop("Gc")
    with pytest.raises(ValueError, match="GcII needs the reference toughness Gc"):
        Physics(Mesh(params), params)


# ----------------------------------------------------------------------
# Checks of the ice state
# ----------------------------------------------------------------------

@pytest.mark.parametrize(("failure_type", "f_s"), [("r2_2L", 0.2e6), ("r2_2L_DP", 0.15e6), ("r2_2L_PSD", 0.24e6)])
def test_inadmissible_tensile_strength_raises_a_clear_error(failure_type, f_s):
    params = _build_params(failure_type)
    params.Materials["Ice"]["f_s"] = f_s
    params.Materials["Ice"].pop("DP_q", None)
    _, model = _model(params)
    with pytest.raises(ValueError, match=f"tensile strength f_t .* for FailureType '{failure_type}'"):
        model.assemble_KF(model.Step_u)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("poros", 1.0, "porosity 'poros' must be finite with 0 <= p < 1"),
    ("poros", -0.1, "porosity 'poros'"),
    ("T", 2.0, "temperature 'T' must be finite and at most 0 C"),
    ("T", np.nan, "temperature 'T'"),
])
def test_invalid_porosity_or_temperature_raises_a_clear_error(field, value, message):
    mesh, model = _model(_build_params())
    mesh.Constant_Field(field, value)
    with pytest.raises(ValueError, match=message):
        model.assemble_KF(model.Step_u)


def test_warns_once_when_the_temperature_was_never_set():
    _, model = _model(_build_params(), temperature=None)
    with pytest.warns(UserWarning, match="temperature 'T' is 0 C everywhere"):
        model.assemble_KF(model.Step_u)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.assemble_KF(model.Step_u)


def test_inadmissible_points_get_a_non_finite_stress_without_an_exception():
    params = _build_params()
    params.Materials["Ice"]["f_s"] = 0.2e6
    _, model = _model(params)
    update = model._update_cve_state
    n = 3
    update.set_properties(np.full(n, 3.0e9), np.full(n, 2.0e9), np.full(n, 1.0e8), [0.1e6, 0.3e6, 0.1e6])
    sizes = (6, 6, 6, 6, 6, 36, 1, 1, 1, 2, 2)    # stress, eta, eta_old, ..., lambda_c, lambda_c_old
    state = [np.zeros(size * n) for size in sizes]
    update(np.full(6 * n, 1.0e-5), 1.0, np.ones(n), *state)
    finite = np.isfinite(state[0].reshape(n, 6)).all(axis=1)
    np.testing.assert_array_equal(finite, [True, False, True])


# ----------------------------------------------------------------------
# Runs, measures and remeshing
# ----------------------------------------------------------------------

def _run_params():
    params = _build_params("r2_2L_PSD")
    params.Lx, params.Ly = 4.0, 2.0
    params.Field_Orders.update(u=1, phasefield=1)
    params.Solution_Steps = {"u": 0, "phasefield": 1}
    params.Models.update({
        "PF": {
            "Type": "Glaciology:PhaseField",
            "Material": "Ice",
            "CDF_field": "PsiField",
            "Irreversible_Method": "Hist",
        },
        "BC_Bottom": {"Type": "Boundary:PrescribedValue", "boundary": "bottom", "field": "u_y", "value": 0.0},
        "BC_Left": {"Type": "Boundary:PrescribedValue", "boundary": "left", "field": "u_x", "value": 0.0},
        "BC_Right": {
            "Type": "Boundary:DisplacementControl",
            "boundary": "right",
            "field": "u_x",
            "rate": 1.0e-4,
            "Dummy": 1.0e12,
        },
    })
    params.ModelNames = list(params.Models)
    params.Linear_Solver = LinearSolver.LU
    params.n_passes, params.max_it = 2, 30
    params.line_search, params.line_search_lims = True, [0.1, 1.0]
    params.rel_tol, params.abs_tol = 1e-6, 1e-6
    params.start_time, params.t, params.end_time, params.dt = 0.0, 0.0, 3.0, 1.0
    params.Plot_Measures = []
    return params


def test_time_solver_run_records_the_measures_and_eta_norm():
    params = _run_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    mesh.Constant_Field("poros", POROSITY)
    y = ufl.SpatialCoordinate(mesh.mesh)[1]
    mesh.Set_From_Expression("T", -25.0 + 10.0 * y)
    TimeSolver(params, physics).run()

    assert params.t == pytest.approx(params.end_time)
    measures = params.Global_Measures
    assert measures["Eta_Max"] > 0.0 and measures["Eta_Active_Area"] > 0.0
    eta = mesh.Get_Field("eta").x.array.reshape(-1, 6)
    np.testing.assert_allclose(mesh.Get_Field("etaNorm").x.array, np.linalg.norm(eta, axis=1))
    assert measures["Eta_Max"] == pytest.approx(mesh.Get_Field("etaNorm").x.array.max())


def test_solid_step_assembles_after_a_refinement():
    params = _build_ice_fracture_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solid = physics.models["Solid"]
    mesh.Constant_Field("poros", POROSITY)
    mesh.Constant_Field("T", TEMPERATURE)
    mesh.Constant_Field("phasefield", 0.5)
    physics.Get_Kmat_Fvec(solid.Step_u)
    assert solid._property_expressions

    assert _install_one_refinement(mesh).installed
    physics.Post_Remesh()
    assert solid._property_expressions == {}
    assert _contains_ufl_terminal(solid.d_e, mesh.Get_Field("phasefield"))
    assert _contains_ufl_terminal(solid.lame, mesh.Get_Field("poros"))
    assert _contains_ufl_terminal(solid.viscosity, mesh.Get_Field("T"))

    K, F = physics.Get_Kmat_Fvec(solid.Step_u)
    assert np.isfinite(K.norm()) and np.isfinite(F.norm())
    np.testing.assert_allclose(mesh.Get_Field("dam_elastic").x.array, 1e-4 + (1.0 - 1e-4) * 0.25)
    np.testing.assert_allclose(mesh.Get_Field("Lame").x.array,
                               IceMaterial().Get_Lame_Parameter(POROSITY, TEMPERATURE), rtol=1e-12)


def test_restricted_to_a_subdomain():
    params = _build_ice_fracture_params()
    params.MeshType, params.L, params.dx = "TwoSquared", 1.0, 0.5
    for name in ("Solid", "PhaseField"):
        params.Models[name]["domain"] = "domain:left"
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]
    for field_name in ("u", "poros", "T", "Lame", "TensileStrength", "PsiField", "phasefield"):
        assert mesh.Get_Field_Support(field_name).domain_names == ("domain:left",)

    mesh.Constant_Field("poros", 0.2)
    mesh.Constant_Field("T", -10.0)
    K, F = model.assemble_KF(model.Step_u)
    assert mesh.Compile_Form(K) is not None
    assert mesh.Compile_Form(F) is not None
    material = IceMaterial()
    np.testing.assert_allclose(mesh.Get_Field("Lame").x.array, material.Get_Lame_Parameter(0.2, -10.0))
    np.testing.assert_allclose(mesh.Get_Field("TensileStrength").x.array, material.Get_Tensile_Strength(0.2, -10.0))


# ----------------------------------------------------------------------
# MPI: the same error on every rank
# ----------------------------------------------------------------------

def _run_mpi_check() -> None:
    """PRIVATE: an invalid temperature in one corner (on one rank) raises the same error on every rank."""
    from mpi4py import MPI

    params = _build_params()
    params.Lx, params.Ly, params.dx = 8.0, 2.0, 0.5
    mesh, model = _model(params)
    x = ufl.SpatialCoordinate(mesh.mesh)
    mesh.Set_From_Expression("T", ufl.conditional(ufl.gt(x[0], 7.0), 1.0, -10.0))
    holds_invalid = bool((mesh.Get_Field("T").x.array > 0.0).any())
    try:
        model.assemble_KF(model.Step_u)
        message = "no error"
    except ValueError as error:
        message = str(error)
    messages = MPI.COMM_WORLD.allgather(message)
    assert not all(MPI.COMM_WORLD.allgather(holds_invalid)), "every rank holds the invalid temperature"
    assert all(m == messages[0] for m in messages), messages
    assert "temperature 'T' must be finite and at most 0 C" in messages[0], messages[0]


@pytest.mark.mpi
def test_invalid_state_raises_on_every_rank():
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")
    cmd = [mpirun, "-n", "2", sys.executable, os.path.abspath(__file__), "--mpi-check"]
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    if result.returncode != 0:
        raise AssertionError(f"MPI check failed:\n{result.stdout}\n{result.stderr}")


if __name__ == "__main__" and "--mpi-check" in sys.argv:
    _run_mpi_check()
