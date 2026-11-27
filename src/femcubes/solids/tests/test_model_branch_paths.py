"""Focused branch tests for solid-model helper and damage paths."""
import importlib
import h5py
import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix
from mpi4py import MPI
from petsc4py import PETSc

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.utils.maths_utils import Tensor_To_Mandel_UFL, nabla_s
from femcubes.solids.failure_surface_plot import plot_FailureSurface as plot_cohesive_failure_surface
from femcubes.solids.linear_elastic import LinearElastic
from femcubes.solids.phase_field import (
    GetCrackDensity,
    PhaseField,
    _smooth_positive_part,
)
from femcubes.solids.stokes import Stokes, plot_viscosity_comparison as plot_stokes_viscosity_comparison
from femcubes.solids.viscoelastic import ViscoElastic

failure_surface_plot_module = importlib.import_module("femcubes.solids.failure_surface_plot")


def _base_params(dim: int = 2):
    class Params(ParamsTemplate):
        pass

    params = Params()
    params.MeshType = "Rectangle"
    params.dx = 1.0
    params.dim = dim
    params.Lx = 2.0
    params.Ly = 1.0
    if dim == 3:
        params.Lz = 1.0
    params.Materials = {}
    params.Models = {}
    params.ModelNames = []
    params.Field_Orders = {}
    params.Solution_Steps = {}
    params.gravity = 0.0
    params.dt = 0.1
    params.t = 0.0
    params.Newmark_Beta = 0.25
    params.Newmark_Gamma = 0.5
    params.Quadrature_Degree = 2
    params.Global_Measures = {}
    return params


def _build_linear_elastic_model(
    *,
    dim: int = 2,
    damage: bool = False,
    driving_force: str = "Stress-Based",
    true_2d: bool = False,
):
    params = _base_params(dim)
    params.Field_Orders = {"u": 1}
    params.Solution_Steps = {"u": 0}
    params.Materials["Ice"] = {
        "Lame": 4.0,
        "Shear": 3.0,
        "Density": 1.0,
        "LE_Damping": 0.0,
        "Gc": 1.0,
        "ell": 0.25,
        "f_t": 2.0,
        "kappa": 1.0e-3,
        "PF_formulation": "AT2",
    }
    params.Materials["Ice"]["CrackDrivingForce"] = driving_force
    params.Models["Solid"] = {
        "Type": "Solid:LinearElastic",
        "Material": "Ice",
        "DamageModel": "PhaseField" if damage else None,
        "Inertia": False,
        "true_2d": true_2d,
    }

    mesh = Mesh(params)
    if damage:
        mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
    model = LinearElastic("Solid", params, mesh)
    mesh.finalize_test_functions()
    model.initialize_fields()
    if damage:
        mesh.Constant_Field("phasefield", 0.2)
    return params, mesh, model


def _build_phasefield_model(
    *,
    irreversible_method: str = "Hist",
    damping: float = 0.0,
    pf_formulation: str = "AT2",
    irreversibility_settings: dict | None = None,
    gc: float = 1.0,
    ell: float = 0.25,
):
    params = _base_params(2)
    params.Field_Orders = {"phasefield": 2 if irreversible_method == "AL" else 1}
    params.Solution_Steps = {"phasefield": 0}
    params.dt = 1.0e-12
    params.t = 0.0
    params.Materials["Ice"] = {
        "Gc": gc,
        "ell": ell,
        "kappa": 1.0e-3,
        "PF_formulation": pf_formulation,
        "pf_damping": damping,
    }
    params.Models["pf"] = {
        "Type": "Solid:PhaseField",
        "Material": "Ice",
        "Irreversible_Method": irreversible_method,
        "CDF_field": "ElasticEnergy",
    }
    if irreversibility_settings is not None:
        params.Models["pf"].update(irreversibility_settings)

    mesh = Mesh(params)
    mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density", ())
    model = PhaseField("pf", params, mesh)
    mesh.finalize_test_functions()
    model.initialize_fields()
    mesh.Zero_Field("ElasticEnergy")
    return params, mesh, model


def _assemble_phasefield_residual_component(mesh, residual_form, field_name):
    field_index = mesh.step_fields[0].index(field_name)
    residual = assemble_vector(fem.form(ufl.extract_blocks(residual_form)[field_index]))
    residual.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    try:
        index_map = mesh.fields[field_name].function_space.dofmap.index_map
        return residual.array[: index_map.size_local].copy()
    finally:
        residual.destroy()


def _assemble_phasefield_tangent_block_norm(mesh, tangent_form, row_field, column_field):
    row_index = mesh.step_fields[0].index(row_field)
    column_index = mesh.step_fields[0].index(column_field)
    matrix_form = fem.form(ufl.extract_blocks(tangent_form)[row_index][column_index])
    matrix = create_matrix(matrix_form)
    assemble_matrix(matrix, matrix_form)
    matrix.assemble()
    try:
        return matrix.norm(PETSc.NormType.FROBENIUS)
    finally:
        matrix.destroy()


def _build_stokes_model(
    *,
    dim: int = 2,
    damage: bool = False,
    inertia: bool = False,
    driving_force: str = "Stress-Based",
    pf_formulation: str = "AT2",
    true_2d: bool = False,
):
    params = _base_params(dim)
    params.Field_Orders = {"v": 2}
    params.Solution_Steps = {"v": 0, "p": 0}
    params.Materials["Ice"] = {
        "Bulk": 10.0,
        "Density": 1.0,
        "Visc_Type": "Newtonian",
        "Viscosity": 5.0,
        "Glen_A": 1.0e-24,
        "Glen_n": 3.0,
        "visc_min": 1.0e1,
        "visc_max": 1.0e8,
        "Inertia": inertia,
        "Gc": 1.0,
        "ell": 0.25,
        "f_t": 2.0,
        "kappa": 1.0e-3,
        "PF_formulation": pf_formulation,
    }
    params.Materials["Ice"]["CrackDrivingForce"] = driving_force
    params.Models["Solid"] = {
        "Type": "Solid:Stokes",
        "Material": "Ice",
        "DamageModel": "PhaseField" if damage else None,
        "true_2d": true_2d,
    }

    mesh = Mesh(params)
    if damage:
        mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
        mesh.Add_Field("phasefieldOld", "Previous phasefield", "CG", 1)
    model = Stokes("Solid", params, mesh)
    mesh.finalize_test_functions()
    model.initialize_fields()
    if damage:
        mesh.Constant_Field("phasefield", 0.2)
    return params, mesh, model


def _build_viscoelastic_model(
    *,
    viscosity_type: str = "Newtonian",
    pf_formulation: str = "AT2",
    damage: bool = False,
    damage_model=None,
    inertia: bool = False,
    gravity: float = 0.0,
    driving_force: str = "Stress-Based",
    true_2d: bool = False,
):
    params = _base_params(2)
    params.gravity = gravity
    params.Field_Orders = {"u": 1}
    params.Solution_Steps = {"u": 0}
    params.Materials["Ice"] = {
        "Lame": 4.0,
        "Shear": 3.0,
        "Bulk": 10.0,
        "Density": 1.0,
        "LE_Damping": 0.0,
        "Visc_Type": viscosity_type,
        "Viscosity": 5.0,
        "Glen_A": 1.0e-24,
        "Glen_n": 3.0,
        "Inertia": inertia,
        "Gc": 1.0,
        "ell": 0.25,
        "f_t": 2.0,
        "kappa": 1.0e-3,
        "PF_formulation": pf_formulation,
    }
    if damage_model is None:
        damage_model = "PhaseField" if damage else None

    params.Materials["Ice"]["CrackDrivingForce"] = driving_force
    params.Models["Solid"] = {
        "Type": "Solid:ViscoElastic",
        "Material": "Ice",
        "DamageModel": damage_model,
        "true_2d": true_2d,
    }

    mesh = Mesh(params)
    if damage_model == "PhaseField":
        mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
    model = ViscoElastic("Solid", params, mesh)
    mesh.finalize_test_functions()
    model.initialize_fields()
    if damage_model == "PhaseField":
        mesh.Constant_Field("phasefield", 0.2)
    return params, mesh, model


def _build_cohesive_viscoelastic_model(
    *, consistent_stiffness=False, plane_stress=False, true_2d=False
):
    params = _base_params(2)
    params.Field_Orders = {"u": 1}
    params.Solution_Steps = {"u": 0}
    params.Materials["Ice"] = {
        "Lame": 4.0,
        "Shear": 3.0,
        "Bulk": 10.0,
        "Density": 2.0,
        "LE_Damping": 0.0,
        "Visc_Type": "Newtonian",
        "Viscosity": 5.0,
        "Inertia": True,
        "Gc": 1.0,
        "ell": 0.25,
        "f_t": 2.0,
        "f_s": 1.5,
        "kappa": 1.0e-3,
    }
    params.Models["Solid"] = {
        "Type": "Solid:Cohesive_ViscoElastic",
        "Material": "Ice",
        "DamageModel": "PhaseField",
        "consistent_stiffness": consistent_stiffness,
        "plane_stress": plane_stress,
        "true_2d": true_2d,
    }

    from femcubes.solids.cohesive_viscoelastic import Cohesive_ViscoElastic

    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
    model = Cohesive_ViscoElastic("Solid", params, mesh)
    mesh.finalize_test_functions()
    model.initialize_fields()
    mesh.Zero_Field("phasefield")
    return params, mesh, model


@pytest.mark.parametrize("configured", [False, True])
def test_cohesive_viscoelastic_selects_configured_global_tangent(configured):
    _, _, model = _build_cohesive_viscoelastic_model(
        consistent_stiffness=configured
    )

    assert model.consistent_stiffness is configured


def test_cohesive_viscoelastic_true_2d_uses_embedded_in_plane_stiffness():
    _, _, model = _build_cohesive_viscoelastic_model(true_2d=True)

    expected_stiffness = np.zeros((6, 6))
    expected_stiffness[0, 0] = expected_stiffness[1, 1] = 10.0
    expected_stiffness[0, 1] = expected_stiffness[1, 0] = 4.0
    expected_stiffness[5, 5] = 6.0
    actual_stiffness = np.asarray(
        [[float(model.D[i, j]) for j in range(6)] for i in range(6)]
    )

    assert model.true_2d
    assert not model.plane_stress
    assert model.viscosity == 10.0
    np.testing.assert_allclose(actual_stiffness, expected_stiffness)


def test_cohesive_viscoelastic_rejects_true_2d_plane_stress_combination():
    with pytest.raises(ValueError, match="cannot both be enabled"):
        _build_cohesive_viscoelastic_model(plane_stress=True, true_2d=True)


def test_linear_elastic_phasefield_update_is_noop_without_damage():
    _, mesh, model = _build_linear_elastic_model(damage=False)

    before = mesh.Get_Field("u").x.array.copy()
    model.Update_PhaseField_Fields()

    np.testing.assert_allclose(mesh.Get_Field("u").x.array, before)


@pytest.mark.parametrize("dim", [2, 3])
def test_linear_elastic_stores_embedded_3d_stress_at_quadrature_points(dim):
    params, mesh, model = _build_linear_elastic_model(dim=dim, damage=False)
    displacement = mesh.Get_Field("u")
    if dim == 2:
        displacement.interpolate(
            lambda x: np.vstack(
                (
                    0.1 * x[0],
                    0.2 * x[1],
                )
            )
        )
        expected_stress = [1.8, 2.4, 1.2, 0.0, 0.0, 0.0]
    else:
        displacement.interpolate(
            lambda x: np.vstack(
                (
                    0.1 * x[0],
                    0.2 * x[1],
                    0.3 * x[2],
                )
            )
        )
        expected_stress = [3.0, 3.6, 4.2, 0.0, 0.0, 0.0]
    displacement.x.scatter_forward()

    model.Update_Weak_Constants(params.t, params.dt, model.My_Step)

    sigma = mesh.Get_Field("sigma")
    assert tuple(sigma.function_space.element.value_shape) == (6, 1)
    np.testing.assert_allclose(
        sigma.x.array.reshape(-1, 6),
        np.tile(np.array(expected_stress), (sigma.x.array.size // 6, 1)),
        atol=1.0e-14,
    )


def test_linear_elastic_true_2d_zeroes_out_of_plane_stress():
    params, mesh, model = _build_linear_elastic_model(true_2d=True)
    mesh.Get_Field("u").interpolate(
        lambda x: np.vstack((0.1 * x[0], 0.2 * x[1]))
    )
    mesh.Get_Field("u").x.scatter_forward()

    model.Update_Weak_Constants(params.t, params.dt, model.My_Step)

    np.testing.assert_allclose(
        mesh.Get_Field("sigma").x.array.reshape(-1, 6),
        np.tile(np.array([1.8, 2.4, 0.0, 0.0, 0.0, 0.0]),
                (mesh.Get_Field("sigma").x.array.size // 6, 1)),
        atol=1.0e-14,
    )


def test_linear_elastic_true_2d_rejects_3d_model():
    with pytest.raises(ValueError, match="only supported for 2-D"):
        _build_linear_elastic_model(dim=3, true_2d=True)


@pytest.mark.parametrize("driving_force", ["FullEnergy", "VolDev", "VolOnly"])
def test_linear_elastic_phasefield_driving_force_variants(driving_force):
    _, mesh, model = _build_linear_elastic_model(damage=True, driving_force=driving_force)

    model.Update_PhaseField_Fields()

    assert np.isfinite(mesh.Get_Field("ElasticEnergy").x.array).all()


def test_linear_elastic_stress_based_phasefield_update_supports_3d():
    _, mesh, model = _build_linear_elastic_model(dim=3, damage=True, driving_force="Stress-Based")

    model.Update_PhaseField_Fields()

    assert np.isfinite(mesh.Get_Field("ElasticEnergy").x.array).all()


def test_linear_elastic_unknown_crack_driving_force_raises():
    _, _, model = _build_linear_elastic_model(damage=True, driving_force="Unknown")

    with pytest.raises(ValueError, match="Unknown CrackDrivingForce type"):
        model.Update_PhaseField_Fields()


def test_phasefield_hist_startup_and_damping_assemble():
    _, mesh, model = _build_phasefield_model(irreversible_method="Hist", damping=1.0)

    assert not model.Higher_Order
    assert "phasefield_grad" not in mesh.fields
    assert "phasefield_gradOld" not in mesh.fields

    K, F = model.assemble_KF(model.My_Step)

    assert K is not None
    assert F is not None


def test_phasefield_reset_step_restores_hist_state():
    _, mesh, model = _build_phasefield_model(irreversible_method="Hist")

    mesh.Constant_Field("phasefield", 0.2)
    mesh.Constant_Field("phasefield_Hist", 0.3)
    model.commit()

    mesh.Constant_Field("phasefield", 0.8)
    mesh.Constant_Field("phasefield_Hist", 0.9)

    model.reset_step()

    np.testing.assert_allclose(mesh.Get_Field("phasefield").x.array, 0.2)
    np.testing.assert_allclose(mesh.Get_Field("phasefield_Hist").x.array, 0.3)


def test_phasefield_reset_step_restores_augmented_lagrangian_state():
    _, mesh, model = _build_phasefield_model(irreversible_method="AL")

    assert "LMultiplier_PFOld" in mesh.fields

    mesh.Constant_Field("phasefield", 0.2)
    mesh.Constant_Field("LMultiplier_PF", 0.3)
    model.commit()

    mesh.Constant_Field("phasefield", 0.8)
    mesh.Constant_Field("LMultiplier_PF", 0.9)

    model.reset_step()

    np.testing.assert_allclose(mesh.Get_Field_For_Output("phasefield").field.x.array, 0.2)
    np.testing.assert_allclose(mesh.Get_Field_For_Output("LMultiplier_PF").field.x.array, 0.3)


def test_phasefield_augmented_lagrangian_uses_configurable_active_set_parameter():
    _, _, default_model = _build_phasefield_model(irreversible_method="AL")
    _, _, configured_model = _build_phasefield_model(
        irreversible_method="AL",
        irreversibility_settings={"Irreversibility_Active_Set_Parameter": 2.5},
    )

    assert default_model.irreversibility_active_set_parameter == pytest.approx(1.0)
    assert configured_model.irreversibility_active_set_parameter == pytest.approx(2.5)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, True])
def test_phasefield_augmented_lagrangian_rejects_invalid_active_set_parameter(value):
    with pytest.raises(ValueError, match="positive finite"):
        _build_phasefield_model(
            irreversible_method="AL",
            irreversibility_settings={"Irreversibility_Active_Set_Parameter": value},
        )


def test_phasefield_augmented_lagrangian_is_scaled_by_gc_over_ell():
    residuals = []
    for gc in (1.0, 3.0):
        _, mesh, model = _build_phasefield_model(
            irreversible_method="AL",
            gc=gc,
            ell=0.25,
        )
        mesh.Constant_Field("phasefield", 0.0)
        mesh.Constant_Field("phasefieldOld", 0.0)
        mesh.Constant_Field("LMultiplier_PF", -0.5)
        _, residual_form = model.assemble_KF(model.My_Step)
        residuals.append(
            _assemble_phasefield_residual_component(
                mesh,
                residual_form,
                "phasefield",
            )
        )

    assert np.linalg.norm(residuals[0]) > 0.0
    np.testing.assert_allclose(residuals[1], 3.0 * residuals[0], rtol=1.0e-12, atol=1.0e-12)


def test_phasefield_augmented_lagrangian_enforces_unclipped_old_endpoint():
    _, mesh, model = _build_phasefield_model(
        irreversible_method="AL",
        irreversibility_settings={"Irreversibility_Active_Set_Parameter": 2.0},
    )
    mesh.Constant_Field("phasefield", 0.99)
    mesh.Constant_Field("phasefieldOld", 1.0)
    mesh.Constant_Field("LMultiplier_PF", -0.5)
    _, residual_form = model.assemble_KF(model.My_Step)
    multiplier_residual = _assemble_phasefield_residual_component(
        mesh,
        residual_form,
        "LMultiplier_PF",
    )

    assert np.linalg.norm(multiplier_residual) > 1.0e-6


def test_phasefield_augmented_lagrangian_has_exact_active_constraint_residual():
    _, mesh, model = _build_phasefield_model(
        irreversible_method="AL",
        irreversibility_settings={"Irreversibility_Active_Set_Parameter": 2.0},
    )
    mesh.Constant_Field("phasefield", 0.4)
    mesh.Constant_Field("phasefieldOld", 0.4)
    mesh.Constant_Field("LMultiplier_PF", -0.5)
    _, residual_form = model.assemble_KF(model.My_Step)
    multiplier_residual = _assemble_phasefield_residual_component(
        mesh,
        residual_form,
        "LMultiplier_PF",
    )

    np.testing.assert_allclose(multiplier_residual, 0.0, rtol=0.0, atol=1.0e-14)


def test_phasefield_augmented_lagrangian_activates_intact_at1_bound_tangent():
    _, mesh, model = _build_phasefield_model(
        irreversible_method="AL",
        pf_formulation="AT1",
    )
    mesh.Constant_Field("phasefield", 0.0)
    mesh.Constant_Field("phasefieldOld", 0.0)
    mesh.Constant_Field("LMultiplier_PF", 0.0)
    tangent_form, _ = model.assemble_KF(model.My_Step)

    phase_multiplier_norm = _assemble_phasefield_tangent_block_norm(
        mesh,
        tangent_form,
        "phasefield",
        "LMultiplier_PF",
    )
    multiplier_phase_norm = _assemble_phasefield_tangent_block_norm(
        mesh,
        tangent_form,
        "LMultiplier_PF",
        "phasefield",
    )

    assert phase_multiplier_norm > 0.0
    assert multiplier_phase_norm > 0.0


@pytest.mark.parametrize(
    ("pf_formulation", "expected_factor"),
    [
        ("AT1", 27.0 / (64.0 * 0.02**2)),
        ("AT2", 1.0 / 0.02**2 - 1.0),
        ("HO-AT1", 27.0 / (64.0 * 0.02**2)),
        ("HO-AT2", 1.0 / 0.02**2 - 1.0),
    ],
)
def test_phasefield_penalty_registers_only_primal_fields_and_assembles(
    pf_formulation,
    expected_factor,
):
    _, mesh, model = _build_phasefield_model(
        irreversible_method="Penalty",
        pf_formulation=pf_formulation,
        irreversibility_settings={
            "Irreversibility_Tolerance": 0.02,
            "Irreversibility_Smoothing": 0.001,
        },
    )

    assert "phasefield_Hist" not in mesh.fields
    assert "phasefield_HistOld" not in mesh.fields
    assert "LMultiplier_PF" not in mesh.fields
    assert "LMultiplier_PFOld" not in mesh.fields
    assert model.irreversibility_tolerance == pytest.approx(0.02)
    assert model.irreversibility_smoothing == pytest.approx(0.001)
    assert model.irreversibility_penalty_factor == pytest.approx(expected_factor)

    mesh.Constant_Field("phasefield", 0.2)
    mesh.Constant_Field("phasefieldOld", 0.4)
    K, F = model.assemble_KF(model.My_Step)

    assert mesh.Compile_Form(K) is not None
    assert mesh.Compile_Form(F) is not None


def test_phasefield_penalty_factor_can_be_overridden():
    _, _, model = _build_phasefield_model(
        irreversible_method="Penalty",
        irreversibility_settings={
            "Irreversibility_Penalty_Factor": 123.0,
        },
    )

    assert model.irreversibility_penalty_factor == pytest.approx(123.0)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"Irreversibility_Tolerance": 0.0}, "positive finite"),
        ({"Irreversibility_Tolerance": 1.0}, "smaller than 1"),
        ({"Irreversibility_Smoothing": 0.0}, "positive finite"),
        (
            {
                "Irreversibility_Tolerance": 0.01,
                "Irreversibility_Smoothing": 0.02,
            },
            "must not exceed",
        ),
        ({"Irreversibility_Penalty_Factor": np.inf}, "positive finite"),
    ],
)
def test_phasefield_penalty_rejects_invalid_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        _build_phasefield_model(
            irreversible_method="Penalty",
            irreversibility_settings=settings,
        )


def test_phasefield_unknown_irreversibility_method_raises():
    with pytest.raises(ValueError, match="Unknown Irreversible_Method"):
        _build_phasefield_model(irreversible_method="Unknown")


@pytest.mark.parametrize(
    ("violation_scale", "expected"),
    [
        (-1.0, (0.0, 0.0, 0.0)),
        (0.0, (0.0, 0.0, 0.0)),
        (0.5, (3.0 / 32.0, 0.5, 1.5)),
        (1.0, (0.5, 1.0, 0.0)),
        (2.0, (1.5, 1.0, 0.0)),
    ],
)
def test_smooth_positive_part_values_and_derivatives(violation_scale, expected):
    domain = dmesh.create_rectangle(
        MPI.COMM_SELF,
        ((0.0, 0.0), (1.0, 1.0)),
        (1, 1),
    )
    smoothing = 0.01
    violation = fem.Constant(domain, violation_scale * smoothing)
    positive_part, first_derivative, second_derivative = _smooth_positive_part(
        violation,
        smoothing,
    )
    measure = ufl.dx(domain=domain)
    area = fem.assemble_scalar(fem.form(1.0 * measure))

    values = (
        fem.assemble_scalar(fem.form(positive_part * measure)) / area / smoothing,
        fem.assemble_scalar(fem.form(first_derivative * measure)) / area,
        fem.assemble_scalar(fem.form(second_derivative * measure))
        / area
        * smoothing,
    )

    np.testing.assert_allclose(values, expected, rtol=1.0e-12, atol=1.0e-12)


def test_phasefield_penalty_reset_restores_committed_phasefield_without_auxiliary_state():
    _, mesh, model = _build_phasefield_model(irreversible_method="Penalty")

    mesh.Constant_Field("phasefield", 0.4)
    model.commit()
    mesh.Constant_Field("phasefield", 0.1)

    model.reset_step()

    np.testing.assert_allclose(mesh.Get_Field("phasefield").x.array, 0.4)
    assert "phasefield_HistOld" not in mesh.fields
    assert "LMultiplier_PFOld" not in mesh.fields


def test_phasefield_unknown_formulation_raises():
    with pytest.raises(ValueError, match="Unknown PF_formulation"):
        _build_phasefield_model(pf_formulation="Unknown")


def test_get_crack_density_higher_order_requires_auxiliary_gradient():
    with pytest.raises(ValueError, match="requires the auxiliary gradient field"):
        GetCrackDensity(0.25, 0.0, model="HO-AT1")


def test_get_crack_density_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown phase-field formulation"):
        GetCrackDensity(0.25, 0.0, model="Unknown")


@pytest.mark.parametrize(
    ("model", "expected_gamma", "expected_f", "expected_k"),
    [
        (
            "AT1",
            lambda ell, p: 3.0 / (8.0 * ell)
            * (p + ell**2 * ufl.inner(ufl.grad(p), ufl.grad(p))),
            lambda ell, p, p_t: 3.0 / (8.0 * ell)
            * (p_t + 2.0 * ell**2 * ufl.inner(ufl.grad(p_t), ufl.grad(p))),
            lambda ell, p_t, p_tr: 3.0 / (8.0 * ell)
            * (2.0 * ell**2 * ufl.inner(ufl.grad(p_t), ufl.grad(p_tr))),
        ),
        (
            "AT2",
            lambda ell, p: 1.0 / (2.0 * ell)
            * (p**2 + ell**2 * ufl.inner(ufl.grad(p), ufl.grad(p))),
            lambda ell, p, p_t: 1.0 / (2.0 * ell)
            * (2.0 * p_t * p + 2.0 * ell**2 * ufl.inner(ufl.grad(p_t), ufl.grad(p))),
            lambda ell, p_t, p_tr: 1.0 / (2.0 * ell)
            * (2.0 * p_t * p_tr + 2.0 * ell**2 * ufl.inner(ufl.grad(p_t), ufl.grad(p_tr))),
        ),
    ],
)
def test_get_crack_density_second_order_matches_standard_forms(
    model, expected_gamma, expected_f, expected_k
):
    domain = dmesh.create_rectangle(MPI.COMM_SELF, ((0.0, 0.0), (2.0, 1.0)), (2, 2))
    x = ufl.SpatialCoordinate(domain)
    dx = ufl.dx(domain=domain)
    ell = 0.25

    p = x[0] + 2.0 * x[1]
    p_t = 1.0 + x[1]
    p_tr = 2.0 - x[0]

    gamma, f_term, k_term = GetCrackDensity(ell, p, model=model, p_t=p_t, p_tr=p_tr)

    gamma_error = fem.assemble_scalar(fem.form((gamma - expected_gamma(ell, p)) * dx))
    f_error = fem.assemble_scalar(fem.form((f_term - expected_f(ell, p, p_t)) * dx))
    k_error = fem.assemble_scalar(fem.form((k_term - expected_k(ell, p_t, p_tr)) * dx))

    assert abs(gamma_error) < 1.0e-12
    assert abs(f_error) < 1.0e-12
    assert abs(k_error) < 1.0e-12


@pytest.mark.parametrize("pf_formulation", ["HO-AT1", "HO-AT2"])
def test_phasefield_higher_order_formulations_add_auxiliary_gradient_and_assemble(pf_formulation):
    _, mesh, model = _build_phasefield_model(
        irreversible_method="Hist" if pf_formulation == "HO-AT2" else "Penalty",
        damping=1.0,
        pf_formulation=pf_formulation,
    )

    assert model.Higher_Order
    assert "phasefield_grad" in mesh.fields
    assert "phasefield_gradOld" in mesh.fields

    mesh.Constant_Field("phasefield", 0.2)
    mesh.Constant_Field("phasefield_grad", 0.1)
    mesh.Constant_Field("ElasticEnergy", 0.4)

    K, F = model.assemble_KF(model.My_Step)

    assert K is not None
    assert F is not None
    mesh.Compile_Form(K)
    mesh.Compile_Form(F)

    before_commit = mesh.Get_Field_For_Output("phasefield_grad").field.x.array.copy()
    model.commit()

    mesh.Zero_Field("phasefield_grad")
    model.reset_step()

    np.testing.assert_allclose(
        mesh.Get_Field_For_Output("phasefield_grad").field.x.array,
        before_commit,
    )

    crack_length = model.Evaluate_Crack_Length(
        mesh.Get_Field("phasefield", step=model.My_Step),
        mesh.Get_Field("phasefieldOld"),
    )
    assert np.isfinite(crack_length)


def test_stokes_compute_viscosity_invalid_type_raises():
    _, _, model = _build_stokes_model(damage=False)
    model.visc_type = "Unknown"

    with pytest.raises(ValueError, match="Unknown viscosity type"):
        model.compute_viscosity(ufl.as_tensor(((0.0, 0.0), (0.0, 0.0))))


def test_stokes_selects_driver_from_material_parameters():
    _, _, model = _build_stokes_model(damage=True, driving_force="VolOnly")
    assert model.DrivingForce_Type == "VolOnly"


def test_stokes_damage_update_and_inertia_assembly_paths():
    _, mesh, model = _build_stokes_model(damage=True, inertia=True)

    K, F = model.assemble_KF(model.My_Step)

    assert K is not None
    assert F is not None
    assert np.isfinite(mesh.Get_Field("ElasticEnergy").x.array).all()


def test_stokes_stores_cauchy_stress_at_quadrature_points():
    params, mesh, model = _build_stokes_model(damage=False)
    mesh.Constant_Field("p", 2.0)

    model.Update_Weak_Constants(params.t, params.dt, model.My_Step)

    sigma = mesh.Get_Field("sigma")
    assert tuple(sigma.function_space.element.value_shape) == (6, 1)
    np.testing.assert_allclose(
        sigma.x.array.reshape(-1, 6),
        np.tile(np.array([-2.0, -2.0, -2.0, 0.0, 0.0, 0.0]), (sigma.x.array.size // 6, 1)),
    )


@pytest.mark.parametrize("pressure", [0.0, 2.0, -3.0])
@pytest.mark.parametrize("damage", [False, True])
def test_stokes_true_2d_zeroes_out_of_plane_stress(pressure, damage):
    params, mesh, model = _build_stokes_model(true_2d=True, damage=damage)
    mesh.Constant_Field("p", pressure)
    degradation = 1.0
    if damage:
        degradation = params.Materials["Ice"]["kappa"] + (1-params.Materials["Ice"]["kappa"])*0.8**2

    model.Update_Weak_Constants(params.t, params.dt, model.My_Step)

    np.testing.assert_allclose(
        mesh.Get_Field("sigma").x.array.reshape(-1, 6),
        np.tile(np.array([-degradation*pressure, -degradation*pressure, 0.0, 0.0, 0.0, 0.0]),
                (mesh.Get_Field("sigma").x.array.size // 6, 1)),
    )


def test_stokes_true_2d_uses_intrinsic_viscosity_invariant():
    _, _, model = _build_stokes_model(true_2d=True, damage=True)
    viscosity_input_shapes = []

    def record_viscosity_input(dev_strain_rate):
        viscosity_input_shapes.append(dev_strain_rate.ufl_shape)
        return 5.0

    model.compute_viscosity = record_viscosity_input

    model.Get_Weak_Form(0.0, 1.0, model.My_Step)
    assert viscosity_input_shapes
    assert all(shape == (2, 2) for shape in viscosity_input_shapes)

    viscosity_input_shapes.clear()
    model.Update_Weak_Constants(0.0, 1.0, model.My_Step)
    assert viscosity_input_shapes
    assert all(shape == (2, 2) for shape in viscosity_input_shapes)


def test_stokes_true_2d_rejects_3d_model():
    with pytest.raises(ValueError, match="only supported for 2-D"):
        _build_stokes_model(dim=3, true_2d=True)


def test_stokes_flow_and_fracture_use_3d_viscosity_invariant_in_2d():
    _, _, model = _build_stokes_model(damage=True)
    viscosity_input_shapes = []

    def record_viscosity_input(dev_strain_rate):
        viscosity_input_shapes.append(dev_strain_rate.ufl_shape)
        return 5.0

    model.compute_viscosity = record_viscosity_input

    model.Get_Weak_Form(0.0, 1.0, model.My_Step)
    assert viscosity_input_shapes
    assert all(shape == (3, 3) for shape in viscosity_input_shapes)

    viscosity_input_shapes.clear()
    model.Update_Weak_Constants(0.0, 1.0, model.My_Step)
    assert viscosity_input_shapes
    assert all(shape == (3, 3) for shape in viscosity_input_shapes)


def test_stokes_phasefield_update_is_noop_without_damage():
    _, _, model = _build_stokes_model(damage=False)

    model.Update_Weak_Constants(0.0, 1.0, model.My_Step)

    assert model.d_e == 1.0
    assert model.d_rho == 1.0


def test_stokes_phasefield_update_supports_3d_at1_branch():
    _, mesh, model = _build_stokes_model(
        dim=3, damage=True, driving_force="Stress-Based", pf_formulation="AT1"
    )

    model.Update_Weak_Constants(0.0, 1.0, model.My_Step)

    assert np.isfinite(mesh.Get_Field("ElasticEnergy").x.array).all()


def test_stokes_wrong_step_returns_none():
    _, _, model = _build_stokes_model()

    assert model.assemble_KF(model.My_Step + 1) is None


def test_stokes_unknown_crack_driving_force_raises():
    _, _, model = _build_stokes_model(damage=True, driving_force="Unknown")

    with pytest.raises(ValueError, match="Unknown CrackDrivingForce type"):
        model.Update_Weak_Constants(0.0, 1.0, model.My_Step)


def test_stokes_plot_viscosity_comparison_runs(monkeypatch):
    params = _base_params(2)
    params.Materials["Ice"] = {
        "Glen_A": 1.0e-24,
        "Glen_n": 3.0,
        "visc_min": 1.0e1,
        "visc_max": 1.0e8,
        "Viscosity": 5.0,
    }

    show_calls = []

    def _fake_show():
        show_calls.append(True)

    monkeypatch.setattr("matplotlib.pyplot.show", _fake_show)

    fig, ax = plot_stokes_viscosity_comparison(params, show=True)

    assert fig is not None
    assert ax is not None
    assert show_calls == [True]


def test_cohesive_plot_failure_surface_runs(tmp_path):
    poisson = 0.35
    params = _base_params(2)
    params.Materials["Ice"] = {
        "Bulk": 1.7e9,
        "Lame": 1.0e9,
        "Shear": 3.5e8,
        "f_t": 0.4e6,
        "f_s": 0.4e6,
        "Poisson": poisson,
        "DP_phi": np.degrees(np.arctan(3.0 / (1.0 + poisson))),
        "DP_q": 9.0e6,
        "FailureType": "r1_DP",
    }
    params.Models["Solid"] = {"Material": "Ice"}

    output_path = tmp_path / "failure_surface.png"
    saved_path = plot_cohesive_failure_surface(
        params.Materials["Ice"],
        np.linspace(-10.0e6, 2.0e6, 41),
        output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()


def test_cohesive_plot_failure_surface_evaluates_true_2d_contour(tmp_path, monkeypatch):
    params = _base_params(2)
    params.Materials["Ice"] = {
        "Bulk": 1.7e9,
        "Lame": 1.0e9,
        "Shear": 3.5e8,
        "f_t": 0.4e6,
        "f_s": 0.4e6,
        "FailureType": "r1",
    }
    strength_calls = []
    original_compute_strength = failure_surface_plot_module.compute_strength_surface_point

    def _record_compute_strength(*args, **kwargs):
        strength_calls.append(kwargs.get("true_2d", False))
        return original_compute_strength(*args, **kwargs)

    monkeypatch.setattr(
        failure_surface_plot_module, "compute_strength_surface_point", _record_compute_strength,
    )

    output_path = tmp_path / "failure_surface_true_2d.png"
    plot_cohesive_failure_surface(
        params.Materials["Ice"],
        np.linspace(-1.0e6, 1.0e6, 9),
        output_path,
    )

    assert output_path.is_file()
    assert any(strength_calls)


def test_cohesive_plot_failure_surface_returns_none_without_output_path():
    params = _base_params(2)
    params.Materials["Ice"] = {
        "Bulk": 1.7e9,
        "Lame": 1.0e9,
        "Shear": 3.5e8,
        "f_t": 0.4e6,
        "f_s": 0.4e6,
        "FailureType": "r1",
    }

    assert (
        plot_cohesive_failure_surface(
            params.Materials["Ice"],
            np.linspace(-1.0e6, 1.0e6, 3),
            None,
        )
        is None
    )


def test_cohesive_plot_failure_surface_returns_none_on_non_root(tmp_path, monkeypatch):
    params = _base_params(2)
    params.Materials["Ice"] = {
        "Bulk": 1.7e9,
        "Lame": 1.0e9,
        "Shear": 3.5e8,
        "f_t": 0.4e6,
        "f_s": 0.4e6,
        "FailureType": "r1",
    }
    monkeypatch.setattr(failure_surface_plot_module, "rank", 1)

    output_path = tmp_path / "should_not_exist.png"
    assert (
        plot_cohesive_failure_surface(
            params.Materials["Ice"],
            np.linspace(-1.0e6, 1.0e6, 3),
            output_path,
        )
        is None
    )
    assert not output_path.exists()


def test_cohesive_plot_failure_surface_rejects_invalid_sampling_input(tmp_path):
    params = _base_params(2)
    params.Materials["Ice"] = {
        "Bulk": 1.7e9,
        "Lame": 1.0e9,
        "Shear": 3.5e8,
        "f_t": 0.4e6,
        "f_s": 0.4e6,
        "FailureType": "r1",
    }

    with pytest.raises(ValueError, match="one-dimensional array"):
        plot_cohesive_failure_surface(
            params.Materials["Ice"],
            np.asarray([[0.0, 1.0]]),
            tmp_path / "invalid_shape.png",
        )

    with pytest.raises(ValueError, match="output file"):
        plot_cohesive_failure_surface(
            params.Materials["Ice"],
            np.linspace(-1.0e6, 1.0e6, 3),
            "",
        )


def test_cohesive_plot_failure_surface_one_lambda_branch_handles_hydrostatic_state(tmp_path):
    params = _base_params(2)
    params.Materials["Ice"] = {
        "Bulk": 1.7e9,
        "Lame": 1.0e9,
        "Shear": 3.5e8,
        "f_t": 0.4e6,
        "f_s": 0.4e6,
        "FailureType": "r2",
    }

    output_path = tmp_path / "failure_surface_r2.png"
    saved_path = plot_cohesive_failure_surface(
        params.Materials["Ice"],
        np.asarray([-1.0e6, 0.0, 1.0e6]),
        output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()


def test_cohesive_plot_failure_surface_uses_3d_constants_for_an_intrinsic_2d_material(tmp_path, monkeypatch):
    """With true_2d=True the material holds the intrinsic 2-D constants; the plane panels use the 3-D ones."""
    youngs, poisson, shear = 9.5e9, 0.35, 9.5e9 / 2.7
    lame_2d = youngs * poisson / (1.0 - poisson**2)
    lame_3d = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    material = {"Lame": lame_2d, "Shear": shear, "Bulk": lame_2d + shear, "f_t": 0.4e6, "f_s": 1.0e6,
                "FailureType": "r2_2L"}
    calls = []
    original = failure_surface_plot_module.compute_strength_surface_point

    def _record(f_t, f_s, bulk, dp_phi, dp_q, failure_type, stress, strain, true_2d=False):
        calls.append((bulk, true_2d, stress.copy(), strain.copy()))
        return original(f_t, f_s, bulk, dp_phi, dp_q, failure_type, stress, strain, true_2d=true_2d)

    monkeypatch.setattr(failure_surface_plot_module, "compute_strength_surface_point", _record)
    plot_cohesive_failure_surface(material, np.linspace(-1.0e6, 1.0e6, 5), tmp_path / "surface.png", true_2d=True)

    three_d = [call for call in calls if not call[1]]
    intrinsic = [call for call in calls if call[1]]
    assert three_d and intrinsic
    np.testing.assert_allclose([call[0] for call in three_d], lame_3d + 2.0 * shear / 3.0, rtol=1e-12)
    np.testing.assert_allclose([call[0] for call in intrinsic], lame_2d + shear, rtol=1e-12)
    for _, _, stress, strain in three_d:
        stiffness = np.full((3, 3), lame_3d) + 2.0 * shear * np.eye(3)
        np.testing.assert_allclose(stiffness @ strain[:3], stress[:3], rtol=1e-9, atol=1e-6)


def test_cohesive_plot_failure_surface_leaves_the_intrinsic_2d_panel_empty_for_opening_only(tmp_path):
    material = {"Bulk": 8.8e9, "Lame": 6.5e9, "Shear": 3.5e9, "f_t": 0.4e6, "f_s": 1.5e6, "FailureType": "r2_2L_PSD"}
    output_path = tmp_path / "failure_surface_psd.png"
    plot_cohesive_failure_surface(material, np.linspace(-2.0e6, 1.0e6, 9), output_path, export_hdf5=True)

    with h5py.File(output_path.with_suffix(".hdf5"), "r") as h5f:
        assert np.isnan(h5f["true_2d/failure_criterion"][()]).all()
        plane_stress = h5f["plane_stress/failure_criterion"][()]
        assert np.isfinite(plane_stress).all()
        assert plane_stress.min() < 0.0 < plane_stress.max()


def test_cohesive_plot_failure_surface_r1_dp(tmp_path):
    """plot_FailureSurface runs without error and produces a file for r1_DP."""
    poisson = 0.35
    params = _base_params(2)
    params.Materials["Ice"] = {
        "Bulk": 1.7e9,
        "Lame": 1.0e9,
        "Shear": 3.5e8,
        "f_t": 0.4e6,
        "f_s": 0.4e6,
        "Poisson": poisson,
        "DP_phi": np.degrees(np.arctan(3.0 / (1.0 + poisson))),
        "DP_q": 9.0e6,
        "FailureType": "r1_DP",
    }

    output_path = tmp_path / "failure_surface_r1_dp.png"
    saved_path = plot_cohesive_failure_surface(
        params.Materials["Ice"],
        np.linspace(-10.0e6, 2.0e6, 41),
        output_path,
        export_hdf5=True,
    )

    assert saved_path == output_path
    assert output_path.is_file()
    hdf5_path = output_path.with_suffix(".hdf5")
    assert hdf5_path.is_file()

    with h5py.File(hdf5_path, "r") as h5f:
        assert h5f.attrs["failure_type"] == "r1_DP"
        np.testing.assert_allclose(
            h5f["plane_stress/sigma_3"],
            0.0,
        )
        assert h5f["plane_stress/failure_criterion"].shape == (41, 41)
        assert h5f["invariant/failure_criterion"].shape[0] >= 80
        assert h5f["sigma_space/axis"].shape[0] == 20
        assert h5f["sigma_space/surface_vertices"].shape[1] == 3
        assert h5f["sigma_space/surface_faces"].shape[1] == 3


def test_cohesive_viscoelastic_lumped_mass_adds_all_displacement_components():
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector

    _, mesh, model = _build_cohesive_viscoelastic_model()
    model.LumpedMass = True
    model.da_du = 7.0
    mesh.fields["a"].x.array[:] = np.arange(len(mesh.fields["a"].x.array), dtype=float) + 1.0
    mesh.fields["a"].x.scatter_forward()

    u_t = mesh.Get_TestFunction("u", model.Step_u)
    u_tr = mesh.Get_Trial_Function("u", model.Step_u)
    u = mesh.Get_Field("u", step=model.Step_u)
    k_form = fem.form(ufl.inner(u_t, u_tr) * ufl.dx)
    f_form = fem.form(ufl.inner(u_t, u) * ufl.dx)
    K = fem.petsc.create_matrix(k_form)
    assemble_matrix(K, k_form)
    K.assemble()
    K.zeroEntries()
    F = assemble_vector(f_form)

    W, dofs, _vals = mesh.Get_Weights_Dofs(["u"], step=model.Step_u)
    a_vals = mesh.Get_Old_For_Lumped(["a"], step=model.Step_u)["a"]

    model.Add_KF(K, F, model.Step_u)
    K.assemble()

    with F.localForm() as F_local:
        np.testing.assert_allclose(F_local.array[dofs["u"]], W * model.rho * a_vals)

    diagonal = F.duplicate()
    K.getDiagonal(diagonal)
    with diagonal.localForm() as diag_local:
        np.testing.assert_allclose(diag_local.array[dofs["u"]], W * model.rho * model.da_du)

    F.set(0.0)
    model.Add_KF(None, F, model.Step_u)
    with F.localForm() as F_local:
        np.testing.assert_allclose(F_local.array[dofs["u"]], W * model.rho * a_vals)

    diagonal.destroy()
    F.destroy()
    K.destroy()


def test_viscoelastic_invalid_viscosity_type_raises_on_init():
    with pytest.raises(NotImplementedError, match="not implemented"):
        _build_viscoelastic_model(viscosity_type="Carreau")


def test_viscoelastic_true_2d_selects_intrinsic_constitutive_mode():
    from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import ViscoElasticKernels
    try:
        ViscoElasticKernels(1.0, 1.0, 1.0, 1.0, True)
    except TypeError:
        pytest.skip("viscoelastic extension must be rebuilt for true_2d support")
    _, _, model = _build_viscoelastic_model(true_2d=True)

    assert model.true_2d
    assert model.viscosity == 2.0 * 5.0


def test_viscoelastic_true_2d_rejects_3d_model():
    params = _base_params(3)
    params.Field_Orders = {"u": 1}
    params.Solution_Steps = {"u": 0}
    params.Materials["Ice"] = {
        "Lame": 4.0, "Shear": 3.0, "Bulk": 10.0, "Density": 1.0,
        "Visc_Type": "Newtonian", "Viscosity": 5.0,
    }
    params.Models["Solid"] = {
        "Type": "Solid:ViscoElastic", "Material": "Ice", "true_2d": True,
    }
    with pytest.raises(ValueError, match="only supported for 2-D"):
        ViscoElastic("Solid", params, Mesh(params))


def test_viscoelastic_damage_assembly_inertia_and_gravity_paths():
    _, mesh, model = _build_viscoelastic_model(
        damage=True,
        inertia=True,
        gravity=-9.81,
        driving_force="Stress-Based",
    )

    K, F = model.assemble_KF(model.My_Step)

    assert K is not None
    assert F is not None
    assert np.isfinite(mesh.Get_Field("ElasticEnergy").x.array).all()


def test_viscoelastic_inertia_updates_and_weak_form_use_later_newmark_state():
    """The model lifecycle must update Newmark state before assembling inertia."""
    params, mesh, model = _build_viscoelastic_model(inertia=True)
    beta = params.Newmark_Beta
    gamma = params.Newmark_Gamma

    def set_constant(field_name, value):
        mesh.Constant_Field(field_name, value)
        mesh.Get_Field(field_name).x.scatter_forward()

    def expected_newmark(u, u_old, v_old, a_old, dt):
        velocity = (
            gamma / (beta * dt) * (u - u_old)
            + (1.0 - gamma / beta) * v_old
            + dt * (1.0 - gamma / (2.0 * beta)) * a_old
        )
        acceleration = (
            (u - u_old) / (beta * dt**2)
            - v_old / (beta * dt)
            - (1.0 / (2.0 * beta) - 1.0) * a_old
        )
        return velocity, acceleration

    dt_first = 0.1
    params.dt = dt_first
    first_state = (0.04, 0.01, 0.2, -0.3)
    set_constant("uOld", first_state[1])
    set_constant("vOld", first_state[2])
    set_constant("aOld", first_state[3])
    set_constant("u", first_state[0])

    model.Update_Weak_Constants(0.0, dt_first, model.My_Step)
    expected_v, expected_a = expected_newmark(*first_state, dt_first)
    np.testing.assert_allclose(mesh.Get_Field("v").x.array, expected_v)
    np.testing.assert_allclose(mesh.Get_Field("a").x.array, expected_a)

    model.commit()

    dt_second = 0.2
    params.dt = dt_second
    second_state = (0.10, first_state[0], expected_v, expected_a)
    set_constant("u", second_state[0])
    model.Update_Weak_Constants(0.1, dt_second, model.My_Step)
    expected_v, expected_a = expected_newmark(*second_state, dt_second)
    np.testing.assert_allclose(mesh.Get_Field("v").x.array, expected_v)
    np.testing.assert_allclose(mesh.Get_Field("a").x.array, expected_a)

    def assemble_vector_values(form_expression):
        vector_form = fem.form(form_expression)
        vector = assemble_vector(vector_form)
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,
            mode=PETSc.ScatterMode.FORWARD,
        )
        with vector.localForm() as local:
            values = local.array.copy()
        vector.destroy()
        return values

    def assemble_matrix_values(form_expression):
        matrix_form = fem.form(form_expression)
        matrix = create_matrix(matrix_form)
        assemble_matrix(matrix, matrix_form)
        matrix.assemble()
        indices = list(range(matrix.getSize()[0]))
        values = np.asarray(matrix.getValues(indices, indices))
        matrix.destroy()
        return values

    time = 0.1
    K_with_inertia, F_with_inertia = model.Get_Weak_Form(
        time, dt_second, model.My_Step
    )
    model.Inertia = False
    K_without_inertia, F_without_inertia = model.Get_Weak_Form(
        time, dt_second, model.My_Step
    )

    u_test = mesh.Get_TestFunction("u", model.My_Step)
    u_trial = mesh.Get_Trial_Function("u", model.My_Step)
    acceleration = mesh.Get_Field("a")
    inertia_force = params.Materials["Ice"]["Density"] * ufl.inner(
        u_test, acceleration
    ) * ufl.dx
    inertia_tangent = (
        params.Materials["Ice"]["Density"]
        / (beta * dt_second**2)
        * ufl.inner(u_test, u_trial)
        * ufl.dx
    )

    np.testing.assert_allclose(
        assemble_vector_values(F_with_inertia)
        - assemble_vector_values(F_without_inertia),
        assemble_vector_values(inertia_force),
    )
    np.testing.assert_allclose(
        assemble_matrix_values(K_with_inertia)
        - assemble_matrix_values(K_without_inertia),
        assemble_matrix_values(inertia_tangent),
    )


def test_viscoelastic_poros_damage_update_uses_porosity_field():
    _, mesh, model = _build_viscoelastic_model(damage_model="poros")
    mesh.Constant_Field("poros", 0.25)
    strain = ufl.as_vector([0.0, 1.0e-4, 0.0, 0.0, 0.0, 0.0])

    model.Update_PhaseField_Fields(strain)

    assert isinstance(model.d_e, ufl.core.expr.Expr)
    assert isinstance(model.d_rho, ufl.core.expr.Expr)
    elastic_energy = mesh.Get_Field("ElasticEnergy").x.array
    assert np.isfinite(elastic_energy).all()
    assert np.any(elastic_energy > 0.0)


def test_viscoelastic_phasefield_update_is_noop_without_damage():
    _, mesh, model = _build_viscoelastic_model(damage=False)
    zero_strain = Tensor_To_Mandel_UFL(
        nabla_s(mesh.Get_Field("u", step=model.My_Step)), model.dim
    )

    before_u = mesh.Get_Field("u").x.array.copy()
    model.Update_PhaseField_Fields(zero_strain)

    np.testing.assert_allclose(mesh.Get_Field("u").x.array, before_u)


@pytest.mark.parametrize("driving_force", ["FullEnergy", "VolDev", "VolOnly"])
def test_viscoelastic_phasefield_driving_force_variants(driving_force):
    _, mesh, model = _build_viscoelastic_model(damage=True, driving_force=driving_force)
    zero_strain = Tensor_To_Mandel_UFL(
        nabla_s(mesh.Get_Field("u", step=model.My_Step)), model.dim
    )

    model.Update_PhaseField_Fields(zero_strain)

    assert np.isfinite(mesh.Get_Field("ElasticEnergy").x.array).all()


def test_viscoelastic_stress_based_at1_phasefield_variant():
    _, mesh, model = _build_viscoelastic_model(
        damage=True, driving_force="Stress-Based", pf_formulation="AT1"
    )
    zero_strain = Tensor_To_Mandel_UFL(
        nabla_s(mesh.Get_Field("u", step=model.My_Step)), model.dim
    )

    model.Update_PhaseField_Fields(zero_strain)

    assert np.isfinite(mesh.Get_Field("ElasticEnergy").x.array).all()


def test_viscoelastic_unknown_crack_driving_force_raises():
    _, _, model = _build_viscoelastic_model(damage=True, driving_force="Unknown")
    zero_strain = Tensor_To_Mandel_UFL(
        nabla_s(model.mesh.Get_Field("u", step=model.My_Step)), model.dim
    )

    with pytest.raises(ValueError, match="Unknown CrackDrivingForce type"):
        model.Update_PhaseField_Fields(zero_strain)


def test_viscoelastic_reset_step_restores_committed_state():
    _, mesh, model = _build_viscoelastic_model()

    mesh.Constant_Field("uOld", 1.0)
    mesh.Constant_Field("vOld", 2.0)
    mesh.Constant_Field("aOld", 3.0)
    mesh.Constant_Field("u", -1.0)
    mesh.Constant_Field("v", -2.0)
    mesh.Constant_Field("a", -3.0)
    mesh.Get_Field("Plastic_Strain_Old").x.array[:] = 4.0
    mesh.Get_Field("Plastic_Strain").x.array[:] = -4.0

    model.reset_step()

    np.testing.assert_allclose(mesh.Get_Field("u").x.array, mesh.Get_Field("uOld").x.array)
    np.testing.assert_allclose(mesh.Get_Field("v").x.array, mesh.Get_Field("vOld").x.array)
    np.testing.assert_allclose(mesh.Get_Field("a").x.array, mesh.Get_Field("aOld").x.array)
    np.testing.assert_allclose(
        mesh.Get_Field("Plastic_Strain").x.array,
        mesh.Get_Field("Plastic_Strain_Old").x.array,
    )


def test_viscoelastic_wrong_step_returns_none():
    _, _, model = _build_viscoelastic_model()

    assert model.assemble_KF(model.My_Step + 1) is None


@pytest.mark.parametrize("pf_formulation", ["AT1", "HO-AT1"])
def test_phasefield_rejects_history_irreversibility_with_at1(pf_formulation):
    # The history field does not keep phi >= 0, which the linear AT1 crack density needs.
    with pytest.raises(ValueError, match="cannot be used with PF_formulation"):
        _build_phasefield_model(irreversible_method="Hist", pf_formulation=pf_formulation)
