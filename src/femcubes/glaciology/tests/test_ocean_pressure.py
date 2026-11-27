"""Model tests for the OceanPressure glaciology boundary condition."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI
from petsc4py import PETSc

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.glaciology  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration
from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.two_squared import Sample_Parameters as TwoSquared_Sample_Parameters
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.utils.maths_utils import nabla_s
from femcubes.glaciology.ocean_pressure import OceanPressure


class OceanPressureConstantDamageField(BaseModel):
    """Test-only model registering a constant phase-field."""

    MODEL_TYPE = "Test:OceanPressureConstantDamageField"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        order = self.params.Field_Orders.get("phasefield", self.params.Field_Orders.get("u", 1))
        self.mesh.Add_Field("phasefield", "Damage", "CG", order)
        self.value = float(self.params.Models[self.name]["value"])

    def initialize_fields(self) -> None:
        self.mesh.Constant_Field("phasefield", self.value)

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        return None


def _elastic_material(youngs: float = 10.0e9, poisson: float = 0.3) -> dict:
    shear = youngs / (2.0 * (1.0 + poisson))
    return {
        "Density": 0.0,
        "Youngs": youngs,
        "Poisson": poisson,
        "LE_Damping": 0.0,
        "Shear": shear,
        "Bulk": youngs / (3.0 * (1.0 - 2.0 * poisson)),
        "Lame": shear * (2.0 * poisson) / (1.0 - 2.0 * poisson),
        "Inertia": False,
    }


def _build_ocean_pressure_elastic_params(phasefield_value=None):
    model_names = ["Solid", "BC_Left", "BC_Right", "BC_Top", "OceanLoad"]
    models = {
        "Solid": {
            "Type": "Solid:LinearElastic",
            "Material": "Ice",
            "DamageModel": None,
        },
        "BC_Left": {
            "Type": "Boundary:PrescribedValue",
            "boundary": "left",
            "field": "u_x",
            "value": 0.0,
        },
        "BC_Right": {
            "Type": "Boundary:PrescribedValue",
            "boundary": "right",
            "field": "u_x",
            "value": 0.0,
        },
        "BC_Top": {
            "Type": "Boundary:PrescribedValue",
            "boundary": "top",
            "field": "u_y",
            "value": 0.0,
        },
        "OceanLoad": {
            "Type": "glaciology:OceanPressure",
            "boundary": "bottom",
            "DamagedForces": phasefield_value is not None,
        },
    }

    if phasefield_value is not None:
        model_names.insert(-1, "Damage")
        models["Damage"] = {
            "Type": "Test:OceanPressureConstantDamageField",
            "value": phasefield_value,
        }

    class Params(ParamsTemplate):
        Global_Measures: dict = {}

        MeshType: str = "Rectangle"
        dx: float = 0.5
        dim: int = 2
        Lx: float = 2.0
        Ly: float = 4.0
        Quadrature_Degree: int = 2

        Materials: dict = {
            "Ice": _elastic_material(),
            "Water": {
                "Density": 1025.0,
                "WaterHeight": 100.0,
            },
        }
        gravity: float = 0.0
        Plot_Fields: list = []

        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 2}
        if phasefield_value is not None:
            Solution_Steps["phasefield"] = 0
            Field_Orders["phasefield"] = 1

        ModelNames = model_names
        Models = models

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 25
        line_search: bool = False
        line_search_lims: list = [0.1, 1.0]
        rel_tol: float = 1.0e-3
        abs_tol: float = 1.0e-3

        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0
        dt: float = 1.0
        Newmark_Beta: float = 0.25
        Newmark_Gamma: float = 0.5

    return Params()


def _build_ocean_pressure_direct_params(*, solution_steps=None, water_height=1.5, boundary="right"):
    class Params(ParamsTemplate):
        MeshType: str = "Rectangle"
        dx: float = 0.5
        dim: int = 2
        Lx: float = 2.0
        Ly: float = 4.0
        Quadrature_Degree: int = 2
        gravity: float = -9.81
        Materials: dict = {
            "Water": {
                "Density": 1025.0,
                "WaterHeight": water_height,
            }
        }
        Solution_Steps = solution_steps if solution_steps is not None else {"v": 0}
        Field_Orders = {"u": 1, "v": 1}
        Models = {
            "OceanLoad": {
                "boundary": boundary,
                "DamagedForces": False,
            }
        }
        ModelNames = []
        Plot_Fields: list = []

    return Params()


def _global_scalar(mesh: Mesh, expr) -> float:
    return mesh.mesh.comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)


def _l2_error(mesh: Mesh, expr) -> float:
    dx = ufl.Measure("dx", domain=mesh.mesh)
    return np.sqrt(_global_scalar(mesh, expr * expr * dx))


def _max_abs_function_error(reference: fem.Function, exact: fem.Function) -> float:
    local = np.abs(reference.x.array - exact.x.array)
    local_max = 0.0 if local.size == 0 else float(np.max(local))
    return reference.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)


def _collapsed_component(mesh: Mesh, field_name: str, component: int) -> fem.Function:
    return mesh.Get_Field_For_Output(field_name).collapse_component(component)


def _interpolate_exact_like(reference: fem.Function, exact_callable) -> fem.Function:
    exact = fem.Function(reference.function_space)
    exact.interpolate(exact_callable)
    exact.x.scatter_forward()
    return exact


def _sigma_yy(mesh: Mesh, lame: float, shear: float):
    u = mesh.Get_Field("u", step=0)
    strain = nabla_s(u)
    return lame * ufl.tr(strain) + 2.0 * shear * strain[1, 1]


def _sigma_xy(mesh: Mesh, shear: float):
    u = mesh.Get_Field("u", step=0)
    return 2.0 * shear * nabla_s(u)[0, 1]


def _run_elastic_case(params, ocean_gravity: float):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    # Keep solid self-weight out of this boundary-condition benchmark while
    # still testing OceanPressure with a physical gravitational acceleration.
    physics.models["OceanLoad"].g = ocean_gravity
    physics.Initialize_Fields()
    solver = TimeSolver(params, physics)
    solver.run()
    return mesh


@pytest.mark.full_model
@pytest.mark.parametrize(
    "phasefield_value",
    [
        pytest.param(None, id="undamaged"),
        pytest.param(0.4, id="damaged"),
    ],
)
@pytest.mark.filterwarnings("error:Expr.ufl_domain:DeprecationWarning")
def test_ocean_pressure_bottom_load_matches_linear_elastic_solution(phasefield_value):
    ocean_gravity = -9.81
    params = _build_ocean_pressure_elastic_params(phasefield_value)
    mesh = _run_elastic_case(params, ocean_gravity)

    lame = params.Materials["Ice"]["Lame"]
    shear = params.Materials["Ice"]["Shear"]
    effective_modulus = lame + 2.0 * shear
    hydrostatic_gradient = params.Materials["Water"]["Density"] * abs(ocean_gravity)
    damage_factor = 1.0
    if phasefield_value is not None:
        damage_factor = (1.0 - 1.0e-6) * (1.0 - phasefield_value) ** 2 + 1.0e-6
    bottom_displacement = (
        damage_factor
        * hydrostatic_gradient
        * params.Materials["Water"]["WaterHeight"]
        / (effective_modulus / params.Ly + damage_factor * hydrostatic_gradient)
    )
    effective_pressure = effective_modulus * bottom_displacement / params.Ly

    ux_func = _collapsed_component(mesh, "u", 0)
    uy_func = _collapsed_component(mesh, "u", 1)
    ux_exact = _interpolate_exact_like(ux_func, lambda x: np.zeros(x.shape[1], dtype=np.float64))
    uy_exact = _interpolate_exact_like(
        uy_func,
        lambda x: effective_pressure * (params.Ly - x[1]) / effective_modulus,
    )

    sigma_yy_error = _l2_error(mesh, _sigma_yy(mesh, lame, shear) + effective_pressure)
    sigma_xy_error = _l2_error(mesh, _sigma_xy(mesh, shear))

    assert _max_abs_function_error(ux_func, ux_exact) < 1.0e-8
    assert _max_abs_function_error(uy_func, uy_exact) < 1.0e-8
    assert sigma_yy_error < 5.0e-3
    assert sigma_xy_error < 1.0e-8


def test_ocean_pressure_displacement_uses_current_boundary_height_and_tangent():
    params = _build_ocean_pressure_direct_params(
        solution_steps={"u": 0},
        water_height=1.5,
        boundary="bottom",
    )
    mesh = Mesh(params)
    mesh.Add_Vector_Field("u", "Displacement", "CG", 1, params.dim)
    mesh.Add_Vector_TestFunction("u", params.dim, params.Solution_Steps["u"])
    mesh.finalize_test_functions()
    model = OceanPressure("OceanLoad", params, mesh)

    vertical_displacement = -0.25
    displacement = mesh.Get_Field("u")
    displacement.interpolate(
        lambda x: np.vstack(
            (
                np.zeros(x.shape[1], dtype=np.float64),
                np.full(x.shape[1], vertical_displacement, dtype=np.float64),
            )
        )
    )
    displacement.x.scatter_forward()

    K, F = model.assemble_KF(model.step)
    unit_y = fem.Function(displacement.function_space)
    unit_y.interpolate(
        lambda x: np.vstack(
            (
                np.zeros(x.shape[1], dtype=np.float64),
                np.ones(x.shape[1], dtype=np.float64),
            )
        )
    )
    unit_y.x.scatter_forward()

    residual_resultant = _global_scalar(mesh, ufl.action(F, unit_y))
    tangent_resultant = _global_scalar(mesh, ufl.action(ufl.action(K, unit_y), unit_y))
    hydrostatic_gradient = params.Materials["Water"]["Density"] * abs(params.gravity)
    expected_residual = -hydrostatic_gradient * (
        params.Materials["Water"]["WaterHeight"] - vertical_displacement
    ) * params.Lx
    expected_tangent = hydrostatic_gradient * params.Lx

    assert residual_resultant == pytest.approx(expected_residual)
    assert tangent_resultant == pytest.approx(expected_tangent)


def _block_resultants(residual_form, tangent_form, unit_field):
    """Return ``unit . F`` and ``unit . K unit`` for one assembled field block."""
    residual = assemble_vector(fem.form(residual_form))
    residual.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    tangent = assemble_matrix(fem.form(tangent_form))
    tangent.assemble()
    product = tangent.createVecLeft()
    try:
        unit = unit_field.x.petsc_vec
        tangent.mult(unit, product)
        return residual.dot(unit), product.dot(unit)
    finally:
        for petsc_object in (residual, tangent, product):
            petsc_object.destroy()


def test_ocean_pressure_velocity_integrates_boundary_motion_with_backward_euler():
    params = _build_ocean_pressure_direct_params(
        water_height=1.5,
        boundary="bottom",
    )
    params.dt = 0.4
    mesh = Mesh(params)
    velocity = mesh.Add_Vector_Field("v", "Velocity", "CG", 1, params.dim)
    mesh.Add_Field("p", "Pressure", "CG", 1)
    mesh.Add_Vector_TestFunction("v", params.dim, params.Solution_Steps["v"])
    mesh.Add_TestFunction("p", params.Solution_Steps["v"])

    vertical_velocity = -0.5
    velocity.interpolate(
        lambda x: np.vstack(
            (
                np.zeros(x.shape[1], dtype=np.float64),
                np.full(x.shape[1], vertical_velocity, dtype=np.float64),
            )
        )
    )
    velocity.x.scatter_forward()
    mesh.finalize_test_functions()
    model = OceanPressure("OceanLoad", params, mesh)
    model.initialize_fields()

    K, F = model.assemble_KF(model.step)
    # The step is monolithic (v, p); test the velocity block of the forms.
    v_block = mesh.step_fields[model.step].index("v")
    F_v = ufl.extract_blocks(F)[v_block]
    K_vv = ufl.extract_blocks(K)[v_block][v_block]
    unit_y = fem.Function(mesh.fields["v"].function_space)
    unit_y.sub(1).interpolate(lambda x: np.ones(x.shape[1], dtype=np.float64))
    unit_y.x.scatter_forward()

    residual_resultant, tangent_resultant = _block_resultants(F_v, K_vv, unit_y)
    hydrostatic_gradient = params.Materials["Water"]["Density"] * abs(params.gravity)
    displacement_increment = params.dt * vertical_velocity
    expected_residual = -hydrostatic_gradient * (
        params.Materials["Water"]["WaterHeight"] - displacement_increment
    ) * params.Lx
    expected_tangent = hydrostatic_gradient * params.dt * params.Lx

    assert residual_resultant == pytest.approx(expected_residual)
    assert tangent_resultant == pytest.approx(expected_tangent)

    model.commit()
    mesh.Zero_Field("v")
    committed_resultant, _ = _block_resultants(F_v, K_vv, unit_y)
    assert committed_resultant == pytest.approx(expected_residual)


def test_ocean_pressure_velocity_branch_integrates_hydrostatic_cutoff():
    params = _build_ocean_pressure_direct_params(water_height=1.5)
    mesh = Mesh(params)
    mesh.Add_Vector_Field("v", "Velocity", "CG", 1, params.dim)
    mesh.Add_Vector_TestFunction("v", params.dim, params.Solution_Steps["v"])
    mesh.finalize_test_functions()
    model = OceanPressure("OceanLoad", params, mesh)

    K, F = model.assemble_KF(model.step)

    unit_x = fem.Function(mesh.Get_Field("v").function_space)
    unit_x.interpolate(
        lambda x: np.vstack(
            (
                np.ones(x.shape[1], dtype=np.float64),
                np.zeros(x.shape[1], dtype=np.float64),
            )
        )
    )
    unit_x.x.scatter_forward()
    resultant = _global_scalar(mesh, ufl.action(F, unit_x))
    expected = (
        0.5
        * params.Materials["Water"]["Density"]
        * abs(params.gravity)
        * params.Materials["Water"]["WaterHeight"] ** 2
    )

    assert model.StokesFormulation
    assert K is not None
    assert abs(resultant - expected) / expected < 1.0e-10
    assert model.assemble_KF(model.step + 1) is None


def test_ocean_pressure_requires_displacement_or_velocity_solution_step():
    params = _build_ocean_pressure_direct_params(solution_steps={"phasefield": 0})
    mesh = Mesh(params)

    with pytest.raises(ValueError, match="requires either 'u' or 'v'"):
        OceanPressure("OceanLoad", params, mesh)


def _build_two_domain_velocity_case(boundary):
    """Return ``(params, mesh, model)`` with the ice velocity on ``domain:left`` only."""
    params = TwoSquared_Sample_Parameters()
    params.gravity = -9.81
    params.dt = 0.5
    params.Materials = {"Water": {"Density": 1025.0, "WaterHeight": 2.0}}
    params.Solution_Steps = {"v": 0}
    params.Field_Orders = {"v": 1}
    params.Models = {"OceanLoad": {"boundary": boundary, "domain": "domain:left"}}
    mesh = Mesh(params)
    mesh.Add_Vector_Field("v", "Velocity", "CG", 1, params.dim, domain="domain:left")
    mesh.Add_Vector_TestFunction("v", params.dim, params.Solution_Steps["v"])
    mesh.finalize_test_functions()
    return params, mesh, OceanPressure("OceanLoad", params, mesh)


def test_ocean_pressure_acts_on_ice_restricted_to_a_subdomain():
    params, mesh, model = _build_two_domain_velocity_case("left")
    model.initialize_fields()
    mesh.Zero_Field("v")

    K, F = model.assemble_KF(model.step)
    unit_x = fem.Function(mesh.Get_Field("v").function_space)
    unit_x.x.array.reshape(-1, 2)[:, 0] = 1.0
    resultant = mesh.mesh.comm.allreduce(fem.assemble_scalar(mesh.Compile_Form(ufl.action(F, unit_x))), op=MPI.SUM)

    # Hydrostatic load on the fully submerged left face (outward normal -x), y in [0, 1].
    expected = -1025.0 * 9.81 * (2.0 - 0.5)
    assert resultant == pytest.approx(expected, rel=1.0e-10)
    assert mesh.Compile_Form(K) is not None

    displacement_name = model.displacement_field_name
    assert mesh.Get_Field_Support(displacement_name).domain_names == ("domain:left",)
    mesh.Get_Field("v").x.array.reshape(-1, 2)[:, 0] = 2.0
    model.commit()
    np.testing.assert_allclose(mesh.Get_Field(displacement_name).x.array.reshape(-1, 2)[:, 0], params.dt * 2.0)


def test_ocean_pressure_rejects_boundaries_outside_the_ice_subdomain():
    with pytest.raises(ValueError, match="Boundary 'right' does not bound domain 'domain:left'"):
        _build_two_domain_velocity_case("right")
