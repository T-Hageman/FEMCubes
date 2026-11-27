"""Analytic benchmark tests for solid-coupled boundary models."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration
from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.utils.maths_utils import nabla_s


class ConstantDamageField(BaseModel):
    """Test-only model registering a constant phase-field."""

    MODEL_TYPE = "Test:ConstantDamageField"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        order = self.params.Field_Orders.get("phasefield", self.params.Field_Orders["u"])
        self.mesh.Add_Field("phasefield", "Damage", "CG", order)
        self.value = float(self.params.Models[self.name]["value"])

    def initialize_fields(self) -> None:
        self.mesh.Constant_Field("phasefield", self.value)

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        return None


class SpatialComponentDirichlet(BaseModel):
    """Test-only spatial Dirichlet helper for analytic boundary data."""

    MODEL_TYPE = "Test:SpatialComponentDirichlet"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        model = self.params.Models[self.name]
        self.boundary = model["boundary"]
        dof = model["field"]
        if dof.endswith("_x"):
            self.field_name = dof[:-2]
            self.component = 0
        elif dof.endswith("_y"):
            self.field_name = dof[:-2]
            self.component = 1
        else:
            raise ValueError("SpatialComponentDirichlet requires a vector component field like 'u_y'.")
        self.profile = model["profile"]

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        return None

    def get_bcs_for_field(self, field_name: str):
        if field_name != self.field_name:
            return None

        function_space, sub_index = self.mesh.Get_Field_Space_For_BC(field_name, self.component)
        if sub_index is None:
            raise ValueError("SpatialComponentDirichlet expects a vector component subspace.")

        boundary_facets = self.mesh.get_boundary_facets(self.boundary)
        component_space = function_space.sub(sub_index)
        value_space, _ = component_space.collapse()
        current_field = self.mesh.Get_Field_For_Output(field_name).collapse_component(self.component)
        bc_value = fem.Function(value_space)

        if self.profile == "linear_uy":
            delta = float(self.params.Models[self.name]["delta"])
            height = self.params.Ly

            def _profile(x):
                return delta * x[1] / height

        else:
            raise ValueError(f"Unknown spatial profile '{self.profile}'.")

        bc_value.interpolate(_profile)
        bc_value.x.array[:] = current_field.x.array[:] - bc_value.x.array[:]
        bc_value.x.scatter_forward()

        dofs = fem.locate_dofs_topological(
            (component_space, value_space),
            self.mesh.mesh.topology.dim - 1,
            boundary_facets,
        )
        return [fem.dirichletbc(bc_value, dofs, component_space)]


def _global_scalar(mesh: Mesh, expr) -> float:
    return mesh.mesh.comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)


def _domain_average(mesh: Mesh, expr) -> float:
    dx = ufl.Measure("dx", domain=mesh.mesh)
    volume = _global_scalar(mesh, 1.0 * dx)
    return _global_scalar(mesh, expr * dx) / volume


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


def _elastic_material(youngs: float = 10.0e9, poisson: float = 0.3) -> dict:
    shear = youngs / (2.0 * (1.0 + poisson))
    return {
        "Density": 917.0,
        "Youngs": youngs,
        "Poisson": poisson,
        "LE_Damping": 0.0,
        "Shear": shear,
        "Bulk": youngs / (3.0 * (1.0 - 2.0 * poisson)),
        "Lame": shear * (2.0 * poisson) / (1.0 - 2.0 * poisson),
        "Inertia": False,
    }


def _build_linear_elastic_params(model_names, models, *, lx=2.0, ly=4.0, mesh_dx=0.5, phasefield_order=None):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}

        MeshType: str = "Rectangle"
        dx: float = mesh_dx
        dim: int = 2
        Lx: float = lx
        Ly: float = ly

        Materials: dict = {"Ice": _elastic_material()}
        gravity: float = 0.0
        Plot_Fields: list = []

        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 2}
        if phasefield_order is not None:
            Field_Orders["phasefield"] = phasefield_order
            Solution_Steps["phasefield"] = 0

        ModelNames = list(model_names)
        Models = dict(models)

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 25
        line_search: bool = False
        line_search_lims: list = [0.1, 1.0]
        rel_tol: float = 1.0e-3
        abs_tol: float = 1.0e-6

        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0
        dt: float = 1.0
        Newmark_Beta: float = 0.25
        Newmark_Gamma: float = 0.5

    return Params()


def _run_linear_elastic_case(params):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solver = TimeSolver(params, physics)
    solver.run()
    return mesh


def _sigma_yy(mesh: Mesh, lame: float, shear: float):
    u = mesh.Get_Field("u", step=0)
    strain = nabla_s(u)
    return lame * ufl.tr(strain) + 2.0 * shear * strain[1, 1]


def _sigma_xy(mesh: Mesh, shear: float):
    u = mesh.Get_Field("u", step=0)
    return 2.0 * shear * nabla_s(u)[0, 1]


@pytest.mark.full_model
def test_prescribed_rate_linear_elastic_uniform_strain_ramp():
    intercept, rate = 1.0e-4, 2.0e-4
    models = {
        "Solid": {"Type": "Solid:LinearElastic", "Material": "Ice", "DamageModel": None},
        "BC_Left": {"Type": "Boundary:PrescribedValue", "boundary": "left", "field": "u_x", "value": 0.0},
        "BC_Right": {"Type": "Boundary:PrescribedValue", "boundary": "right", "field": "u_x", "value": 0.0},
        "BC_Bottom": {"Type": "Boundary:PrescribedValue", "boundary": "bottom", "field": "u_y", "value": 0.0},
        "BC_Top": {"Type": "Boundary:PrescribedRate", "boundary": "top", "field": "u_y",
                   "value": intercept, "rate": rate},
    }
    params = _build_linear_elastic_params(list(models), models, mesh_dx=1.0)
    params.start_time, params.end_time, params.dt = 0.2, 0.8, 0.2
    mesh = _run_linear_elastic_case(params)
    displacement = _collapsed_component(mesh, "u", 1)
    target = intercept+rate*params.end_time
    exact = _interpolate_exact_like(displacement, lambda x: target*x[1]/params.Ly)
    assert _max_abs_function_error(displacement, exact) < 1e-11
    material = params.Materials["Ice"]
    expected_stress = (material["Lame"]+2*material["Shear"])*target/params.Ly
    assert _domain_average(mesh, _sigma_yy(mesh, material["Lame"], material["Shear"])) == pytest.approx(expected_stress)


@pytest.mark.full_model
def test_prescribed_value_linear_elastic_uniform_strain_patch():
    delta = 4.0e-4
    models = {
        "Solid": {
            "Type": "Solid:LinearElastic",
            "Material": "Ice",
            "DamageModel": None
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
        "BC_Bottom": {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "u_y",
            "value": 0.0,
        },
        "BC_Top": {
            "Type": "Boundary:PrescribedValue",
            "boundary": "top",
            "field": "u_y",
            "value": delta,
        },
        "BC_Left_Profile": {
            "Type": "Test:SpatialComponentDirichlet",
            "boundary": "left",
            "field": "u_y",
            "profile": "linear_uy",
            "delta": delta,
        },
        "BC_Right_Profile": {
            "Type": "Test:SpatialComponentDirichlet",
            "boundary": "right",
            "field": "u_y",
            "profile": "linear_uy",
            "delta": delta,
        },
    }
    params = _build_linear_elastic_params(
        ["Solid", "BC_Left", "BC_Right", "BC_Bottom", "BC_Top", "BC_Left_Profile", "BC_Right_Profile"],
        models,
    )
    params.abs_tol = 1.0e-5
    params.rel_tol = 1.0e-4
    mesh = _run_linear_elastic_case(params)

    lame = params.Materials["Ice"]["Lame"]
    shear = params.Materials["Ice"]["Shear"]
    effective_modulus = lame + 2.0 * shear
    target_sigma_yy = effective_modulus * delta / params.Ly
    target_pressure = -lame * delta / params.Ly

    ux_func = _collapsed_component(mesh, "u", 0)
    uy_func = _collapsed_component(mesh, "u", 1)
    p_func = mesh.Get_Field_For_Output("p").collapse_component(0)

    ux_exact = _interpolate_exact_like(ux_func, lambda x: np.zeros(x.shape[1], dtype=np.float64))
    uy_exact = _interpolate_exact_like(uy_func, lambda x: delta * x[1] / params.Ly)

    assert _max_abs_function_error(ux_func, ux_exact) < 1.0e-11
    assert _max_abs_function_error(uy_func, uy_exact) < 1.0e-11

    sigma_yy_error = _l2_error(mesh, _sigma_yy(mesh, lame, shear) - target_sigma_yy)
    sigma_xy_error = _l2_error(mesh, _sigma_xy(mesh, shear))
    pressure_error = _max_abs_function_error(
        p_func,
        _interpolate_exact_like(p_func, lambda x: np.full(x.shape[1], target_pressure, dtype=np.float64)),
    )

    assert sigma_yy_error < 1.0e-6
    assert sigma_xy_error < 1.0e-8
    assert pressure_error < 1.0e-7
    assert abs(_domain_average(mesh, _sigma_yy(mesh, lame, shear)) - target_sigma_yy) / abs(target_sigma_yy) < 1.0e-12


@pytest.mark.full_model
def test_displacement_control_linear_elastic_tension_bar():
    delta = 4.0e-4
    models = {
        "Solid": {
            "Type": "Solid:LinearElastic",
            "Material": "Ice",
            "DamageModel": None
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
        "BC_Bottom": {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "u_y",
            "value": 0.0,
        },
        "BC_Top": {
            "Type": "Boundary:DisplacementControl",
            "boundary": "top",
            "field": "u_y",
            "rate": delta,
            "Dummy": 1.0e14,
        },
    }
    params = _build_linear_elastic_params(
        ["Solid", "BC_Left", "BC_Right", "BC_Bottom", "BC_Top"],
        models,
    )
    mesh = _run_linear_elastic_case(params)

    lame = params.Materials["Ice"]["Lame"]
    shear = params.Materials["Ice"]["Shear"]
    effective_modulus = lame + 2.0 * shear
    expected_force = effective_modulus * delta / params.Ly * params.Lx

    ux_func = _collapsed_component(mesh, "u", 0)
    uy_func = _collapsed_component(mesh, "u", 1)
    ux_exact = _interpolate_exact_like(ux_func, lambda x: np.zeros(x.shape[1], dtype=np.float64))
    uy_exact = _interpolate_exact_like(uy_func, lambda x: delta * x[1] / params.Ly)

    measured_force = params.Global_Measures["BC_Top_F"]
    measured_force_max = params.Global_Measures["BC_Top_F_max"]

    assert _max_abs_function_error(ux_func, ux_exact) < 2.0e-8
    assert _max_abs_function_error(uy_func, uy_exact) < 2.0e-8
    assert abs(measured_force - expected_force) / abs(expected_force) < 2.0e-4
    assert abs(measured_force_max - expected_force) / abs(expected_force) < 2.0e-4


@pytest.mark.full_model
@pytest.mark.parametrize(
    "phasefield_value",
    [
        pytest.param(None, id="undamaged"),
        pytest.param(0.4, id="damaged"),
    ],
)
def test_external_force_linear_elastic_traction_bar_matches_analytic_solution(phasefield_value):
    traction = 1.5e6
    model_names = ["Solid", "BC_Left", "BC_Right", "BC_Bottom", "Load"]
    models = {
        "Solid": {
            "Type": "Solid:LinearElastic",
            "Material": "Ice",
            "DamageModel": None
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
        "BC_Bottom": {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "u_y",
            "value": 0.0,
        },
        "Load": {
            "Type": "Boundary:ExternalForce",
            "boundary": "top",
            "field": "u_y",
            "value": traction,
            "DamagedForces": phasefield_value is not None,
        },
    }
    phasefield_order = None
    if phasefield_value is not None:
        model_names.append("Damage")
        models["Damage"] = {
            "Type": "Test:ConstantDamageField",
            "value": phasefield_value,
        }
        phasefield_order = 1

    params = _build_linear_elastic_params(model_names, models, phasefield_order=phasefield_order)
    mesh = _run_linear_elastic_case(params)

    lame = params.Materials["Ice"]["Lame"]
    shear = params.Materials["Ice"]["Shear"]
    effective_modulus = lame + 2.0 * shear
    damage_factor = 1.0
    if phasefield_value is not None:
        damage_factor = (1.0 - 1.0e-6) * (1.0 - phasefield_value) ** 2 + 1.0e-6
    effective_traction = damage_factor * traction

    ux_func = _collapsed_component(mesh, "u", 0)
    uy_func = _collapsed_component(mesh, "u", 1)
    ux_exact = _interpolate_exact_like(ux_func, lambda x: np.zeros(x.shape[1], dtype=np.float64))
    uy_exact = _interpolate_exact_like(
        uy_func,
        lambda x: effective_traction * x[1] / effective_modulus,
    )

    sigma_yy_error = _l2_error(mesh, _sigma_yy(mesh, lame, shear) - effective_traction)
    sigma_xy_error = _l2_error(mesh, _sigma_xy(mesh, shear))

    assert _max_abs_function_error(ux_func, ux_exact) < 1.0e-8
    assert _max_abs_function_error(uy_func, uy_exact) < 1.0e-8
    assert sigma_yy_error < 5.0e-3
    assert sigma_xy_error < 1.0e-8
    assert abs(_domain_average(mesh, _sigma_yy(mesh, lame, shear)) - effective_traction) / abs(effective_traction) < 1.0e-8
