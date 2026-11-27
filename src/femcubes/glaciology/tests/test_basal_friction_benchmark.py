"""Analytic benchmark for basal friction coupled to Stokes flow."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.glaciology  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration
from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.time_solver import TimeSolver

_BASAL_PENALTY_STIFFNESS = 1.0e12
_BASAL_NORMAL_VELOCITY = -1.0e-12


def _linear_friction_normal_load(params) -> float:
    bottom_vy = params.Models["BottomVy"]["value"]
    return _BASAL_PENALTY_STIFFNESS * max(-bottom_vy, 0.0)


def _linear_friction_ratio(params) -> float:
    material = params.Materials["Ice"]
    normal_load = _linear_friction_normal_load(params)
    return material["Linear_Friction_C"] * normal_load / material["Viscosity"]


class ProfileDirichlet(BaseModel):
    """Test-only profile Dirichlet BC used to realize exact slab flow."""

    MODEL_TYPE = "Test:ProfileDirichlet"

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
            raise ValueError("ProfileDirichlet requires a vector component field like 'v_x'.")
        self.target_top_velocity = float(model["target_top_velocity"])

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        return None

    def get_bcs_for_field(self, field_name: str):
        if field_name != self.field_name:
            return None

        space, sub_index = self.mesh.Get_Field_Space_For_BC(self.field_name, self.component)
        if sub_index is None:
            raise ValueError("ProfileDirichlet expects the target vector component to resolve to a subspace.")

        boundary_facets = self.mesh.get_boundary_facets(self.boundary)
        current_component = self.mesh.Get_Field_For_Output(self.field_name).collapse_component(self.component)
        component_space = space.sub(sub_index)
        value_space, _ = component_space.collapse()
        bc_value = fem.Function(value_space)

        ratio = _linear_friction_ratio(self.params)
        height = self.params.Ly
        top_velocity = self.target_top_velocity

        def _profile(x):
            return top_velocity * (1.0 + ratio * x[1]) / (1.0 + ratio * height)

        bc_value.interpolate(_profile)
        bc_value.x.array[:] = current_component.x.array[:] - bc_value.x.array[:]
        bc_value.x.scatter_forward()

        dofs = fem.locate_dofs_topological(
            (component_space, value_space),
            self.mesh.mesh.topology.dim - 1,
            boundary_facets,
        )
        return [fem.dirichletbc(bc_value, dofs, component_space)]


def _global_scalar(mesh: Mesh, expr) -> float:
    return mesh.mesh.comm.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)


def _boundary_average(mesh: Mesh, expr, boundary_name: str) -> float:
    ds = mesh.get_Boundary_Integrator()(mesh.get_boundary_tag(boundary_name))
    measure = _global_scalar(mesh, 1.0 * ds)
    return _global_scalar(mesh, expr * ds) / measure


def _max_abs_function_error(reference: fem.Function, exact: fem.Function) -> float:
    local = np.abs(reference.x.array - exact.x.array)
    local_max = 0.0 if local.size == 0 else float(np.max(local))
    return reference.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)


def _interpolate_exact_like(reference: fem.Function, exact_callable) -> fem.Function:
    exact = fem.Function(reference.function_space)
    exact.interpolate(exact_callable)
    exact.x.scatter_forward()
    return exact


def _build_basal_friction_params():
    class Params(ParamsTemplate):
        MeshType: str = "Rectangle"
        dx: float = 0.25
        dim: int = 2
        Lx: float = 1.0
        Ly: float = 1.0

        Materials: dict = {
            "Ice": {
                "Density": 917.0,
                "Bulk": 1.0e9,
                "Visc_Type": "Newtonian",
                "Viscosity": 4.0,
                "Linear_Friction_C": 10.0,
                "Inertia": False,
            }
        }

        gravity: float = 0.0
        Plot_Fields: list = []

        Solution_Steps = {"v": 0, "p": 0}
        Field_Orders = {"v": 2, "p": 1}

        ModelNames = [
            "Solid",
            "TopVx",
            "TopVy",
            "BottomVy",
            "LeftVy",
            "RightVy",
            "PressurePin",
            "LeftProfile",
            "RightProfile",
            "Friction",
        ]
        Models = {
            "Solid": {
                "Type": "Solid:Stokes",
                "Material": "Ice",
                "DamageModel": None,
            },
            "TopVx": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "top",
                "field": "v_x",
                "value": 1.0,
            },
            "TopVy": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "top",
                "field": "v_y",
                "value": _BASAL_NORMAL_VELOCITY,
            },
            "BottomVy": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "bottom",
                "field": "v_y",
                "value": _BASAL_NORMAL_VELOCITY,
            },
            "LeftVy": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "left",
                "field": "v_y",
                "value": _BASAL_NORMAL_VELOCITY,
            },
            "RightVy": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "right",
                "field": "v_y",
                "value": _BASAL_NORMAL_VELOCITY,
            },
            "PressurePin": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "left",
                "field": "p",
                "value": 0.0,
            },
            "LeftProfile": {
                "Type": "Test:ProfileDirichlet",
                "boundary": "left",
                "field": "v_x",
                "target_top_velocity": 1.0,
            },
            "RightProfile": {
                "Type": "Test:ProfileDirichlet",
                "boundary": "right",
                "field": "v_x",
                "target_top_velocity": 1.0,
            },
            "Friction": {
                "Type": "glaciology:BasalFriction",
                "boundary": "bottom",
                "FrictionModel": "Linear",
            },
        }

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 20
        line_search: bool = False
        line_search_lims: list = [0.1, 1.0]
        rel_tol: float = 1.0e-10
        abs_tol: float = 1.0e-12

        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0
        dt: float = 1.0

    return Params()


@pytest.mark.full_model
@pytest.mark.filterwarnings("error:Expr.ufl_domain:DeprecationWarning")
def test_basal_friction_stokes_matches_linear_robin_sliding_slab():
    params = _build_basal_friction_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solver = TimeSolver(params, physics)
    solver.run()

    viscosity = params.Materials["Ice"]["Viscosity"]
    friction = params.Materials["Ice"]["Linear_Friction_C"]
    top_velocity = params.Models["TopVx"]["value"]
    height = params.Ly
    normal_load = _linear_friction_normal_load(params)
    effective_friction = friction * normal_load
    ratio = effective_friction / viscosity
    expected_slip = top_velocity / (1.0 + ratio * height)
    expected_tau = effective_friction * expected_slip

    vx_func = mesh.Get_Field_For_Output("v").collapse_component(0)
    vy_func = mesh.Get_Field_For_Output("v").collapse_component(1)
    p_func = mesh.Get_Field_For_Output("p").collapse_component(0)

    vx_exact = _interpolate_exact_like(
        vx_func,
        lambda x: top_velocity * (1.0 + ratio * x[1]) / (1.0 + ratio * height),
    )
    vy_exact = _interpolate_exact_like(
        vy_func,
        lambda x: np.full(x.shape[1], _BASAL_NORMAL_VELOCITY, dtype=np.float64),
    )

    assert _max_abs_function_error(vx_func, vx_exact) < 1.0e-11
    assert _max_abs_function_error(vy_func, vy_exact) < 1.0e-11
    assert _max_abs_function_error(p_func, _interpolate_exact_like(p_func, lambda x: np.zeros(x.shape[1], dtype=np.float64))) < 1.0e-11

    v = mesh.Get_Field("v", step=0)
    normal = ufl.FacetNormal(mesh.mesh)
    normal_load_expr = _BASAL_PENALTY_STIFFNESS * ufl.dot(v, normal)
    positive_normal_load_expr = ufl.conditional(normal_load_expr > 0.0, normal_load_expr, 0.0)

    avg_slip = _boundary_average(mesh, v[0], "bottom")
    avg_normal_load = _boundary_average(mesh, positive_normal_load_expr, "bottom")
    avg_tau = _boundary_average(
        mesh,
        params.Materials["Ice"]["Linear_Friction_C"] * positive_normal_load_expr * v[0],
        "bottom",
    )

    assert abs(avg_slip - expected_slip) / abs(expected_slip) < 1.0e-11
    assert abs(avg_normal_load - normal_load) / abs(normal_load) < 1.0e-11
    assert abs(avg_tau - expected_tau) / abs(expected_tau) < 1.0e-11
