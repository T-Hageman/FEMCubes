"""Full-model benchmarks for ``PorePressure``."""

import numpy as np
import pytest
import ufl

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.poromechanics.pore_pressure import PorePressure

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.poromechanics  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration


def _elastic_material(
    *,
    youngs: float = 10.0e9,
    poisson: float = 0.3,
    density: float = 917.0,
    permeability0: float = 1.0e-6,
    biot=None,
):
    shear = youngs / (2.0 * (1.0 + poisson))
    material = {
        "Density": density,
        "Youngs": youngs,
        "Poisson": poisson,
        "Shear": shear,
        "Bulk": youngs / (3.0 * (1.0 - 2.0 * poisson)),
        "Lame": shear * (2.0 * poisson) / (1.0 - 2.0 * poisson),
        "LE_Damping": 0.0,
        "Inertia": False,
        "Biot_n": 1.0,
        "poros_crit": 0.8,
        "permeability0": permeability0,
        "permeability_exp": 3.0,
        "Saturation_p_ref": 1.0e3,
        "Saturation_m": 0.6,
    }
    if biot is not None:
        material["Biot"] = biot
    return material


def _build_params(*, saturated: bool, monolithic: bool, initial_poros: float = 0.25, biot=None):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}

        MeshType: str = "Rectangle"
        dx: float = 0.5
        dim: int = 2
        Lx: float = 2.0
        Ly: float = 2.0
        Quadrature_Degree: int = 2

        Materials: dict = {
            "Ice": _elastic_material(biot=biot),
            "Water": {
                "Density": 1000.0,
                "Bulk": 2.2e9,
                "Viscosity": 1.0e-3,
            },
        }

        # PorePressure gravity is set explicitly after model construction so
        # this test isolates pore-pressure flow from solid self-weight.
        gravity: float = 0.0
        Plot_Fields: list = []

        Solution_Steps = {"u": 0, "pw": 0 if monolithic else 1}
        Field_Orders = {"u": 2, "pw": 1}

        Models = {
            "Solid": {
                "Type": "Solid:LinearElastic",
                "Material": "Ice",
                "DamageModel": None,
            },
            "porous": {
                "Type": "PoroMechanics:PorePressure",
                "Material": "Ice",
                "Fluid": "Water",
                "Initial_Porosity": initial_poros,
                "Saturated": saturated,
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
            "BC_TopPressure": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "top",
                "field": "pw",
                "value": 0.0,
            },
        }
        ModelNames = ["Solid", "porous", "BC_Left", "BC_Right", "BC_Bottom", "BC_TopPressure"]

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 2
        max_it: int = 50
        line_search: bool = False
        line_search_lims: list = [0.01, 1.0]
        rel_tol: float = 1.0e-8
        abs_tol: float = 1.0e-9

        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0e6
        dt: float = 1.0e6
        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0

    return Params()


def _set_pore_pressure_gravity(physics, gravity: float):
    model = physics.models["porous"]
    model.gravity = gravity
    model.gVec = ufl.as_vector([0.0, gravity])


def _run_model(params, *, pore_pressure_gravity: float = 0.0, initialize=None):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    _set_pore_pressure_gravity(physics, pore_pressure_gravity)
    physics.Initialize_Fields()
    if initialize is not None:
        initialize(mesh, physics)
    solver = TimeSolver(params, physics)
    solver.run()
    return mesh, physics


def _field_array(mesh, field_name: str):
    return mesh.Get_Field_For_Output(field_name).field.x.array


def _vector_field_array(mesh, field_name: str):
    field = mesh.Get_Field_For_Output(field_name).field
    return field.x.array.reshape((-1, 2))


@pytest.mark.full_model
@pytest.mark.parametrize(
    ("use_lumped", "biot"),
    [
        pytest.param(True, None, id="lumped-variable-biot"),
        pytest.param(False, None, id="consistent-variable-biot"),
        pytest.param(True, 0.35, id="lumped-constant-biot"),
    ],
)
def test_pore_pressure_saturated_hydrostatic_column(use_lumped, biot, monkeypatch):
    monkeypatch.setattr(PorePressure, "USE_LUMPED", use_lumped)
    gravity = -9.81
    params = _build_params(saturated=True, monolithic=True, biot=biot)

    mesh, physics = _run_model(params, pore_pressure_gravity=gravity)

    pw = _field_array(mesh, "pw")
    qw = _vector_field_array(mesh, "qw")
    expected_pmax = (
        params.Materials["Water"]["Density"]
        * abs(gravity)
        * params.Ly
    )

    np.testing.assert_allclose(mesh.Get_Field("pwOld").x.array, pw)
    assert abs(np.max(pw) - expected_pmax) / expected_pmax < 1.0e-3
    assert abs(np.min(pw)) < 1.0e-8 * expected_pmax
    assert np.max(np.linalg.norm(qw, axis=1)) < 1.0e-8
    assert physics.models["porous"].assemble_KF(99) is None


@pytest.mark.full_model
@pytest.mark.parametrize("use_lumped", [True, False], ids=["lumped", "consistent"])
def test_pore_pressure_unsaturated_infiltration_with_lens_and_reset(use_lumped, monkeypatch):
    monkeypatch.setattr(PorePressure, "USE_LUMPED", use_lumped)
    p_ref = 1.0e3
    inlet_pressure = 100.0
    params = _build_params(saturated=False, monolithic=False, initial_poros=0.5)
    params.Models["BC_TopPressure"]["value"] = inlet_pressure
    params.Models["BC_BottomPressure"] = {
        "Type": "Boundary:PrescribedValue",
        "boundary": "bottom",
        "field": "pw",
        "value": -p_ref,
    }
    params.ModelNames.append("BC_BottomPressure")
    params.end_time = 10.0
    params.dt = 10.0

    def _initialize(mesh, _physics):
        mesh.Constant_Field("pw", -p_ref)
        mesh.Constant_Field("pwOld", -p_ref)
        x = ufl.SpatialCoordinate(mesh.mesh)
        in_lens = ufl.And(
            ufl.And(x[0] >= 0.75, x[0] <= 1.25),
            ufl.And(x[1] >= 0.75, x[1] <= 1.25),
        )
        mesh.Set_From_Expression("poros", ufl.conditional(in_lens, 0.0, 0.5))

    mesh, physics = _run_model(params, initialize=_initialize)

    pw = _field_array(mesh, "pw")
    sw = mesh.Get_Field("Sw").x.array
    qw = _vector_field_array(mesh, "qw")
    poros = mesh.Get_Field("poros").x.array

    assert np.min(pw) >= -1.001 * p_ref
    assert np.max(pw) <= 1.001 * inlet_pressure
    assert np.all(np.isfinite(sw))
    assert np.min(sw) >= -1.0e-12
    assert np.max(sw) <= 1.0 + 1.0e-12
    assert np.max(sw) - np.min(sw) > 0.1
    assert np.min(poros) == pytest.approx(0.0)
    assert np.mean(qw[:, 1]) < 0.0

    committed_pw = mesh.Get_Field("pwOld").x.array.copy()
    mesh.Zero_Field("pw")
    physics.Reset_Step()
    np.testing.assert_allclose(mesh.Get_Field("pw").x.array, committed_pw)
