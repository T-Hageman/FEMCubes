"""Model tests for the LoweringOceanPressure glaciology boundary condition."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

import femcubes.glaciology  # noqa: F401 - ensure model registration
from femcubes.core.base_model import BaseModel
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.glaciology.lowering_ocean_pressure import LoweringOceanPressure


def _params(**settings):
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
                "WaterHeight": 3.0,
            }
        }
        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 1}
        Models = {
            "Ocean": {"boundary": "bottom", **settings},
        }
        ModelNames = []
        Plot_Fields: list = []

    return Params()


def _model(**settings):
    params = _params(**settings)
    mesh = Mesh(params)
    mesh.Add_Vector_Field("u", "Displacement", "CG", 1, params.dim)
    mesh.Add_Vector_TestFunction("u", params.dim, params.Solution_Steps["u"])
    mesh.finalize_test_functions()
    return params, mesh, LoweringOceanPressure("Ocean", params, mesh)


def test_lowering_ocean_pressure_is_registered():
    assert BaseModel.get_model_class("glaciology:LoweringOceanPressure") is LoweringOceanPressure


def test_lowering_ocean_pressure_water_height_law():
    _, _, model = _model(LoweringTime=100.0, FinalWaterHeight=1.0, LoweringStart=10.0)
    assert model.water_height(-50.0) == pytest.approx(3.0)
    assert model.water_height(10.0) == pytest.approx(3.0)
    assert model.water_height(60.0) == pytest.approx(2.0)
    assert model.water_height(110.0) == pytest.approx(1.0)
    assert model.water_height(500.0) == pytest.approx(1.0)


@pytest.mark.parametrize("time, dt", [(-20.0, 5.0), (30.0, 20.0), (200.0, 10.0)])
def test_lowering_ocean_pressure_bottom_resultant_follows_the_level(time, dt):
    params, mesh, model = _model(LoweringTime=100.0, FinalWaterHeight=1.0, LoweringStart=10.0)
    time_constant = fem.Constant(mesh.mesh, PETSc.ScalarType(time))
    dt_constant = fem.Constant(mesh.mesh, PETSc.ScalarType(dt))
    _, F = model.Get_Weak_Form(time_constant, dt_constant, model.step)

    unit_y = fem.Function(mesh.Get_Field("u").function_space)
    unit_y.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.ones(x.shape[1]))))
    unit_y.x.scatter_forward()
    resultant = mesh.mesh.comm.allreduce(fem.assemble_scalar(fem.form(ufl.action(F, unit_y))), op=MPI.SUM)

    # Undeformed bottom (y = 0) with outward normal -y: resultant -rho g h_w(t + dt) Lx
    expected = -params.Materials["Water"]["Density"] * abs(params.gravity) * model.water_height(time + dt) * params.Lx
    assert resultant == pytest.approx(expected)


@pytest.mark.parametrize("settings", [{}, {"LoweringTime": 0.0}, {"LoweringTime": -1.0}])
def test_lowering_ocean_pressure_rejects_missing_or_non_positive_lowering_time(settings):
    with pytest.raises(ValueError, match="LoweringTime"):
        _model(**settings)
