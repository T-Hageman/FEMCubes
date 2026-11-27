"""Shared fixtures for the physics test suite."""

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration
import pytest

from femcubes.core.params_template import ParamsTemplate


def _build_linear_elastic_params():
    class Params(ParamsTemplate):
        MeshType: str = "Rectangle"
        dx: float = 1.0
        dim: int = 2
        Lx: float = 2.0
        Ly: float = 3.0

        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 917.0,
            "Youngs": 10.0e9,
            "Poisson": 0.3,
            "LE_Damping": 1.0e6,
        }
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = (
            Materials["Ice"]["Shear"]
            * (2.0 * Materials["Ice"]["Poisson"])
            / (1.0 - 2.0 * Materials["Ice"]["Poisson"])
        )

        gravity: float = 0.0
        Plot_Fields: list = []

        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 2}

        Models = {}
        ModelNames = ["Solid", "BC_Left", "BC_Bottom", "BC_Right"]
        Models["Solid"] = {
            "Type": "Solid:LinearElastic",
            "Material": "Ice",
            "DamageModel": None,
        }
        Models["BC_Left"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "left",
            "field": "u_x",
            "value": 0.0,
        }
        Models["BC_Bottom"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "u_y",
            "value": 0.0,
        }
        Models["BC_Right"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "right",
            "field": "u_x",
            "value": 0.0,
        }

        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0
        dt: float = 0.25
        Newmark_Beta: float = 0.25
        Newmark_Gamma: float = 0.5

    return Params()


def _build_stokes_params():
    class Params(ParamsTemplate):
        MeshType: str = "Rectangle"
        dx: float = 1.0
        dim: int = 2
        Lx: float = 2.0
        Ly: float = 3.0

        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 917.0,
            "Bulk": 10.0e9,
            "Visc_Type": "Newtonian",
            "Viscosity": 1.0e13,
            "Glen_n": 3.0,
            "Glen_A": 1.0e-24,
            "visc_max": 1.0e20,
            "visc_min": 1.0e4,
        }

        gravity: float = 0.0
        Plot_Fields: list = []

        Solution_Steps = {"v": 0, "p": 0}
        Field_Orders = {"v": 2, "p": 1}

        Models = {}
        ModelNames = ["Solid", "BC_Left", "BC_Bottom", "BC_Right"]
        Models["Solid"] = {
            "Type": "Solid:Stokes",
            "Material": "Ice",
            "DamageModel": None,
        }
        Models["BC_Left"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "left",
            "field": "v_x",
            "value": 0.0,
        }
        Models["BC_Bottom"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "v_y",
            "value": 0.0,
        }
        Models["BC_Right"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "right",
            "field": "v_x",
            "value": 0.0,
        }

        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0
        dt: float = 0.25

    return Params()


@pytest.fixture
def linear_elastic_params():
    return _build_linear_elastic_params()


@pytest.fixture
def stokes_params():
    return _build_stokes_params()
