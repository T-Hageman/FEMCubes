"""Tests for the PrescribedFlux electrochemistry model."""

import pytest
import ufl

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.electrochemistry.prescribed_flux import PrescribedFlux


def test_prescribed_flux_uses_mesh_boundary_integrator(monkeypatch):
    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2

    params = TestParams()
    params.Solution_Steps = {"C": 3}
    params.Models = {
        "Flux": {
            "boundary": "left",
            "Species": ["Na+"],
            "flux": [1.2],
        }
    }

    mesh = Mesh(params)
    mesh.Add_Field("Na+", "Sodium", "CG", 1)
    mesh.Add_TestFunction("Na+", 3)
    mesh.finalize_test_functions()
    mesh.Constant_Field("Na+", 1.0)

    model = PrescribedFlux("Flux", params, mesh)
    original = mesh.get_Boundary_Integrator
    calls = []

    def _wrapped_integrator():
        calls.append(True)
        return original()

    monkeypatch.setattr(mesh, "get_Boundary_Integrator", _wrapped_integrator)

    model.assemble_KF(model.step)

    assert len(calls) == 1


def test_prescribed_flux_into_a_subdomain_field_reaches_the_linear_steady_state():
    # Lattice hydrogen only exists in the right square: a flux q enters
    # through its right edge and leaves through the interface, where C_L = 0.
    # The steady profile is linear, C_L(x) = q (x - L) / D, which integrates
    # over the square to q L^2 Ly / (2 D). Dimensionless values keep the
    # residuals well above round-off, so the steady state is resolved tightly.
    import femcubes.boundary  # noqa: F401 - ensure model registration
    import femcubes.electrochemistry  # noqa: F401 - ensure model registration
    from femcubes import Physics, TimeSolver

    flux, diffusivity = 1.0, 1.0

    class Params(ParamsTemplate):
        MeshType = "TwoSquared"
        dim = 2
        L = 1.0
        Ly = 0.2
        dx = 0.1
        Global_Measures: dict = {}
        Materials: dict = {"Steel": {"D_L": diffusivity}}
        Solution_Steps = {"C": 0, "CL": 0}
        Field_Orders = {"CL": 2}
        ModelNames: list = ["Hydrogen", "Influx", "Sink"]
        Models: dict = {
            "Hydrogen": {"Type": "ElectroChemistry:HydrogenDiffusion", "Material": "Steel", "domain": "domain:right"},
            "Influx": {"Type": "ElectroChemistry:PrescribedFlux", "boundary": "right", "Species": ["CL"], "flux": [flux], "domain": "domain:right"},
            "Sink": {"Type": "Boundary:PrescribedValue", "boundary": "interface", "field": "CL", "value": 0.0, "domain": "domain:right"},
        }
        # Steps much longer than the diffusion time L^2/D: each backward-Euler
        # step reduces the transient by a factor ~250.
        start_time = 0.0
        end_time = 2000.0
        dt = 100.0
        TimeStepControl = {"Type": "None"}
        rel_tol = 1e-12
        abs_tol = 1e-12
        Plot_Fields: list = []
        Plot_Measures: list = []

    params = Params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    expected = flux * params.L**2 * params.Ly / (2.0 * diffusivity)
    assert mesh.Get_Field_Support("CL").domain_names == ("domain:right",)
    assert abs(params.Global_Measures["Hydrogen_Hydrogen"] - expected) <= 1e-9 * expected


def _two_squared_flux_mesh(models):
    """Return params and a two-squares mesh with ``CL`` only on ``domain:right``."""
    from femcubes.mesh_generators.two_squared import Sample_Parameters

    params = Sample_Parameters()
    params.Solution_Steps = {"C": 0}
    params.Models = models
    mesh = Mesh(params)
    mesh.Add_Field("CL", "Lattice hydrogen", "CG", 1, domain="domain:right")
    mesh.Add_TestFunction("CL", 0)
    mesh.finalize_test_functions()
    return params, mesh


def test_prescribed_flux_on_a_subdomain_field_integrates_over_its_boundary():
    import numpy as np
    from dolfinx import fem

    params, mesh = _two_squared_flux_mesh(
        {"Influx": {"boundary": "right", "Species": ["CL"], "flux": [1.5], "domain": "domain:right"}}
    )
    model = PrescribedFlux("Influx", params, mesh)
    _, residual = model.assemble_KF(model.step)

    probe = fem.Function(mesh.Get_Field("CL").function_space)
    probe.x.array[:] = 1.0
    local = fem.assemble_scalar(mesh.Compile_Form(ufl.action(residual, probe)))
    assert np.isclose(mesh.mesh.comm.allreduce(local), -1.5 * params.L)


def test_prescribed_flux_rejects_boundaries_outside_the_field_support_or_domain():
    params, mesh = _two_squared_flux_mesh(
        {
            "Outside": {"boundary": "top", "Species": ["CL"], "flux": [1.0]},
            "Wrong": {"boundary": "right", "Species": ["CL"], "flux": [1.0], "domain": "domain:left"},
        }
    )

    outside = PrescribedFlux("Outside", params, mesh)
    with pytest.raises(ValueError, match="cannot act on field 'CL'"):
        outside.assemble_KF(outside.step)
    with pytest.raises(ValueError, match="does not bound domain 'domain:left'"):
        PrescribedFlux("Wrong", params, mesh)
