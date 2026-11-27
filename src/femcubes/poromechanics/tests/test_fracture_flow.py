"""Focused tests for ``femcubes.poromechanics.fracture_flow``."""

import pytest
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector

import femcubes.poromechanics  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - integration-test model registration
from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.poromechanics.fracture_flow import FractureFlow
from femcubes.poromechanics.pore_pressure import PorePressure
from femcubes.poromechanics.shared_poroelasticity import (
    single_saturation_from_pressure,
)


def _build_params(*, saturated=True, monolithic=False, gravity=0.0):
    class Params(ParamsTemplate):
        pass

    params = Params()
    params.MeshType = "Rectangle"
    params.dx = 0.5
    params.dim = 2
    params.Lx = 1.0
    params.Ly = 1.0
    params.gravity = gravity
    params.dt = 1.0
    params.t = 0.0
    params.Newmark_Beta = 0.25
    params.Newmark_Gamma = 0.5
    params.Quadrature_Degree = 2
    params.Global_Measures = {}
    params.Field_Orders = {"u": 1, "pw": 1, "phasefield": 1}
    params.Solution_Steps = {
        "u": 0,
        "pw": 1,
        "phasefield": 1 if monolithic else 2,
    }
    params.Materials = {
        "Ice": {
            "Density": 917.0,
            "Youngs": 1.0e10,
            "Poisson": 0.3,
            "Shear": 1.0e10 / (2.0 * (1.0 + 0.3)),
            "Lame": 1.0e10 * 0.3 / ((1.0 + 0.3) * (1.0 - 2.0 * 0.3)),
            "Bulk": 1.0e10,
            "Inertia": False,
            "Gc": 1.0,
            "ell": 0.1,
            "f_t": 1.0e6,
            "kappa": 1.0e-6,
            "PF_formulation": "AT2",
            "permeability0": 1.0e-12,
            "permeability_exp": 3.0,
            "fracture_permeability": 2.0e-9,
            "poros_crit": 0.8,
            "Saturation_p_ref": 2.0e3,
            "Saturation_m": 0.6,
        },
        "Water": {
            "Density": 1000.0,
            "Bulk": 2.2e9,
            "Viscosity": 1.0e-3,
        },
    }
    params.Models = {
        "porous": {
            "Type": "PoroMechanics:PorePressure",
            "Material": "Ice",
            "Fluid": "Water",
            "Initial_Porosity": 0.1,
            "Saturated": saturated,
        },
        "fracture_flow": {
            "Type": "PoroMechanics:FractureFlow",
            "Material": "Ice",
            "Fluid": "Water",
            "Saturated": saturated,
        },
    }
    params.ModelNames = ["porous", "fracture_flow"]
    return params


def _build_model(*, saturated=True, monolithic=False, gravity=0.0):
    params = _build_params(saturated=saturated, monolithic=monolithic, gravity=gravity)
    mesh = Mesh(params)
    PorePressure("porous", params, mesh)
    mesh.Add_Field(
        "phasefield",
        "Phasefield [-]",
        "CG",
        params.Field_Orders["phasefield"],
    )
    mesh.Add_TestFunction("phasefield", params.Solution_Steps["phasefield"])
    model = FractureFlow("fracture_flow", params, mesh)
    mesh.finalize_test_functions()
    model.initialize_fields()
    return params, mesh, model


def _residual_norm(model):
    tangent, residual = model.assemble_KF(model.Step_pw)
    fem.form(tangent)
    vector = assemble_vector(fem.form(residual))
    return vector.norm()


def test_fracture_flow_is_registered():
    assert (
        BaseModel.get_model_class("PoroMechanics:FractureFlow")
        is FractureFlow
    )


def test_fracture_flow_scales_with_phasefield():
    _, mesh, model = _build_model()
    pressure = mesh.Get_Field("pw")
    pressure.interpolate(lambda x: x[0])
    pressure.x.scatter_forward()

    mesh.Constant_Field("phasefield", 0.0)
    assert _residual_norm(model) == pytest.approx(0.0, abs=1.0e-20)

    mesh.Constant_Field("phasefield", 0.25)
    quarter_norm = _residual_norm(model)
    assert quarter_norm > 0.0

    mesh.Constant_Field("phasefield", 0.5)
    half_norm = _residual_norm(model)
    assert half_norm == pytest.approx(2.0 * quarter_norm)


def test_fracture_flow_monolithic_pressure_phasefield_tangent_builds():
    _, _, model = _build_model(monolithic=True)

    tangent, residual = model.Get_Weak_Form(0.0, 1.0, model.Step_pw)

    fem.form(tangent)
    fem.form(residual)


def test_fracture_flow_assembles_additively_through_physics():
    params = _build_params()
    params.Models.update(
        {
            "solid": {
                "Type": "Solid:LinearElastic",
                "Material": "Ice",
                "DamageModel": "PhaseField",
            },
            "damage": {
                "Type": "Solid:PhaseField",
                "Material": "Ice",
                "Irreversible_Method": "Hist",
                "CDF_field": "ElasticEnergy",
            },
        }
    )
    # Put the additive model first to ensure it does not rely on constructor
    # order; all shared fields are resolved only after model registration.
    params.ModelNames = ["fracture_flow", "porous", "solid", "damage"]
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    pressure = mesh.Get_Field("pw")
    pressure.interpolate(lambda x: x[0])
    pressure.x.scatter_forward()
    mesh.Constant_Field("phasefield", 0.5)

    tangent, residual = physics.Get_Kmat_Fvec(params.Solution_Steps["pw"])

    assert tangent is not None
    assert residual is not None
    assert residual.norm() > 0.0


def test_fracture_flow_includes_gravity_in_driving_force():
    _, zero_g_mesh, zero_g_model = _build_model(gravity=0.0)
    gravity_params, gravity_mesh, gravity_model = _build_model(gravity=-9.81)
    zero_g_mesh.Constant_Field("phasefield", 0.5)
    gravity_mesh.Constant_Field("phasefield", 0.5)

    zero_g_norm = _residual_norm(zero_g_model)
    gravity_norm = _residual_norm(gravity_model)

    assert zero_g_norm == pytest.approx(0.0, abs=1.0e-20)
    assert gravity_norm > 0.0

    hydrostatic_pressure = gravity_mesh.Get_Field("pw")
    rho_f = gravity_params.Materials["Water"]["Density"]
    gravity = gravity_params.gravity
    hydrostatic_pressure.interpolate(lambda x: rho_f * gravity * x[1])
    hydrostatic_pressure.x.scatter_forward()

    assert _residual_norm(gravity_model) == pytest.approx(0.0, abs=1.0e-14)


def test_shared_unsaturated_relation_matches_expected_saturation():
    p_ref = 2.0e3
    m = 0.6

    saturation, d_saturation, _ = single_saturation_from_pressure(
        -p_ref,
        p_ref,
        m,
    )

    assert saturation == pytest.approx(2.0**-m)
    assert d_saturation > 0.0
    assert single_saturation_from_pressure(1.0, p_ref, m) == (1.0, 0.0, 0.0)


def test_fracture_flow_rejects_missing_phasefield_step():
    params = _build_params()
    params.Solution_Steps.pop("phasefield")
    mesh = Mesh(params)

    with pytest.raises(ValueError, match="requires 'phasefield'"):
        FractureFlow("fracture_flow", params, mesh)


def _build_two_squared_model(fracture_domain):
    """Return a fracture-flow model whose coupled fields live on ``domain:left``."""
    from femcubes.mesh_generators.two_squared import Sample_Parameters

    params = _build_params()
    sample = Sample_Parameters()
    for attr in ("MeshType", "L", "dx", "dx_interface"):
        setattr(params, attr, getattr(sample, attr))
    params.Ly = params.L
    params.Models["porous"]["domain"] = "domain:left"
    params.Models["fracture_flow"]["domain"] = fracture_domain
    mesh = Mesh(params)
    PorePressure("porous", params, mesh)
    mesh.Add_Field("phasefield", "Phasefield [-]", "CG", 1, domain="domain:left")
    mesh.Add_TestFunction("phasefield", params.Solution_Steps["phasefield"])
    model = FractureFlow("fracture_flow", params, mesh)
    mesh.finalize_test_functions()
    mesh.Constant_Field("poros", 0.1)
    return params, mesh, model


def test_fracture_flow_on_a_subdomain_integrates_over_it():
    import ufl
    from mpi4py import MPI

    params, mesh, model = _build_two_squared_model("domain:left")
    mesh.Get_Field("pw").interpolate(lambda x: x[0])
    mesh.Constant_Field("phasefield", 0.5)

    _, residual = model.assemble_KF(model.Step_pw)
    probe = fem.Function(mesh.Get_Field("pw").function_space)
    probe.interpolate(lambda x: x[0])
    local = fem.assemble_scalar(mesh.Compile_Form(ufl.action(residual, probe)))

    # A unit pressure gradient drives k_f / mu * phasefield through the unit left square.
    mobility = params.Materials["Ice"]["fracture_permeability"] / params.Materials["Water"]["Viscosity"]
    assert MPI.COMM_WORLD.allreduce(local, op=MPI.SUM) == pytest.approx(mobility * 0.5 * params.L**2)


@pytest.mark.parametrize("fracture_domain", ["domain:right", "domain"])
def test_fracture_flow_rejects_domains_outside_the_pressure_support(fracture_domain):
    _, _, model = _build_two_squared_model(fracture_domain)

    with pytest.raises(ValueError, match="requires field 'pw' on that domain"):
        model.assemble_KF(model.Step_pw)
