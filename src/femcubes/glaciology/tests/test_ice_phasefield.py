"""Tests for the ice-specific phase-field fracture model."""

import numpy as np
import pytest
import ufl
from dolfinx import fem

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.glaciology.ice_phasefield import IcePhaseField
from femcubes.solids.phase_field import PhaseField


def _build_params(pf_formulation="AT2"):
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        dx = 0.5
        dim = 2
        Lx = 1.0
        Ly = 1.0

        Materials = {
            "Ice": {
                "ell": 0.2,
                "PF_formulation": pf_formulation,
                "kappa": 1.0e-6,
            }
        }

        Solution_Steps = {"phasefield": 0}
        Field_Orders = {"phasefield": 2, "Porosity": 0, "Temperature": 1}

        Models = {
            "pf": {
                "Type": "Glaciology:PhaseField",
                "Material": "Ice",
                "CDF_field": "ElasticEnergy",
                # History irreversibility needs AT2/HO-AT2; AT1 needs a bound on the phase field
                "Irreversible_Method": "Penalty" if pf_formulation in ("AT1", "HO-AT1") else "Hist",
            }
        }
        ModelNames = ["pf"]

        t = 0.0
        dt = 1.0

    return Params()


@pytest.mark.parametrize(
    ("pf_formulation", "gc_factor"),
    [("AT1", 8.0 / 3.0), ("AT2", None), ("HO-AT1", 8.0 / 3.0), ("HO-AT2", 256.0 / 27.0)],
)
def test_ice_phasefield_uses_local_ice_fracture_toughness_and_base_behavior(
    pf_formulation, gc_factor
):
    params = _build_params(pf_formulation)
    mesh = Mesh(params)
    mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", ())
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    model = physics.models["pf"]
    assert isinstance(model, IcePhaseField)
    assert isinstance(model, PhaseField)
    assert "Gc" not in params.Materials["Ice"]
    assert mesh.field_order["poros"] == params.Field_Orders["Porosity"]
    assert mesh.field_order["T"] == params.Field_Orders["Temperature"]

    porosity = 0.2
    temperature = -5.0
    mesh.Constant_Field("poros", porosity)
    mesh.Constant_Field("T", temperature)
    mesh.Add_Field("Gc_check", "Fracture Toughness [J/m²]", "DG", 0)
    mesh.Set_From_Expression("Gc_check", model.Gc)

    tensile_strength = 0.2e6  # independent of the temperature
    youngs_modulus = 9.5e9 * (1.0 - porosity) ** 4
    expected_gc = (10.0 * (1.0 - porosity) if pf_formulation == "AT2" else
                   gc_factor * tensile_strength**2 / youngs_modulus * params.Materials["Ice"]["ell"])
    np.testing.assert_allclose(mesh.Get_Field("Gc_check").x.array, expected_gc)

    K, F = model.assemble_KF(model.My_Step)
    assert K is not None
    assert F is not None


def test_ice_phasefield_penalty_assembles_with_spatial_fracture_toughness():
    params = _build_params("AT2")
    params.Models["pf"].update(
        {
            "Irreversible_Method": "Penalty",
            "Irreversibility_Tolerance": 0.02,
            "Irreversibility_Smoothing": 0.001,
        }
    )
    mesh = Mesh(params)
    mesh.Add_Quadrature_Field(
        "ElasticEnergy",
        "Elastic Energy Density [J/m³]",
        (),
    )
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["pf"]

    mesh.Constant_Field("phasefieldOld", 0.4)
    mesh.Constant_Field("phasefield", 0.2)
    K, F = model.assemble_KF(model.My_Step)

    assert fem.form(K) is not None
    assert fem.form(F) is not None
    assert "phasefield_Hist" not in mesh.fields
    assert "LMultiplier_PF" not in mesh.fields


def test_ice_phasefield_restricted_to_a_subdomain():
    params = _build_params("AT2")
    params.MeshType, params.L = "TwoSquared", 1.0
    params.Models["pf"]["domain"] = "domain:left"
    mesh = Mesh(params)
    mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", (), domain="domain:left")
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["pf"]

    for field_name in ("poros", "T", "phasefield", "phasefield_Hist"):
        assert mesh.Get_Field_Support(field_name).domain_names == ("domain:left",)

    mesh.Constant_Field("poros", 0.2)
    mesh.Constant_Field("T", -5.0)
    mesh.Constant_Field("phasefield", 0.1)
    K, F = model.assemble_KF(model.My_Step)

    assert mesh.Compile_Form(K) is not None
    residual = mesh.mesh.comm.allreduce(fem.assemble_scalar(mesh.Compile_Form(ufl.action(F, mesh.Get_Field("phasefield")))))
    assert np.isfinite(residual) and residual > 0.0


@pytest.mark.parametrize("correction", [{"Coefficient": 0.5, "Element_Size": 0.25},
                                        {"Coefficient": 0.5, "Source_Field": "ElasticEnergy",
                                         "Source_Threshold": -1.0}])
def test_gc_mesh_correction_uses_the_spatially_varying_toughness(correction):
    def corrected_toughness(parent):
        params = _build_params("AT2")
        params.Models["pf"]["Gc_Mesh_Correction"] = correction
        if parent:
            params.Models["pf"]["Type"] = "Solid:PhaseField"
            params.Materials["Ice"]["Gc"] = 1.0
        mesh = Mesh(params)
        mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", ())
        physics = Physics(mesh, params)
        physics.Initialize_Fields()
        model = physics.models["pf"]
        if not parent:
            x = ufl.SpatialCoordinate(mesh.mesh)
            mesh.Set_From_Expression("poros", 0.1 + 0.2 * x[0])
            mesh.Constant_Field("T", -5.0)
        K, F = model.assemble_KF(model.My_Step)
        assert mesh.Compile_Form(K) is not None and mesh.Compile_Form(F) is not None
        return mesh

    mesh = corrected_toughness(parent=False)
    factor = corrected_toughness(parent=True).Get_Field("Gc_eff").x.array      # correction factor of Gc = 1
    assert np.all(factor < 1.0)
    gc_eff = mesh.Get_Field("Gc_eff").x.array
    assert np.ptp(gc_eff) > 0
    np.testing.assert_allclose(gc_eff, 10.0 * (1.0 - mesh.Get_Field("poros").x.array) * factor, rtol=1e-12)


def test_post_remesh_resets_the_mesh_correction_and_rebinds_the_toughness():
    from femcubes.core.physics.tests.test_model_remesh import _contains_ufl_terminal, _install_one_refinement

    params = _build_params("AT2")
    params.Models["pf"]["Gc_Mesh_Correction"] = {"Coefficient": 0.5, "Element_Size": 0.25}
    mesh = Mesh(params)
    mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", ())
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["pf"]
    mesh.Constant_Field("poros", 0.2)
    mesh.Constant_Field("T", -5.0)
    physics.Get_Kmat_Fvec(model.My_Step)
    assert model._gc_correction_key is not None

    assert _install_one_refinement(mesh).installed
    physics.Post_Remesh()
    assert model._gc_correction_key is None
    assert _contains_ufl_terminal(model.Gc, mesh.Get_Field("poros"))
    K, F = physics.Get_Kmat_Fvec(model.My_Step)
    assert np.isfinite(K.norm()) and np.isfinite(F.norm())
    np.testing.assert_allclose(mesh.Get_Field("Gc_eff").x.array, 10.0 * 0.8 / (1.0 + 0.5 * 0.25 / 0.2), rtol=1e-12)


def test_invalid_porosity_raises_a_clear_error():
    params = _build_params("AT2")
    mesh = Mesh(params)
    mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", ())
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["pf"]
    mesh.Constant_Field("poros", 1.2)
    with pytest.raises(ValueError, match="porosity 'poros' must be finite with 0 <= p < 1"):
        model.assemble_KF(model.My_Step)
