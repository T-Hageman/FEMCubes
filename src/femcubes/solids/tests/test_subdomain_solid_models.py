"""Solid models restricted to a subdomain through their ``"domain"`` setting."""

import numpy as np
import pytest

from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.two_squared import Sample_Parameters
from femcubes.core.physics.physics import Physics
from femcubes.solids.tests.test_split_viscoelastic import _build_params as _visco_params

MODEL_TYPES = (
    "Solid:ViscoElastic",
    "Solid:SplitViscoElastic",
    "Solid:SplitLinearElastic",
    "Solid:Cohesive_LinearElastic",
    "Solid:Cohesive_ViscoElastic",
    "Solid:Cohesive_ViscoElastic_Kernel",
    "Solid:Stokes",
)


def _build_params(model_type, solid_domains):
    """Return two-square parameters with one solid model per entry of ``solid_domains``.

    ``solid_domains`` maps model names to ``(domain, material)``. The phase
    field always lives on the full mesh.
    """
    params = _visco_params()
    sample = Sample_Parameters()
    params.MeshType, params.L, params.dx, params.dx_interface = sample.MeshType, sample.L, 0.5, 0.25
    ice = params.Materials["Ice"]
    ice.update({"f_s": 3.0, "Inertia": False})
    params.Materials["Stiff"] = {**ice, "Lame": 2.0 * ice["Lame"], "Shear": 2.0 * ice["Shear"], "Viscosity": 2.0 * ice["Viscosity"], "Bulk": 2.0 * ice["Bulk"]}
    if model_type == "Solid:Stokes":
        params.Solution_Steps = {"v": 0, "p": 0, "phasefield": 1}
        params.Field_Orders = {"v": 2, "phasefield": 1}
        for material in params.Materials.values():
            material["CrackDrivingForce"] = "Stress-Based"
    pf = params.Models["pf"]
    params.Models = {"pf": pf}
    for name, (domain, material) in solid_domains.items():
        settings = {"Type": model_type, "Material": material, "DamageModel": "PhaseField"}
        if model_type == "Solid:SplitViscoElastic":
            settings["CreepCoupling"] = "EffectiveStress"
        if domain is not None:
            settings["domain"] = domain
        params.Models[name] = settings
    params.ModelNames = [*solid_domains, "pf"]
    return params


def _assemble(params):
    """Build the physics, set a smooth primary field and assemble the mechanics step."""
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    mesh.Constant_Field("phasefield", 0.3)
    primary = "v" if "v" in params.Solution_Steps else "u"
    mesh.Get_Field(primary).interpolate(lambda x: np.vstack((1.0e-3 * x[0] + 2.0e-3 * x[1], -1.0e-3 * x[0] * x[1])))
    mesh.Get_Field(primary).x.scatter_forward()
    K, F = physics.Get_Kmat_Fvec(0)
    return physics, primary, K, F


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_on_a_subdomain_only_creates_its_fields_there(model_type):
    params = _build_params(model_type, {"Solid": ("domain:left", "Ice")})
    physics, primary, K, F = _assemble(params)
    mesh = physics.mesh

    for field_name in (primary, "sigma"):
        support = mesh.Get_Field_Support(field_name)
        assert support.domain_names == ("domain:left",)
        assert not support.is_full
    step_dofs = sum(
        mesh.fields[name].function_space.dofmap.index_map.size_global * mesh.fields[name].function_space.dofmap.index_map_bs
        for name in mesh.step_fields[0]
    )
    assert F.getSize() == step_dofs
    assert np.isfinite(K.norm()) and K.norm() > 0.0
    assert np.isfinite(F.norm()) and F.norm() > 0.0


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_models_on_neighbouring_subdomains_each_update_only_their_own_points(model_type):
    split = _build_params(model_type, {"Left": ("domain:left", "Ice"), "Right": ("domain:right", "Stiff")})
    physics, primary, _K, F = _assemble(split)
    mesh = physics.mesh
    references = {material: _assemble(_build_params(model_type, {"Solid": (None, material)}))[0].mesh for material in ("Ice", "Stiff")}

    assert mesh.Get_Field_Support(primary).is_full
    assert np.isfinite(F.norm())
    for domain, material in (("domain:left", "Ice"), ("domain:right", "Stiff")):
        dofs = mesh.Get_Field_Dofs("sigma", domain)
        expected = references[material].Get_Field("sigma").x.array[dofs]
        assert np.linalg.norm(expected) > 0.0
        np.testing.assert_allclose(mesh.Get_Field("sigma").x.array[dofs], expected, rtol=1.0e-10, atol=1.0e-14 * np.abs(expected).max())


def test_subdomain_viscoelastic_model_commits_and_resets_only_on_its_domain():
    params = _build_params("Solid:ViscoElastic", {"Left": ("domain:left", "Ice"), "Right": ("domain:right", "Stiff")})
    physics, _primary, _K, _F = _assemble(params)
    mesh = physics.mesh
    right_dofs = mesh.Get_Field_Dofs("Plastic_Strain", "domain:right")
    left_dofs = mesh.Get_Field_Dofs("Plastic_Strain", "domain:left")
    before = mesh.Get_Field("Plastic_Strain_Old").x.array.copy()

    physics.models["Left"].commit()

    after = mesh.Get_Field("Plastic_Strain_Old").x.array
    np.testing.assert_array_equal(after[right_dofs], before[right_dofs])
    np.testing.assert_array_equal(after[left_dofs], mesh.Get_Field("Plastic_Strain").x.array[left_dofs])
