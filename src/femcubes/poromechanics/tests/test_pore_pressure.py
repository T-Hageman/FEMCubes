"""Focused tests for ``femcubes.poromechanics.pore_pressure``."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
from mpi4py import MPI

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.glaciology.ice_material import IceMaterial
from femcubes.poromechanics.pore_pressure import PorePressure


def _build_params(
    *,
    dim: int = 2,
    monolithic: bool = False,
    gravity: float = -9.81,
    biot=None,
    solid_density: float = 917.0,
    fluid_density: float = 1000.0,
):
    class Params(ParamsTemplate):
        pass

    params = Params()
    params.MeshType = "Rectangle"
    params.dx = 1.0
    params.dim = dim
    params.Lx = 2.0
    params.Ly = 3.0
    if dim == 3:
        params.Lz = 1.0
    params.gravity = gravity
    params.dt = 1.0
    params.t = 0.0
    params.Newmark_Beta = 0.25
    params.Newmark_Gamma = 0.5
    params.Quadrature_Degree = 2
    params.Global_Measures = {}

    params.Field_Orders = {"u": 1, "pw": 1}
    if monolithic:
        params.Solution_Steps = {"u": 0, "pw": 0}
    else:
        params.Solution_Steps = {"u": 0, "pw": 1}

    params.Materials = {
        "Ice": {
            "Density": solid_density,
            "Youngs": 10.0e9,
            "Poisson": 0.3,
            "Bulk": 10.0e9 / (3.0 * (1.0 - 2.0 * 0.3)),
            "Biot_n": 2,
            "poros_crit": 0.8,
            "permeability0": 1.0e-12,
            "permeability_exp": 3.0,
        },
        "Water": {
            "Density": fluid_density,
            "Bulk": 2.2e9,
            "Viscosity": 1.0e-3,
        },
    }
    if biot is not None:
        params.Materials["Ice"]["Biot"] = biot

    params.Models = {
        "porous": {
            "Type": "PoroMechanics:PorePressure",
            "Material": "Ice",
            "Fluid": "Water",
            "Initial_Porosity": 0.1,
            "Saturated": True,
        }
    }
    params.ModelNames = ["porous"]
    return params


def _build_model(
    *,
    dim: int = 2,
    monolithic: bool = False,
    gravity: float = -9.81,
    biot=None,
    solid_density: float = 917.0,
    fluid_density: float = 1000.0,
):
    params = _build_params(
        dim=dim,
        monolithic=monolithic,
        gravity=gravity,
        biot=biot,
        solid_density=solid_density,
        fluid_density=fluid_density,
    )
    mesh = Mesh(params)
    model = PorePressure("porous", params, mesh)
    mesh.finalize_test_functions()
    model.initialize_fields()
    return params, mesh, model


def test_pore_pressure_constant_biot_path_builds_forms():
    _, _, model = _build_model(biot=0.35)

    assert model.Biot is not None
    np.testing.assert_allclose(np.asarray(model.Biot.value), 0.35)

    K, F = model.assemble_KF(model.Step_pw)

    assert K is not None
    assert F is not None
    fem.form(K)
    fem.form(F)


def test_pore_pressure_uses_ice_bulk_modulus_fallback():
    params = _build_params()
    params.Materials["Ice"].pop("Bulk")
    mesh = Mesh(params)

    model = PorePressure("porous", params, mesh)

    expected = IceMaterial().Get_Bulk_Modulus(porosity=0.0, temperature=0.0)
    assert model.bulk_s == pytest.approx(expected)


def test_pore_pressure_requires_bulk_modulus_for_non_ice_material():
    params = _build_params()
    params.Materials["Rock"] = params.Materials.pop("Ice")
    params.Materials["Rock"].pop("Bulk")
    params.Models["porous"]["Material"] = "Rock"
    mesh = Mesh(params)

    with pytest.raises(ValueError, match="Bulk modulus must be specified"):
        PorePressure("porous", params, mesh)


def test_pore_pressure_assemble_kf_returns_none_for_inactive_step():
    _, _, model = _build_model()

    assert model.assemble_KF(99) is None


@pytest.mark.parametrize("dim", [2, 3])
def test_pore_pressure_step_u_includes_gravity_density_correction(dim):
    params, mesh, gravity_model = _build_model(dim=dim, gravity=-9.81)
    _, _, zero_g_model = _build_model(dim=dim, gravity=0.0)

    _, F_gravity = gravity_model.assemble_KF(gravity_model.Step_u)
    _, F_zero = zero_g_model.assemble_KF(zero_g_model.Step_u)

    vec_gravity = assemble_vector(fem.form(F_gravity))
    vec_zero = assemble_vector(fem.form(F_zero))

    assert vec_gravity.norm() > 0.0
    assert vec_zero.norm() == 0.0
    for axis in range(dim):
        space, component_index = mesh.Get_Field_Space_For_BC("u", axis)
        probe = fem.Function(space)
        probe.sub(component_index).interpolate(lambda x: np.ones(x.shape[1]))
        probe.x.scatter_forward()
        local = fem.assemble_scalar(fem.form(ufl.action(F_gravity, probe)))
        work = MPI.COMM_WORLD.allreduce(local, op=MPI.SUM)
        volume = params.Lx * params.Ly * (params.Lz if dim == 3 else 1.0)
        expected = (-params.gravity * params.Materials["Water"]["Density"]
                    * params.Models["porous"]["Initial_Porosity"] * volume
                    if axis == dim-1 else 0.0)
        assert work == pytest.approx(expected, rel=1e-12, abs=1e-12)
    vec_gravity.destroy()
    vec_zero.destroy()


def test_pore_pressure_reset_step_restores_old_pressure():
    _, mesh, model = _build_model()

    mesh.Constant_Field("pwOld", 12.5)
    mesh.Zero_Field("pw")

    result = model.reset_step()

    assert result is None
    np.testing.assert_allclose(mesh.Get_Field("pw").x.array, 12.5)


def _build_two_squared_params(**kwargs):
    """Return ``_build_params`` on the two-squares sample mesh."""
    from femcubes.mesh_generators.two_squared import Sample_Parameters

    params = _build_params(**kwargs)
    sample = Sample_Parameters()
    for attr in ("MeshType", "L", "dx", "dx_interface"):
        setattr(params, attr, getattr(sample, attr))
    params.Ly = params.L  # two unit squares
    params.Models["porous"]["domain"] = "domain:left"
    return params


def test_pore_pressure_on_a_subdomain_registers_and_integrates_there_only():
    from femcubes.core.physics.physics import Physics

    params = _build_two_squared_params(gravity=-9.81)
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["porous"]

    for name in ("u", "uOld", "pw", "pwOld", "poros", "qw"):
        assert mesh.Get_Field_Support(name).domain_names == ("domain:left",)

    # The fluid weight only acts on the left unit square.
    _, F_u = model.assemble_KF(model.Step_u)
    space, component = mesh.Get_Field_Space_For_BC("u", 1)
    probe = fem.Function(space)
    probe.sub(component).interpolate(lambda x: np.ones(x.shape[1]))
    work = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(mesh.Compile_Form(ufl.action(F_u, probe))), op=MPI.SUM)
    expected = -params.gravity * params.Materials["Water"]["Density"] * params.Models["porous"]["Initial_Porosity"] * params.L**2
    assert work == pytest.approx(expected, rel=1e-12)

    # Lumped storage is assembled on the subdomain pressure DOFs.
    tangent, residual = physics.Get_Kmat_Fvec(model.Step_pw)
    assert tangent is not None and residual is not None


def test_pore_pressure_lumped_storage_rejects_pressure_shared_between_domains():
    from femcubes.core.physics.physics import Physics

    params = _build_two_squared_params()
    params.Models["porous_right"] = dict(params.Models["porous"], domain="domain:right")
    params.ModelNames = ["porous", "porous_right"]
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    with pytest.raises(NotImplementedError, match="requires 'pw' to live on that domain only"):
        physics.Get_Kmat_Fvec(params.Solution_Steps["pw"])
