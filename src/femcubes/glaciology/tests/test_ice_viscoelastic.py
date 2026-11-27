import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.utils.mpi_utils import mprint
from femcubes.glaciology.ice_material import IceMaterial
from femcubes.solids.split_viscoelastic import SplitViscoElastic
from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import SplitViscoElasticKernels

import femcubes.boundary
import femcubes.glaciology


def _build_params(include_plots: bool = False):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}

        MeshType: str = "Rectangle"
        dx: float = 0.5
        dim: int = 2
        Lx: float = 1.0
        Ly: float = 4.0
        Quadrature_Degree: int = 2

        Materials: dict = {}
        Materials["Ice"] = {
            "LE_Damping": 0.0,
            "Inertia": False,
        }

        gravity: float = 0.0

        if include_plots:
            Plot_Fields: list = ["u_x", "u_y", "poros", "T"]
            Plot_Measures: list = [("BC_Top_u", "BC_Top_F")]
        else:
            Plot_Fields: list = []

        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 2, "T": 1, "Porosity": 0}

        Models = {}
        ModelNames = ["Solid", "BC_Bottom", "BC_Left", "BC_Top"]
        Models["Solid"] = {
            "Type": "Glaciology:ViscoElastic",
            "Material": "Ice",
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
        Models["BC_Top"] = {
            "Type": "Boundary:DisplacementControl",
            "boundary": "top",
            "field": "u_y",
            "rate": 1.0e-6 * Ly,
            "Dummy": 1.0e14,
        }

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 100
        line_search: bool = False
        line_search_lims: list = [0.01, 1.0]
        rel_tol: float = 1e-6
        abs_tol: float = 1e-8

        output_dir: str = None

        start_time: float = 0.0
        end_time: float = 100.0 * 50
        dt: float = 100.0
        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0

    return Params()


@pytest.fixture
def InputParams():
    return _build_params()


def test_ice_viscoelastic_porosity_order_falls_back_to_pw():
    params = _build_params()
    params.Field_Orders = {"u": 2, "T": 1, "pw": 1}

    mesh = Mesh(params)
    Physics(mesh, params)

    poros = mesh.Get_Field("poros")
    assert poros.function_space.element.basix_element.degree == 1


def test_ice_viscoelastic_selects_driver_from_material_parameters():
    params = _build_params()
    params.Materials["Ice"].update(CrackDrivingForce="Spectral", ell=0.1)
    params.Models["Solid"]["DamageModel"] = "PhaseField"
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Damage", "CG", 1)
    physics = Physics(mesh, params)
    assert physics.models["Solid"].DrivingForce_Type == "Spectral"


def test_ice_viscoelastic_assembly_interpolates_ufl_damage_expression():
    params = _build_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]

    mesh.Constant_Field("poros", 0.25)
    mesh.Constant_Field("T", -10.0)
    model.d_e = 1.0 - mesh.Get_Field("poros")

    K, F = model.assemble_KF(model.My_Step)

    assert K is not None
    assert F is not None
    np.testing.assert_allclose(mesh.Get_Field("dam_elastic").x.array, 0.75)


@pytest.mark.full_model
class TestIceViscoElastic:
    def test_two_layer_glen_creep_matches_series_solution(self, InputParams):
        mesh = Mesh(InputParams)
        physics = Physics(mesh, InputParams)
        physics.Initialize_Fields()

        temperature = -10.0
        phi_bottom = 0.0
        phi_top = 0.2
        x = ufl.SpatialCoordinate(mesh.mesh)
        poros_expr = ufl.conditional(x[1] < 0.5 * InputParams.Ly, phi_bottom, phi_top)
        mesh.Set_From_Expression("poros", poros_expr)
        mesh.Constant_Field("T", temperature)

        solver = TimeSolver(InputParams, physics)
        solver.run()

        # The layers are loaded in series, so the vertical stress is common.
        # For the 2-D plane-strain kernel benchmark, each layer satisfies
        # eps_dot_i = A_i * (sigma_yy / 2)^n. Equal layer thicknesses therefore
        # give A_eff = 0.5 * (A_bottom + A_top).
        material = IceMaterial()
        A_bottom = material.Get_Creep_A(phi_bottom, temperature)
        A_top = material.Get_Creep_A(phi_top, temperature)
        A_eff = 0.5 * (A_bottom + A_top)
        glen_n = material.Get_Creep_n(phi_bottom, temperature)
        strain_rate = InputParams.Models["BC_Top"]["rate"] / InputParams.Ly
        sigma_yy = 2.0 * (strain_rate / A_eff) ** (1.0 / glen_n)
        force_expected = sigma_yy * InputParams.Lx
        force_measured = abs(InputParams.Global_Measures.get("BC_Top_F", 0.0))
        rel_err = abs(force_measured - force_expected) / abs(force_expected)

        creep_field = mesh.Get_Field("Creep_Stress_Scale").x.array
        elastic_energy = mesh.Get_Field("ElasticEnergy").x.array

        mprint(
            f"Two-layer ice viscoelastic force comparison: "
            f"measured={force_measured:.6e}, expected={force_expected:.6e}, "
            f"rel_err={rel_err:.3e}"
        )
        assert np.all(np.isfinite(creep_field))
        assert np.max(creep_field) > np.min(creep_field)
        assert np.all(np.isfinite(elastic_energy))
        assert rel_err < 0.2


if __name__ == "__main__":
    params = _build_params(True)
    TestIceViscoElastic().test_two_layer_glen_creep_matches_series_solution(params)


@pytest.mark.parametrize("split", ["FullEnergy", "VolDev", "VolOnly", "Spectral", "NoTension", "Stress-Based"])
@pytest.mark.parametrize("coupling", ["EffectiveStress", "DegradedStress", "FullDamagedStress", "PartialExplicit"])
def test_phasefield_drivers_support_spatially_varying_ice_properties(split, coupling):
    params = _build_params()
    params.Materials["Ice"].update(CrackDrivingForce=split, ell=0.1)
    params.Models["Solid"]["DamageModel"] = "PhaseField"
    params.Models["Solid"].update(true_2d=True, CreepCoupling=coupling)
    params.ModelNames = ["Solid"]
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Damage", "CG", 1)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]
    x = ufl.SpatialCoordinate(mesh.mesh)
    mesh.Set_From_Expression("poros", 0.1+0.05*x[1])
    mesh.Set_From_Expression("T", -10.0+x[1])
    space = mesh.Get_Field("ElasticEnergy").function_space

    def sample(expression):
        value = fem.Function(space)
        value.interpolate(fem.Expression(expression, space.element.interpolation_points))
        value.x.scatter_forward()
        return value.x.array.copy()

    strain = 1e-4
    mesh.Get_Field("u").interpolate(lambda x: strain*x[:2])
    mesh.Get_Field("u").x.scatter_forward()
    mesh.Constant_Field("phasefield", 0.6)
    for temperature in (-10.0, -5.0):
        mesh.Set_From_Expression("T", temperature+x[1])
        model.assemble_KF(0)
        lame, shear = sample(model.lame), sample(model.shear)
        expected = (2*lame+2*shear)*strain**2
        if split == "Stress-Based":
            strength = 2*((2*lame+2*shear)*strain)**2/model.f_t**2
            expected = sample(model.Gc)/(4*model.ell)*(strength-1)
        actual = mesh.Get_Field("ElasticEnergy").x.array
        assert np.ptp(actual) > 0
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


def _build_damaged_ice(split="NoTension", coupling="EffectiveStress", true_2d=False, dim=2):
    params = _build_params()
    params.dim = dim
    params.Lz = 1.0
    params.ModelNames = ["Solid"]
    params.Materials["Ice"].update(CrackDrivingForce=split, ell=0.1, kappa=1e-3)
    params.Models["Solid"].update(DamageModel="PhaseField", CreepCoupling=coupling, true_2d=true_2d)
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Damage", "CG", 1)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    mesh.Constant_Field("phasefield", 0.6)
    mesh.Constant_Field("poros", 0.2)
    mesh.Constant_Field("T", -10.0)
    return params, mesh, physics, physics.models["Solid"]


def _integral(form):
    return MPI.COMM_WORLD.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


@pytest.mark.parametrize("split", ["FullEnergy", "VolDev", "VolOnly", "Spectral", "NoTension", "Stress-Based"])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_ice_mechanics_degrades_only_selected_stress_once(split, sign):
    params, mesh, _, model = _build_damaged_ice(split, true_2d=True)
    assert isinstance(model, SplitViscoElastic)
    strain = sign*1e-4
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: strain*x[:2])
    u.x.scatter_forward()
    _, residual = model.assemble_KF(0)
    material = IceMaterial()
    lame = material.Get_Lame_Parameter(0.2, -10.0)
    shear = material.Get_Shear_Modulus(0.2, -10.0)
    factor = 1e-3+(1-1e-3)*0.4**2 if sign > 0 or split == "FullEnergy" else 1.0
    expected = np.zeros(6)
    expected[:2] = factor*(2*lame+2*shear)*strain
    np.testing.assert_allclose(mesh.Get_Field("sigma").x.array.reshape(-1, 6),
                               np.broadcast_to(expected, mesh.Get_Field("sigma").x.array.reshape(-1, 6).shape),
                               rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(_integral(ufl.action(residual, u)),
                               4*factor*(lame+shear)*strain**2*params.Lx*params.Ly, rtol=1e-12)


@pytest.mark.parametrize("split", ["FullEnergy", "VolDev", "VolOnly", "Spectral", "NoTension", "Stress-Based"])
@pytest.mark.parametrize("coupling", ["EffectiveStress", "DegradedStress", "FullDamagedStress", "PartialExplicit"])
@pytest.mark.parametrize("true_2d", [False, True])
def test_ice_algorithmic_tangent_and_variable_properties(split, coupling, true_2d):
    params, mesh, _, model = _build_damaged_ice(split, coupling, true_2d)
    x = ufl.SpatialCoordinate(mesh.mesh)
    mesh.Set_From_Expression("poros", 0.1+0.05*x[1])
    mesh.Set_From_Expression("T", -12.0+x[1])
    u = mesh.Get_Field("u")
    previous_strain = None
    if coupling == "PartialExplicit":
        mesh.Get_Field("uOld").interpolate(
            lambda x: 1e-4*np.vstack((0.13*x[0]+0.02*x[1], -0.17*x[1]+0.02*x[0])),
        )
        mesh.Get_Field("uOld").x.scatter_forward()
        previous_strain = 1e-4*np.array([[0.13, -0.17, 0, 0, 0, np.sqrt(2)*0.02]])
    probe = fem.Function(u.function_space)
    test = fem.Function(u.function_space)
    probe.interpolate(lambda x: 1e-4*np.vstack((0.3*x[0]+0.2*x[1], -0.1*x[1]+0.1*x[0])))
    test.interpolate(lambda x: 1e-4*np.vstack((-0.2*x[0]+0.3*x[1], 0.4*x[1]+0.1*x[0])))
    probe.x.scatter_forward()
    test.x.scatter_forward()

    def assemble(delta):
        u.interpolate(lambda x: 1e-4*np.vstack((0.17*x[0]+0.04*x[1], -0.23*x[1]+0.04*x[0])))
        u.x.array[:] += delta*probe.x.array
        u.x.scatter_forward()
        return model.assemble_KF(0)

    tangent, _ = assemble(0.0)
    analytical = _integral(ufl.action(ufl.action(tangent, probe), test))
    step = 1e-5
    _, plus = assemble(step)
    plus_work = _integral(ufl.action(plus, test))
    _, minus = assemble(-step)
    minus_work = _integral(ufl.action(minus, test))
    np.testing.assert_allclose(analytical, (plus_work-minus_work)/(2*step), rtol=2e-6, atol=1e-9)
    assert np.linalg.norm(mesh.Get_Field("Plastic_Strain").x.array) > 0
    stress = mesh.Get_Field("sigma").x.array.reshape(-1, 6)
    positive = mesh.Get_Field("sigma_plus").x.array.reshape(-1, 6)
    negative = mesh.Get_Field("sigma_minus").x.array.reshape(-1, 6)
    g = mesh.Get_Field("dam_elastic").x.array
    np.testing.assert_allclose(stress, g[:, None]*positive+negative)

    # Each point must use its own sampled material values and the live dt,
    # including when the solver constant differs from params.dt.
    dt = fem.Constant(mesh.mesh, params.dt/2)
    model.Update_Weak_Constants(0.0, dt, 0)
    strains = mesh.Get_Field("Total_Strain").x.array.reshape(-1, 6)
    old = mesh.Get_Field("Plastic_Strain_Old").x.array.reshape(-1, 6)
    for point, (lame, shear, viscosity) in enumerate(zip(
        mesh.Get_Field("Lame").x.array, mesh.Get_Field("Shear").x.array, mesh.Get_Field("Creep_Stress_Scale").x.array,
    )):
        reference = SplitViscoElasticKernels(viscosity, 3.0, lame, shear, split, coupling, true_2d)(
            strains[point:point+1], old[point:point+1], g[point:point+1], params.dt/2,
            old_total_strains=previous_strain,
        )
        np.testing.assert_allclose(mesh.Get_Field("sigma").x.array.reshape(-1, 6)[point], reference["stress"][0])
        np.testing.assert_allclose(mesh.Get_Field("dsigma_dStrainMandel").x.array.reshape(-1, 6, 6)[point],
                                   reference["tangent"][0])


@pytest.mark.parametrize("dim", [2, 3])
def test_partial_explicit_ice_history_and_stress_survive_live_refinement(dim):
    from femcubes.core.physics.tests.test_physics_remesh import _refine_live_mesh

    _, mesh, physics, model = _build_damaged_ice(coupling="PartialExplicit", dim=dim)
    gradient = 1e-4*np.diag([0.2, -0.1] if dim == 2 else [0.2, -0.1, 0.04])
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: gradient@x[:dim])
    u.x.scatter_forward()
    model.assemble_KF(0)
    physics.commit()
    u.interpolate(lambda x: 1.3*gradient@x[:dim])
    u.x.scatter_forward()
    physics.Get_Kmat_Fvec(0)
    expected_stress = mesh.Get_Field("sigma").x.array.reshape(-1, 6)[0].copy()
    expected_creep = mesh.Get_Field("Plastic_Strain").x.array.reshape(-1, 6)[0].copy()
    assert np.linalg.norm(expected_creep) > 0
    assert _refine_live_mesh(mesh).installed
    physics.Post_Remesh()
    tangent, residual = physics.Get_Kmat_Fvec(0)
    assert np.isfinite(tangent.norm()) and np.isfinite(residual.norm())
    for name, expected in (("sigma", expected_stress), ("Plastic_Strain", expected_creep)):
        actual = mesh.Get_Field(name).x.array.reshape(-1, 6)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape),
                                   rtol=1e-10, atol=1e-12*np.linalg.norm(expected))


@pytest.mark.parametrize("inertia", [False, True])
def test_ice_density_gravity_and_inertia_remain_active_after_damage(inertia):
    params = _build_params()
    params.ModelNames = ["Solid"]
    params.gravity = -2.0
    params.Materials["Ice"].update(CrackDrivingForce="NoTension", ell=0.1, Inertia=inertia)
    params.Models["Solid"]["DamageModel"] = "PhaseField"
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Damage", "CG", 1)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]
    mesh.Constant_Field("poros", 0.25)
    mesh.Constant_Field("T", -10.0)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.full(x.shape[1], 0.2))))
    u.x.scatter_forward()
    probe = fem.Function(u.function_space)
    probe.interpolate(lambda x: np.vstack((np.zeros(x.shape[1]), np.ones(x.shape[1]))))
    probe.x.scatter_forward()
    rho = 917.0*0.75
    volume = params.Lx*params.Ly
    mass_tangent = rho/(params.Newmark_Beta*params.dt**2) if inertia else 0.0
    for damage in (0.0, 0.8, 1.0):
        mesh.Constant_Field("phasefield", damage)
        tangent, residual = model.assemble_KF(0)
        assert model.d_rho == 1.0
        np.testing.assert_allclose(_integral(ufl.action(residual, probe)),
                                   volume*(-rho*params.gravity+0.2*mass_tangent), rtol=1e-12)
        np.testing.assert_allclose(_integral(ufl.action(ufl.action(tangent, probe), probe)),
                                   volume*mass_tangent, rtol=1e-12, atol=1e-10)


@pytest.mark.parametrize("formulation", ["AT1", "AT2", "HO-AT1", "HO-AT2"])
@pytest.mark.parametrize("coupling", ["EffectiveStress", "DegradedStress", "FullDamagedStress", "PartialExplicit"])
def test_ice_strength_driver_uses_updated_elastic_state(formulation, coupling):
    params = _build_params()
    params.ModelNames = ["Solid", "pf"]
    params.Solution_Steps = {"u": 0, "phasefield": 1}
    params.Field_Orders["phasefield"] = 2
    params.Materials["Ice"].update(CrackDrivingForce="Stress-Based", ell=0.1,
                                  PF_formulation=formulation)
    params.Models["Solid"].update(DamageModel="PhaseField", CreepCoupling=coupling)
    irreversibility = "Penalty" if formulation in ("AT1", "HO-AT1") else "Hist"  # Hist needs AT2/HO-AT2
    params.Models["pf"] = {"Type": "Glaciology:PhaseField", "Material": "Ice",
                            "CDF_field": "ElasticEnergy", "Irreversible_Method": irreversibility}
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]
    x = ufl.SpatialCoordinate(mesh.mesh)
    mesh.Set_From_Expression("poros", 0.1+0.05*x[1])
    mesh.Set_From_Expression("T", -12.0+x[1])
    mesh.Constant_Field("phasefield", 0.6)
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: 1e-4*np.vstack((0.17*x[0]+0.04*x[1], -0.23*x[1]+0.04*x[0])))
    u.x.scatter_forward()
    if coupling == "PartialExplicit":
        mesh.Copy_Field("u", "uOld")
    model.assemble_KF(0)
    assert np.linalg.norm(mesh.Get_Field("Plastic_Strain").x.array) > 0
    space = mesh.Get_Field("ElasticEnergy").function_space

    def sample(expression):
        field = fem.Function(space)
        field.interpolate(fem.Expression(expression, space.element.interpolation_points))
        field.x.scatter_forward()
        return field.x.array

    positive = mesh.Get_Field("sigma_plus").x.array.reshape(-1, 6)
    strength = np.sum(positive**2, axis=1)/model.f_t**2
    gc = sample(model.Gc)
    expected = gc/(4*model.ell)*(strength-1) if formulation in ("AT2", "HO-AT2") else 3*gc/(16*model.ell)*strength
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, expected, rtol=1e-12)
    stiffness, residual = physics.Get_Kmat_Fvec(1)
    assert np.isfinite(stiffness.norm()) and np.isfinite(residual.norm())


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_ice_three_dimensional_hydrostatic_split(sign):
    params, mesh, _, model = _build_damaged_ice(dim=3)
    strain = sign*1e-4
    u = mesh.Get_Field("u")
    u.interpolate(lambda x: strain*x)
    u.x.scatter_forward()
    _, residual = model.assemble_KF(0)
    material = IceMaterial()
    lame = material.Get_Lame_Parameter(0.2, -10.0)
    shear = material.Get_Shear_Modulus(0.2, -10.0)
    factor = 1e-3+(1-1e-3)*0.4**2 if sign > 0 else 1.0
    expected = np.zeros(6)
    expected[:3] = factor*(3*lame+2*shear)*strain
    stress = mesh.Get_Field("sigma").x.array.reshape(-1, 6)
    np.testing.assert_allclose(stress, np.broadcast_to(expected, stress.shape),
                               rtol=1e-12, atol=1e-12*np.max(np.abs(expected)))
    np.testing.assert_allclose(_integral(ufl.action(residual, u)),
                               3*factor*(3*lame+2*shear)*strain**2*params.Lx*params.Ly*params.Lz, rtol=1e-12)


@pytest.mark.parametrize("shared_temperature", [False, True])
def test_ice_viscoelastic_restricted_to_a_subdomain(shared_temperature):
    params = _build_params()
    params.MeshType, params.L, params.Ly, params.dx = "TwoSquared", 1.0, 1.0, 0.25
    params.ModelNames = ["Solid"]
    params.Materials["Ice"].update(CrackDrivingForce="Stress-Based", ell=0.1)
    params.Models["Solid"].update(DamageModel="PhaseField", true_2d=True, domain="domain:left")
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Damage", "CG", 1, domain="domain:left")
    if shared_temperature:
        # e.g. a thermal model on the full mesh: material expressions then mix meshes.
        mesh.Add_Field("T", "Temperature [C]", "CG", params.Field_Orders["T"])
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]

    for field_name in ("u", "poros", "Lame", "Creep_Stress_Scale", "sigma_plus", "ElasticEnergy",
                       "TensileStrength", "FractureToughness"):
        assert mesh.Get_Field_Support(field_name).domain_names == ("domain:left",)
    assert mesh.Get_Field_Support("T").is_full == shared_temperature

    mesh.Constant_Field("poros", 0.2)
    mesh.Constant_Field("T", -10.0)
    mesh.Constant_Field("phasefield", 0.6)
    strain = 1e-4
    mesh.Get_Field("u").interpolate(lambda x: strain*x[:2])
    mesh.Get_Field("u").x.scatter_forward()
    K, F = model.assemble_KF(model.My_Step)

    assert mesh.Compile_Form(K) is not None
    assert mesh.Compile_Form(F) is not None
    np.testing.assert_allclose(mesh.Get_Field("Lame").x.array, IceMaterial().Get_Lame_Parameter(0.2, -10.0))
    lame, shear, Gc = (mesh.Evaluate_At_Field_Points("ElasticEnergy", e).reshape(-1) for e in (model.lame, model.shear, model.Gc))
    strength = 2*((2*lame+2*shear)*strain)**2/model.f_t**2
    np.testing.assert_allclose(mesh.Get_Field("ElasticEnergy").x.array, Gc/(4*model.ell)*(strength-1), rtol=1e-10)


@pytest.mark.parametrize("true_2d", [False, True])
def test_creep_scale_equals_the_parent_at_uniform_porosity_and_temperature(true_2d):
    porosity, temperature = 0.1, -20.0
    material = IceMaterial()

    def state(parent):
        params = _build_params()
        params.ModelNames = ["Solid"]
        params.Models["Solid"]["true_2d"] = true_2d
        if parent:
            params.Models["Solid"]["Type"] = "Solid:SplitViscoElastic"
            params.Materials["Ice"].update(
                Lame=material.Get_Lame_Parameter(porosity, temperature),
                Shear=material.Get_Shear_Modulus(porosity, temperature),
                Bulk=material.Get_Bulk_Modulus(porosity, temperature),
                Density=material.Get_Density(porosity, temperature), Visc_Type="Glen",
                Glen_A=float(material.Get_Creep_A(porosity, temperature)), Glen_n=3.0)
        params.dt = 1.0e7
        mesh = Mesh(params)
        physics = Physics(mesh, params)
        physics.Initialize_Fields()
        if not parent:
            mesh.Constant_Field("poros", porosity)
            mesh.Constant_Field("T", temperature)
        u = mesh.Get_Field("u")
        u.interpolate(lambda x: 1e-4*np.vstack((0.3*x[0]+0.2*x[1], -0.1*x[1])))
        u.x.scatter_forward()
        physics.models["Solid"].assemble_KF(0)
        return mesh, {name: mesh.Get_Field(name).x.array.copy() for name in ("sigma", "Plastic_Strain")}

    mesh, ice = state(False)
    _, reference = state(True)
    for name, expected in reference.items():
        np.testing.assert_allclose(ice[name], expected, rtol=1e-10, atol=1e-10*np.abs(expected).max(), err_msg=name)
    assert np.abs(reference["Plastic_Strain"]).max() > 0.0
    np.testing.assert_allclose(mesh.Get_Field("Creep_Stress_Scale").x.array,
                               material.Get_ViscoPlastic_viscosity(porosity, temperature, dim=2 if true_2d else 3),
                               rtol=1e-12)


@pytest.mark.parametrize(("field", "value", "message"), [
    ("poros", 1.0, "porosity 'poros' must be finite with 0 <= p < 1"),
    ("T", 1.0, "temperature 'T' must be finite and at most 0 C"),
])
def test_invalid_porosity_or_temperature_raises_a_clear_error(field, value, message):
    _, mesh, _, model = _build_damaged_ice("Stress-Based")
    mesh.Constant_Field(field, value)
    with pytest.raises(ValueError, match=message):
        model.assemble_KF(0)


def test_warns_once_when_the_temperature_was_never_set():
    params = _build_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["Solid"]
    with pytest.warns(UserWarning, match="temperature 'T' is 0 C everywhere"):
        model.assemble_KF(0)


def test_material_entry_is_required():
    params = _build_params()
    params.Models["Solid"].pop("Material")
    with pytest.raises(KeyError, match="Material"):
        Physics(Mesh(params), params)


def test_the_true2d_alias_is_not_read():
    params = _build_params()
    params.Models["Solid"]["True2D"] = True
    model = Physics(Mesh(params), params).models["Solid"]
    assert model.true_2d is False
