"""Remesh lifecycle tests for built-in models with mesh-local state."""

from types import SimpleNamespace

from dolfinx import fem
import numpy as np
import pytest

from femcubes.boundary.external_force import ExternalForce
from femcubes.boundary.external_force_from_file import ExternalForceFromFile
from femcubes.boundary.torsion import AppliedTorsion
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.mesh.refinement import (
    CellIndicatorResult,
    MeshMarkerState,
    expand_refinement_buffer,
    refine_mesh_topology,
    snapshot_registered_fields,
    transfer_mesh_markers,
    transfer_quadrature_fields,
    transfer_standard_fields,
)
from femcubes.core.physics.physics import Physics
from femcubes.electrochemistry.electrolyte import Electrolyte
from femcubes.electrochemistry.prescribed_flux import PrescribedFlux
from femcubes.glaciology.basal_friction import BasalFriction
from femcubes.glaciology.ice_phasefield import IcePhaseField
from femcubes.glaciology.level_set import LevelSet
from femcubes.glaciology.ocean_pressure import OceanPressure
from femcubes.glaciology.shallow_ice import ShallowIce
from femcubes.poromechanics.pore_pressure import PorePressure

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.glaciology  # noqa: F401 - ensure model registration


def _install_one_refinement(mesh_wrapper):
    """Install one refined cell with every registered marker and field."""

    marker_state = MeshMarkerState.from_mesh_wrapper(mesh_wrapper)
    field_snapshot = snapshot_registered_fields(mesh_wrapper)
    mesh = mesh_wrapper.mesh
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    marked = np.zeros(cells.size, dtype=bool)
    marked[cells.size // 2] = True
    indicator = CellIndicatorResult(
        cells=cells,
        values=marked.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=marked.copy(),
        size_mask=np.ones(cells.size, dtype=bool),
        marked_mask=marked.copy(),
    )
    selection = expand_refinement_buffer(SimpleNamespace(mesh=mesh), indicator, 0, target_h=0.1)
    artifacts = refine_mesh_topology(mesh, selection)
    artifacts = transfer_mesh_markers(marker_state, artifacts)
    transferred = transfer_standard_fields(field_snapshot, artifacts)
    transferred = transfer_quadrature_fields(
        field_snapshot,
        artifacts,
        transferred,
    )
    return mesh_wrapper.Install_Refined_State(artifacts, transferred)


def _contains_ufl_terminal(expression, target):
    """Return whether a UFL expression graph contains one exact terminal."""

    pending = [expression]
    visited = set()
    while pending:
        current = pending.pop()
        if current is target:
            return True
        if id(current) in visited:
            continue
        visited.add(id(current))
        pending.extend(getattr(current, "ufl_operands", ()))
    return False


def _assert_constant_on_live_mesh(constant, mesh_wrapper):
    assert constant.ufl_domain().ufl_cargo() is mesh_wrapper.mesh._cpp_object


def _build_ice_fracture_params():
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 1.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        Quadrature_Degree = 2

        Global_Measures = {}
        Materials = {
            "Ice": {
                "ell": 0.25,
                "f_s": 2.0e5,
                "PF_formulation": "AT2",
                "FailureType": "r1",
                "Inertia": False,
                "kappa": 1.0e-4,
            }
        }
        gravity = 0.0
        Plot_Fields = []
        Plot_Measures = []
        Solution_Steps = {"u": 0, "phasefield": 1}
        Field_Orders = {
            "u": 1,
            "phasefield": 1,
            "Porosity": 0,
            "Temperature": 1,
        }
        ModelNames = ["Solid", "PhaseField"]
        Models = {
            "Solid": {
                "Type": "Glaciology:Cohesive_ViscoElastic",
                "Material": "Ice",
                "DamageModel": "PhaseField",
            },
            "PhaseField": {
                "Type": "Glaciology:PhaseField",
                "Material": "Ice",
                "CDF_field": "PsiField",
                "Irreversible_Method": "Hist",
            },
        }
        start_time = 0.0
        end_time = 1.0
        t = 0.0
        dt = 0.1
        Newmark_Beta = 0.25
        Newmark_Gamma = 0.5
        output_dir = None

    return Params()


def _build_ice_viscoelastic_params():
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 1.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        Quadrature_Degree = 2

        Global_Measures = {}
        Materials = {"Ice": {"Inertia": False, "LE_Damping": 0.0}}
        gravity = 0.0
        Plot_Fields = []
        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 1, "Porosity": 0, "Temperature": 1}
        ModelNames = ["Solid"]
        Models = {
            "Solid": {
                "Type": "Glaciology:ViscoElastic",
                "Material": "Ice",
            }
        }
        t = 0.0
        dt = 0.1
        Newmark_Beta = 0.25
        Newmark_Gamma = 0.5
        output_dir = None

    return Params()


def test_ice_phasefield_penalty_preserves_committed_bound_and_assembles_after_remesh():
    params = _build_ice_fracture_params()
    params.Models["PhaseField"].update(
        {
            "Irreversible_Method": "Penalty",
            "Irreversibility_Tolerance": 0.02,
            "Irreversibility_Smoothing": 0.001,
        }
    )
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    model = physics.models["PhaseField"]

    mesh.Constant_Field("phasefieldOld", 0.4)
    mesh.Constant_Field("phasefield", 0.2)
    K_before, F_before = physics.Get_Kmat_Fvec(model.My_Step)
    assert np.isfinite(K_before.norm())
    assert np.isfinite(F_before.norm())

    result = _install_one_refinement(mesh)
    physics.Post_Remesh()

    assert result.installed
    np.testing.assert_allclose(mesh.Get_Field("phasefieldOld").x.array, 0.4)
    assert "phasefield_Hist" not in mesh.fields
    assert "LMultiplier_PF" not in mesh.fields

    K_after, F_after = physics.Get_Kmat_Fvec(model.My_Step)
    assert np.isfinite(K_after.norm())
    assert np.isfinite(F_after.norm())


def test_ice_viscoelastic_rebinds_material_and_expression_caches():
    params = _build_ice_viscoelastic_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solid = physics.models["Solid"]

    physics.Get_Kmat_Fvec(0)
    assert solid._total_strain_expr is not None
    assert solid._property_expressions.get("Lame") is not None
    old_porosity = mesh.Get_Field("poros")
    old_temperature = mesh.Get_Field("T")

    _install_one_refinement(mesh)
    physics.Post_Remesh()

    porosity = mesh.Get_Field("poros")
    temperature = mesh.Get_Field("T")
    assert porosity is not old_porosity
    assert temperature is not old_temperature
    assert solid._total_strain_expr is None
    assert solid._damage_expr is None
    assert solid._property_expressions == {}
    assert _contains_ufl_terminal(solid.lame, porosity)
    assert _contains_ufl_terminal(solid.viscosity, porosity)
    assert _contains_ufl_terminal(solid.viscosity, temperature)

    K_mat, F_vec = physics.Get_Kmat_Fvec(0)
    assert np.isfinite(K_mat.norm())
    assert np.isfinite(F_vec.norm())


def _build_shallow_ice_params():
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 2.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        Global_Measures = {}
        gravity = -9.81
        Materials = {
            "Ice": {
                "C_base": 1.0e-4,
                "Friction_Model": "Weertman",
                "Front_Ocean_Stress": False,
            },
            "Water": {"Density": 1030.0, "WaterHeight": 0.0},
        }
        Solution_Steps = {"v": 0, "h": 1, "LS": 2}
        Field_Orders = {"v": 1, "h": 0, "LS": 1, "Temperature": 1}
        Level_Set = {"Transition_Width": 0.25}
        Models = {
            "Shallow": {"Type": "Glaciology:ShallowIce"},
            "Margin": {
                "Type": "Glaciology:LevelSet",
                "Velocity_Field": "v",
            },
        }
        dt = 0.1

    return Params()


def test_shallow_ice_and_level_set_forms_compile_after_remesh():
    params = _build_shallow_ice_params()
    mesh = Mesh(params)
    shallow = ShallowIce("Shallow", params, mesh)
    level_set = LevelSet("Margin", params, mesh)
    mesh.finalize_test_functions()
    old_temperature = mesh.Get_Field("T")

    _install_one_refinement(mesh)
    shallow.Post_Remesh()
    level_set.Post_Remesh()

    temperature = mesh.Get_Field("T")
    assert temperature is not old_temperature
    assert _contains_ufl_terminal(shallow.creep_A, temperature)
    for model, step in (
        (shallow, shallow.step_V),
        (shallow, shallow.step_H),
        (level_set, level_set.step_LS),
    ):
        K_form, F_form = model.assemble_KF(step)
        assert mesh.Compile_Form(K_form) is not None
        assert mesh.Compile_Form(F_form) is not None


def _build_pore_pressure_params():
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 1.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        gravity = 0.0
        Materials = {
            "Solid": {
                "Bulk": 1.0e9,
                "permeability0": 1.0e-10,
                "permeability_exp": 2.0,
                "Biot": 0.8,
            },
            "Water": {
                "Bulk": 2.0e9,
                "Density": 1000.0,
                "Viscosity": 1.0e-3,
            },
        }
        Solution_Steps = {"u": 0, "pw": 0}
        Field_Orders = {"u": 1, "pw": 1}
        Models = {
            "Pore": {
                "Type": "PoroMechanics:PorePressure",
                "Material": "Solid",
                "Fluid": "Water",
                "Saturated": True,
            }
        }
        dt = 0.1

    return Params()


def test_pore_pressure_constants_preserve_values_on_new_domain():
    params = _build_pore_pressure_params()
    mesh = Mesh(params)
    model = PorePressure("Pore", params, mesh)
    mesh.finalize_test_functions()
    model.Biot.value = 0.7
    model.Sw.value = 0.9
    model.SwOld.value = 0.85
    model.dSw.value = 0.05
    model.ddSw.value = -0.02
    old_constants = tuple(
        getattr(model, name) for name in ("Biot", "Sw", "SwOld", "dSw", "ddSw")
    )

    _install_one_refinement(mesh)
    model.Post_Remesh()

    expected = (0.7, 0.9, 0.85, 0.05, -0.02)
    for name, old_constant, value in zip(
        ("Biot", "Sw", "SwOld", "dSw", "ddSw"),
        old_constants,
        expected,
    ):
        constant = getattr(model, name)
        assert constant is not old_constant
        _assert_constant_on_live_mesh(constant, mesh)
        np.testing.assert_allclose(constant.value, value)

    K_form, F_form = model.assemble_KF(model.Step_pw)
    assert mesh.Compile_Form(K_form) is not None
    assert mesh.Compile_Form(F_form) is not None


def test_electrolyte_forms_compile_from_transferred_fields_after_remesh():
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 1.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        Species = {
            "Na": {
                "Diffusivity": 1.0e-9,
                "Charge": 1.0,
                "C0": 1.0,
            }
        }
        Volume_Reactions = {}
        C_ref = 1.0
        Solution_Steps = {"C": 0, "E": 0}
        Field_Orders = {"C": 1, "E": 1}
        Models = {"Electrolyte": {"Type": "ElectroChemistry:Electrolyte", "Activity_Model": "ideal"}}
        dt = 0.1

    params = Params()
    mesh = Mesh(params)
    model = Electrolyte("Electrolyte", params, mesh)
    mesh.finalize_test_functions()

    _install_one_refinement(mesh)
    model.Post_Remesh()

    K_form, F_form = model.assemble_KF(model.C_Step)
    assert mesh.Compile_Form(K_form) is not None
    assert mesh.Compile_Form(F_form) is not None


def _build_boundary_params():
    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 1.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        gravity = -9.81
        t = 0.0
        dt = 0.1
        Solution_Steps = {"u": 0, "C": 1}
        Field_Orders = {"u": 1}
        Materials = {
            "Ice": {
                "Linear_Friction_C": 0.5,
                "Weertman_C": 1.0,
                "Weertman_m": 1.0,
            },
            "Water": {"Density": 1000.0, "WaterHeight": 0.75},
        }
        Models = {}

    return Params()


@pytest.fixture
def refined_boundary_models(tmp_path):
    """Return representative boundary models after one installed remesh."""
    params = _build_boundary_params()
    x_file = tmp_path / "force_x.csv"
    t_file = tmp_path / "force_t.csv"
    force_file = tmp_path / "force.csv"
    np.savetxt(x_file, np.array([0.0, 1.0]), delimiter=",")
    np.savetxt(t_file, np.array([0.0, 1.0]), delimiter=",")
    np.savetxt(force_file, np.array([[1.0, 2.0], [3.0, 4.0]]), delimiter=",")
    params.Models = {
        "Twist": {
            "boundary": "right",
            "plane": "xy",
            "rate": 0.5,
            "Dummy": 10.0,
        },
        "FileLoad": {
            "boundary": "right",
            "field": "u_x",
            "coordinate": 1,
            "file_x": str(x_file),
            "file_t": str(t_file),
            "file_F": str(force_file),
        },
        "Load": {"boundary": "right", "field": "u_x", "value": 2.0},
        "Flux": {
            "boundary": "left",
            "Species": ["Na"],
            "flux": [1.2],
        },
        "Friction": {
            "boundary": "bottom",
            "FrictionModel": "Linear",
        },
        "Ocean": {"boundary": "right"},
    }

    mesh = Mesh(params)
    mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 2)
    mesh.Add_Vector_Field("uOld", "Old Displacement", "CG", 1, 2)
    mesh.Add_Field("Na", "Sodium", "CG", 1)
    mesh.Add_Vector_TestFunction("u", 2, 0)
    mesh.Add_TestFunction("Na", 1)
    mesh.finalize_test_functions()
    models = {
        "torsion": AppliedTorsion("Twist", params, mesh),
        "file": ExternalForceFromFile("FileLoad", params, mesh),
        "force": ExternalForce("Load", params, mesh),
        "flux": PrescribedFlux("Flux", params, mesh),
        "basal": BasalFriction("Friction", params, mesh),
        "ocean": OceanPressure("Ocean", params, mesh),
    }
    models["torsion"].cos_omega.value = 0.6
    models["torsion"].sin_omega.value = 0.8
    models["flux"].u_target.value = 1.5
    old_force_function = models["file"]._force_func
    interpolant = models["file"]._interpolant

    _install_one_refinement(mesh)
    for model in models.values():
        model.Post_Remesh()

    return SimpleNamespace(
        mesh=mesh,
        models=models,
        old_force_function=old_force_function,
        interpolant=interpolant,
    )


def test_torsion_constants_use_new_mesh_and_preserve_angle(refined_boundary_models):
    mesh = refined_boundary_models.mesh
    model = refined_boundary_models.models["torsion"]
    for constant, value in (
        (model.cos_omega, 0.6),
        (model.sin_omega, 0.8),
    ):
        _assert_constant_on_live_mesh(constant, mesh)
        np.testing.assert_allclose(constant.value, value)

    K_form, F_form = model.assemble_KF(model.step)
    assert mesh.Compile_Form(K_form) is not None
    assert mesh.Compile_Form(F_form) is not None


def test_external_force_file_helper_uses_new_space_and_retains_data(
    refined_boundary_models,
):
    mesh = refined_boundary_models.mesh
    model = refined_boundary_models.models["file"]
    assert model._force_func is not refined_boundary_models.old_force_function
    assert model._force_func.function_space.mesh is mesh.mesh
    assert model._interpolant is refined_boundary_models.interpolant

    K_form, F_form = model.assemble_KF(model.step)
    assert K_form is None
    assert mesh.Compile_Form(F_form) is not None


def test_external_force_forms_use_the_refined_mesh(refined_boundary_models):
    mesh = refined_boundary_models.mesh
    model = refined_boundary_models.models["force"]

    K_form, F_form = model.assemble_KF(model.step)
    assert K_form is None
    assert mesh.Compile_Form(F_form) is not None


def test_prescribed_flux_refreshes_constant_and_form(refined_boundary_models):
    mesh = refined_boundary_models.mesh
    model = refined_boundary_models.models["flux"]
    _assert_constant_on_live_mesh(model.u_target, mesh)
    np.testing.assert_allclose(model.u_target.value, 1.5)

    _, F_form = model.assemble_KF(model.step)
    assert mesh.Compile_Form(F_form) is not None


@pytest.mark.parametrize("name", ["basal", "ocean"])
def test_glaciology_boundary_forms_use_the_refined_mesh(
    refined_boundary_models,
    name,
):
    mesh = refined_boundary_models.mesh
    model = refined_boundary_models.models[name]

    K_form, F_form = model.assemble_KF(model.step)
    if K_form is not None:
        assert mesh.Compile_Form(K_form) is not None
    assert mesh.Compile_Form(F_form) is not None
