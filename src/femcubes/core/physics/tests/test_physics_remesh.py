"""Integration tests for the physics lifecycle across a live remesh."""

from types import SimpleNamespace

import numpy as np
import pytest

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration
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
from femcubes.solids.cohesive_linear_elastic import Cohesive_LinearElastic
from femcubes.solids.cohesive_viscoelastic import Cohesive_ViscoElastic
from femcubes.solids.viscoelastic import ViscoElastic


def _build_cohesive_params():
    """Return a tiny cohesive viscoelastic/phase-field problem."""

    class Params(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 1.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        Quadrature_Degree = 2

        Global_Measures = {}
        Materials = {
            "Solid": {
                "Density": 1.0,
                "Youngs": 1.0e6,
                "Poisson": 0.25,
                "LE_Damping": 0.0,
                "Visc_Type": "Newtonian",
                "Viscosity": 1.0e6,
                "ell": 0.25,
                "f_t": 1.0e3,
                "f_s": 2.0e3,
                "Gc": 1.0,
                "PF_formulation": "AT2",
                "Inertia": False,
                "kappa": 1.0e-4,
            }
        }
        Materials["Solid"]["Shear"] = Materials["Solid"]["Youngs"] / (
            2.0 * (1.0 + Materials["Solid"]["Poisson"])
        )
        Materials["Solid"]["Bulk"] = Materials["Solid"]["Youngs"] / (
            3.0 * (1.0 - 2.0 * Materials["Solid"]["Poisson"])
        )
        Materials["Solid"]["Lame"] = (
            2.0
            * Materials["Solid"]["Shear"]
            * Materials["Solid"]["Poisson"]
            / (1.0 - 2.0 * Materials["Solid"]["Poisson"])
        )

        gravity = 0.0
        Plot_Fields = []
        Plot_Measures = []
        Solution_Steps = {"u": 0, "phasefield": 1}
        Field_Orders = {"u": 1, "phasefield": 1}

        ModelNames = ["Solid", "PhaseField", "BC_Left", "BC_Bottom", "Load"]
        Models = {
            "Solid": {
                "Type": "Solid:Cohesive_ViscoElastic",
                "Material": "Solid",
                "DamageModel": "PhaseField",
            },
            "PhaseField": {
                "Type": "Solid:PhaseField",
                "Material": "Solid",
                "CDF_field": "PsiField",
                "Irreversible_Method": "Hist",
            },
            "BC_Left": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "left",
                "field": "u_x",
                "value": 0.0,
            },
            "BC_Bottom": {
                "Type": "Boundary:PrescribedValue",
                "boundary": "bottom",
                "field": "u_y",
                "value": 0.0,
            },
            "Load": {
                "Type": "Boundary:DisplacementControl",
                "boundary": "right",
                "field": "u_x",
                "rate": 0.01,
                "Dummy": 1.0e5,
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


def _refine_live_mesh(mesh_wrapper):
    """Refine one owned cell and install all markers and registered fields."""

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


def test_cohesive_phase_field_physics_assembles_after_live_remesh():
    params = _build_cohesive_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    solid = physics.models["Solid"]
    load = physics.models["Load"]

    physics.Get_Kmat_Fvec(0)
    assert solid._strain_expr is not None
    assert solid._damage_expr is not None

    load.F_max = 8.0
    load.FOld = 4.0
    load.uOld = 0.2
    load.F0 = 12.0
    load.changed_tmax = True
    load.ArcLen = 0.75
    old_target = load.u_target
    target_value = np.asarray(old_target.value).copy()
    model_ids = {name: id(model) for name, model in physics.models.items()}
    global_measures = params.Global_Measures

    result = _refine_live_mesh(mesh)
    physics.Post_Remesh()

    assert result.installed
    assert result.epoch_after == 1
    assert solid._strain_expr is None
    assert solid._damage_expr is None
    assert load.u_target is not old_target
    np.testing.assert_allclose(load.u_target.value, target_value)
    assert load.u_target.ufl_domain().ufl_cargo() is mesh.mesh._cpp_object
    assert (load.F_max, load.FOld, load.uOld, load.F0) == (8.0, 4.0, 0.2, 12.0)
    assert load.changed_tmax
    assert load.ArcLen == 0.75
    assert {name: id(model) for name, model in physics.models.items()} == model_ids
    assert params.Global_Measures is global_measures
    assert physics._assembly_cache == {}
    assert physics.K_mat is None
    assert physics.F_vec is None

    K_phase, F_phase = physics.Get_Kmat_Fvec(1)
    K_solid, F_solid = physics.Get_Kmat_Fvec(0)

    assert np.isfinite(K_phase.norm())
    assert np.isfinite(F_phase.norm())
    assert np.isfinite(K_solid.norm())
    assert np.isfinite(F_solid.norm())
    assert solid._strain_expr is not None
    assert solid._damage_expr is not None
