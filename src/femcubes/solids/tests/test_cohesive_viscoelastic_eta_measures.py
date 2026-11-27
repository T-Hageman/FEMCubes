"""Global measures of the cohesive fracture strain written by Cohesive_ViscoElastic."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from femcubes.solids.tests.test_model_branch_paths import _build_cohesive_viscoelastic_model


def _set_eta_norm(mesh, value, cell_mask=None):
    """Set ``etaNorm`` at every integration point, or only in the cells of ``cell_mask``."""
    field = mesh.Get_Field("etaNorm")
    if cell_mask is None:
        field.x.array[:] = value
    else:
        dofmap = field.function_space.dofmap
        for cell in np.flatnonzero(cell_mask):
            field.x.array[dofmap.cell_dofs(cell)] = value
    field.x.scatter_forward()


def _domain_area(mesh, cell_mask=None):
    domain = mesh.Get_Field("etaNorm").function_space.mesh
    if cell_mask is None:
        indicator = 1.0
    else:
        V0 = fem.functionspace(domain, ("DG", 0))
        indicator = fem.Function(V0)
        indicator.x.array[:] = 0.0
        for cell in np.flatnonzero(cell_mask):
            indicator.x.array[V0.dofmap.cell_dofs(cell)] = 1.0
        indicator.x.scatter_forward()
    local = fem.assemble_scalar(fem.form(indicator * ufl.dx(domain=domain)))
    return domain.comm.allreduce(local, op=MPI.SUM)


def _left_half_cells(mesh):
    domain = mesh.Get_Field("etaNorm").function_space.mesh
    tdim = domain.topology.dim
    n_local = domain.topology.index_map(tdim).size_local
    x = domain.geometry.x
    x_mid = 0.5 * (x[:, 0].min() + x[:, 0].max())
    domain.topology.create_connectivity(tdim, 0)
    cell_to_vertex = domain.topology.connectivity(tdim, 0)
    centres = np.array([x[cell_to_vertex.links(c), 0].mean() for c in range(n_local)])
    return centres < x_mid


def test_eta_measures_are_registered_and_start_at_zero():
    params, _, model = _build_cohesive_viscoelastic_model()
    assert params.Global_Measures["Eta_Max"] == 0.0
    assert params.Global_Measures["Eta_Active_Area"] == 0.0
    model.Update_Global_Measures()
    assert params.Global_Measures["Eta_Max"] == 0.0
    assert params.Global_Measures["Eta_Active_Area"] == 0.0


def test_eta_measures_follow_the_fracture_strain_field():
    params, mesh, model = _build_cohesive_viscoelastic_model()

    _set_eta_norm(mesh, 0.3)
    model.Update_Global_Measures()
    assert np.isclose(params.Global_Measures["Eta_Max"], 0.3)
    assert np.isclose(params.Global_Measures["Eta_Active_Area"], _domain_area(mesh))

    # Values below the activation threshold count towards the maximum but not the area.
    _set_eta_norm(mesh, 1.0e-9)
    model.Update_Global_Measures()
    assert np.isclose(params.Global_Measures["Eta_Max"], 1.0e-9)
    assert params.Global_Measures["Eta_Active_Area"] == 0.0


def test_eta_active_area_counts_only_the_active_cells():
    params, mesh, model = _build_cohesive_viscoelastic_model()
    left = _left_half_cells(mesh)
    _set_eta_norm(mesh, 0.0)
    _set_eta_norm(mesh, 0.05, cell_mask=left)
    model.Update_Global_Measures()
    assert np.isclose(params.Global_Measures["Eta_Max"], 0.05)
    expected = _domain_area(mesh, cell_mask=left)
    assert 0.0 < expected < _domain_area(mesh)
    assert np.isclose(params.Global_Measures["Eta_Active_Area"], expected)


def test_eta_active_threshold_option():
    params, mesh, model = _build_cohesive_viscoelastic_model()
    model.eta_active_threshold = 0.5
    model._eta_area_form = None
    _set_eta_norm(mesh, 0.3)
    model.Update_Global_Measures()
    assert np.isclose(params.Global_Measures["Eta_Max"], 0.3)
    assert params.Global_Measures["Eta_Active_Area"] == 0.0


def test_negative_eta_active_threshold_is_rejected():
    from femcubes import Mesh
    from femcubes.solids.cohesive_viscoelastic import Cohesive_ViscoElastic

    params, _, _ = _build_cohesive_viscoelastic_model()
    params.Models["Solid"]["Eta_Active_Threshold"] = -1.0
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", 1)
    with pytest.raises(ValueError, match="Eta_Active_Threshold"):
        Cohesive_ViscoElastic("Solid", params, mesh)
