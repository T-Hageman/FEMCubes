"""Tests for axisymmetric (r-z) meshes: radius-weighted integrals and model support."""

import numpy as np
import pytest
from dolfinx import fem

from femcubes.core.base_model import BaseModel
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.mesh_generators.plate_with_defect import Sample_Parameters


@pytest.fixture
def disc_mesh():
    params = Sample_Parameters()
    params.Axisymmetric = True
    return Mesh(params)


def _sum(mesh, values):
    return mesh.mesh.comm.allreduce(float(np.sum(values)))


class TestAxisymmetricIntegrals:
    def test_plane_meshes_are_not_axisymmetric(self):
        mesh = Mesh(Sample_Parameters())

        assert not mesh.axisymmetric
        assert mesh.Get_Integration_Weight() == 1

    def test_domain_integrals_carry_the_radius(self, disc_mesh):
        params = disc_mesh.params
        one = fem.Constant(disc_mesh.mesh, 1.0)

        volume = fem.assemble_scalar(fem.form(one * disc_mesh.Get_Domain_Integrator()))

        assert np.isclose(disc_mesh.mesh.comm.allreduce(volume), params.W**2 * params.H / 2.0, rtol=1e-10)

    def test_boundary_integrals_carry_the_radius(self, disc_mesh):
        params = disc_mesh.params
        one = fem.Constant(disc_mesh.mesh, 1.0)
        ds = disc_mesh.get_Boundary_Integrator()

        metal = fem.assemble_scalar(fem.form(one * ds(disc_mesh.get_boundary_tags("metal"))))
        defect = fem.assemble_scalar(fem.form(one * ds(disc_mesh.get_boundary_tags("metal:defect"))))

        assert np.isclose(disc_mesh.mesh.comm.allreduce(metal), params.W**2 / 2.0, rtol=1e-10)
        assert np.isclose(disc_mesh.mesh.comm.allreduce(defect), params.R_defect**2 / 2.0, rtol=1e-10)

    @pytest.mark.parametrize("degree", [1, 2])
    def test_lumped_weights_sum_to_the_radius_weighted_volume(self, disc_mesh, degree):
        params = disc_mesh.params
        disc_mesh.Add_Field("c", "c", "CG", degree)
        disc_mesh.Add_TestFunction("c", step=0)
        disc_mesh.finalize_test_functions()

        weights, _dofs, _vals = disc_mesh.Get_Weights_Dofs(["c"], 0)

        assert np.all(weights > 0.0)
        assert np.isclose(_sum(disc_mesh, weights), params.W**2 * params.H / 2.0, rtol=1e-10)

    @pytest.mark.parametrize("degree", [1, 2])
    def test_interface_weights_are_radius_weighted_and_positive_on_the_axis(self, disc_mesh, degree):
        params = disc_mesh.params
        for name, domain in (("c", "domain"), ("theta", "metal")):
            disc_mesh.Add_Field(name, name, "CG", degree, domain=domain)
            disc_mesh.Add_TestFunction(name, step=0)
        disc_mesh.finalize_test_functions()

        weights, dofs, _vals = disc_mesh.Get_Interface_Weights_Dofs("metal", ["c", "theta"], 0)
        r = disc_mesh.Get_Field("c").function_space.tabulate_dof_coordinates()[:, 0] if degree == 1 else None

        assert np.all(weights > 0.0)
        assert np.isclose(_sum(disc_mesh, weights), params.W**2 / 2.0, rtol=1e-10)
        if r is not None:
            # P1: the lumped weights also reproduce the first radial moment, int r^2 dr = W^3 / 3,
            # up to the trapezoidal-rule error of the metal discretisation.
            moment = _sum(disc_mesh, weights * r[dofs["c"]])
            assert np.isclose(moment, params.W**3 / 3.0, rtol=1e-2)


    def test_residual_rows_are_scaled_by_the_inverse_mean_radius(self, disc_mesh):
        params = disc_mesh.params
        disc_mesh.Add_Field("c", "c", "CG", 1)
        disc_mesh.Add_Global_Field("E_m", "metal potential")
        for name in ("c", "E_m"):
            disc_mesh.Add_TestFunction(name, step=0)
        disc_mesh.finalize_test_functions()

        scale = disc_mesh.Get_Residual_Row_Scale(0)
        offsets = disc_mesh.get_step_block_offsets(0)
        x = disc_mesh.Get_Field("c").function_space.tabulate_dof_coordinates()[: offsets[1], 0]

        assert scale.size == offsets[-1]
        # the mean radius of a nodal basis function lies within one element of its node, and is positive on the axis
        assert np.all(np.abs(1.0 / scale[: offsets[1]] - x) <= params.dx)
        assert np.all(scale[: offsets[1]] > 0.0)
        # the global row: the mean radius of the domain, W / 2
        assert np.allclose(scale[offsets[1]:], 2.0 / params.W)

    def test_plane_meshes_have_no_residual_row_scale(self):
        mesh = Mesh(Sample_Parameters())
        mesh.Add_Field("c", "c", "CG", 1)
        mesh.Add_TestFunction("c", step=0)
        mesh.finalize_test_functions()

        assert mesh.Get_Residual_Row_Scale(0) is None


class _PlanarOnly(BaseModel):
    MODEL_TYPE = "Test:PlanarOnly"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)


class TestAxisymmetricValidation:
    def test_axisymmetric_meshes_must_be_two_dimensional(self):
        params = Sample_Parameters()
        params.dim = 3
        params.Axisymmetric = True

        with pytest.raises(ValueError, match="2-D"):
            Mesh(params)

    def test_models_without_axisymmetric_support_are_rejected(self, disc_mesh):
        params = disc_mesh.params
        params.Models = {"planar": {"Type": "Test:PlanarOnly"}}
        params.ModelNames = ["planar"]
        params.dt = 1.0

        with pytest.raises(ValueError, match="does not support axisymmetric"):
            Physics(disc_mesh, params)
