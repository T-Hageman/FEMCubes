"""Tests for _MeshLumpedOperators functionality."""

import numpy as np
import pytest
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix
from ufl import dx, inner

from femcubes.core.params_template import ParamsTemplate


@pytest.fixture
def Mesh_Test_Params():
    """Create a Params instance configured for mesh lumping tests."""

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2

    return TestParams()


def _assemble_mass_row_sum(mesh, field_name: str, step: int) -> np.ndarray:
    """Assemble the consistent mass matrix row sums for one scalar or vector field."""
    test_function = mesh.Get_TestFunction(field_name, step=step)
    trial_function = mesh.Get_Trial_Function(field_name, step=step)
    integrand = (
        test_function * trial_function
        if len(test_function.ufl_shape) == 0
        else inner(test_function, trial_function)
    )

    mass_matrix = assemble_matrix(fem.form(integrand * dx))
    mass_matrix.assemble()

    row_sum = mass_matrix.createVecRight()
    mass_matrix.getRowSum(row_sum)

    with row_sum.localForm() as row_sum_local:
        return row_sum_local.array.copy()


class TestMeshLumpedOperators:
    """Test lumped-matrix helpers through the public Mesh API."""

    @pytest.fixture
    def mesh_with_single_scalar_step(self, Mesh_Test_Params):
        """Create a mesh with one scalar field registered for one solve step."""
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("c", "Concentration", "CG", 1)
        mesh.Add_TestFunction("c", step=0)
        mesh.finalize_test_functions()
        mesh.Constant_Field("c", 2.5)
        return mesh

    @pytest.fixture
    def mesh_with_mixed_scalar_step(self, Mesh_Test_Params):
        """Create a mesh with two scalar fields combined into one mixed step."""
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("c1", "Species 1", "CG", 1)
        mesh.Add_Field("c2", "Species 2", "CG", 1)
        mesh.Add_TestFunction("c1", step=0)
        mesh.Add_TestFunction("c2", step=0)
        mesh.finalize_test_functions()
        mesh.Constant_Field("c1", 1.25)
        mesh.Constant_Field("c2", -0.75)
        return mesh

    @pytest.fixture
    def mesh_with_old_fields(self, Mesh_Test_Params):
        """Create a mesh with standalone fields used as old-step storage."""
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("c1Old", "Species 1 Old", "CG", 1)
        mesh.Add_Field("c2Old", "Species 2 Old", "CG", 1)
        mesh.Add_Vector_Field("a", "Acceleration", "CG", 1, Mesh_Test_Params.dim)
        mesh.Constant_Field("c1Old", 0.5)
        mesh.Constant_Field("c2Old", -1.0)
        mesh.fields["a"].x.array[:] = np.arange(len(mesh.fields["a"].x.array), dtype=float)
        return mesh

    @pytest.fixture
    def mesh_with_single_vector_step(self, Mesh_Test_Params):
        """Create a mesh with one vector field registered for one solve step."""
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Vector_Field("u", "Displacement", "CG", 1, Mesh_Test_Params.dim)
        mesh.Add_Vector_TestFunction("u", Mesh_Test_Params.dim, step=0)
        mesh.finalize_test_functions()
        mesh.fields["u"].x.array[:] = np.arange(len(mesh.fields["u"].x.array), dtype=float)
        return mesh

    def test_get_weights_dofs_requires_at_least_one_field(self, mesh_with_single_scalar_step):
        """The public lumping helper should reject an empty field list."""
        mesh = mesh_with_single_scalar_step

        with pytest.raises(ValueError, match="At least one field name"):
            mesh.Get_Weights_Dofs([], step=0)

    def test_get_weights_dofs_matches_mass_row_sum_for_single_field(
        self, mesh_with_single_scalar_step
    ):
        """Lumped weights should equal the row sums of the consistent mass matrix."""
        mesh = mesh_with_single_scalar_step

        weights, dofs, vals = mesh.Get_Weights_Dofs(["c"], step=0)
        row_sum = _assemble_mass_row_sum(mesh, "c", step=0)

        assert dofs["c"].dtype == np.int32
        np.testing.assert_array_equal(dofs["c"], np.arange(len(weights), dtype=np.int32))
        np.testing.assert_allclose(weights, row_sum)
        np.testing.assert_allclose(vals["c"], 2.5)
        assert np.all(weights > 0.0)

    def test_get_weights_dofs_returns_mixed_field_slices_and_values(
        self, mesh_with_mixed_scalar_step
    ):
        """Mixed-step lumping should return per-field DOF slices and field values."""
        mesh = mesh_with_mixed_scalar_step

        weights, dofs, vals = mesh.Get_Weights_Dofs(["c1", "c2"], step=0)
        row_sum = _assemble_mass_row_sum(mesh, "c1", step=0)

        assert set(dofs) == {"c1", "c2"}
        assert set(vals) == {"c1", "c2"}
        assert len(dofs["c1"]) == len(weights)
        assert len(dofs["c2"]) == len(weights)
        assert np.intersect1d(dofs["c1"], dofs["c2"]).size == 0
        np.testing.assert_allclose(weights, row_sum[dofs["c1"]])
        np.testing.assert_allclose(vals["c1"], 1.25)
        np.testing.assert_allclose(vals["c2"], -0.75)

    def test_get_weights_dofs_matches_mass_row_sum_for_vector_field(
        self, mesh_with_single_vector_step
    ):
        """Vector-field lumping should return one weight per owned component DOF."""
        mesh = mesh_with_single_vector_step

        weights, dofs, vals = mesh.Get_Weights_Dofs(["u"], step=0)
        row_sum = _assemble_mass_row_sum(mesh, "u", step=0)

        assert dofs["u"].dtype == np.int32
        np.testing.assert_array_equal(dofs["u"], np.arange(len(weights), dtype=np.int32))
        np.testing.assert_allclose(weights, row_sum)
        np.testing.assert_allclose(vals["u"], mesh.fields["u"].x.array[: len(weights)])
        assert np.all(weights > 0.0)

    def test_get_old_for_lumped_returns_values_for_standalone_fields(self, mesh_with_old_fields):
        """Standalone old fields should be returned directly from their own storage."""
        mesh = mesh_with_old_fields

        vals_old = mesh.Get_Old_For_Lumped(["c1Old", "c2Old"], step=0)

        np.testing.assert_allclose(vals_old["c1Old"], 0.5)
        np.testing.assert_allclose(vals_old["c2Old"], -1.0)

    def test_get_old_for_lumped_returns_values_for_vector_fields(self, mesh_with_old_fields):
        """Vector old fields should be returned as flattened owned component values."""
        mesh = mesh_with_old_fields

        vals_old = mesh.Get_Old_For_Lumped(["a"], step=0)

        np.testing.assert_allclose(vals_old["a"], mesh.fields["a"].x.array)

    def test_get_old_for_lumped_returns_values_for_mixed_fields(self, mesh_with_mixed_scalar_step):
        """Mixed fields should be sliced back out of the shared mixed storage."""
        mesh = mesh_with_mixed_scalar_step

        vals_old = mesh.Get_Old_For_Lumped(["c1", "c2"], step=0)

        np.testing.assert_allclose(vals_old["c1"], 1.25)
        np.testing.assert_allclose(vals_old["c2"], -0.75)
