"""
Tests for _MeshDOFs functionality.

The main body of this module targets the public _MeshDOFs API. A small
white-box section remains near the end for hard-to-reach Copy_Field
branches that are not practical to trigger through the full Mesh object.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
from types import SimpleNamespace
from dolfinx import fem

from femcubes.core.params_template import ParamsTemplate
import femcubes.core.mesh._mesh_dofs as mesh_dofs_module
from femcubes.core.mesh._mesh_dofs import FieldDefinition, FieldOutputView, ProjectedFieldOutput, _MeshDOFs


@pytest.fixture
def Mesh_Test_Params():
    """Create a Params instance configured for Mesh testing."""

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2
        pass

    params = TestParams()
    return params


class _MeshDOFsHarnessBase:
    """Minimal base used only for white-box branch coverage."""

    def __init__(self, params):
        self.params = params


class _MeshDOFsHarness(_MeshDOFs, _MeshDOFsHarnessBase):
    """Small harness for edge cases that are awkward to reach from Mesh."""

    def __init__(self):
        self.mesh = None
        super().__init__(SimpleNamespace())


class _DummyArray:
    """Array holder mimicking the subset of the DOLFINx vector API used here."""

    def __init__(self, values):
        self.array = np.array(values, dtype=float)
        self.scatter_calls = 0

    def scatter_forward(self):
        self.scatter_calls += 1


class _DummyField:
    """Fake field used to test Copy_Field control flow."""

    def __init__(self, function_space, values):
        self.function_space = function_space
        self.x = _DummyArray(values)
        self.interpolated_with = None

    def interpolate(self, value):
        self.interpolated_with = value


class TestFieldManagement:
    """Test field creation and retrieval."""

    @pytest.fixture
    def mesh_with_params(self, Mesh_Test_Params):
        """Fixture to create a mesh for testing."""
        from femcubes.core.mesh.mesh import Mesh

        params = Mesh_Test_Params
        mesh = Mesh(params)
        return mesh

    def test_add_scalar_field(self, mesh_with_params):
        """Test adding a scalar field."""
        mesh = mesh_with_params

        mesh.Add_Field("temperature", "Temperature [K]", "CG", 1)
        mesh.Print_Fields()

        assert "temperature" in mesh.fields
        assert mesh.fields["temperature"] is not None
        assert isinstance(mesh.fields["temperature"], fem.Function)

    def test_add_vector_field(self, mesh_with_params):
        """Test adding a vector field."""
        mesh = mesh_with_params

        mesh.Add_Vector_Field("velocity", "Velocity [m/s]", "CG", 2, 2)

        assert "velocity" in mesh.fields
        assert mesh.fields["velocity"] is not None
        assert isinstance(mesh.fields["velocity"], fem.Function)
        assert "Vector" in mesh.field_type["velocity"]

    def test_add_quadrature_field(self, mesh_with_params):
        """Test adding a quadrature field."""
        mesh = mesh_with_params

        mesh.Add_Quadrature_Field("stress", "Stress at quadrature points [Pa]", (3, 3))

        assert "stress" in mesh.fields
        assert mesh.fields["stress"] is not None
        assert isinstance(mesh.fields["stress"], fem.Function)

    def test_add_quadrature_field_returns_existing(self, mesh_with_params):
        """Re-adding a quadrature field should return the existing object."""
        mesh = mesh_with_params

        original = mesh.Add_Quadrature_Field("stress", "Stress at quadrature points [Pa]", (3, 3))
        result = mesh.Add_Quadrature_Field("stress", "Ignored rename", (3, 3))

        assert result is original

    def test_add_quadrature_field_caches_spaces_by_degree_and_shape(self, mesh_with_params):
        """Quadrature-space caching should reuse identical degree/shape spaces only."""
        mesh = mesh_with_params

        stress = mesh.Add_Quadrature_Field("stress", "Stress", (3, 3))
        strain = mesh.Add_Quadrature_Field("strain", "Strain", (3, 3))
        damage = mesh.Add_Quadrature_Field("damage", "Damage", None)

        assert stress.function_space is strain.function_space
        assert stress.function_space is not damage.function_space
        assert len(mesh._quadrature_spaces) == 2
        assert not hasattr(mesh, "_Q_tensor")

    def test_get_field(self, mesh_with_params):
        """Test retrieving a field."""
        mesh = mesh_with_params
        mesh.Add_Field("pressure", "Pressure [Pa]", "CG", 1)

        retrieved_field = mesh.Get_Field("pressure")
        assert retrieved_field is mesh.fields["pressure"]

    def test_zero_field(self, mesh_with_params):
        """Test creating a field and setting it to zero."""
        mesh = mesh_with_params
        mesh.Add_Field("pressure", "Pressure [Pa]", "CG", 1)

        mesh.Zero_Field("pressure")
        np.testing.assert_array_almost_equal(mesh.fields["pressure"].x.array, 0.0)

    def test_constant_field(self, mesh_with_params):
        """Test creating a field and setting it to a constant."""
        mesh = mesh_with_params
        mesh.Add_Field("pressure", "Pressure [Pa]", "CG", 1)

        mesh.Constant_Field("pressure", 3.25)
        np.testing.assert_array_almost_equal(mesh.fields["pressure"].x.array, 3.25)

    def test_field_not_found(self, mesh_with_params):
        """Test that getting a non-existent field raises an error."""
        mesh = mesh_with_params

        with pytest.raises(ValueError, match="not found"):
            mesh.Get_Field("nonexistent_field")

    def test_field_order_storage(self, mesh_with_params):
        """Test that field order is stored correctly."""
        mesh = mesh_with_params

        mesh.Add_Field("field1", "Field 1", "CG", 1)
        mesh.Add_Field("field2", "Field 2", "CG", 2)
        mesh.Add_Field("field3", "Field 3", "DG", 0)

        assert mesh.field_order["field1"] == 1
        assert mesh.field_order["field2"] == 2
        assert mesh.field_order["field3"] == 0

    def test_field_type_storage(self, mesh_with_params):
        """Test that field type is stored correctly."""
        mesh = mesh_with_params

        mesh.Add_Field("scalar_cg", "Scalar CG", "CG", 1)
        mesh.Add_Field("scalar_dg", "Scalar DG", "DG", 0)
        mesh.Add_Vector_Field("vector_cg", "Vector CG", "CG", 1, 2)

        assert mesh.field_type["scalar_cg"] == "CG"
        assert mesh.field_type["scalar_dg"] == "DG"
        assert mesh.field_type["vector_cg"] == "CG Vector"

    def test_field_definitions_capture_scalar_and_vector_storage(self, mesh_with_params):
        """Every CG/DG registration should retain mesh-independent metadata."""
        mesh = mesh_with_params

        mesh.Add_Field("scalar_cg", "Scalar CG", "CG", 2)
        mesh.Add_Field("scalar_dg", "Scalar DG", "DG", 1)
        mesh.Add_Vector_Field("vector_cg", "Vector CG", "CG", 2, 2)
        mesh.Add_Vector_Field("vector_dg", "Vector DG", "DG", 1, 3)

        assert mesh.field_definitions["scalar_cg"] == FieldDefinition(
            name="scalar_cg",
            description="Scalar CG",
            family="CG",
            degree=2,
        )
        assert mesh.field_definitions["scalar_dg"] == FieldDefinition(
            name="scalar_dg",
            description="Scalar DG",
            family="DG",
            degree=1,
        )
        assert mesh.field_definitions["vector_cg"].value_shape == (2,)
        assert mesh.field_definitions["vector_dg"].value_shape == (3,)
        assert mesh.field_definitions["vector_dg"].family == "DG"

    def test_field_definitions_capture_scalar_and_tensor_quadrature_storage(self, mesh_with_params):
        """Quadrature definitions should retain degree and complete value shape."""
        mesh = mesh_with_params
        mesh.params.Quadrature_Degree = 3

        mesh.Add_Quadrature_Field("scalar_q", "Scalar IP", ())
        mesh.Add_Quadrature_Field("tangent_q", "Tangent IP", (6, 6))

        assert mesh.field_definitions["scalar_q"] == FieldDefinition(
            name="scalar_q",
            description="Scalar IP",
            family="Quadrature",
            degree=-1,
            value_shape=(),
            quadrature_degree=3,
        )
        assert mesh.field_definitions["tangent_q"].value_shape == (6, 6)
        assert mesh.field_definitions["tangent_q"].quadrature_degree == 3

    def test_compatible_field_reregistration_preserves_original_function_and_label(self, mesh_with_params):
        """A label change alone should retain the established registration behavior."""
        mesh = mesh_with_params

        original = mesh.Add_Vector_Field("u", "Displacement", "CG", 2, 2)
        repeated = mesh.Add_Vector_Field("u", "Ignored replacement label", "CG", 2, 2)

        assert repeated is original
        assert mesh.field_definitions["u"].description == "Displacement"
        assert original.name == "Displacement"

    def test_conflicting_field_reregistration_is_rejected(self, mesh_with_params):
        """Family, degree, shape, and quadrature-degree conflicts must be explicit."""
        mesh = mesh_with_params

        mesh.Add_Field("family_conflict", "Field", "CG", 1)
        with pytest.raises(ValueError, match="family_conflict.*incompatible"):
            mesh.Add_Field("family_conflict", "Field", "DG", 1)

        mesh.Add_Field("degree_conflict", "Field", "CG", 1)
        with pytest.raises(ValueError, match="degree_conflict.*incompatible"):
            mesh.Add_Field("degree_conflict", "Field", "CG", 2)

        mesh.Add_Vector_Field("shape_conflict", "Field", "CG", 1, 2)
        with pytest.raises(ValueError, match="shape_conflict.*incompatible"):
            mesh.Add_Vector_Field("shape_conflict", "Field", "CG", 1, 3)

        mesh.params.Quadrature_Degree = 2
        mesh.Add_Quadrature_Field("quadrature_conflict", "Field", ())
        mesh.params.Quadrature_Degree = 3
        with pytest.raises(ValueError, match="quadrature_conflict.*incompatible"):
            mesh.Add_Quadrature_Field("quadrature_conflict", "Field", ())

    @pytest.mark.parametrize(
        ("field_name", "register"),
        [
            ("scalar_cg", lambda mesh: mesh.Add_Field("scalar_cg", "Scalar CG", "CG", 2)),
            ("vector_dg", lambda mesh: mesh.Add_Vector_Field("vector_dg", "Vector DG", "DG", 1, 2)),
            ("tensor_q", lambda mesh: mesh.Add_Quadrature_Field("tensor_q", "Tensor IP", (6, 6))),
        ],
    )
    def test_field_definition_recreates_equivalent_function_on_another_mesh(
        self,
        mesh_with_params,
        Mesh_Test_Params,
        field_name,
        register,
    ):
        """Detached recreation should preserve the element contract on a new mesh."""
        from femcubes.core.mesh.mesh import Mesh

        source_mesh = mesh_with_params
        source_mesh.params.Quadrature_Degree = 3
        source = register(source_mesh)
        target_mesh = Mesh(Mesh_Test_Params)
        definition = source_mesh.field_definitions[field_name]

        recreated = source_mesh._create_function_from_definition(definition, target_mesh.mesh)

        assert recreated.function_space.mesh is target_mesh.mesh
        assert recreated.name == definition.description
        assert tuple(recreated.function_space.element.value_shape) == tuple(source.function_space.element.value_shape)
        if definition.family == "Quadrature":
            assert recreated.function_space.element.interpolation_points.shape == source.function_space.element.interpolation_points.shape
        else:
            assert recreated.function_space.element.basix_element.degree == source.function_space.element.basix_element.degree
        assert field_name not in target_mesh.fields


class TestTestFunctions:
    """Test test function creation and management."""

    @pytest.fixture
    def mesh_with_fields(self, Mesh_Test_Params):
        """Fixture to create a mesh with fields for testing."""
        from femcubes.core.mesh.mesh import Mesh

        params = Mesh_Test_Params
        mesh = Mesh(params)
        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_Vector_Field("v", "Velocity", "CG", 2, 2)

        return mesh

    def test_add_scalar_test_function(self, mesh_with_fields):
        """Test adding a test function for a scalar field."""
        mesh = mesh_with_fields

        mesh.Add_TestFunction("p", step=0)

        test_key = "p_test_step0"
        assert test_key in mesh.test_functions
        assert mesh.test_functions[test_key] is not None

    def test_add_vector_test_function(self, mesh_with_fields):
        """Test adding a test function for a vector field."""
        mesh = mesh_with_fields

        mesh.Add_Vector_TestFunction("v", nDim=2, step=0)

        test_key = "v_test_step0"
        assert test_key in mesh.test_functions
        assert mesh.test_functions[test_key] is not None

    def test_get_test_function(self, mesh_with_fields):
        """Test retrieving a test function."""
        mesh = mesh_with_fields

        original = mesh.Add_TestFunction("p", step=0)
        retrieved = mesh.Get_TestFunction("p", step=0)

        assert retrieved is original

    def test_test_function_field_tracking(self, mesh_with_fields):
        """Test that fields are tracked per step."""
        mesh = mesh_with_fields

        mesh.Add_TestFunction("p", step=0)
        mesh.Add_Vector_TestFunction("v", nDim=2, step=0)

        assert 0 in mesh.step_fields
        assert "p" in mesh.step_fields[0]
        assert "v" in mesh.step_fields[0]

    def test_wrong_test_function_type(self, mesh_with_fields):
        """Test that using the wrong test-function helper raises."""
        mesh = mesh_with_fields

        with pytest.raises(ValueError, match="vector field"):
            mesh.Add_TestFunction("v", step=0)

        with pytest.raises(ValueError, match="scalar field"):
            mesh.Add_Vector_TestFunction("p", nDim=2, step=0)


class TestFieldCopy:
    """Test field copying functionality."""

    @pytest.fixture
    def mesh_with_fields(self, Mesh_Test_Params):
        """Fixture to create a mesh with fields."""
        from femcubes.core.mesh.mesh import Mesh

        params = Mesh_Test_Params
        mesh = Mesh(params)

        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_Field("p_old", "Old Pressure", "CG", 1)
        mesh.Add_Vector_Field("v", "Velocity", "CG", 2, 2)
        mesh.Add_Vector_Field("v_old", "Old Velocity", "CG", 2, 2)

        return mesh

    def test_copy_scalar_field(self, mesh_with_fields):
        """Test copying a scalar field."""
        mesh = mesh_with_fields

        mesh.fields["p"].x.array[:] = 100.0
        mesh.fields["p_old"].x.array[:] = 0.0

        mesh.Copy_Field("p", "p_old")

        np.testing.assert_array_almost_equal(
            mesh.fields["p_old"].x.array,
            mesh.fields["p"].x.array,
        )

    def test_copy_vector_field(self, mesh_with_fields):
        """Test copying a vector field."""
        mesh = mesh_with_fields

        mesh.fields["v"].x.array[:] = 5.0
        mesh.fields["v_old"].x.array[:] = 0.0

        mesh.Copy_Field("v", "v_old")

        np.testing.assert_array_almost_equal(
            mesh.fields["v_old"].x.array,
            mesh.fields["v"].x.array,
        )


class TestMixedSpaces:
    """Test mixed function space functionality."""

    @pytest.fixture
    def mesh_for_mixed_space(self, Mesh_Test_Params):
        """Fixture to create a mesh ready for mixed space testing."""
        from femcubes.core.mesh.mesh import Mesh

        params = Mesh_Test_Params
        mesh = Mesh(params)

        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_Vector_Field("v", "Velocity", "CG", 2, 2)
        mesh.Add_TestFunction("p", step=0)
        mesh.Add_Vector_TestFunction("v", nDim=2, step=0)

        return mesh

    def test_build_mixed_space(self, mesh_for_mixed_space):
        """Multi-field steps get a block layout with one block per field."""
        mesh = mesh_for_mixed_space
        mesh.finalize_test_functions()

        block_layout = mesh.mixed_spaces[0]
        assert block_layout is not None
        assert block_layout["field_names"] == ["p", "v"]
        assert block_layout["function_spaces"][0] is mesh.fields["p"].function_space
        assert block_layout["function_spaces"][1] is mesh.fields["v"].function_space

    def test_block_test_and_trial_functions_belong_to_their_field_blocks(self, mesh_for_mixed_space):
        """Test/trial functions of a block step are the per-field block arguments."""
        mesh = mesh_for_mixed_space
        mesh.finalize_test_functions()

        p_test = mesh.Get_TestFunction("p", 0)
        v_trial = mesh.Get_Trial_Function("v", 0)
        assert p_test.part() == 0 and p_test.ufl_function_space() == mesh.fields["p"].function_space
        assert v_trial.part() == 1 and v_trial.ufl_function_space() == mesh.fields["v"].function_space
        assert v_trial.ufl_shape == (2,)

    def test_build_mixed_space_preserves_field_registration_order(self, Mesh_Test_Params):
        """Block ordering should follow registration order, not sorting."""
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Vector_Field("zeta", "Velocity", "CG", 1, 2)
        mesh.Add_Field("alpha", "Pressure", "CG", 1)
        mesh.Add_Vector_TestFunction("zeta", nDim=2, step=0)
        mesh.Add_TestFunction("alpha", step=0)

        mesh.finalize_test_functions()

        assert mesh.mixed_spaces[0]["field_names"] == ["zeta", "alpha"]
        zeta_size = mesh.fields["zeta"].function_space.dofmap.index_map.size_local * 2
        alpha_size = mesh.fields["alpha"].function_space.dofmap.index_map.size_local
        assert mesh.get_step_block_offsets(0) == [0, zeta_size, zeta_size + alpha_size]

    def test_get_field_from_mixed_space(self, mesh_for_mixed_space):
        """Fields of a block step are returned as their own functions."""
        mesh = mesh_for_mixed_space
        mesh.finalize_test_functions()

        assert mesh.Get_Field("p", step=0) is mesh.fields["p"]
        assert mesh.Get_Field("v", step=0) is mesh.fields["v"]

    def test_set_zero_mixed_space_field(self, mesh_for_mixed_space):
        """Zeroing one field of a block step leaves the other fields untouched."""
        mesh = mesh_for_mixed_space
        mesh.finalize_test_functions()
        mesh.Constant_Field("p", 10.0)
        mesh.Constant_Field("v", 10.0)

        mesh.Zero_Field("p")

        np.testing.assert_array_almost_equal(mesh.fields["p"].x.array, 0.0)
        np.testing.assert_array_almost_equal(mesh.fields["v"].x.array, 10.0)

    def test_set_constant_mixed_space_vector_field(self, mesh_for_mixed_space):
        """Setting a vector field of a block step to a constant only changes that field."""
        mesh = mesh_for_mixed_space
        mesh.finalize_test_functions()

        mesh.Constant_Field("v", 4.5)

        np.testing.assert_array_almost_equal(mesh.fields["p"].x.array, 0.0)
        np.testing.assert_array_almost_equal(mesh.fields["v"].x.array, 4.5)

    def test_single_field_no_mixed_space(self, mesh_for_mixed_space):
        """Test that a single field does not create a mixed space."""
        mesh = mesh_for_mixed_space

        mesh.Add_Field("temp", "Temperature", "CG", 1)
        mesh.Add_TestFunction("temp", step=1)
        mesh.finalize_test_functions()

        assert 1 in mesh.mixed_spaces
        assert mesh.mixed_spaces[1] is None


class TestStepVectorUpdates:
    """Test step-wise solution updates from linear solver vectors."""

    def test_update_step_from_vector_single_field(self, Mesh_Test_Params):
        """Single-field steps should update the field directly."""
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_TestFunction("p", step=0)

        du = mesh.fields["p"].x.petsc_vec.duplicate()
        with du.localForm() as loc_du:
            loc_du.array[:] = 1.75

        mesh.update_step_from_vector(du, step=0)

        np.testing.assert_allclose(mesh.fields["p"].x.array, 1.75)
        du.destroy()

    def test_update_step_from_vector_mixed_space_updates_each_field_from_its_block(self, Mesh_Test_Params):
        """Block-step increments are added to each field from its own block."""
        from dolfinx.fem.petsc import create_vector

        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
        mesh.Add_TestFunction("p", step=0)
        mesh.Add_Vector_TestFunction("v", nDim=2, step=0)
        mesh.finalize_test_functions()

        du = create_vector(mesh.mixed_spaces[0]["function_spaces"], kind="mpi")
        offsets = mesh.get_step_block_offsets(0)
        with du.localForm() as loc_du:
            loc_du.array[offsets[0]:offsets[1]] = 1.25
            loc_du.array[offsets[1]:offsets[2]] = -0.5

        mesh.update_step_from_vector(du, step=0)
        du.destroy()

        np.testing.assert_allclose(mesh.fields["p"].x.array, 1.25)
        np.testing.assert_allclose(mesh.fields["v"].x.array, -0.5)

    def test_update_step_from_vector_requires_mixed_space_for_multi_field_step(self, Mesh_Test_Params):
        """Multiple fields in one step must be finalized into a block layout first."""
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
        mesh.Add_TestFunction("p", step=0)
        mesh.Add_Vector_TestFunction("v", nDim=2, step=0)

        du = mesh.fields["p"].x.petsc_vec.duplicate()
        with pytest.raises(ValueError, match="no block layout"):
            mesh.update_step_from_vector(du, step=0)
        du.destroy()

    def test_update_step_from_vector_requires_registered_step(self, Mesh_Test_Params):
        """Updating an unknown step should fail through the public API."""
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_TestFunction("p", step=0)

        du = mesh.fields["p"].x.petsc_vec.duplicate()
        with pytest.raises(ValueError, match="No fields registered for step 99"):
            mesh.update_step_from_vector(du, step=99)
        du.destroy()


class TestMeshDOFsPublicApiEdgeCases:
    """Additional coverage for public _MeshDOFs API behavior."""

    @pytest.fixture
    def mesh_dofs_fixture(self, Mesh_Test_Params):
        """Mesh configured to exercise advanced DOF utilities."""
        from femcubes.core.mesh.mesh import Mesh

        params = Mesh_Test_Params
        mesh = Mesh(params)
        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
        mesh.Add_Field("temp", "Temperature", "CG", 1)
        mesh.Add_Field("p_copy", "Pressure Copy", "CG", 1)
        mesh.Add_Vector_Field("v_copy", "Velocity Copy", "CG", 1, 2)

        mesh.Add_TestFunction("p", step=0)
        mesh.Add_Vector_TestFunction("v", nDim=2, step=0)
        mesh.Add_TestFunction("temp", step=2)

        mesh.finalize_test_functions()
        return mesh

    def test_get_field_handles_plain_steps(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        assert mesh.Get_Field("temp", step=2) is mesh.fields["temp"]
        assert mesh.Get_Field("temp") is mesh.fields["temp"]

    def test_get_field_for_output_plain_when_mixed_present(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        output_view = mesh.Get_Field_For_Output("temp")

        assert isinstance(output_view, FieldOutputView)
        assert output_view.field is mesh.fields["temp"]
        assert output_view.comp_count == 1

    def test_get_field_for_output_returns_own_storage_of_block_step_fields(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        output_view = mesh.Get_Field_For_Output("v")

        assert isinstance(output_view, FieldOutputView)
        assert output_view.field is mesh.fields["v"]
        assert output_view.comp_count == 2

    def test_get_field_for_output_missing(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        with pytest.raises(ValueError, match="Field 'ghost' not found"):
            mesh.Get_Field_For_Output("ghost")

    def test_get_field_space_for_bc_variants(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        space_p, idx_p = mesh.Get_Field_Space_For_BC("p")
        assert space_p is mesh.fields["p"].function_space and idx_p is None
        space_v, idx_v = mesh.Get_Field_Space_For_BC("v", component=1)
        assert space_v is mesh.fields["v"].function_space and idx_v == 1
        plain_space, component = mesh.Get_Field_Space_For_BC("temp")
        assert component is None
        assert plain_space == mesh.fields["temp"].function_space

    def test_get_field_space_for_bc_missing(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        with pytest.raises(ValueError, match="not found"):
            mesh.Get_Field_Space_For_BC("unknown")

    def test_prepare_field_for_output_returns_projected_scalar_and_vector_data(
        self, mesh_dofs_fixture
    ):
        mesh = mesh_dofs_fixture
        mesh.Constant_Field("temp", 2.0)
        mesh.Constant_Field("v", 3.0)

        scalar_output = mesh.Prepare_Field_For_Output("temp")
        vector_output = mesh.Prepare_Field_For_Output("v_x")

        assert isinstance(scalar_output, ProjectedFieldOutput)
        assert scalar_output.field_spec == "temp"
        assert scalar_output.title == "Temperature"
        np.testing.assert_allclose(scalar_output.values, 2.0)

        assert isinstance(vector_output, ProjectedFieldOutput)
        assert vector_output.field_spec == "v_x"
        assert "X component" in vector_output.title
        np.testing.assert_allclose(vector_output.values, 3.0)

    def test_prepare_field_for_output_treats_non_component_underscores_as_field_names(
        self, Mesh_Test_Params
    ):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("phasefield_Hist", "Phasefield History", "CG", 1)
        mesh.Constant_Field("phasefield_Hist", 4.0)

        output = mesh.Prepare_Field_For_Output("phasefield_Hist")

        assert output is not None
        assert output.field_name == "phasefield_Hist"
        assert output.title == "Phasefield History"
        np.testing.assert_allclose(output.values, 4.0)

    def test_prepare_field_for_output_uses_cell_local_expression_for_dg_fields(
        self, Mesh_Test_Params
    ):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("height", "Thickness", "DG", 1)
        mesh.fields["height"].x.array[:] = 3.25

        output = mesh.Prepare_Field_For_Output("height")

        assert output is not None
        assert output.field_name == "height"
        assert output.title == "Thickness"
        np.testing.assert_allclose(output.values, 3.25)

    def test_prepare_field_for_output_uses_dg0_for_quadrature_fields(
        self, mesh_dofs_fixture
    ):
        """Quadrature fields should be exported as cell-wise DG0 quantities."""
        mesh = mesh_dofs_fixture
        stress_q = mesh.Add_Quadrature_Field("stress_q", "Stress Q")
        stress_q.x.array[:] = 2.5

        output = mesh.Prepare_Field_For_Output("stress_q")

        assert output is not None
        assert output.function_space is mesh._dg_spaces[0]
        np.testing.assert_allclose(output.values, 2.5)

    def test_get_interior_facet_integrator_returns_dS_measure(self, mesh_dofs_fixture):
        measure = mesh_dofs_fixture.Get_Interior_Facet_Integrator()

        assert measure.integral_type() == "interior_facet"

    def test_field_output_view_rejects_out_of_range_component(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        output_view = mesh.Get_Field_For_Output("temp")

        with pytest.raises(IndexError, match="out of range"):
            output_view.collapse_component(1)

    def test_prepare_field_for_output_tensor_edge_cases(self, Mesh_Test_Params, monkeypatch):
        from femcubes.core.mesh.mesh import Mesh
        from dolfinx import fem

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("temperature", "Temperature", "CG", 1)

        assert mesh.Prepare_Field_For_Output("ghost_xx") is None
        assert mesh.Prepare_Field_For_Output("temperature_xx") is None

        tensor_element = mesh._create_element("CG", 1, shape=(2, 2))
        tensor_space = fem.functionspace(mesh.mesh, tensor_element)
        sigma = fem.Function(tensor_space, name="Stress")
        sigma.x.array[:] = np.linspace(0.0, 1.0, sigma.x.array.size)
        mesh.fields["sigma"] = sigma
        mesh.field_order["sigma"] = 1
        mesh.field_type["sigma"] = "CG Tensor"

        assert mesh.Prepare_Field_For_Output("sigma_xz") is None

        original_expression = mesh_dofs_module.fem.Expression
        monkeypatch.setattr(
            mesh_dofs_module.fem,
            "Expression",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        try:
            assert mesh.Prepare_Field_For_Output("sigma_xx") is None
        finally:
            monkeypatch.setattr(mesh_dofs_module.fem, "Expression", original_expression)

    def test_get_trial_function_mixed_scalar_and_vector(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        mixed_info = mesh.mixed_spaces[0]

        p_trial = mesh.Get_Trial_Function("p", step=0)
        assert p_trial == mixed_info["trial_functions"][0]
        assert mesh.Get_Trial_Function("p", step=0) is p_trial

        v_trial = mesh.Get_Trial_Function("v", step=0)
        assert v_trial.ufl_shape == (2,)
        assert v_trial == mixed_info["trial_functions"][1]

    def test_get_trial_function_plain_field_and_missing(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture

        temp_trial = mesh.Get_Trial_Function("temp", step=2)
        assert mesh.trial_functions["temp_trial_step2"] is temp_trial
        assert temp_trial.ufl_shape == ()

        with pytest.raises(ValueError, match="Field 'ghost' not found"):
            mesh.Get_Trial_Function("ghost", step=0)

    def test_copy_field_from_mixed_scalar_and_vector(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        mesh.fields["p"].x.array[:] = np.linspace(0.0, 1.0, mesh.fields["p"].x.array.size)
        mesh.fields["v"].x.array[:] = np.linspace(0.0, 2.0, mesh.fields["v"].x.array.size)
        mesh.fields["p_copy"].x.array[:] = 0.0
        mesh.fields["v_copy"].x.array[:] = 0.0

        mesh.Copy_Field("p", "p_copy")
        mesh.Copy_Field("v", "v_copy")

        np.testing.assert_allclose(mesh.fields["p_copy"].x.array, mesh.fields["p"].x.array)
        np.testing.assert_allclose(mesh.fields["v_copy"].x.array, mesh.fields["v"].x.array)

    def test_copy_field_missing_destination(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        with pytest.raises(ValueError, match="Destination field"):
            mesh.Copy_Field("p", "does_not_exist")

    def test_copy_field_missing_source(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        with pytest.raises(ValueError, match="Source field"):
            mesh.Copy_Field("does_not_exist", "p")

    def test_get_test_function_missing(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        with pytest.raises(ValueError, match="not found"):
            mesh.Get_TestFunction("temp", step=99)

    def test_add_field_invalid_space(self, Mesh_Test_Params):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        with pytest.raises(ValueError, match="Unsupported function space"):
            mesh.Add_Field("bad", "Bad Space", "Foo", 1)

    def test_add_field_returns_existing(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        original = mesh.fields["p"]
        result = mesh.Add_Field("p", "Pressure", "CG", 1)
        assert result is original

    def test_add_vector_field_returns_existing(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        original = mesh.fields["v"]
        result = mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
        assert result is original

    def test_add_test_function_reuses_existing(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        first = mesh.Add_TestFunction("p", step=0)
        second = mesh.Add_TestFunction("p", step=0)
        assert first is second

    def test_add_test_function_missing_field(self, Mesh_Test_Params):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        with pytest.raises(ValueError, match="Field 'missing' not found"):
            mesh.Add_TestFunction("missing", step=0)

    def test_add_vector_test_function_dimension_warning(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        mesh.Add_Vector_TestFunction("v", nDim=3, step=5)
        assert "v_test_step5" in mesh.test_functions

    def test_finalize_test_functions_is_idempotent(self, mesh_dofs_fixture):
        mesh = mesh_dofs_fixture
        original = mesh.mixed_spaces[0]
        mesh.finalize_test_functions()
        assert mesh.mixed_spaces[0] is original

    def test_finalize_test_functions_with_no_registered_steps(self, Mesh_Test_Params):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.finalize_test_functions()
        assert mesh.mixed_spaces == {}

    def test_set_from_expression_scalar(self, mesh_dofs_fixture):
        """Test setting a scalar field from a UFL expression."""
        from ufl import SpatialCoordinate

        mesh = mesh_dofs_fixture
        x = SpatialCoordinate(mesh.mesh)
        expr = x[0] + x[1]
        mesh.Set_From_Expression("p", expr)

        p_field = mesh.fields["p"]
        dof_coords = p_field.function_space.tabulate_dof_coordinates()

        for i in range(min(10, len(dof_coords))):
            expected = dof_coords[i, 0] + dof_coords[i, 1]
            actual = p_field.x.array[i]
            assert np.isclose(actual, expected, rtol=1e-10), (
                f"At DOF {i}: expected {expected}, got {actual}"
            )

    def test_set_from_expression_vector(self, mesh_dofs_fixture):
        """Test setting a vector field from a UFL expression."""
        from ufl import SpatialCoordinate, as_vector

        mesh = mesh_dofs_fixture
        x = SpatialCoordinate(mesh.mesh)
        expr = as_vector([2.0 * x[0], 3.0 * x[1]])
        mesh.Set_From_Expression("v", expr)

        v_field = mesh.fields["v"]
        dof_coords = v_field.function_space.tabulate_dof_coordinates()
        num_dofs_per_component = len(dof_coords) // 2

        for i in range(min(5, num_dofs_per_component)):
            expected_vx = 2.0 * dof_coords[i, 0]
            actual_vx = v_field.x.array[2 * i]
            assert np.isclose(actual_vx, expected_vx, rtol=1e-10), (
                f"At DOF {i}, x-component: expected {expected_vx}, got {actual_vx}"
            )

            expected_vy = 3.0 * dof_coords[i, 1]
            actual_vy = v_field.x.array[2 * i + 1]
            assert np.isclose(actual_vy, expected_vy, rtol=1e-10), (
                f"At DOF {i}, y-component: expected {expected_vy}, got {actual_vy}"
            )

    def test_set_from_expression_missing_field(self, mesh_dofs_fixture):
        """Test that setting an undefined field raises ValueError."""
        from ufl import SpatialCoordinate

        mesh = mesh_dofs_fixture
        x = SpatialCoordinate(mesh.mesh)
        expr = x[0]

        with pytest.raises(ValueError, match="Field 'nonexistent' not found"):
            mesh.Set_From_Expression("nonexistent", expr)


class TestCopyFieldWhiteBoxEdgeCases:
    """White-box coverage for public Copy_Field edge branches."""

    def test_copy_field_quadrature_same_space_uses_direct_array_copy(self):
        """Quadrature fields in the same space should bypass interpolation."""
        mesh = _MeshDOFsHarness()
        quad_space = SimpleNamespace(element=SimpleNamespace(family="Quadrature"))
        source = _DummyField(quad_space, [1.0, 2.0, 3.0])
        dest = _DummyField(quad_space, [0.0, 0.0, 0.0])

        mesh.fields["dest"] = dest
        mesh.field_type["source"] = ""
        mesh.field_type["dest"] = ""
        mesh.Get_Field_For_Output = lambda field_name: FieldOutputView(
            field_name=field_name,
            field=source,
            comp_count=1,
        )

        mesh.Copy_Field("source", "dest")

        np.testing.assert_allclose(dest.x.array, source.x.array)
        assert dest.interpolated_with is None
        assert dest.x.scatter_calls == 1

    def test_copy_field_quadrature_different_space_uses_expression(self, monkeypatch):
        """Quadrature detection should fall back to Expression interpolation when spaces differ."""
        mesh = _MeshDOFsHarness()

        class _RaisesBasixElement:
            family = None

            @property
            def basix_element(self):
                raise RuntimeError("No basix element available")

        source_space = SimpleNamespace(element=_RaisesBasixElement())
        dest_space = SimpleNamespace(
            element=SimpleNamespace(
                family=None,
                basix_element=SimpleNamespace(family="Quadrature"),
                interpolation_points="quadrature-points",
            )
        )

        source = _DummyField(source_space, [9.0, 8.0])
        dest = _DummyField(dest_space, [0.0, 0.0])

        created = {}

        def _fake_expression(field, interpolation_points):
            created["field"] = field
            created["points"] = interpolation_points
            return ("expr", field, interpolation_points)

        monkeypatch.setattr(mesh_dofs_module.fem, "Expression", _fake_expression)

        mesh.fields["dest"] = dest
        mesh.field_type["source"] = ""
        mesh.field_type["dest"] = ""
        mesh.Get_Field_For_Output = lambda field_name: FieldOutputView(
            field_name=field_name,
            field=source,
            comp_count=1,
        )

        mesh.Copy_Field("source", "dest")

        assert created["field"] is source
        assert created["points"] == "quadrature-points"
        assert dest.interpolated_with == ("expr", source, "quadrature-points")
        assert dest.x.scatter_calls == 1

    def test_copy_field_nonquadrature_basix_path_uses_regular_interpolation(self):
        """Non-quadrature basix metadata should stay on the standard interpolation path."""
        mesh = _MeshDOFsHarness()
        source_space = SimpleNamespace(
            element=SimpleNamespace(family=None, basix_element=SimpleNamespace(family=object()))
        )
        dest_space = SimpleNamespace(
            element=SimpleNamespace(family=None, basix_element=SimpleNamespace(family=object()))
        )

        source = _DummyField(source_space, [5.0])
        dest = _DummyField(dest_space, [0.0])

        mesh.fields["dest"] = dest
        mesh.field_type["source"] = ""
        mesh.field_type["dest"] = ""
        mesh.Get_Field_For_Output = lambda field_name: FieldOutputView(
            field_name=field_name,
            field=source,
            comp_count=1,
        )

        mesh.Copy_Field("source", "dest")

        assert dest.interpolated_with is source
        assert dest.x.scatter_calls == 1


class TestMeshDOFsWhiteBoxOutputHelpers:
    """White-box coverage for private output-projection helpers."""

    def test_output_component_helper_branches(self):
        """Numeric suffix parsing and component-index helpers should cover row/column edge cases."""
        mesh = _MeshDOFsHarness()
        mesh.fields["vector"] = SimpleNamespace(
            function_space=SimpleNamespace(element=SimpleNamespace(value_shape=(2,)))
        )
        mesh.fields["row_tensor"] = SimpleNamespace(
            function_space=SimpleNamespace(element=SimpleNamespace(value_shape=(1, 2)))
        )

        assert mesh._parse_numeric_output_component("vector", "123") is None
        assert mesh._resolve_vector_component_index("2") == 1
        assert mesh._resolve_tensor_component_indices("2", (1, 2)) == (0, 1)

        with pytest.raises(KeyError):
            mesh._resolve_vector_component_index("theta")

        with pytest.raises(KeyError):
            mesh._resolve_tensor_component_indices("theta", (2, 2))

        with pytest.raises(KeyError):
            mesh._resolve_mandel_component("11")

    def test_is_quadrature_field_private_metadata_paths(self):
        """Quadrature detection should cover family strings, basix strings, and unsupported metadata."""
        mesh = _MeshDOFsHarness()

        def _view_for_element(element):
            field = _DummyField(SimpleNamespace(element=element), [0.0])
            field.name = "Dummy"
            return FieldOutputView(
                field_name="dummy",
                field=field,
                comp_count=1,
            )

        class _RaisesBasixElement:
            family = None
            value_shape = ()

            @property
            def basix_element(self):
                raise RuntimeError("No basix element available")

        assert mesh._is_quadrature_field(
            "dummy", _view_for_element(SimpleNamespace(family="Quadrature", value_shape=()))
        )
        assert not mesh._is_quadrature_field(
            "dummy", _view_for_element(SimpleNamespace(family="CG", value_shape=()))
        )
        assert not mesh._is_quadrature_field("dummy", _view_for_element(_RaisesBasixElement()))
        assert mesh._is_quadrature_field(
            "dummy",
            _view_for_element(
                SimpleNamespace(
                    family=None,
                    basix_element=SimpleNamespace(family="Quadrature"),
                    value_shape=(),
                )
            ),
        )
        assert not mesh._is_quadrature_field(
            "dummy",
            _view_for_element(
                SimpleNamespace(
                    family=None,
                    basix_element=SimpleNamespace(family=object()),
                    value_shape=(),
                )
            ),
        )

    def test_project_quadrature_source_to_output_space_returns_none_on_projection_failure(self):
        """Quadrature projection failures should be logged and converted to ``None``."""
        mesh = _MeshDOFsHarness()
        mesh._resolve_output_dg_space = lambda field_name, V_dg: V_dg
        mesh._get_cached_output_projection = lambda field_spec, V_dg: (
            SimpleNamespace(
                x=SimpleNamespace(
                    array=np.array([0.0]),
                    petsc_vec=None,
                    scatter_forward=lambda: None,
                )
            ),
            None,
        )

        def _raise_projection_failure(_V_dg):
            raise RuntimeError("boom")

        mesh._get_cached_output_mass_projection = _raise_projection_failure

        assert (
            mesh._project_quadrature_source_to_output_space(
                "history_q", "history_q", "History Q", 1.0, object(), "cell"
            )
            is None
        )

    def test_project_quadrature_source_to_output_space_drops_ghost_cell_values(
        self, monkeypatch
    ):
        """Cell-data quadrature plots should return only owned DG0 entries."""
        mesh = _MeshDOFsHarness()

        class _FakeExpr:
            def __mul__(self, _other):
                return self

            __rmul__ = __mul__

        class _FakeVec:
            def ghostUpdate(self, *args, **kwargs):
                return None

            def destroy(self):
                return None

        class _FakeKSP:
            def solve(self, _rhs_vec, _petsc_vec):
                return None

        V_dg = SimpleNamespace(
            dofmap=SimpleNamespace(
                index_map=SimpleNamespace(size_local=3),
                index_map_bs=1,
            )
        )
        field_dg = SimpleNamespace(
            x=SimpleNamespace(
                array=np.array([1.0, 2.0, 3.0, 99.0, 100.0]),
                petsc_vec=object(),
                scatter_forward=lambda: None,
            )
        )

        mesh._resolve_output_dg_space = lambda field_name, candidate: candidate
        mesh._get_cached_output_projection = lambda field_spec, candidate: (field_dg, None)
        mesh._get_cached_output_mass_projection = lambda candidate: (_FakeExpr(), None, _FakeKSP())
        mesh.Get_Field_Support = lambda field_name: SimpleNamespace(is_full=True)
        monkeypatch.setattr(mesh_dofs_module.fem, "form", lambda expr: expr)
        monkeypatch.setattr(mesh_dofs_module, "assemble_vector", lambda _form: _FakeVec())

        projected = mesh._project_quadrature_source_to_output_space(
            "strain_q", "strain_q_zz", "Strain ZZ", _FakeExpr(), V_dg, "cell"
        )

        assert projected.association == "cell"
        np.testing.assert_allclose(projected.values, [1.0, 2.0, 3.0])

    def test_project_tensor_component_returns_none_when_index_extraction_fails(self):
        """Tensor projection should return ``None`` when Mandel or tensor indexing fails."""
        mesh = _MeshDOFsHarness()

        class _RaisingIndexedField:
            def __init__(self, value_shape):
                self.name = "Stress"
                self.function_space = SimpleNamespace(
                    element=SimpleNamespace(value_shape=value_shape, family="CG")
                )

            def __getitem__(self, _key):
                raise RuntimeError("boom")

        mandel_field = _RaisingIndexedField((6,))
        mesh.fields["sigma_mandel"] = mandel_field
        mesh.field_type["sigma_mandel"] = "CG Tensor"
        mesh.Get_Field_For_Output = lambda field_name: FieldOutputView(
            field_name=field_name,
            field=mandel_field if field_name == "sigma_mandel" else tensor_field,
            comp_count=6 if field_name == "sigma_mandel" else 4,
        )

        assert mesh._project_tensor_field_component("sigma_mandel", "xx", "sigma_mandel_xx") is None

        tensor_field = _RaisingIndexedField((2, 2))
        mesh.fields["sigma_tensor"] = tensor_field
        mesh.field_type["sigma_tensor"] = "CG Tensor"

        assert mesh._project_tensor_field_component("sigma_tensor", "xx", "sigma_tensor_xx") is None


class TestMeshDOFsPrint:
    """Coverage for print_dofs_info output paths."""

    def test_print_dofs_info_no_steps(self, Mesh_Test_Params, capsys):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.print_dofs_info()
        captured = capsys.readouterr().out
        assert "Degrees of freedom per solution step" in captured
        assert "No solution steps defined." in captured

    def test_print_dofs_info_empty_step(self, Mesh_Test_Params, capsys):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.step_fields[0] = []
        mesh.print_dofs_info()
        captured = capsys.readouterr().out
        assert "Step 0: No fields" in captured

    def test_print_dofs_info_scalar_and_vector(self, Mesh_Test_Params, capsys):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.Add_Field("p", "Pressure", "CG", 1)
        mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
        mesh.step_fields[0] = ["p", "v"]
        mesh.print_dofs_info()
        captured = capsys.readouterr().out
        assert "Step 0:" in captured
        assert "p:" in captured
        assert "v:" in captured
        assert "components" in captured

    def test_print_dofs_info_skips_missing_field(self, Mesh_Test_Params, capsys):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        mesh.step_fields[0] = ["ghost"]
        mesh.print_dofs_info()
        captured = capsys.readouterr().out
        assert "Step 0:" in captured

    def test_global_solution_dof_count_counts_registered_fields_once(
        self,
        Mesh_Test_Params,
    ):
        from femcubes.core.mesh.mesh import Mesh

        mesh = Mesh(Mesh_Test_Params)
        pressure = mesh.Add_Field("p", "Pressure", "CG", 1)
        velocity = mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
        mesh.step_fields[0] = ["p", "v"]
        mesh.step_fields[1] = ["p", "missing"]

        expected = sum(
            field.function_space.dofmap.index_map.size_global
            * field.function_space.dofmap.index_map_bs
            for field in (pressure, velocity)
        )
        assert mesh.get_global_solution_dof_count() == expected


def _build_mpi_mesh():
    """Create a mesh large enough that each MPI rank owns non-zero DOFs."""
    from femcubes.core.mesh.mesh import Mesh

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 6.0
        Ly = 3.0
        dx = 0.5
        dim = 2

    return Mesh(TestParams())


def _owned_entries(function_space) -> int:
    """Return the number of owned vector entries for a function space."""
    dofmap = function_space.dofmap
    return dofmap.index_map.size_local * dofmap.index_map_bs


def _global_entries(function_space) -> int:
    """Return the global number of vector entries for a function space."""
    dofmap = function_space.dofmap
    return dofmap.index_map.size_global * dofmap.index_map_bs


def _assert_partitioned_ownership(function_space, comm) -> None:
    """Check that rank ownership ranges form a contiguous partition."""
    dofmap = function_space.dofmap
    bs = dofmap.index_map_bs
    local_start, local_end = dofmap.index_map.local_range
    gathered_ranges = comm.allgather((local_start * bs, local_end * bs))

    expected_start = 0
    for start, end in gathered_ranges:
        assert start == expected_start
        assert end >= start
        expected_start = end

    assert expected_start == _global_entries(function_space)


def _run_mpi_dof_distribution_checks() -> None:
    """Validate scalar, vector, and DG0 DOF ownership across ranks."""
    from femcubes.core.utils.mpi_utils import comm, size, isolate_jit_cache

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("MPI DOF distribution checks require at least 2 ranks.")

    mesh = _build_mpi_mesh()
    mesh.Add_Field("p", "Pressure", "CG", 1)
    mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
    mesh.Add_Field("damage", "Damage", "DG", 0)

    p_space = mesh.fields["p"].function_space
    v_space = mesh.fields["v"].function_space
    damage_space = mesh.fields["damage"].function_space

    p_owned = _owned_entries(p_space)
    v_owned = _owned_entries(v_space)
    damage_owned = _owned_entries(damage_space)

    assert p_owned > 0
    assert v_owned > 0
    assert damage_owned > 0
    assert v_owned == 2 * p_owned
    assert _global_entries(v_space) == 2 * _global_entries(p_space)

    _assert_partitioned_ownership(p_space, comm)
    _assert_partitioned_ownership(v_space, comm)
    _assert_partitioned_ownership(damage_space, comm)

    tdim = mesh.mesh.topology.dim
    local_cells = mesh.mesh.topology.index_map(tdim).size_local
    global_cells = mesh.mesh.topology.index_map(tdim).size_global
    assert damage_owned == local_cells
    assert comm.allreduce(damage_owned) == global_cells


def _run_mpi_mixed_space_distribution_checks() -> None:
    """Validate mixed-space DOF accounting across ranks."""
    from femcubes.core.utils.mpi_utils import comm, size, isolate_jit_cache

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("MPI mixed-space checks require at least 2 ranks.")

    mesh = _build_mpi_mesh()
    mesh.Add_Field("p", "Pressure", "CG", 1)
    mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
    mesh.Add_TestFunction("p", step=0)
    mesh.Add_Vector_TestFunction("v", nDim=2, step=0)
    mesh.finalize_test_functions()

    p_space = mesh.fields["p"].function_space
    v_space = mesh.fields["v"].function_space

    p_owned = _owned_entries(p_space)
    v_owned = _owned_entries(v_space)
    p_global = _global_entries(p_space)
    v_global = _global_entries(v_space)

    # The step vector holds one contiguous block per field on every rank.
    offsets = mesh.get_step_block_offsets(0)
    assert offsets == [0, p_owned, p_owned + v_owned]
    assert comm.allreduce(offsets[-1]) == p_global + v_global
    _assert_partitioned_ownership(p_space, comm)
    _assert_partitioned_ownership(v_space, comm)

    p_output = mesh.Get_Field_For_Output("p")
    v_output = mesh.Get_Field_For_Output("v")
    assert isinstance(p_output, FieldOutputView)
    assert isinstance(v_output, FieldOutputView)
    assert p_output.field is mesh.fields["p"]
    assert v_output.field is mesh.fields["v"]


def _run_mpi_distributed_field_operation_checks() -> None:
    """Validate DOF-preserving field operations on distributed data."""
    from femcubes.core.utils.mpi_utils import comm, rank, size, isolate_jit_cache

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("MPI field-operation checks require at least 2 ranks.")

    mesh = _build_mpi_mesh()
    mesh.Add_Field("p", "Pressure", "CG", 1)
    mesh.Add_Field("p_copy", "Pressure Copy", "CG", 1)
    mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
    mesh.Add_Vector_Field("v_copy", "Velocity Copy", "CG", 1, 2)

    p_owned = _owned_entries(mesh.fields["p"].function_space)
    v_owned = _owned_entries(mesh.fields["v"].function_space)

    mesh.fields["p"].x.array[:] = -1.0
    mesh.fields["v"].x.array[:] = -1.0
    mesh.fields["p"].x.array[:p_owned] = rank + 1.0
    mesh.fields["v"].x.array[:v_owned] = rank + 2.0
    mesh.fields["p"].x.scatter_forward()
    mesh.fields["v"].x.scatter_forward()

    mesh.Copy_Field("p", "p_copy")
    mesh.Copy_Field("v", "v_copy")
    assert np.allclose(mesh.fields["p_copy"].x.array[:p_owned], rank + 1.0)
    assert np.allclose(mesh.fields["v_copy"].x.array[:v_owned], rank + 2.0)

    mesh.Constant_Field("p", 7.5)
    mesh.Zero_Field("v")
    assert np.allclose(mesh.fields["p"].x.array[:p_owned], 7.5)
    assert np.allclose(mesh.fields["v"].x.array[:v_owned], 0.0)

    owned_pressure_counts = comm.allgather(p_owned)
    owned_velocity_counts = comm.allgather(v_owned)
    assert sum(owned_pressure_counts) == _global_entries(mesh.fields["p"].function_space)
    assert sum(owned_velocity_counts) == _global_entries(mesh.fields["v"].function_space)


def _run_mpi_step_global_index_checks() -> None:
    """Validate step-vector to global index conversion against DOLFINx block matrices."""
    import ufl
    from dolfinx.fem.petsc import create_matrix

    from femcubes.core.mesh.mesh import Mesh
    from femcubes.mesh_generators.two_squared import Sample_Parameters
    from femcubes.core.utils.mpi_utils import isolate_jit_cache, size

    isolate_jit_cache()
    if size < 2:
        raise RuntimeError("MPI step-index checks require at least 2 ranks.")

    mesh = Mesh(Sample_Parameters())
    mesh.Add_Vector_Field("u", "Displacement", "CG", 2, 2, domain="domain:left")
    mesh.Add_Field("c", "Concentration", "CG", 1, domain="domain:right")
    mesh.Add_Vector_TestFunction("u", nDim=2, step=0)
    mesh.Add_TestFunction("c", step=0)
    mesh.finalize_test_functions()
    u_t, c_t = mesh.Get_TestFunction("u", 0), mesh.Get_TestFunction("c", 0)
    u_tr, c_tr = mesh.Get_Trial_Function("u", 0), mesh.Get_Trial_Function("c", 0)
    K = ufl.inner(u_t, u_tr) * mesh.Get_Domain_Integrator("domain:left") + c_t * c_tr * mesh.Get_Domain_Integrator("domain:right")
    matrix = create_matrix(mesh.Compile_Form(K), kind="mpi")
    matrix_globals = matrix.getLGMap()[0].getIndices()

    # Step vector: all owned entries, then all ghosts. Block matrix: per
    # block, owned entries followed by that block's ghosts.
    maps = [mesh.fields[name].function_space.dofmap for name in ("u", "c")]
    owned = [m.index_map.size_local * m.index_map_bs for m in maps]
    ghosts = [m.index_map.num_ghosts * m.index_map_bs for m in maps]
    vector_indices, matrix_indices = [], []
    for block in range(2):
        vector_indices.append(sum(owned[:block]) + np.arange(owned[block]))
        vector_indices.append(sum(owned) + sum(ghosts[:block]) + np.arange(ghosts[block]))
        matrix_start = sum(owned[:block]) + sum(ghosts[:block])
        matrix_indices.append(matrix_start + np.arange(owned[block] + ghosts[block]))
    vector_indices = np.concatenate(vector_indices)
    matrix_indices = np.concatenate(matrix_indices)

    np.testing.assert_array_equal(
        mesh.get_step_global_indices(0, vector_indices), matrix_globals[matrix_indices]
    )
    matrix.destroy()


def _run_mpi_case(case_name: str) -> None:
    """Dispatch a self-launched MPI validation case."""
    cases = {
        "dof-distribution": _run_mpi_dof_distribution_checks,
        "mixed-space-distribution": _run_mpi_mixed_space_distribution_checks,
        "field-operations": _run_mpi_distributed_field_operation_checks,
        "step-global-indices": _run_mpi_step_global_index_checks,
    }
    if case_name not in cases:
        raise ValueError(f"Unknown MPI case '{case_name}'.")
    cases[case_name]()


def _run_mpi_subprocess(case_name: str) -> None:
    """Launch this test module under mpirun for one MPI-specific check."""
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")

    try:
        import mpi4py  # noqa: F401
    except ImportError:
        pytest.skip("mpi4py not available")

    project_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) if not pythonpath else f"{project_root}{os.pathsep}{pythonpath}"

    cmd = [mpirun, "-n", "2", sys.executable, __file__, "--mpi-check", case_name]
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    result = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        output = "\n".join([result.stdout, result.stderr]).strip()
        raise AssertionError(f"MPI DOF check '{case_name}' failed:\n{output}")


@pytest.mark.mpi
def test_mpi_dof_distribution_across_ranks():
    """Scalar, vector, and DG0 spaces should partition owned DOFs consistently."""
    _run_mpi_subprocess("dof-distribution")


@pytest.mark.mpi
def test_mpi_mixed_space_dof_distribution_across_ranks():
    """Mixed-space ownership should equal the sum of its constituent fields."""
    _run_mpi_subprocess("mixed-space-distribution")


@pytest.mark.mpi
def test_mpi_distributed_field_operations_preserve_owned_dofs():
    """Field operations should behave correctly on rank-local owned DOFs."""
    _run_mpi_subprocess("field-operations")


@pytest.mark.mpi
def test_mpi_step_global_indices_match_block_matrix_numbering():
    """Step-vector indices (owned and ghost) map to the block matrix's global rows."""
    _run_mpi_subprocess("step-global-indices")


if __name__ == "__main__" and "--mpi-check" in sys.argv:
    case_index = sys.argv.index("--mpi-check") + 1
    if case_index >= len(sys.argv):
        raise SystemExit("Missing MPI case name.")
    _run_mpi_case(sys.argv[case_index])
