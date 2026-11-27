"""Tests for mesh-side Dirichlet boundary-condition helpers."""

import numpy as np
import pytest
from types import SimpleNamespace
from dolfinx import fem

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
import femcubes.core.mesh._mesh_bc as mesh_bc_module
from femcubes.core.mesh._mesh_bc import _MeshBC


class _MeshBCHarnessBase:
    """Minimal cooperative base for white-box _MeshBC tests."""

    axisymmetric = False

    def __init__(self, params):
        self.params = params


class _MeshBCHarness(_MeshBC, _MeshBCHarnessBase):
    """Small harness for testing boundary-integrator cache behavior."""

    def __init__(self):
        self.mesh = object()
        self.facet_tags = object()
        super().__init__(SimpleNamespace())


@pytest.fixture
def MeshBC_Test_Params():
    """Create a simple rectangle-mesh parameter set for BC helper tests."""

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2

    return TestParams()


@pytest.fixture
def bc_test_mesh(MeshBC_Test_Params):
    """Create one mesh with plain and mixed fields for BC helper tests."""
    mesh = Mesh(MeshBC_Test_Params)

    mesh.Add_Field("temp", "Temperature", "CG", 1)
    mesh.Add_Vector_Field("velocity", "Velocity", "CG", 1, 2)
    mesh.Add_Field("p", "Pressure", "CG", 1)
    mesh.Add_Vector_Field("v", "Mixed Velocity", "CG", 1, 2)

    mesh.Add_TestFunction("p", 0)
    mesh.Add_Vector_TestFunction("v", 2, 0)
    mesh.finalize_test_functions()

    mesh.Constant_Field("temp", 2.0)
    mesh.Constant_Field("velocity", 1.0)
    mesh.Constant_Field("p", 1.5)
    mesh.Constant_Field("v", 2.0)

    return mesh


class TestMeshDirichletBCHelpers:
    """Public tests for the mesh-side incremental Dirichlet BC API."""

    def test_get_boundary_integrator_reuses_cached_measure_until_mesh_changes(
        self, monkeypatch
    ):
        harness = _MeshBCHarness()
        created = []

        def _fake_measure(name, domain=None, subdomain_data=None):
            created.append((name, domain, subdomain_data))
            return {
                "name": name,
                "domain": domain,
                "subdomain_data": subdomain_data,
                "ordinal": len(created),
            }

        monkeypatch.setattr(mesh_bc_module.ufl, "Measure", _fake_measure)
        monkeypatch.setattr(harness, "_exterior_facet_tags", lambda: harness.facet_tags)

        first = harness.get_Boundary_Integrator()
        second = harness.get_Boundary_Integrator()

        assert first is second
        assert len(created) == 1

        harness.mesh = object()
        refreshed = harness.get_Boundary_Integrator()

        assert refreshed is not first
        assert len(created) == 2

    def test_build_increment_dirichlet_bcs_for_plain_scalar_field(self, bc_test_mesh):
        bcs = bc_test_mesh.Build_Increment_Dirichlet_BCs("temp", "left", 5.0)

        assert len(bcs) == 1
        np.testing.assert_allclose(bcs[0].g.x.array, -3.0)
        assert bcs[0].dof_indices()[0].size > 0

    def test_build_increment_dirichlet_bcs_skips_dg_fields(self, bc_test_mesh):
        bc_test_mesh.Add_Field("height", "Thickness", "DG", 0)

        bcs = bc_test_mesh.Build_Increment_Dirichlet_BCs("height", "left", 5.0)

        assert bcs == []

    def test_build_increment_dirichlet_bcs_for_plain_vector_field(self, bc_test_mesh):
        bcs = bc_test_mesh.Build_Increment_Dirichlet_BCs("velocity", "left", [3.0, 4.0])

        assert len(bcs) == 2
        np.testing.assert_allclose(bcs[0].g.x.array, -2.0)
        np.testing.assert_allclose(bcs[1].g.x.array, -3.0)
        assert bcs[0].dof_indices()[0].size > 0
        assert bcs[1].dof_indices()[0].size > 0

    def test_build_increment_dirichlet_bcs_for_explicit_vector_component(self, bc_test_mesh):
        bcs = bc_test_mesh.Build_Increment_Dirichlet_BCs(
            "velocity", "left", 4.0, component=1
        )

        assert len(bcs) == 1
        np.testing.assert_allclose(bcs[0].g.x.array, -3.0)
        assert bcs[0].dof_indices()[0].size > 0

    def test_build_increment_dirichlet_bcs_reuses_objects_and_updates_values(
        self, bc_test_mesh
    ):
        first = bc_test_mesh.Build_Increment_Dirichlet_BCs(
            "velocity", "left", 4.0, component=1
        )

        bc_test_mesh.Constant_Field("velocity", 2.0)
        second = bc_test_mesh.Build_Increment_Dirichlet_BCs(
            "velocity", "left", 5.0, component=1
        )

        assert second[0] is first[0]
        np.testing.assert_allclose(second[0].g.x.array, -3.0)

    def test_invalidating_boundary_cache_rebuilds_incremental_bc(
        self, bc_test_mesh
    ):
        first = bc_test_mesh.Build_Increment_Dirichlet_BCs(
            "temp", "left", 5.0
        )

        bc_test_mesh._invalidate_boundary_cache()
        second = bc_test_mesh.Build_Increment_Dirichlet_BCs(
            "temp", "left", 5.0
        )

        assert second[0] is not first[0]

    def test_build_increment_dirichlet_bcs_for_mixed_scalar_field(self, bc_test_mesh):
        bcs = bc_test_mesh.Build_Increment_Dirichlet_BCs("p", "left", 5.0)

        assert len(bcs) == 1
        np.testing.assert_allclose(bcs[0].g.x.array, -3.5)
        assert bcs[0].dof_indices()[0].size > 0

    def test_build_increment_dirichlet_bcs_for_mixed_vector_field(self, bc_test_mesh):
        bcs = bc_test_mesh.Build_Increment_Dirichlet_BCs("v", "left", [5.0, 6.0])

        assert len(bcs) == 2
        np.testing.assert_allclose(bcs[0].g.x.array, -3.0)
        np.testing.assert_allclose(bcs[1].g.x.array, -4.0)
        assert bcs[0].dof_indices()[0].size > 0
        assert bcs[1].dof_indices()[0].size > 0

    def test_build_increment_dirichlet_bcs_rejects_wrong_component_target_count(
        self, bc_test_mesh
    ):
        with pytest.raises(ValueError, match="Expected 2 component targets"):
            bc_test_mesh.Build_Increment_Dirichlet_BCs("v", "left", [1.0, 2.0, 3.0])

    def test_normalize_bc_targets_accepts_numpy_arrays(self, bc_test_mesh):
        targets = bc_test_mesh._normalize_bc_targets(np.array([3.0, 4.0]), 2)

        assert targets == [3.0, 4.0]

    def test_normalize_bc_targets_broadcasts_scalar_targets(self, bc_test_mesh):
        targets = bc_test_mesh._normalize_bc_targets(5.0, 2)

        assert targets == [5.0, 5.0]

    def test_populate_increment_bc_function_handles_full_vector_space(self, bc_test_mesh):
        class _FakeSubFunction:
            def __init__(self, values):
                self.x = SimpleNamespace(array=np.asarray(values, dtype=float))

        class _FakeFunction:
            def __init__(self, values, shape):
                self.function_space = SimpleNamespace(
                    element=SimpleNamespace(value_shape=shape)
                )
                self._subs = [_FakeSubFunction(comp_values) for comp_values in values]

            def sub(self, comp_idx):
                return self._subs[comp_idx]

        current_field = _FakeFunction([[1.0, 1.0], [1.0, 1.0]], (2,))
        bc_func = _FakeFunction([[0.0, 0.0], [0.0, 0.0]], (2,))

        bc_test_mesh._populate_increment_bc_function(
            bc_func, current_field, np.array([3.0, 4.0])
        )

        np.testing.assert_allclose(bc_func.sub(0).x.array, -2.0)
        np.testing.assert_allclose(bc_func.sub(1).x.array, -3.0)
