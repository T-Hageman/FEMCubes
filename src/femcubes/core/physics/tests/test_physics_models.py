"""Tests for the public model-lifecycle API on ``Physics``."""

import numpy as np

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics


def _assert_field_equal(mesh, field_name_a, field_name_b):
    np.testing.assert_allclose(
        mesh.fields[field_name_a].x.array,
        mesh.fields[field_name_b].x.array,
    )


class TestPhysicsModelLifecycle:
    def test_physics_initializes_linear_elastic_models(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)

        assert list(physics.models) == linear_elastic_params.ModelNames
        assert physics.models["Solid"].MODEL_TYPE == "Solid:LinearElastic"

    def test_physics_initializes_stokes_models(self, stokes_params):
        mesh = Mesh(stokes_params)
        physics = Physics(mesh, stokes_params)

        assert list(physics.models) == stokes_params.ModelNames
        assert physics.models["Solid"].MODEL_TYPE == "Solid:Stokes"

    def test_initialize_fields_zeroes_linear_elastic_fields(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)

        for field_name in ("u", "uOld", "v", "vOld", "a", "aOld", "p"):
            mesh.Constant_Field(field_name, 1.25)

        physics.Initialize_Fields()

        for field_name in ("u", "uOld", "v", "vOld", "a", "aOld", "p"):
            assert np.allclose(mesh.fields[field_name].x.array, 0.0)

    def test_initialize_fields_zeroes_stokes_fields(self, stokes_params):
        mesh = Mesh(stokes_params)
        physics = Physics(mesh, stokes_params)

        for field_name in ("v", "vOld", "p", "pOld"):
            mesh.Constant_Field(field_name, 2.5)

        physics.Initialize_Fields()

        for field_name in ("v", "vOld", "p", "pOld"):
            assert np.allclose(mesh.fields[field_name].x.array, 0.0)

    def test_commit_and_reset_step_restore_linear_elastic_state(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        mesh.Constant_Field("u", 1.25)
        physics.commit()

        for field_name, old_field_name in (("u", "uOld"), ("v", "vOld"), ("a", "aOld")):
            _assert_field_equal(mesh, field_name, old_field_name)

        mesh.Constant_Field("u", -4.0)
        mesh.Constant_Field("v", -3.0)
        mesh.Constant_Field("a", -2.0)
        physics.Reset_Step()

        for field_name, old_field_name in (("u", "uOld"), ("v", "vOld"), ("a", "aOld")):
            _assert_field_equal(mesh, field_name, old_field_name)

    def test_commit_and_reset_step_restore_stokes_state(self, stokes_params):
        mesh = Mesh(stokes_params)
        physics = Physics(mesh, stokes_params)
        physics.Initialize_Fields()

        mesh.Constant_Field("v", 0.75)
        mesh.Constant_Field("p", 1.5)
        physics.commit()

        _assert_field_equal(mesh, "v", "vOld")
        _assert_field_equal(mesh, "p", "pOld")

        mesh.Constant_Field("v", -0.25)
        mesh.Constant_Field("p", -1.0)
        physics.Reset_Step()

        _assert_field_equal(mesh, "v", "vOld")
        _assert_field_equal(mesh, "p", "pOld")

    def test_post_remesh_destroys_assembly_and_dispatches_in_model_order(
        self,
        linear_elastic_params,
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)

        class _Destroyable:
            def __init__(self):
                self.destroyed = False

            def destroy(self):
                self.destroyed = True

        K_mat = _Destroyable()
        F_vec = _Destroyable()
        physics._assembly_cache = {0: {"K_mat": K_mat, "F_vec": F_vec}}
        physics.K_mat = K_mat
        physics.F_vec = F_vec
        model_ids = {name: id(model) for name, model in physics.models.items()}
        global_measures = linear_elastic_params.Global_Measures
        global_measures["retained"] = 3.5
        calls = []
        for model_name, model in physics.models.items():
            model.Post_Remesh = lambda name=model_name: calls.append(name)

        physics.Post_Remesh()

        assert calls == linear_elastic_params.ModelNames
        assert {name: id(model) for name, model in physics.models.items()} == model_ids
        assert linear_elastic_params.Global_Measures is global_measures
        assert linear_elastic_params.Global_Measures["retained"] == 3.5
        assert K_mat.destroyed
        assert F_vec.destroyed
        assert physics._assembly_cache == {}
        assert physics.K_mat is None
        assert physics.F_vec is None
