"""Tests for the public K/F assembly API on ``Physics``."""

import importlib
import numpy as np
import pytest

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.physics._physics_kf import _PhysicsKF

physics_kf_module = importlib.import_module("femcubes.core.physics._physics_kf")


class _PhysicsKFHarness(_PhysicsKF):
    """Minimal harness for targeted white-box tests of private helpers."""

    def __init__(self):
        pass


class TestPhysicsAssemblyApi:
    def test_get_fields_for_step_linear_elastic(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)

        fields_to_solve = physics.get_fields_for_step(0)

        assert [name for name, _ in fields_to_solve] == ["u"]

    def test_get_fields_for_step_stokes(self, stokes_params):
        mesh = Mesh(stokes_params)
        physics = Physics(mesh, stokes_params)

        fields_to_solve = physics.get_fields_for_step(0)

        assert {name for name, _ in fields_to_solve} == {"v", "p"}

    def test_get_boundary_conditions_for_linear_elastic_step(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        bcs = physics.get_boundary_conditions(step=0)

        assert len(bcs) == 3

    def test_get_boundary_conditions_without_step_returns_all_active_bcs(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        bcs = physics.get_boundary_conditions()

        assert len(bcs) == 3

    def test_get_kmat_fvec_linear_elastic(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        K_mat, F_vec = physics.Get_Kmat_Fvec(0)

        assert K_mat is physics.K_mat
        assert F_vec is physics.F_vec
        assert K_mat.getSize()[0] == K_mat.getSize()[1] > 0
        assert F_vec.getSize() == K_mat.getSize()[0]

    def test_get_kmat_fvec_reuses_previous_objects_when_layout_is_unchanged(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        K_first, F_first = physics.Get_Kmat_Fvec(0)
        K_second, F_second = physics.Get_Kmat_Fvec(0)

        assert K_first is K_second
        assert F_first is F_second

    def test_get_kmat_fvec_reuses_compiled_forms_but_runs_model_updates(
        self, linear_elastic_params, monkeypatch
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        original_form = physics_kf_module.fem.form
        compiled_inputs = []
        update_calls = []
        weak_form_calls = []
        original_update = physics.models["Solid"].Update_Weak_Constants
        original_get_weak_form = physics.models["Solid"].Get_Weak_Form

        def _counted_form(form, **kwargs):
            compiled_inputs.append(form)
            return original_form(form, **kwargs)

        def _counted_update(time_constant, dt_constant, step):
            update_calls.append(step)
            return original_update(time_constant, dt_constant, step)

        def _counted_get_weak_form(time_constant, dt_constant, step):
            weak_form_calls.append(step)
            return original_get_weak_form(time_constant, dt_constant, step)

        monkeypatch.setattr(physics_kf_module.fem, "form", _counted_form)
        physics.models["Solid"].Update_Weak_Constants = _counted_update
        physics.models["Solid"].Get_Weak_Form = _counted_get_weak_form

        physics.Get_Kmat_Fvec(0)
        physics.Get_Kmat_Fvec(0)

        assert update_calls == [0, 0]
        assert weak_form_calls == [0]
        assert len(compiled_inputs) == 2

    def test_timestep_constant_updates_without_rebinding_forms(
        self, linear_elastic_params, monkeypatch
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        original_form = physics_kf_module.fem.form
        compiled_inputs = []

        def _counted_form(form, **kwargs):
            compiled_inputs.append(form)
            return original_form(form, **kwargs)

        monkeypatch.setattr(physics_kf_module.fem, "form", _counted_form)

        physics.Get_Kmat_Fvec(0)
        linear_elastic_params.dt *= 0.5
        physics.Get_Kmat_Fvec(0)

        assert len(compiled_inputs) == 2
        assert float(physics.dt_constant.value) == linear_elastic_params.dt

    def test_model_refresh_request_rebuilds_cached_weak_form(
        self, linear_elastic_params
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        model = physics.models["Solid"]
        original_update = model.Update_Weak_Constants
        original_get_weak_form = model.Get_Weak_Form
        update_count = 0
        weak_form_count = 0

        def _request_second_refresh(time_constant, dt_constant, step):
            nonlocal update_count
            update_count += 1
            original_update(time_constant, dt_constant, step)
            return update_count == 2

        def _count_weak_forms(time_constant, dt_constant, step):
            nonlocal weak_form_count
            weak_form_count += 1
            return original_get_weak_form(time_constant, dt_constant, step)

        model.Update_Weak_Constants = _request_second_refresh
        model.Get_Weak_Form = _count_weak_forms

        physics.Get_Kmat_Fvec(0)
        physics.Get_Kmat_Fvec(0)

        assert weak_form_count == 2

    def test_compiled_form_cache_ignores_time_when_form_identity_is_unchanged(
        self, linear_elastic_params, monkeypatch
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        original_form = physics_kf_module.fem.form
        compiled_inputs = []

        def _counted_form(form, **kwargs):
            compiled_inputs.append(form)
            return original_form(form, **kwargs)

        monkeypatch.setattr(physics_kf_module.fem, "form", _counted_form)

        physics.Get_Kmat_Fvec(0)
        linear_elastic_params.t = (
            linear_elastic_params.start_time + linear_elastic_params.dt
        )
        physics.Get_Kmat_Fvec(0)

        assert len(compiled_inputs) == 2

    def test_compiled_form_cache_does_not_use_mesh_epoch_as_validity_rule(
        self, linear_elastic_params, monkeypatch
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        original_form = physics_kf_module.fem.form
        compiled_inputs = []

        def _counted_form(form, **kwargs):
            compiled_inputs.append(form)
            return original_form(form, **kwargs)

        monkeypatch.setattr(physics_kf_module.fem, "form", _counted_form)

        K_form, _ = physics.models["Solid"].assemble_KF(0)
        physics._get_or_compile_form(0, "K", K_form)
        mesh.remesh_epoch += 1
        physics._get_or_compile_form(0, "K", K_form)

        assert len(compiled_inputs) == 1

    def test_cached_compiled_forms_follow_updated_field_values(
        self, linear_elastic_params
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        physics.Get_Kmat_Fvec(0)
        mesh.fields["u"].x.array[:] = 1.0e-5
        mesh.fields["u"].x.scatter_forward()
        cached_residual = physics.Get_Fvec(0).copy()

        physics._compiled_form_cache.clear()
        fresh_residual = physics.Get_Fvec(0)

        np.testing.assert_allclose(cached_residual.array, fresh_residual.array)
        cached_residual.destroy()

    def test_full_assembly_passes_bcs_directly_without_model_additions(
        self, linear_elastic_params, monkeypatch
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        original_assemble_matrix = physics_kf_module.assemble_matrix
        assembled_bcs = []

        def _counted_assemble_matrix(matrix, form, bcs=None):
            assembled_bcs.append(bcs)
            return original_assemble_matrix(matrix, form, bcs=bcs)

        monkeypatch.setattr(
            physics_kf_module,
            "assemble_matrix",
            _counted_assemble_matrix,
        )

        physics.Get_Kmat_Fvec(0)

        assert len(assembled_bcs) == 1
        assert len(assembled_bcs[0]) == 3

    def test_fast_and_algebraic_assembly_paths_produce_the_same_system(
        self, linear_elastic_params
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        fast_matrix, fast_residual = physics.Get_Kmat_Fvec(0)
        reference_matrix = fast_matrix.copy()
        reference_residual = fast_residual.copy()

        physics.models["Solid"].Add_KF = lambda K_mat, F_vec, step: None
        legacy_matrix, legacy_residual = physics.Get_Kmat_Fvec(0)
        difference = legacy_matrix.copy()
        difference.axpy(-1.0, reference_matrix)

        assert difference.norm() == pytest.approx(0.0, abs=1.0e-12)
        np.testing.assert_allclose(
            legacy_residual.array,
            reference_residual.array,
        )

        difference.destroy()
        reference_matrix.destroy()
        reference_residual.destroy()

    def test_get_fvec_matches_full_assembly_and_reuses_vector(
        self, linear_elastic_params
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        _, F_full = physics.Get_Kmat_Fvec(0)
        expected = F_full.array.copy()
        F_only = physics.Get_Fvec(0)

        assert F_only is F_full
        assert F_only is physics.F_vec
        assert np.allclose(F_only.array, expected)

    def test_assembly_profiling_reports_exclusive_subphases(
        self,
        linear_elastic_params,
    ):
        linear_elastic_params.PROFILE_TIMINGS = True
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        physics.Get_Kmat_Fvec(0)
        physics.Get_Fvec(0)

        timings = physics._runtime_profiler.snapshot()
        counts = physics._runtime_profiler.snapshot_counts()
        assembly_categories = (
            "form_evaluation",
            "form_cache_lookup",
            "form_compilation",
            "local_assembly",
            "assembly_communication",
            "boundary_conditions",
            "model_assembly",
            "assembly_bookkeeping",
        )
        assert "assembly" not in timings
        assert all(timings[category] >= 0.0 for category in assembly_categories)
        assert sum(timings[category] for category in assembly_categories) > 0.0
        assert counts["form_evaluation"] == 2
        assert counts["form_cache_lookup"] == 4
        assert counts["form_compilation"] == 2
        assert counts["form_cache_hit"] == 2
        assert counts["form_cache_miss"] == 2
        assert counts["assembly_bookkeeping"] == 2
        assert counts["matrix_residual_assembly"] == 1
        assert counts["residual_only_assembly"] == 1

    def test_get_fvec_passes_no_matrix_to_model_algebraic_additions(
        self, linear_elastic_params
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        add_kf_calls = []

        def _add_kf(K_mat, F_vec, step):
            add_kf_calls.append((K_mat, F_vec.getSize(), step))

        physics.models["Solid"].Add_KF = _add_kf

        F_vec = physics.Get_Fvec(0)

        assert F_vec.getSize() > 0
        assert add_kf_calls == [(None, F_vec.getSize(), 0)]

    def test_full_assembly_replaces_force_only_workspace_without_leaking_it(
        self, linear_elastic_params
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        F_only = physics.Get_Fvec(0)
        K_mat, F_full = physics.Get_Kmat_Fvec(0)

        assert K_mat is physics.K_mat
        assert F_full is physics.F_vec
        assert F_full is not F_only

    def test_get_kmat_fvec_calls_model_add_kf_and_debug_timing(self, linear_elastic_params, capsys):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        physics.DEBUG_PRINT_TIMING = True

        add_kf_calls = []

        def _add_kf(K_mat, F_vec, step):
            add_kf_calls.append((K_mat.getSize(), F_vec.getSize(), step))

        physics.models["Solid"].Add_KF = _add_kf

        physics.Get_Kmat_Fvec(0)

        assert add_kf_calls
        assert add_kf_calls[0][2] == 0

        captured = capsys.readouterr().out
        assert "Getting BCs duration" in captured
        assert "Assembling K/F total duration" in captured
        assert "'Solid'" in captured

    def test_full_assembly_defers_bcs_when_model_additions_are_active(
        self, linear_elastic_params, monkeypatch
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()
        original_assemble_matrix = physics_kf_module.assemble_matrix
        assembled_bcs = []

        physics.models["Solid"].Add_KF = lambda K_mat, F_vec, step: None

        def _counted_assemble_matrix(matrix, form, bcs=None):
            assembled_bcs.append(bcs)
            return original_assemble_matrix(matrix, form, bcs=bcs)

        monkeypatch.setattr(
            physics_kf_module,
            "assemble_matrix",
            _counted_assemble_matrix,
        )

        physics.Get_Kmat_Fvec(0)

        assert assembled_bcs == [None]

    def test_get_kmat_fvec_stokes(self, stokes_params):
        mesh = Mesh(stokes_params)
        physics = Physics(mesh, stokes_params)
        physics.Initialize_Fields()

        K_mat, F_vec = physics.Get_Kmat_Fvec(0)

        assert K_mat is physics.K_mat
        assert F_vec is physics.F_vec
        assert K_mat.getSize()[0] == K_mat.getSize()[1] > 0
        assert F_vec.getSize() == K_mat.getSize()[0]

    def test_get_kmat_fvec_raises_clear_error_when_no_forms_are_assembled(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        physics.models["Solid"].Get_Weak_Form = (
            lambda time_constant, dt_constant, step: None
        )

        with pytest.raises(RuntimeError, match="Step 0 assembled no bilinear or linear form"):
            physics.Get_Kmat_Fvec(0)

    def test_get_kmat_fvec_raises_clear_error_when_bilinear_form_is_missing(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        K_form, F_form = physics.models["Solid"].assemble_KF(0)
        physics.models["Solid"].Get_Weak_Form = (
            lambda time_constant, dt_constant, step: (None, F_form)
        )

        with pytest.raises(RuntimeError, match="Step 0 assembled no bilinear form"):
            physics.Get_Kmat_Fvec(0)

    def test_get_fvec_allows_residual_without_bilinear_form(
        self, linear_elastic_params
    ):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        _, F_form = physics.models["Solid"].assemble_KF(0)
        physics.models["Solid"].Get_Weak_Form = (
            lambda time_constant, dt_constant, step: (None, F_form)
        )

        F_vec = physics.Get_Fvec(0)

        assert F_vec.getSize() > 0

    def test_get_kmat_fvec_raises_clear_error_when_linear_form_is_missing(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)
        physics.Initialize_Fields()

        K_form, F_form = physics.models["Solid"].assemble_KF(0)
        physics.models["Solid"].Get_Weak_Form = (
            lambda time_constant, dt_constant, step: (K_form, None)
        )

        with pytest.raises(RuntimeError, match="Step 0 assembled no linear form"):
            physics.Get_Kmat_Fvec(0)

    def test_update_global_measures_is_safe_for_models_without_hook(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)

        physics.Update_Global_Measures()

    def test_update_global_measures_calls_model_hook(self, linear_elastic_params):
        mesh = Mesh(linear_elastic_params)
        physics = Physics(mesh, linear_elastic_params)

        calls = []

        def _update_global_measures():
            calls.append("called")

        physics.models["Solid"].Update_Global_Measures = _update_global_measures

        physics.Update_Global_Measures()

        assert calls == ["called"]


class TestPhysicsKFWhiteBoxEdgeCases:
    def test_destroy_assembled_objects_clears_cached_workspaces_and_active_refs(self):
        harness = _PhysicsKFHarness()

        class _Destroyable:
            def __init__(self):
                self.destroyed = False

            def destroy(self):
                self.destroyed = True

        K0 = _Destroyable()
        F0 = _Destroyable()
        K1 = _Destroyable()
        F1 = _Destroyable()
        harness._assembly_cache = {
            0: {"K_mat": K0, "F_vec": F0},
            1: {"K_mat": K1, "F_vec": F1},
        }
        harness.K_mat = K1
        harness.F_vec = F1
        harness._compiled_form_cache = {
            (0, "K"): {"compiled": object()},
        }

        harness._destroy_assembled_objects()

        assert K0.destroyed is True
        assert F0.destroyed is True
        assert K1.destroyed is True
        assert F1.destroyed is True
        assert harness._assembly_cache == {}
        assert harness._compiled_form_cache == {}
        assert harness.K_mat is None
        assert harness.F_vec is None

    def test_destroy_assembled_objects_also_destroys_untracked_active_refs(self):
        harness = _PhysicsKFHarness()

        class _Destroyable:
            def __init__(self):
                self.destroyed = False

            def destroy(self):
                self.destroyed = True

        K_mat = _Destroyable()
        F_vec = _Destroyable()
        harness._assembly_cache = {}
        harness.K_mat = K_mat
        harness.F_vec = F_vec

        harness._destroy_assembled_objects()

        assert K_mat.destroyed
        assert F_vec.destroyed
        assert harness.K_mat is None
        assert harness.F_vec is None

    def test_get_kmat_fvec_warns_and_reallocates_when_reuse_fails(self, monkeypatch, capsys):
        harness = _PhysicsKFHarness()
        harness.DEBUG_PRINT_TIMING = False
        harness.mesh = type(
            "MeshStub",
            (),
            {
                "step_fields": {0: ["u"]},
                "fields": {"u": object()},
                "mixed_spaces": {0: None},
                "get_entity_maps": lambda self: [],
                "Is_Global_Field": lambda self, name: False,
            },
        )()
        harness.models = {
            "Solid": type(
                "ModelStub",
                (),
                {
                    "name": "Solid",
                    "Update_Weak_Constants": (
                        lambda self, time_constant, dt_constant, step: False
                    ),
                    "Get_Weak_Form": (
                        lambda self, time_constant, dt_constant, step: (
                            "K_form",
                            "F_form",
                        )
                    ),
                    "Add_KF": lambda self, K, F, step: None,
                },
            )()
        }
        harness._assembly_cache = {}
        harness._model_weak_form_cache = {}
        harness._sync_weak_constants = lambda: None
        harness.time_constant = 0.0
        harness.dt_constant = 1.0
        harness.K_mat = None
        harness.F_vec = None

        class _FakeVecLocalForm:
            def __init__(self, vec):
                self.vec = vec

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def set(self, value):
                self.vec.zeroed = True

        class _FakeVec:
            def __init__(self, name):
                self.name = name
                self.destroyed = False
                self.zeroed = False

            def localForm(self):
                return _FakeVecLocalForm(self)

            def ghostUpdate(self, **kwargs):
                pass

            def destroy(self):
                self.destroyed = True

        class _FakeMat:
            def __init__(self, name):
                self.name = name
                self.destroyed = False
                self.strict_reuse = False

            def zeroEntries(self):
                pass

            def assemble(self):
                pass

            def setOption(self, option, value):
                self.strict_reuse = bool(value)

            def destroy(self):
                self.destroyed = True

        counters = {"mat": 0, "vec": 0}

        def _fake_form(form, **kwargs):
            return form

        def _fake_create_matrix(form):
            counters["mat"] += 1
            return _FakeMat(f"K{counters['mat']}")

        def _fake_assemble_vector(*args):
            if len(args) == 1:
                counters["vec"] += 1
                return _FakeVec(f"F{counters['vec']}")
            vec, form = args
            return vec

        def _fake_assemble_matrix(mat, form, bcs=None):
            if mat.strict_reuse:
                raise RuntimeError("changing sparsity")
            return mat

        monkeypatch.setattr(physics_kf_module.fem, "form", _fake_form)
        monkeypatch.setattr(physics_kf_module, "create_matrix", _fake_create_matrix)
        monkeypatch.setattr(physics_kf_module, "assemble_vector", _fake_assemble_vector)
        monkeypatch.setattr(physics_kf_module, "assemble_matrix", _fake_assemble_matrix)

        first_K, first_F = harness.Get_Kmat_Fvec(0)
        second_K, second_F = harness.Get_Kmat_Fvec(0)

        assert first_K is not second_K
        assert first_F is not second_F
        assert first_K.destroyed is True
        assert first_F.destroyed is True
        assert harness.K_mat is second_K
        assert harness.F_vec is second_F
        assert "Reusing K/F allocation for step 0 failed" in capsys.readouterr().out

    def test_apply_bcs_to_matrix_handles_empty_rows_and_fallback_matrix_api(self):
        harness = _PhysicsKFHarness()

        class _BC:
            def dof_indices(self):
                return np.array([1, 4], dtype=np.int32), 0

        class _Matrix:
            def __init__(self):
                self.rows = None
                self.diag = None

            def zeroRowsColumns(self, rows, diag):
                self.rows = rows
                self.diag = diag

        matrix = _Matrix()
        harness.mesh = type("MeshStub", (), {"mixed_spaces": {0: None}})()
        harness._apply_bcs_to_matrix(matrix, [_BC()], 0)

        assert matrix.rows.size == 0
        assert matrix.diag == 1.0

    def test_get_fields_for_step_handles_missing_step_fields(self, capsys):
        harness = _PhysicsKFHarness()
        harness.mesh = type("MeshStub", (), {})()

        assert harness.get_fields_for_step(0) == []
        assert "No step fields registered in mesh" in capsys.readouterr().out

    def test_get_fields_for_step_warns_about_missing_registered_field(self, capsys):
        harness = _PhysicsKFHarness()
        field = object()
        harness.mesh = type(
            "MeshStub",
            (),
            {"step_fields": {0: ["u", "ghost"]}, "fields": {"u": field}},
        )()

        assert harness.get_fields_for_step(0) == [("u", field)]
        assert "registered but is missing from mesh.fields" in capsys.readouterr().out

    def test_get_boundary_conditions_handles_missing_step_fields(self):
        harness = _PhysicsKFHarness()
        harness.mesh = type("MeshStub", (), {})()
        harness.models = {}

        assert harness.get_boundary_conditions() == []

    def test_zero_vector_uses_set_when_local_form_is_unavailable(self):
        harness = _PhysicsKFHarness()

        class _Vec:
            def __init__(self):
                self.values = []

            def set(self, value):
                self.values.append(value)

        vec = _Vec()
        harness._zero_vector(vec)

        assert vec.values == [0.0]

    def test_zero_vector_raises_when_vec_cannot_be_cleared(self):
        harness = _PhysicsKFHarness()

        with pytest.raises(TypeError, match="Cannot zero PETSc vector"):
            harness._zero_vector(object())
