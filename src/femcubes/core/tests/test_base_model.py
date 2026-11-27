"""Tests for ``femcubes.core.base_model``."""

import pytest

from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate


class _DummyParams(ParamsTemplate):
    pass


class _RegisteredTestModel(BaseModel):
    MODEL_TYPE = "Test:Registered"

    def Get_Weak_Form(self, time_constant, dt_constant, step):
        return None


def test_get_model_class_returns_registered_model():
    assert BaseModel.get_model_class("Test:Registered") is _RegisteredTestModel


def test_get_model_class_reports_available_models_for_unknown_type():
    with pytest.raises(ValueError, match="Unknown model type: Test:Missing"):
        BaseModel.get_model_class("Test:Missing")

    try:
        BaseModel.get_model_class("Test:Missing")
    except ValueError as exc:
        message = str(exc)

    assert "Available:" in message
    assert "Test:Registered" in message


def test_base_model_init_sets_common_attributes():
    params = _DummyParams()
    mesh = object()
    model = _RegisteredTestModel("Dummy", params, mesh)

    assert model.name == "Dummy"
    assert model.params is params
    assert model.mesh is mesh
    assert model.type == "Test:Registered"


def test_base_model_get_weak_form_must_be_overridden():
    params = _DummyParams()
    model = BaseModel("Base", params, object())

    with pytest.raises(NotImplementedError, match="must implement Get_Weak_Form"):
        model.Get_Weak_Form(0.0, 1.0, 0)


def test_base_model_optional_hooks_default_to_noop():
    params = _DummyParams()
    model = _RegisteredTestModel("Dummy", params, object())

    assert model.initialize_fields() is None
    assert model.Post_Remesh() is None
    assert model.get_bcs_for_field("u") is None
    assert model.Update_Weak_Constants(0.0, 1.0, 0) is False
    assert model.commit() is None
    assert model.reset_step() is None
