"""Tests for ``femcubes.core._discover``."""

import sys
import types

import pytest

from femcubes.core import _discover
from femcubes.core._discover import discover_models, load_plugins
from femcubes.core.base_model import BaseModel
from femcubes.core.mesh.generator_base import MeshGeneratorBase


def test_discover_models_imports_non_test_modules(monkeypatch):
    walked = [
        (None, "femcubes.fake.alpha", False),
        (None, "femcubes.fake.tests.beta", False),
        (None, "femcubes.fake.test_gamma", False),
        (None, "femcubes.fake.delta", False),
    ]
    imported = []

    monkeypatch.setattr("femcubes.core._discover.pkgutil.walk_packages", lambda path, prefix: walked)
    monkeypatch.setattr("femcubes.core._discover.importlib.import_module", imported.append)

    discover_models("femcubes.fake", ["unused"])

    assert imported == ["femcubes.fake.alpha", "femcubes.fake.delta"]


def test_discover_models_raises_on_import_failure_by_default(monkeypatch):
    walked = [(None, "femcubes.fake.alpha", False)]

    def _raise_import_error(modname):
        raise RuntimeError("boom")

    monkeypatch.setattr("femcubes.core._discover.pkgutil.walk_packages", lambda path, prefix: walked)
    monkeypatch.setattr("femcubes.core._discover.importlib.import_module", _raise_import_error)

    with pytest.raises(ImportError, match="Failed to import model module femcubes.fake.alpha: boom"):
        discover_models("femcubes.fake", ["unused"])

def test_discover_models_can_warn_in_non_strict_mode(monkeypatch):
    walked = [(None, "femcubes.fake.alpha", False)]

    def _raise_import_error(modname):
        raise RuntimeError("boom")

    monkeypatch.setattr("femcubes.core._discover.pkgutil.walk_packages", lambda path, prefix: walked)
    monkeypatch.setattr("femcubes.core._discover.importlib.import_module", _raise_import_error)

    with pytest.warns(UserWarning, match="Failed to import model module femcubes.fake.alpha: boom"):
        discover_models("femcubes.fake", ["unused"], strict=False)


class _FakeEntryPoint:
    """Stand-in for ``importlib.metadata.EntryPoint``."""

    def __init__(self, name, loader):
        self.name = name
        self.value = f"fake_plugin_{name}"
        self._loader = loader

    def load(self):
        return self._loader()


@pytest.fixture
def plugin_entry_points(monkeypatch):
    """Replace the installed entry points by a list the test fills in."""
    entry_points = []
    queried_groups = []

    def _entry_points(*, group):
        queried_groups.append(group)
        return list(entry_points)

    monkeypatch.setattr(_discover.importlib.metadata, "entry_points", _entry_points)
    monkeypatch.setattr(_discover, "_handled_plugins", set())
    return types.SimpleNamespace(items=entry_points, groups=queried_groups)


def test_load_plugins_imports_each_plugin_once(plugin_entry_points):
    calls = []
    plugin_entry_points.items.append(_FakeEntryPoint("beta", lambda: calls.append("beta")))
    plugin_entry_points.items.append(_FakeEntryPoint("alpha", lambda: calls.append("alpha")))

    assert load_plugins() == ["alpha", "beta"]
    assert load_plugins() == []

    assert calls == ["alpha", "beta"]
    assert set(plugin_entry_points.groups) == {"femcubes.plugins"}


def test_load_plugins_picks_up_plugins_added_later(plugin_entry_points):
    calls = []
    plugin_entry_points.items.append(_FakeEntryPoint("alpha", lambda: calls.append("alpha")))
    load_plugins()
    plugin_entry_points.items.append(_FakeEntryPoint("beta", lambda: calls.append("beta")))

    assert load_plugins() == ["beta"]
    assert calls == ["alpha", "beta"]


def test_load_plugins_warns_on_broken_plugin_and_loads_the_rest(plugin_entry_points):
    calls = []

    def _broken():
        raise RuntimeError("boom")

    plugin_entry_points.items.append(_FakeEntryPoint("broken", _broken))
    plugin_entry_points.items.append(_FakeEntryPoint("working", lambda: calls.append("working")))

    with pytest.warns(UserWarning, match="Failed to import FEMCubes plugin 'broken'.*boom"):
        loaded = load_plugins()

    assert loaded == ["working"]
    assert calls == ["working"]


def test_load_plugins_raises_on_broken_plugin_in_strict_mode(plugin_entry_points):
    def _broken():
        raise RuntimeError("boom")

    plugin_entry_points.items.append(_FakeEntryPoint("broken", _broken))

    with pytest.raises(ImportError, match="Failed to import FEMCubes plugin 'broken'"):
        load_plugins(strict=True)


def test_unknown_model_type_is_looked_up_in_plugins(plugin_entry_points, monkeypatch):
    monkeypatch.setattr(BaseModel, "_registry", dict(BaseModel._registry))

    def _define_model():
        class PluginModel(BaseModel):
            MODEL_TYPE = "TestPlugin:Model"

        return PluginModel

    plugin_entry_points.items.append(_FakeEntryPoint("models", _define_model))

    assert "TestPlugin:Model" not in BaseModel._registry
    model_cls = BaseModel.get_model_class("TestPlugin:Model")
    assert model_cls.MODEL_TYPE == "TestPlugin:Model"


def test_unknown_mesh_type_is_looked_up_in_plugins(plugin_entry_points, monkeypatch):
    monkeypatch.setattr(MeshGeneratorBase, "_registry", dict(MeshGeneratorBase._registry))

    def _define_generator():
        class PluginGenerator(MeshGeneratorBase):
            MESH_TYPE = "TestPluginMesh"

        return PluginGenerator

    plugin_entry_points.items.append(_FakeEntryPoint("meshes", _define_generator))

    assert "TestPluginMesh" not in MeshGeneratorBase._registry
    assert MeshGeneratorBase.get_generator("TestPluginMesh").MESH_TYPE == "TestPluginMesh"


def test_unknown_type_without_plugin_still_lists_the_available_types(plugin_entry_points):
    with pytest.raises(ValueError, match="Unknown model type: Nope:Missing. Available:"):
        BaseModel.get_model_class("Nope:Missing")
    assert plugin_entry_points.groups == ["femcubes.plugins"]


def test_plugin_entry_point_of_an_installed_distribution_is_imported(tmp_path, monkeypatch):
    """A real ``.dist-info`` entry point on ``sys.path`` is found and imported."""
    (tmp_path / "femcubes_testplugin.py").write_text("LOADED = True\n")
    dist_info = tmp_path / "femcubes_testplugin-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: femcubes-testplugin\nVersion: 0.1\n"
    )
    (dist_info / "entry_points.txt").write_text(
        "[femcubes.plugins]\ntestplugin = femcubes_testplugin\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_discover, "_handled_plugins", set())
    monkeypatch.delitem(sys.modules, "femcubes_testplugin", raising=False)

    try:
        assert "testplugin" in load_plugins()
        assert sys.modules["femcubes_testplugin"].LOADED is True
    finally:
        sys.modules.pop("femcubes_testplugin", None)
