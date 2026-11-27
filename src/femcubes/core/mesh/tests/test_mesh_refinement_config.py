"""Tests for reusable mesh-refinement configuration contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib

import pytest

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.refinement import (
    FieldThresholdIndicatorConfig,
    RefinementConfig,
    normalize_refinement_config,
)


def _indicator(**overrides):
    values = {"field": "phasefield", "threshold": 0.1}
    values.update(overrides)
    return FieldThresholdIndicatorConfig(**values)


def _enabled_config(**overrides):
    values = {"enabled": True, "indicator": _indicator(), "target_h": 0.25}
    values.update(overrides)
    return RefinementConfig(**values)


def test_refinement_package_exports_configuration_contracts():
    assert FieldThresholdIndicatorConfig.__module__.endswith("refinement.config")
    assert RefinementConfig.__module__.endswith("refinement.config")
    assert callable(normalize_refinement_config)


def test_refinement_package_resolves_configuration_exports_lazily(monkeypatch):
    package = importlib.reload(importlib.import_module("femcubes.core.mesh.refinement"))
    for export_name in package.__all__:
        package.__dict__.pop(export_name, None)

    imports = []

    def _fake_import_module(module_name):
        imports.append(module_name)

        class _StubModule:
            RefinementConfig = "refinement-config"

        return _StubModule()

    monkeypatch.setattr(package, "import_module", _fake_import_module)

    assert package.RefinementConfig == "refinement-config"
    assert imports == ["femcubes.core.mesh.refinement.config"]
    assert "CellIndicatorResult" in dir(package)
    package.__dict__.pop("RefinementConfig", None)
    importlib.reload(package)


def test_refinement_defaults_are_disabled_and_immutable():
    config = RefinementConfig()

    assert config.enabled is False
    assert config.indicator is None
    assert config.target_h is None
    assert config.buffer_layers == 1
    assert config.check_interval == 1
    assert config.max_levels_per_step == 1
    assert config.max_events is None
    assert config.start_step == 0
    assert config.repartition is False
    assert config.repartition_imbalance_threshold == pytest.approx(1.5)
    assert config.replot_on_refinement is False
    assert config.save_remesh_animations is False
    assert config.interpolation_padding == pytest.approx(1.0e-12)

    with pytest.raises(FrozenInstanceError):
        config.enabled = True


def test_valid_typed_configuration_retains_indicator_and_visualization_options():
    indicator = _indicator(
        threshold=2.5,
        reduction="cell_mean",
        comparison="less_equal",
        use_absolute_value=True,
    )
    config = RefinementConfig(
        enabled=True,
        indicator=indicator,
        target_h=0.125,
        buffer_layers=2,
        check_interval=3,
        max_levels_per_step=2,
        max_events=7,
        start_step=4,
        repartition=True,
        repartition_imbalance_threshold=2.25,
        replot_on_refinement=True,
        save_remesh_animations=True,
        interpolation_padding=0.0,
    )

    assert config.indicator is indicator
    assert config.target_h == pytest.approx(0.125)
    assert config.buffer_layers == 2
    assert config.check_interval == 3
    assert config.max_levels_per_step == 2
    assert config.max_events == 7
    assert config.start_step == 4
    assert config.repartition is True
    assert config.repartition_imbalance_threshold == pytest.approx(2.25)
    assert config.replot_on_refinement is True
    assert config.save_remesh_animations is True
    assert config.interpolation_padding == 0.0


def test_normalize_accepts_none_and_returns_typed_input_unchanged():
    assert normalize_refinement_config(None) == RefinementConfig()

    config = _enabled_config()
    assert normalize_refinement_config(config) is config


def test_normalize_accepts_modern_nested_mapping_without_mutating_it():
    indicator_mapping = {
        "field": "PsiField",
        "threshold": 10.0,
        "reduction": "cell_max",
        "comparison": "greater",
    }
    mapping = {
        "indicator": indicator_mapping,
        "target_h": 0.5,
        "replot_on_refinement": False,
        "save_remesh_animations": True,
    }

    config = normalize_refinement_config(mapping)

    assert config.enabled is True
    assert config.indicator == FieldThresholdIndicatorConfig(**indicator_mapping)
    assert config.target_h == pytest.approx(0.5)
    assert config.replot_on_refinement is False
    assert config.save_remesh_animations is True
    assert mapping["indicator"] is indicator_mapping
    assert "enabled" not in mapping


def test_empty_mapping_selects_disabled_default():
    assert normalize_refinement_config({}) == RefinementConfig()


def test_enabled_refinement_requires_indicator_and_target_h():
    with pytest.raises(ValueError, match="indicator.*required"):
        RefinementConfig(enabled=True, target_h=0.25)

    with pytest.raises(ValueError, match="target_h.*required"):
        RefinementConfig(enabled=True, indicator=_indicator())

    with pytest.raises(TypeError, match="indicator.*FieldThresholdIndicatorConfig"):
        RefinementConfig(indicator="phasefield")


@pytest.mark.parametrize("reduction", ["cell_max", "cell_min", "cell_mean"])
@pytest.mark.parametrize("comparison", ["greater_equal", "greater", "less_equal", "less"])
def test_all_documented_indicator_modes_are_valid(reduction, comparison):
    indicator = _indicator(reduction=reduction, comparison=comparison)
    assert indicator.reduction == reduction
    assert indicator.comparison == comparison


def test_indicator_rejects_invalid_field_threshold_and_modes():
    with pytest.raises(ValueError, match="non-empty string"):
        _indicator(field="")
    with pytest.raises(ValueError, match="leading or trailing"):
        _indicator(field=" phasefield")
    with pytest.raises(ValueError, match="finite real"):
        _indicator(threshold=float("nan"))
    with pytest.raises(ValueError, match="Unknown refinement reduction"):
        _indicator(reduction="maximum")
    with pytest.raises(ValueError, match="Unknown refinement comparison"):
        _indicator(comparison="approximately")
    with pytest.raises(TypeError, match="use_absolute_value.*bool"):
        _indicator(use_absolute_value=1)


@pytest.mark.parametrize("target_h", [0.0, -1.0, float("inf"), float("nan"), True])
def test_refinement_rejects_invalid_target_h(target_h):
    with pytest.raises(ValueError, match="target_h"):
        _enabled_config(target_h=target_h)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("buffer_layers", -1),
        ("buffer_layers", 1.5),
        ("check_interval", 0),
        ("max_levels_per_step", 0),
        ("max_events", 0),
        ("start_step", -1),
        ("repartition_imbalance_threshold", 0.999),
        ("repartition_imbalance_threshold", float("nan")),
        ("repartition_imbalance_threshold", float("inf")),
        ("repartition_imbalance_threshold", True),
        ("interpolation_padding", -1.0),
    ],
)
def test_refinement_rejects_invalid_limits(name, value):
    with pytest.raises(ValueError, match=name):
        _enabled_config(**{name: value})


@pytest.mark.parametrize(
    "name",
    ["enabled", "repartition", "replot_on_refinement", "save_remesh_animations"],
)
def test_refinement_boolean_options_require_booleans(name):
    values = {
        "enabled": True,
        "indicator": _indicator(),
        "target_h": 0.25,
        name: 1,
    }
    with pytest.raises(TypeError, match=f"{name}.*bool"):
        RefinementConfig(**values)


def test_animation_is_independent_of_replotting_and_field_movie_configuration():
    config = _enabled_config(replot_on_refinement=False, save_remesh_animations=True)

    assert config.replot_on_refinement is False
    assert config.save_remesh_animations is True
    assert "record_movie" not in RefinementConfig.__dataclass_fields__


def test_repartitioning_accepts_an_explicit_imbalance_trigger():
    config = _enabled_config(
        repartition=True,
        repartition_imbalance_threshold=1.25,
    )

    assert config.repartition is True
    assert config.repartition_imbalance_threshold == pytest.approx(1.25)


def test_normalize_rejects_unknown_and_unsupported_inputs():
    with pytest.raises(ValueError, match="Unknown MeshRefinement parameter.*mystery"):
        normalize_refinement_config({"mystery": 1})
    with pytest.raises(ValueError, match="Unknown MeshRefinement parameter") as exc_info:
        normalize_refinement_config(
            {"Variable": "phasefield", "Threshold": 0.1, "dx_fine": 0.25}
        )
    assert all(name in str(exc_info.value) for name in ("Variable", "Threshold", "dx_fine"))
    with pytest.raises(TypeError, match="RefinementConfig, mapping, or None"):
        normalize_refinement_config("phasefield")
    with pytest.raises(TypeError, match="indicator.*mapping"):
        normalize_refinement_config({"indicator": "phasefield", "target_h": 0.25})
    with pytest.raises(ValueError, match="Invalid refinement indicator mapping"):
        normalize_refinement_config(
            {
                "indicator": {
                    "field": "phasefield",
                    "threshold": 0.1,
                    "unknown": True,
                },
                "target_h": 0.25,
            }
        )
    with pytest.raises(ValueError, match="Invalid MeshRefinement mapping"):
        normalize_refinement_config({"enabled": 1})


def test_params_template_supplies_disabled_typed_default():
    class Params(ParamsTemplate):
        pass

    first = Params()
    second = Params()

    assert first.MeshRefinement == RefinementConfig()
    assert second.MeshRefinement == RefinementConfig()


def test_params_template_normalizes_modern_mapping_per_instance():
    class Params(ParamsTemplate):
        MeshRefinement = {
            "indicator": {"field": "phasefield", "threshold": 0.1},
            "target_h": 0.25,
        }

    first = Params()
    second = Params()

    assert first.MeshRefinement == _enabled_config()
    assert second.MeshRefinement == first.MeshRefinement
    assert first.MeshRefinement is not Params.MeshRefinement
