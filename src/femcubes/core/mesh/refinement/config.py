"""Configuration contracts for adaptive mesh refinement.

This module deliberately has no DOLFINx dependency.  Refinement inputs can
therefore be validated while parameter objects are constructed, before a mesh
or physics model exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any


_REDUCTIONS = frozenset({"cell_max", "cell_min", "cell_mean"})
_COMPARISONS = frozenset({"greater_equal", "greater", "less_equal", "less"})
_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "indicator",
        "target_h",
        "buffer_layers",
        "check_interval",
        "max_levels_per_step",
        "max_events",
        "start_step",
        "repartition",
        "repartition_imbalance_threshold",
        "replot_on_refinement",
        "save_remesh_animations",
        "interpolation_padding",
    }
)


@dataclass(frozen=True)
class FieldThresholdIndicatorConfig:
    """Select cells by reducing and thresholding a registered scalar field.

    Parameters
    ----------
    field : str
        Name of a registered CG, DG, or quadrature field.
    threshold : float
        Finite threshold used by ``comparison``.
    reduction : {"cell_max", "cell_min", "cell_mean"}
        Reduction applied to field samples within each cell.
    comparison : {"greater_equal", "greater", "less_equal", "less"}
        Comparison between the reduced cell value and ``threshold``.
    use_absolute_value : bool
        Apply the reduction to absolute sample values when true.
    """

    field: str
    threshold: float
    reduction: str = "cell_max"
    comparison: str = "greater_equal"
    use_absolute_value: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("Refinement indicator 'field' must be a non-empty string.")
        if self.field != self.field.strip():
            raise ValueError("Refinement indicator 'field' must not contain leading or trailing whitespace.")
        _validate_finite_real("Refinement indicator 'threshold'", self.threshold)
        if self.reduction not in _REDUCTIONS:
            choices = ", ".join(sorted(_REDUCTIONS))
            raise ValueError(f"Unknown refinement reduction {self.reduction!r}. Expected one of: {choices}.")
        if self.comparison not in _COMPARISONS:
            choices = ", ".join(sorted(_COMPARISONS))
            raise ValueError(f"Unknown refinement comparison {self.comparison!r}. Expected one of: {choices}.")
        _validate_bool("Refinement indicator 'use_absolute_value'", self.use_absolute_value)


@dataclass(frozen=True)
class RefinementConfig:
    """Validated configuration for refinement-only adaptive meshing.

    Refinement is disabled by default.  When enabled, both ``indicator`` and
    ``target_h`` are required. Optional repartitioning is performed only after
    refinement fields and markers have first been transferred to the
    identity-partitioned refined mesh.

    ``buffer_layers`` sets a physical padding radius of
    ``buffer_layers * target_h`` around all threshold-selected cells.
    It does not count layers of cells in the current mesh.

    ``interpolation_padding`` is retained for existing configurations;
    ancestry-based field transfers do not use geometric search padding.
    """

    enabled: bool = False
    indicator: FieldThresholdIndicatorConfig | None = None
    target_h: float | None = None
    buffer_layers: int = 1
    check_interval: int = 1
    max_levels_per_step: int = 1
    max_events: int | None = None
    start_step: int = 0
    repartition: bool = False
    repartition_imbalance_threshold: float = 1.5
    replot_on_refinement: bool = False
    save_remesh_animations: bool = False
    interpolation_padding: float = 1.0e-12

    def __post_init__(self) -> None:
        _validate_bool("Refinement 'enabled'", self.enabled)
        _validate_bool("Refinement 'repartition'", self.repartition)
        _validate_bool("Refinement 'replot_on_refinement'", self.replot_on_refinement)
        _validate_bool("Refinement 'save_remesh_animations'", self.save_remesh_animations)

        if self.indicator is not None and not isinstance(self.indicator, FieldThresholdIndicatorConfig):
            raise TypeError("Refinement 'indicator' must be a FieldThresholdIndicatorConfig or None.")

        if self.target_h is not None:
            _validate_positive_real("Refinement 'target_h'", self.target_h)
        _validate_nonnegative_int("Refinement 'buffer_layers'", self.buffer_layers)
        _validate_positive_int("Refinement 'check_interval'", self.check_interval)
        _validate_positive_int("Refinement 'max_levels_per_step'", self.max_levels_per_step)
        if self.max_events is not None:
            _validate_positive_int("Refinement 'max_events'", self.max_events)
        _validate_nonnegative_int("Refinement 'start_step'", self.start_step)
        _validate_at_least_one_real(
            "Refinement 'repartition_imbalance_threshold'",
            self.repartition_imbalance_threshold,
        )
        _validate_nonnegative_real("Refinement 'interpolation_padding'", self.interpolation_padding)

        if self.enabled and self.indicator is None:
            raise ValueError("Refinement 'indicator' is required when refinement is enabled.")
        if self.enabled and self.target_h is None:
            raise ValueError("Refinement 'target_h' is required when refinement is enabled.")


def normalize_refinement_config(value: RefinementConfig | Mapping[str, Any] | None) -> RefinementConfig:
    """Return a validated :class:`RefinementConfig` from supported input.

    ``None`` selects the disabled default.  Modern mappings use the dataclass
    field names and may supply ``indicator`` either as a typed object or a
    nested mapping.

    The input mapping is copied and never mutated.
    """

    if value is None:
        return RefinementConfig()
    if isinstance(value, RefinementConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("MeshRefinement must be a RefinementConfig, mapping, or None.")

    config_values = dict(value)
    unknown = set(config_values) - _CONFIG_KEYS
    if unknown:
        names = ", ".join(sorted(repr(name) for name in unknown))
        raise ValueError(f"Unknown MeshRefinement parameter(s): {names}.")

    indicator = config_values.get("indicator")
    if isinstance(indicator, Mapping):
        try:
            config_values["indicator"] = FieldThresholdIndicatorConfig(**dict(indicator))
        except TypeError as exc:
            raise ValueError(f"Invalid refinement indicator mapping: {exc}") from exc
    elif indicator is not None and not isinstance(indicator, FieldThresholdIndicatorConfig):
        raise TypeError("MeshRefinement 'indicator' must be a mapping or FieldThresholdIndicatorConfig.")

    if "enabled" not in config_values:
        config_values["enabled"] = "indicator" in config_values or "target_h" in config_values

    try:
        return RefinementConfig(**config_values)
    except TypeError as exc:
        raise ValueError(f"Invalid MeshRefinement mapping: {exc}") from exc


def _validate_bool(name: str, value: Any) -> None:
    """PRIVATE: Require a real boolean rather than relying on truthiness."""

    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_finite_real(name: str, value: Any) -> None:
    """PRIVATE: Require a finite real scalar, excluding booleans."""

    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number.")


def _validate_positive_real(name: str, value: Any) -> None:
    """PRIVATE: Require a finite real scalar greater than zero."""

    _validate_finite_real(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _validate_nonnegative_real(name: str, value: Any) -> None:
    """PRIVATE: Require a finite real scalar greater than or equal to zero."""

    _validate_finite_real(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _validate_at_least_one_real(name: str, value: Any) -> None:
    """PRIVATE: Require a finite real scalar greater than or equal to one."""

    _validate_finite_real(name, value)
    if value < 1:
        raise ValueError(f"{name} must be greater than or equal to one.")


def _validate_positive_int(name: str, value: Any) -> None:
    """PRIVATE: Require an integer greater than zero, excluding booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer greater than or equal to one.")


def _validate_nonnegative_int(name: str, value: Any) -> None:
    """PRIVATE: Require a nonnegative integer, excluding booleans."""

    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
