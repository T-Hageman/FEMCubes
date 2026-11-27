"""Tests for ``femcubes.core.params_template``."""

import pytest

from femcubes.core.params_template import ParamsTemplate


class _DefaultParams(ParamsTemplate):
    pass


class _ConfiguredParams(ParamsTemplate):
    Global_Measures = {"Damage": 1.0}
    line_search_lims = (0.2, 0.8)
    output_fields = ("u", "p")
    Plot_Fields = ("u_x",)
    Plot_RO_Fields = ("u_x", "h")
    Plot_Measures = ("Damage",)


class _ElasticParams(ParamsTemplate):
    Materials = {
        "Ice": {
            "Youngs": 10.0,
            "Poisson": 0.25,
        }
    }


def test_params_template_initializes_fresh_default_mutables():
    first = _DefaultParams()
    second = _DefaultParams()

    assert first.Global_Measures == {}
    assert first.PROFILE_TIMINGS is False
    assert first.LBFGS_History == 10
    assert first.LBFGS_LineSearch_MaxIter == 20
    assert first.LBFGS_LineSearch_Tolerance == pytest.approx(0.9)
    assert first.LBFGS_LineSearch_MaxResidualGrowth == pytest.approx(2.0)
    assert first.LBFGS_StagnationIterations == 5
    assert first.LBFGS_RecoveryLineSearchTolerance == pytest.approx(0.1)
    assert first.LBFGS_PowellDamping == pytest.approx(0.2)
    assert first.line_search_lims == [0.01, 1.0]
    assert first.output_fields == []
    assert first.Plot_Fields == []
    assert first.Plot_RO_Fields == []
    assert first.Plot_Measures == []

    first.Global_Measures["Damage"] = 2.0
    first.line_search_lims.append(2.0)
    first.output_fields.append("u")
    first.Plot_Fields.append("u_x")
    first.Plot_RO_Fields.append("u_x")
    first.Plot_Measures.append("Damage")

    assert second.Global_Measures == {}
    assert second.line_search_lims == [0.01, 1.0]
    assert second.output_fields == []
    assert second.Plot_Fields == []
    assert second.Plot_RO_Fields == []
    assert second.Plot_Measures == []


def test_params_template_copies_configured_mutables_to_instance_lists_and_dicts():
    params = _ConfiguredParams()

    assert params.Global_Measures == {"Damage": 1.0}
    assert params.line_search_lims == [0.2, 0.8]
    assert params.output_fields == ["u", "p"]
    assert params.Plot_Fields == ["u_x"]
    assert params.Plot_RO_Fields == ["u_x", "h"]
    assert params.Plot_Measures == ["Damage"]

    assert isinstance(params.Global_Measures, dict)
    assert isinstance(params.line_search_lims, list)
    assert isinstance(params.output_fields, list)
    assert isinstance(params.Plot_Fields, list)
    assert isinstance(params.Plot_RO_Fields, list)
    assert isinstance(params.Plot_Measures, list)


def test_set_elastic_params_derives_bulk_lame_and_shear():
    params = _ElasticParams()

    params._SetElasticParams("Ice")

    assert params.Materials["Ice"]["Bulk"] == pytest.approx(6.666666666666667)
    assert params.Materials["Ice"]["Lame"] == pytest.approx(4.0)
    assert params.Materials["Ice"]["Shear"] == pytest.approx(4.0)
