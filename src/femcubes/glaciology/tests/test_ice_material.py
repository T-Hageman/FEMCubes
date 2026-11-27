"""Tests for ice-specific material helper relations."""

import numpy as np
import pytest

from femcubes.glaciology.ice_material import IceMaterial

R = 8.31446261815324
N = 3.0
T_GLEN_TRANSITION = 263.15
A_GLEN_COLD = 3.615e-13
Q_GLEN_COLD = 60e3
Q_GLEN_TEMPERATE = 139e3
A_GLEN_TRANSITION = A_GLEN_COLD * np.exp(
    -Q_GLEN_COLD / (R * T_GLEN_TRANSITION)
)
POROSITY_GLEN_BETA = 4.0


def test_elastic_and_strength_relations_are_consistent():
    material = IceMaterial()
    porosity = 0.2
    temperature = -5.0

    youngs = 9.5e9 * (1.0 - porosity) ** 4
    poisson = 0.35

    np.testing.assert_allclose(
        material.Get_Bulk_Modulus(porosity, temperature),
        youngs / (3.0 * (1.0 - 2.0 * poisson)),
    )
    np.testing.assert_allclose(
        material.Get_Shear_Modulus(porosity, temperature),
        youngs / (2.0 * (1.0 + poisson)),
    )
    np.testing.assert_allclose(
        material.Get_Lame_Parameter(porosity, temperature),
        youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
    )
    np.testing.assert_allclose(material.Get_Tensile_Strength(porosity, temperature), 0.2e6)
    np.testing.assert_allclose(material.Get_Creep_n(porosity, temperature), N)


@pytest.mark.parametrize(
    ("pf_formulation", "gc_factor"),
    [
        ("AT2", None),
        ("HO-AT2", 256.0 / 27.0),
        ("AT1", 8.0 / 3.0),
        ("HO-AT1", 8.0 / 3.0),
    ],
)
def test_fracture_energy_uses_phasefield_formulation(pf_formulation, gc_factor):
    material = IceMaterial()
    porosity = 0.2
    temperature = -5.0
    ell = 0.1

    tensile_strength = material.Get_Tensile_Strength(porosity, temperature)
    youngs_modulus = material.Get_Youngs_Modulus(porosity, temperature)
    if pf_formulation == "AT2":
        # IceMaterial currently reserves AT2 for its prescribed cohesive toughness.
        expected = 10.0 * (1.0 - porosity)
    else:
        expected = gc_factor * tensile_strength**2 * ell / youngs_modulus
    np.testing.assert_allclose(
        material.Get_Gc(porosity, temperature, ell, PF_formulation=pf_formulation),
        expected,
    )


def test_fracture_energy_rejects_unknown_phasefield_formulation():
    material = IceMaterial()

    with pytest.raises(ValueError, match="Unknown Phase Field formulation"):
        material.Get_Gc(0.0, -10.0, 0.1, PF_formulation="Unknown")


def test_creep_a_uses_cold_and_temperate_arrhenius_branches():
    material = IceMaterial()

    cold_temperature = -20.0
    cold_T = cold_temperature + 273.15
    cold_expected = A_GLEN_TRANSITION * np.exp(
        -Q_GLEN_COLD
        / R
        * (1.0 / cold_T - 1.0 / T_GLEN_TRANSITION)
    )

    temperate_temperature = -5.0
    temperate_T = temperate_temperature + 273.15
    temperate_expected = A_GLEN_TRANSITION * np.exp(
        -Q_GLEN_TEMPERATE
        / R
        * (1.0 / temperate_T - 1.0 / T_GLEN_TRANSITION)
    )

    np.testing.assert_allclose(
        material.Get_Creep_A(0.0, cold_temperature),
        cold_expected,
    )
    np.testing.assert_allclose(
        material.Get_Creep_A(0.0, temperate_temperature),
        temperate_expected,
    )


def test_creep_a_is_continuous_at_branch_transition():
    material = IceMaterial()

    np.testing.assert_allclose(
        material.Get_Creep_A(0.0, -10.0),
        A_GLEN_TRANSITION,
    )


def test_creep_a_preserves_porosity_softening():
    material = IceMaterial()
    porosity = 0.1

    dense_a = material.Get_Creep_A(0.0, -15.0)
    porous_a = material.Get_Creep_A(porosity, -15.0)

    expected_factor = np.exp(POROSITY_GLEN_BETA * porosity / (1.0 - porosity))
    np.testing.assert_allclose(porous_a, dense_a * expected_factor)


def test_stokes_viscosity_returns_glen_base_coefficient():
    material = IceMaterial()
    temperature = -5.0

    A = material.Get_Creep_A(0.0, temperature)
    expected = 0.5 * A ** (-1.0 / N)

    np.testing.assert_allclose(
        material.Get_Viscosity_Stokes(0.0, temperature),
        expected,
    )


def test_creep_relations_accept_ufl_expressions():
    ufl = pytest.importorskip("ufl")
    material = IceMaterial()

    porosity = ufl.variable(ufl.as_ufl(0.1))
    temperature = ufl.variable(ufl.as_ufl(-5.0))

    creep_a = material.Get_Creep_A(porosity, temperature)
    stokes_viscosity = material.Get_Viscosity_Stokes(porosity, temperature)
    viscoplastic_viscosity = material.Get_ViscoPlastic_viscosity(
        porosity,
        temperature,
    )

    assert isinstance(creep_a, ufl.core.expr.Expr)
    assert isinstance(stokes_viscosity, ufl.core.expr.Expr)
    assert isinstance(viscoplastic_viscosity, ufl.core.expr.Expr)


def test_plot_relations_creates_requested_temperature_porosity_plot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    material = IceMaterial()

    fig, ax = material.Plot_Relations(
        ToPlot="Get_Poisson",
        num_temperature=4,
        num_porosity=3,
        show=False,
    )

    try:
        assert ax.get_title() == "Poisson"
        assert ax.get_xlabel() == "Temperature [C]"
        assert ax.get_ylabel() == "Porosity [-]"
        assert len(fig.axes) == 2
        assert len(ax.collections) == 1
    finally:
        plt.close(fig)


def test_plot_relations_returns_axis_array_for_multiple_relations():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    material = IceMaterial()

    fig, axes = material.Plot_Relations(
        ToPlot=["Density", "Get_Youngs_Modulus"],
        num_temperature=4,
        num_porosity=3,
        show=False,
    )

    try:
        assert axes.shape == (2,)
        assert [ax.get_title() for ax in axes] == ["Density", "Youngs Modulus"]
        assert len(fig.axes) == 4
    finally:
        plt.close(fig)


def test_plot_relations_defaults_to_all_relations_and_removes_unused_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    material = IceMaterial()

    fig, axes = material.Plot_Relations(
        num_temperature=2,
        num_porosity=2,
        show=False,
    )

    try:
        assert axes.shape == (10,)
        assert all(ax.get_title() for ax in axes)
    finally:
        plt.close(fig)


def test_plot_relations_rejects_unknown_relation():
    material = IceMaterial()

    with pytest.raises(ValueError, match="Unknown material relation"):
        material.Plot_Relations(ToPlot="Thermal_Conductivity", show=False)


def test_plot_relations_rejects_empty_relation_list():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    material = IceMaterial()

    with pytest.raises(ValueError, match="at least one relation"):
        material.Plot_Relations(ToPlot=[], show=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature_range": (-10.0,)}, "temperature_range must contain"),
        ({"porosity_range": (0.0, 0.5, 0.6)}, "porosity_range must contain"),
        ({"temperature_range": (0.0, -1.0)}, "temperature_range must be"),
        ({"temperature_range": (-274.0, -273.0)}, "above absolute zero"),
        ({"porosity_range": (-0.1, 0.5)}, "0 <= porosity < 1"),
        ({"porosity_range": (0.0, 1.0)}, "0 <= porosity < 1"),
        ({"porosity_range": (0.5, 0.5)}, "porosity_range must be"),
        ({"num_temperature": 1}, "num_temperature must be"),
        ({"num_porosity": 1}, "num_porosity must be"),
    ],
)
def test_plot_relations_rejects_invalid_grid_options(kwargs, message):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    material = IceMaterial()

    with pytest.raises(ValueError, match=message):
        material.Plot_Relations(ToPlot="Density", show=False, **kwargs)


def test_plot_relations_show_is_non_blocking_by_default(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    material = IceMaterial()
    show_call = {}
    pause_call = {}

    def fake_show(*, block=None):
        show_call["block"] = block

    def fake_pause(interval):
        pause_call["interval"] = interval

    monkeypatch.setattr(plt, "show", fake_show)
    monkeypatch.setattr(plt, "pause", fake_pause)

    fig, ax = material.Plot_Relations(
        ToPlot="Density",
        num_temperature=4,
        num_porosity=3,
    )

    try:
        assert show_call == {"block": False}
        assert pause_call == {"interval": 0.1}
    finally:
        plt.close(fig)


def test_color_normalisation_handles_edge_cases():
    matplotlib = pytest.importorskip("matplotlib")
    from matplotlib.colors import LogNorm

    material = IceMaterial()

    assert material._Get_Color_Normalisation(np.array([np.nan]), LogNorm) is None
    assert material._Get_Color_Normalisation(np.array([0.0, 1.0]), LogNorm) is None

    norm = material._Get_Color_Normalisation(np.array([1.0, 1.0e4]), LogNorm)
    assert isinstance(norm, LogNorm)


def test_tensile_strength_is_constant_with_the_shape_of_the_inputs():
    material = IceMaterial()
    temperature = np.linspace(-50.0, 0.0, 6)
    np.testing.assert_array_equal(material.Get_Tensile_Strength(0.0, temperature), np.full(6, 0.2e6))
    np.testing.assert_array_equal(material.Get_Tensile_Strength(np.zeros((2, 1)), temperature), np.full((2, 6), 0.2e6))
    assert material.Get_Tensile_Strength(0.3, -10.0) == material.TENSILE_STRENGTH


@pytest.mark.parametrize("dim", [2, 3])
def test_viscoplastic_stress_scale_gives_glens_creep_rate_in_the_dimension(dim):
    material = IceMaterial()
    A = material.Get_Creep_A(0.1, -15.0)
    eta = material.Get_ViscoPlastic_viscosity(0.1, -15.0, dim=dim)
    # f = sqrt(d/2 s:s) = sqrt(d) tau_e; the flow rule (f/eta)^n df/dsigma must equal A tau_e^(n-1) s
    tau = 1.0e5
    rate = (np.sqrt(dim) * tau / eta) ** N * (dim / 2.0) / (np.sqrt(dim) * tau)
    np.testing.assert_allclose(rate, A * tau ** (N - 1.0), rtol=1e-12)
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        material.Get_ViscoPlastic_viscosity(0.1, -15.0, dim=1)
