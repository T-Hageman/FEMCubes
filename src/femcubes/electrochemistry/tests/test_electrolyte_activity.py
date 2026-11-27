"""Activity models, volume-reaction types and immobile solids of the Electrolyte model.

The activity coefficients are checked against the closed-form Davies and SIT
expressions, all rate derivatives against finite differences, and the
assembled tangent of the full (non-ideal) model against finite differences
of its residual.
"""

import math

import numpy as np
import pytest

import femcubes.electrochemistry  # noqa: F401 - ensure model registration
from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics
from femcubes.electrochemistry._activity import ActivityModel

ACTIVITY_PARAMETERS = {
    "A": 0.509, "B_a": 1.5, "b_Davies": 0.3,
    "SIT_epsilon": {"Na+|Cl-": 0.03, "Na+|OH-": 0.04, "H+|Cl-": 0.12, "Fe2+|Cl-": 0.17},
}
CHARGES = {"Na+": 1.0, "Cl-": -1.0, "H+": 1.0, "OH-": -1.0, "Fe2+": 2.0, "FeOH+": 1.0, "O2": 0.0, "H2": 0.0}


def _build_params(activity_model="SIT", reactions=None):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        MeshType: str = "Rectangle"
        dim: int = 2
        Lx: float = 1.0e-3
        Ly: float = 2.0e-4
        dx: float = 1.0e-4
        Quadrature_Degree: int = 3

        Species: dict = {
            "Na+": {"Diffusivity": 1.33e-9, "Charge": 1.0, "C0": "Neutrality"},
            "Cl-": {"Diffusivity": 2.03e-9, "Charge": -1.0, "C0": 500.0},
            "H+": {"Diffusivity": 9.31e-9, "Charge": 1.0, "C0": 1.0e-7},
            "OH-": {"Diffusivity": 5.27e-9, "Charge": -1.0, "C0": 100.0},
            "Fe2+": {"Diffusivity": 7.19e-10, "Charge": 2.0, "C0": 1.0e-3},
            "FeOH+": {"Diffusivity": 7.0e-10, "Charge": 1.0, "C0": 1.0e-3},
            "O2": {"Diffusivity": 2.1e-9, "Charge": 0.0, "C0": 0.25},
            "H2": {"Diffusivity": 5.13e-9, "Charge": 0.0, "C0": 1.0},
            "Fe(OH)2(s)": {"Solid": True, "C0": 0.5},
        }
        C_ref: float = 1000.0
        Activity_Parameters: dict = ACTIVITY_PARAMETERS
        Volume_Reactions: dict = {}
        Materials: dict = {}

        Solution_Steps = {"C": 0, "E": 0}
        Field_Orders = {"C": 1, "E": 1}
        ModelNames: list = ["Electrolyte"]
        Models: dict = {"Electrolyte": {"Type": "ElectroChemistry:Electrolyte", "Activity_Model": activity_model}}

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 20
        line_search: bool = False
        rel_tol: float = 1e-8
        abs_tol: float = 1e-14
        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0
        dt: float = 1.0
        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0
        TimeStepControl = {"Type": "None"}
        Plot_Fields: list = []
        Plot_Measures: list = []

    params = Params()
    params.Volume_Reactions = dict(REACTIONS if reactions is None else reactions)
    return params


REACTIONS = {
    "Water": {"Type": "Equilibrium", "Normalised": True, "K_eq": 1.0e-14, "k_eq": 1.0e3,
              "species_in": ["H+", "OH-"], "stoich_in": [1.0, 1.0], "species_out": [], "stoich_out": []},
    "Hydrolysis": {"Type": "Equilibrium", "Normalised": True, "K_eq": 10**-4.5, "k_eq": 1.0e3,
                   "species_in": ["Fe2+", "OH-"], "stoich_in": [1.0, 1.0], "species_out": ["FeOH+"], "stoich_out": [1.0]},
    "Plain": {"Type": "Equilibrium", "K_eq": 2.0, "k_dummy": 1.0e2,
              "species_in": ["FeOH+"], "stoich_in": [1.0], "species_out": ["Fe2+", "OH-"], "stoich_out": [1.0, 1.0]},
    "Oxidation": {"Type": "Kinetic", "k_fwd": 5.0e5, "k_rev": 0.0,
                  "species_in": ["FeOH+", "O2", "H+"], "stoich_in": [1.0, 0.25, 1.0], "orders_in": [1.0, 1.0, 0.0],
                  "species_out": ["Fe2+"], "stoich_out": [1.0]},
    "Fe(OH)2(s)": {"Type": "Precipitation", "solid": "Fe(OH)2(s)", "K_s": 10**12.95, "k_p": 1.0,
                   "species": ["Fe2+", "H+"], "stoich": [1.0, -2.0]},
    "H2 degassing": {"Type": "Degassing", "species": "H2", "C_sat": 0.78, "k_g": 1.0},
}


def _electrolyte(activity_model="SIT", reactions=None, floor=None):
    params = _build_params(activity_model, reactions)
    if floor is not None:
        params.Models["Electrolyte"]["Concentration_Floor"] = floor
    physics = Physics(Mesh(params), params)
    return params, physics, physics.models["Electrolyte"]


def _random_state(n=5, seed=4):
    rng = np.random.default_rng(seed)
    return {
        "Na+": rng.uniform(100.0, 600.0, n), "Cl-": rng.uniform(100.0, 600.0, n),
        "H+": rng.uniform(1e-8, 1e-4, n), "OH-": rng.uniform(1.0, 100.0, n),
        "Fe2+": rng.uniform(1e-4, 1e-1, n), "FeOH+": rng.uniform(1e-4, 1e-1, n),
        "O2": rng.uniform(0.05, 0.3, n), "H2": rng.uniform(0.5, 1.5, n),
        "Fe(OH)2(s)": rng.uniform(0.0, 1.0, n),
    }


# ----------------------------------------------------------------------
# Activity coefficients
# ----------------------------------------------------------------------

@pytest.mark.parametrize("model", ["Davies", "SIT"])
def test_activity_coefficients_match_the_closed_forms(model):
    C = {"Na+": np.array([400.0]), "Cl-": np.array([300.0]), "H+": np.array([1e-4]),
         "OH-": np.array([100.0]), "Fe2+": np.array([1.0]), "FeOH+": np.array([0.5]),
         "O2": np.array([0.2]), "H2": np.array([0.0])}
    ln_gamma, _ = ActivityModel(model, ACTIVITY_PARAMETERS, CHARGES).Ln_Gamma(C)

    m = {s: C[s][0] / 1000.0 for s in C}
    I = 0.5 * sum(CHARGES[s] ** 2 * m[s] for s in C)
    A = ACTIVITY_PARAMETERS["A"]
    for s in C:
        z2 = CHARGES[s] ** 2
        if model == "Davies":
            expected = -A * z2 * (math.sqrt(I) / (1.0 + math.sqrt(I)) - 0.3 * I)
        else:
            expected = -z2 * A * math.sqrt(I) / (1.0 + 1.5 * math.sqrt(I))
            for pair, eps in ACTIVITY_PARAMETERS["SIT_epsilon"].items():
                cation, anion = pair.split("|")
                if s == cation:
                    expected += eps * m[anion]
                elif s == anion:
                    expected += eps * m[cation]
        assert ln_gamma[s][0] / math.log(10.0) == pytest.approx(expected, rel=1e-12, abs=1e-15), s


@pytest.mark.parametrize("model", ["Davies", "SIT"])
def test_activity_coefficient_derivatives_match_finite_differences(model):
    activity = ActivityModel(model, ACTIVITY_PARAMETERS, CHARGES)
    rng = np.random.default_rng(5)
    C = {s: rng.uniform(1.0, 500.0, 5) for s in CHARGES}
    ln_gamma, d_ln_gamma = activity.Ln_Gamma(C)
    for k in C:
        step = 1e-4
        plus, _ = activity.Ln_Gamma({**C, k: C[k] + step})
        minus, _ = activity.Ln_Gamma({**C, k: C[k] - step})
        for s in C:
            numerical = (plus[s] - minus[s]) / (2.0 * step)
            np.testing.assert_allclose(d_ln_gamma[s].get(k, 0.0), numerical, rtol=1e-6, atol=1e-12, err_msg=f"{s}/{k}")


def test_ideal_activity_coefficients_are_one():
    ln_gamma, d_ln_gamma = ActivityModel("ideal", None, CHARGES).Ln_Gamma({s: np.array([1.0, 2.0]) for s in CHARGES})
    assert all(np.all(v == 0.0) for v in ln_gamma.values())
    assert all(not d for d in d_ln_gamma.values())


def test_invalid_activity_settings_are_rejected():
    with pytest.raises(ValueError, match="Activity_Model must be one of"):
        ActivityModel("Debye", ACTIVITY_PARAMETERS, CHARGES)
    with pytest.raises(ValueError, match="requires params.Activity_Parameters"):
        ActivityModel("SIT", None, CHARGES)
    with pytest.raises(ValueError, match="cation\\|anion"):
        ActivityModel("SIT", {**ACTIVITY_PARAMETERS, "SIT_epsilon": {"Cl-|Na+": 0.03}}, CHARGES)
    params = _build_params()
    del params.Models["Electrolyte"]["Activity_Model"]
    with pytest.raises(ValueError, match="requires the setting 'Activity_Model'"):
        Physics(Mesh(params), params)


# ----------------------------------------------------------------------
# Volume reactions
# ----------------------------------------------------------------------

@pytest.mark.parametrize("model", ["ideal", "SIT"])
def test_reaction_derivatives_match_finite_differences(model):
    _params, _physics, electrolyte = _electrolyte(model)
    state = _random_state()
    production, derivatives = electrolyte.Production_Rates(state)
    for k, value in state.items():
        step = 1e-6 * max(np.abs(value).max(), 1e-6)
        plus, _ = electrolyte.Production_Rates({**state, k: value + step})
        minus, _ = electrolyte.Production_Rates({**state, k: value - step})
        for s in production:
            numerical = (plus[s] - minus[s]) / (2.0 * step)
            resolution = 1e-10 * np.abs(production[s]).max() / step
            np.testing.assert_allclose(derivatives[s].get(k, 0.0), numerical, rtol=1e-5, atol=resolution,
                                       err_msg=f"d production({s}) / d C({k})")


def test_normalised_equilibria_relax_at_k_eq_whatever_their_constant():
    """A perturbed normalised equilibrium decays at about k_eq, for constants spanning 40 decades."""
    for logK in (-30.0, -4.5, 10.0):
        reaction = {"Type": "Equilibrium", "Normalised": True, "K_eq": 10**logK, "k_eq": 1.0e3,
                    "species_in": ["Fe2+", "OH-"], "stoich_in": [1.0, 1.0], "species_out": ["FeOH+"], "stoich_out": [1.0]}
        _params, _physics, electrolyte = _electrolyte("ideal", {"R": reaction}, floor=1.0e-60)
        a_OH = 0.1
        # Equilibrium: a_Fe a_OH = K a_FeOH; total iron 1 mol/m^3.
        free = 1.0 / (1.0 + a_OH / 10**logK)
        bound = 1.0 / (1.0 + 10**logK / a_OH)
        state = {s: np.array([v]) for s, v in {"Na+": 100.0, "Cl-": 0.0, "H+": 1e-7, "OH-": 1000.0 * a_OH,
                                               "Fe2+": free, "FeOH+": bound, "O2": 0.0, "H2": 0.0,
                                               "Fe(OH)2(s)": 0.0}.items()}
        production, derivatives = electrolyte.Production_Rates(state)
        assert abs(production["Fe2+"][0]) < 1e-9 * 1.0e3
        # Relaxation rate of the iron exchange Fe2+ <-> FeOH+ (OH- in large excess).
        relaxation = -(derivatives["Fe2+"]["Fe2+"][0] - derivatives["Fe2+"]["FeOH+"][0])
        assert 0.5e3 <= relaxation <= 1.0e3 + 1e-9, logK


def _precipitation_rate(omega, C_T, C_s, k_p=1.0, epsilon=1.0e-3):
    """Expected precipitation rate: -k_p FB(C_s, -C_T ln Omega), smoothed with delta = epsilon C_T."""
    a, b, delta = C_s, -C_T * np.log(omega), epsilon * C_T
    return -k_p * (a + b - np.sqrt(a**2 + b**2 + 2.0 * delta**2))


def test_precipitation_forms_and_redissolves_the_solid():
    reaction = {"Type": "Precipitation", "solid": "Fe(OH)2(s)", "K_s": 10**12.95, "k_p": 1.0,
                "species": ["Fe2+", "H+"], "stoich": [1.0, -2.0]}
    _params, _physics, electrolyte = _electrolyte("ideal", {"P": reaction})
    base = {s: np.zeros(3) for s in _random_state()}
    a_H = 1e-10
    omega = np.array([4.0, 0.5, 0.5])
    C_Fe = 1000.0 * omega * 10**12.95 * a_H**2
    state = {**base, "H+": np.full(3, 1000.0 * a_H), "Fe2+": C_Fe, "Fe(OH)2(s)": np.array([0.0, 2.0, 0.0])}
    production, _ = electrolyte.Production_Rates(state)
    expected = _precipitation_rate(omega, C_Fe, state["Fe(OH)2(s)"])
    np.testing.assert_allclose(production["Fe(OH)2(s)"], expected, rtol=1e-12)
    # Undersaturated without solid: only the smoothing remainder (C_s ln Omega -> -epsilon^2 C_T).
    assert expected[0] > 0.0 > expected[1] and 0.0 < expected[2] < 1e-5 * C_Fe[2]
    # Supersaturated without solid: 2 k_p C_T ln(Omega); undersaturated with ample solid: -k_p C_T |ln Omega|.
    np.testing.assert_allclose(expected[:2], [2.0 * C_Fe[0] * np.log(4.0), -C_Fe[1] * np.log(2.0)], rtol=1e-5)
    np.testing.assert_allclose(production["Fe2+"], -production["Fe(OH)2(s)"])
    # The 2 H+ released per precipitation are split between H+ and OH- (consumed) by their abundance.
    w = 1.0e-12 / (1.0e-12 + state["H+"])           # OH- at the default concentration floor
    np.testing.assert_allclose(production["H+"], 2.0 * (1.0 - w) * production["Fe(OH)2(s)"], rtol=1e-12)
    np.testing.assert_allclose(production["OH-"], -2.0 * w * production["Fe(OH)2(s)"], rtol=1e-12)


def test_precipitation_is_smooth_across_saturation_with_the_solid_present():
    """With solid present the rate is ~ k_p C_T ln(Omega) on both sides of saturation: no switch at Omega = 1."""
    reaction = {"Type": "Precipitation", "solid": "Fe(OH)2(s)", "K_s": 10**12.95, "k_p": 1.0,
                "species": ["Fe2+", "H+"], "stoich": [1.0, -2.0]}
    _params, _physics, electrolyte = _electrolyte("ideal", {"P": reaction})
    a_H = 1e-10
    omega = np.array([0.99, 0.999, 1.001, 1.01])
    C_Fe = 1000.0 * omega * 10**12.95 * a_H**2
    state = {**{s: np.zeros(4) for s in _random_state()}, "H+": np.full(4, 1000.0 * a_H), "Fe2+": C_Fe,
             "Fe(OH)2(s)": np.full(4, 1.0)}
    production, derivatives = electrolyte.Production_Rates(state)
    np.testing.assert_allclose(production["Fe(OH)2(s)"], C_Fe * np.log(omega), rtol=1e-3)
    slope = derivatives["Fe(OH)2(s)"]["Fe2+"]            # d rate / d C_Fe = ln(Omega) + 1 ~ 1, continuous
    np.testing.assert_allclose(slope, 1.0 + np.log(omega), rtol=1e-3)


def test_precipitation_leaves_a_vanishing_solid_when_undersaturated():
    """The rate vanishes where C_s (-ln Omega) = epsilon^2 C_T: undersaturated, the solid decays to ~0 but stays positive."""
    reaction = {"Type": "Precipitation", "solid": "Fe(OH)2(s)", "K_s": 10**12.95, "k_p": 1.0, "epsilon": 1.0e-2,
                "species": ["Fe2+", "H+"], "stoich": [1.0, -2.0]}
    _params, _physics, electrolyte = _electrolyte("ideal", {"P": reaction})
    a_H, omega = 1e-10, 0.5
    C_Fe = 1000.0 * omega * 10**12.95 * a_H**2
    C_s = 1.0e-4 * C_Fe / -np.log(omega)
    state = {**{s: np.zeros(3) for s in _random_state()}, "H+": np.full(3, 1000.0 * a_H), "Fe2+": np.full(3, C_Fe),
             "Fe(OH)2(s)": np.array([0.5 * C_s, C_s, 2.0 * C_s])}
    production, _ = electrolyte.Production_Rates(state)
    rate = production["Fe(OH)2(s)"]
    assert rate[0] > 0.0 and rate[2] < 0.0
    assert abs(rate[1]) < 1e-12 * C_Fe


@pytest.mark.parametrize("change, message", [({"width": 0.01}, "width"), ({"epsilon": 0.0}, "epsilon")])
def test_precipitation_rejects_invalid_smoothing(change, message):
    reaction = {"Type": "Precipitation", "solid": "Fe(OH)2(s)", "K_s": 1.0, "k_p": 1.0,
                "species": ["Fe2+", "OH-"], "stoich": [1.0, 2.0], **change}
    with pytest.raises(ValueError, match=message):
        _electrolyte("ideal", {"P": reaction})


def test_precipitation_forms_share_the_rate_by_abundance():
    """Equivalent dissolution forms give one saturation ratio; each form takes its complex's share of the rate."""
    log_K_OH = 4.0                       # Fe2+ + OH- = FeOH+ (formation constant of the complex)
    forms = [{"species": ["Fe2+", "OH-"], "stoich": [1.0, 2.0], "K_s": 10**-15.0},
             {"species": ["FeOH+", "OH-"], "stoich": [1.0, 1.0], "K_s": 10**(-15.0 + log_K_OH)}]
    reaction = {"Type": "Precipitation", "solid": "Fe(OH)2(s)", "k_p": 1.0, "forms": forms}
    _params, _physics, electrolyte = _electrolyte("ideal", {"P": reaction})
    a_OH = np.array([1e-1, 1e-6])               # FeOH+ dominates at the first node, Fe2+ at the second
    a_Fe = 3.0 * 10**-15.0 / a_OH**2            # Omega = 3 at both nodes (complex at equilibrium)
    state = {**{s: np.zeros(2) for s in _random_state()}, "OH-": 1000.0 * a_OH, "Fe2+": 1000.0 * a_Fe,
             "FeOH+": 1000.0 * a_Fe * 10**log_K_OH * a_OH}
    production, _ = electrolyte.Production_Rates(state)
    C_T = state["FeOH+"] + state["Fe2+"]
    share_FeOH = state["FeOH+"] / C_T
    rate = _precipitation_rate(3.0, C_T, 0.0)
    np.testing.assert_allclose(production["Fe(OH)2(s)"], rate, rtol=1e-10)
    np.testing.assert_allclose(production["FeOH+"], -rate * share_FeOH, rtol=1e-10)
    np.testing.assert_allclose(production["Fe2+"], -rate * (1.0 - share_FeOH), rtol=1e-10)
    w = state["OH-"] / (state["OH-"] + 1.0e-12)     # H+ at the default concentration floor
    np.testing.assert_allclose(production["OH-"], -w * rate * (1.0 + (1.0 - share_FeOH)), rtol=1e-12)


def test_precipitation_forms_derivatives_match_finite_differences():
    log_K_OH = 4.0
    forms = [{"species": ["Fe2+", "OH-"], "stoich": [1.0, 2.0], "K_s": 10**-15.0},
             {"species": ["FeOH+", "OH-"], "stoich": [1.0, 1.0], "K_s": 10**(-15.0 + log_K_OH)}]
    _params, _physics, electrolyte = _electrolyte("SIT", {"P": {"Type": "Precipitation", "solid": "Fe(OH)2(s)",
                                                                "k_p": 1.0, "forms": forms}})
    state = _random_state()
    state["OH-"] = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    state["Fe2+"] = np.array([1e-3, 1e-6, 1e-8, 1e-11, 1e-13])
    production, derivatives = electrolyte.Production_Rates(state)
    for k, value in state.items():
        step = 1e-6 * np.maximum(np.abs(value), 1e-12)          # per node: the concentrations span 10 decades
        plus, _ = electrolyte.Production_Rates({**state, k: value + step})
        minus, _ = electrolyte.Production_Rates({**state, k: value - step})
        for s in production:
            numerical = (plus[s] - minus[s]) / (2.0 * step)
            resolution = 1e-10 * np.abs(production[s]) / step
            error = np.abs(derivatives[s].get(k, 0.0) - numerical)
            assert np.all(error <= 1e-5 * np.abs(numerical) + resolution), f"d production({s}) / d C({k})"


def test_precipitation_takes_oh_equivalents_from_the_abundant_ion():
    """In acid the OH- of a precipitation comes from H+ release, in base from OH- (equivalent at the water equilibrium)."""
    reaction = {"Type": "Precipitation", "solid": "Fe(OH)2(s)", "K_s": 1.0e-25, "k_p": 1.0,
                "species": ["Fe2+", "OH-"], "stoich": [1.0, 2.0]}
    _params, _physics, electrolyte = _electrolyte("ideal", {"P": reaction})
    state = {**{s: np.zeros(2) for s in _random_state()}, "Fe2+": np.array([1.0, 1.0]),
             "OH-": np.array([1.0e-6, 100.0]), "H+": np.array([1.0e-2, 1.0e-10])}
    production, _ = electrolyte.Production_Rates(state)
    rate = production["Fe(OH)2(s)"]
    assert np.all(rate > 0.0)                                   # supersaturated at both nodes
    w = state["OH-"] / (state["OH-"] + state["H+"])
    np.testing.assert_allclose(production["OH-"], -2.0 * w * rate, rtol=1e-12)
    np.testing.assert_allclose(production["H+"], 2.0 * (1.0 - w) * rate, rtol=1e-12)
    assert abs(production["OH-"][0]) < 1e-3 * abs(production["H+"][0])     # acid: H+ released
    assert abs(production["H+"][1]) <= 1e-9 * abs(production["OH-"][1])    # base: OH- consumed


def test_degassing_removes_only_the_excess_gas():
    reaction = {"Type": "Degassing", "species": "H2", "C_sat": 0.78, "k_g": 2.0}
    _params, _physics, electrolyte = _electrolyte("ideal", {"G": reaction})
    state = {**{s: np.zeros(2) for s in _random_state()}, "H2": np.array([1.0, 0.5])}
    production, _ = electrolyte.Production_Rates(state)
    np.testing.assert_allclose(production["H2"], [-2.0 * 0.22, 0.0])


def test_invalid_reactions_are_rejected():
    with pytest.raises(ValueError, match="unknown Type"):
        _electrolyte("ideal", {"R": {"Type": "Fast", "species_in": ["H+"], "stoich_in": [1.0]}})
    with pytest.raises(ValueError, match="only take part in a Precipitation"):
        _electrolyte("ideal", {"R": {"Type": "Kinetic", "k_fwd": 1.0, "species_in": ["Fe(OH)2(s)"], "stoich_in": [1.0]}})
    with pytest.raises(ValueError, match="must be a species with 'Solid': True"):
        _electrolyte("ideal", {"R": {"Type": "Precipitation", "solid": "Fe2+", "K_s": 1.0, "k_p": 1.0,
                                     "species": ["H+"], "stoich": [1.0]}})
    with pytest.raises(ValueError, match="unknown species"):
        _electrolyte("ideal", {"R": {"Type": "Kinetic", "k_fwd": 1.0, "species_in": ["Fe3+"], "stoich_in": [1.0]}})


# ----------------------------------------------------------------------
# Full model: tangent of the assembled system
# ----------------------------------------------------------------------

@pytest.mark.parametrize("model", ["ideal", "SIT"])
def test_assembled_tangent_matches_finite_differences(model):
    """K v equals the directional derivative of the residual, including the activity term of the flux."""
    params, physics, _electrolyte_model = _electrolyte(model)
    mesh = physics.mesh
    physics.Initialize_Fields()
    # Smooth, non-uniform fields so that every flux term is active.
    x = mesh.Get_Field("Cl-").function_space.tabulate_dof_coordinates()[:, 0] / params.Lx
    profiles = {"Cl-": 500.0 * (1.0 + 0.5 * x), "OH-": 100.0 * (1.0 - 0.4 * x), "Fe2+": 1.0 + x,
                "FeOH+": 0.5 * (1.0 + x * x), "H2": 1.0 + 0.2 * x, "Fe(OH)2(s)": 0.5 + 0.1 * x}
    for name, values in profiles.items():
        mesh.Get_Field(name).x.array[:] = values
    mesh.Get_Field("Na+").x.array[:] = 500.0 * (1.0 + 0.5 * x) + 100.0 * (1.0 - 0.4 * x) - 2.0 * (1.0 + x) - 0.5 * (1.0 + x * x)
    mesh.Get_Field("E").x.array[:] = 1.0e-3 * np.sin(3.0 * x)

    K, F = physics.Get_Kmat_Fvec(0)
    F0 = F.copy()
    direction = F.duplicate()
    rng = np.random.default_rng(7)
    offsets = mesh.get_step_block_offsets(0)
    values = rng.uniform(-1.0, 1.0, direction.getLocalSize())
    # Scale each field block of the direction to the size of that field.
    for name, start, stop in zip(mesh.step_fields[0], offsets[:-1], offsets[1:]):
        scale = 1.0e-3 if name == "E" else 1.0e-2 * max(np.abs(mesh.Get_Field(name).x.array).max(), 1.0e-6)
        values[start:stop] *= scale
    direction.setArray(values)
    Kv = F.duplicate()
    K.mult(direction, Kv)

    eps = 1.0e-6
    step = direction.copy()
    step.scale(eps)
    mesh.update_step_from_vector(step, 0)
    F_plus = physics.Get_Fvec(0).copy()
    step.scale(-2.0)
    mesh.update_step_from_vector(step, 0)
    F_minus = physics.Get_Fvec(0).copy()

    numerical = (F_plus.getArray() - F_minus.getArray()) / (2.0 * eps)
    scale = np.abs(Kv.getArray()).max()
    np.testing.assert_allclose(Kv.getArray(), numerical, rtol=1e-4, atol=1e-6 * scale)
    assert np.abs(F0.getArray()).max() > 0.0


def _graded_state(params, physics):
    """Smooth, non-uniform, electroneutral fields (every flux term active)."""
    mesh = physics.mesh
    physics.Initialize_Fields()
    x = mesh.Get_Field("Cl-").function_space.tabulate_dof_coordinates()[:, 0] / params.Lx
    for name, values in {"Cl-": 500.0 * (1.0 + 0.5 * x), "OH-": 100.0 * (1.0 - 0.4 * x), "Fe2+": 1.0 + x}.items():
        mesh.Get_Field(name).x.array[:] = values
    mesh.Get_Field("Na+").x.array[:] = 500.0 * (1.0 + 0.5 * x) + 100.0 * (1.0 - 0.4 * x) - 2.0 * (1.0 + x) - 1.0e-3
    mesh.Get_Field("E").x.array[:] = 1.0e-3 * np.sin(3.0 * x)


def test_lagged_activity_tangent_keeps_the_residual_and_drops_only_the_activity_coupling():
    """'lagged' changes the tangent only: same residual, and the same tangent with ideal activities."""
    residuals, tangents = {}, {}
    for mode in ("full", "lagged"):
        params = _build_params("SIT")
        params.Models["Electrolyte"]["Activity_Tangent"] = mode
        physics = Physics(Mesh(params), params)
        _graded_state(params, physics)
        K, F = physics.Get_Kmat_Fvec(0)
        residuals[mode], tangents[mode] = F.getArray().copy(), K.norm()
    np.testing.assert_allclose(residuals["lagged"], residuals["full"], rtol=1e-13, atol=1e-20)
    assert tangents["lagged"] != pytest.approx(tangents["full"], rel=1e-12)     # the activity coupling is left out


def test_invalid_activity_tangent_is_rejected():
    params = _build_params("SIT")
    params.Models["Electrolyte"]["Activity_Tangent"] = "approximate"
    with pytest.raises(ValueError, match="Activity_Tangent"):
        Physics(Mesh(params), params)
