"""Tests of the PassivatingSurface model.

Reference values come from the material-point model of the corrosion study
(CorrosionResearch WP2), exported with its parameter set to
``data/passivating_surface_reference.json``: at given coverages, potential and
activities, the net rates of R1-R14 and the currents must agree, and the
coverages of the point-model steady states must be steady here too.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import femcubes.electrochemistry  # noqa: F401 - ensure model registration
from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.electrochemistry.passivating_surface import STATES, STEPS

FARADAY = 96485.33212
REFERENCE = json.loads((Path(__file__).parent / "data" / "passivating_surface_reference.json").read_text())
CHARGES = {"Na+": 1.0, "Cl-": -1.0, "H+": 1.0, "OH-": -1.0, "Fe2+": 2.0, "Fe(OH)4-": -1.0, "O2": 0.0, "H2": 0.0}
ACTIVITY_PARAMETERS = {"A": 0.509, "B_a": 1.5, "b_Davies": 0.3,
                       "SIT_epsilon": {"Na+|Cl-": 0.03, "Na+|OH-": 0.04, "H+|Cl-": 0.12, "Fe2+|Cl-": 0.17}}


def _build_params(surface=None, activity_model="ideal", end_time=1.0, dt=1.0):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        MeshType: str = "Rectangle"
        dim: int = 2
        Lx: float = 4.0e-4
        Ly: float = 1.0e-3
        dx: float = 2.0e-4
        Quadrature_Degree: int = 3

        Species: dict = {
            "Na+": {"Diffusivity": 1.33e-9, "Charge": 1.0, "C0": "Neutrality"},
            "Cl-": {"Diffusivity": 2.03e-9, "Charge": -1.0, "C0": 300.0},
            "H+": {"Diffusivity": 9.31e-9, "Charge": 1.0, "C0": 1.0e-10},
            "OH-": {"Diffusivity": 5.27e-9, "Charge": -1.0, "C0": 100.0},
            "Fe2+": {"Diffusivity": 7.19e-10, "Charge": 2.0, "C0": 1.0e-6},
            "Fe(OH)4-": {"Diffusivity": 8.0e-10, "Charge": -1.0, "C0": 1.0e-3},
            "O2": {"Diffusivity": 2.0e-9, "Charge": 0.0, "C0": 0.26},
            "H2": {"Diffusivity": 5.13e-9, "Charge": 0.0, "C0": 0.0},
        }
        C_ref: float = 1000.0
        Activity_Parameters: dict = ACTIVITY_PARAMETERS
        Volume_Reactions: dict = {
            "Water": {"Type": "Equilibrium", "Normalised": True, "K_eq": 1.0e-14, "k_eq": 1.0e3,
                      "species_in": ["H+", "OH-"], "stoich_in": [1.0, 1.0], "species_out": [], "stoich_out": []},
        }
        Materials: dict = {}

        Solution_Steps = {"C": 0, "E": 0, "theta": 0}
        Field_Orders = {"C": 1, "E": 1, "theta": 1}
        ModelNames: list = ["Electrolyte", "Bulk", "Surface"]
        Models: dict = {
            "Electrolyte": {"Type": "ElectroChemistry:Electrolyte", "Activity_Model": activity_model},
            "Bulk": {"Type": "ElectroChemistry:OpenBoundary", "boundary": "top", "E": 0.0},
        }

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 30
        line_search: bool = False
        rel_tol: float = 1e-8
        abs_tol: float = 1e-14
        output_dir: str = None
        start_time: float = 0.0
        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0
        TimeStepControl = {"Type": "None"}
        Plot_Fields: list = []
        Plot_Measures: list = []

    params = Params()
    params.end_time, params.dt = end_time, dt
    settings = copy.deepcopy(REFERENCE["settings"] if surface is None else surface)
    settings["Activity_Model"] = activity_model
    settings.setdefault("domain", "bottom")
    params.Models["Surface"] = settings
    return params


def _surface(**changes):
    params = _build_params(**changes)
    physics = Physics(Mesh(params), params)
    return params, physics, physics.models["Surface"]


def _nodal_state(state, n=1):
    """Field values of a reference state (ideal activities, phi = -E so that E_m - phi = E with E_m = 0)."""
    vals = {f"theta_{s}": np.full(n, state["coverages"][s]) for s in STATES}
    for s, a in state["activities"].items():
        vals[s] = np.full(n, 1000.0 * a)
    vals["Na+"] = np.full(n, 100.0)
    vals["E"] = np.full(n, -state["E"])
    return vals


# ----------------------------------------------------------------------
# Regression against the material-point model
# ----------------------------------------------------------------------

@pytest.mark.parametrize("state", REFERENCE["states"], ids=[s["label"] for s in REFERENCE["states"]])
def test_rates_match_the_material_point_model(state):
    settings = {**REFERENCE["settings"], "Metal_Potential": 0.0}
    _params, _physics, surface = _surface(surface=settings)
    rates = surface.Reaction_Rates(_nodal_state(state))
    for name in STEPS:
        total = rates[name][0][0] + (rates[f"{name}_water"][0][0] if f"{name}_water" in rates else 0.0)
        # A net rate is a difference of gross rates: compare to the round-off of the gross rate.
        assert total == pytest.approx(state["rates"][name], rel=1e-7, abs=1e-10 * state["gross_rates"][name] + 1e-30), name

    cathodic = -FARADAY * (4.0 * rates["ORR"][0][0] + 2.0 * rates["HER_acid"][0][0] + 2.0 * rates["HER_base"][0][0])
    assert cathodic == pytest.approx(state["i_cathodic"], rel=1e-7)


@pytest.mark.parametrize("state", [s for s in REFERENCE["states"] if s["steady"]], ids=lambda s: s["label"])
def test_point_model_steady_states_are_steady(state):
    settings = {**REFERENCE["settings"], "Metal_Potential": 0.0}
    _params, _physics, surface = _surface(surface=settings)
    vals = _nodal_state(state)
    production, _ = surface.Production_Rates(vals)
    for state_name in STATES:
        # Round-off scale: the gross rates of the steps changing this state.
        scale = sum(g for name, g in state["gross_rates"].items()
                    if state_name in (STEPS[name].reactant, STEPS[name].product))
        assert abs(production[f"theta_{state_name}"][0]) <= 1e-9 * scale + 1e-30, state_name
    scale = max(state["gross_rates"].values())
    assert sum(production[f"theta_{s}"][0] for s in STATES) == pytest.approx(0.0, abs=1e-12 * scale)


# ----------------------------------------------------------------------
# Contract checks: tangent, conservation, input validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("activity_model", ["ideal", "SIT"])
def test_rate_derivatives_match_finite_differences(activity_model):
    settings = {**REFERENCE["settings"], "Metal_Potential": "Charge_Conservation", "Initial_Metal_Potential": -0.3}
    _params, _physics, surface = _surface(surface=settings, activity_model=activity_model)
    rng = np.random.default_rng(2)
    n = 4
    theta = rng.dirichlet(np.ones(len(STATES)), n).T
    vals = {f"theta_{s}": theta[i] for i, s in enumerate(STATES)}
    vals.update({"Na+": rng.uniform(100.0, 500.0, n), "Cl-": rng.uniform(10.0, 500.0, n), "H+": rng.uniform(1e-4, 1.0, n),
                 "OH-": rng.uniform(1e-4, 10.0, n), "Fe2+": rng.uniform(1e-4, 1.0, n), "Fe(OH)4-": rng.uniform(1e-4, 1.0, n),
                 "O2": rng.uniform(0.05, 0.3, n), "H2": rng.uniform(0.01, 0.8, n), "E": rng.uniform(-0.05, 0.05, n),
                 "E_m": rng.uniform(-0.6, -0.2, n)})
    production, derivatives = surface.Production_Rates(vals)
    for column, value in vals.items():
        step = 1e-6 * max(np.abs(value).max(), 1e-3) if column not in ("E", "E_m") else 1e-7
        plus, _ = surface.Production_Rates({**vals, column: value + step})
        minus, _ = surface.Production_Rates({**vals, column: value - step})
        for row in production:
            numerical = (plus[row] - minus[row]) / (2.0 * step)
            resolution = 1e-9 * np.abs(production[row]).max() / step
            np.testing.assert_allclose(derivatives[row][column], numerical, rtol=1e-5, atol=resolution,
                                       err_msg=f"d {row} / d {column}")


def test_charge_and_iron_are_conserved():
    """Charge released into the solution equals the electrons taken by the metal; iron released = dissolution."""
    settings = {**REFERENCE["settings"], "Metal_Potential": 0.0}
    params, _physics, surface = _surface(surface=settings)
    for state in REFERENCE["states"]:
        vals = _nodal_state(state)
        production, _ = surface.Production_Rates(vals)
        rates = surface.Reaction_Rates(vals)
        released_charge = sum(CHARGES[s] * production[s][0] for s in ("OH-", "H+", "Cl-", "Fe2+", "Fe(OH)4-", "O2", "H2"))
        scale = max(abs(production["charge"][0]), max(abs(r[0][0]) for r in rates.values()))
        assert released_charge + production["charge"][0] == pytest.approx(0.0, abs=1e-10 * scale)
        dissolution = sum(rates[name][0][0] for name in ("R3", "R5", "R8", "R14", "R12", "R13"))
        assert production["Fe2+"][0] + production["Fe(OH)4-"][0] == pytest.approx(dissolution, rel=1e-12, abs=1e-14 * scale)


def test_inconsistent_thermodynamics_are_rejected():
    settings = copy.deepcopy(REFERENCE["settings"])
    settings["Reactions"]["R3"]["E0"] += 0.01
    with pytest.raises(ValueError, match="do not close the reaction cycles"):
        _surface(surface=settings)


@pytest.mark.parametrize("change, message", [
    (lambda s: s["Reactions"].pop("R14"), "must list exactly"),
    (lambda s: s["Reactions"]["R8"].pop("logK"), "needs \\['logK', 'k'\\]"),
    (lambda s: s.update(Initial_Coverages={"FeOOH": 0.8, "FeOH2": 0.4}), "sum to at most one"),
    (lambda s: s.update(Initial_Coverages={"Fe": 0.5}), "unknown or bare states"),
    (lambda s: s.update(Slow_Down={"factor": 0.1, "steps": ["R99"]}), "not steps or water forms"),
    (lambda s: s.update(Slow_Down={"factor": 2.0, "steps": []}), "must lie in"),
    (lambda s: s.update(Water_Pathway={"R2": {"i0_w": 1.0}}), "has no water form"),
    (lambda s: s.update(Metal_Potential="floating"), "Metal_Potential must be"),
    (lambda s: s.update(Coverage_Hold=float("nan")), "Coverage_Hold must be a time"),
    (lambda s: s.pop("Activity_Model"), "requires the setting 'Activity_Model'"),
])
def test_invalid_settings_are_rejected(change, message):
    settings = copy.deepcopy(REFERENCE["settings"])
    change(settings)
    params = _build_params(surface=settings)
    if "Activity_Model" not in settings:
        params.Models["Surface"].pop("Activity_Model")
    with pytest.raises(ValueError, match=message):
        Physics(Mesh(params), params)


def test_an_empty_facet_group_is_rejected():
    params = _build_params(surface={**REFERENCE["settings"], "Metal_Potential": 0.0, "domain": "bottom"})
    params.Lx, params.dx = 4.0e-4, 2.0e-4
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.models["Surface"].domain = "no_facets_here"
    with pytest.raises((ValueError, KeyError)):
        physics.Initialize_Fields()


def test_models_sharing_the_coverages_keep_their_own_initial_state():
    """A defect and the surrounding surface (two models, shared fields) start from their own coverages."""
    params = _build_params(surface={**REFERENCE["settings"], "Metal_Potential": 0.0, "domain": "bottom",
                                    "Initial_Coverages": {"FeOOH": 0.9, "FeOH2": 0.1}})
    params.Models["Side"] = {**params.Models["Surface"], "domain": "left", "Initial_Coverages": {"FeOCl": 0.97}}
    params.ModelNames = params.ModelNames + ["Side"]
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    _w, _d, bottom = mesh.Get_Interface_Weights_Dofs("bottom", ["theta_FeOOH", "theta_FeOCl", "theta_Fe"], None)
    _w, _d, side = mesh.Get_Interface_Weights_Dofs("left", ["theta_FeOOH", "theta_FeOCl", "theta_Fe"], None)
    corner = np.isclose(bottom["theta_FeOCl"], 0.97)          # the shared corner node takes the last model's state
    assert corner.sum() <= 2
    np.testing.assert_allclose(bottom["theta_FeOOH"][~corner], 0.9)
    np.testing.assert_allclose(side["theta_FeOCl"], 0.97)
    np.testing.assert_allclose(side["theta_Fe"], 0.03)


def test_net_current_balances_at_free_corrosion():
    """Net_Current is the charge balance of the coupled model: zero at its converged metal potential."""
    passive = next(s for s in REFERENCE["states"] if s["label"].startswith("passive"))
    settings = {**REFERENCE["settings"], "Metal_Potential": "Charge_Conservation", "Initial_Metal_Potential": -0.3,
                "Initial_Coverages": {s: v for s, v in passive["coverages"].items() if s != "Fe"}}
    params = _build_params(surface=settings, end_time=1.0, dt=1.0)
    params.rel_tol, params.abs_tol = 1e-13, 1e-11
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()
    surface = physics.models["Surface"]
    E_m = params.Global_Measures["Surface_Metal_Potential"]
    scale = params.Global_Measures["Surface_ORR_Current"]
    assert abs(surface.Net_Current(E_m)) < 1e-4 * scale
    assert surface.Net_Current(E_m - 0.05) > 0.0 > surface.Net_Current(E_m + 0.05)


def test_potential_hold_keeps_the_initial_metal_potential():
    passive = next(s for s in REFERENCE["states"] if s["label"].startswith("passive"))
    settings = {**REFERENCE["settings"], "Metal_Potential": "Charge_Conservation", "Initial_Metal_Potential": -0.25,
                "Potential_Hold": 2.5, "Initial_Coverages": {s: v for s, v in passive["coverages"].items() if s != "Fe"}}
    params = _build_params(surface=settings, end_time=4.0, dt=1.0)
    params.rel_tol, params.abs_tol = 1e-13, 1e-11
    history = []
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    write_outputs = mesh.Write_Outputs
    mesh.Write_Outputs = lambda step, t: (write_outputs(step, t), history.append((t, params.Global_Measures["Surface_Metal_Potential"])))
    TimeSolver(params, physics).run()
    held = [E for t, E in history if t <= 2.5 + 1e-12]
    assert held and np.allclose(held, -0.25, atol=1e-12)
    assert abs(history[-1][1] - (-0.25)) > 1e-3                          # released: charge conservation
    surface = physics.models["Surface"]
    assert abs(surface.Net_Current(history[-1][1])) < 1e-4 * params.Global_Measures["Surface_ORR_Current"]


def _history(params, physics, measures):
    """PRIVATE: run and record (t, measures...) at every output."""
    history = []
    mesh = physics.mesh
    write_outputs = mesh.Write_Outputs
    mesh.Write_Outputs = lambda step, t: (write_outputs(step, t),
                                          history.append((t, *[params.Global_Measures[m] for m in measures])))
    TimeSolver(params, physics).run()
    return history


def test_coverage_hold_keeps_the_coverages_while_the_surface_reacts():
    """The passive state of pH 13 held 50 mV above its own potential: the coverages keep their values before the
    release at t = 0 while the surface reacts (anodic current), and move towards FeOOH after it."""
    passive = next(s for s in REFERENCE["states"] if s["label"].startswith("passive"))
    coverages = {s: v for s, v in passive["coverages"].items() if s != "Fe"}
    settings = {**REFERENCE["settings"], "Metal_Potential": passive["E"] + 0.05, "Initial_Coverages": coverages,
                "Coverage_Hold": 0.0}
    params = _build_params(surface=settings, end_time=2.0, dt=1.0)
    params.start_time = -2.0
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    history = _history(params, physics, ["Surface_Coverage_FeOOH", "Surface_Anodic_Current"])
    held = [(t, theta, i_a) for t, theta, i_a in history if t <= 1e-12]
    released = [(t, theta) for t, theta, _i_a in history if t > 1e-12]
    assert [round(t) for t, _, _ in held] == [-1, 0]                    # outputs after each step
    assert all(theta == pytest.approx(coverages["FeOOH"], abs=1e-14) for _t, theta, _i_a in held)
    assert all(i_a > 0.0 for _t, _theta, i_a in held)                   # the held surface reacts
    assert released and released[-1][1] > coverages["FeOOH"] + 1e-3


def test_coverage_hold_with_charge_conservation_balances_the_metal_potential():
    """Without a potential hold (the default), the metal potential balances the currents at the held coverages
    during the initialisation period t < 0."""
    passive = next(s for s in REFERENCE["states"] if s["label"].startswith("passive"))
    coverages = {s: v for s, v in passive["coverages"].items() if s != "Fe"}
    settings = {**REFERENCE["settings"], "Metal_Potential": "Charge_Conservation", "Initial_Metal_Potential": -0.25,
                "Initial_Coverages": coverages, "Coverage_Hold": 0.0}
    params = _build_params(surface=settings, end_time=1.0, dt=1.0)
    params.start_time = -2.0
    params.rel_tol, params.abs_tol = 1e-13, 1e-11
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    history = _history(params, physics, ["Surface_Metal_Potential", "Surface_Coverage_FeOOH"])
    E_m, theta = history[0][1], history[0][2]                           # t = -1: held coverages
    assert abs(E_m - (-0.25)) > 1e-3
    assert theta == pytest.approx(coverages["FeOOH"], abs=1e-14)
    assert abs(physics.models["Surface"].Net_Current(history[-1][1])) < 1e-4 * params.Global_Measures["Surface_ORR_Current"]


def test_slow_down_scales_rates_but_not_equilibria():
    state = REFERENCE["states"][0]
    base = {**REFERENCE["settings"], "Metal_Potential": 0.0}
    slowed = {**base, "Slow_Down": {"factor": 1.0e-3, "steps": ["R10", "R1_water"]}}
    rates = _surface(surface=base)[2].Reaction_Rates(_nodal_state(state))
    rates_slow = _surface(surface=slowed)[2].Reaction_Rates(_nodal_state(state))
    for key in rates:
        expected = 1.0e-3 if key in ("R10", "R1_water") else 1.0
        gross = state["gross_rates"].get(key.removesuffix("_water"), 0.0)
        assert rates_slow[key][0][0] == pytest.approx(expected * rates[key][0][0], rel=1e-12, abs=1e-14 * gross + 1e-30), key


# ----------------------------------------------------------------------
# Coupled model
# ----------------------------------------------------------------------

def test_coupled_passive_surface_runs_and_reports():
    """A passive surface under an alkaline chloride electrolyte at free corrosion: converged steps, consistent measures."""
    passive = next(s for s in REFERENCE["states"] if s["label"].startswith("passive"))
    settings = {**REFERENCE["settings"], "Metal_Potential": "Charge_Conservation", "Initial_Metal_Potential": -0.3,
                "Initial_Coverages": {s: v for s, v in passive["coverages"].items() if s != "Fe"}}
    params = _build_params(surface=settings, end_time=10.0, dt=1.0)
    # The charge balance is small next to the coverage residuals: converge tightly (to the round-off floor).
    params.rel_tol, params.abs_tol = 1e-13, 1e-11
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    measures = params.Global_Measures
    length = params.Lx
    # Free corrosion: the anodic current balances the cathodic currents.
    assert measures["Surface_Anodic_Current"] == pytest.approx(
        measures["Surface_ORR_Current"] + measures["Surface_HER_Current"], rel=1e-5)
    assert -0.6 < measures["Surface_Metal_Potential"] < 0.2
    coverage_sum = sum(measures[f"Surface_Coverage_{s}"] for s in STATES)
    assert coverage_sum == pytest.approx(1.0, abs=1e-10)
    assert measures["Surface_Coverage_FeOH2"] + measures["Surface_Coverage_FeOOH"] > 0.5
    depth = mesh.Get_Field("Dissolved_Depth").x.array
    assert np.all(depth >= 0.0) and depth.max() > 0.0
    assert measures["Surface_Dissolution_Current"] / length < 1e-2      # passive: below 1e-2 A/m^2
    # The dissolution routes split the iron release.
    assert measures["Surface_Hydroxide_Route_Rate"] + measures["Surface_Chloride_Route_Rate"] == pytest.approx(
        measures["Surface_Fe2_Rate"] + measures["Surface_Fe3_Rate"], rel=1e-10, abs=1e-30)
    # The current-density output fields integrate to the global measures.
    surface = physics.models["Surface"]
    for field_name, measure in (("Anodic_Current_Density", "Anodic_Current"), ("Dissolution_Current_Density", "Dissolution_Current"),
                                ("ORR_Current_Density", "ORR_Current"), ("HER_Current_Density", "HER_Current")):
        weights, _dofs, vals = mesh.Get_Interface_Weights_Dofs(surface.domain, [field_name], None)
        assert np.sum(weights * vals[field_name]) == pytest.approx(measures[f"Surface_{measure}"], rel=1e-10, abs=1e-20)
