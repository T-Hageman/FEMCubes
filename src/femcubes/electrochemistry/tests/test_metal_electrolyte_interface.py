"""Tests of the metal-electrolyte interface and hydrogen diffusion, on an electrolyte next to a loaded steel plate."""

import copy

import numpy as np
import pytest

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.electrochemistry  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration
from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver


def _coarse_params(include_interface: bool = True, end_time: float = 120.0):
    """PRIVATE: an NaCl electrolyte (domain:left) next to a hydrogen-absorbing, loaded steel plate (domain:right)."""
    class Params(ParamsTemplate):
        Global_Measures: dict = {}

        MeshType: str = "TwoSquared"
        dim: int = 2
        L: float = 1.0e-3
        dx: float = 2.5e-4
        dx_interface: float = 1.0e-4
        Quadrature_Degree: int = 3

        pH = 7.0
        Species: dict = {
            "Na+": {"Diffusivity": 1.33e-9, "Charge": 1.0, "C0": "Neutrality"},
            "Cl-": {"Diffusivity": 2.03e-9, "Charge": -1.0, "C0": 600.0},
            "H+": {"Diffusivity": 9.31e-9, "Charge": 1.0, "C0": 10**(3 - pH)},
            "OH-": {"Diffusivity": 5.27e-9, "Charge": -1.0, "C0": 10**(3 + pH - 14)},
        }
        C_ref = 1000.0
        Volume_Reactions: dict = {
            "Water": {"Type": "Equilibrium", "K_eq": 1.0e-14, "k_dummy": 1.0e6, "species_in": ["H+", "OH-"],
                      "stoich_in": [1.0, 1.0], "species_out": [], "stoich_out": []},
        }

        Materials: dict = {}
        Materials["Steel"] = {"Density": 7.85e3, "Youngs": 210.0e9, "Poisson": 0.3, "Inertia": False, "D_L": 1.0e-9,
                              "C_L0": 0.0}
        steel = Materials["Steel"]
        steel["Shear"] = steel["Youngs"] / (2.0 * (1.0 + steel["Poisson"]))
        steel["Lame"] = steel["Shear"] * (2.0 * steel["Poisson"]) / (1.0 - 2.0 * steel["Poisson"])
        gravity: float = 0.0

        Solution_Steps = {"C": 0, "E": 0, "CL": 0, "theta": 0, "u": 1}
        Field_Orders = {"C": 2, "E": 2, "CL": 2, "theta": 2, "u": 2}
        Models: dict = {
            "Electrolyte": {"Type": "ElectroChemistry:Electrolyte", "Activity_Model": "ideal", "domain": "domain:left"},
            "BC_Electrolyte": {"Type": "ElectroChemistry:OpenBoundary", "boundary": "left", "E": 0.0, "domain": "domain:left"},
            "Hydrogen": {"Type": "ElectroChemistry:HydrogenDiffusion", "Material": "Steel", "domain": "domain:right"},
            "Interface": {
                "Type": "ElectroChemistry:MetalElectrolyteInterface",
                "domain": "interface",
                "Metal_Potential": -0.6,
                "N_ads": 1.0e-3,
                "N_L": 1.0e6,
                "Reactions": {
                    "Volmer_acid": {"k": 1.0e-4, "k_rev": 1.0e-10, "alpha": 0.5, "E_eq": 0.0},
                    "Heyrovsky_acid": {"k": 1.0e-10, "alpha": 0.3, "E_eq": 0.0},
                    "Volmer_base": {"k": 1.0e-8, "k_rev": 1.0e-13, "alpha": 0.5, "E_eq": 0.0},
                    "Heyrovsky_base": {"k": 1.0e-10, "alpha": 0.3, "E_eq": 0.0},
                    "Tafel": {"k": 1.0e-6},
                    "Absorption": {"k": 1.0e1, "k_rev": 7.0e5},
                },
            },
            "Solid": {"Type": "Solid:LinearElastic", "Material": "Steel", "domain": "domain:right"},
            "BC_Bottom_x": {"Type": "Boundary:PrescribedValue", "boundary": "bottom:right", "field": "u_x", "value": 0.0,
                            "domain": "domain:right"},
            "BC_Bottom_y": {"Type": "Boundary:PrescribedValue", "boundary": "bottom:right", "field": "u_y", "value": 0.0,
                            "domain": "domain:right"},
            "BC_Top": {"Type": "Boundary:DisplacementControl", "boundary": "top:right", "field": "u_y", "rate": 1.0e-9,
                       "Dummy": 1.0e16, "domain": "domain:right"},
        }
        ModelNames: list = ["Electrolyte", "BC_Electrolyte", "Hydrogen", "Solid", "BC_Bottom_x", "BC_Bottom_y", "BC_Top"]
        if include_interface:
            ModelNames.append("Interface")

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 2
        max_it: int = 30
        line_search: bool = False
        rel_tol: float = 1e-6
        abs_tol: float = 1e-9

        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0
        start_time: float = 0.0
        dt: float = 60.0
        TimeStepControl = {"Type": "None"}

        output_dir: str = None
        write_vtk: bool = False
        write_hdf5: bool = False
        Plot_Fields: list = []
        Plot_Measures: list = []

    params = Params()
    params.end_time = end_time
    return params


def _interface_model(params):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    return mesh, physics, physics.models["Interface"]


@pytest.fixture(scope="module")
def interface_setup():
    return _interface_model(_coarse_params())


def test_interface_coverage_only_exists_on_the_interface(interface_setup):
    mesh, _physics, _model = interface_setup

    support = mesh.Get_Field_Support("theta_ads")
    assert support.is_interface
    assert support.domain_names == ("interface",)
    assert mesh.Get_Field_Support("H+").domain_names == ("domain:left",)
    assert mesh.Get_Field_Support("CL").domain_names == ("domain:right",)
    assert mesh.mixed_spaces[0]["field_names"][-1] == "theta_ads"


def test_production_rate_derivatives_match_finite_differences(interface_setup):
    _mesh, _physics, model = interface_setup
    rng = np.random.default_rng(1)
    n = 5
    state = {
        "H+": rng.uniform(1e-4, 1e-1, n),
        "OH-": rng.uniform(1e-4, 1e-1, n),
        "E": rng.uniform(-0.05, 0.05, n),
        "theta_ads": rng.uniform(0.1, 0.9, n),
        "CL": rng.uniform(0.0, 50.0, n),
    }
    production, derivatives = model.Production_Rates(state)

    # The productions are linear in the concentrations and smooth in the
    # potential and coverage; a moderate step avoids round-off from the large
    # absorption terms (~1e7 mol/(m^2 s)) while keeping central differences exact enough.
    for column, value in state.items():
        step = 1e-3 * max(np.abs(value).max(), 1e-3)
        plus = {**state, column: value + step}
        minus = {**state, column: value - step}
        production_plus, _ = model.Production_Rates(plus)
        production_minus, _ = model.Production_Rates(minus)
        for row in production:
            numerical = (production_plus[row] - production_minus[row]) / (2.0 * step)
            # Derivatives below what a difference quotient can resolve count as zero.
            resolution = 1e-12 * np.abs(production[row]).max() / step
            np.testing.assert_allclose(derivatives[row][column], numerical, rtol=1e-5, atol=resolution,
                                       err_msg=f"d {row} / d {column}")


def test_reactions_conserve_hydrogen(interface_setup):
    # Every adsorbed or absorbed hydrogen comes from H+ or water, so the
    # coverage, lattice and electrolyte productions must balance against
    # the evolved H2 (2 per Tafel, 1 per Heyrovsky step).
    _mesh, _physics, model = interface_setup
    state = {"H+": np.array([1e-2]), "OH-": np.array([1e-4]), "E": np.array([0.0]),
             "theta_ads": np.array([0.4]), "CL": np.array([10.0])}
    original = copy.deepcopy(model.reactions)
    try:
        for name in ("Tafel", "Heyrovsky_acid", "Heyrovsky_base"):
            model.reactions[name]["k"] = 0.0
        production, _ = model.Production_Rates(state)
    finally:
        model.reactions = original

    # Without H2 evolution: hydrogen gained by the coverage and the metal
    # equals H+ consumed plus OH- produced (from water).
    assert np.isclose(production["theta_ads"] + production["CL"], -production["H+"] + production["OH-"], rtol=1e-12)


def test_interface_terms_require_one_monolithic_step():
    params = _coarse_params()
    params.Solution_Steps["CL"] = 1

    with pytest.raises(ValueError, match="solved in one step"):
        Physics(Mesh(params), params)


@pytest.mark.full_model
def test_without_interface_the_metal_stays_hydrogen_free():
    params = _coarse_params(include_interface=False)
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    assert params.Global_Measures["Hydrogen_Hydrogen"] == 0.0
    assert params.Global_Measures["BC_Top_F"] > 0.0


@pytest.mark.full_model
def test_with_interface_the_metal_absorbs_hydrogen():
    params = _coarse_params(include_interface=True)
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    assert params.Global_Measures["Hydrogen_Hydrogen"] > 0.0
    assert 0.0 < params.Global_Measures["Interface_Coverage"] < 1.0
