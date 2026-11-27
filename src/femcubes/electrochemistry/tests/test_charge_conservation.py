"""Tests of the charge-conserving metal potential and of corrosion at a metal-electrolyte interface.

An electrolyte strip touches a corroding metal (TwoSquared, ``domain:left``
and its ``interface``); corrosion (Fe -> Fe2+ + 2e-) is supported by oxygen
reduction (O2 + 2H2O + 4e- -> 4OH-). With charge conservation the metal
potential is one global unknown that balances both currents.
"""

import numpy as np
import pytest
from mpi4py import MPI
from scipy.optimize import brentq

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.electrochemistry  # noqa: F401 - ensure model registration
from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver

FARADAY = 96485.3329
TEMPERATURE = 293.15
C_O2, C_OH = 1.0, 0.01   # bulk O2 and OH- (pH 11), mol/m^3
REACTIONS = {
    "Corrosion": {"k": 0.1 / FARADAY, "k_rev": 0.1 / FARADAY, "alpha": 0.5, "E_eq": -0.4},
    "Oxygen": {"k": 1e-4 / FARADAY, "k_rev": 1e-6 / FARADAY, "alpha": 0.5, "E_eq": 0.4},
}


def _build_params(applied_current=0.0, dt=1.0e-3, end_time=1.0e-3):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        MeshType: str = "TwoSquared"
        dim: int = 2
        Quadrature_Degree: int = 3

        Species: dict = {
            "Na+": {"Diffusivity": 1.3e-9, "Charge": 1.0, "C0": "Neutrality"},
            "Cl-": {"Diffusivity": 2.0e-9, "Charge": -1.0, "C0": 600.0},
            "H+": {"Diffusivity": 9.3e-9, "Charge": 1.0, "C0": 1.0e-8 / C_OH},
            "OH-": {"Diffusivity": 5.3e-9, "Charge": -1.0, "C0": C_OH},
            "Fe2+": {"Diffusivity": 1.4e-9, "Charge": 2.0, "C0": 0.0},
            "FeOH+": {"Diffusivity": 1.0e-9, "Charge": 1.0, "C0": 0.0},
            "O2": {"Diffusivity": 1.0e-9, "Charge": 0.0, "C0": C_O2},
        }
        C_ref: float = 1.0
        Volume_Reactions: dict = {
            "Water": {"Type": "Equilibrium", "K_eq": 1.0e-8, "k_dummy": 1.0e6,
                      "species_in": ["H+", "OH-"], "stoich_in": [1.0, 1.0], "species_out": [], "stoich_out": []},
            "Iron": {"k_fwd": 0.1, "k_rev": 1.0e-3,
                     "species_in": ["Fe2+"], "stoich_in": [1.0], "species_out": ["FeOH+", "H+"], "stoich_out": [1.0, 1.0]},
        }
        Materials: dict = {}

        Solution_Steps = {"C": 0, "E": 0}
        Field_Orders = {"C": 1, "E": 1}
        ModelNames: list = ["Electrolyte", "Bulk", "Interface"]
        Models: dict = {
            "Electrolyte": {"Type": "ElectroChemistry:Electrolyte", "Activity_Model": "ideal", "domain": "domain:left"},
            "Bulk": {"Type": "ElectroChemistry:OpenBoundary", "boundary": "left", "E": 0.0, "domain": "domain:left"},
            "Interface": {
                "Type": "ElectroChemistry:MetalElectrolyteInterface",
                "domain": "interface",
                "Metal_Potential": "Charge_Conservation",
                "Initial_Metal_Potential": -0.3,
                "Applied_Current": applied_current,
                "Temperature": TEMPERATURE,
                "Reactions": REACTIONS,
            },
        }

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 50
        line_search: bool = False
        rel_tol: float = 1e-10
        abs_tol: float = 1e-14
        start_time: float = 0.0
        TimeStepControl = {"Type": "None"}
        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0
        output_dir: str = None
        Plot_Fields: list = []
        Plot_Measures: list = []

    params = Params()
    params.L, params.Ly, params.dx, params.dx_interface = 1.0e-3, 2.0e-4, 1.0e-4, 5.0e-5
    params.dt, params.end_time = dt, end_time
    return params


def _run(params):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()
    return mesh, physics


def _mixed_potential(applied_current_density=0.0):
    """Metal potential balancing corrosion and oxygen reduction at the bulk state (phi = 0)."""
    f = FARADAY / (8.3145 * TEMPERATURE)
    c, o = REACTIONS["Corrosion"], REACTIONS["Oxygen"]

    def net_cathodic_current(E_m):
        nu_c = -c["k_rev"] * np.exp((1.0 - c["alpha"]) * f * (E_m - c["E_eq"]))
        nu_o = (o["k"] * C_O2 * np.exp(-o["alpha"] * f * (E_m - o["E_eq"]))
                - o["k_rev"] * C_OH * np.exp((1.0 - o["alpha"]) * f * (E_m - o["E_eq"])))
        return FARADAY * (2.0 * nu_c + 4.0 * nu_o) - applied_current_density

    return brentq(net_cathodic_current, -1.0, 0.4, xtol=1e-14)


def test_global_metal_potential_is_one_dof_in_the_electrolyte_step():
    params = _build_params()
    mesh = Mesh(params)
    Physics(mesh, params)

    assert mesh.Is_Global_Field("E_m")
    assert "E_m" in mesh.mixed_spaces[0]["field_names"]
    assert mesh.fields["E_m"].function_space.dofmap.index_map.size_global == 1
    assert mesh.Prepare_Field_For_Output("E_m") is None


def test_first_assembly_preallocates_the_couplings_of_the_metal_potential():
    # E_m couples to every interface node through Add_KF; inserting these entries without preallocation
    # reallocates the whole matrix many times (quadratic in the number of nodes)
    params = _build_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    K, _F = physics.Get_Kmat_Fvec(0)
    charge_dof = mesh.Get_Global_Field_Dof("E_m", 0)
    start, end = K.getOwnershipRange()

    assert K.getInfo()["mallocs"] == 0
    if start <= charge_dof < end:
        columns, _values = K.getRow(charge_dof)
        assert columns.size > 1              # the charge balance row couples to the interface nodes


def test_electrolyte_reports_the_pH():
    params = _build_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    assert mesh.Get_Field_Support("pH").domain_names == mesh.Get_Field_Support("H+").domain_names
    # Bulk pH: C_H+ = 1e-8 / C_OH- mol/m^3 = 1e-9 mol/L.
    np.testing.assert_allclose(mesh.fields["pH"].x.array, 9.0, rtol=1e-12)

    TimeSolver(params, physics).run()

    # Output is NaN outside the electrolyte (domain:left).
    inside = np.isfinite(mesh.Prepare_Field_For_Output("pH").values)
    pH = mesh.Prepare_Field_For_Output("pH").values[inside]
    C_H = mesh.Prepare_Field_For_Output("H+").values[inside]
    # The field is the interpolant of -log10(C_H+ / 1000).
    np.testing.assert_allclose(pH, 3.0 - np.log10(C_H), atol=1e-3)
    # Oxygen reduction raised the pH at the metal.
    assert mesh.mesh.comm.allreduce(float(pH.max(initial=-np.inf)), op=MPI.MAX) > 9.1


def test_rate_derivatives_include_the_metal_potential():
    params = _build_params()
    params.Models["Interface"]["Reactions"] = {
        **REACTIONS,
        "Volmer_acid": {"k": 1e-6, "k_rev": 1e-10, "alpha": 0.5, "E_eq": 0.0},
        "Heyrovsky_base": {"k": 1e-8, "alpha": 0.3, "E_eq": 0.0},
    }
    params.Models["Interface"]["N_ads"] = 1e-3
    params.Solution_Steps["theta"], params.Field_Orders["theta"] = 0, 1
    model = Physics(Mesh(params), params).models["Interface"]
    rng = np.random.default_rng(3)
    n = 4
    state = {
        "H+": rng.uniform(1e-4, 1e-1, n), "OH-": rng.uniform(1e-4, 1e-1, n), "Fe2+": rng.uniform(0.0, 1.0, n),
        "O2": rng.uniform(0.1, 1.0, n), "E": rng.uniform(-0.05, 0.05, n), "theta_ads": rng.uniform(0.1, 0.9, n),
        "E_m": rng.uniform(-0.5, -0.2, n),
    }
    production, derivatives = model.Production_Rates(state)

    assert set(production) == {"H+", "OH-", "Fe2+", "O2", "theta_ads", "charge"}
    for column, value in state.items():
        step = 1e-4 * max(np.abs(value).max(), 1e-3)
        plus, _ = model.Production_Rates({**state, column: value + step})
        minus, _ = model.Production_Rates({**state, column: value - step})
        for row in production:
            numerical = (plus[row] - minus[row]) / (2.0 * step)
            resolution = 1e-12 * np.abs(production[row]).max() / step
            np.testing.assert_allclose(derivatives[row][column], numerical, rtol=1e-5, atol=resolution,
                                       err_msg=f"d {row} / d {column}")


def test_corrosion_conserves_iron_and_charge():
    params = _build_params()
    model = Physics(Mesh(params), params).models["Interface"]
    state = {"H+": np.array([1e-6]), "OH-": np.array([1e-2]), "Fe2+": np.array([0.5]), "O2": np.array([1.0]),
             "E": np.array([0.0]), "E_m": np.array([-0.35])}

    production, _ = model.Production_Rates(state)

    # Each Fe2+ dissolved releases two electrons, each O2 reduced takes four.
    assert np.isclose(production["charge"], -2.0 * production["Fe2+"] - 4.0 * production["O2"], rtol=1e-12)
    assert np.isclose(production["OH-"], -4.0 * production["O2"], rtol=1e-12)


@pytest.mark.parametrize("applied_current", [0.0, -2.0e-3])
def test_metal_potential_balances_corrosion_and_oxygen_reduction(applied_current):
    params = _build_params(applied_current=applied_current)

    mesh, _physics = _run(params)

    measures = params.Global_Measures
    length = params.Ly
    # The total current of the reactions equals the applied current.
    np.testing.assert_allclose(measures["Interface_Current"], applied_current, atol=1e-9 * abs(measures["Interface_Corrosion_Current"]))
    assert measures["Interface_Corrosion_Current"] > 0.0
    # After a short step the surface concentrations are still at the bulk
    # state, so E_m - phi is the mixed potential (phi: ohmic drop at the surface).
    weights, _dofs, vals = mesh.Get_Interface_Weights_Dofs("interface", ["E"], 0)
    comm = mesh.mesh.comm
    phi_surface = comm.allreduce(np.sum(weights * vals["E"])) / comm.allreduce(weights.sum())
    expected = _mixed_potential(applied_current / length) + phi_surface
    np.testing.assert_allclose(mesh.Get_Global_Value("E_m"), expected, atol=2e-4)
    assert measures["Interface_Metal_Potential"] == mesh.Get_Global_Value("E_m")


def test_fixed_metal_potentials_must_be_numbers():
    params = _build_params()
    params.Models["Interface"]["Metal_Potential"] = "Charge conservation"

    with pytest.raises(ValueError, match="Metal_Potential must be a number"):
        Physics(Mesh(params), params)
