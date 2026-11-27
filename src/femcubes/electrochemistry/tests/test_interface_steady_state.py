"""Steady-state verification of electrolyte | interface | metal hydrogen transport.

A thin strip (quasi one-dimensional) holds an electrolyte on the left and a
metal on the right. Hydrogen ions diffuse from the bulk electrolyte (left
boundary) to the interface, are adsorbed (Volmer), recombine (Tafel) or are
absorbed into the metal, and leave the metal through its right boundary,
where the lattice concentration is zero.

With a large excess of supporting electrolyte, migration and ohmic losses are
negligible, so at steady state both H+ and the lattice hydrogen have linear
profiles and the interface obeys three algebraic balances:

    D_H (C_H,bulk - C_H,i) / L = nu_V          (H+ supply = Volmer)
    nu_V = 2 nu_T + nu_A                       (surface coverage)
    nu_A = D_L C_L,i / L                       (absorption = metal outflow)

with the rates of :class:`MetalElectrolyteInterface` at the interface
potential difference E_m - 0. These are solved with SciPy and compared to the
finite element solution.
"""

import numpy as np
import pytest
from scipy.optimize import fsolve

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.electrochemistry  # noqa: F401 - ensure model registration
from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver

L = 1.0e-3          # Length of each half of the strip (m)
LY = 1.0e-4         # Strip height (m)
C_H_BULK = 1.0      # Bulk H+ concentration, pH 3 (mol/m^3)
D_H = 9.3e-9        # H+ diffusivity (m^2/s)
D_L = 1.0e-9        # Lattice hydrogen diffusivity (m^2/s)
E_M = -0.2          # Metal potential (V_SHE)
N_L = 1.0e6         # Lattice site density (mol/m^3)
TEMPERATURE = 293.15
REACTIONS = {
    "Volmer_acid": {"k": 1.0e-6, "k_rev": 1.0e-10, "alpha": 0.5, "E_eq": 0.0},
    "Tafel": {"k": 1.0e-5},
    "Absorption": {"k": 1.0e-11, "k_rev": 1.0e-9},
}


def _build_params():
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        MeshType: str = "TwoSquared"
        dim: int = 2
        Quadrature_Degree: int = 3

        Species: dict = {
            "Na+": {"Diffusivity": 1.33e-9, "Charge": 1.0, "C0": "Neutrality"},
            "Cl-": {"Diffusivity": 2.03e-9, "Charge": -1.0, "C0": 600.0},
            "H+": {"Diffusivity": D_H, "Charge": 1.0, "C0": C_H_BULK},
            "OH-": {"Diffusivity": 5.27e-9, "Charge": -1.0, "C0": 1.0e-8},
        }
        C_ref: float = 1000.0
        Volume_Reactions: dict = {}
        Materials: dict = {"Steel": {"D_L": D_L, "C_L0": 0.0}}

        Solution_Steps = {"C": 0, "E": 0, "CL": 0, "theta": 0}
        Field_Orders = {"C": 2, "E": 2, "CL": 2, "theta": 2}
        ModelNames: list = ["Electrolyte", "Bulk", "Hydrogen", "Outflow", "Interface"]
        Models: dict = {
            "Electrolyte": {"Type": "ElectroChemistry:Electrolyte", "Activity_Model": "ideal", "domain": "domain:left"},
            "Bulk": {"Type": "ElectroChemistry:OpenBoundary", "boundary": "left", "E": 0.0, "domain": "domain:left"},
            "Hydrogen": {"Type": "ElectroChemistry:HydrogenDiffusion", "Material": "Steel", "domain": "domain:right"},
            "Outflow": {"Type": "Boundary:PrescribedValue", "boundary": "right", "field": "CL", "value": 0.0, "domain": "domain:right"},
            "Interface": {
                "Type": "ElectroChemistry:MetalElectrolyteInterface",
                "domain": "interface",
                "Metal_Potential": E_M,
                "N_ads": 1.0e-3,
                "N_L": N_L,
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

        # Backward Euler with very large steps converges to the steady state.
        start_time: float = 0.0
        end_time: float = 1.0e8
        dt: float = 1.0e7
        TimeStepControl = {"Type": "None"}
        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0
        output_dir: str = None
        Plot_Fields: list = []
        Plot_Measures: list = []

    params = Params()
    params.L, params.Ly, params.dx, params.dx_interface = L, LY, 1.0e-4, 2.0e-5
    return params


def _reference_interface_state():
    """Solve the steady interface balances for (C_H,i, theta, C_L,i)."""
    f = 96485.3329 / (8.3145 * TEMPERATURE)
    volmer, tafel, absorption = REACTIONS["Volmer_acid"], REACTIONS["Tafel"], REACTIONS["Absorption"]
    eta = E_M - volmer["E_eq"]

    def residual(x):
        C_H, theta, C_L = x
        nu_V = (volmer["k"] * C_H * (1.0 - theta) * np.exp(-volmer["alpha"] * f * eta)
                - volmer["k_rev"] * theta * np.exp((1.0 - volmer["alpha"]) * f * eta))
        nu_T = tafel["k"] * theta**2
        nu_A = absorption["k"] * (N_L - C_L) * theta - absorption["k_rev"] * C_L * (1.0 - theta)
        return [
            D_H * (C_H_BULK - C_H) / L - nu_V,
            nu_V - 2.0 * nu_T - nu_A,
            nu_A - D_L * C_L / L,
        ]

    solution, _info, converged, message = fsolve(residual, [0.5 * C_H_BULK, 0.5, 1.0], full_output=True, xtol=1e-14)
    assert converged == 1, message
    return solution


@pytest.mark.full_model
def test_steady_state_matches_the_interface_balances():
    params = _build_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    C_H_ref, theta_ref, C_L_ref = _reference_interface_state()
    # Sanity of the chosen regime: every process matters.
    assert 0.05 < C_H_ref / C_H_BULK < 0.95 and 0.05 < theta_ref < 0.95 and C_L_ref > 0.0

    _weights, _dofs, vals = mesh.Get_Interface_Weights_Dofs("interface", ["H+", "theta_ads", "CL", "E"], 0)
    comm = mesh.mesh.comm
    count = comm.allreduce(vals["H+"].size)

    def interface_mean(name):
        return comm.allreduce(vals[name].sum()) / count

    np.testing.assert_allclose(interface_mean("H+"), C_H_ref, rtol=2e-3)
    np.testing.assert_allclose(interface_mean("theta_ads"), theta_ref, rtol=2e-3)
    np.testing.assert_allclose(interface_mean("CL"), C_L_ref, rtol=2e-3)
    assert abs(interface_mean("E")) < 1e-3  # negligible ohmic drop

    # Linear lattice profile from C_L,i at the interface to zero at the right.
    np.testing.assert_allclose(params.Global_Measures["Hydrogen_Hydrogen"], 0.5 * C_L_ref * L * LY, rtol=2e-3)
    np.testing.assert_allclose(params.Global_Measures["Interface_Coverage"], theta_ref, rtol=2e-3)
