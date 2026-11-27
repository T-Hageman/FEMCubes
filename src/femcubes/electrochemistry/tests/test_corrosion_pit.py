"""Short, coarse corrosion-pit runs under charge-conservation conditions."""

import numpy as np
import pytest
from mpi4py import MPI

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.electrochemistry  # noqa: F401 - ensure model registration
from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver

FARADAY = 96485.3329
HYDROGEN_REACTIONS = {
    "Volmer_acid": {"k": 2.0e-6 / FARADAY, "k_rev": 0.0, "alpha": 0.5, "E_eq": 0.0},
    "Tafel": {"k": 1.0e2},
}


def _build_params(width=4.0e-2, end_time=1800.0):
    """PRIVATE: an NaCl electrolyte (pH 12) above an iron surface with a pit, on a coarse mesh (narrow by default)."""
    class Params(ParamsTemplate):
        Global_Measures: dict = {}

        MeshType: str = "CorrosionPit"
        dim: int = 2
        W: float = width
        H: float = 20.0e-3
        R_pit: float = 5.0e-3
        dx: float = 4.0e-3
        dx_metal: float = 1.0e-3
        Quadrature_Degree: int = 3

        pH = 12.0
        Species: dict = {
            "Na+": {"Diffusivity": 1.3e-9, "Charge": 1.0, "C0": "Neutrality"},
            "Cl-": {"Diffusivity": 2.0e-9, "Charge": -1.0, "C0": 600.0},
            "H+": {"Diffusivity": 9.3e-9, "Charge": 1.0, "C0": 10**(3 - pH)},
            "OH-": {"Diffusivity": 5.3e-9, "Charge": -1.0, "C0": 10**(3 + pH - 14)},
            "Fe2+": {"Diffusivity": 1.4e-9, "Charge": 2.0, "C0": 0.0},
            "FeOH+": {"Diffusivity": 1.0e-9, "Charge": 1.0, "C0": 0.0},
            "O2": {"Diffusivity": 1.0e-9, "Charge": 0.0, "C0": 1.0},
        }
        C_ref = 1.0
        Volume_Reactions: dict = {
            "Water": {"Type": "Equilibrium", "K_eq": 1.0e-8, "k_dummy": 1.0e12, "species_in": ["H+", "OH-"],
                      "stoich_in": [1.0, 1.0], "species_out": [], "stoich_out": []},
            "Iron": {"k_fwd": 0.1, "k_rev": 1.0e-3, "species_in": ["Fe2+"], "stoich_in": [1.0],
                     "species_out": ["FeOH+", "H+"], "stoich_out": [1.0, 1.0]},
        }
        Materials: dict = {}

        Solution_Steps = {"C": 0, "E": 0, "theta": 0}
        Field_Orders = {"C": 2, "E": 2, "theta": 2}
        ModelNames: list = ["Electrolyte", "Bulk", "Pit", "Outer"]
        surface = {
            "Type": "ElectroChemistry:MetalElectrolyteInterface",
            "Metal_Potential": "Charge_Conservation",
            "Metal_Potential_Field": "E_m",
            "Initial_Metal_Potential": -0.3,
            "N_ads": 1.0e-3,
        }
        Models: dict = {
            "Electrolyte": {"Type": "ElectroChemistry:Electrolyte", "Activity_Model": "ideal", "Concentration_Floor": 1.0e-30},
            "Bulk": {"Type": "ElectroChemistry:OpenBoundary", "boundary": "top", "E": 0.0},
            "Pit": {**surface, "domain": "metal:pit", "Reactions": {
                **HYDROGEN_REACTIONS,
                "Corrosion": {"k": 0.1 / FARADAY, "k_rev": 0.1 / FARADAY, "alpha": 0.5, "E_eq": -0.4}}},
            "Outer": {**surface, "domain": "metal:outer", "Reactions": {
                **HYDROGEN_REACTIONS,
                "Oxygen": {"k": 1.0e-4 / FARADAY, "k_rev": 1.0e-6 / FARADAY, "alpha": 0.5, "E_eq": 0.4}}},
        }

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 30
        line_search: bool = False
        Positive_Fields: list = ["H+", "OH-"]
        rel_tol: float = 1e-8
        abs_tol: float = 1e-12

        start_time: float = 0.0
        end_time: float = 1800.0
        dt: float = 300.0
        TimeStepControl = {"Type": "StepWise", "time_list": [0.0], "dt_list": [3600.0], "dt_grow_factor": 1.05}
        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0

        output_dir: str = None
        write_vtk: bool = False
        write_hdf5: bool = False
        Plot_Fields: list = []
        Plot_Measures: list = []

    params = Params()
    params.end_time = end_time
    return params


@pytest.mark.full_model
def test_corrosion_is_balanced_by_the_cathodic_reactions():
    params = _build_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    TimeSolver(params, physics).run()

    measures = params.Global_Measures
    corrosion = measures["Pit_Corrosion_Current"]
    cathodic = measures["Pit_Hydrogen_Current"] + measures["Outer_Hydrogen_Current"] + measures["Outer_Oxygen_Current"]
    assert corrosion > 0.0
    # No current source: the electrons released by corrosion are all consumed.
    np.testing.assert_allclose(cathodic, corrosion, rtol=1e-8)
    # Both surfaces see the one metal potential, which left its initial guess.
    assert measures["Pit_Metal_Potential"] == measures["Outer_Metal_Potential"] == mesh.Get_Global_Value("E_m")
    assert -0.6 < mesh.Get_Global_Value("E_m") < -0.3
    # Corrosion releases iron in the pit.
    _weights, _dofs, vals = mesh.Get_Interface_Weights_Dofs("metal:pit", ["Fe2+"], 0)
    assert mesh.mesh.comm.allreduce(float(vals["Fe2+"].max(initial=0.0)), op=MPI.MAX) > 0.0


@pytest.mark.full_model
def test_wide_pit_acidifies_without_rejected_steps():
    # The 300 mm wide electrolyte of the example: its large cathode drives the pit to acidify within seconds,
    # with OH- falling by orders of magnitude at the pit wall. Linear Newton updates overshot OH- below zero,
    # where the floored reaction rates have no derivative, and every step diverged (dt fell below 1e-3 s).
    params = _build_params(width=0.3, end_time=600.0)
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    def reject_step():
        raise AssertionError(f"The time step from t = {params.t:.4e} s was rejected (retried with dt = {params.dt:.4e} s).")

    physics.Reset_Step = reject_step
    TimeSolver(params, physics).run()

    measures = params.Global_Measures
    assert measures["Pit_Corrosion_Current"] > 0.0
    cathodic = measures["Pit_Hydrogen_Current"] + measures["Outer_Hydrogen_Current"] + measures["Outer_Oxygen_Current"]
    np.testing.assert_allclose(cathodic, measures["Pit_Corrosion_Current"], rtol=1e-8)
    # The pit wall has acidified while OH- stayed positive.
    _weights, _dofs, vals = mesh.Get_Interface_Weights_Dofs("metal:pit", ["H+", "OH-"], 0)
    assert mesh.mesh.comm.allreduce(float(vals["H+"].max(initial=0.0)), op=MPI.MAX) > 1.0e-6
    assert mesh.mesh.comm.allreduce(float(vals["OH-"].min(initial=np.inf)), op=MPI.MIN) > 0.0
