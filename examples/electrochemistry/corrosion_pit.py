"""Corrosion of a pit under charge-conservation conditions.

An NaCl electrolyte (pH 12, with dissolved oxygen) fills a rectangle above a
metal surface with a semicircular pit (``MeshType = "CorrosionPit"``),
following the pencil-electrode case of Hageman, Andrade & Martinez-Paneda,
"Corrosion rates under charge-conservation conditions", Electrochim. Acta
461 (2023) 142624:

* **Electrolyte** -- Na+, Cl-, H+, OH-, Fe2+, FeOH+ and O2 with the
  electrolyte potential (Nernst-Planck transport, electroneutrality, water
  auto-ionisation and Fe2+ hydrolysis), held at its bulk state on ``top``.
* **Pit wall** (``metal:pit``) -- the anodic surface: iron corrodes
  (Fe -> Fe2+ + 2e-); hydrogen evolves (H+ + e- -> H_ads, 2 H_ads -> H2)
  once the pit acidifies.
* **Outer surface** (``metal:outer``) -- the cathodic surface: oxygen
  reduction (O2 + 2H2O + 4e- -> 4OH-) and hydrogen evolution.

The metal is not connected to any current source, so the electrons released
by corrosion must be consumed by the cathodic reactions. The metal potential
``E_m`` is therefore not imposed but solved for, as one global unknown,
from the charge conservation of the whole metal surface; both surface models
share it.

The pH and the concentrations are plotted and written to VTK files,
together with the surface currents and the metal potential over time.

Run with::

    python examples/electrochemistry/corrosion_pit.py
    mpirun -n 4 python examples/electrochemistry/corrosion_pit.py

Options: ``--quick`` (short, coarse smoke run on a narrow electrolyte),
``--no-plots``, ``--output-dir DIR`` (default ``results/corrosion_pit`` next
to this script; VTK files are written to its ``data`` subfolder).
"""
from pathlib import Path

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.electrochemistry  # noqa: F401

FARADAY = 96485.3329  # Faraday constant (C/mol)

# Hydrogen evolution 2H+ + 2e- -> H2 at the rate k_h C_H+ exp(-alpha f eta) of
# the corrosion paper, as a Volmer step (one electron per H) followed by a
# fast Tafel recombination that keeps the coverage negligible.
HYDROGEN_REACTIONS = {
    "Volmer_acid": {"k": 2.0e-6 / FARADAY, "k_rev": 0.0, "alpha": 0.5, "E_eq": 0.0},
    "Tafel": {"k": 1.0e2},
}


def build_params(quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Return the parameters of the pit; ``quick`` gives a narrow, coarse, 30-minute run."""
    results_dir = output_dir  # the class body below has its own output_dir parameter

    class Params(ParamsTemplate):
        Global_Measures: dict = {}      # Scalar results of the models (currents, metal potential), filled during the run

        ## Mesh: electrolyte above a metal surface with a pit, read by the mesh generator named in MeshType
        MeshType: str = "CorrosionPit"
        dim: int = 2
        W: float = 300.0e-3               # Width of the electrolyte (m)
        H: float = 20.0e-3               # Height of the electrolyte above the metal (m)
        R_pit: float = 5.0e-3           # Pit radius (m)
        dx: float = 2.0e-3              # Element size away from the metal (m)
        dx_metal: float = 2.5e-4        # Element size on the metal surface (m)
        Quadrature_Degree: int = 3

        ## Electrolyte: the dissolved species, read by the electrolyte, boundary and surface models.
        # Per species: diffusivity (m^2/s), charge number (-) and initial concentration C0 (mol/m^3);
        # "Neutrality" gives the one species whose C0 makes the initial state electroneutral.
        pH = 12.0                       # Bulk pH, sets the initial H+ and OH- concentrations
        Species: dict = {}
        Species["Na+"] = {"Diffusivity": 1.3e-9, "Charge": 1.0, "C0": "Neutrality"}
        Species["Cl-"] = {"Diffusivity": 2.0e-9, "Charge": -1.0, "C0": 600.0}
        Species["H+"] = {"Diffusivity": 9.3e-9, "Charge": 1.0, "C0": 10**(3 - pH)}
        Species["OH-"] = {"Diffusivity": 5.3e-9, "Charge": -1.0, "C0": 10**(3 + pH - 14)}
        Species["Fe2+"] = {"Diffusivity": 1.4e-9, "Charge": 2.0, "C0": 0.0}
        Species["FeOH+"] = {"Diffusivity": 1.0e-9, "Charge": 1.0, "C0": 0.0}
        Species["O2"] = {"Diffusivity": 1.0e-9, "Charge": 0.0, "C0": 1.0}

        # Volume reactions in the electrolyte; their rates use the activities a = C / C_ref (ideal solution)
        C_ref = 1.0                     # Reference concentration of the activities (mol/m^3)
        Volume_Reactions: dict = {}
        Volume_Reactions["Water"] = {   # H2O <-> H+ + OH-, in instant equilibrium
            "Type": "Equilibrium",      # Penalty rate k_dummy * (a_H+ a_OH- - K_eq)
            "K_eq": 1.0e-8,             # Water auto-ionisation constant (-), Kw in (mol/m^3)^2 for C_ref = 1 mol/m^3
            "k_dummy": 1.0e12,           # Penalty rate enforcing the equilibrium (mol/(m^3 s))
            "species_in": ["H+", "OH-"],
            "stoich_in": [1.0, 1.0],
            "species_out": [],
            "stoich_out": [],
        }
        Volume_Reactions["Iron"] = {    # Fe2+ + H2O <-> FeOH+ + H+, kinetic (the default "Type")
            "k_fwd": 0.1,               # Forward rate constant (mol/(m^3 s)), times a_Fe2+
            "k_rev": 1.0e-3,            # Backward rate constant (mol/(m^3 s)), times a_FeOH+ a_H+
            "species_in": ["Fe2+"],
            "stoich_in": [1.0],
            "species_out": ["FeOH+", "H+"],
            "stoich_out": [1.0, 1.0],
        }
        Materials: dict = {}

        ## Models: the fields, their solution steps and the models acting on them
        # Concentrations C, electrolyte potential E and hydrogen coverage theta, solved together in one step
        Solution_Steps = {"C": 0, "E": 0, "theta": 0}
        Field_Orders = {"C": 2, "E": 2, "theta": 2}  # Polynomial order of each field

        Models: dict = {}
        ModelNames: list = ["Electrolyte", "Bulk", "Pit", "Outer"]  # The models to build, in this order

        # The volume reactions use concentrations floored at Concentration_Floor, which must lie far
        # below the OH- of the acidified pit (about 1e-9 mol/m^3 at pH 2)
        Models["Electrolyte"] = {"Type": "ElectroChemistry:Electrolyte",
                                 "Activity_Model": "ideal",  # Activity coefficients 1
                                 "Concentration_Floor": 1.0e-30  # (mol/m^3)
        }
        # Well-mixed bulk electrolyte on top: every species held at C0 and the potential at E (V)
        Models["Bulk"] = {"Type": "ElectroChemistry:OpenBoundary",
                          "boundary": "top",
                          "E": 0.0
        }
        # Anodic pit wall and cathodic outer surface share the metal potential E_m,
        # which follows from the conservation of electrons in the metal.
        surface = {
            "Type": "ElectroChemistry:MetalElectrolyteInterface",
            "Metal_Potential": "Charge_Conservation",  # Solve for E_m instead of imposing it
            "Metal_Potential_Field": "E_m",  # Name of the shared metal-potential unknown
            "Initial_Metal_Potential": -0.3,  # (V_SHE)
            "N_ads": 1.0e-3,            # Adsorption site density (mol/m^2)
        }
        # Per reaction: rate constants k and k_rev (giving rates in mol/(m^2 s)), charge-transfer
        # coefficient alpha (-) and equilibrium potential E_eq (V_SHE); "domain" names the facet group
        Models["Pit"] = {**surface,
                         "domain": "metal:pit",
                         "Reactions": {
                             **HYDROGEN_REACTIONS,
                             "Corrosion": {"k": 0.1 / FARADAY, "k_rev": 0.1 / FARADAY, "alpha": 0.5, "E_eq": -0.4},
                         }
        }
        Models["Outer"] = {**surface,
                           "domain": "metal:outer",
                           "Reactions": {
                               **HYDROGEN_REACTIONS,
                               "Oxygen": {"k": 1.0e-4 / FARADAY, "k_rev": 1.0e-6 / FARADAY, "alpha": 0.5, "E_eq": 0.4},
                           }
        }

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        n_passes: int = 1               # Only one solution step, so a single pass
        max_it: int = 30                # Maximum Newton iterations per step
        line_search: bool = False
        # H+ and OH- change by orders of magnitude where the pit acidifies: their Newton updates
        # are taken in log space, so they cannot overshoot below zero (or below the floor)
        Positive_Fields: list = ["H+", "OH-"]
        rel_tol: float = 1e-8           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1e-12          # ... or below abs_tol

        ## Time: 300 s steps, growing by 5% per step up to 1 h (the end time is set below the class)
        start_time: float = 0.0
        dt: float = 300.0
        # "StepWise": dt grows by dt_grow_factor per step up to dt_list[i], the step from time_list[i] on
        TimeStepControl = {"Type": "StepWise",
                           "time_list": [0.0],
                           "dt_list": [3600.0],
                           "dt_grow_factor": 1.05
        }
        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0

        ## Output: VTK files of the fields every 10 steps, figures of the fields and surface currents
        output_dir: str = str(results_dir / "data") if results_dir is not None else None  # Folder of the field files
        write_vtk: bool = plots and not quick
        write_hdf5: bool = False
        output_fields: list = ["pH", "H+", "OH-", "Fe2+", "FeOH+", "O2", "E", "theta_ads"]  # Fields written to the files
        output_interval: int = 10       # Write the fields every n-th step
        Plot_Fields: list = ["pH", "H+", "OH-", "Fe2+", "FeOH+", "O2", "E"] if plots else []  # Fields plotted during the run
        # Global measures plotted as (x, y): "t" is the time, the others are "<model name>_<quantity>"
        Plot_Measures: list = [
            ("t", "Pit_Metal_Potential"),
            ("t", "Pit_Corrosion_Current"),
            ("t", "Pit_Hydrogen_Current"),
            ("t", "Outer_Hydrogen_Current"),
            ("t", "Outer_Oxygen_Current"),
        ] if plots else []
        record_movie: bool = False

    params = Params()
    params.end_time = 7 * 24 * 3600.0
    if quick:
        # A narrow electrolyte on a coarse mesh, to keep the smoke run short
        params.W, params.dx, params.dx_metal = 4.0e-2, 4.0e-3, 1.0e-3
        params.end_time = 1800.0
    return params


def run(quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Run the corrosion pit and save the figures to ``output_dir``."""
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__)
    params = build_params(quick=quick, plots=plots, output_dir=output_dir)

    mesh = Mesh(params)
    mesh.print_mesh_groups()
    if plots:
        mesh.plot_mesh()

    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    measures = params.Global_Measures
    mprint(f"Metal potential: {measures['Pit_Metal_Potential']:.4f} V_SHE")
    mprint(f"Corrosion current (pit): {measures['Pit_Corrosion_Current']:.4e} A/m")
    mprint(f"Hydrogen current (pit / outer): {measures['Pit_Hydrogen_Current']:.4e} / {measures['Outer_Hydrogen_Current']:.4e} A/m")
    mprint(f"Oxygen current (outer): {measures['Outer_Oxygen_Current']:.4e} A/m")
    if plots:
        mesh.SavePlotsToFolder(str(output_dir))
        mprint(f"Figures saved to {output_dir}")
        mesh.Close_Plots()
    return mesh, physics


if __name__ == "__main__":
    args = parse_example_arguments(__file__, __doc__)
    run(quick=args.quick, plots=args.plots, output_dir=args.output_dir)
