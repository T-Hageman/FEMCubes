"""An electrolyte next to a hydrogen-absorbing, loaded metal.

Two adjacent squares form one mesh (``MeshType = "TwoSquared"``):

* **domain:left** -- an NaCl electrolyte: ionic concentrations and the
  electrolyte potential (Nernst-Planck transport with electroneutrality and
  water auto-ionisation), held at its bulk state on the ``left`` boundary.
* **domain:right** -- a steel plate: linear-elastic deformation (fixed at the
  bottom, displacement-controlled at the top) and diffusion of lattice
  hydrogen.

Each field only has degrees of freedom on the side where its models act.
The electrolyte and hydrogen fields are solved together in one monolithic
step, even though they live on different subdomains; the displacements are
solved in a second step.

By default, both sides are coupled through
the hydrogen evolution reaction at the interface (Volmer, Heyrovsky, Tafel,
and absorption steps, with parameters following Hageman & Martinez-Paneda,
CMAME 2023): the adsorbed hydrogen coverage is an additional field living on
the interface only. With ``--no-interface`` the two sides are
independent.

Run with::

    python examples/electrochemistry/electrolyte_metal.py
    mpirun -n 4 python examples/electrochemistry/electrolyte_metal.py --no-interface

Options: ``--no-interface`` (uncoupled electrolyte and metal), ``--quick``
(short, coarse smoke run), ``--no-plots``, ``--output-dir DIR`` (default
``results/electrolyte_metal`` next to this script).
"""
from pathlib import Path

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.electrochemistry  # noqa: F401
import femcubes.solids  # noqa: F401

# Extra command-line options and their defaults
OPTIONS = {"no_interface": False}


def build_params(include_interface: bool = True, quick: bool = False, plots: bool = True):
    """Return the parameters; ``quick`` gives a coarse, two-minute run."""
    class Params(ParamsTemplate):
        Global_Measures: dict = {}      # Scalar results of the models (force, hydrogen content, coverage), filled during the run

        ## Mesh: two squares sharing an interface, read by the mesh generator named in MeshType
        MeshType: str = "TwoSquared"
        dim: int = 2
        L: float = 1.0e-3               # Side length of each square (m)
        dx: float = 1.0e-4              # Element size away from the interface (m)
        dx_interface: float = 2.0e-5    # Element size at the interface (m)
        Quadrature_Degree: int = 3

        ## Electrolyte (domain:left): the dissolved species, read by the electrolyte, boundary and interface models.
        # Per species: diffusivity (m^2/s), charge number (-) and initial concentration C0 (mol/m^3);
        # "Neutrality" gives the one species whose C0 makes the initial state electroneutral.
        pH = 7.0                        # Bulk pH, sets the initial H+ and OH- concentrations
        Species: dict = {}
        Species["Na+"] = {"Diffusivity": 1.33e-9, "Charge": 1.0, "C0": "Neutrality"}
        Species["Cl-"] = {"Diffusivity": 2.03e-9, "Charge": -1.0, "C0": 600.0}
        Species["H+"] = {"Diffusivity": 9.31e-9, "Charge": 1.0, "C0": 10**(3 - pH)}
        Species["OH-"] = {"Diffusivity": 5.27e-9, "Charge": -1.0, "C0": 10**(3 + pH - 14)}

        # Volume reactions in the electrolyte; their rates use the activities a = C / C_ref (ideal solution)
        C_ref = 1000.0                  # Reference concentration of the activities (mol/m^3), i.e. activities in mol/L
        Volume_Reactions: dict = {}
        Volume_Reactions["Water"] = {   # H2O <-> H+ + OH-, in instant equilibrium
            "Type": "Equilibrium",      # Penalty rate k_dummy * (a_H+ a_OH- - K_eq)
            "K_eq": 1.0e-14,            # Water auto-ionisation constant (-)
            "k_dummy": 1.0e6,           # Penalty rate enforcing the equilibrium (mol/(m^3 s))
            "species_in": ["H+", "OH-"],
            "stoich_in": [1.0, 1.0],
            "species_out": [],
            "stoich_out": [],
        }

        ## Metal (domain:right): material of the solid and hydrogen-diffusion models
        Materials: dict = {}
        Materials["Steel"] = {
            "Density": 7.85e3,          # Density (kg/m^3)
            "Youngs": 210.0e9,          # Young's modulus (Pa)
            "Poisson": 0.3,             # Poisson's ratio
            "Inertia": False,           # Quasi-static loading
            "D_L": 1.0e-9,              # Lattice hydrogen diffusivity (m^2/s)
            "C_L0": 0.0,                # Initial lattice hydrogen concentration (mol/m^3)
        }
        # Elastic constants derived from Youngs and Poisson, as read by the solid model
        Materials["Steel"]["Shear"] = Materials["Steel"]["Youngs"] / (2.0 * (1.0 + Materials["Steel"]["Poisson"]))
        Materials["Steel"]["Lame"] = Materials["Steel"]["Shear"] * (2.0 * Materials["Steel"]["Poisson"]) / (1.0 - 2.0 * Materials["Steel"]["Poisson"])
        gravity: float = 0.0            # Gravitational acceleration along the last axis (m/s^2)

        ## Models: the fields, their staggered solution steps and the models acting on them
        # Step 0: concentrations C, electrolyte potential E, lattice hydrogen CL and coverage theta; step 1: displacements u
        Solution_Steps = {"C": 0, "E": 0, "CL": 0, "theta": 0, "u": 1}
        Field_Orders = {"C": 2, "E": 2, "CL": 2, "theta": 2, "u": 2}  # Polynomial order of each field

        # Each entry is one model, built from its "Type"; "domain" restricts it to a subdomain or facet group.
        # Only the models in ModelNames are built, in this order.
        Models: dict = {}
        ModelNames: list = ["Electrolyte", "BC_Electrolyte", "Hydrogen", "Solid", "BC_Bottom_x", "BC_Bottom_y", "BC_Top"]

        Models["Electrolyte"] = {
            "Type": "ElectroChemistry:Electrolyte",
            "Activity_Model": "ideal",          # dilute electrolyte (gamma = 1)
            "domain": "domain:left",
        }
        # Bulk electrolyte on the far left: every species held at C0 and the potential at E (V)
        Models["BC_Electrolyte"] = {
            "Type": "ElectroChemistry:OpenBoundary",
            "boundary": "left",
            "E": 0.0,
            "domain": "domain:left",
        }
        Models["Hydrogen"] = {
            "Type": "ElectroChemistry:HydrogenDiffusion",
            "Material": "Steel",
            "domain": "domain:right",
        }
        # Hydrogen evolution and absorption at the metal-electrolyte interface
        if include_interface:
            ModelNames.append("Interface")
        Models["Interface"] = {
            "Type": "ElectroChemistry:MetalElectrolyteInterface",
            "domain": "interface",
            "Metal_Potential": -0.6,    # Imposed metal potential E_m (V_SHE)
            "N_ads": 1.0e-3,            # Adsorption site density (mol/m^2)
            "N_L": 1.0e6,               # Interstitial lattice site density (mol/m^3)
            # Per reaction: rate constants k and k_rev (giving rates in mol/(m^2 s)), charge-transfer
            # coefficient alpha (-) and equilibrium potential E_eq (V_SHE)
            "Reactions": {
                "Volmer_acid":    {"k": 1.0e-4,  "k_rev": 1.0e-10, "alpha": 0.5, "E_eq": 0.0},
                "Heyrovsky_acid": {"k": 1.0e-10,                   "alpha": 0.3, "E_eq": 0.0},
                "Volmer_base":    {"k": 1.0e-8,  "k_rev": 1.0e-13, "alpha": 0.5, "E_eq": 0.0},
                "Heyrovsky_base": {"k": 1.0e-10,                   "alpha": 0.3, "E_eq": 0.0},
                "Tafel":          {"k": 1.0e-6},
                "Absorption":     {"k": 1.0e1,   "k_rev": 7.0e5},
            },
        }
        Models["Solid"] = {
            "Type": "Solid:LinearElastic",
            "Material": "Steel",
            "domain": "domain:right",
        }
        Models["BC_Bottom_x"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom:right",
            "field": "u_x",
            "value": 0.0,
            "domain": "domain:right",
        }
        Models["BC_Bottom_y"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom:right",
            "field": "u_y",
            "value": 0.0,
            "domain": "domain:right",
        }
        # Fixed bottom and displacement-controlled top, recording BC_Top_u / BC_Top_F
        Models["BC_Top"] = {
            "Type": "Boundary:DisplacementControl",
            "boundary": "top:right",
            "field": "u_y",
            "rate": 1.0e-9,             # Imposed displacement rate (m/s)
            "Dummy": 1.0e16,            # Penalty stiffness enforcing the displacement (Pa/m)
            "domain": "domain:right",
        }

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        n_passes: int = 2               # Maximum passes over the staggered steps; stops early once all have converged
        max_it: int = 30                # Maximum Newton iterations per step and pass
        line_search: bool = False
        rel_tol: float = 1e-6           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1e-9           # ... or below abs_tol

        ## Time: 10 minutes in constant 60 s steps
        Newmark_Beta: float = 0.5625    # Newmark parameters, read by the solid model also without inertia
        Newmark_Gamma: float = 1.0
        start_time: float = 0.0
        end_time: float = 600.0
        dt: float = 60.0
        TimeStepControl = {"Type": "None"}  # Constant time step (reduced only after a failed step)

        ## Output: field files and live plots
        output_dir: str = None          # Folder of the field files (None: no files written)
        write_vtk: bool = False
        write_hdf5: bool = False

        Plot_Fields: list = ["H+", "OH-", "E", "CL", "u_y"] + (["theta_ads"] if include_interface else [])  # Fields plotted during the run
        Plot_Measures: list = [("BC_Top_u", "BC_Top_F")]  # Global measures plotted as (x, y)
        if not plots:
            Plot_Fields, Plot_Measures = [], []
        record_movie: bool = False

    params = Params()
    if quick:
        params.dx, params.dx_interface = 2.5e-4, 1.0e-4
        params.end_time = 120.0
    return params


def run(include_interface: bool = True, quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Run the electrolyte | metal example and save the figures to ``output_dir``."""
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__)
    params = build_params(include_interface, quick=quick, plots=plots)

    mesh = Mesh(params)
    mesh.print_mesh_groups()
    if plots:
        mesh.plot_mesh()

    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    mprint(f"Top force: {params.Global_Measures['BC_Top_F']:.4e} N/m at u = {params.Global_Measures['BC_Top_u']:.4e} m")
    mprint(f"Lattice hydrogen in the metal: {params.Global_Measures['Hydrogen_Hydrogen']:.4e} mol/m")
    if include_interface:
        mprint(f"Average surface coverage: {params.Global_Measures['Interface_Coverage']:.4e}")
    if plots:
        mesh.SavePlotsToFolder(str(output_dir))
        mprint(f"Figures saved to {output_dir}")
        mesh.Close_Plots()
    return mesh, physics


if __name__ == "__main__":
    args = parse_example_arguments(__file__, __doc__, options=OPTIONS)
    run(include_interface=not args.options["no_interface"], quick=args.quick, plots=args.plots,
        output_dir=args.output_dir)
