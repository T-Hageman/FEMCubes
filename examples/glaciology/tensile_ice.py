"""Tensile loading of a notched ice block, with a choice of rheology.

A 200 x 130 m ice block with surface notches is fixed on its left edge and
bed, and pulled at a constant rate on its right edge after a 50-hour spin-up.
The same set-up runs with

* ``stokes`` / ``glen`` -- a viscous fluid (Newtonian or Glen) with
  phase-field damage,
* ``le`` / ``ve`` -- standard (AT2) phase-field fracture, elastic or
  viscoelastic,
* ``le_cohesive`` / ``ve_cohesive`` -- the cohesive phase-field model, with
  and without Glen creep.

Run with::

    python examples/glaciology/tensile_ice.py --case ve_cohesive
    mpirun -n 4 python examples/glaciology/tensile_ice.py --case glen --n-notches 3

Options: ``--ell`` (phase-field length scale, m), ``--n-notches``,
``--quick`` (short smoke run), ``--no-plots``, ``--output-dir DIR`` (default
``results/tensile_ice/<case>`` next to this script).
"""
from pathlib import Path

import numpy as np

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.maths_utils import ft_to_Gc
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.glaciology  # noqa: F401
import femcubes.solids  # noqa: F401

# Solid model of each case (--case), and the extra command-line options with their defaults
SOLID_MODELS = {
    "stokes": "Solid:Stokes",
    "glen": "Solid:Stokes",
    "le": "Solid:LinearElastic",
    "ve": "Solid:ViscoElastic",
    "le_cohesive": "Solid:Cohesive_LinearElastic",
    "ve_cohesive": "Solid:Cohesive_ViscoElastic",
}
CASES = list(SOLID_MODELS)
OPTIONS = {"ell": 10.0, "n_notches": 1}


def build_params(case: str = "stokes", ell: float = 10.0, n_notches: int = 1, quick: bool = False, plots: bool = True,
                 output_dir: Path | None = None):
    """Return the parameters of one rheology; ``quick`` gives a short run."""
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}.")
    if ell <= 0.0 or n_notches < 0:
        raise ValueError("ell must be positive and n_notches non-negative.")
    cohesive = case in ("le_cohesive", "ve_cohesive")
    dof = "v" if case in ("stokes", "glen") else "u"
    results_dir = output_dir  # the class body below has its own output_dir parameter

    class Params(ParamsTemplate):
        Global_Measures: dict = {}      # Scalar results of the models (forces, crack depth), filled during the run

        ## Mesh: geometry and element size, read by the mesh generator named in MeshType
        MeshType: str = "Rectangle_Notches"
        dx: float = ell / 4             # Element size (m)
        dim: int = 2
        Lx: float = 200.0               # Block length (m)
        Ly: float = 130.0               # Ice thickness (m); the bed is y = 0
        nNotches: int = n_notches       # Number of surface notches, evenly spaced along x
        Notch_Width: float = 1.0        # (m)
        Notch_Depth: float = 25.0       # Depth below the surface (m)
        Quadrature_Degree: int = 3      # Quadrature degree of the point-wise fields (stresses, internal variables)

        ## Materials (not every entry is used by every rheology)
        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 910.0,           # (kg/m^3)
            "Youngs": 9.5e9,            # Young's modulus (Pa)
            "Poisson": 0.35,
            "Viscosity": 3e9,           # Dynamic viscosity (Pa.s), Newtonian Stokes only
            "Visc_Type": "Newtonian" if case == "stokes" else "Glen",  # Creep law (none in the elastic models)
            "Glen_n": 3.0,              # Glen's flow law exponent
            "Glen_A": 1e-24,            # Glen's flow law rate factor (Pa^-n s^-1)
            "visc_max": 1.0e20,         # Maximum viscosity (Pa.s), Glen Stokes only
            "visc_min": 1.0e4,          # Minimum viscosity (Pa.s), Glen Stokes only
            "LE_Damping": 1.0e2,        # Damping coefficient (Pa s/m^2); le_cohesive: a rate (1/s), times the density below
            "ell": ell,                 # Phase-field length scale (m)
            "f_t": 0.4e6,               # Tensile strength (Pa)
            "f_s": 1.0e6,               # Shear strength (Pa), cohesive models only
            "Gc": 1e3,                  # Fracture energy (J/m^2); set from f_t below for the non-cohesive models
            "DP_q": 9.0e6,              # Maximum confined shear strength of the hardening law (Pa)
            "pf_damping": 0.0,          # Viscous damping of the phase field (J s/m^3)
            "PF_formulation": "HO-AT2",  # Phase-field crack density: "AT1", "AT2", or their higher-order "HO-" forms
            "FailureType": "r1_DP",     # Cohesive failure potential with Drucker-Prager-like compression hardening
            "Inertia": True,            # Include the inertia term
            "kappa": 1.0e-6,            # Residual stiffness of fully damaged material (-)
        }
        ice = Materials["Ice"]
        ice["DP_phi"] = np.degrees(np.arctan(3.0 / (1.0 + ice["Poisson"])))  # Friction angle (deg)
        # Elastic constants derived from Youngs and Poisson, as read by the solid models
        ice["Shear"] = ice["Youngs"] / (2.0 * (1.0 + ice["Poisson"]))
        ice["Bulk"] = ice["Youngs"] / (3.0 * (1.0 - 2.0 * ice["Poisson"]))
        ice["Lame"] = ice["Shear"] * (2.0 * ice["Poisson"]) / (1.0 - 2.0 * ice["Poisson"])
        if case == "le_cohesive":
            ice["LE_Damping"] *= ice["Density"]
        if not cohesive:
            ice["Gc"] = ft_to_Gc(ice["f_t"], ice["Youngs"], ice["ell"])
        gravity: float = 0.0            # Gravitational acceleration along the last axis (m/s^2)

        ## Models: the fields, their staggered solution steps and the models acting on them
        Solution_Steps = {dof: 0, "phasefield": 1}  # Staggered step of each field: displacements (velocities) first
        Field_Orders = {dof: 2, "phasefield": 2}  # Polynomial order of each field
        # Each entry is one model, built from its "Type"; boundary conditions are models as well
        Models: dict = {
            "Solid": {"Type": SOLID_MODELS[case],
                      "Material": "Ice",
                      "DamageModel": "PhaseField"
            },
            # Phase field driven by the solid's PsiField (cohesive) or ElasticEnergy; "Hist": the driving force
            # is kept at its maximum over time (history field)
            "pf": {"Type": "Solid:PhaseField",
                   "Material": "Ice",
                   "CDF_field": "PsiField" if cohesive else "ElasticEnergy",
                   "Irreversible_Method": "Hist"
            },
            "BC_Bottom": {"Type": "Boundary:PrescribedValue",
                          "boundary": "bottom",
                          "field": dof + "_y",
                          "value": 0.0
            },
            "BC_Left": {"Type": "Boundary:PrescribedValue",
                        "boundary": "left",
                        "field": dof + "_x",
                        "value": 0.0
            },
            # Right edge held during the spin-up (t < 0), then pulled at "rate" through a penalty of stiffness
            # "Dummy" (Pa/m); the run ends 50 steps after the force has dropped below F_Cutoff times its peak
            "BC_Right": {"Type": "Boundary:DisplacementControl",
                         "boundary": "right",
                         "field": dof + "_x",
                         "rate": 1e-4,
                         "Dummy": 1e12,
                         "F_Cutoff": 0.05
            },
            "CD": {"Type": "Solid:CrackDepthMeasure"},  # Writes the global measure Crack_Depth
        }
        ModelNames = list(Models)       # The models to build, in this order

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        n_passes: int = 5               # Maximum passes over the staggered steps; stops early once all have converged
        max_it: int = 30                # Maximum Newton iterations per step and pass
        line_search: bool = True
        line_search_lims: list = [1e-1, 1.0]  # Smallest and largest line-search step factor
        rel_tol: float = 1e-3           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1e-4           # ... or below abs_tol

        ## Time: 50 h spin-up with hourly steps, then one hour of loading
        start_time: float = -50 * 3600.0
        end_time: float = 3600.0
        dt: float = 3600.0
        Newmark_Beta: float = 0.5625    # Newmark scheme with numerical damping
        Newmark_Gamma: float = 1.0
        # "Measure": a step whose Crack_Length_Change (from Solid:PhaseField) exceeds Target_Change (m) is
        # repeated with half the time step (down to dt_min). Otherwise dt grows by 1.2 per step up to dt_list[i],
        # the largest step from time_list[i] on: 3600 s before t = 0, 1 s after (dt_max is used only without
        # time_list). The step that would cross t = 0 is shortened to end there.
        TimeStepControl = {"Type": "Measure",
                           "time_list": [-1e99, 0],
                           "dt_list": [3600, 1.0],
                           "dt_min": 1.0e-8,
                           "dt_max": 3600.0,
                           "Measure": "Crack_Length_Change",
                           "Target_Change": 1.0 * dx
        }

        ## Output: field files and live plots
        output_dir: str = None          # Folder of the field files (None: no files written)
        write_vtk: bool = False
        write_hdf5: bool = False
        Plot_Fields: list = ["phasefield", dof + "_x", dof + "_y"] if plots else []  # Fields plotted during the run
        Plot_Measures: list = [("t", "BC_Right_F")] if plots else []  # Global measures plotted as (x, y); "t" is the time
        record_movie: bool = plots and not quick  # Record the field plots as movies in movie_folder
        movie_fps: int = 10
        movie_folder: str = str(results_dir) if results_dir is not None else None

    params = Params()
    if quick:
        params.start_time, params.end_time = -2 * 3600.0, 0.01
    return params


def run(case: str = "stokes", ell: float = 10.0, n_notches: int = 1, quick: bool = False, plots: bool = True,
        output_dir: Path | None = None):
    """Run one rheology and save its figures to ``output_dir``."""
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__, case)
    params = build_params(case, ell=ell, n_notches=n_notches, quick=quick, plots=plots, output_dir=output_dir)

    mesh = Mesh(params)
    if plots:
        mesh.plot_mesh()
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()

    if plots:
        mesh.SavePlotsToFolder(str(output_dir))
        mprint(f"Figures saved to {output_dir}")
        mesh.Close_Plots()
    return mesh, physics


if __name__ == "__main__":
    args = parse_example_arguments(__file__, __doc__, cases=CASES, options=OPTIONS)
    run(args.case, **args.options, quick=args.quick, plots=args.plots, output_dir=args.output_dir)
