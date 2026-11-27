"""Surface crevasses in a floating ice shelf with a temperature profile through the ice.

A floating ice shelf near its front spreads under its own weight. The ice
model ``Glaciology:Cohesive_ViscoElastic`` takes its creep rate (Glen's
flow law) from the local temperature, so with a cold surface and a warm
base the basal ice relaxes its stress faster, and the longitudinal tension
moves into the cold upper ice. Surface crevasses grow from six notches.
Two cases:

* ``profile`` -- temperature linear from -20 C at the surface to -2 C at
  the base.
* ``uniform`` -- the mean of the profile, -11 C, everywhere.

The crevasses are dry: the ocean pressure acts on the front and the base,
not in the cracks.

Run with::

    mpirun -n 8 python examples/glaciology/ice_shelf_front.py --case profile

Options: ``--ell`` (phase-field length scale, m), ``--days`` (simulated
time, days), ``--quick`` (short, coarse smoke run), ``--no-plots``,
``--output-dir DIR`` (default ``results/ice_shelf_front/<case>`` next to
this script).
"""
from pathlib import Path

import numpy as np
import ufl

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.glaciology  # noqa: F401
import femcubes.solids  # noqa: F401

# Cases selected with --case (see the module docstring) and extra options with their defaults
CASES = ["profile", "uniform"]
OPTIONS = {"ell": 10.0, "days": 10.0}

SHELF_LENGTH = 600.0            # Distance from the inland edge (x = 0) to the front (x = SHELF_LENGTH) (m)
THICKNESS = 200.0               # Ice thickness (m); the base is y = 0 before the shelf settles
NOTCH_SPACING = 100.0           # (m); the first notch half a spacing from the inland edge
ICE_DENSITY = 917.0             # Density of IceMaterial at zero porosity (kg/m^3)
WATER_DENSITY = 1025.0          # (kg/m^3)
T_SURFACE = -20.0               # Temperature at the surface, case "profile" (C)
T_BASE = -2.0                   # Temperature at the base, case "profile" (C)


def build_params(case: str = "profile", ell: float = 10.0, days: float = 10.0, quick: bool = False,
                 plots: bool = True, output_dir: Path | None = None):
    """Return the parameters of one case; ``quick`` gives a short run."""
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}.")
    if ell <= 0.0 or days <= 0.0:
        raise ValueError("ell and days must be positive.")
    results_dir = output_dir  # the class body below has its own output_dir parameter

    class Params(ParamsTemplate):
        Global_Measures: dict = {}      # Scalar results of the models (crack depth, crack length), filled during the run

        ## Mesh: geometry and element sizes, read by the mesh generator named in MeshType
        MeshType: str = "Rectangle_Notches"
        dx: float = 25.0                # Background element size (m)
        dim: int = 2
        Lx: float = SHELF_LENGTH
        Ly: float = THICKNESS
        Notch_Positions: tuple = tuple(np.arange(0.5 * NOTCH_SPACING, SHELF_LENGTH, NOTCH_SPACING))  # Notch centres along x (m)
        Notch_Width: float = 0.5        # (m)
        Notch_Depth: float = 10.0       # Depth below the surface (m)
        Quadrature_Degree: int = 3      # Quadrature degree of the point-wise fields (stresses, internal variables)

        ## Mesh refinement: refine where the cohesive strain etaNorm (written by the solid model) appears
        MeshRefinement = RefinementConfig(
            enabled=True,
            # Mark the cells whose largest etaNorm value is at least 3e-12
            indicator=FieldThresholdIndicatorConfig(field="etaNorm", threshold=3e-12, reduction="cell_max",
                                                    comparison="greater_equal"),
            target_h=ell / 4,           # Refine the marked cells down to this element size (m)
            buffer_layers=4,            # Also refine within buffer_layers * target_h of the marked cells
            check_interval=1,           # Check for refinement every n-th time step
            max_levels_per_step=6,      # Refinement passes per time step, each re-solving the same step
            max_events=int(1e99),       # Limit on the total number of refinements (none here)
            repartition=True,           # Rebalance the cells over the MPI ranks after a refinement ...
            repartition_imbalance_threshold=1.1,  # ... when max / mean cells per rank exceeds this
            replot_on_refinement=plots,  # Redraw the mesh plot after each refinement
            save_remesh_animations=plots and not quick,  # Save mesh_refinement.mp4 in movie_folder
        )

        ## Materials: one entry per material, referred to by name from the models. The elastic constants,
        ## density, Glen's flow law and tensile strength of the ice come from IceMaterial, as functions of
        ## the temperature T and porosity poros (zero here).
        Materials: dict = {}
        Materials["Water"] = {          # Read by glaciology:OceanPressure
            "Density": WATER_DENSITY,   # (kg/m^3)
            "WaterHeight": ICE_DENSITY / WATER_DENSITY * THICKNESS,  # Sea level (m): the shelf floats at y >= 0
        }
        Materials["Ice"] = {
            "ell": ell,                 # Phase-field length scale (m)
            "f_s": 1.5e6,               # Strength in compression (Pa) of the strength surface
            "FailureType": "r2_2L_PSD",  # Cohesive strength surface: opening-only (principal directions)
            "PF_formulation": "AT2",    # Phase-field crack density
            "kappa": 1.0e-3,            # Residual stiffness of fully damaged material (-)
            "LE_Damping": 1.0e4,        # Damping coefficient (Pa s/m^2)
            "Inertia": True,            # Include the inertia term
            "pf_damping": 0.0,          # Viscous damping of the phase field (J s/m^3)
        }
        gravity: float = -9.81          # Gravitational acceleration along the last axis (m/s^2)

        ## Models: the fields, their staggered solution steps and the models acting on them
        Solution_Steps = {"u": 0, "phasefield": 1}  # Staggered step of each field: displacements first
        # Polynomial order of each field; the temperature T (CG) and porosity poros (DG) are set, not solved for
        Field_Orders = {"u": 2, "phasefield": 1, "Temperature": 1, "Porosity": 0}
        # Each entry is one model, built from its "Type"; boundary conditions are models as well
        Models: dict = {
            "Solid": {"Type": "Glaciology:Cohesive_ViscoElastic",
                      "Material": "Ice"
            },
            # Phase field driven by the solid's PsiField; irreversibility through a history field. Gc is
            # corrected for the element size h in the crack: Gc / (1 + 0.33 h / ell)
            "pf": {"Type": "Glaciology:PhaseField",
                   "Material": "Ice",
                   "CDF_field": "PsiField",
                   "Irreversible_Method": "Hist",
                   "Gc_Mesh_Correction": {"Coefficient": 0.33,
                                          "Source_Field": "etaNorm",
                                          "Source_Threshold": 1.0e-8,
                                          "Search_Radius": 2.0,
                                          "Reduction": "mean"
                   }
            },
            # Inland edge held horizontally (a symmetry plane of a wider shelf)
            "BC_Left": {"Type": "Boundary:PrescribedValue",
                        "boundary": "left",
                        "field": "u_x",
                        "value": 0.0
            },
            # Hydrostatic water pressure on the front and the base; the base pressure also holds the shelf
            # vertically, as it follows the current height of the base
            "Front": {"Type": "glaciology:OceanPressure",
                      "boundary": "right"
            },
            "Base": {"Type": "glaciology:OceanPressure",
                     "boundary": "bottom"
            },
            "CD": {"Type": "Solid:CrackDepthMeasure"},  # Writes the global measure Crack_Depth
        }
        ModelNames = list(Models)       # The models to build, in this order

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        Step_Solvers: dict = {0: "Newton", 1: "Newton"}  # Nonlinear solver per step
        n_passes: int = 10              # Maximum passes over the staggered steps; stops early once all have converged
        max_it: int = 20                # Maximum Newton iterations per step and pass
        line_search: bool = True
        line_search_lims: list = [1.0e-1, 1.0]  # Smallest and largest line-search step factor
        rel_tol: float = 1e-5           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1e-1           # ... or below abs_tol (N/m)

        ## Time: gravity and the ocean act from t = 0; steps of up to 3 hours
        start_time: float = 0.0
        end_time: float = days * 86400.0
        dt: float = 3 * 3600.0
        Newmark_Beta: float = 0.5625    # Newmark scheme with numerical damping
        Newmark_Gamma: float = 1.0
        # "Measure": a step whose Crack_Length_Change (from the phase-field model) exceeds Target_Change (2 ell)
        # is repeated with dt divided by dt_cut_factor (down to dt_min); otherwise dt grows by 1.2 per step up to
        # dt_list[0] (dt_max is used only without time_list).
        TimeStepControl = {"Type": "Measure",
                           "time_list": [0.0],
                           "dt_list": [3 * 3600.0],
                           "dt_min": 1.0e-3,
                           "dt_max": 3 * 3600.0,
                           "dt_cut_factor": 10.0,
                           "Measure": "Crack_Length_Change",
                           "Target_Change": 2.0 * ell
        }

        ## Output: field files and live plots
        output_dir: str = None          # Folder of the field files (None: no files written)
        write_vtk: bool = False
        write_hdf5: bool = False
        Plot_Fields: list = ["phasefield", "sigma_xx", "u_y", "T"]  # Fields plotted during the run
        Plot_Measures: list = [("t", "Crack_Depth")]  # Global measures plotted as (x, y); "t" is the time
        if not plots:
            Plot_Fields, Plot_Measures = [], []
        record_movie: bool = plots and not quick  # Record the field plots as movies in movie_folder
        movie_fps: int = 10
        movie_folder: str = str(results_dir) if results_dir is not None else None

    params = Params()
    if quick:
        params.end_time = 2 * params.dt
    return params


def set_temperature(mesh, case: str) -> None:
    """Set the temperature field T (C): linear through the thickness, or its mean everywhere."""
    x = ufl.SpatialCoordinate(mesh.mesh)
    if case == "profile":
        temperature = T_BASE + (T_SURFACE - T_BASE) * x[1] / THICKNESS
    else:
        temperature = 0.5 * (T_BASE + T_SURFACE) + 0.0 * x[1]
    mesh.Set_From_Expression("T", temperature)


def run(case: str = "profile", ell: float = 10.0, days: float = 10.0, quick: bool = False, plots: bool = True,
        output_dir: Path | None = None):
    """Run one case and save its figures to ``output_dir``."""
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__, case)
    params = build_params(case, ell=ell, days=days, quick=quick, plots=plots, output_dir=output_dir)

    mesh = Mesh(params)
    if plots:
        mesh.plot_mesh()
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    set_temperature(mesh, case)     # After Initialize_Fields, which sets T to zero
    TimeSolver(params, physics).run()

    if plots:
        mesh.SavePlotsToFolder(str(output_dir))
        mprint(f"Figures saved to {output_dir}")
        mesh.Close_Plots()
    return mesh, physics


if __name__ == "__main__":
    args = parse_example_arguments(__file__, __doc__, cases=CASES, options=OPTIONS)
    run(args.case, **args.options, quick=args.quick, plots=args.plots, output_dir=args.output_dir)
