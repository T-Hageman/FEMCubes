"""Creep and cohesive fracture of an ice plate with a hole.

The plate-with-hole benchmark of Hageman (2026, submitted to CMAME), Section 3:
a 1 x 1 m ice plate with a central hole of radius 0.1 m, clamped at the bottom
and pulled at the top at a constant nominal strain rate. The ice creeps by
Glen's law and fractures through the strength-based cohesive phase-field model
(``Solid:Cohesive_ViscoElastic``, intrinsic 2-D). At the default (medium) rate
both creep and fracture matter: creep relaxes the stress concentration at the
hole before cracks start from its sides and run to the plate edges. The
default length scale, ell = 0.1 m, is the largest of the paper's length-scale
study, so the run is short. Adaptive mesh refinement follows the cohesive
strain to an element size of ell / 4.

Run with::

    mpirun -n 8 python examples/fracture/viscoelastic_fracture.py
    mpirun -n 8 python examples/fracture/viscoelastic_fracture.py --rate high --gc 10

Options: ``--rate`` (``low``, ``medium`` or ``high``: 1.0, 1.5 or 2.5e-9 1/s),
``--gc`` (fracture energy, J/m^2; the paper uses 1, 2.5, 5 and 10), ``--ell``
(phase-field length scale, m; the paper uses 0.005 to 0.1), ``--quick``
(short, coarse smoke run), ``--no-plots``, ``--output-dir DIR`` (default
``results/viscoelastic_fracture`` next to this script).
"""
from pathlib import Path

import numpy as np

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint, rank
from femcubes.solids.failure_surface_plot import plot_FailureSurface

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.solids  # noqa: F401

# Nominal strain rates (1/s) of the paper, selected with --rate
RATES = {"low": 1.0e-9, "medium": 1.5e-9, "high": 2.5e-9}
# Extra command-line options with their defaults
OPTIONS = {"rate": "medium", "gc": 5.0, "ell": 0.1}


def build_params(rate: str = "medium", gc: float = 5.0, ell: float = 0.1, quick: bool = False, plots: bool = True,
                 output_dir: Path | None = None):
    """Return the parameters of the run; ``quick`` gives a coarse, short run."""
    if rate not in RATES:
        raise ValueError(f"Unknown rate {rate!r}; choose from {list(RATES)}.")
    if gc <= 0.0 or ell <= 0.0:
        raise ValueError("gc and ell must be positive.")
    strain_rate = RATES[rate]

    results_dir = output_dir  # the class body below has its own output_dir parameter

    class Params(ParamsTemplate):
        Global_Measures: dict = {}      # Scalar results of the models (forces, crack length), filled during the run

        ## Mesh: geometry and element sizes, read by the mesh generator named in MeshType
        MeshType: str = "FullPlateWithHole"
        dim: int = 2
        Lx: float = 1.0                 # Plate width (m)
        Ly: float = 1.0                 # Plate height (m)
        R: float = 0.1                  # Hole radius (m)
        dx: float = 0.03                # Background element size (m)
        # Element size within R_Refine of the hole, graded to dx at 2 R_Refine (m). The paper uses 0.002 m;
        # 0.01 m gives the same peak load at a fifth of the cost. Kept at most ell / 4, so that the cracks
        # start in elements that already resolve the phase field.
        dxmin: float = min(0.01, ell / 4)
        R_Refine: float = 0.025         # Width of the pre-refined ring around the hole (m)
        Quadrature_Degree: int = 3      # Quadrature degree of the point-wise fields (stresses, internal variables)

        ## Mesh refinement: refine where the cohesive strain etaNorm (written by the solid model) appears
        MeshRefinement = RefinementConfig(
            enabled=True,
            # Mark the cells whose largest etaNorm value is at least 1e-12
            indicator=FieldThresholdIndicatorConfig(field="etaNorm", threshold=1e-12, reduction="cell_max",
                                                    comparison="greater_equal"),
            target_h=ell / 4,           # Refine the marked cells down to this element size (m)
            buffer_layers=6,            # Also refine within buffer_layers * target_h of the marked cells
            check_interval=1,           # Check for refinement every n-th time step
            max_levels_per_step=1,      # One refinement pass per time step, which then re-solves the step
            max_events=int(1e99),       # Limit on the total number of refinements (none here)
            repartition=True,           # Rebalance the cells over the MPI ranks after a refinement ...
            repartition_imbalance_threshold=1.1,  # ... when max / mean cells per rank exceeds this
            replot_on_refinement=plots,  # Redraw the mesh plot after each refinement
            save_remesh_animations=False,  # No mesh_refinement.mp4
        )

        ## Materials: one entry per material, referred to by name from the models
        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 910.0,           # (kg/m^3)
            "Youngs": 9.5e9,            # Young's modulus (Pa)
            "Poisson": 0.35,
            "LE_Damping": 1.0e9,        # Damping coefficient (Pa s/m^2), 1.1e6 1/s times the density
            "Visc_Type": "Glen",        # Creep law: "Newtonian" (needs "Viscosity"), "Glen" or "None"
            "Glen_n": 3.0,              # Glen's flow law exponent
            "Glen_A": 1e-24,            # Glen's flow law rate factor (Pa^-n s^-1)
            "ell": ell,                 # Phase-field length scale (m), used by Solid:PhaseField
            "f_t": 0.4e6,               # Tensile strength (Pa)
            "f_s": 0.6e6,               # Shear (compressive) strength (Pa)
            "Gc": gc,                   # Fracture energy (J/m^2)
            "PF_formulation": "AT2",    # Phase-field crack density: "AT1", "AT2", or their higher-order "HO-" forms
            "FailureType": "r2_2L",     # Cohesive failure potential (strength surface) of the solid model
            "Inertia": True,            # Include the inertia term
            "kappa": 1.0e-3,            # Residual stiffness of fully damaged material (-)
        }
        ice = Materials["Ice"]
        # Elastic constants of the intrinsic 2-D material (true_2d below), as read by the solid model
        ice["Shear"] = ice["Youngs"] / (2.0 * (1.0 + ice["Poisson"]))
        ice["Lame"] = ice["Youngs"] * ice["Poisson"] / (1.0 - ice["Poisson"]**2)
        ice["Bulk"] = ice["Lame"] + ice["Shear"]
        gravity: float = 0.0            # Gravitational acceleration along the last axis (m/s^2)

        ## Models: the fields, their staggered solution steps and the models acting on them
        Solution_Steps = {"u": 0, "phasefield": 1}  # Staggered step of each field: displacements first, then the phase field
        Step_Solvers = {0: "Newton", 1: "Newton"}  # Nonlinear solver per step
        Field_Orders = {"u": 2, "phasefield": 1}  # Polynomial order of each field
        Models: dict = {
            # true_2d: intrinsic 2-D material (no out-of-plane stress or strain)
            "Solid": {"Type": "Solid:Cohesive_ViscoElastic",
                      "Material": "Ice",
                      "DamageModel": "PhaseField",
                      "plane_stress": False,
                      "true_2d": True,
                      "consistent_stiffness": True
            },
            # Phase field driven by the solid's PsiField, irreversible through a history field. The fracture
            # energy is corrected for the element size, Gc / (1 + 0.33 h / ell), with h the mean size of the
            # cells with cohesive strain within 2 ell
            "pf": {"Type": "Solid:PhaseField",
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
            # Bottom edge clamped
            "BC_Bottom": {"Type": "Boundary:PrescribedValue",
                          "boundary": "bottom",
                          "field": "u_y",
                          "value": 0.0
            },
            "BC_Bottom2": {"Type": "Boundary:PrescribedValue",
                           "boundary": "bottom",
                           "field": "u_x",
                           "value": 0.0
            },
            # Top edge moved up at strain_rate * Ly (m/s), free to move sideways, through a penalty of stiffness
            # "Dummy" (Pa/m); the run ends 50 steps after the reaction force has dropped below F_Cutoff times its peak
            "BC_Top": {"Type": "Boundary:DisplacementControl",
                       "boundary": "top",
                       "field": "u_y",
                       "rate": strain_rate * Ly,
                       "Dummy": 1e12,
                       "F_Cutoff": 0.1
            },
            # No phase field in a strip along the clamped bottom edge
            "NoPF": {"Type": "Solid:No_PhaseField",
                     "x_range": [-0.1, 1.1],
                     "y_range": [-0.1, 0.1]
            },
        }
        ModelNames = list(Models)       # The models to build, in this order

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        n_passes_force: bool = False    # True: always run all n_passes staggered passes
        max_it: int = 20                # Maximum Newton iterations per step and pass
        n_passes: int = 10              # Maximum passes over the staggered steps; stops early once all have converged
        line_search: bool = True
        line_search_lims: list = [1e-1, 1.0]  # Smallest and largest line-search step factor
        rel_tol: float = 1e-6           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1e-3           # ... or below abs_tol

        ## Time: top displacement of 0.1 mm (0.35 mm for Gc >= 10), in steps of at most 3e-7 strain
        start_time: float = 0.0
        end_time: float = (3.5e-4 if gc >= 10.0 else 1.0e-4) / (strain_rate * Ly)
        dt: float = 3e-7 / strain_rate
        Newmark_Beta: float = 0.5625    # Damped Newmark scheme, 0.25 (1 + a)^2 and 0.5 + a with a = 0.5
        Newmark_Gamma: float = 1.0
        # "Measure": a step whose Crack_Length_Change (from Solid:PhaseField) exceeds Target_Change (m) is
        # repeated with half the time step (down to dt_min). Otherwise dt grows by 1.2 per step up to dt_list[i],
        # the largest step from time_list[i] on (dt_max is used only without time_list).
        TimeStepControl = {"Type": "Measure",
                           "time_list": [-1e99, 0],
                           "dt_list": [60, dt],
                           "dt_min": 1.0e-14,
                           "dt_max": dt,
                           "Measure": "Crack_Length_Change",
                           "Target_Change": 0.0025
        }

        ## Output: time history file and live plots
        output_dir: str = str(results_dir / "Data") if results_dir is not None else None  # time_data.hdf5 (global measures)
        write_vtk: bool = False         # No field files
        write_hdf5: bool = False
        Plot_Fields: list = ["phasefield", "u_y", "sigma_yy"] if plots else []  # Fields plotted during the run
        Plot_Measures: list = [("BC_Top_u", "BC_Top_F"), ("t", "Crack_Length")] if plots else []  # Global measures plotted as (x, y); "t" is the time
        record_movie: bool = plots and not quick  # Record the field plots as movies in movie_folder
        movie_fps: int = 10
        movie_folder: str = str(results_dir) if results_dir is not None else None

    params = Params()
    if quick:
        params.dx, params.dxmin = 0.1, 0.02
        params.end_time = params.start_time + 5 * params.dt
    return params


def run(rate: str = "medium", gc: float = 5.0, ell: float = 0.1, quick: bool = False, plots: bool = True,
        output_dir: Path | None = None):
    """Run the plate and save its figures to ``output_dir``."""
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__)
    params = build_params(rate, gc=gc, ell=ell, quick=quick, plots=plots, output_dir=output_dir)
    mprint(f"Strain rate {RATES[rate]:.1e} 1/s, Gc = {gc} J/m^2, ell = {ell} m, end time {params.end_time:.3e} s")

    if plots and rank == 0:
        ice = params.Materials["Ice"]
        plot_FailureSurface(ice, np.linspace(-1.5 * ice["f_s"], 1.5 * ice["f_t"], 200),
                            save_path=output_dir / "failure_surface.png", true_2d=True)

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
    args = parse_example_arguments(__file__, __doc__, options=OPTIONS)
    run(**args.options, quick=args.quick, plots=args.plots, output_dir=args.output_dir)
