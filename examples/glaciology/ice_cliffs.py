"""Crevasses at a grounded ice cliff as the water in front of it drains, with a choice of rheology.

A smaller version of the ice-cliff case of Hageman (2026, submitted, Section 4.2): a 300 x 200 m
grounded ice cliff with surface notches every 50 m, loaded by its own weight and by the water
against its face (right edge). During a 100-hour spin-up the water stands at flotation, so the ice
is in the state of a buttressed cliff, with or without creep. From t = 0 the water is lowered at the
rate of the paper (from flotation to zero in one hour), by default down to ``--water-end`` = 80 m,
and crevasses grow from the notches. The same set-up runs with

* ``le_cohesive`` / ``ve_cohesive`` -- the cohesive phase-field model with the opening-only
  strength surface of the paper, without and with Glen creep (``ve_cohesive`` takes several hours),
* ``le_split`` / ``ve_split`` -- standard (AT1) phase-field fracture of an elastic or viscoelastic
  solid, with the stress split as the driving force,
* ``stokes`` / ``glen`` -- a viscous fluid (Newtonian or Glen) with phase-field damage.

The bed (``--base``) is ``free_slip`` (as in the paper), ``frozen`` (no slip), ``floating`` (ocean
pressure on the bed) or ``friction`` (linear basal friction with coefficient ``--friction``).

Run with::

    mpirun -n 8 python examples/glaciology/ice_cliffs.py
    mpirun -n 8 python examples/glaciology/ice_cliffs.py --case ve_cohesive --water-end 0

Options: ``--ell`` (phase-field length scale, m), ``--water-end`` (water level at the end of the
lowering, m), ``--base``, ``--friction`` (s/m), ``--quick``
(short, coarse smoke run), ``--no-plots``, ``--output-dir DIR`` (default
``results/ice_cliffs/<case>`` next to this script).
"""
from pathlib import Path

import numpy as np

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.maths_utils import ft_to_Gc
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint, rank
from femcubes.solids.failure_surface_plot import plot_FailureSurface

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.glaciology  # noqa: F401
import femcubes.solids  # noqa: F401

# Solid model of each case (--case); the bed conditions (--base) and the extra command-line options
SOLID_MODELS = {
    "le_cohesive": "Solid:Cohesive_LinearElastic",
    "ve_cohesive": "Solid:Cohesive_ViscoElastic",
    "le_split": "Solid:SplitLinearElastic",
    "ve_split": "Solid:SplitViscoElastic",
    "stokes": "Solid:Stokes",
    "glen": "Solid:Stokes",
}
CASES = list(SOLID_MODELS)
BASES = ("free_slip", "frozen", "floating", "friction")
OPTIONS = {"ell": 10.0, "water_end": 80.0, "base": "free_slip", "friction": 0.0}

# Geometry: the paper's cliff is 500 m long; 300 m keeps the run short (five notches instead of nine)
CLIFF_LENGTH = 300.0            # (m); the cliff face is the right edge x = CLIFF_LENGTH
CLIFF_HEIGHT = 200.0            # Ice thickness (m); the bed is y = 0
NOTCH_SPACING = 50.0            # (m); the first notch half a spacing from the left edge, none within half a spacing of the face
WATER_DENSITY = 1025.0          # (kg/m^3)
FLOTATION = 910.0 / WATER_DENSITY * CLIFF_HEIGHT  # Water level of a floating cliff, the level before the lowering (m)
LOWERING_RATE = FLOTATION / 3600.0  # Lowering rate of the paper: from flotation to zero in one hour (m/s)


def build_params(case: str = "le_cohesive", ell: float = 10.0, water_end: float = 80.0, base: str = "free_slip",
                 friction: float = 0.0, quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Return the parameters of one rheology; ``quick`` gives a coarse, short run."""
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}.")
    if base not in BASES:
        raise ValueError(f"Unknown base {base!r}; choose from {BASES}.")
    if ell <= 0.0:
        raise ValueError("ell must be positive.")
    if not 0.0 <= water_end < FLOTATION:
        raise ValueError(f"water_end must be at least 0 and below the flotation level {FLOTATION:.1f} m.")
    lowering_time = (FLOTATION - water_end) / LOWERING_RATE  # Time to lower the water to water_end (s)
    cohesive = case in ("le_cohesive", "ve_cohesive")
    fluid = case in ("stokes", "glen")
    dof = "v" if fluid else "u"
    results_dir = output_dir  # the class body below has its own output_dir parameter

    class Params(ParamsTemplate):
        Global_Measures: dict = {}      # Scalar results of the models (crack depth, crack length), filled during the run

        ## Mesh: geometry and element sizes, read by the mesh generator named in MeshType
        MeshType: str = "Rectangle_Notches"
        dx: float = 25.0                # Background element size (m)
        dim: int = 2
        Lx: float = CLIFF_LENGTH
        Ly: float = CLIFF_HEIGHT
        Notch_Positions: tuple = tuple(np.arange(0.5 * NOTCH_SPACING, CLIFF_LENGTH - 0.5 * NOTCH_SPACING,
                                                 NOTCH_SPACING))  # Notch centres along x (m)
        Notch_Width: float = 0.5        # (m)
        Notch_Depth: float = 10.0       # Depth below the surface (m)
        Quadrature_Degree: int = 3      # Quadrature degree of the point-wise fields (stresses, internal variables)

        ## Mesh refinement: refine where the cohesive strain etaNorm (cohesive cases) or the phase field appears
        MeshRefinement = RefinementConfig(
            enabled=True,
            # Mark the cells whose largest indicator value is at least the threshold
            indicator=FieldThresholdIndicatorConfig(field="etaNorm" if cohesive else "phasefield",
                                                    threshold=3e-12 if cohesive else 0.05,
                                                    reduction="cell_max", comparison="greater_equal"),
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

        ## Materials (not every entry is used by every rheology)
        Materials: dict = {}
        Materials["Water"] = {          # Read by glaciology:LoweringOceanPressure
            "Density": WATER_DENSITY,   # (kg/m^3)
            "WaterHeight": FLOTATION,   # Water level before the lowering (m, bed at y = 0)
        }
        Materials["Ice"] = {
            "Density": 910.0,           # (kg/m^3)
            "Youngs": 9.5e9,            # Young's modulus (Pa)
            "Poisson": 0.35,
            "Viscosity": 3e9,           # Dynamic viscosity (Pa.s), Newtonian Stokes only
            "bulk_visc": 0.0,           # Bulk viscosity (Pa.s), Stokes only
            "Visc_Type": "Glen" if case != "le_cohesive" else "None",  # Creep law: "Newtonian", "Glen" or "None"; set per case below for Stokes
            "Glen_n": 3.0,              # Glen's flow law exponent
            "Glen_A": 1e-24,            # Glen's flow law rate factor (Pa^-n s^-1)
            "LE_Damping": 1.0e4,        # Damping coefficient (Pa s/m^2)
            "visc_max": 1.0e18,         # Maximum viscosity (Pa.s), Glen Stokes only
            "visc_min": 1.0e4,          # Minimum viscosity (Pa.s), Glen Stokes only
            "ell": ell,                 # Phase-field length scale (m)
            "f_t": 0.4e6,               # Tensile strength (Pa)
            "f_s": 1.5e6,               # Strength in compression (Pa) of the strength surface, cohesive models only
            "Gc": 10.0,                 # Fracture energy (J/m^2); set from f_t below for the non-cohesive models
            "pf_damping": 0.0,          # Viscous damping of the phase field (J s/m^3)
            "CrackDrivingForce": "NoTension",  # Energy split of the crack driving force (Split* models: also of the stress); not cohesive models
            "PF_formulation": "AT2",    # Phase-field crack density: "AT1", "AT2", or their higher-order "HO-" forms
            "FailureType": "r2_2L_PSD",  # Cohesive strength surface: opening-only (principal directions), cohesive models only
            "Inertia": True,            # Include the inertia term
            "kappa": 1.0e-3,            # Residual stiffness of fully damaged material (-)
            "Linear_Friction_C": friction,  # Linear basal friction coefficient (s/m), base "friction" only
        }
        # Elastic constants derived from Youngs and Poisson, as read by the solid models
        ice = Materials["Ice"]
        ice["Shear"] = ice["Youngs"] / (2.0 * (1.0 + ice["Poisson"]))
        ice["Bulk"] = ice["Youngs"] / (3.0 * (1.0 - 2.0 * ice["Poisson"]))
        ice["Lame"] = ice["Shear"] * (2.0 * ice["Poisson"]) / (1.0 - 2.0 * ice["Poisson"])
        if not cohesive:
            ice["PF_formulation"] = "HO-AT1"
            ice["Gc"] = ft_to_Gc(ice["f_t"], ice["Youngs"], ice["ell"], ice["PF_formulation"])
        if fluid:
            ice.update(Visc_Type="Newtonian" if case == "stokes" else "Glen", LE_Damping=1e3,
                       CrackDrivingForce="Stress-Based")
        gravity: float = -9.81          # Gravitational acceleration along the last axis (m/s^2)

        ## Models: the fields, their staggered solution steps and the models acting on them
        Solution_Steps = {dof: 0, "phasefield": 1}  # Staggered step of each field: displacements (velocities) first
        Field_Orders = {dof: 2, "phasefield": 1 if cohesive else 2}  # Polynomial order of each field
        # Each entry is one model, built from its "Type"; boundary conditions are models as well
        Models: dict = {
            "Solid": {"Type": SOLID_MODELS[case],
                      "Material": "Ice",
                      "DamageModel": "PhaseField"
            },
            # Phase field driven by the solid's PsiField (cohesive) or ElasticEnergy; irreversibility through
            # a history field (cohesive) or an augmented Lagrangian ("AL")
            "pf": {"Type": "Solid:PhaseField",
                   "Material": "Ice",
                   "CDF_field": "PsiField" if cohesive else "ElasticEnergy",
                   "Irreversible_Method": "Hist" if cohesive else "AL"
            },
            "BC_Left": {"Type": "Boundary:PrescribedValue",
                        "boundary": "left",
                        "field": dof + "_x",
                        "value": 0.0
            },
            # Water against the cliff face: at flotation up to t = 0, then lowered to water_end at LOWERING_RATE
            "Ocean": {"Type": "glaciology:LoweringOceanPressure",
                      "boundary": "right",
                      "LoweringTime": lowering_time,
                      "FinalWaterHeight": water_end
            },
            "CD": {"Type": "Solid:CrackDepthMeasure"},  # Writes the global measure Crack_Depth
        }
        if cohesive:
            # Correct Gc for the element size h in the crack: Gc / (1 + 0.33 h / ell)
            Models["pf"]["Gc_Mesh_Correction"] = {"Coefficient": 0.33,
                                                  "Source_Field": "etaNorm",
                                                  "Source_Threshold": 1.0e-8,
                                                  "Search_Radius": 2.0,
                                                  "Reduction": "mean"
            }
        if case == "ve_split":
            Models["Solid"]["CreepCoupling"] = "PartialExplicit"  # or EffectiveStress, DegradedStress, FullDamagedStress
        # Bed conditions selected with --base. The ocean pressure and the friction are degraded with the phase
        # field (DamagedForces) only for the standard phase-field models; the cohesive models need no degradation.
        if base == "floating":
            Models["BC_Bottom"] = {"Type": "glaciology:OceanPressure",
                                   "boundary": "bottom",
                                   "DamagedForces": not cohesive
            }
        elif base == "friction":
            Models["BC_Bottom"] = {"Type": "glaciology:BasalFriction",
                                   "boundary": "bottom",
                                   "DamagedForces": not cohesive,
                                   "FrictionModel": "Linear"
            }
        else:
            Models["BC_Bottom"] = {"Type": "Boundary:PrescribedValue",
                                   "boundary": "bottom",
                                   "field": dof + "_y",
                                   "value": 0.0
            }
            if base == "frozen":
                Models["BC_Bottom2"] = {"Type": "Boundary:PrescribedValue",
                                        "boundary": "bottom",
                                        "field": dof + "_x",
                                        "value": 0.0
                }
        ModelNames = list(Models)       # The models to build, in this order

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        Step_Solvers: dict = {0: "Newton", 1: "Newton"}  # Nonlinear solver per step
        n_passes: int = 10              # Maximum passes over the staggered steps; stops early once all have converged
        max_it: int = 20 if cohesive else 50  # Maximum Newton iterations per step and pass
        line_search: bool = True
        line_search_lims: list = [1.0e-1, 1.0] if cohesive or case == "glen" else [1.0e-2, 1.0]  # Smallest and largest line-search step factor
        rel_tol: float = 1e-5           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1e-1           # ... or below abs_tol (N/m; much lower values are below the round-off of this problem)

        ## Time: 100 h spin-up with hourly steps, then the lowering of the water to water_end
        start_time: float = -100 * 3600.0
        end_time: float = lowering_time
        dt: float = 3600.0
        Newmark_Beta: float = 0.5625    # Newmark scheme with numerical damping
        Newmark_Gamma: float = 1.0
        # "Measure": a step whose Crack_Length_Change (from Solid:PhaseField) exceeds Target_Change (2 ell) is
        # repeated with half the time step (down to dt_min). Otherwise dt grows by 1.2 per step up to dt_list[i],
        # the largest step from time_list[i] on: 3600 s before t = 0, 1 s for the first 10 s, then 30 s
        # (dt_max is used only without time_list). The step that would cross t = 0 is shortened to end there.
        TimeStepControl = {"Type": "Measure",
                           "time_list": [-1e99, 0.0, 10.0],
                           "dt_list": [3600.0, 1.0, 30.0],
                           "dt_min": 1.0e-6,
                           "dt_max": 3600.0,
                           "Measure": "Crack_Length_Change",
                           "Target_Change": 2.0 * ell
        }

        ## Output: field files and live plots
        output_dir: str = None          # Folder of the field files (None: no files written)
        write_vtk: bool = False
        write_hdf5: bool = False
        Plot_Fields: list = ["phasefield", dof + "_x", dof + "_y"] + (["p"] if fluid else [])  # Fields plotted during the run
        Plot_Measures: list = [("t", "Crack_Depth")]  # Global measures plotted as (x, y); "t" is the time
        if not plots:
            Plot_Fields, Plot_Measures = [], []
        record_movie: bool = plots and not quick  # Record the field plots as movies in movie_folder
        movie_fps: int = 10
        movie_folder: str = str(results_dir) if results_dir is not None else None

    params = Params()
    mprint(f"Cohesive length (tension): {params.ice['Youngs'] * params.ice['Gc'] / params.ice['f_t']**2:.2e} m")
    if quick:
        params.start_time, params.end_time = -2 * 3600.0, 1.0e-4
    return params


def run(case: str = "le_cohesive", ell: float = 10.0, water_end: float = 80.0, base: str = "free_slip",
        friction: float = 0.0, quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Run one rheology and save its figures to ``output_dir``."""
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__, case)
    params = build_params(case, ell=ell, water_end=water_end, base=base, friction=friction, quick=quick, plots=plots,
                          output_dir=output_dir)

    if plots and rank == 0 and case in ("le_cohesive", "ve_cohesive"):
        plot_FailureSurface(params.Materials["Ice"], np.linspace(-2.0e7, 1.0e7, 201),
                            save_path=output_dir / "failure_surface.png")

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
