"""Cohesive phase-field fracture of plates with holes and notches.

The strength-based cohesive model (``Solid:Cohesive_LinearElastic``) on three
geometries:

* ``full_plate`` -- square plate with a central elliptical hole under
  displacement-controlled compression.
* ``single_edge_notched`` -- single-edge-notched plate loaded in shear.
* ``dynamic`` -- single-edge-notched plate under a sudden tensile traction,
  giving dynamic crack branching.

Adaptive mesh refinement follows the cohesive strain (``etaNorm``). The
displacements, the phase field, the crack length and the load-displacement
curve are plotted.

Run with::

    python examples/fracture/cohesive_phasefield.py --case full_plate
    mpirun -n 8 python examples/fracture/cohesive_phasefield.py --case dynamic

Options: ``--quick`` (short, coarse smoke run), ``--no-plots``,
``--output-dir DIR`` (default ``results/cohesive_phasefield/<case>`` next to
this script).
"""
from pathlib import Path

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.solids  # noqa: F401

# Cases selected with --case; see the module docstring.
CASES = ["full_plate", "single_edge_notched", "dynamic"]


def build_params(case: str = "full_plate", quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Return the parameters of one case; ``quick`` gives a coarse, short run."""
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}.")

    results_dir = output_dir  # the class body below has its own output_dir parameter

    class Params(ParamsTemplate):
        ## Mesh: geometry and element sizes, read by the mesh generator named in MeshType
        dim: int = 2
        ell = 0.01                      # Phase-field length scale (m), also sets the refinement size and crack-growth step
        if case == "full_plate":
            MeshType: str = "FullPlateWithHole"
            dx: float = 0.05            # Background element size (m)
            Lx: float = 1.0             # Plate width (m)
            Ly: float = 1.0             # Plate height (m)
            R: float = 0.1              # Hole semi-axis along x before the rotation (m)
            ellipticity: float = 1.1    # Semi-axis ratio x / y of the elliptical hole (-)
            rotation_angle: float = 45.0  # Rotation of the hole about the plate centre (deg)
        elif case == "single_edge_notched":
            MeshType: str = "SingleEdgeNotched"
            dx: float = 0.05            # Element size (m)
            Lx: float = 1.0             # Plate width (m)
            Ly: float = 1.0             # Plate height (m)
            LNotch: float = 0.5         # Notch length from the left edge, at mid-height (m)
            NotchHeight: float = 0.001  # Notch opening (m)
        else:
            MeshType: str = "SingleEdgeNotched"
            dx: float = 0.025
            Lx: float = 1.0
            Ly: float = 0.5
            LNotch: float = 0.25
            NotchHeight: float = 0.001
        Quadrature_Degree: int = 3      # Quadrature degree of the point-wise fields (stresses, internal variables)

        ## Mesh refinement: refine where the cohesive strain etaNorm (written by the solid model) appears
        MeshRefinement = RefinementConfig(
            enabled=True,
            # Mark the cells whose largest etaNorm value is at least 1e-8
            indicator=FieldThresholdIndicatorConfig(field="etaNorm", threshold=1e-8, reduction="cell_max",
                                                    comparison="greater_equal"),
            target_h=ell / 2,           # Refine the marked cells down to this element size (m)
            buffer_layers=2,            # Also refine within buffer_layers * target_h of the marked cells
            check_interval=1,           # Check for refinement every n-th time step
            max_levels_per_step=10,     # Refinement passes per time step, each re-solving the same step
            max_events=int(1e99),       # Limit on the total number of refinements (none here)
            repartition=True,           # Rebalance the cells over the MPI ranks after a refinement ...
            repartition_imbalance_threshold=1.1,  # ... when max / mean cells per rank exceeds this
            replot_on_refinement=plots,  # Redraw the mesh plot after each refinement
            save_remesh_animations=plots and not quick,  # Save mesh_refinement.mp4 in movie_folder
        )

        ## Materials: one entry per material, referred to by name from the models
        Materials: dict = {}
        Materials["Iron"] = {
            "Density": 8.0e3,           # (kg/m^3)
            "Youngs": 200e9,            # Young's modulus (Pa)
            "Poisson": 0.3,
            "LE_Damping": 1.0e6 * 8.0e3,  # Damping coefficient (Pa s/m^2): a rate of 1e6 1/s times the density
            "ell": ell,                 # Phase-field length scale (m), used by Solid:PhaseField
            "f_t": 150e6,               # Tensile strength (Pa)
            "f_s": 150e6,               # Shear strength (Pa)
            "Gc": 1e4,                  # Fracture energy (J/m^2)
            "pf_damping": 0.0,          # Viscous damping of the phase field (J s/m^3)
            "PF_formulation": "HO-AT2",  # Phase-field crack density: "AT1", "AT2", or their higher-order "HO-" forms
            "FailureType": "r1",        # Cohesive failure potential (strength surface) of the solid model
            "DP_phi": 0,                # Friction angle of the Drucker-Prager-like compression hardening (deg), *_DP* failure types only
            "DP_q": 300.0e6,            # Maximum confined shear strength of the hardening law (Pa), *_DP* failure types only
            "Inertia": True,            # Include the inertia term
            "kappa": 1.0e-6,            # Residual stiffness of fully damaged material (-)
        }
        if case == "dynamic":
            Materials["Iron"].update(LE_Damping=0.0, Gc=1e3, FailureType="r2")
        # Elastic constants derived from Youngs and Poisson, as read by the solid model
        iron = Materials["Iron"]
        iron["Shear"] = iron["Youngs"] / (2.0 * (1.0 + iron["Poisson"]))
        iron["Bulk"] = iron["Youngs"] / (3.0 * (1.0 - 2.0 * iron["Poisson"]))
        iron["Lame"] = iron["Shear"] * (2.0 * iron["Poisson"]) / (1.0 - 2.0 * iron["Poisson"])
        gravity: float = 0.0            # Gravitational acceleration along the last axis (m/s^2)

        ## Models: the fields, their staggered solution steps and the models acting on them
        Solution_Steps = {"u": 0, "phasefield": 1}  # Staggered step of each field: displacements first, then the phase field
        Step_Solvers = {0: "Newton"}    # Nonlinear solver per step (default "Newton")
        Field_Orders = {"u": 2, "phasefield": 2}  # Polynomial order of each field

        # Each entry is one model, built from its "Type"; boundary conditions are models as well.
        # The solid writes the crack driving force PsiField, which the phase-field model reads.
        Models: dict = {
            "Solid": {"Type": "Solid:Cohesive_LinearElastic",
                      "Material": "Iron",
                      "DamageModel": "PhaseField"
            },
            # Irreversible_Method "Hist": the driving force is kept at its maximum over time (history field)
            "pf": {"Type": "Solid:PhaseField",
                   "Material": "Iron",
                   "CDF_field": "PsiField",
                   "Irreversible_Method": "Hist"
            },
        }
        if case == "dynamic":
            # Sudden tensile traction (Pa) on the top and bottom, degraded with the phase field (DamagedForces)
            Models["BC_Bottom"] = {"Type": "Boundary:ExternalForce",
                                   "boundary": "bottom",
                                   "field": "u_y",
                                   "value": -10.0e6,
                                   "DamagedForces": True
            }
            Models["BC_Top"] = {"Type": "Boundary:ExternalForce",
                                "boundary": "top",
                                "field": "u_y",
                                "value": 10.0e6,
                                "DamagedForces": True
            }
        else:
            # Fixed bottom; the top moves at "rate" (m/s), enforced by a penalty of stiffness "Dummy" (Pa/m).
            # The run ends 50 steps after the reaction force BC_Top_F has dropped below F_Cutoff times its peak.
            Models["BC_Bottom"] = {"Type": "Boundary:PrescribedValue",
                                   "boundary": "bottom",
                                   "field": "u_y",
                                   "value": 0.0
            }
            Models["BC_Bottom2"] = {"Type": "Boundary:PrescribedValue",
                                    "boundary": "bottom",
                                    "field": "u_x",
                                    "value": 0.0
            }
            if case == "full_plate":  # Compression
                Models["BC_Top"] = {"Type": "Boundary:DisplacementControl",
                                    "boundary": "top",
                                    "field": "u_y",
                                    "rate": -3.0e-6,
                                    "Dummy": 1.0e16,
                                    "F_Cutoff": 0.1
                }
            else:  # Shear
                Models["BC_Top"] = {"Type": "Boundary:DisplacementControl",
                                    "boundary": "top",
                                    "field": "u_x",
                                    "rate": 1.0e-6,
                                    "Dummy": 1.0e14,
                                    "F_Cutoff": 0.1
                }
                Models["BC_Top2"] = {"Type": "Boundary:PrescribedValue",
                                     "boundary": "top",
                                     "field": "u_y",
                                     "value": 0.0
                }
        ModelNames = list(Models)       # The models to build, in this order

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        n_passes: int = 3               # Maximum passes over the staggered steps; stops early once all have converged
        max_it: int = 50                # Maximum Newton iterations per step and pass
        line_search: bool = True
        line_search_lims: list = [1e-2, 1.0]  # Smallest and largest line-search step factor
        rel_tol: float = 1e-3           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1.0e2 if case == "dynamic" else 1.0  # ... or below abs_tol

        ## Time: time span (s), Newmark parameters and adaptive time step
        start_time: float = 0.0
        if case == "dynamic":
            alpha = 0.5                 # Numerical damping of the Newmark scheme (0: none)
            Newmark_Beta: float = 0.25 * (1 + alpha)**2
            Newmark_Gamma: float = 0.5 + alpha
            end_time: float = 1.0e-3
            dt: float = 1e-5
            crack_increment = ell * 0.1  # Largest crack growth per step (m)
        else:
            Newmark_Beta: float = 0.5625  # The damped scheme above with alpha = 0.5
            Newmark_Gamma: float = 1.0
            end_time: float = 200.0 * 670.0
            dt: float = 10.0
            crack_increment = ell
        # "Measure": a step whose Crack_Length_Change (from Solid:PhaseField) exceeds Target_Change is repeated
        # with half the time step (down to dt_min). Otherwise dt grows by 1.2 per step up to dt_list[i], the
        # largest step from time_list[i] on (dt_max is used only without time_list).
        TimeStepControl = {"Type": "Measure",
                           "time_list": [0.0],
                           "dt_list": [dt],
                           "dt_min": 1.0e-10,
                           "dt_max": dt,
                           "Measure": "Crack_Length_Change",
                           "Target_Change": crack_increment
        }

        ## Output: field files and live plots
        output_dir: str = None          # Folder of the field files (None: no files written)
        write_vtk: bool = False
        write_hdf5: bool = False
        Plot_Fields: list = ["u_x", "u_y", "phasefield"] if plots else []  # Fields plotted during the run
        # Global measures plotted against each other, as (x, y); "t" is the time
        Plot_Measures: list = [("t", "Crack_Length")] if case == "dynamic" else [("t", "Crack_Length"), ("BC_Top_u", "BC_Top_F")]
        if not plots:
            Plot_Measures = []
        record_movie: bool = plots and not quick  # Record the field plots as movies in movie_folder
        movie_fps: int = 10
        movie_folder: str = str(results_dir) if results_dir is not None else None

    params = Params()
    if quick:
        params.dx *= 2
        params.end_time = params.start_time + 5 * params.dt
    return params


def run(case: str = "full_plate", quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Run one case and save its figures to ``output_dir``."""
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__, case)
    params = build_params(case, quick=quick, plots=plots, output_dir=output_dir)

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
    args = parse_example_arguments(__file__, __doc__, cases=CASES)
    run(args.case, quick=args.quick, plots=args.plots, output_dir=args.output_dir)
