"""Advance of an ice sheet over a bed, with the shallow-ice/shallow-shelf equations.

Ice flows in from the left edge of a 20 x 5 km domain (the upper half of a
symmetric valley) with a thickness of 100 m and spreads over the bed. The
ice margin is tracked by a level set (``Glaciology:LevelSet``), basal friction
follows Budd's law, and the ice/ocean pressure difference acts on the front.
The bed is

* ``slope`` -- rising towards the left edge at a constant slope of 2 %, or
* ``flat``.

Run with::

    python examples/glaciology/shallow_ice.py
    mpirun -n 4 python examples/glaciology/shallow_ice.py --case flat

Options: ``--quick`` (short smoke run), ``--no-plots``, ``--output-dir DIR``
(default ``results/shallow_ice/<case>`` next to this script).
"""
from pathlib import Path

import ufl

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.mpi_utils import isolate_jit_cache, mprint

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.glaciology  # noqa: F401
import femcubes.solids  # noqa: F401

# Bed shapes selected with --case, and the length of a year (s)
CASES = ["slope", "flat"]
YEAR = 365.25 * 24 * 3600.0


def build_params(quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Return the parameters (the bed is set in :func:`run`); ``quick`` gives a short run."""
    results_dir = output_dir  # the class body below has its own output_dir parameter

    class Params(ParamsTemplate):
        Global_Measures: dict = {}      # Scalar results of the models, filled during the run

        ## Mesh: plan view of the domain, read by the mesh generator named in MeshType
        MeshType: str = "Rectangle"
        dx: float = 200.0               # Element size (m)
        dim: int = 2
        Lx: float = 20e3                # Length in the flow direction (m)
        Ly: float = 5e3                 # Half-width of the valley (m)
        Quadrature_Degree: int = 3

        ## Ice and ice margin: material entries read by Glaciology:ShallowIce, and the level-set settings
        Materials: dict = {}
        Materials["Ice"] = {
            "Inertia": False,
            "kappa": 1.0e-6,
            "f_t": 0.1e6,               # Tensile strength (Pa)
            "visc_min": 1.0e3,          # Minimum Glen viscosity (Pa.s)
            "visc_max": 1.0e15,         # Maximum Glen viscosity (Pa.s)
            "Friction_Model": "Budd",   # Basal friction law ("Budd" or "Weertman")
            "C_base": 1.0e3,            # Basal friction coefficient: drag C_base * N * v for Budd (s/m), C_base * v for Weertman (Pa s/m)
            "Front_Ocean_Stress": True,  # Ice/ocean pressure difference on the level-set front
        }
        # Weertman coefficient giving about Budd's drag for 100 m of ice on a dry bed (N = rho g h)
        if Materials["Ice"]["Friction_Model"].lower() == "weertman":
            Materials["Ice"]["C_base"] *= 9.81 * 100 * 917
        UseLevelSet = True
        # Transition_Width: half-width of the smoothed ice / no-ice transition around LS = 0 (m).
        # Every Reinitialize_Every steps LS is reset to the signed distance from the margin, after moving
        # ice thinner than Reinitialize_Min_Thickness (m) outside it.
        Level_Set = {"Transition_Width": 4.0 * dx,
                     "Reinitialize_Every": 10,
                     "Reinitialize_Min_Thickness": 1.0e-2
        }
        gravity: float = -9.81          # Gravitational acceleration (m/s^2)

        ## Models: depth-averaged velocity v, thickness h and level set LS, each in its own staggered step
        Solution_Steps = {"v": 0, "h": 1, "LS": 2}
        Field_Orders = {"v": 2, "h": 1, "LS": 1}  # Polynomial order of each field (h is discontinuous)
        # Each entry is one model, built from its "Type"; boundary conditions are models as well.
        # ShallowIceBC: velocity components (m/s) imposed strongly, an inflow thickness "h" (m) by a penalty.
        Models: dict = {
            "Interior": {"Type": "Glaciology:ShallowIce"},
            "BC_Bottom": {"Type": "Glaciology:ShallowIceBC",  # Valley centre line
                          "boundary": "bottom",
                          "v_y": 0.0
            },
            "BC_Left": {"Type": "Glaciology:ShallowIceBC",  # Inflow
                        "boundary": "left",
                        "v_x": 1.0e-5,
                        "v_y": 0.0,
                        "h": 100.0
            },
            "BC_Top": {"Type": "Glaciology:ShallowIceBC",  # Valley side
                       "boundary": "top",
                       "v_y": 0.0
            },
            "IceMargin": {"Type": "Glaciology:LevelSet",  # Advects LS with the ice velocity
                          "Velocity_Field": "v"
            },
        }
        ModelNames = list(Models)       # The models to build, in this order

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        n_passes: int = 1               # One pass over the staggered steps v, h, LS per time step
        max_it: int = 50                # Maximum Newton iterations per step
        line_search: bool = True
        line_search_lims: list = [0.1, 1.0]  # Smallest and largest line-search step factor
        rel_tol: float = 1e-3           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1e-10          # ... or below abs_tol

        ## Time: 100 years in steps of 0.1 year (s); without TimeStepControl, dt changes only after a failed step
        start_time: float = -50 * YEAR
        end_time: float = 50 * YEAR
        dt: float = 0.1 * YEAR

        ## Output: field files and live plots
        output_dir: str = None          # Folder of the field files (None: no files written)
        write_vtk: bool = False
        write_hdf5: bool = False
        ShallowIce_Print_Diagnostics: bool = True  # Print diagnostics of the shallow-ice model
        Plot_Fields: list = ["v_x", "v_y", "h", "z", "LS_phase"] if plots else []  # Fields plotted during the run
        Plot_RO_Fields: list = ["h"] if plots else []  # Fields drawn on a 3-D view of the bed z and the ice surface
        RO_zscale: float = 10.0         # Vertical exaggeration of that 3-D view
        Plot_Measures: list = []
        record_movie: bool = plots and not quick  # Record the field plots as movies in movie_folder
        movie_fps: int = 10
        movie_folder: str = str(results_dir) if results_dir is not None else None

    params = Params()
    if quick:
        params.end_time = params.start_time + 3 * params.dt
    return params


def run(case: str = "slope", quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Run the ice sheet over the chosen bed and save its figures to ``output_dir``."""
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}.")
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__, case)
    params = build_params(quick=quick, plots=plots, output_dir=output_dir)

    mesh = Mesh(params)
    if plots:
        mesh.plot_mesh()
    physics = Physics(mesh, params)
    physics.Initialize_Fields()

    # Bed elevation
    x = ufl.SpatialCoordinate(mesh.mesh)[0]
    mesh.Set_From_Expression("z", 0.02 * (params.Lx - x) if case == "slope" else 0.0 * x)

    # Initial state: 100 m of ice up to the initial front
    mesh.Constant_Field("T", 0.0)  # Uniform temperature for the material properties
    initial_front = max(4.0 * params.dx, params.Level_Set["Transition_Width"])
    mesh.Set_From_Expression("h", ufl.conditional(ufl.le(x, initial_front), 100, 0.0))
    mesh.Set_From_Expression("LS", initial_front - x)
    mesh.Copy_Field("LS", "LSOld")
    # The initial state is set after the models are initialised, so commit it
    # once to make hOld, vOld and LSOld match it.
    physics.commit()

    TimeSolver(params, physics).run()

    if plots:
        mesh.SavePlotsToFolder(str(output_dir))
        mprint(f"Figures saved to {output_dir}")
        mesh.Close_Plots()
    return mesh, physics


if __name__ == "__main__":
    args = parse_example_arguments(__file__, __doc__, cases=CASES)
    run(args.case, quick=args.quick, plots=args.plots, output_dir=args.output_dir)
