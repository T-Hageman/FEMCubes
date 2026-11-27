"""Notched tensile bar split into several physics domains.

A double-edge-notched steel bar is loaded in tension through its left and
right ends. The mesh (``MeshType = "NotchedTensileBar"``) consists of three
conforming subdomains:

* ``domain:left`` and ``domain:right`` -- grips and shank, linear elastic.
* ``domain:centre`` -- the notched region, linear elastic with phase-field
  fracture, so the phase field only has degrees of freedom here.

Every model names the (sub)domain it acts on through its ``"domain"`` entry,
and boundaries follow the same ``<boundary>:<subdomain>`` scheme, e.g.
``top`` covers ``top:left``, ``top:centre`` and ``top:right``. The bar is
clamped on its left end and loaded by a displacement-controlled right end
until it has fractured.

Current restrictions: adaptive mesh refinement and Dirichlet boundary
conditions on subdomain-only fields are not yet supported.

Run with::

    python examples/fracture/multidomain_tensile_bar.py
    mpirun -n 4 python examples/fracture/multidomain_tensile_bar.py

Options: ``--quick`` (short, coarse smoke run), ``--no-plots``,
``--output-dir DIR`` (default ``results/multidomain_tensile_bar`` next to
this script).
"""
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem

from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.utils.example_utils import default_output_dir, parse_example_arguments
from femcubes.core.utils.maths_utils import ft_to_Gc
from femcubes.core.utils.mpi_utils import comm, isolate_jit_cache, mprint

# Importing the model packages registers their models with Physics.
import femcubes.boundary  # noqa: F401
import femcubes.solids  # noqa: F401


def build_params(quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Return the parameters of the bar; ``quick`` gives a coarse, quickly damaging, short run."""
    results_dir = output_dir  # the class body below has its own output_dir parameter

    class Params(ParamsTemplate):
        Global_Measures: dict = {}      # Scalar results of the models (forces, crack length), filled during the run

        ## Mesh: geometry and element sizes, read by the mesh generator named in MeshType
        MeshType: str = "NotchedTensileBar"
        dim: int = 2                    # Only 2-D is supported
        Lx: float = 0.1                 # Bar length in the loading direction (m)
        Ly: float = 0.02                # Bar height (m)
        Lx_centre: float = 0.02         # Width of the central fracture domain (m)
        Notch_Width: float = 1.0e-3     # Width of each notch (m)
        Notch_Depth: float = 4.0e-3     # Depth of each notch (m)
        Notch_Sides: str = "both"       # Notch the "top", "bottom", or "both" edges
        dx: float = 4.0e-3 if quick else 2.0e-3         # Element size in the outer domains (m)
        dx_centre: float = 1.0e-3 if quick else 2.5e-4  # Element size in the centre domain (m)
        Quadrature_Degree: int = 3      # Quadrature degree of the point-wise fields (stresses, internal variables)

        ## Materials: one entry per material, referred to by name from the models
        Materials: dict = {}
        Materials["Steel"] = {
            "Density": 7.85e3,          # (kg/m^3)
            "Youngs": 210.0e9,          # Young's modulus (Pa)
            "Poisson": 0.3,
            "Inertia": False,           # Quasi-static loading
            "ell": 2.0e-3 if quick else 1.0e-3,     # Phase-field length scale (m), ~4 centre elements
            "f_t": 100.0e6 if quick else 500.0e6,   # Tensile strength (Pa)
            "kappa": 1.0e-6,            # Residual stiffness of fully damaged material (-)
            "PF_formulation": "AT2",    # Phase-field crack density: "AT1", "AT2", or their higher-order "HO-" forms
            "pf_damping": 0.0,          # Viscous damping of the phase field (J s/m^3)
        }
        # Elastic constants read by the solid models, and the fracture energy (J/m^2) that gives the
        # tensile strength f_t for this length scale
        steel = Materials["Steel"]
        steel["Shear"] = steel["Youngs"] / (2.0 * (1.0 + steel["Poisson"]))
        steel["Lame"] = steel["Shear"] * (2.0 * steel["Poisson"]) / (1.0 - 2.0 * steel["Poisson"])
        steel["Gc"] = ft_to_Gc(steel["f_t"], steel["Youngs"], steel["ell"], steel["PF_formulation"])
        gravity: float = 0.0            # Gravitational acceleration along the last axis (m/s^2)

        ## Models: displacements in all subdomains, the phase field only in domain:centre
        Solution_Steps = {"u": 0, "phasefield": 1}  # Staggered step of each field: displacements first, then the phase field
        Field_Orders = {"u": 2, "phasefield": 2}  # Polynomial order of each field
        # Each entry is one model, built from its "Type"; "domain" restricts it to a subdomain of the mesh.
        # The damaged centre writes the crack driving force ElasticEnergy ("Stress-Based" by default), read by PF_Centre.
        Models: dict = {
            "Solid_Left": {"Type": "Solid:LinearElastic",
                           "Material": "Steel",
                           "domain": "domain:left"
            },
            "Solid_Right": {"Type": "Solid:LinearElastic",
                            "Material": "Steel",
                            "domain": "domain:right"
            },
            "Solid_Centre": {"Type": "Solid:LinearElastic",
                             "Material": "Steel",
                             "DamageModel": "PhaseField",
                             "domain": "domain:centre"
            },
            "PF_Centre": {"Type": "Solid:PhaseField",
                          "Material": "Steel",
                          "CDF_field": "ElasticEnergy",
                          "Irreversible_Method": "Hist",
                          "domain": "domain:centre"
            },
            # Clamped left end
            "BC_Left_x": {"Type": "Boundary:PrescribedValue",
                          "boundary": "left",
                          "field": "u_x",
                          "value": 0.0,
                          "domain": "domain:left"
            },
            "BC_Left_y": {"Type": "Boundary:PrescribedValue",
                          "boundary": "left",
                          "field": "u_y",
                          "value": 0.0,
                          "domain": "domain:left"
            },
            # Displacement-controlled right end at "rate" (m/s), through a penalty of stiffness "Dummy" (Pa/m),
            # recording BC_Right_u and BC_Right_F; the run ends 50 steps after the force has dropped below
            # F_Cutoff = 5% of its peak
            "BC_Right": {"Type": "Boundary:DisplacementControl",
                         "boundary": "right",
                         "field": "u_x",
                         "rate": 1.0e-6,
                         "Dummy": 1.0e16,
                         "F_Cutoff": 0.05,
                         "domain": "domain:right"
            },
        }
        ModelNames = list(Models)       # The models to build, in this order

        ## Solver: nonlinear and linear solvers of each time step
        Linear_Solver: LinearSolver = LinearSolver.LU  # Direct solver (MUMPS)
        n_passes: int = 3               # Maximum passes over the staggered steps; stops early once all have converged
        max_it: int = 50                # Maximum Newton iterations per step and pass
        line_search: bool = True
        line_search_lims: list = [1e-2, 1.0]  # Smallest and largest line-search step factor
        rel_tol: float = 1e-3           # Converged when the RMS residual is below rel_tol times that after the first update ...
        abs_tol: float = 1.0            # ... or below abs_tol

        ## Time: up to 1 mm imposed displacement
        Newmark_Beta: float = 0.5625    # Newmark parameters, read by the solid models also without inertia
        Newmark_Gamma: float = 1.0
        start_time: float = 0.0
        end_time: float = 15.0 if quick else 1000.0
        dt: float = 5.0
        # "Measure": a step whose Crack_Length_Change (from Solid:PhaseField) exceeds Target_Change (m) is
        # repeated with half the time step (down to dt_min). Otherwise dt grows by 1.2 per step up to dt_list[i],
        # the largest step from time_list[i] on (dt_max is used only without time_list).
        TimeStepControl = {"Type": "Measure",
                           "time_list": [0.0],
                           "dt_list": [dt],
                           "dt_min": 1.0e-8,
                           "dt_max": dt,
                           "Measure": "Crack_Length_Change",
                           "Target_Change": steel["ell"]
        }

        ## Output: field files and live plots
        output_dir: str = None          # Folder of the field files (None: no files written)
        write_vtk: bool = False
        write_hdf5: bool = False
        Interactive_Plots: bool = True  # Keep the plot windows interactive, e.g. to zoom into the mesh plot
        Plot_Fields: list = ["u_x", "phasefield"] if plots else []  # Fields plotted during the run
        Plot_Measures: list = [("BC_Right_u", "BC_Right_F")] if plots else []  # Global measures plotted as (x, y)
        record_movie: bool = plots and not quick  # Record the field plots as movies in movie_folder
        movie_fps: int = 10
        movie_folder: str = str(results_dir) if results_dir is not None else None

    return Params()


def print_domain_summary(mesh):
    """Print the global cell count and area of every named domain."""
    dx = ufl.Measure("dx", domain=mesh.mesh, subdomain_data=mesh.cell_tags)
    one = fem.Constant(mesh.mesh, 1.0)
    num_owned_cells = mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_local

    mprint("\n=== Domain Summary ===")
    for name in mesh.domain_names():
        cells = mesh.get_domain_cells(name)
        num_cells = comm.allreduce(int(np.count_nonzero(cells < num_owned_cells)))
        area = comm.allreduce(fem.assemble_scalar(fem.form(one * dx(mesh.get_domain_tags(name)))))
        mprint(f"  {name}: {num_cells} cells, area = {area:.4e} m^2")


def run(quick: bool = False, plots: bool = True, output_dir: Path | None = None):
    """Run the tensile test until the bar has fractured and save the figures to ``output_dir``."""
    isolate_jit_cache()
    output_dir = output_dir if output_dir is not None else default_output_dir(__file__)
    params = build_params(quick=quick, plots=plots, output_dir=output_dir)

    mesh = Mesh(params)
    mesh.print_mesh_groups()
    print_domain_summary(mesh)
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
    args = parse_example_arguments(__file__, __doc__)
    run(quick=args.quick, plots=args.plots, output_dir=args.output_dir)
