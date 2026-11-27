"""Base parameter template shared by all FEMCubes simulations.

Concrete simulations typically define a small subclass of
:class:`ParamsTemplate` and populate it with mesh, material, model, and
time-integration settings.  The template supplies:

* sane solver and output defaults
* fresh per-instance copies of mutable configuration attributes
* a convenience helper for deriving isotropic elastic constants
"""

from femcubes.core.mesh.refinement.config import RefinementConfig, normalize_refinement_config
from femcubes.core.solvers.solver_enums import LinearSolver


class ParamsTemplate:
    """Base template for simulation parameters.

    Subclass this and set the attributes relevant to your simulation.
    Only ``Global_Measures`` and the solver defaults are provided here;
    everything else (mesh, materials, models, time) must be defined by
    the user in a concrete subclass.

    Notes
    -----
    Most examples in FEMCubes define a nested ``Params`` class inside a
    test or script:

    .. code-block:: python

       class Params(ParamsTemplate):
           MeshType = "Rectangle"
           dim = 2
           Lx = 10.0
           Ly = 5.0
           dx = 0.5
           Materials = {...}
           Models = {...}
           ModelNames = [...]
           Solution_Steps = {"u": 0}
           Field_Orders = {"u": 2}
           start_time = 0.0
           end_time = 1.0
           dt = 0.1

    Instantiating the subclass copies mutable class attributes like
    ``Global_Measures`` or ``Plot_Fields`` onto the instance so tests or
    runs never share state accidentally.
    """

    DEBUG_PRINT_TIMING: bool = False
    DEBUG_FIELD_CONVERGENCE: bool = False
    PROFILE_TIMINGS: bool = False

    # ── Solver defaults (override as needed) ────────────────────────
    Linear_Solver: LinearSolver = LinearSolver.LU
    n_passes: int = 1
    max_it: int = 100
    LBFGS_History: int = 10
    LBFGS_LineSearch_MaxIter: int = 20
    LBFGS_LineSearch_Tolerance: float = 0.9
    LBFGS_LineSearch_MaxResidualGrowth: float = 2.0
    LBFGS_StagnationIterations: int = 5
    LBFGS_RecoveryLineSearchTolerance: float = 0.1
    LBFGS_PowellDamping: float = 0.2
    line_search: bool = True
    rel_tol: float = 1e-6
    abs_tol: float = 1e-8
    rel_tol_max_residual: float = float("inf")
    Stagnation_Iterations: int = 0
    Stagnation_Accept_Factor: float | None = None
    Positive_Fields: list = []
    Fraction_Fields: list = []
    Max_Field_Update: dict = {}
    Positive_Update_Factor: float = 1.0e-4

    # ── Output defaults ─────────────────────────────────────────────
    output_interval: int = 10
    output_dir: str = "Results"
    write_vtk: bool = False
    write_hdf5: bool = False
    save_Mesh_every_step: bool = False
    record_movie: bool = False
    movie_fps: int = 10
    RO_zscale: float = 1.0

    # ── Mesh-refinement defaults ─────────────────────────────────────
    MeshRefinement: RefinementConfig = RefinementConfig()

    def __init__(self):
        """Create instance-local copies of mutable configuration values."""
        # Create fresh copies of mutable defaults so instances never
        # share state through the class-level objects.  Only set a
        # default when the subclass hasn't already defined the attribute
        # (as a class-level annotation or assignment).
        if not hasattr(self, "Global_Measures"):
            self.Global_Measures = {}
        else:
            self.Global_Measures = dict(self.Global_Measures)

        if not hasattr(self, "line_search_lims"):
            self.line_search_lims = [0.01, 1.0]
        else:
            self.line_search_lims = list(self.line_search_lims)

        if not hasattr(self, "output_fields"):
            self.output_fields = []
        else:
            self.output_fields = list(self.output_fields)

        if not hasattr(self, "Plot_Fields"):
            self.Plot_Fields = []
        else:
            self.Plot_Fields = list(self.Plot_Fields)

        if not hasattr(self, "Plot_Isosurfaces"):
            self.Plot_Isosurfaces = {}
        else:
            self.Plot_Isosurfaces = dict(self.Plot_Isosurfaces)

        if not hasattr(self, "Plot_RO_Fields"):
            self.Plot_RO_Fields = []
        else:
            self.Plot_RO_Fields = list(self.Plot_RO_Fields)

        if not hasattr(self, "Plot_Measures"):
            self.Plot_Measures = []
        else:
            self.Plot_Measures = list(self.Plot_Measures)

        self.MeshRefinement = normalize_refinement_config(self.MeshRefinement)

    # ── Helpers ──────────────────────────────────────────────────────
    def _SetElasticParams(self, matName: str):
        """Derive isotropic elastic constants from ``Youngs`` and ``Poisson``.

        Parameters
        ----------
        matName : str
            Key inside :attr:`Materials` naming the material to update.

        Notes
        -----
        The helper adds or overwrites the following entries in
        ``self.Materials[matName]``:

        * ``"Bulk"``
        * ``"Lame"``
        * ``"Shear"``
        """
        E = self.Materials[matName]["Youngs"]
        nu = self.Materials[matName]["Poisson"]
        lame = E * nu / ((1 + nu) * (1 - 2 * nu))
        shear = E / (2 * (1 + nu))
        bulk = E / (3 * (1 - 2 * nu))
        self.Materials[matName]["Bulk"] = bulk
        self.Materials[matName]["Lame"] = lame
        self.Materials[matName]["Shear"] = shear
