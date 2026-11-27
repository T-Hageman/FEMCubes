"""Composite mesh class assembled from the FEMCubes mesh mix-ins.

The public :class:`Mesh` class is the main entry point used by physics
models, solvers, and user scripts.  It combines mesh generation, field
registration, named boundary/domain access, output, plotting, and
lumped-operator helpers behind one object.
"""

from typing import Any
from femcubes.core.utils.mpi_utils import mprint

from ._mesh_base import _MeshBase
from ._mesh_bc import _MeshBC
from ._mesh_gen import _MeshGen
from ._mesh_dofs import _MeshDOFs
from ._mesh_groups import _MeshGroups
from ._mesh_plotting import _MeshPlotting
from ._mesh_reduced_order_plotting import _MeshReducedOrderPlotting
from ._mesh_refinement import _MeshRefinement
from ._mesh_end import _MeshEnd
from ._mesh_outputs import _MeshOutputs
from ._mesh_lumped_operators import _MeshLumpedOperators

class Mesh(_MeshBase, _MeshDOFs, _MeshGroups, _MeshBC, _MeshGen, _MeshRefinement, _MeshLumpedOperators, _MeshPlotting, _MeshReducedOrderPlotting, _MeshOutputs, _MeshEnd):
    """Composite mesh class for FEMCubes simulations.

    ``Mesh`` is intentionally implemented as a composition of mix-ins so
    that the mesh-related responsibilities remain logically separated in
    the code base while still presenting one object to the rest of the
    framework.  In normal use, models and solvers should interact with
    this composite class rather than with the individual mix-ins.

    The class assembles the following responsibilities:

    * `_MeshBase` -- shared state and DOLFINx mesh reference
    * `_MeshDOFs` -- scalar / vector field management and function spaces
    * `_MeshGroups` -- named boundary and domain markers
    * `_MeshBC` -- Dirichlet boundary-condition construction helpers
    * `_MeshGen` -- delegates mesh creation to registered generators
    * `_MeshRefinement` -- runs and installs adaptive-refinement transactions
    * `_MeshLumpedOperators` -- lumped mass-matrix helpers
    * `_MeshPlotting` -- interactive PyVista visualization
    * `_MeshReducedOrderPlotting` -- 3-D reduced-order surface plots
    * `_MeshOutputs` -- VTK / HDF5 file output
    * `_MeshEnd` -- MRO termination

    Parameters
    ----------
    params : ParamsTemplate
        Simulation parameter object.  Must define at least ``dim``,
        ``MeshType``, and the geometry parameters expected by the chosen
        mesh generator.

    Attributes
    ----------
    params : ParamsTemplate
        Simulation parameter object passed at construction time.
    mesh : dolfinx.mesh.Mesh
        Generated DOLFINx mesh.
    cell_tags : dolfinx.mesh.MeshTags or None
        Cell-wise domain tags produced by the selected mesh generator.
    facet_tags : dolfinx.mesh.MeshTags or None
        Facet-wise boundary tags produced by the selected mesh generator.
    fields : dict[str, dolfinx.fem.Function]
        Registered finite-element fields keyed by field name.
    boundary_markers : dict[str, int]
        Mapping from boundary names to integer facet tags.
    domain_markers : dict[str, int]
        Mapping from domain names to integer cell tags.
    remesh_epoch : int
        Monotonically increasing installed-mesh epoch, initially zero.
    last_refinement_report : RefinementReport or None
        Report from the most recently installed successful refinement.

    Notes
    -----
    The constructor immediately generates the mesh and initialises all
    mesh-related subsystems.  By the time ``Mesh(params)`` returns, the
    object is ready for field registration, boundary lookup, plotting,
    and output configuration.
    """

    def __init__(self, params: Any) -> None:
        super().__init__(params)
        mprint("Mesh generation complete.")
