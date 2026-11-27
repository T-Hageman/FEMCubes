import numpy as np
from dolfinx import mesh as dmesh
from mpi4py import MPI

from femcubes.core.base_model import BaseModel
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm


class CrackDepthMeasure(BaseModel):
    r"""Depth of a phase-field crack below the top of the mesh.

    A post-processing model that follows the growth of surface cracks, such
    as crevasses in glaciers, through a single global measure. The depth is
    the vertical distance between the highest mesh vertex and the lowest
    vertex where :math:`\phi > 0.5`, along the last coordinate axis, taking
    :math:`\phi = 0.5` as the edge of the diffuse crack. Supports 2-D and
    3-D.

    .. rubric:: Fields

    * ``phasefield``: read only, provided by :doc:`phase_field`, evaluated at
      the mesh vertices. The model defines no fields.

    .. rubric:: Weak form

    None.

    .. rubric:: Outside the weak form

    * Global measure ``Crack_Depth``
      :math:`= \max_v x_d - \min_{v:\,\phi(v)>0.5} x_d` over the vertices
      :math:`v` (of the domain group ``domain``; both extremes use the same
      vertices), zero when no vertex has :math:`\phi > 0.5`. Updated with the
      other global measures; initialized to 0.
    * No Dirichlet conditions or state variables.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                               # REQUIRED: 2 or 3
            Solution_Steps = {"phasefield": 1}    # REQUIRED: must contain "phasefield"
            Models = {
                "CrackDepth": {
                    "Type": "Solid:CrackDepthMeasure",  # REQUIRED
                    "domain": "domain",                 # OPTIONAL (default "domain", the full mesh): domain group;
                                                        # the phase field must exist on all of it
                },
            }
    """

    MODEL_TYPE = "Solid:CrackDepthMeasure"

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        self.dim = params.dim
        self.My_Step = params.Solution_Steps["phasefield"]

        self.params.Global_Measures["Crack_Depth"] = 0.0

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):

        return

    def Evaluate_Crack_Depth(self, phasefield, Oldphasefield) -> float:
        """Evaluate, store and return the crack depth of a phase-field function.

        ``phasefield`` is a DOLFINx function, or a field view exposing
        ``collapse_component``; ``Oldphasefield`` is unused.
        """
        Depth = np.inf

        # Get minimum y coordinate where phasefield is above 0.5
        field_output = phasefield
        if hasattr(field_output, "collapse_component"):
            field_func = field_output.collapse_component(0)
        else:
            field_func = field_output
        field_func.x.scatter_forward()

        field_mesh = field_func.function_space.mesh
        vertices, cells = self._domain_vertices(field_mesh)
        points = dmesh.compute_midpoints(field_mesh, 0, vertices)
        max_y_local = float(np.max(points[:, self.dim-1])) if vertices.size > 0 else -np.inf
        if vertices.size > 0:
            pf_vals = field_func.eval(points, cells)
            if pf_vals.ndim > 1:
                pf_vals = pf_vals[:, 0]
            mask = pf_vals > 0.5
            if np.any(mask):
                Depth = float(np.min(points[mask, self.dim-1]))

        # sync across processes
        min_Depth = comm.allreduce(Depth, op=MPI.MIN)
        max_y = comm.allreduce(max_y_local, op=MPI.MAX)

        crack_depth = 0.0 if not np.isfinite(min_Depth) else max(0.0, max_y - min_Depth)

        self.params.Global_Measures["Crack_Depth"] = crack_depth
        return crack_depth

    def Update_Global_Measures(self) -> None:
        phi = self.mesh.Get_Field_For_Output("phasefield")
        self.Evaluate_Crack_Depth(phi, None)
        return

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _domain_vertices(self, field_mesh):
        """PRIVATE: Return the owned vertices of the model's domain on ``field_mesh``, each with one adjacent cell."""
        tdim = field_mesh.topology.dim
        field_mesh.topology.create_connectivity(tdim, 0)
        field_mesh.topology.create_connectivity(0, tdim)
        cell_to_vertex = field_mesh.topology.connectivity(tdim, 0)
        cells = self._domain_cells(field_mesh)
        num_owned = field_mesh.topology.index_map(0).size_local
        vertices = np.concatenate([cell_to_vertex.links(cell) for cell in cells]) if cells.size else np.empty(0, dtype=np.int32)
        adjacent = np.repeat(cells, [cell_to_vertex.links(cell).size for cell in cells]) if cells.size else np.empty(0, dtype=np.int32)
        vertices, first = np.unique(vertices, return_index=True)
        owned = vertices < num_owned
        return vertices[owned].astype(np.int32), adjacent[first][owned].astype(np.int32)

    def _domain_cells(self, field_mesh):
        """PRIVATE: Return the local cells of ``field_mesh`` in the model's domain."""
        tdim = field_mesh.topology.dim
        if self.domain == "domain":
            index_map = field_mesh.topology.index_map(tdim)
            return np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
        parent_cells = np.asarray(self.mesh.get_domain_cells(self.domain), dtype=np.int32)
        if field_mesh is self.mesh.mesh:
            return parent_cells
        support = self.mesh.Get_Field_Support("phasefield")
        if support.mesh is not field_mesh:
            raise ValueError("Solid:CrackDepthMeasure can only restrict the registered phase field to a domain.")
        cells = np.asarray(support.cell_map.sub_topology_to_topology(parent_cells, inverse=True), dtype=np.int32)
        if np.any(cells < 0):
            raise ValueError(f"Solid:CrackDepthMeasure needs the phase field on all of domain '{self.domain}'.")
        return cells
