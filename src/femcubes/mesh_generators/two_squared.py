from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D two-squares configuration."""

    class TwoSquaredSampleParameters(ParamsTemplate):
        MeshType = "TwoSquared"
        dim = 2
        L = 1.0
        dx = 0.2
        dx_interface = 0.05

    return TwoSquaredSampleParameters()


class TwoSquaredMeshGenerator(MeshGeneratorBase):
    """Generate two adjacent 2-D squares sharing one interface.

    The left square spans ``[0, L] x [0, Ly]`` and the right square
    ``[L, 2L] x [0, Ly]``, with ``Ly = L`` unless given (e.g. a thin strip
    for quasi one-dimensional problems). Both squares are conforming subdomains of one
    mesh, so models can act on either side and be coupled through the
    vertical interface at ``x = L``. Only 2-D is supported; passing
    ``dim=3`` raises ``NotImplementedError``.

    Required parameters
    -------------------
    dim : int
        Spatial dimension. Only 2 is supported.
    L : float
        Side length of each square.
    dx : float
        Element size away from the interface.

    Optional parameters
    -------------------
    Ly : float
        Height of both rectangles. Defaults to ``L`` (two squares).
    dx_interface : float
        Element size at the interface, grading back to ``dx`` over a
        distance of ``2 * dx`` on both sides. Defaults to ``dx``.

    Named markers
    -------------
    Domains: ``domain`` = ``domain:left`` + ``domain:right``.
    Boundaries: ``bottom`` and ``top`` (each split into ``:left`` and
    ``:right``), ``left`` = ``left:left``, ``right`` = ``right:right``.
    Interior facets: ``interface`` (between ``domain:left`` and
    ``domain:right``).

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "TwoSquared"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim != 2:
            raise NotImplementedError(
                "TwoSquared only supports 2-D meshes (dim=2). "
                "3-D support is not yet implemented."
            )
        return self._generate_2d(params)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Build the two squares with a conforming interface."""
        L = float(params.L)
        Ly = float(getattr(params, "Ly", L))
        dx = float(params.dx)
        dx_interface = float(getattr(params, "dx_interface", dx))
        if L <= 0.0 or Ly <= 0.0:
            raise ValueError("L and Ly must be positive.")
        if dx <= 0.0 or dx_interface <= 0.0:
            raise ValueError("dx and dx_interface must be positive.")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("two_squared")

        left = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, L, Ly)
        right = gmsh.model.occ.addRectangle(L, 0.0, 0.0, L, Ly)
        # Fragmenting makes both squares share the interface curve, so the
        # mesh is conforming across it.
        gmsh.model.occ.fragment([(2, left)], [(2, right)])
        gmsh.model.occ.synchronize()

        tol = 1e-9 * L

        def side_of(dim, tag):
            return "left" if gmsh.model.occ.getCenterOfMass(dim, tag)[0] < L else "right"

        domain_surfaces = {"domain:left": [], "domain:right": []}
        for _, tag in gmsh.model.getEntities(2):
            domain_surfaces[f"domain:{side_of(2, tag)}"].append(tag)

        domain_markers = {"domain:left": 1, "domain:right": 2}
        for name, marker in domain_markers.items():
            gmsh.model.addPhysicalGroup(2, domain_surfaces[name], marker)
            gmsh.model.setPhysicalName(2, marker, name)

        boundary_groups = {
            "bottom:left": [],
            "bottom:right": [],
            "top:left": [],
            "top:right": [],
            "left:left": [],
            "right:right": [],
            "interface": [],
        }
        for _, tag in gmsh.model.getEntities(1):
            x0, y0, _z0, x1, y1, _z1 = self._exact_bounding_box(1, tag)
            vertical = np.isclose(x0, x1, atol=tol)
            horizontal = np.isclose(y0, y1, atol=tol)
            if horizontal and np.isclose(y0, 0.0, atol=tol):
                boundary_groups[f"bottom:{side_of(1, tag)}"].append(tag)
            elif horizontal and np.isclose(y0, Ly, atol=tol):
                boundary_groups[f"top:{side_of(1, tag)}"].append(tag)
            elif vertical and np.isclose(x0, 0.0, atol=tol):
                boundary_groups["left:left"].append(tag)
            elif vertical and np.isclose(x0, 2.0 * L, atol=tol):
                boundary_groups["right:right"].append(tag)
            elif vertical and np.isclose(x0, L, atol=tol):
                boundary_groups["interface"].append(tag)
            else:
                raise RuntimeError(
                    f"TwoSquared could not classify boundary curve {tag} "
                    f"with bounds ({x0}, {y0}) - ({x1}, {y1})."
                )  # pragma: no cover

        boundary_markers = {name: marker for marker, name in enumerate(boundary_groups, start=1)}
        for name, marker in boundary_markers.items():
            gmsh.model.addPhysicalGroup(1, boundary_groups[name], marker)
            gmsh.model.setPhysicalName(1, marker, name)

        self._set_mesh_size_fields(boundary_groups["interface"], dx, dx_interface)
        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers=domain_markers,
            boundary_markers=boundary_markers,
        )

        gmsh.finalize()

        return generated

    @staticmethod
    def _set_mesh_size_fields(interface_curves, dx, dx_interface) -> None:
        """PRIVATE: Grade from ``dx_interface`` at the interface to ``dx`` at ``2 * dx``."""
        fields = gmsh.model.mesh.field
        fields.add("Distance", 1)
        fields.setNumbers(1, "CurvesList", list(interface_curves))
        fields.add("Threshold", 2)
        fields.setNumber(2, "InField", 1)
        fields.setNumber(2, "SizeMin", dx_interface)
        fields.setNumber(2, "SizeMax", dx)
        fields.setNumber(2, "DistMin", 0.0)
        fields.setNumber(2, "DistMax", 2.0 * dx)
        fields.setAsBackgroundMesh(2)

        # Sizes come exclusively from the background field.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
