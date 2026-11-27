from typing import Any

import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D rectangle with a refined top-layer configuration."""

    class RectangleRefinedTopSampleParameters(ParamsTemplate):
        MeshType = "Rectangle_RefinedTop"
        dim = 2
        Lx = 1000.0
        Ly = 300.0
        Ly_refined = 20.0
        dx_min = 1.0
        dx_max = 20.0

    return RectangleRefinedTopSampleParameters()


class RectangleRefinedTopMeshGenerator(MeshGeneratorBase):
    """Generate a 2-D rectangle with a finely meshed top layer.

    The domain is ``Lx`` × ``Ly``. The top ``Ly_refined`` metres use
    element size ``dx_min``; the remaining bulk uses ``dx_max``.
    Only 2-D is supported; passing ``dim=3`` raises ``NotImplementedError``.

    Required parameters
    -------------------
    Lx : float
        Width of the domain [m].
    Ly : float
        Full height of the domain [m].
    Ly_refined : float
        Height of the finely meshed top layer [m].  Must be < ``Ly``.
    dx_min : float
        Element size in the refined top layer [m].
    dx_max : float
        Element size in the coarse bulk region [m].
    dim : int
        Spatial dimension.  Only 2 is supported.

    Optional parameters
    -------------------
    AlignWithRectangle : dict or None
        When provided, embed a conforming sub-rectangle in the mesh so that
        element edges align exactly with its boundaries.  The dict must
        contain ``x_max``, ``y_min`` and ``y_max``, and may contain
        ``x_min`` (default 0); all in metres. Useful when a field
        discontinuity (e.g. an ice lens) coincides with a known rectangle
        and jagged element crossings are undesirable.
    Lx_topleft : float
        Width of the ``top-left`` boundary group [m]. A vertical edge at
        ``x = Lx_topleft`` in the top layer splits the top surface into a
        narrow inlet strip and the remaining top. Defaults to ``dx_min``
        (a strip of about one element).

    Boundary groups
    ---------------
    ``bottom``, ``right`` and ``left``: the outer edges ``y = 0``,
    ``x = Lx`` and ``x = 0``. ``top-left``: the top edge from ``x = 0`` to
    ``x = Lx_topleft``, intended for a localised water-influx condition.
    ``top``: the remaining top edge, from ``x = Lx_topleft`` to ``x = Lx``.
    """

    MESH_TYPE = "Rectangle_RefinedTop"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim != 2:
            raise NotImplementedError(
                "Rectangle_RefinedTop only supports 2-D meshes (dim=2). "
                "3-D support is not yet implemented."
            )
        return self._generate_2d(params)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Build the 2-D rectangle with a refined top layer."""
        LX = params.Lx
        LY = params.Ly
        LY_ref = params.Ly_refined
        dx_min = params.dx_min
        dx_max = params.dx_max
        Lx_topleft = getattr(params, "Lx_topleft", dx_min)
        align = getattr(params, "AlignWithRectangle", None)

        if LY_ref >= LY:
            raise ValueError(
                f"Ly_refined ({LY_ref}) must be less than Ly ({LY})."
            )

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rectangle_refined_top")

        # Two stacked rectangles sharing the horizontal interface at y = LY - LY_ref.
        y_interface = LY - LY_ref

        bulk = gmsh.model.occ.addRectangle(0, 0,           0, LX, y_interface)
        top  = gmsh.model.occ.addRectangle(0, y_interface, 0, LX, LY_ref)

        # A vertical line at x = Lx_topleft through the top strip creates a
        # separate top-left boundary group of that exact width.
        p_influx_bot = gmsh.model.occ.addPoint(Lx_topleft, y_interface, 0)
        p_influx_top = gmsh.model.occ.addPoint(Lx_topleft, LY, 0)
        influx_divider = gmsh.model.occ.addLine(p_influx_bot, p_influx_top)

        # Optionally embed a third rectangle so element edges align with it.
        if align is not None:
            ax_min = align.get("x_min", 0.0)
            ax_max = align["x_max"]
            ay_min = align["y_min"]
            ay_max = align["y_max"]
            lens = gmsh.model.occ.addRectangle(
                ax_min, ay_min, 0, ax_max - ax_min, ay_max - ay_min
            )
            # Fragment all surfaces and the influx divider together.
            gmsh.model.occ.fragment([(2, bulk), (2, top)], [(2, lens), (1, influx_divider)])
        else:
            gmsh.model.occ.fragment([(2, bulk), (2, top)], [(1, influx_divider)])

        gmsh.model.occ.synchronize()

        # ---- Physical groups ----------------------------------------
        # Collect all 2-D surfaces (should be exactly 2 after fragment).
        all_surfaces = [tag for _, tag in gmsh.model.occ.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, all_surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        # Tag boundary edges by position.
        tol = 1e-6 * max(LX, LY)
        bottom_lines, top_left_lines, top_lines, left_lines, right_lines = [], [], [], [], []
        for _, tag in gmsh.model.occ.getEntities(1):
            xmin, ymin, _, xmax, ymax, _ = self._exact_bounding_box(1, tag)
            if abs(ymin) < tol and abs(ymax) < tol:
                bottom_lines.append(tag)
            elif abs(ymin - LY) < tol and abs(ymax - LY) < tol:
                # Split top into top-left (x in [0, Lx_topleft]) and top (remainder).
                if xmin < tol and xmax <= Lx_topleft + tol:
                    top_left_lines.append(tag)
                else:
                    top_lines.append(tag)
            elif abs(xmin) < tol and abs(xmax) < tol:
                left_lines.append(tag)
            elif abs(xmin - LX) < tol and abs(xmax - LX) < tol:
                right_lines.append(tag)

        gmsh.model.addPhysicalGroup(1, bottom_lines, 1)
        gmsh.model.setPhysicalName(1, 1, "Bottom")
        gmsh.model.addPhysicalGroup(1, right_lines, 2)
        gmsh.model.setPhysicalName(1, 2, "Right")
        gmsh.model.addPhysicalGroup(1, top_lines, 3)
        gmsh.model.setPhysicalName(1, 3, "Top")
        gmsh.model.addPhysicalGroup(1, left_lines, 4)
        gmsh.model.setPhysicalName(1, 4, "Left")
        gmsh.model.addPhysicalGroup(1, top_left_lines, 5)
        gmsh.model.setPhysicalName(1, 5, "Top_Left")

        # ---- Mesh size fields ----------------------------------------
        # Field 1: coarse size everywhere.
        gmsh.model.mesh.field.add("MathEval", 1)
        gmsh.model.mesh.field.setString(1, "F", str(dx_max))

        # Field 2: fine size in the top layer (y > y_interface).
        gmsh.model.mesh.field.add("MathEval", 2)
        gmsh.model.mesh.field.setString(2, "F", str(dx_min))

        # Field 3: restrict field 2 to the top surface region.
        gmsh.model.mesh.field.add("Restrict", 3)
        gmsh.model.mesh.field.setNumber(3, "InField", 2)
        gmsh.model.mesh.field.setNumbers(3, "SurfacesList",
                                         [tag for _, tag in gmsh.model.occ.getEntities(2)
                                          if self._exact_bounding_box(2, tag)[1] >= y_interface - tol])

        # Field 4: take the minimum of the two size fields.
        gmsh.model.mesh.field.add("Min", 4)
        gmsh.model.mesh.field.setNumbers(4, "FieldsList", [1, 3])
        gmsh.model.mesh.field.setAsBackgroundMesh(4)

        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers={"bottom": 1, "right": 2, "top": 3, "left": 4, "top-left": 5},
        )

        gmsh.finalize()
        return generated
