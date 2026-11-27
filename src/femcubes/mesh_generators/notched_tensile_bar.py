from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


_NOTCH_SIDES = ("both", "top", "bottom")


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D double-edge-notched tensile bar configuration."""

    class NotchedTensileBarSampleParameters(ParamsTemplate):
        MeshType = "NotchedTensileBar"
        dim = 2
        Lx = 4.0
        Ly = 1.0
        Lx_centre = 1.0
        Notch_Width = 0.1
        Notch_Depth = 0.2
        dx = 0.2
        dx_centre = 0.05

    return NotchedTensileBarSampleParameters()


class NotchedTensileBarMeshGenerator(MeshGeneratorBase):
    """Generate a 2-D notched tensile bar split into three cell domains.

    The bar spans ``[0, Lx] x [0, Ly]`` and is intended to be loaded in
    tension through its ``left`` and ``right`` ends. Rectangular notches of
    width ``Notch_Width`` and depth ``Notch_Depth`` are cut into the top
    and/or bottom edge at ``x = Lx / 2``.

    The bar is partitioned into three conforming cell domains: a centre
    region of width ``Lx_centre`` around the notches, and the two outer
    regions on either side of it. The vertical lines separating them are
    tagged as interior facet groups, so that multi-domain models can couple
    the regions or restrict fields to a subset of them. Only 2-D is
    supported; passing ``dim=3`` raises ``NotImplementedError``.

    Required parameters
    -------------------
    dim : int
        Spatial dimension. Only 2 is supported.
    Lx, Ly : float
        Length (loading direction) and height of the bar.
    Lx_centre : float
        Width of the centre domain, centred at ``x = Lx / 2``. Must be larger
        than ``Notch_Width`` and smaller than ``Lx``.
    Notch_Width, Notch_Depth : float
        Width and depth of each rectangular notch.
    dx : float
        Element size in the outer domains.

    Optional parameters
    -------------------
    dx_centre : float
        Element size in the centre domain. Defaults to ``dx``.
    Notch_Sides : str
        Which edges are notched: ``"both"`` (default), ``"top"``, or
        ``"bottom"``.

    Named markers
    -------------
    Markers are named ``<group>:<subdomain>``; each group name (e.g.
    ``domain`` or ``top``) resolves to the union of its parts.

    Domains: ``domain`` = ``domain:left`` + ``domain:centre`` +
    ``domain:right``.
    Boundaries: ``bottom`` and ``top`` (each split into ``:left``,
    ``:centre`` and ``:right``), ``left`` = ``left:left``,
    ``right`` = ``right:right``, ``notch`` = ``notch:centre``.
    Interior facets: ``interface`` = ``interface:left`` (between
    ``domain:left`` and ``domain:centre``) + ``interface:right`` (between
    ``domain:centre`` and ``domain:right``).

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "NotchedTensileBar"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim != 2:
            raise NotImplementedError(
                "NotchedTensileBar only supports 2-D meshes (dim=2). "
                "3-D support is not yet implemented."
            )
        return self._generate_2d(params)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Build the 2-D three-domain notched tensile bar."""
        geometry = self._resolve_geometry(params)
        LX = geometry["Lx"]
        LY = geometry["Ly"]
        x_c0 = geometry["x_centre_min"]
        x_c1 = geometry["x_centre_max"]
        dx = geometry["dx"]
        dx_centre = geometry["dx_centre"]

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("notched_tensile_bar")

        outer_left = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, x_c0, LY)
        centre = gmsh.model.occ.addRectangle(x_c0, 0.0, 0.0, x_c1 - x_c0, LY)
        outer_right = gmsh.model.occ.addRectangle(x_c1, 0.0, 0.0, LX - x_c1, LY)

        notch_tools = [
            (2, gmsh.model.occ.addRectangle(x0, y0, 0.0, x1 - x0, y1 - y0))
            for x0, y0, x1, y1 in geometry["notch_boxes"]
        ]
        gmsh.model.occ.cut([(2, centre)], notch_tools, removeObject=True, removeTool=True)

        # Fragmenting makes the three regions share their interface curves,
        # so the resulting mesh is conforming across the domain boundaries.
        gmsh.model.occ.fragment(
            [(2, outer_left), (2, outer_right)],
            [entity for entity in gmsh.model.occ.getEntities(2) if entity[1] not in (outer_left, outer_right)],
        )
        gmsh.model.occ.synchronize()

        tol = 1e-9 * max(LX, LY)
        notch_flank_x = [x for x0, _y0, x1, _y1 in geometry["notch_boxes"] for x in (x0, x1)]
        notch_root_y = [y1 if y0 == 0.0 else y0 for _x0, y0, _x1, y1 in geometry["notch_boxes"]]

        def subdomain_of(dim, tag):
            x_mid = gmsh.model.occ.getCenterOfMass(dim, tag)[0]
            if x_mid < x_c0:
                return "left"
            if x_mid > x_c1:
                return "right"
            return "centre"

        def on_any(value, targets):
            return any(np.isclose(value, target, atol=tol) for target in targets)

        domain_surfaces = {"domain:left": [], "domain:centre": [], "domain:right": []}
        for _, tag in gmsh.model.getEntities(2):
            domain_surfaces[f"domain:{subdomain_of(2, tag)}"].append(tag)

        if any(not surfaces for surfaces in domain_surfaces.values()):
            raise RuntimeError(
                "NotchedTensileBar mesh generation failed to create all three domains."
            )  # pragma: no cover

        domain_markers = {"domain:left": 1, "domain:centre": 2, "domain:right": 3}
        for name, marker in domain_markers.items():
            gmsh.model.addPhysicalGroup(2, domain_surfaces[name], marker)
            gmsh.model.setPhysicalName(2, marker, name)

        boundary_groups = {
            "bottom:left": [],
            "bottom:centre": [],
            "bottom:right": [],
            "top:left": [],
            "top:centre": [],
            "top:right": [],
            "left:left": [],
            "right:right": [],
            "notch:centre": [],
            "interface:left": [],
            "interface:right": [],
        }
        for _, tag in gmsh.model.getEntities(1):
            x0, y0, _z0, x1, y1, _z1 = self._exact_bounding_box(1, tag)
            vertical = np.isclose(x0, x1, atol=tol)
            horizontal = np.isclose(y0, y1, atol=tol)
            if horizontal and np.isclose(y0, 0.0, atol=tol):
                boundary_groups[f"bottom:{subdomain_of(1, tag)}"].append(tag)
            elif horizontal and np.isclose(y0, LY, atol=tol):
                boundary_groups[f"top:{subdomain_of(1, tag)}"].append(tag)
            elif vertical and np.isclose(x0, 0.0, atol=tol):
                boundary_groups["left:left"].append(tag)
            elif vertical and np.isclose(x0, LX, atol=tol):
                boundary_groups["right:right"].append(tag)
            elif vertical and np.isclose(x0, x_c0, atol=tol):
                boundary_groups["interface:left"].append(tag)
            elif vertical and np.isclose(x0, x_c1, atol=tol):
                boundary_groups["interface:right"].append(tag)
            elif (vertical and on_any(x0, notch_flank_x)) or (horizontal and on_any(y0, notch_root_y)):
                boundary_groups["notch:centre"].append(tag)
            else:
                raise RuntimeError(
                    f"NotchedTensileBar could not classify boundary curve {tag} "
                    f"with bounds ({x0}, {y0}) - ({x1}, {y1})."
                )  # pragma: no cover

        boundary_markers = {name: marker for marker, name in enumerate(boundary_groups, start=1)}
        for name, marker in boundary_markers.items():
            gmsh.model.addPhysicalGroup(1, boundary_groups[name], marker)
            gmsh.model.setPhysicalName(1, marker, name)

        self._set_mesh_size_fields(
            domain_surfaces["domain:centre"],
            boundary_groups["interface:left"] + boundary_groups["interface:right"],
            dx,
            dx_centre,
        )
        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers=domain_markers,
            boundary_markers=boundary_markers,
        )

        gmsh.finalize()

        return generated

    @staticmethod
    def _set_mesh_size_fields(centre_surfaces, interface_curves, dx, dx_centre) -> None:
        """PRIVATE: Use ``dx_centre`` in the centre domain and ``dx`` elsewhere.

        The outer domains grade from ``dx_centre`` at the interfaces back to
        ``dx`` over a distance of ``2 * dx``, so that the fine centre size does
        not leak far into the outer domains.
        """
        centre_curves = [
            tag
            for _, tag in gmsh.model.getBoundary(
                [(2, tag) for tag in centre_surfaces], combined=False, oriented=False
            )
        ]

        fields = gmsh.model.mesh.field
        fields.add("MathEval", 1)
        fields.setString(1, "F", str(dx))

        fields.add("MathEval", 2)
        fields.setString(2, "F", str(dx_centre))
        fields.add("Restrict", 3)
        fields.setNumber(3, "InField", 2)
        fields.setNumbers(3, "SurfacesList", list(centre_surfaces))
        fields.setNumbers(3, "CurvesList", centre_curves)

        fields.add("Distance", 4)
        fields.setNumbers(4, "CurvesList", list(interface_curves))
        fields.add("Threshold", 5)
        fields.setNumber(5, "InField", 4)
        fields.setNumber(5, "SizeMin", dx_centre)
        fields.setNumber(5, "SizeMax", dx)
        fields.setNumber(5, "DistMin", 0.0)
        fields.setNumber(5, "DistMax", 2.0 * dx)

        fields.add("Min", 6)
        fields.setNumbers(6, "FieldsList", [1, 3, 5])
        fields.setAsBackgroundMesh(6)

        # Sizes come exclusively from the background field.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    def _resolve_geometry(self, params) -> dict[str, Any]:
        """PRIVATE: Validate the bar parameters and return derived geometry."""
        LX = float(params.Lx)
        LY = float(params.Ly)
        Lx_centre = float(params.Lx_centre)
        notch_width = float(params.Notch_Width)
        notch_depth = float(params.Notch_Depth)
        dx = float(params.dx)
        dx_centre = float(getattr(params, "dx_centre", dx))
        notch_sides = getattr(params, "Notch_Sides", "both")

        if LX <= 0.0 or LY <= 0.0:
            raise ValueError("Lx and Ly must be positive.")
        if dx <= 0.0 or dx_centre <= 0.0:
            raise ValueError("dx and dx_centre must be positive.")
        if Lx_centre <= 0.0 or Lx_centre >= LX:
            raise ValueError("Lx_centre must be positive and smaller than Lx.")
        if notch_width <= 0.0 or notch_width >= Lx_centre:
            raise ValueError("Notch_Width must be positive and smaller than Lx_centre.")
        if notch_sides not in _NOTCH_SIDES:
            raise ValueError(f"Notch_Sides must be one of {_NOTCH_SIDES}, got {notch_sides!r}.")

        n_notches = 2 if notch_sides == "both" else 1
        if notch_depth <= 0.0 or n_notches * notch_depth >= LY:
            limit = "Ly / 2" if n_notches == 2 else "Ly"
            raise ValueError(f"Notch_Depth must be positive and smaller than {limit}.")

        x_notch0 = 0.5 * (LX - notch_width)
        x_notch1 = 0.5 * (LX + notch_width)
        notch_boxes = []
        if notch_sides in ("both", "bottom"):
            notch_boxes.append((x_notch0, 0.0, x_notch1, notch_depth))
        if notch_sides in ("both", "top"):
            notch_boxes.append((x_notch0, LY - notch_depth, x_notch1, LY))

        return {
            "Lx": LX,
            "Ly": LY,
            "x_centre_min": 0.5 * (LX - Lx_centre),
            "x_centre_max": 0.5 * (LX + Lx_centre),
            "dx": dx,
            "dx_centre": dx_centre,
            "notch_boxes": notch_boxes,
        }
