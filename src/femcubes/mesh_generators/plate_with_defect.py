from typing import Any

import gmsh
import numpy as np

from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase, model_to_mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D plate-with-defect configuration."""

    class PlateWithDefectSampleParameters(ParamsTemplate):
        MeshType = "PlateWithDefect"
        dim = 2
        W = 4.0
        H = 1.0
        R_defect = 0.5
        dx = 0.4
        dx_metal = 0.1

    return PlateWithDefectSampleParameters()


class PlateWithDefectMeshGenerator(MeshGeneratorBase):
    """Generate an electrolyte layer on a flat metal plate with a circular defect, using symmetry.

    The metal surface is the plane ``y = 0`` (2-D) or ``z = 0`` (3-D) and is
    not meshed; the electrolyte layer of thickness ``H`` above it is. The
    defect is a disk of radius ``R_defect`` centred at the origin, on the
    symmetry axis (2-D: half domain ``0 <= x <= W``) or on both symmetry
    planes (3-D: quarter domain ``0 <= x, y <= W``). The metal surface is
    split into the defect and the surrounding outer surface, so surface
    models (e.g. initial coverages) can differ between them.

    Supported dimensions
    --------------------
    2 (half domain) and 3 (quarter domain).

    Required parameters
    -------------------
    dim : int
        Spatial dimension, 2 or 3.
    W : float
        Extent of the domain from the defect centre (2-D: width; 3-D: side of the square).
    H : float
        Thickness of the electrolyte layer.
    R_defect : float
        Defect radius; must be smaller than ``W``.
    dx : float
        Element size away from the metal.

    Optional parameters
    -------------------
    dx_metal : float
        Element size on the metal surface, grading to ``dx`` over a distance
        of ``2 * dx``. Defaults to ``dx``.
    dx_defect : float
        Element size at the edge of the defect (where the coverages and
        currents change fastest), grading to ``dx_metal`` over a distance
        of ``R_defect``. Defaults to ``dx_metal``.
    dx_pit : float
        Element size on the whole defect surface (an active pit, where the
        electrolyte changes most). Defaults to ``dx_metal`` (no separate
        refinement).
    grading : float
        If given (> 0), element sizes grow linearly with the distance ``d``
        from each refined feature instead of the fixed ramps above:
        ``min(dx, dx_metal + grading * d_metal, dx_pit + grading * d_pit,
        dx_defect + grading * d_edge)``. With ``grading`` ~ 0.1-0.3 a thick
        electrolyte (e.g. 10 mm) needs fine elements only near the metal
        and the defect, and coarse ones (``dx``) in the bulk.
    lateral_grading : float
        If given (> 0), element sizes also grow with the distance ``d`` from
        the defect edge along the plate, beyond ``fine_extent``, through the
        whole layer: ``min(dx_far, max(size, dx_metal + lateral_grading *
        max(0, d - fine_extent)))``, with ``size`` the sizes above. For long
        plates under a thin layer, where the layer would otherwise be meshed
        at the (small) through-thickness size over its whole length.
        Requires ``dx_far``.
    dx_far : float
        Largest element size far from the defect (with ``lateral_grading``).
    fine_extent : float
        Distance from the defect edge over which the sizes are not coarsened
        (with ``lateral_grading``). Defaults to 0.
    weak_spots : sequence of (centre, half_width)
        2-D only: segments ``[centre - half_width, centre + half_width]`` of
        the outer surface marked as separate metal surfaces
        (``metal:spot1``, ``metal:spot2``, ... numbered outwards from the
        defect), e.g. to
        give grains or inclusions their own surface model. They must lie
        within ``(R_defect, W)`` and must not overlap. Their edges are
        refined like the defect edge (``dx_defect``) and, with
        ``lateral_grading``, kept fine over ``feature_extent`` around their
        edges.
    defect_depth : float
        2-D only: the defect is a recess (a pit) of this depth below the
        metal surface, ``[0, R_defect] x [-defect_depth, 0]``, whose bottom
        and wall are both ``metal:defect``. Defaults to 0 (a flat defect).
    voids : sequence of (centre, radius, mouth)
        2-D only: cavities in the metal under the outer surface (e.g. voids
        at a steel-concrete interface), each a circle of ``radius`` below
        the surface that opens into the electrolyte through a mouth
        ``[centre - mouth, centre + mouth]`` (``0 < mouth < radius``); the
        circle's centre lies ``sqrt(radius^2 - mouth^2)`` below the surface.
        Their walls are separate metal surfaces (``metal:void1``, ...
        numbered outwards from the defect), refined to ``dx_void``. They
        must lie within ``(R_defect, W)`` and must not overlap each other
        or the weak spots.
    feature_extent : float
        Distance from the edges of the weak spots and the mouths of the
        voids over which the sizes are not coarsened (with
        ``lateral_grading``). Defaults to ``fine_extent``.
    dx_void : float
        Element size on the void walls, grading back to the other sizes
        over one void radius. Defaults to ``dx_defect``.
    Axisymmetric : bool
        2-D only (mesh option, see the mesh documentation): the half domain
        is the r-z plane of a disc of radius ``W`` with a circular defect at
        its centre; weak spots and voids become rings.

    Named markers
    -------------
    Domains: ``domain`` (the electrolyte).
    Boundaries: ``top`` (the outer boundary of the layer, ``y = H`` or
    ``z = H``), ``far`` (the side(s) at ``x = W``, and ``y = W`` in 3-D),
    ``symmetry`` (``x = 0``, and ``y = 0`` in 3-D), and ``metal`` =
    ``metal:defect`` + ``metal:outer`` (+ ``metal:spot1``, ... with
    ``weak_spots``, and ``metal:void1``, ... with ``voids``).
    """

    MESH_TYPE = "PlateWithDefect"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        W, H, R = float(params.W), float(params.H), float(params.R_defect)
        dx = float(params.dx)
        dx_metal = float(getattr(params, "dx_metal", dx))
        dx_defect = float(getattr(params, "dx_defect", dx_metal))
        dx_pit = float(getattr(params, "dx_pit", dx_metal))
        grading = getattr(params, "grading", None)
        lateral_grading = getattr(params, "lateral_grading", None)
        if params.dim not in (2, 3):
            raise ValueError(f"PlateWithDefect supports dim = 2 or 3, not {params.dim}.")
        if W <= 0.0 or H <= 0.0 or R <= 0.0:
            raise ValueError("W, H and R_defect must be positive.")
        if R >= W:
            raise ValueError("R_defect must be smaller than W.")
        if dx <= 0.0 or dx_metal <= 0.0 or dx_defect <= 0.0 or dx_pit <= 0.0:
            raise ValueError("dx, dx_metal, dx_defect and dx_pit must be positive.")
        if grading is not None and not float(grading) > 0.0:
            raise ValueError(f"grading must be positive, not {grading}.")
        if lateral_grading is not None:
            if not float(lateral_grading) > 0.0:
                raise ValueError(f"lateral_grading must be positive, not {lateral_grading}.")
            if not float(getattr(params, "dx_far", 0.0)) > 0.0:
                raise ValueError("lateral_grading requires a positive dx_far.")
            if not float(getattr(params, "fine_extent", 0.0)) >= 0.0:
                raise ValueError("fine_extent must not be negative.")
            if not float(getattr(params, "feature_extent", 0.0)) >= 0.0:
                raise ValueError("feature_extent must not be negative.")

        spots = self._validated_spots(getattr(params, "weak_spots", None), params.dim, R, W)
        depth = float(getattr(params, "defect_depth", 0.0))
        if not depth >= 0.0:
            raise ValueError(f"defect_depth must not be negative, not {depth}.")
        if depth > 0.0 and params.dim != 2:
            raise ValueError("defect_depth is supported in 2-D only.")
        voids = self._validated_voids(getattr(params, "voids", None), params.dim, R, W, spots)
        dx_void = float(getattr(params, "dx_void", dx_defect))
        if not dx_void > 0.0:
            raise ValueError(f"dx_void must be positive, not {dx_void}.")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate_with_defect")
        if params.dim == 2:
            boundary_groups, defect_edge, feature_edges = self._build_2d(W, H, R, spots, depth, voids)
        else:
            boundary_groups, defect_edge = self._build_3d(W, H, R)
            feature_edges = []

        boundary_dim = params.dim - 1
        boundary_markers = {name: marker for marker, name in enumerate(boundary_groups, start=1)}
        for name, marker in boundary_markers.items():
            gmsh.model.addPhysicalGroup(boundary_dim, boundary_groups[name], marker)
            gmsh.model.setPhysicalName(boundary_dim, marker, name)

        metal = [tag for name, tags in boundary_groups.items() if name.startswith("metal:") for tag in tags]
        if grading is None:
            size_field = self._set_mesh_size_fields(params.dim, metal, defect_edge + feature_edges, dx, dx_metal, dx_defect, R,
                                                    boundary_groups["metal:defect"], dx_pit)
        else:
            size_field = self._set_graded_size_fields(params.dim, metal, defect_edge + feature_edges, boundary_groups["metal:defect"], dx,
                                                      dx_metal, dx_defect, dx_pit, float(grading))
        if lateral_grading is not None:
            fine_extent = float(getattr(params, "fine_extent", 0.0))
            size_field = self._coarsen_laterally(params.dim, defect_edge, size_field, dx_metal, float(lateral_grading),
                                                 float(params.dx_far), fine_extent, feature_edges,
                                                 float(getattr(params, "feature_extent", fine_extent)))
        if voids:
            size_field = self._refine_voids(size_field, [tag for name, tags in boundary_groups.items()
                                                         if name.startswith("metal:void") for tag in tags],
                                            dx, dx_void, min(radius for _c, radius, _m in voids))
        gmsh.model.mesh.field.setAsBackgroundMesh(size_field)
        self._sizes_from_background_only()
        gmsh.model.mesh.generate(params.dim)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=params.dim),
            domain_markers={"domain": 1},
            boundary_markers=boundary_markers,
        )
        gmsh.finalize()
        return generated

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validated_spots(spots, dim, R, W) -> list:
        """PRIVATE: The weak spots as sorted (centre, half_width) pairs; reject invalid ones."""
        if spots is None:
            return []
        spots = [(float(centre), float(half)) for centre, half in spots]
        if spots and dim != 2:
            raise ValueError("weak_spots are supported in 2-D only.")
        spots.sort()
        edge = R
        for centre, half in spots:
            if not half > 0.0:
                raise ValueError(f"weak_spots half-widths must be positive, not {half}.")
            if not (centre - half > edge and centre + half < W):
                raise ValueError(f"The weak spot at {centre} (half-width {half}) must lie within (R_defect, W) "
                                 "without overlapping another spot.")
            edge = centre + half
        return spots

    @staticmethod
    def _validated_voids(voids, dim, R, W, spots) -> list:
        """PRIVATE: The voids as sorted (centre, radius, mouth) triples; reject invalid ones."""
        if voids is None:
            return []
        voids = sorted((float(c), float(radius), float(mouth)) for c, radius, mouth in voids)
        if voids and dim != 2:
            raise ValueError("voids are supported in 2-D only.")
        for centre, radius, mouth in voids:
            if not 0.0 < mouth < radius:
                raise ValueError(f"Each void needs 0 < mouth < radius, not mouth = {mouth}, radius = {radius}.")
        # footprints along the surface: the void circles and the spot segments
        footprints = sorted([(c - radius, c + radius) for c, radius, _m in voids] + [(c - h, c + h) for c, h in spots])
        edge = R
        for low, high in footprints:
            if not (low > edge and high < W):
                raise ValueError(f"The void or weak spot over [{low}, {high}] must lie within (R_defect, W) "
                                 "without overlapping another void or spot.")
            edge = high
        return voids

    def _build_2d(self, W, H, R, spots=(), depth=0.0, voids=()):
        """PRIVATE: Half domain [0, W] x [0, H] with the metal edge split at x = R, at the edges of the weak spots and
        at the mouths of the voids; with ``depth``, the defect is a recess [0, R] x [-depth, 0] (bottom and wall both
        ``metal:defect``). Returns the boundary groups, the refined corners of the defect, and those of the spots and voids."""
        occ = gmsh.model.occ
        groups = {"top": [], "far": [], "symmetry": [], "metal:defect": [], "metal:outer": []}
        corners = []
        start = occ.addPoint(0.0, -depth, 0.0)
        path, loop = [start], []                       # boundary points and curves, in order

        def add_curve(group, curve):
            groups[group].append(curve)
            loop.append(curve)

        def go_to(x, y, group):
            point = occ.addPoint(x, y, 0.0)
            add_curve(group, occ.addLine(path[-1], point))
            path.append(point)
            return point

        if depth > 0.0:
            corners.append(go_to(R, -depth, "metal:defect"))
        corners.append(go_to(R, 0.0, "metal:defect"))
        defect_corners, corners = corners, []
        # features along the surface, outwards: weak spots (segments) and voids (circles below a mouth)
        features = sorted([(c - h, c + h, f"metal:spot{i + 1}", None) for i, (c, h) in enumerate(spots)]
                          + [(c - m, c + m, f"metal:void{i + 1}", (c, radius, m)) for i, (c, radius, m) in enumerate(voids)])
        for low, high, group, void in features:
            groups[group] = []
            corners.append(go_to(low, 0.0, "metal:outer"))
            if void is None:
                corners.append(go_to(high, 0.0, group))
                continue
            c, radius, m = void
            depth_centre = np.sqrt(radius**2 - m**2)
            centre = occ.addPoint(c, -depth_centre, 0.0)
            bottom = occ.addPoint(c, -depth_centre - radius, 0.0)
            mouth = occ.addPoint(high, 0.0, 0.0)
            # two arcs (each less than pi) from the mouth down around the circle and back up
            add_curve(group, occ.addCircleArc(path[-1], centre, bottom))
            add_curve(group, occ.addCircleArc(bottom, centre, mouth))
            occ.remove([(0, centre)])
            path.append(mouth)
            corners.append(mouth)
        go_to(W, 0.0, "metal:outer")
        go_to(W, H, "far")
        go_to(0.0, H, "top")
        add_curve("symmetry", occ.addLine(path[-1], start))
        surface = occ.addPlaneSurface([occ.addCurveLoop(loop)])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], 1)
        gmsh.model.setPhysicalName(2, 1, "domain")
        # the spot and void groups numbered outwards, after the fixed ones
        ordered = {name: groups[name] for name in ("top", "far", "symmetry", "metal:defect", "metal:outer")}
        for kind in ("metal:spot", "metal:void"):
            ordered.update({name: tags for name, tags in groups.items() if name.startswith(kind)})
        return ordered, defect_corners, corners

    def _build_3d(self, W, H, R):
        """PRIVATE: Quarter domain [0, W]^2 x [0, H] with a quarter-disk defect on the metal face."""
        occ = gmsh.model.occ
        box = occ.addBox(0.0, 0.0, 0.0, W, W, H)
        disk = occ.addDisk(0.0, 0.0, 0.0, R, R)
        square = occ.addRectangle(0.0, 0.0, 0.0, W, W)
        quarter, _map = occ.intersect([(2, disk)], [(2, square)])
        occ.fragment([(3, box)], quarter)
        occ.synchronize()
        volumes = [tag for _dim, tag in gmsh.model.getEntities(3)]
        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "domain")

        tol = 1e-9 * W
        groups = {"top": [], "far": [], "symmetry": [], "metal:defect": [], "metal:outer": []}
        for _dim, tag in gmsh.model.getEntities(2):
            x0, y0, z0, x1, y1, z1 = self._exact_bounding_box(2, tag)
            if np.isclose(z0, 0.0, atol=tol) and np.isclose(z1, 0.0, atol=tol):
                groups["metal:defect" if x1 <= R + tol and y1 <= R + tol else "metal:outer"].append(tag)
            elif np.isclose(z0, H, atol=tol) and np.isclose(z1, H, atol=tol):
                groups["top"].append(tag)
            elif (np.isclose(x0, 0.0, atol=tol) and np.isclose(x1, 0.0, atol=tol)) or \
                    (np.isclose(y0, 0.0, atol=tol) and np.isclose(y1, 0.0, atol=tol)):
                groups["symmetry"].append(tag)
            elif (np.isclose(x0, W, atol=tol) and np.isclose(x1, W, atol=tol)) or \
                    (np.isclose(y0, W, atol=tol) and np.isclose(y1, W, atol=tol)):
                groups["far"].append(tag)
            else:
                raise RuntimeError(f"PlateWithDefect could not classify surface {tag}.")  # pragma: no cover
        # The defect edge: the arc bounding the defect face(s), away from the symmetry planes.
        edge = []
        for _dim, tag in gmsh.model.getBoundary([(2, t) for t in groups["metal:defect"]], oriented=False):
            x0, y0, _z0, x1, y1, _z1 = self._exact_bounding_box(1, tag)
            if x1 > tol and y1 > tol:
                edge.append(tag)
        return groups, edge

    @staticmethod
    def _set_mesh_size_fields(dim, metal, defect_edge, dx, dx_metal, dx_defect, R, defect=(), dx_pit=None) -> int:
        """PRIVATE: Grade from ``dx_metal`` on the metal, ``dx_pit`` on the defect and ``dx_defect`` at its edge to ``dx``."""
        fields = gmsh.model.mesh.field
        fields.add("Distance", 1)
        fields.setNumbers(1, "CurvesList" if dim == 2 else "SurfacesList", list(metal))
        fields.setNumber(1, "Sampling", 200)
        fields.add("Threshold", 2)
        fields.setNumber(2, "InField", 1)
        fields.setNumber(2, "SizeMin", dx_metal)
        fields.setNumber(2, "SizeMax", dx)
        fields.setNumber(2, "DistMin", 0.0)
        fields.setNumber(2, "DistMax", 2.0 * dx)

        fields.add("Distance", 3)
        fields.setNumbers(3, "PointsList" if dim == 2 else "CurvesList", list(defect_edge))
        fields.setNumber(3, "Sampling", 200)
        fields.add("Threshold", 4)
        fields.setNumber(4, "InField", 3)
        fields.setNumber(4, "SizeMin", dx_defect)
        fields.setNumber(4, "SizeMax", dx)
        fields.setNumber(4, "DistMin", 0.0)
        fields.setNumber(4, "DistMax", R)
        size_fields = [2, 4]

        if dx_pit is not None and dx_pit < dx_metal:
            fields.add("Distance", 6)
            fields.setNumbers(6, "CurvesList" if dim == 2 else "SurfacesList", list(defect))
            fields.setNumber(6, "Sampling", 200)
            fields.add("Threshold", 7)
            fields.setNumber(7, "InField", 6)
            fields.setNumber(7, "SizeMin", dx_pit)
            fields.setNumber(7, "SizeMax", dx)
            fields.setNumber(7, "DistMin", 0.0)
            fields.setNumber(7, "DistMax", R)
            size_fields.append(7)

        fields.add("Min", 5)
        fields.setNumbers(5, "FieldsList", size_fields)
        return 5

    @staticmethod
    def _set_graded_size_fields(dim, metal, defect_edge, defect, dx, dx_metal, dx_defect, dx_pit, grading) -> int:
        """PRIVATE: Sizes growing linearly (rate ``grading``) with the distance from the metal, the defect and its edge, capped at ``dx``."""
        fields = gmsh.model.mesh.field
        size_fields = []
        features = [("CurvesList" if dim == 2 else "SurfacesList", list(metal), dx_metal),
                    ("CurvesList" if dim == 2 else "SurfacesList", list(defect), dx_pit),
                    ("PointsList" if dim == 2 else "CurvesList", list(defect_edge), dx_defect)]
        for i, (kind, entities, size) in enumerate(features):
            distance, graded = 10 + 2 * i, 11 + 2 * i
            fields.add("Distance", distance)
            fields.setNumbers(distance, kind, entities)
            fields.setNumber(distance, "Sampling", 400)
            fields.add("MathEval", graded)
            fields.setString(graded, "F", f"min({dx}, {size} + {grading} * F{distance})")
            size_fields.append(graded)
        fields.add("Min", 20)
        fields.setNumbers(20, "FieldsList", size_fields)
        return 20

    @staticmethod
    def _coarsen_laterally(dim, defect_edge, size_field, dx_metal, grading, dx_far, fine_extent, features=(),
                           feature_extent=0.0) -> int:
        """PRIVATE: Field ``min(dx_far, max(size, dx_metal + grading * max(0, d - fine_extent)))``, ``d`` the distance from the
        defect edge; near the corners of spots and voids (``features``) the same with their distance and ``feature_extent``."""
        fields = gmsh.model.mesh.field
        fields.add("Distance", 30)
        fields.setNumbers(30, "PointsList" if dim == 2 else "CurvesList", list(defect_edge))
        fields.setNumber(30, "Sampling", 400)
        # Max/Min fields rather than one MathEval referencing ``size_field``: gmsh hangs on nested MathEval fields
        fields.add("MathEval", 35)
        fields.setString(35, "F", f"{dx_metal} + {grading} * max(0, F30 - {fine_extent})")
        allowed = 35
        if features:
            fields.add("Distance", 36)
            fields.setNumbers(36, "PointsList", list(features))
            fields.setNumber(36, "Sampling", 400)
            fields.add("MathEval", 37)
            fields.setString(37, "F", f"{dx_metal} + {grading} * max(0, F36 - {feature_extent})")
            fields.add("Min", 38)
            fields.setNumbers(38, "FieldsList", [35, 37])
            allowed = 38
        fields.add("Max", 32)
        fields.setNumbers(32, "FieldsList", [size_field, allowed])
        fields.add("MathEval", 33)
        fields.setString(33, "F", f"{dx_far}")
        fields.add("Min", 34)
        fields.setNumbers(34, "FieldsList", [32, 33])
        return 34

    @staticmethod
    def _refine_voids(size_field, void_curves, dx, dx_void, radius) -> int:
        """PRIVATE: Field ``min(size, dx_void + rate * d)``, ``d`` the distance from the void walls: ``dx_void`` grows
        to ``dx`` over one void radius and keeps growing, so it never limits the sizes elsewhere (e.g. coarsened)."""
        fields = gmsh.model.mesh.field
        rate = max((dx - dx_void) / radius, 0.1)
        fields.add("Distance", 40)
        fields.setNumbers(40, "CurvesList", list(void_curves))
        fields.setNumber(40, "Sampling", 400)
        fields.add("MathEval", 41)
        fields.setString(41, "F", f"{dx_void} + {rate} * F40")
        fields.add("Min", 42)
        fields.setNumbers(42, "FieldsList", [size_field, 41])
        return 42

    @staticmethod
    def _sizes_from_background_only() -> None:
        """PRIVATE: Take element sizes exclusively from the background field."""
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
