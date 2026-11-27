from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import mprint, comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D damage-challenge configuration."""

    class DamageChallengeSampleParameters(ParamsTemplate):
        MeshType = "DamageChallenge"
        dim = 2
        Lx = 4.0
        Ly = 2.0
        dx = 0.2
        Lx_BC = 1.0
        Lx_notch = 2.0
        Notch_Thickness = 0.05
        Ly_notch = 1.0
        Load_Width = 0.2
        BC_Width = 0.2

    return DamageChallengeSampleParameters()


class DamageChallengeMeshGenerator(MeshGeneratorBase):
    """Generate a damage-challenge benchmark specimen (Gmsh).

    Typically a plate with pre-existing notch(es) and specific loading
    zones.  Expects ``params`` to define ``Lx``, ``Ly``, ``Lx_BC``,
    ``Lx_notch``, ``Notch_Thickness``, ``Ly_notch``, ``dx``, and ``dim``.
    The optional ``Load_Width`` and ``BC_Width`` (default ``dx``) give the
    exact widths of the ``top_load`` and ``bottom_left`` / ``bottom_right``
    boundary windows. In 3-D (thickness ``Lz``), ``Lx_notch`` may be a
    pair ``[x0, x1]`` for a slanted notch, and the faces follow the box
    names: ``y = 0`` is ``left``, ``y = Ly`` is ``right``, and ``bottom`` /
    ``top`` are ``z = 0`` / ``z = Lz``.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "DamageChallenge"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim == 2:
            return self._generate_2d(params)
        elif params.dim == 3:
            return self._generate_3d(params)
        else:
            raise NotImplementedError("Only 2D and 3D DamageChallenge mesh generation is implemented.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 2-D damage-challenge mesh variant."""
        LX = params.Lx
        LY = params.Ly
        LNotch = params.Lx_notch
        notch_thickness = params.Notch_Thickness
        notch_height = params.Ly_notch
        Lx_BC = params.Lx_BC
        dx = params.dx

        load_width = getattr(params, "Load_Width", dx)
        bc_width = getattr(params, "BC_Width", dx)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("damage_challenge")
        notch_x0 = LNotch - 0.5 * notch_thickness
        notch_x1 = LNotch + 0.5 * notch_thickness

        self._require_positive_window_widths(load_width, bc_width)
        # The window lines are exactly as wide as requested.
        load_half = 0.5 * load_width
        bc_half = 0.5 * bc_width

        point_coords = {}
        point_cache = {}

        def _add_point(x, y):
            key = (round(x, 12), round(y, 12))
            existing = point_cache.get(key)
            if existing is not None:
                return existing
            tag = gmsh.model.geo.addPoint(x, y, 0.0, dx)
            point_coords[tag] = (x, y)
            point_cache[key] = tag
            return tag

        def _sorted_unique(vals):
            vals = sorted(set(vals))
            return vals

        bottom_left_seg = (Lx_BC - bc_half, Lx_BC + bc_half)
        bottom_right_seg = (LX - Lx_BC - bc_half, LX - Lx_BC + bc_half)
        top_load_seg = (0.5 * LX - load_half, 0.5 * LX + load_half)

        def _segment_in_range(seg, x0, x1):
            return seg[1] > x0 and seg[0] < x1

        if not _segment_in_range(bottom_left_seg, 0.0, notch_x0) and not _segment_in_range(
            bottom_left_seg, notch_x1, LX
        ):
            mprint("Warning: bottom_left segment overlaps notch; it may be skipped.")
        if not _segment_in_range(bottom_right_seg, 0.0, notch_x0) and not _segment_in_range(
            bottom_right_seg, notch_x1, LX
        ):
            mprint("Warning: bottom_right segment overlaps notch; it may be skipped.")

        bottom_left_x = []
        if notch_x0 > 0.0:
            bottom_left_x = [0.0, notch_x0]
            if _segment_in_range(bottom_left_seg, 0.0, notch_x0):
                bottom_left_x.extend(
                    [max(0.0, bottom_left_seg[0]), min(notch_x0, bottom_left_seg[1])]
                )
        bottom_right_x = []
        if notch_x1 < LX:
            bottom_right_x = [notch_x1, LX]
            if _segment_in_range(bottom_right_seg, notch_x1, LX):
                bottom_right_x.extend(
                    [max(notch_x1, bottom_right_seg[0]), min(LX, bottom_right_seg[1])]
                )

        top_x = [0.0, LX, top_load_seg[0], top_load_seg[1]]
        top_x = [x for x in top_x if 0.0 <= x <= LX]

        bottom_left_x = _sorted_unique([x for x in bottom_left_x if 0.0 <= x <= notch_x0])
        bottom_right_x = _sorted_unique([x for x in bottom_right_x if notch_x1 <= x <= LX])
        top_x = _sorted_unique(top_x)

        bottom_lines = []
        top_lines = []
        left_lines = []
        right_lines = []
        notch_lines = []
        bottom_left_line = None
        bottom_right_line = None
        top_load_line = None

        def _add_line(p1, p2, store, tag_name=None):
            line = gmsh.model.geo.addLine(p1, p2)
            store.append(line)
            return line

        # Bottom left segment
        prev = _add_point(0.0, 0.0)
        for x in bottom_left_x[1:]:
            p = _add_point(x, 0.0)
            line = _add_line(prev, p, bottom_lines)
            mid_x = 0.5 * (point_coords[prev][0] + point_coords[p][0])
            if abs(mid_x - Lx_BC) <= bc_half + 1.0e-12 and bottom_left_line is None:
                bottom_left_line = line
            prev = p

        # Notch
        p_notch_bl = prev
        p_notch_up = _add_point(notch_x0, notch_height)
        notch_lines.append(gmsh.model.geo.addLine(p_notch_bl, p_notch_up))
        p_notch_top = _add_point(notch_x1, notch_height)
        notch_lines.append(gmsh.model.geo.addLine(p_notch_up, p_notch_top))
        p_notch_br = _add_point(notch_x1, 0.0)
        notch_lines.append(gmsh.model.geo.addLine(p_notch_top, p_notch_br))

        # Bottom right segment
        prev = p_notch_br
        for x in bottom_right_x[1:]:
            p = _add_point(x, 0.0)
            line = _add_line(prev, p, bottom_lines)
            mid_x = 0.5 * (point_coords[prev][0] + point_coords[p][0])
            if abs(mid_x - (LX - Lx_BC)) <= bc_half + 1.0e-12 and bottom_right_line is None:
                bottom_right_line = line
            prev = p

        # Right edge
        p_right_top = _add_point(LX, LY)
        right_lines.append(gmsh.model.geo.addLine(prev, p_right_top))

        # Top segment (right to left)
        prev = p_right_top
        for x in sorted(top_x, reverse=True)[1:]:
            p = _add_point(x, LY)
            line = _add_line(prev, p, top_lines)
            mid_x = 0.5 * (point_coords[prev][0] + point_coords[p][0])
            if abs(mid_x - 0.5 * LX) <= load_half + 1.0e-12 and top_load_line is None:
                top_load_line = line
            prev = p

        # Left edge
        p_left_bottom = _add_point(0.0, 0.0)
        left_lines.append(gmsh.model.geo.addLine(prev, p_left_bottom))

        cl = gmsh.model.geo.addCurveLoop(
            bottom_lines + notch_lines + right_lines + top_lines + left_lines
        )
        surface = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        if bottom_left_line is not None:
            gmsh.model.addPhysicalGroup(1, [bottom_left_line], 7)
            gmsh.model.setPhysicalName(1, 7, "Bottom_Left")
        if bottom_right_line is not None:
            gmsh.model.addPhysicalGroup(1, [bottom_right_line], 8)
            gmsh.model.setPhysicalName(1, 8, "Bottom_Right")
        if top_load_line is not None:
            gmsh.model.addPhysicalGroup(1, [top_load_line], 6)
            gmsh.model.setPhysicalName(1, 6, "Top_Load")

        # Each facet carries one tag: the windows are not part of top/bottom.
        windows = {bottom_left_line, bottom_right_line, top_load_line}
        bottom_lines = [line for line in bottom_lines if line not in windows]
        top_lines = [line for line in top_lines if line not in windows]

        if bottom_lines:
            gmsh.model.addPhysicalGroup(1, bottom_lines, 1)
            gmsh.model.setPhysicalName(1, 1, "Bottom")
        if right_lines:
            gmsh.model.addPhysicalGroup(1, right_lines, 2)
            gmsh.model.setPhysicalName(1, 2, "Right")
        if top_lines:
            gmsh.model.addPhysicalGroup(1, top_lines, 3)
            gmsh.model.setPhysicalName(1, 3, "Top")
        if left_lines:
            gmsh.model.addPhysicalGroup(1, left_lines, 4)
            gmsh.model.setPhysicalName(1, 4, "Left")
        if notch_lines:
            gmsh.model.addPhysicalGroup(1, notch_lines, 5)
            gmsh.model.setPhysicalName(1, 5, "Notch")

        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers={
                "bottom": 1,
                "right": 2,
                "top": 3,
                "left": 4,
                "notch": 5,
                "top_load": 6,
                "bottom_left": 7,
                "bottom_right": 8,
            },
        )

        gmsh.finalize()

        return generated

    def _generate_3d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 3-D damage-challenge mesh variant."""
        LX = params.Lx
        LY = params.Ly
        LZ = params.Lz
        LNotch = params.Lx_notch
        notch_thickness = params.Notch_Thickness
        notch_height = params.Ly_notch
        Lx_BC = params.Lx_BC
        dx = params.dx

        load_width = getattr(params, "Load_Width", dx)
        bc_width = getattr(params, "BC_Width", dx)

        if isinstance(LNotch, (list, tuple, np.ndarray)):
            if len(LNotch) != 2:
                raise ValueError("Lx_notch must have 2 components for 3D diagonal notches.")
            notch_x0 = float(LNotch[0])
            notch_x1 = float(LNotch[1])
        else:
            notch_x0 = float(LNotch)
            notch_x1 = float(LNotch)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("damage_challenge_3d")

        plate = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)

        notch = gmsh.model.occ.addBox(
            -0.5 * notch_thickness, 0, -0.5 * LZ,
            notch_thickness, notch_height, 2.0 * LZ,
        )
        if abs(notch_x1 - notch_x0) > 0.0:
            theta = np.arctan2((notch_x1 - notch_x0), LZ)
            gmsh.model.occ.rotate([(3, notch)], 0, 0, 0, 0, 1, 0, theta)
        gmsh.model.occ.translate([(3, notch)], notch_x0, 0, 0)

        cut = gmsh.model.occ.cut([(3, plate)], [(3, notch)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        volumes = [tag for dim, tag in cut[0] if dim == 3]
        self._require_positive_window_widths(load_width, bc_width)
        volumes = self._split_faces_at_windows(volumes, [
            (Lx_BC, bc_width, 0.0),
            (LX - Lx_BC, bc_width, 0.0),
            (0.5 * LX, load_width, LY),
        ], LZ)
        if not volumes:
            raise RuntimeError("DamageChallenge 3D mesh generation failed to create a volume.")  # pragma: no cover

        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "Domain")

        boundary_groups = {
            "bottom": [], "right": [], "top": [], "left": [],
            "notch": [], "top_load": [], "bottom_left": [],
            "bottom_right": [], "front": [], "back": []
        }
        max_dim = max(LX, LY, LZ, notch_thickness, notch_height, load_width, bc_width)
        tol = max(1e-6 * max_dim, 1e-9)

        for vol in volumes:
            for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False):
                xmin, ymin, zmin, xmax, ymax, zmax = self._exact_bounding_box(2, tag)
                x_planar = np.isclose(xmin, xmax, atol=tol)
                y_planar = np.isclose(ymin, ymax, atol=tol)
                z_planar = np.isclose(zmin, zmax, atol=tol)

                def in_window(x_centre, width):
                    return xmin >= x_centre - 0.5 * width - tol and xmax <= x_centre + 0.5 * width + tol

                if y_planar and np.isclose(ymin, 0.0, atol=tol):
                    if in_window(Lx_BC, bc_width):
                        boundary_groups["bottom_left"].append(tag)
                    elif in_window(LX - Lx_BC, bc_width):
                        boundary_groups["bottom_right"].append(tag)
                    else:
                        boundary_groups["left"].append(tag)
                elif y_planar and np.isclose(ymin, LY, atol=tol):
                    if in_window(0.5 * LX, load_width):
                        boundary_groups["top_load"].append(tag)
                    else:
                        boundary_groups["right"].append(tag)
                elif x_planar and np.isclose(xmin, 0.0, atol=tol):
                    boundary_groups["front"].append(tag)
                elif x_planar and np.isclose(xmin, LX, atol=tol):
                    boundary_groups["back"].append(tag)
                elif z_planar:
                    if np.isclose(zmin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif np.isclose(zmin, LZ, atol=tol):
                        boundary_groups["top"].append(tag)
                else:
                    boundary_groups["notch"].append(tag)

        empty_windows = [name for name in ("top_load", "bottom_left", "bottom_right") if not boundary_groups[name]]
        if empty_windows:
            gmsh.finalize()
            raise ValueError(
                f"DamageChallenge windows {empty_windows} lie entirely within the notch; "
                "widen Load_Width / BC_Width or move Lx_BC."
            )

        surface_markers = {
            "bottom": (1, "Bottom"), "right": (2, "Right"),
            "top": (3, "Top"), "left": (4, "Left"),
            "notch": (5, "Notch"), "top_load": (6, "Top_Load"),
            "bottom_left": (7, "Bottom_Left"), "bottom_right": (8, "Bottom_Right"),
            "front": (9, "Front"), "back": (10, "Back")
        }
        for name, (marker, physical_name) in surface_markers.items():
            surfaces = boundary_groups[name]
            if surfaces:
                gmsh.model.addPhysicalGroup(2, surfaces, marker)
                gmsh.model.setPhysicalName(2, marker, physical_name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
        gmsh.model.mesh.generate(3)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=3),
            domain_markers={"domain": 1},
            boundary_markers={
                "bottom": 1,
                "right": 2,
                "top": 3,
                "left": 4,
                "notch": 5,
                "top_load": 6,
                "bottom_left": 7,
                "bottom_right": 8,
                "front": 9,
                "back": 10,
            },
        )

        gmsh.finalize()

        return generated

    @staticmethod
    def _require_positive_window_widths(load_width, bc_width) -> None:
        """PRIVATE: Reject non-positive load and support window widths."""
        if load_width <= 0.0 or bc_width <= 0.0:
            raise ValueError("Load_Width and BC_Width must be positive.")

    @staticmethod
    def _split_faces_at_windows(volumes, windows, LZ):
        """PRIVATE: Split the y = const faces of the specimen at the load/support windows.

        ``windows`` holds ``(x_centre, width, y)`` per window; each becomes a
        separate boundary surface spanning the thickness ``LZ``, so it can
        carry its own physical group. Returns the volumes after splitting.
        """
        rectangles = []
        for x_centre, width, y in windows:
            rectangle = gmsh.model.occ.addRectangle(x_centre - 0.5 * width, 0.0, 0.0, width, LZ)
            # From the xy-plane to the xz-plane (z = old y), then to height y.
            gmsh.model.occ.rotate([(2, rectangle)], 0, 0, 0, 1, 0, 0, 0.5 * np.pi)
            gmsh.model.occ.translate([(2, rectangle)], 0, y, 0)
            rectangles.append((2, rectangle))
        fragments, _ = gmsh.model.occ.fragment([(3, volume) for volume in volumes], rectangles)
        gmsh.model.occ.synchronize()

        volumes = [tag for dim, tag in fragments if dim == 3]
        # Window parts outside the specimen (e.g. over the notch) bound nothing.
        boundary = {tag for _, tag in gmsh.model.getBoundary([(3, volume) for volume in volumes], oriented=False)}
        dangling = [(2, tag) for _, tag in gmsh.model.getEntities(2) if tag not in boundary]
        if dangling:
            gmsh.model.occ.remove(dangling, recursive=True)
            gmsh.model.occ.synchronize()
        return volumes
