from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D rectangle-with-top-notches configuration."""

    class RectangleNotchesSampleParameters(ParamsTemplate):
        MeshType = "Rectangle_Notches"
        dim = 2
        Lx = 4.0
        Ly = 2.0
        dx = 0.2
        nNotches = 3
        Notch_Width = 0.25
        Notch_Depth = 0.5

    return RectangleNotchesSampleParameters()


class RectangleNotchesMeshGenerator(MeshGeneratorBase):
    """Generate a rectangle or box with rectangular top-surface notches.

    Expects ``params`` to define ``Lx``, ``Ly`` (and ``Lz`` in 3-D),
    ``dx``, and ``dim``. When notches are requested, the generator also
    uses ``nNotches`` (default 0) plus ``Notch_Width`` (default ``2 dx``)
    and ``Notch_Depth`` (default ``5 dx``). The notch centres may
    optionally be provided explicitly through ``Notch_Positions``, which
    then replaces ``nNotches``; otherwise they are distributed evenly
    across the specimen width. In 2-D the notches are cut down from
    ``y = Ly``; in 3-D from ``z = Lz``, through the full ``y`` extent unless
    ``Notch_Y_Ranges`` limits them: either one ``(y_min, y_max)`` pair for
    all notches or one pair per notch, in the order of the notch centres.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "Rectangle_Notches"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim == 2:
            return self._generate_2d(params)
        elif params.dim == 3:
            return self._generate_3d(params)
        else:
            raise NotImplementedError(
                "Only 2D and 3D rectangle-with-notches mesh generation is implemented."
            )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 2-D rectangle-with-notches mesh variant."""
        LX = params.Lx
        LY = params.Ly
        dx = params.dx
        notch_intervals, notch_depth = self._resolve_notch_intervals(params)
        has_notches = bool(notch_intervals)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rectangle_notches")

        plate = gmsh.model.occ.addRectangle(0, 0, 0, LX, LY)

        if has_notches:
            notch_tools = [
                (
                    2,
                    gmsh.model.occ.addRectangle(
                        x0,
                        LY - notch_depth,
                        0,
                        x1 - x0,
                        notch_depth,
                    ),
                )
                for x0, x1 in notch_intervals
            ]
            cut = gmsh.model.occ.cut(
                [(2, plate)],
                notch_tools,
                removeObject=True,
                removeTool=True,
            )
            surfaces = [tag for dim, tag in cut[0] if dim == 2]
        else:
            surfaces = [plate]

        gmsh.model.occ.synchronize()

        if not surfaces:
            raise RuntimeError(
                "Rectangle_Notches mesh generation failed to create a surface."
            )  # pragma: no cover

        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        boundary_groups = {
            "bottom": [],
            "right": [],
            "top": [],
            "left": [],
        }
        if has_notches:
            boundary_groups["notch"] = []

        max_dim = max(LX, LY)
        tol = max(1e-6 * max_dim, 1e-9)

        for surf in surfaces:
            for _, tag in gmsh.model.getBoundary(
                [(2, surf)],
                oriented=False,
                recursive=False,
            ):
                xmin, ymin, _zmin, xmax, ymax, _zmax = self._exact_bounding_box(1, tag)
                x_planar = np.isclose(xmin, xmax, atol=tol)
                y_planar = np.isclose(ymin, ymax, atol=tol)
                if y_planar and np.isclose(ymin, 0.0, atol=tol):
                    boundary_groups["bottom"].append(tag)
                elif y_planar and np.isclose(ymin, LY, atol=tol):
                    boundary_groups["top"].append(tag)
                elif x_planar and np.isclose(xmin, 0.0, atol=tol):
                    boundary_groups["left"].append(tag)
                elif x_planar and np.isclose(xmin, LX, atol=tol):
                    boundary_groups["right"].append(tag)
                elif has_notches:
                    boundary_groups["notch"].append(tag)

        line_markers = {
            "bottom": (1, "Bottom"),
            "right": (2, "Right"),
            "top": (3, "Top"),
            "left": (4, "Left"),
        }
        if has_notches:
            line_markers["notch"] = (5, "Notch")

        for name, (marker, physical_name) in line_markers.items():
            segments = boundary_groups[name]
            if segments:
                gmsh.model.addPhysicalGroup(1, segments, marker)
                gmsh.model.setPhysicalName(1, marker, physical_name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
        gmsh.model.mesh.generate(2)

        boundary_markers = {
            "bottom": 1,
            "right": 2,
            "top": 3,
            "left": 4,
        }
        if has_notches:
            boundary_markers["notch"] = 5

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers=boundary_markers,
        )

        gmsh.finalize()

        return generated

    def _generate_3d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 3-D rectangle-with-notches mesh variant."""
        LX = params.Lx
        LY = params.Ly
        LZ = params.Lz
        dx = params.dx
        notch_intervals, notch_depth = self._resolve_notch_intervals(params)
        has_notches = bool(notch_intervals)
        y_ranges = self._resolve_notch_y_ranges(params, len(notch_intervals))

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rectangle_notches_3d")

        plate = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)

        if has_notches:
            notch_tools = [
                (
                    3,
                    gmsh.model.occ.addBox(
                        x0,
                        y0,
                        LZ - notch_depth,
                        x1 - x0,
                        y1 - y0,
                        notch_depth,
                    ),
                )
                for (x0, x1), (y0, y1) in zip(notch_intervals, y_ranges)
            ]
            cut = gmsh.model.occ.cut(
                [(3, plate)],
                notch_tools,
                removeObject=True,
                removeTool=True,
            )
            volumes = [tag for dim, tag in cut[0] if dim == 3]
        else:
            volumes = [plate]

        gmsh.model.occ.synchronize()

        if not volumes:
            raise RuntimeError(
                "Rectangle_Notches 3D mesh generation failed to create a volume."
            )  # pragma: no cover

        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "Domain")

        boundary_groups = {
            "bottom": [],
            "top": [],
            "front": [],
            "back": [],
            "left": [],
            "right": [],
        }
        if has_notches:
            boundary_groups["notch"] = []

        max_dim = max(LX, LY, LZ)
        tol = max(1e-6 * max_dim, 1e-9)

        for vol in volumes:
            for _, tag in gmsh.model.getBoundary(
                [(3, vol)],
                oriented=False,
                recursive=False,
            ):
                xmin, ymin, zmin, xmax, ymax, zmax = self._exact_bounding_box(2, tag)
                x_planar = np.isclose(xmin, xmax, atol=tol)
                y_planar = np.isclose(ymin, ymax, atol=tol)
                z_planar = np.isclose(zmin, zmax, atol=tol)

                if x_planar:
                    if np.isclose(xmin, 0.0, atol=tol):
                        boundary_groups["front"].append(tag)
                    elif np.isclose(xmin, LX, atol=tol):
                        boundary_groups["back"].append(tag)
                    elif has_notches:
                        boundary_groups["notch"].append(tag)
                elif y_planar:
                    if np.isclose(ymin, 0.0, atol=tol):
                        boundary_groups["left"].append(tag)
                    elif np.isclose(ymin, LY, atol=tol):
                        boundary_groups["right"].append(tag)
                    # Interior y-planar faces are the ends of notches shorter than the width
                    elif has_notches:
                        boundary_groups["notch"].append(tag)
                elif z_planar:
                    if np.isclose(zmin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif np.isclose(zmin, LZ, atol=tol):
                        boundary_groups["top"].append(tag)
                    elif has_notches:
                        boundary_groups["notch"].append(tag)
                elif has_notches:  # pragma: no cover
                    boundary_groups["notch"].append(tag)  # pragma: no cover

        surface_markers = {
            "bottom": (1, "Bottom"),
            "top": (2, "Top"),
            "front": (3, "Front"),
            "back": (4, "Back"),
            "left": (5, "Left"),
            "right": (6, "Right"),
        }
        if has_notches:
            surface_markers["notch"] = (7, "Notch")

        for name, (marker, physical_name) in surface_markers.items():
            surfaces = boundary_groups[name]
            if surfaces:
                gmsh.model.addPhysicalGroup(2, surfaces, marker)
                gmsh.model.setPhysicalName(2, marker, physical_name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
        gmsh.model.mesh.generate(3)

        boundary_markers = {
            "bottom": 1,
            "top": 2,
            "front": 3,
            "back": 4,
            "left": 5,
            "right": 6,
        }
        if has_notches:
            boundary_markers["notch"] = 7

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=3),
            domain_markers={"domain": 1},
            boundary_markers=boundary_markers,
        )

        gmsh.finalize()

        return generated

    def _resolve_notch_intervals(self, params):
        """PRIVATE: Validate notch inputs and return sorted x-intervals."""
        LX = float(params.Lx)
        depth_limit = float(params.Ly if params.dim == 2 else params.Lz)
        dx = float(params.dx)
        n_notches = int(getattr(params, "nNotches", 0))
        notch_positions = getattr(params, "Notch_Positions", None)

        if notch_positions is None:
            if n_notches <= 0:
                return [], 0.0
            notch_positions = [
                LX * (i + 0.5) / n_notches
                for i in range(n_notches)
            ]
        else:
            notch_positions = [float(x0) for x0 in notch_positions]
            if not notch_positions:
                return [], 0.0

        notch_width = float(getattr(params, "Notch_Width", 2.0 * dx))
        notch_depth = float(getattr(params, "Notch_Depth", 5.0 * dx))

        if notch_width <= 0.0 or notch_width >= LX:
            raise ValueError("Notch_Width must be positive and smaller than Lx.")
        if notch_depth <= 0.0 or notch_depth >= depth_limit:
            depth_axis = "Ly" if params.dim == 2 else "Lz"
            raise ValueError(f"Notch_Depth must be positive and smaller than {depth_axis}.")

        tol = max(1e-9, 1e-9 * LX)
        self._notch_order = sorted(range(len(notch_positions)), key=lambda i: notch_positions[i])
        intervals = []
        for x0 in notch_positions:
            x_min = x0 - 0.5 * notch_width
            x_max = x0 + 0.5 * notch_width
            if x_min < -tol or x_max > LX + tol:
                raise ValueError("Notch centres and widths must keep all notches within [0, Lx].")
            intervals.append((max(0.0, x_min), min(LX, x_max)))

        intervals.sort()
        for prev, curr in zip(intervals, intervals[1:]):
            if curr[0] < prev[1] - tol:
                raise ValueError("Top notches must not overlap.")

        return intervals, notch_depth

    def _resolve_notch_y_ranges(self, params, n_notches):
        """PRIVATE: Validate ``Notch_Y_Ranges`` (3-D) and return one ``(y_min, y_max)`` per sorted notch."""
        LY = float(params.Ly)
        ranges = getattr(params, "Notch_Y_Ranges", None)
        if n_notches == 0:
            return []
        if ranges is None:
            return [(0.0, LY)] * n_notches
        ranges = [tuple(float(v) for v in pair) for pair in ranges]
        if any(len(pair) != 2 for pair in ranges):
            raise ValueError("Notch_Y_Ranges must hold (y_min, y_max) pairs.")
        if len(ranges) == 1:
            ranges = ranges * n_notches
        if len(ranges) != n_notches:
            raise ValueError("Notch_Y_Ranges needs one (y_min, y_max) pair, or one pair per notch.")
        tol = max(1e-9, 1e-9 * LY)
        for y0, y1 in ranges:
            if not (-tol <= y0 < y1 <= LY + tol):
                raise ValueError("Each Notch_Y_Ranges pair must satisfy 0 <= y_min < y_max <= Ly.")
        ranges = [(max(0.0, y0), min(LY, y1)) for y0, y1 in ranges]
        return [ranges[i] for i in self._notch_order]
