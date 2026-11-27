from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D quarter-plate-with-hole configuration."""

    class PlateWithHoleSampleParameters(ParamsTemplate):
        MeshType = "PlateWithHole"
        dim = 2
        Lx = 4.0
        Ly = 2.0
        R = 0.5
        dx = 0.2

    return PlateWithHoleSampleParameters()


class PlateWithHoleMeshGenerator(MeshGeneratorBase):
    """Generate a quarter-plate with a circular hole (Gmsh).

    Uses symmetry so only one quarter of the plate is meshed: the hole is
    centred at the origin and the domain is ``[0, Lx] x [0, Ly]`` outside
    it (thickness ``Lz`` along ``z`` in 3-D). Expects ``params`` to define
    ``Lx``, ``Ly`` (and ``Lz`` in 3-D), ``R`` (hole radius), ``dx`` and
    ``dim``. The hole is always circular; for an elliptical hole use
    ``FullPlateWithHole``.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "PlateWithHole"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim == 2:
            return self._generate_2d(params)
        elif params.dim == 3:
            return self._generate_3d(params)
        else:
            raise NotImplementedError("Only 2D and 3D plate-with-hole mesh generation is implemented.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 2-D quarter-plate-with-hole mesh variant."""
        LX = params.Lx
        LY = params.Ly
        R = params.R
        dx = params.dx

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate_with_hole")

        p0 = gmsh.model.geo.addPoint(0, 0, 0)
        p1 = gmsh.model.geo.addPoint(R, 0, 0)
        p2 = gmsh.model.geo.addPoint(LX, 0, 0)
        p3 = gmsh.model.geo.addPoint(LX, LY, 0)
        p4 = gmsh.model.geo.addPoint(0, LY, 0)
        p5 = gmsh.model.geo.addPoint(0, R, 0)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p5)
        l5 = gmsh.model.geo.addCircleArc(p5, p0, p1)

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5])
        surface = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.geo.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface], 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        gmsh.model.addPhysicalGroup(1, [l1], 1)
        gmsh.model.setPhysicalName(1, 1, "Bottom")
        gmsh.model.addPhysicalGroup(1, [l2], 2)
        gmsh.model.setPhysicalName(1, 2, "Right")
        gmsh.model.addPhysicalGroup(1, [l3], 3)
        gmsh.model.setPhysicalName(1, 3, "Top")
        gmsh.model.addPhysicalGroup(1, [l4], 4)
        gmsh.model.setPhysicalName(1, 4, "Left")
        gmsh.model.addPhysicalGroup(1, [l5], 5)
        gmsh.model.setPhysicalName(1, 5, "Hole")

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers={"bottom": 1, "right": 2, "top": 3, "left": 4, "hole": 5},
        )

        gmsh.finalize()

        return generated

    def _generate_3d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 3-D quarter-plate-with-hole mesh variant."""
        LX = params.Lx
        LY = params.Ly
        LZ = params.Lz
        R = params.R
        dx = params.dx

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate_with_hole_3d")

        box = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)
        cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, LZ, R)
        cut = gmsh.model.occ.cut([(3, box)], [(3, cylinder)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        volumes = [tag for dim, tag in cut[0] if dim == 3]
        if not volumes:
            raise RuntimeError("Plate-with-hole 3D mesh generation failed to create a volume.")  # pragma: no cover

        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "Domain")

        boundary_groups = {
            "bottom": [], "right": [], "top": [],
            "left": [], "hole": [], "front": [], "back": []
        }
        max_dim = max(LX, LY, LZ, R)
        tol = max(1e-6 * max_dim, 1e-9)

        for vol in volumes:
            for _, tag in gmsh.model.getBoundary([(3, vol)], oriented=False, recursive=False):
                xmin, ymin, zmin, xmax, ymax, zmax = self._exact_bounding_box(2, tag)
                x_planar = np.isclose(xmin, xmax, atol=tol)
                y_planar = np.isclose(ymin, ymax, atol=tol)
                z_planar = np.isclose(zmin, zmax, atol=tol)

                if x_planar:
                    if np.isclose(xmin, 0.0, atol=tol):
                        boundary_groups["front"].append(tag)
                    elif np.isclose(xmin, LX, atol=tol):
                        boundary_groups["back"].append(tag)
                elif y_planar:
                    if np.isclose(ymin, 0.0, atol=tol):
                        boundary_groups["left"].append(tag)
                    elif np.isclose(ymin, LY, atol=tol):
                        boundary_groups["right"].append(tag)
                elif z_planar:
                    if np.isclose(zmin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif np.isclose(zmin, LZ, atol=tol):
                        boundary_groups["top"].append(tag)
                else:
                    boundary_groups["hole"].append(tag)

        surface_markers = {
            "bottom": (1, "Bottom"), "top": (2, "Top"),
            "front": (3, "Front"), "back": (4, "Back"),
            "left": (5, "Left"), "right": (6, "Right"),
            "hole": (7, "Hole")
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
                "top": 2,
                "front": 3,
                "back": 4,
                "left": 5,
                "right": 6,
                "hole": 7,
            },
        )

        gmsh.finalize()

        return generated
