from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D rectangle configuration."""

    class RectangleSampleParameters(ParamsTemplate):
        MeshType = "Rectangle"
        dim = 2
        Lx = 4.0
        Ly = 2.0
        dx = 0.2

    return RectangleSampleParameters()


class RectangleMeshGenerator(MeshGeneratorBase):
    """Generate an unstructured mesh of a rectangle (2-D) or box (3-D).

    Expects ``params`` to define ``Lx``, ``Ly`` (and ``Lz`` in 3-D),
    ``dx`` (element size), and ``dim`` (2 or 3). The domain is
    ``[0, Lx] x [0, Ly]``, or ``[0, Lx] x [0, Ly] x [0, Lz]`` in 3-D.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "Rectangle"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim == 2:
            return self._generate_2d(params)
        elif params.dim == 3:
            return self._generate_3d(params)
        else:
            raise NotImplementedError("Only 2D and 3D rectangle mesh generation is implemented.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 2-D rectangular mesh variant."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rectangle")

        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(params.Lx, 0, 0)
        p3 = gmsh.model.geo.addPoint(params.Lx, params.Ly, 0)
        p4 = gmsh.model.geo.addPoint(0, params.Ly, 0)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
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

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), params.dx)
        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers={"bottom": 1, "right": 2, "top": 3, "left": 4},
        )

        gmsh.finalize()

        return generated

    def _generate_3d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 3-D rectangular mesh variant."""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")

        volume = gmsh.model.occ.addBox(0, 0, 0, params.Lx, params.Ly, params.Lz)
        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(3, [volume], 1)
        gmsh.model.setPhysicalName(3, 1, "Domain")

        boundary_groups = {
            "front": [], "back": [], "left": [],
            "right": [], "bottom": [], "top": []
        }
        max_dim = max(params.Lx, params.Ly, params.Lz)
        tol = max(1e-6 * max_dim, 1e-9)
        for _, tag in gmsh.model.occ.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = self._exact_bounding_box(2, tag)
            if np.isclose(xmin, xmax, atol=tol):
                if np.isclose(xmin, 0.0, atol=tol):
                    boundary_groups["front"].append(tag)
                elif np.isclose(xmin, params.Lx, atol=tol):
                    boundary_groups["back"].append(tag)
            elif np.isclose(ymin, ymax, atol=tol):
                if np.isclose(ymin, 0.0, atol=tol):
                    boundary_groups["left"].append(tag)
                elif np.isclose(ymin, params.Ly, atol=tol):
                    boundary_groups["right"].append(tag)
            elif np.isclose(zmin, zmax, atol=tol):
                if np.isclose(zmin, 0.0, atol=tol):
                    boundary_groups["bottom"].append(tag)
                elif np.isclose(zmin, params.Lz, atol=tol):
                    boundary_groups["top"].append(tag)

        surface_markers = {
            "front": (1, "Front"), "back": (2, "Back"),
            "left": (3, "Left"), "right": (4, "Right"),
            "bottom": (5, "Bottom"), "top": (6, "Top")
        }
        for name, (marker, physical_name) in surface_markers.items():
            surfaces = boundary_groups[name]
            gmsh.model.addPhysicalGroup(2, surfaces, marker)
            gmsh.model.setPhysicalName(2, marker, physical_name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), params.dx)
        gmsh.model.mesh.generate(3)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=3),
            domain_markers={"domain": 1},
            boundary_markers={
                "front": 1,
                "back": 2,
                "left": 3,
                "right": 4,
                "bottom": 5,
                "top": 6,
            },
        )

        gmsh.finalize()

        return generated
