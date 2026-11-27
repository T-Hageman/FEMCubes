from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D rectangle-with-centre-notch configuration."""

    class RectangleCentreNotchSampleParameters(ParamsTemplate):
        MeshType = "Rectangle_CentreNotch"
        dim = 2
        Lx = 4.0
        Ly = 2.0
        dx = 0.2
        Notch_Width = 0.25
        Notch_Depth = 0.5
        dx_notch = 0.05

    return RectangleCentreNotchSampleParameters()


class RectangleCentreNotchMeshGenerator(MeshGeneratorBase):
    """Generate a 2-D rectangle with a rectangular hole at its centre.

    Expects ``params`` to define ``Lx``, ``Ly``, ``dx``, ``dx_notch`` and
    ``dim`` (must be 2). The hole is ``Notch_Width`` wide (in ``x``) and
    ``Notch_Depth`` high (in ``y``), centred at ``(Lx/2, Ly/2)``. Elements
    of size ``dx_notch`` fill a box that extends ``5 dx_notch`` beyond the
    hole. The named boundaries are ``bottom``, ``right``, ``top`` and
    ``left``; the hole boundary is not tagged.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "Rectangle_CentreNotch"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim == 2:
            return self._generate_2d(params)
        # elif params.dim == 3:
        #     return self._generate_3d(params)
        else:
            raise NotImplementedError(
                "Only 2D rectangle-with-centre-notch mesh generation is implemented."
            )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 2-D rectangle-with-centre-notch mesh variant."""

        LX = params.Lx
        LY = params.Ly
        dx = params.dx
        dx_small = params.dx_notch

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("rectangle_notches")

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------

        plate = gmsh.model.occ.addRectangle(0, 0, 0, LX, LY)

        x0 = LX / 2 - params.Notch_Width / 2
        x1 = LX / 2 + params.Notch_Width / 2

        notch = gmsh.model.occ.addRectangle(
            x0,
            LY / 2 - params.Notch_Depth / 2,
            0,
            params.Notch_Width,
            params.Notch_Depth,
        )

        cut = gmsh.model.occ.cut(
            [(2, plate)],
            [(2, notch)],
            removeObject=True,
            removeTool=True,
        )

        surfaces = [tag for dim, tag in cut[0] if dim == 2]

        gmsh.model.occ.synchronize()

        if not surfaces:
            raise RuntimeError("Rectangle_Notches mesh generation failed.")

        # ------------------------------------------------------------------
        # Physical groups
        # ------------------------------------------------------------------

        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        boundary_groups = {
            "bottom": [],
            "right": [],
            "top": [],
            "left": [],
        }

        max_dim = max(LX, LY)
        tol = max(1e-6 * max_dim, 1e-9)

        for surf in surfaces:
            for _, tag in gmsh.model.getBoundary(
                [(2, surf)],
                oriented=False,
                recursive=False,
            ):
                xmin, ymin, _, xmax, ymax, _ = self._exact_bounding_box(1, tag)

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

        line_markers = {
            "bottom": (1, "Bottom"),
            "right": (2, "Right"),
            "top": (3, "Top"),
            "left": (4, "Left"),
        }

        for name, (marker, physical_name) in line_markers.items():
            if boundary_groups[name]:
                gmsh.model.addPhysicalGroup(1, boundary_groups[name], marker)
                gmsh.model.setPhysicalName(1, marker, physical_name)

        # ------------------------------------------------------------------
        # Mesh sizing
        # ------------------------------------------------------------------

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)

        # Disable the default sizing heuristics
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # Refinement box dimensions
        pad = 5 * dx_small

        field = gmsh.model.mesh.field.add("Box")

        gmsh.model.mesh.field.setNumber(field, "VIn", dx_small)
        gmsh.model.mesh.field.setNumber(field, "VOut", dx)

        gmsh.model.mesh.field.setNumber(field, "XMin", x0 - pad)
        gmsh.model.mesh.field.setNumber(field, "XMax", x1 + pad)

        gmsh.model.mesh.field.setNumber(
            field,
            "YMin",
            LY / 2 - params.Notch_Depth / 2 - pad,
        )
        gmsh.model.mesh.field.setNumber(
            field,
            "YMax",
            LY / 2 + params.Notch_Depth / 2 + pad,
        )

        # Smooth transition width
        gmsh.model.mesh.field.setNumber(field, "Thickness", 2 * dx)

        gmsh.model.mesh.field.setAsBackgroundMesh(field)

        # ------------------------------------------------------------------
        # Generate mesh
        # ------------------------------------------------------------------

        gmsh.model.mesh.generate(2)

        boundary_markers = {
            "bottom": 1,
            "right": 2,
            "top": 3,
            "left": 4,
        }

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers=boundary_markers,
        )

        gmsh.finalize()

        return generated
