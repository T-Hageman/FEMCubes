from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D single-edge-notched configuration."""

    class SingleEdgeNotchedSampleParameters(ParamsTemplate):
        MeshType = "SingleEdgeNotched"
        dim = 2
        Lx = 1.0
        Ly = 0.4
        LNotch = 0.2
        NotchHeight = 0.04
        dx = 0.02

    return SingleEdgeNotchedSampleParameters()


class SingleEdgeNotchedMeshGenerator(MeshGeneratorBase):
    """Generate a plate with a single-edge notch (Gmsh).

    Expects ``params`` to define ``Lx``, ``Ly`` (and ``Lz`` in 3-D),
    ``LNotch`` (notch length from ``x = 0``), ``dx`` and ``dim``. The
    optional ``NotchHeight`` (default ``dx``) is the notch height, centred
    at ``y = Ly/2``.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "SingleEdgeNotched"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim == 2:
            return self._generate_2d(params)
        elif params.dim == 3:
            return self._generate_3d(params)
        else:
            raise NotImplementedError("Only 2D and 3D single-edge-notched mesh generation is implemented.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 2-D single-edge-notched mesh variant."""
        LX = params.Lx
        LY = params.Ly
        LNotch = params.LNotch
        dx = params.dx

        notch_height = getattr(params, "NotchHeight", dx)
        notch_height = max(notch_height, 1.0e-6 * LY)
        notch_y0 = 0.5 * (LY - notch_height)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("single_edge_notched")

        plate = gmsh.model.occ.addRectangle(0, 0, 0, LX, LY)
        notch = gmsh.model.occ.addRectangle(0, notch_y0, 0, LNotch, notch_height)
        cut = gmsh.model.occ.cut([(2, plate)], [(2, notch)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        surfaces = [tag for dim, tag in cut[0] if dim == 2]
        if not surfaces:
            raise RuntimeError("Single-edge-notched mesh generation failed to create a surface.")  # pragma: no cover

        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        boundary_groups = {
            "bottom": [], "right": [], "top": [], "left": [], "notch": []
        }
        max_dim = max(LX, LY, LNotch)
        tol = max(1e-6 * max_dim, 1e-9)

        for surf in surfaces:
            for _, tag in gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False):
                xmin, ymin, zmin, xmax, ymax, zmax = self._exact_bounding_box(1, tag)
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
                else:
                    boundary_groups["notch"].append(tag)

        surface_markers = {
            "bottom": (1, "Bottom"), "right": (2, "Right"),
            "top": (3, "Top"), "left": (4, "Left"), "notch": (5, "Notch")
        }
        for name, (marker, physical_name) in surface_markers.items():
            segments = boundary_groups[name]
            if segments:
                gmsh.model.addPhysicalGroup(1, segments, marker)
                gmsh.model.setPhysicalName(1, marker, physical_name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers={"bottom": 1, "right": 2, "top": 3, "left": 4, "notch": 5},
        )

        gmsh.finalize()

        return generated

    def _generate_3d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 3-D single-edge-notched mesh variant."""
        LX = params.Lx
        LY = params.Ly
        LZ = params.Lz
        LNotch = params.LNotch
        dx = params.dx

        notch_height = getattr(params, "NotchHeight", dx)
        notch_height = max(notch_height, 1.0e-6 * LY)
        notch_y0 = 0.5 * (LY - notch_height)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("single_edge_notched_3d")

        plate = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)
        notch = gmsh.model.occ.addBox(0, notch_y0, 0, LNotch, notch_height, LZ)
        cut = gmsh.model.occ.cut([(3, plate)], [(3, notch)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        volumes = [tag for dim, tag in cut[0] if dim == 3]
        if not volumes:
            raise RuntimeError("Single-edge-notched 3D mesh generation failed to create a volume.")  # pragma: no cover

        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "Domain")

        boundary_groups = {
            "bottom": [], "right": [], "top": [],
            "left": [], "notch": [], "front": [], "back": []
        }
        max_dim = max(LX, LY, LZ, LNotch)
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
                    else:
                        boundary_groups["notch"].append(tag)
                elif y_planar:
                    if np.isclose(ymin, 0.0, atol=tol):
                        boundary_groups["left"].append(tag)
                    elif np.isclose(ymin, LY, atol=tol):
                        boundary_groups["right"].append(tag)
                    else:
                        boundary_groups["notch"].append(tag)
                elif z_planar:
                    if np.isclose(zmin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif np.isclose(zmin, LZ, atol=tol):
                        boundary_groups["top"].append(tag)
                else:
                    boundary_groups["notch"].append(tag)  # pragma: no cover

        surface_markers = {
            "bottom": (1, "Bottom"), "top": (2, "Top"),
            "front": (3, "Front"), "back": (4, "Back"),
            "left": (5, "Left"), "right": (6, "Right"),
            "notch": (7, "Notch")
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
                "notch": 7,
            },
        )

        gmsh.finalize()

        return generated
