from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D full plate-with-hole configuration."""

    class FullPlateWithHoleSampleParameters(ParamsTemplate):
        MeshType = "FullPlateWithHole"
        dim = 2
        Lx = 4.0
        Ly = 4.0
        R = 0.5
        dx = 0.2
        ellipticity = 1.0
        rotation_angle = 0.0

    return FullPlateWithHoleSampleParameters()


class FullPlateWithHoleMeshGenerator(MeshGeneratorBase):
    """Generate a full plate with an elliptical hole (Gmsh).

    Unlike ``PlateWithHole``, the complete plate is meshed (no symmetry).
    Expects ``params`` to define ``Lx``, ``Ly`` (and ``Lz`` in 3-D), ``R``,
    ``dx`` and ``dim``. Optional: ``ellipticity`` (default 1; the hole has
    semi-axes ``R`` and ``R / ellipticity``) and ``rotation_angle``
    (default 0, degrees, counter-clockwise). In 2-D only, ``dxmin``
    (default ``dx``) is the element size within ``R_Refine`` (default
    ``2 R``) of the hole, grading to ``dx`` at ``2 R_Refine``; the 3-D mesh
    has size ``dx`` everywhere.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "FullPlateWithHole"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim == 2:
            return self._generate_2d(params)
        elif params.dim == 3:
            return self._generate_3d(params)
        else:
            raise NotImplementedError("Only 2D and 3D full plate-with-hole mesh generation is implemented.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params, *, vertical_symmetry=False) -> GeneratedMeshResult:
        """PRIVATE: Generate the 2-D plate or its right half using the same sizing."""
        LX = params.Lx
        LY = params.Ly
        R = params.R
        dx = params.dx
        dx_small = getattr(params, "dxmin", dx)
        R_Refine = getattr(params, "R_Refine", 2.0 * R)
        ellipticity = getattr(params, "ellipticity", 1.0)
        rotation_angle = getattr(params, "rotation_angle", 0.0)
        x_start = 0.5 * LX if vertical_symmetry else 0.0
        left_boundary = "symmetry" if vertical_symmetry else "left"

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("half_plate_with_hole" if vertical_symmetry else "full_plate_with_hole")

        plate = gmsh.model.occ.addRectangle(x_start, 0, 0, LX - x_start, LY)

        Rx = R
        Ry = R / ellipticity
        hole = gmsh.model.occ.addDisk(0.5 * LX, 0.5 * LY, 0, Rx, Ry)

        if not np.isclose(rotation_angle, 0.0):
            gmsh.model.occ.rotate(
                [(2, hole)],
                0.5 * LX, 0.5 * LY, 0,
                0, 0, 1,
                np.radians(rotation_angle)
            )

        cut = gmsh.model.occ.cut(
            [(2, plate)],
            [(2, hole)],
            removeObject=True,
            removeTool=True
        )

        gmsh.model.occ.synchronize()

        surfaces = [tag for dim, tag in cut[0] if dim == 2]
        if not surfaces:
            raise RuntimeError(
                "Full plate-with-hole mesh generation failed to create a surface."
            )  # pragma: no cover

        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "Domain")

        boundary_groups = {
            "bottom": [],
            "right": [],
            "top": [],
            left_boundary: [],
            "hole": [],
        }

        max_dim = max(LX, LY, R)
        tol = max(1e-6 * max_dim, 1e-9)

        for surf in surfaces:
            for _, tag in gmsh.model.getBoundary(
                [(2, surf)], oriented=False, recursive=False
            ):
                xmin, ymin, zmin, xmax, ymax, zmax = self._exact_bounding_box(1, tag)

                x_planar = np.isclose(xmin, xmax, atol=tol)
                y_planar = np.isclose(ymin, ymax, atol=tol)

                if y_planar and np.isclose(ymin, 0.0, atol=tol):
                    boundary_groups["bottom"].append(tag)
                elif y_planar and np.isclose(ymin, LY, atol=tol):
                    boundary_groups["top"].append(tag)
                elif x_planar and np.isclose(xmin, x_start, atol=tol):
                    boundary_groups[left_boundary].append(tag)
                elif x_planar and np.isclose(xmin, LX, atol=tol):
                    boundary_groups["right"].append(tag)
                else:
                    boundary_groups["hole"].append(tag)

        surface_markers = {
            "bottom": (1, "Bottom"),
            "right": (2, "Right"),
            "top": (3, "Top"),
            left_boundary: (4, "Symmetry" if vertical_symmetry else "Left"),
            "hole": (5, "Hole"),
        }

        for name, (marker, physical_name) in surface_markers.items():
            if boundary_groups[name]:
                gmsh.model.addPhysicalGroup(1, boundary_groups[name], marker)
                gmsh.model.setPhysicalName(1, marker, physical_name)

        # ------------------------------------------------------------------
        # Mesh sizing
        # ------------------------------------------------------------------

        # Global mesh size
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)

        # Distance field from the hole boundary
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            distance,
            "CurvesList",
            boundary_groups["hole"]
        )
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)

        # Threshold field controlling mesh size
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", dx_small)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", dx)

        # Keep dx_small within 5*dx_small of the hole,
        # then transition to dx by 10*dx_small.
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", R_Refine)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", 2*R_Refine)

        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)

        # Prevent Gmsh's default size heuristics overriding the field
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers={
                "bottom": 1,
                "right": 2,
                "top": 3,
                left_boundary: 4,
                "hole": 5,
            },
        )

        gmsh.finalize()

        return generated

    def _generate_3d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Generate the 3-D full-plate-with-hole mesh variant."""
        LX = params.Lx
        LY = params.Ly
        LZ = params.Lz
        R = params.R
        dx = params.dx
        ellipticity = getattr(params, "ellipticity", 1.0)
        rotation_angle = getattr(params, "rotation_angle", 0.0)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("full_plate_with_hole_3d")

        box = gmsh.model.occ.addBox(0, 0, 0, LX, LY, LZ)
        Rx = R
        Ry = R / ellipticity
        disk = gmsh.model.occ.addDisk(0.5 * LX, 0.5 * LY, 0, Rx, Ry)
        if not np.isclose(rotation_angle, 0.0):
            gmsh.model.occ.rotate(
                [(2, disk)],
                0.5 * LX, 0.5 * LY, 0,
                0, 0, 1,
                np.radians(rotation_angle)
            )
        extrusion = gmsh.model.occ.extrude([(2, disk)], 0, 0, LZ)
        tool_volumes = [(dim, tag) for dim, tag in extrusion if dim == 3]
        cut = gmsh.model.occ.cut([(3, box)], tool_volumes, removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()

        volumes = [tag for dim, tag in cut[0] if dim == 3]
        if not volumes:
            raise RuntimeError("Full plate-with-hole 3D mesh generation failed to create a volume.")  # pragma: no cover

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
