from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal cylindrical mesh configuration."""

    class CylinderSampleParameters(ParamsTemplate):
        MeshType = "Cylinder"
        dim = 3
        R = 0.5
        Lz = 1.0
        dx = 0.15
        notch_r = 0.0
        notch_h = 0.5
        notch_dh = 0.2

    return CylinderSampleParameters()


class CylinderMeshGenerator(MeshGeneratorBase):
    """Generate a 3-D cylindrical mesh (Gmsh).

    Expects ``params`` to define ``R`` (radius), ``Lz`` (height),
    ``dx``, and ``dim`` (must be 3). The axis is the ``z`` axis. With
    ``notch_r > 0`` (default 0) a circumferential groove of radial depth
    ``notch_r`` and height ``notch_dh`` (default ``dx``) is cut into the
    outer surface, centred at ``z = notch_h`` (default ``Lz / 2``).

    With ``slot_depth > 0`` (default 0) a short planar slot is cut into the
    outer surface instead, centred at ``(R, 0, slot_h)`` (default
    ``slot_h = Lz / 2``): radial depth ``slot_depth``, length ``slot_length``
    (default ``2 slot_depth``) along the surface and width ``slot_width``
    (default ``dx / 2``). Its trace on the surface makes the angle
    ``slot_angle`` (degrees, default 45) with the circumferential direction,
    descending in ``z`` towards ``+y``: with the default, the slot lies
    perpendicular to the largest principal stress of a torsion that rotates
    the top anticlockwise about ``+z`` relative to the bottom. The groove and
    the slot cannot be combined.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "Cylinder"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim != 3:
            raise NotImplementedError(
                "Cylinder mesh generation is only implemented for 3D."
            )

        R = params.R
        Lz = params.Lz
        dx = params.dx
        notch_r = getattr(params, "notch_r", 0.0)
        notch_h = getattr(params, "notch_h", 0.5 * Lz)
        notch_dh = getattr(params, "notch_dh", dx)
        has_notch = notch_r > 0.0
        slot_depth = getattr(params, "slot_depth", 0.0)
        has_slot = slot_depth > 0.0
        if has_notch and has_slot:
            raise ValueError("Cylinder: notch_r and slot_depth cannot both be set.")
        if has_slot:
            slot_length = getattr(params, "slot_length", 2.0 * slot_depth)
            slot_width = getattr(params, "slot_width", 0.5 * dx)
            slot_h = getattr(params, "slot_h", 0.5 * Lz)
            slot_angle = getattr(params, "slot_angle", 45.0)
            if not slot_depth < R:
                raise ValueError("Cylinder: slot_depth must be smaller than R.")
            if slot_length <= 0.0 or slot_width <= 0.0:
                raise ValueError("Cylinder: slot_length and slot_width must be positive.")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cylinder")

        cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, Lz, R)

        if has_notch:
            notch_z0 = notch_h - 0.5 * notch_dh
            outer_ring = gmsh.model.occ.addCylinder(
                0, 0, notch_z0, 0, 0, notch_dh, R + 1.0e-3
            )
            inner_ring = gmsh.model.occ.addCylinder(
                0, 0, notch_z0, 0, 0, notch_dh, R - notch_r
            )
            ring_cut = gmsh.model.occ.cut(
                [(3, outer_ring)], [(3, inner_ring)],
                removeObject=True, removeTool=True,
            )
            ring_vols = [(d, t) for d, t in ring_cut[0] if d == 3]
            body_cut = gmsh.model.occ.cut(
                [(3, cyl)], ring_vols,
                removeObject=True, removeTool=True,
            )
            volumes = [t for d, t in body_cut[0] if d == 3]
        elif has_slot:
            # Box from the slot floor to beyond the surface; its length runs along y and its width along z,
            # then it is turned about the x axis so that its trace makes slot_angle with the circumference.
            overshoot = 0.1 * R
            slot = gmsh.model.occ.addBox(
                R - slot_depth, -0.5 * slot_length, -0.5 * slot_width,
                slot_depth + overshoot, slot_length, slot_width,
            )
            gmsh.model.occ.rotate([(3, slot)], 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -np.deg2rad(slot_angle))
            gmsh.model.occ.translate([(3, slot)], 0.0, 0.0, slot_h)
            body_cut = gmsh.model.occ.cut(
                [(3, cyl)], [(3, slot)],
                removeObject=True, removeTool=True,
            )
            volumes = [t for d, t in body_cut[0] if d == 3]
        else:
            volumes = [cyl]

        gmsh.model.occ.synchronize()

        if not volumes:
            raise RuntimeError(  # pragma: no cover
                "Cylinder mesh generation failed to create a volume."
            )

        gmsh.model.addPhysicalGroup(3, volumes, 1)
        gmsh.model.setPhysicalName(3, 1, "Domain")

        boundary_groups = {"bottom": [], "top": [], "outer": []}
        if has_notch or has_slot:
            boundary_groups["notch"] = []

        max_dim = max(R, Lz)
        tol = max(1e-6 * max_dim, 1e-9)

        for vol in volumes:
            for _, tag in gmsh.model.getBoundary(
                [(3, vol)], oriented=False, recursive=False
            ):
                xmin, ymin, zmin, xmax, ymax, zmax = (
                    self._exact_bounding_box(2, tag)
                )
                z_planar = np.isclose(zmin, zmax, atol=tol)

                if z_planar:
                    if np.isclose(zmin, 0.0, atol=tol):
                        boundary_groups["bottom"].append(tag)
                    elif np.isclose(zmin, Lz, atol=tol):
                        boundary_groups["top"].append(tag)
                    else:
                        if has_notch:
                            boundary_groups["notch"].append(tag)
                else:
                    if has_notch:
                        # Curved extents are sampled, so compare against the
                        # radius halfway between the notch root and the
                        # outer surface rather than requiring an exact 2R.
                        x_extent = xmax - xmin
                        if x_extent > 2.0 * R - notch_r:
                            boundary_groups["outer"].append(tag)
                        else:
                            boundary_groups["notch"].append(tag)
                    elif has_slot:
                        # The slot faces are small; the cylindrical surface spans the full diameter
                        if xmax - xmin > R:
                            boundary_groups["outer"].append(tag)
                        else:
                            boundary_groups["notch"].append(tag)
                    else:
                        boundary_groups["outer"].append(tag)

        surface_markers = {
            "bottom": (1, "Bottom"),
            "top":    (2, "Top"),
            "outer":  (3, "Outer"),
        }
        if has_notch or has_slot:
            surface_markers["notch"] = (4, "Notch")

        for name, (marker, physical_name) in surface_markers.items():
            surfaces = boundary_groups[name]
            if surfaces:
                gmsh.model.addPhysicalGroup(2, surfaces, marker)
                gmsh.model.setPhysicalName(2, marker, physical_name)

        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), dx)
        gmsh.model.mesh.generate(3)

        boundary_markers = {"bottom": 1, "top": 2, "outer": 3}
        if has_notch or has_slot:
            boundary_markers["notch"] = 4

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=3),
            domain_markers={"domain": 1},
            boundary_markers=boundary_markers,
        )

        gmsh.finalize()

        return generated
