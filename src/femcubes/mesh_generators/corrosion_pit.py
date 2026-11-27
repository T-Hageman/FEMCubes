from typing import Any

import numpy as np
import gmsh
from femcubes.core.mesh.generator_base import model_to_mesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D corrosion pit configuration."""

    class CorrosionPitSampleParameters(ParamsTemplate):
        MeshType = "CorrosionPit"
        dim = 2
        W = 4.0
        H = 2.0
        R_pit = 0.5
        dx = 0.4
        dx_metal = 0.1

    return CorrosionPitSampleParameters()


class CorrosionPitMeshGenerator(MeshGeneratorBase):
    """Generate an electrolyte above a metal surface with a semicircular pit.

    The electrolyte fills the rectangle ``[0, W] x [0, H]`` plus a half disk
    of radius ``R_pit`` below its bottom edge, centred at ``x = W / 2``: the
    pit dug into the metal surface ``y = 0``. The metal itself is not
    meshed; its surface is the boundary group ``metal``, split into the pit
    wall and the flat outer surface, so surface reactions can differ between
    them. Only 2-D is supported; passing ``dim=3`` raises
    ``NotImplementedError``.

    Required parameters
    -------------------
    dim : int
        Spatial dimension. Only 2 is supported.
    W, H : float
        Width and height of the electrolyte above the metal surface.
    R_pit : float
        Pit radius; must be smaller than ``W / 2`` and ``H``.
    dx : float
        Element size away from the metal.

    Optional parameters
    -------------------
    dx_metal : float
        Element size on the metal surface, grading back to ``dx`` over a
        distance of ``2 * dx``. Defaults to ``dx``.

    Named markers
    -------------
    Domains: ``domain`` (the electrolyte).
    Boundaries: ``top``, ``left``, ``right``, and ``metal`` =
    ``metal:pit`` (the semicircular pit wall) + ``metal:outer`` (the flat
    surface on both sides of the pit).

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE = "CorrosionPit"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, params: Any) -> GeneratedMeshResult:
        if params.dim != 2:
            raise NotImplementedError(
                "CorrosionPit only supports 2-D meshes (dim=2). "
                "3-D support is not yet implemented."
            )
        return self._generate_2d(params)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_2d(self, params) -> GeneratedMeshResult:
        """PRIVATE: Build the electrolyte rectangle with the pit below it."""
        W, H, R = float(params.W), float(params.H), float(params.R_pit)
        dx = float(params.dx)
        dx_metal = float(getattr(params, "dx_metal", dx))
        if W <= 0.0 or H <= 0.0 or R <= 0.0:
            raise ValueError("W, H and R_pit must be positive.")
        if R >= 0.5 * W or R >= H:
            raise ValueError("R_pit must be smaller than W / 2 and H.")
        if dx <= 0.0 or dx_metal <= 0.0:
            raise ValueError("dx and dx_metal must be positive.")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("corrosion_pit")

        rectangle = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, W, H)
        pit = gmsh.model.occ.addDisk(0.5 * W, 0.0, 0.0, R, R)
        # The upper half of the disk lies inside the rectangle, so the union
        # adds the half disk below the metal surface.
        gmsh.model.occ.fuse([(2, rectangle)], [(2, pit)])
        gmsh.model.occ.synchronize()

        surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "domain")

        tol = 1e-9 * W
        boundary_groups = {"top": [], "left": [], "right": [], "metal:outer": [], "metal:pit": []}
        for _, tag in gmsh.model.getEntities(1):
            x0, y0, _z0, x1, y1, _z1 = self._exact_bounding_box(1, tag)
            if y0 < -tol:
                boundary_groups["metal:pit"].append(tag)
            elif np.isclose(y0, H, atol=tol) and np.isclose(y1, H, atol=tol):
                boundary_groups["top"].append(tag)
            elif np.isclose(x0, 0.0, atol=tol) and np.isclose(x1, 0.0, atol=tol):
                boundary_groups["left"].append(tag)
            elif np.isclose(x0, W, atol=tol) and np.isclose(x1, W, atol=tol):
                boundary_groups["right"].append(tag)
            elif np.isclose(y0, 0.0, atol=tol) and np.isclose(y1, 0.0, atol=tol):
                boundary_groups["metal:outer"].append(tag)
            else:
                raise RuntimeError(
                    f"CorrosionPit could not classify boundary curve {tag} "
                    f"with bounds ({x0}, {y0}) - ({x1}, {y1})."
                )  # pragma: no cover

        boundary_markers = {name: marker for marker, name in enumerate(boundary_groups, start=1)}
        for name, marker in boundary_markers.items():
            gmsh.model.addPhysicalGroup(1, boundary_groups[name], marker)
            gmsh.model.setPhysicalName(1, marker, name)

        metal_curves = boundary_groups["metal:outer"] + boundary_groups["metal:pit"]
        self._set_mesh_size_fields(metal_curves, dx, dx_metal)
        gmsh.model.mesh.generate(2)

        generated = self._result_from_gmsh(
            model_to_mesh(gmsh.model, comm, rank=0, gdim=2),
            domain_markers={"domain": 1},
            boundary_markers=boundary_markers,
        )

        gmsh.finalize()

        return generated

    @staticmethod
    def _set_mesh_size_fields(metal_curves, dx, dx_metal) -> None:
        """PRIVATE: Grade from ``dx_metal`` on the metal surface to ``dx`` at ``2 * dx``."""
        fields = gmsh.model.mesh.field
        fields.add("Distance", 1)
        fields.setNumbers(1, "CurvesList", list(metal_curves))
        fields.setNumber(1, "Sampling", 200)
        fields.add("Threshold", 2)
        fields.setNumber(2, "InField", 1)
        fields.setNumber(2, "SizeMin", dx_metal)
        fields.setNumber(2, "SizeMax", dx)
        fields.setNumber(2, "DistMin", 0.0)
        fields.setNumber(2, "DistMax", 2.0 * dx)
        fields.setAsBackgroundMesh(2)

        # Sizes come exclusively from the background field.
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
