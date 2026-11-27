"""Half of the full plate-with-hole geometry, cut at its vertical centreline."""

from typing import Any

import numpy as np

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.generator_base import GeneratedMeshResult
from .full_plate_with_hole import FullPlateWithHoleMeshGenerator


def Sample_Parameters() -> ParamsTemplate:
    """Return a minimal 2-D half-plate-with-hole configuration."""

    class HalfPlateWithHoleSampleParameters(ParamsTemplate):
        MeshType = "HalfPlateWithHole"
        dim = 2
        Lx = 4.0
        Ly = 4.0
        R = 0.5
        dx = 0.2

    return HalfPlateWithHoleSampleParameters()


class HalfPlateWithHoleMeshGenerator(FullPlateWithHoleMeshGenerator):
    """Generate the right half of a 2-D plate with a centred hole.

    ``Lx`` and ``Ly`` remain the dimensions of the *full* plate. The mesh
    occupies ``Lx/2 <= x <= Lx`` and ``0 <= y <= Ly``, with the right half
    of the hole at ``(Lx/2, Ly/2)`` removed. Required parameters are ``dim``,
    ``Lx``, ``Ly``, ``R``, and ``dx``. Optional ``dxmin`` and ``R_Refine``
    use the same distance-based hole sizing as ``FullPlateWithHole``.

    The named boundaries are ``bottom``, ``right``, ``top``, ``hole``, and
    ``symmetry``. The last comprises both straight segments at ``x=Lx/2``;
    the hole arc is excluded. The domain is named ``domain``.

    Circular and axis-aligned elliptical holes are supported. The optional
    ``ellipticity`` and ``rotation_angle`` follow ``FullPlateWithHole``;
    a noncircular ellipse must be rotated by a multiple of 90 degrees to
    preserve reflection symmetry. Only ``dim=2`` is supported.

    The generator imposes no boundary conditions. For symmetric mechanics,
    prescribe ``u_x=0`` on ``symmetry`` and leave ``u_y`` free there. With
    the usual phase-field gradient term, leave the phase field unconstrained
    there to retain its natural zero-normal-gradient condition. Top reactions
    are half-plate resultants: double them to compare with the full plate.
    """

    MESH_TYPE = "HalfPlateWithHole"

    def generate(self, params: Any) -> GeneratedMeshResult:
        """Build a symmetry-compatible half plate with the full-plate sizing."""
        if params.dim != 2:
            raise NotImplementedError("HalfPlateWithHole supports only dim=2.")
        for name in ("Lx", "Ly", "R", "dx", "dxmin", "R_Refine", "ellipticity"):
            default = {"dxmin": params.dx, "R_Refine": 2 * params.R, "ellipticity": 1.0}.get(name)
            value = getattr(params, name, default)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"HalfPlateWithHole requires positive finite {name}.")
        angle = getattr(params, "rotation_angle", 0.0)
        if not np.isfinite(angle):
            raise ValueError("HalfPlateWithHole requires a finite rotation_angle.")
        ellipticity = getattr(params, "ellipticity", 1.0)
        if ellipticity != 1.0 and not np.isclose(angle / 90, round(angle / 90), rtol=0, atol=1e-12):
            raise ValueError("A noncircular hole must have rotation_angle a multiple of 90 degrees for vertical symmetry.")
        angle = np.radians(angle)
        rx, ry = params.R, params.R / ellipticity
        extent_x = np.hypot(rx * np.cos(angle), ry * np.sin(angle))
        extent_y = np.hypot(rx * np.sin(angle), ry * np.cos(angle))
        if extent_x >= params.Lx / 2 or extent_y >= params.Ly / 2:
            raise ValueError("The hole must fit strictly inside the full plate.")
        return self._generate_2d(params, vertical_symmetry=True)
