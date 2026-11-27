"""Base mesh mix-in that stores shared state for the composite mesh object."""

from __future__ import annotations

from typing import Any, Dict

import ufl
from dolfinx import fem


class _MeshBase:
    """Root mix-in that stores the parameter object and the DOLFINx mesh.

    This is the first class in the Mesh MRO chain.  It forwards
    ``params`` to all subsequent mix-ins via cooperative ``super()``.

    Attributes
    ----------
    params : ParamsTemplate
        The simulation parameter object.

    Notes
    -----
    ``_MeshBase`` intentionally does very little.  Its main purpose is
    to provide a stable place for attributes shared by all later mesh
    mix-ins and to start the cooperative ``super().__init__`` chain.
    """

    def __init__(self, params: Any) -> None:

        self.params = params
        self.remesh_epoch = 0
        self.last_refinement_report = None
        if self.axisymmetric and getattr(params, "dim", None) != 2:
            raise ValueError(f"Axisymmetric meshes must be 2-D (the r-z plane), not dim = {getattr(params, 'dim', None)}.")

        super().__init__(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def axisymmetric(self) -> bool:
        """Whether the 2-D mesh is the r-z plane of an axisymmetric domain (``params.Axisymmetric``).

        The x coordinate is then the radius r, and all domain, boundary and
        lumped integrals carry the weight r (integrals per radian: multiply
        by 2 pi for the full revolution).
        """
        return bool(getattr(self.params, "Axisymmetric", False))

    def Get_Integration_Weight(self, domain=None):
        """Return the integration weight: the radius ``x[0]`` of ``domain`` (default: the mesh) if axisymmetric, else 1."""
        if not self.axisymmetric:
            return 1
        return ufl.SpatialCoordinate(self.mesh if domain is None else domain)[0]


class WeightedMeasure:
    """A UFL measure that multiplies every integrand by a weight (the radius of axisymmetric meshes).

    ``integrand * measure`` gives ``(integrand * weight) * measure``, and calling it (``measure(tags)``)
    restricts the underlying measure as for a plain UFL measure.
    """

    def __init__(self, measure, weight):
        self.measure = measure
        self.weight = weight

    def __call__(self, *args, **kwargs):
        return WeightedMeasure(self.measure(*args, **kwargs), self.weight)

    def __rmul__(self, integrand):
        return (integrand * self.weight) * self.measure
