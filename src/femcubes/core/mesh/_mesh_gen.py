"""Mesh-generation mix-in for the composite :class:`femcubes.core.mesh.mesh.Mesh`."""

from typing import Any

from femcubes.core.mesh.generator_base import MeshGeneratorBase
# Importing the generator package registers every bundled generator.
import femcubes.mesh_generators  # noqa: F401


class _MeshGen:
    """Mix-in that delegates mesh creation to auto-registered generators.

    On initialisation, looks up the generator class whose
    ``MESH_TYPE`` matches ``params.MeshType`` and assigns the
    structured result returned by :meth:`~MeshGeneratorBase.generate`
    onto the mesh wrapper.

    Public methods are part of the mesh wrapper API. Underscore-prefixed
    helpers are internal implementation details.

    Attributes
    ----------
    mesh : dolfinx.mesh.Mesh or None
        Generated DOLFINx mesh returned by the selected generator.
    cell_tags : dolfinx.mesh.MeshTags or None
        Cell-wise tags returned by the generator.  These typically encode
        named subdomains or material regions.
    facet_tags : dolfinx.mesh.MeshTags or None
        Facet-wise tags returned by the generator.  These typically
        encode named boundaries used for boundary conditions.

    Notes
    -----
    The selected generator is resolved from ``params.MeshType`` through
    the shared mesh-generator registry.  Generators return a structured
    :class:`GeneratedMeshResult`, and this mix-in copies those artifacts
    onto the composite mesh object.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, params: Any) -> None:
        # Populated by the selected mesh generator with the final DOLFINx mesh.
        self.mesh = None
        # Cell-wise MeshTags identifying named domains / subdomains.
        self.cell_tags = None
        # Facet-wise MeshTags identifying named boundaries.
        self.facet_tags = None

        generator = MeshGeneratorBase.get_generator(params.MeshType)
        generated = generator.generate(params)

        self.mesh = generated.mesh
        self.cell_tags = generated.cell_tags
        self.facet_tags = generated.facet_tags
        self.domain_markers.clear()
        for name, tag in generated.domain_markers.items():
            self.add_domain_marker(name, tag)

        self.boundary_markers.clear()
        for name, tag in generated.boundary_markers.items():
            self.add_boundary_marker(name, tag)

        super().__init__(params)
