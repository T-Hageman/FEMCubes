"""Base class for mesh generators with auto-registration."""
from dataclasses import dataclass, field
from typing import Any, Dict, Type

import gmsh
import numpy as np
from dolfinx.io.gmsh import model_to_mesh as _dolfinx_model_to_mesh
from dolfinx.mesh import GhostMode, create_cell_partitioner

from femcubes.core._discover import load_plugins


# Shared-facet ghost mode is required for DG interior-facet integrals (dS)
# across MPI partition boundaries.  Using it for all meshes is safe: it is a
# superset of shared-vertex mode and incurs only a small extra memory cost.
_shared_facet_partitioner = create_cell_partitioner(GhostMode.shared_facet)


def model_to_mesh(model, comm, rank, **kwargs):
    """Wrapper around ``dolfinx.io.gmsh.model_to_mesh`` that always enables
    ``GhostMode.shared_facet`` so interior-facet DG integrals work in parallel.
    """
    kwargs.setdefault("partitioner", _shared_facet_partitioner)
    return _dolfinx_model_to_mesh(model, comm, rank, **kwargs)



@dataclass
class GeneratedMeshResult:
    """Structured return value for mesh generators."""

    mesh: Any
    cell_tags: Any = None
    facet_tags: Any = None
    domain_markers: Dict[str, int] = field(default_factory=dict)
    boundary_markers: Dict[str, int] = field(default_factory=dict)


class MeshGeneratorBase:
    """Base class for all mesh generators.

    Subclasses must set ``MESH_TYPE`` to a non-empty string that matches
    the ``params.MeshType`` value, and implement :meth:`generate`.

    Public methods are part of the generator interface. Underscore-prefixed
    helpers are internal implementation details.
    """

    MESH_TYPE: str = ""

    # Auto-populated registry: mesh type string -> generator class
    _registry: Dict[str, Type["MeshGeneratorBase"]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_generator(cls, type_string: str) -> "MeshGeneratorBase":
        """Look up and instantiate a mesh generator by its type string."""
        if type_string not in cls._registry:
            # The type may be provided by an installed plugin.
            load_plugins()
        if type_string not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise NotImplementedError(
                f"Mesh type '{type_string}' not implemented. Available: {available}"
            )
        return cls._registry[type_string]()

    def generate(self, params: Any) -> GeneratedMeshResult:
        """Generate and return the mesh artifacts for ``params``."""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        mesh_type = getattr(cls, "MESH_TYPE", "")
        if mesh_type:
            MeshGeneratorBase._registry[mesh_type] = cls

    @staticmethod
    def _exact_bounding_box(dim: int, tag: int, samples_per_curve: int = 65) -> tuple[float, ...]:
        """PRIVATE: Return an unpadded ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

        ``gmsh.model.occ.getBoundingBox`` pads every box by the OCC tolerance
        (~1e-7). That is not negligible compared to small geometries, so
        planarity and position checks based on it misclassify entities.
        This helper samples the entity's curves instead: straight edges and
        planar faces are bounded exactly, while extents of curved edges are
        accurate to the sampling resolution (relative error
        ``~1 - cos(pi / samples_per_curve)``).

        Surfaces and volumes are bounded through their boundary curves.
        """
        points = MeshGeneratorBase._sample_entity_points(dim, tag, samples_per_curve)
        return (*points.min(axis=0), *points.max(axis=0))

    @staticmethod
    def _sample_entity_points(dim: int, tag: int, samples_per_curve: int):
        """PRIVATE: Sample points on a gmsh entity via its (boundary) curves."""
        if dim == 0:
            return np.asarray(gmsh.model.getValue(0, tag, []), dtype=float).reshape(1, 3)
        if dim == 1:
            (t_min,), (t_max,) = gmsh.model.getParametrizationBounds(1, tag)
            parameters = np.linspace(t_min, t_max, samples_per_curve)
            return np.asarray(gmsh.model.getValue(1, tag, parameters), dtype=float).reshape(-1, 3)
        boundary = gmsh.model.getBoundary([(dim, tag)], combined=False, oriented=False)
        return np.concatenate([
            MeshGeneratorBase._sample_entity_points(sub_dim, abs(sub_tag), samples_per_curve)
            for sub_dim, sub_tag in boundary
        ])

    @staticmethod
    def _result_from_gmsh(
        result,
        *,
        domain_markers: Dict[str, int] | None = None,
        boundary_markers: Dict[str, int] | None = None,
    ) -> GeneratedMeshResult:
        """PRIVATE: Convert ``model_to_mesh`` output into a structured generator result."""
        return GeneratedMeshResult(
            mesh=result[0],
            cell_tags=result[1] if len(result) > 1 else None,
            facet_tags=result[2] if len(result) > 2 else None,
            domain_markers=dict(domain_markers or {}),
            boundary_markers=dict(boundary_markers or {}),
        )
