"""Named boundary and domain marker helpers for the composite mesh object.

Marker names may be hierarchical, using ``":"`` as separator. Each stored
marker (a *part*) maps one name such as ``"domain:left"`` or ``"top:centre"``
to one integer tag in the mesh tags. A *group* name such as ``"domain"`` or
``"top"`` is never stored itself: it resolves to the union of all parts whose
name starts with ``"domain:"`` or ``"top:"``. Single-domain meshes simply
store plain names such as ``"domain"`` and ``"top"`` as parts.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping
import numpy as np
from dolfinx import la
from mpi4py import MPI
from femcubes.core.utils.mpi_utils import mprint


MARKER_SEPARATOR = ":"


class _MeshGroups:
    """Mix-in for managing named boundary and domain markers.

    Maintains two dictionaries -- :attr:`boundary_markers` and
    :attr:`domain_markers` -- that map human-readable names to the
    integer tags stored in the ``MeshTags`` objects produced by the
    mesh generator. All lookup methods also accept group names, which
    resolve to every stored part below them (e.g. ``"top"`` covers
    ``"top:left"`` and ``"top:right"``).

    Attributes
    ----------
    boundary_markers : dict
        ``{name: int}`` for boundary facet groups.
    domain_markers : dict
        ``{name: int}`` for cell/domain groups.

    Public methods are part of the mesh wrapper API. Underscore-prefixed
    helpers are internal implementation details.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, params: Any) -> None:
        # Initialize boundary and domain name mappings
        self.boundary_markers: Dict[str, int] = {}
        self.domain_markers: Dict[str, int] = {}
        
        super().__init__(params)
    
    def add_boundary_marker(self, name: str, tag: int) -> None:
        """Register or replace one named boundary marker.

        Parameters
        ----------
        name : str
            Human-readable boundary name such as ``"top"`` or ``"left"``,
            or a part such as ``"top:centre"``. A name cannot be both a
            part and the group of other parts.
        tag : int
            Integer tag stored in :attr:`facet_tags` for that boundary.
        """
        _validate_marker_name(self.boundary_markers, name, "Boundary")
        self.boundary_markers[name] = tag
    
    def add_domain_marker(self, name: str, tag: int) -> None:
        """Register or replace one named domain marker.

        Parameters
        ----------
        name : str
            Human-readable domain name such as ``"domain"`` or ``"fluid"``,
            or a part such as ``"domain:left"``. A name cannot be both a
            part and the group of other parts.
        tag : int
            Integer tag stored in :attr:`cell_tags` for that domain.
        """
        _validate_marker_name(self.domain_markers, name, "Domain")
        self.domain_markers[name] = tag
    
    def get_boundary_tag(self, name: str) -> int:
        """Return the integer tag associated with one named boundary.

        Parameters
        ----------
        name : str
            Boundary name to resolve.

        Returns
        -------
        int
            Integer tag associated with ``name`` in :attr:`facet_tags`.

        Raises
        ------
        KeyError
            If the requested boundary name is not registered.
        ValueError
            If ``name`` is a group covering more than one part; use
            :meth:`get_boundary_tags` for those.
        """
        return _single_marker_tag(self.boundary_markers, name, "Boundary", "get_boundary_tags")
    
    def get_domain_tag(self, name: str) -> int:
        """Return the integer tag associated with one named domain.

        Parameters
        ----------
        name : str
            Domain name to resolve.

        Returns
        -------
        int
            Integer tag associated with ``name`` in :attr:`cell_tags`.

        Raises
        ------
        KeyError
            If the requested domain name is not registered.
        ValueError
            If ``name`` is a group covering more than one part; use
            :meth:`get_domain_tags` for those.
        """
        return _single_marker_tag(self.domain_markers, name, "Domain", "get_domain_tags")

    def get_boundary_tags(self, name: str) -> tuple[int, ...]:
        """Return all integer tags covered by one boundary part or group.

        The result can be passed directly to a UFL measure, e.g.
        ``ds(mesh.get_boundary_tags("top"))``.

        Raises
        ------
        KeyError
            If ``name`` is neither a registered part nor a group of parts.
        """
        return _resolve_marker_tags(self.boundary_markers, name, "Boundary")

    def get_domain_tags(self, name: str) -> tuple[int, ...]:
        """Return all integer tags covered by one domain part or group.

        The result can be passed directly to a UFL measure, e.g.
        ``dx(mesh.get_domain_tags("domain"))``.

        Raises
        ------
        KeyError
            If ``name`` is neither a registered part nor a group of parts.
        """
        return _resolve_marker_tags(self.domain_markers, name, "Domain")
    
    def get_boundary_facets(self, name: str) -> np.ndarray:
        """Return the facet indices belonging to one named boundary.

        Parameters
        ----------
        name : str
            Registered boundary name, or a group of boundary parts.

        Returns
        -------
        numpy.ndarray
            Array of local facet indices returned by
            ``self.facet_tags.find(tag)``, merged over all tags of a group.
        """
        return _find_tagged_entities(self.facet_tags, self.get_boundary_tags(name))

    def get_domain_cells(self, name: str) -> np.ndarray:
        """Return the cell indices belonging to one named domain.

        Parameters
        ----------
        name : str
            Registered domain name, or a group of domain parts.

        Returns
        -------
        numpy.ndarray
            Array of local cell indices returned by
            ``self.cell_tags.find(tag)``, merged over all tags of a group.
        """
        return _find_tagged_entities(self.cell_tags, self.get_domain_tags(name))
    
    def Require_Boundary_On_Domain(self, boundary_name: str, domain: str) -> None:
        """Check that a named boundary bounds a (sub)domain.

        Every facet of ``boundary_name`` must be a facet of a cell of
        ``domain``; the full ``"domain"`` accepts every (registered)
        boundary. Boundary models use this to validate their boundary and
        their optional ``"domain"`` setting.
        Collective over the mesh communicator.

        Raises
        ------
        KeyError
            If ``boundary_name`` or ``domain`` is not a registered name.
        ValueError
            If part of the boundary does not bound ``domain``.
        """
        facets = self.get_boundary_facets(boundary_name)  # also rejects unknown boundary names
        if domain == "domain":
            return
        topology = self.mesh.topology
        tdim = topology.dim
        cell_map = topology.index_map(tdim)
        in_domain = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=bool)
        in_domain[self.get_domain_cells(domain)] = True
        topology.create_connectivity(tdim - 1, tdim)
        facet_to_cell = topology.connectivity(tdim - 1, tdim)

        # A facet's cells may sit on different ranks: accumulate on the facet owner.
        touches = la.vector(topology.index_map(tdim - 1), dtype=np.int32)
        touches.array[facets] = [in_domain[facet_to_cell.links(facet)].any() for facet in facets]
        touches.scatter_reverse(la.InsertMode.add)
        owned = facets[facets < topology.index_map(tdim - 1).size_local]
        outside = bool(np.any(touches.array[owned] == 0))
        if self.mesh.comm.allreduce(outside, op=MPI.LOR):
            raise ValueError(f"Boundary '{boundary_name}' does not bound domain '{domain}'.")

    def list_boundaries(self) -> Dict[str, int]:
        """Return a copy of the boundary-name to tag mapping."""
        return self.boundary_markers.copy()
    
    def list_domains(self) -> Dict[str, int]:
        """Return a copy of the domain-name to tag mapping."""
        return self.domain_markers.copy()

    def boundary_names(self) -> list[str]:
        """Return boundary names in registration order, each group directly before its first part."""
        return _marker_names(self.boundary_markers)

    def domain_names(self) -> list[str]:
        """Return domain names in registration order, each group directly before its first part."""
        return _marker_names(self.domain_markers)
    
    def print_mesh_groups(self) -> None:
        """Print registered domain and boundary names together with their tags."""
        mprint("\n=== Mesh Groups ===")
        
        if self.domain_markers:
            mprint("\nDomains:")
            self._print_marker_names(self.domain_markers)
        
        if self.boundary_markers:
            mprint("\nBoundaries:")
            self._print_marker_names(self.boundary_markers)
        
        mprint("==================\n")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_marker_names(markers: Mapping[str, int]) -> None:
        """PRIVATE: Print parts with their tag and groups with their tag list."""
        for name in _marker_names(markers):
            indent = "  " * (name.count(MARKER_SEPARATOR) + 1)
            if name in markers:
                mprint(f"{indent}{name}: tag={markers[name]}")
            else:
                mprint(f"{indent}{name}: tags={list(_resolve_marker_tags(markers, name, 'Domain'))}")


# ----------------------------------------------------------------------
# Internal Helpers: Name Resolution
# ----------------------------------------------------------------------
# Shared with the refinement artifacts, which expose the same name API.


def _marker_names(markers: Mapping[str, int]) -> list[str]:
    """PRIVATE: Return group and part names, each group before its first part."""
    names = []
    for name in markers:
        parts = name.split(MARKER_SEPARATOR)
        for depth in range(1, len(parts) + 1):
            prefix = MARKER_SEPARATOR.join(parts[:depth])
            if prefix not in names:
                names.append(prefix)
    return names


_KIND_PLURALS = {"Boundary": "boundaries", "Domain": "domains"}


def _resolve_marker_tags(markers: Mapping[str, int], name: str, kind: str) -> tuple[int, ...]:
    """PRIVATE: Resolve a part or group name to the tags it covers."""
    if name in markers:
        return (markers[name],)
    prefix = name + MARKER_SEPARATOR
    tags = tuple(tag for part, tag in markers.items() if part.startswith(prefix))
    if not tags:
        raise KeyError(f"{kind} '{name}' not found. Available {_KIND_PLURALS[kind]}: {_marker_names(markers)}")
    return tags


def _single_marker_tag(markers: Mapping[str, int], name: str, kind: str, plural_method: str) -> int:
    """PRIVATE: Resolve a name that must cover exactly one stored part."""
    tags = _resolve_marker_tags(markers, name, kind)
    if len(tags) != 1:
        raise ValueError(
            f"{kind} '{name}' covers {len(tags)} parts with tags {list(tags)}; "
            f"use {plural_method}() to resolve all of them."
        )
    return tags[0]


def _find_tagged_entities(mesh_tags, tags: tuple[int, ...]) -> np.ndarray:
    """PRIVATE: Return the sorted local entities carrying any of ``tags``."""
    if len(tags) == 1:
        return mesh_tags.find(tags[0])
    return np.unique(np.concatenate([mesh_tags.find(tag) for tag in tags]))


def _validate_marker_name(markers: Mapping[str, int], name: str, kind: str) -> None:
    """PRIVATE: Reject empty name components and part/group name clashes."""
    if not name or any(not part for part in name.split(MARKER_SEPARATOR)):
        raise ValueError(f"{kind} name '{name}' must not be empty or contain empty ':'-separated parts.")
    prefix = name + MARKER_SEPARATOR
    for existing in markers:
        if existing.startswith(prefix) or name.startswith(existing + MARKER_SEPARATOR):
            raise ValueError(
                f"{kind} name '{name}' clashes with registered part '{existing}': "
                "a name cannot be both a part and a group of other parts."
            )
