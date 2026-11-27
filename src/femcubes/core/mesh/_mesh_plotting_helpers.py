"""Neutral mesh-geometry preparation and PyVista actor helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pyvista as pv

from femcubes.core.mesh._mesh_groups import MARKER_SEPARATOR as _MARKER_SEPARATOR


_PARTITION_COLORS = (
    "lightblue",
    "lightgreen",
    "lightyellow",
    "lightcoral",
    "plum",
    "peachpuff",
    "lightcyan",
    "pink",
)

# Strong, mutually distinct colors so that boundary overlays stay
# distinguishable from each other and from the pale partition colors.
# Multi-domain meshes easily have more than ten boundary parts.
_BOUNDARY_COLORS = (
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "yellow",
    "saddlebrown",
    "lime",
    "navy",
    "deeppink",
    "teal",
    "olive",
    "darkviolet",
    "chocolate",
    "deepskyblue",
    "maroon",
    "darkgreen",
    "black",
)


def add_mesh_partitions(
    plotter,
    all_topologies: Sequence[np.ndarray],
    all_cell_types: Sequence[np.ndarray],
    all_geometries: Sequence[np.ndarray],
    dimension: int,
    backend=None,
) -> None:
    """Add gathered rank-local mesh geometry using stable partition colors."""

    backend = pv if backend is None else backend
    for rank_index, (topology, cell_types, geometry) in enumerate(
        zip(all_topologies, all_cell_types, all_geometries)
    ):
        rank_grid = backend.UnstructuredGrid(topology, cell_types, geometry)
        rank_color = _PARTITION_COLORS[rank_index % len(_PARTITION_COLORS)]
        plotter.add_mesh(
            rank_grid,
            show_edges=True,
            color=rank_color,
            edge_color="black",
            opacity=0.5 if dimension == 2 else 1.0,
            line_width=1,
            label=f"Rank {rank_index}",
        )


def configure_mesh_camera(plotter, dimension: int) -> None:
    """Apply the common axes, view, and legend configuration."""

    plotter.show_axes()
    if dimension == 2:
        plotter.view_xy()
    else:
        plotter.view_isometric()
    plotter.add_legend()


def collect_boundary_plot_data(
    mesh,
    facet_tags,
    boundary_markers: Mapping[str, int] | None,
    tdim: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Return local named-boundary geometry from explicit mesh/tag inputs."""

    local_boundary_data: dict[str, dict[str, Any]] = {}
    if facet_tags is None or not boundary_markers:
        return local_boundary_data
    if tdim is None:
        tdim = mesh.topology.dim

    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(tdim, 0)

    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    cell_to_vertex = mesh.topology.connectivity(tdim, 0)
    geom_dofmap = mesh.geometry.dofmap

    for name, tag in boundary_markers.items():
        facets = facet_tags.find(tag)
        geom_point_to_index: dict[int, int] = {}
        boundary_points: list[np.ndarray] = []
        boundary_lines: list[list[int]] = []
        boundary_faces: list[list[int]] = []

        for facet in facets:
            cells = facet_to_cell.links(facet)
            if len(cells) == 0:
                continue

            cell = cells[0]
            geom_dofs = geom_dofmap[cell]
            topology_vertices = cell_to_vertex.links(cell)
            facet_topology_vertices = facet_to_vertex.links(facet)
            facet_indices: list[int] = []

            for topology_vertex in facet_topology_vertices:
                local_position = np.where(topology_vertices == topology_vertex)[0]
                if len(local_position) == 0:
                    continue

                geometry_dof = int(geom_dofs[local_position[0]])
                if geometry_dof not in geom_point_to_index:
                    geom_point_to_index[geometry_dof] = len(boundary_points)
                    boundary_points.append(mesh.geometry.x[geometry_dof].copy())
                facet_indices.append(geom_point_to_index[geometry_dof])

            if len(facet_indices) == 2:
                boundary_lines.append([2, facet_indices[0], facet_indices[1]])
            elif len(facet_indices) >= 3:
                boundary_faces.append([len(facet_indices), *facet_indices])

        if boundary_points:
            local_boundary_data[name] = {
                "points": np.asarray(boundary_points),
                "lines": boundary_lines,
                "faces": boundary_faces,
            }

    return local_boundary_data


def add_named_boundary_overlays(
    plotter,
    all_boundary_data: Sequence[Mapping[str, dict[str, Any]]],
    boundary_markers: Mapping[str, int] | None,
    backend=None,
) -> None:
    """Add gathered named-boundary fragments and labels to a plotter."""

    if not boundary_markers:
        return

    colors = boundary_color_map(boundary_markers)
    for name, tag in boundary_markers.items():
        color = colors.get(name, "gray")
        all_centers = []
        for rank_index, rank_boundary_data in enumerate(all_boundary_data):
            data = rank_boundary_data.get(name)
            if data is None:
                continue

            points = data["points"]
            lines = data["lines"]
            faces = data["faces"]
            if len(points) == 0 or not (faces or lines):
                continue

            boundary_mesh = build_boundary_mesh(points, lines, faces, backend=backend)
            if boundary_mesh is not None:
                label = f"{name} (tag={tag})" if rank_index == 0 else None
                plotter.add_mesh(
                    boundary_mesh,
                    color=color,
                    line_width=5 if lines else 1,
                    opacity=1.0 if lines else 0.35,
                    label=label,
                )
            all_centers.append(np.mean(points, axis=0))

        if all_centers:
            center = np.mean(all_centers, axis=0)
            plotter.add_point_labels(
                [center],
                [f"{name} (tag={tag})"],
                font_size=12,
                point_size=8,
                text_color=color,
                point_color=color,
                always_visible=True,
                bold=True,
            )


def add_domain_label(
    plotter,
    all_geometries: Sequence[np.ndarray],
    domain_markers: Mapping[str, int] | None,
) -> None:
    """Add a named domain label at the gathered geometry center."""

    mesh_center = np.mean(np.concatenate(all_geometries), axis=0)
    plotter.add_point_labels(
        [mesh_center],
        [domain_label_text(domain_markers)],
        font_size=14,
        point_size=10,
        text_color="darkgray",
        point_color="darkgray",
        always_visible=True,
        bold=True,
    )


def build_boundary_mesh(points, lines, faces, backend=None):
    """Build a PyVista boundary surface or polyline from connectivity."""

    backend = pv if backend is None else backend
    if faces:
        return backend.PolyData(points, faces=np.hstack(faces))
    if lines:
        return backend.PolyData(points, lines=np.hstack(lines))
    return None


def boundary_color_map(boundary_markers: Mapping[str, int]) -> dict[str, str]:
    """Assign deterministic display colors to named boundaries."""

    return {
        name: _BOUNDARY_COLORS[index % len(_BOUNDARY_COLORS)]
        for index, name in enumerate(boundary_markers)
    }


def domain_label_text(domain_markers: Mapping[str, int] | None) -> str:
    """Return the stable label for a mesh domain.

    Multi-domain meshes are labelled by the group of their first part, e.g.
    ``domain (tags=1, 2, 3)`` for ``domain:left``, ``domain:centre`` and
    ``domain:right``.
    """

    if not domain_markers:
        return "Domain"
    name = next(iter(domain_markers)).split(_MARKER_SEPARATOR)[0]
    tags = [tag for part, tag in domain_markers.items() if part == name or part.startswith(name + _MARKER_SEPARATOR)]
    if len(tags) == 1 and name in domain_markers:
        return f"{name} (tag={tags[0]})"
    return f"{name} (tags={', '.join(str(tag) for tag in tags)})"
