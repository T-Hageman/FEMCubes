"""Persistent mesh-plot tests and a manually runnable refinement demo.

Run the visual demo from the repository root with::

    python femcubes/core/mesh/tests/test_mesh_refinement_visualization.py

The demo moves a synthetic fracture-like indicator across a rectangle and
updates the ordinary FEMCubes mesh plot after each successful refinement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from dolfinx import mesh as dmesh
from mpi4py import MPI
import numpy as np
import pytest

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.refinement import (
    BufferedCellSelection,
    CellIndicatorResult,
    MeshMarkerState,
    RefinementConfig,
    expand_refinement_buffer,
    refine_mesh_topology,
    transfer_mesh_markers,
)


@pytest.fixture
def tracked_mesh_pyvista(monkeypatch):
    """Replace PyVista with a lightweight persistent-plot interaction tracker."""

    import femcubes.core.mesh._mesh_plotting as mesh_plotting
    import femcubes.core.mesh._mesh_plotting_helpers as plot_helpers

    class TrackingPlotter:
        instances = []
        fail_open_movie = False
        fail_write_frame = False

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.meshes = []
            self.labels = []
            self.texts = []
            self.shows = []
            self.opened_movies = []
            self.written_frames = 0
            self.update_calls = 0
            self.clear_calls = 0
            self.close_calls = 0
            self.disable_calls = 0
            self.axes_calls = 0
            self.xy_calls = 0
            self.isometric_calls = 0
            self.legend_calls = 0
            TrackingPlotter.instances.append(self)

        def add_mesh(self, *args, **kwargs):
            self.meshes.append((args, kwargs))

        def add_point_labels(self, *args, **kwargs):
            self.labels.append((args, kwargs))

        def add_text(self, *args, **kwargs):
            self.texts.append((args, kwargs))

        def show_axes(self):
            self.axes_calls += 1

        def view_xy(self):
            self.xy_calls += 1

        def view_isometric(self):
            self.isometric_calls += 1

        def add_legend(self):
            self.legend_calls += 1

        def show(self, *args, **kwargs):
            self.shows.append((args, kwargs))

        def disable(self):
            self.disable_calls += 1

        def open_movie(self, path, framerate):
            if TrackingPlotter.fail_open_movie:
                raise RuntimeError("movie open failed")
            self.opened_movies.append((path, framerate))

        def write_frame(self):
            if TrackingPlotter.fail_write_frame:
                raise RuntimeError("frame write failed")
            self.written_frames += 1

        def update(self):
            self.update_calls += 1

        def clear(self):
            self.clear_calls += 1
            self.meshes.clear()
            self.labels.clear()
            self.texts.clear()

        def close(self):
            self.close_calls += 1

    backend = SimpleNamespace(
        Plotter=TrackingPlotter,
        UnstructuredGrid=lambda topology, cell_types, geometry: (
            "grid",
            topology,
            cell_types,
            geometry,
        ),
        PolyData=lambda points, **kwargs: ("polydata", points, kwargs),
        OFF_SCREEN=True,
    )
    monkeypatch.setattr(mesh_plotting, "pv", backend)
    monkeypatch.setattr(plot_helpers, "pv", backend)
    yield TrackingPlotter


def _selection(mesh, marked_mask, size_mask=None, buffer_layers=0):
    tdim = mesh.topology.dim
    cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
    marked_mask = np.asarray(marked_mask, dtype=bool)
    if size_mask is None:
        size_mask = np.ones(cells.size, dtype=bool)
    size_mask = np.asarray(size_mask, dtype=bool)
    indicator = CellIndicatorResult(
        cells=cells,
        values=marked_mask.astype(float),
        diameters=mesh.h(tdim, cells),
        threshold_mask=marked_mask.copy(),
        size_mask=size_mask,
        marked_mask=marked_mask & size_mask,
    )
    return expand_refinement_buffer(SimpleNamespace(mesh=mesh), indicator, buffer_layers, target_h=0.1)


def _one_cell_refinement(mesh, marker_state=None):
    cell_count = mesh.topology.index_map(mesh.topology.dim).size_local
    marked = np.zeros(cell_count, dtype=bool)
    marked[cell_count // 2] = True
    artifacts = refine_mesh_topology(mesh, _selection(mesh, marked))
    if marker_state is not None:
        artifacts = transfer_mesh_markers(marker_state, artifacts)
    return artifacts


def _remesh_movie_path(mesh_wrapper) -> Path:
    folder = getattr(mesh_wrapper.params, "movie_folder", None)
    if folder is None:
        folder = Path(getattr(mesh_wrapper.params, "output_dir", "Results")) / "movies"
    return Path(folder) / "mesh_refinement.mp4"


def test_mesh_plot_can_be_disabled_without_creating_a_plotter(
    refinement_mesh,
    tracked_mesh_pyvista,
):
    result = refinement_mesh.plot_mesh(show_window=False)

    assert not result.visible_updated
    assert not result.movie_frame_written
    assert tracked_mesh_pyvista.instances == []


@pytest.mark.parametrize(
    ("options", "exception", "message"),
    [
        ({"show_window": 1}, TypeError, "show_window must be a bool"),
        ({"movie_path": ""}, ValueError, "movie_path must not be empty"),
        (
            {"movie_path": "mesh.mp4", "movie_fps": 0},
            ValueError,
            "movie_fps must be an integer",
        ),
        ({"annotation": 3}, TypeError, "annotation must be a string"),
    ],
)
def test_persistent_mesh_plot_controls_reject_invalid_inputs(
    refinement_mesh,
    tracked_mesh_pyvista,
    options,
    exception,
    message,
):
    with pytest.raises(exception, match=message):
        refinement_mesh.plot_mesh(**options)

    assert tracked_mesh_pyvista.instances == []


def test_refined_topology_updates_existing_mesh_plotter_in_place(
    refinement_mesh,
    tracked_mesh_pyvista,
):
    initial = refinement_mesh.plot_mesh(annotation="Mesh epoch 0")
    marker_state = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    refined = _one_cell_refinement(refinement_mesh.mesh, marker_state)

    updated = refinement_mesh.plot_mesh(
        refined.refined_mesh,
        facet_tags=refined.facet_tags,
        boundary_markers=refined.boundary_markers,
        domain_markers=refined.domain_markers,
        annotation=f"Mesh epoch 1\nGlobal cells: {refined.report.global_cells_after}",
    )

    assert initial.visible_updated
    assert updated.visible_updated
    assert len(tracked_mesh_pyvista.instances) == 1
    plotter = tracked_mesh_pyvista.instances[0]
    assert plotter.clear_calls == 1
    assert plotter.update_calls == 1
    assert len(plotter.shows) == 1
    assert plotter.close_calls == 0
    assert any(
        f"Global cells: {refined.report.global_cells_after}" in args[0]
        for args, _kwargs in plotter.texts
    )
    label_text = [text for args, _kwargs in plotter.labels for text in args[1]]
    for boundary_name in refinement_mesh.boundary_names():
        assert any(boundary_name in text for text in label_text)


def test_animation_only_refinement_reuses_one_offscreen_mesh_plotter(
    refinement_mesh,
    tmp_path,
    tracked_mesh_pyvista,
):
    refinement_mesh.params.output_dir = str(tmp_path)
    refinement_mesh.params.movie_fps = 17
    movie_path = _remesh_movie_path(refinement_mesh)
    initial = refinement_mesh.plot_mesh(
        annotation="Mesh epoch 0",
        show_window=False,
        movie_path=movie_path,
    )
    marker_state = MeshMarkerState.from_mesh_wrapper(refinement_mesh)
    refined = _one_cell_refinement(refinement_mesh.mesh, marker_state)

    updated = refinement_mesh.plot_mesh(
        refined.refined_mesh,
        facet_tags=refined.facet_tags,
        boundary_markers=refined.boundary_markers,
        domain_markers=refined.domain_markers,
        annotation="Mesh epoch 1",
        show_window=False,
        movie_path=movie_path,
    )

    assert not initial.visible_updated
    assert not updated.visible_updated
    assert initial.movie_frame_written
    assert updated.movie_frame_written
    assert len(tracked_mesh_pyvista.instances) == 1
    plotter = tracked_mesh_pyvista.instances[0]
    assert plotter.kwargs["off_screen"] is True
    assert plotter.opened_movies == [(str(movie_path), 17)]
    assert plotter.written_frames == 2
    assert plotter.clear_calls == 1
    assert plotter.update_calls == 1


def test_noop_refinement_workflow_does_not_update_or_append_frame(
    refinement_mesh,
    tmp_path,
    tracked_mesh_pyvista,
):
    movie_path = tmp_path / "mesh_refinement.mp4"
    refinement_mesh.plot_mesh(show_window=False, movie_path=movie_path)
    mesh = refinement_mesh.mesh
    cell_count = mesh.topology.index_map(mesh.topology.dim).size_local
    no_op = refine_mesh_topology(mesh, _selection(mesh, np.zeros(cell_count, dtype=bool)))
    plotter = tracked_mesh_pyvista.instances[0]

    if no_op.report.occurred:
        refinement_mesh.plot_mesh(
            no_op.refined_mesh,
            show_window=False,
            movie_path=movie_path,
        )

    assert not no_op.report.occurred
    assert plotter.written_frames == 1
    assert plotter.clear_calls == 0
    assert plotter.update_calls == 0


@pytest.mark.parametrize(
    ("failure", "warning"),
    [("open", "Could not open mesh movie"), ("write", "Could not write mesh movie frame")],
)
def test_mesh_movie_failures_warn_without_closing_persistent_plot(
    refinement_mesh,
    tmp_path,
    tracked_mesh_pyvista,
    failure,
    warning,
):
    tracked_mesh_pyvista.fail_open_movie = failure == "open"
    tracked_mesh_pyvista.fail_write_frame = failure == "write"

    with pytest.warns(RuntimeWarning, match=warning):
        result = refinement_mesh.plot_mesh(
            show_window=False,
            movie_path=tmp_path / "mesh_refinement.mp4",
        )

    assert not result.movie_frame_written
    assert len(tracked_mesh_pyvista.instances) == 1
    assert tracked_mesh_pyvista.instances[0].close_calls == 0


def test_close_plots_finalizes_the_persistent_mesh_plotter(
    refinement_mesh,
    tmp_path,
    tracked_mesh_pyvista,
):
    refinement_mesh.plot_mesh(
        show_window=True,
        movie_path=tmp_path / "mesh_refinement.mp4",
    )
    plotter = tracked_mesh_pyvista.instances[0]

    refinement_mesh.Close_Plots()

    assert plotter.close_calls == 1
    assert refinement_mesh._mesh_plotter is None
    assert refinement_mesh._mesh_plot_movie_path is None


def run_visual_refinement_demo(
    output_dir: str | Path = "Results/mesh_refinement_demo",
    *,
    steps: int = 6,
    animation_only: bool = False,
) -> Path:
    """Update one mesh plot through refinements driven by a moving field."""

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("The manual topology demo currently runs in serial only.")
    if steps < 1:
        raise ValueError("steps must be at least one.")

    class DemoParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 2.0
        Ly = 1.0
        dx = 0.25
        dim = 2
        movie_fps = 2
        Interactive_Plots = True

    from femcubes.core.mesh.mesh import Mesh

    output_dir = Path(output_dir)
    mesh_wrapper = Mesh(DemoParams())
    mesh_wrapper.params.output_dir = str(output_dir)
    config = RefinementConfig(
        replot_on_refinement=not animation_only,
        save_remesh_animations=True,
    )
    movie_path = _remesh_movie_path(mesh_wrapper)
    mesh = mesh_wrapper.mesh
    marker_state = MeshMarkerState.from_mesh_wrapper(mesh_wrapper)
    mesh_wrapper.plot_mesh(
        annotation="Mesh epoch 0\nGlobal cells: "
        f"{mesh.topology.index_map(mesh.topology.dim).size_global}",
        show_window=config.replot_on_refinement,
        movie_path=movie_path if config.save_remesh_animations else None,
    )

    target_h = 0.005
    for epoch in range(1, steps + 1):
        tdim = mesh.topology.dim
        cells = np.arange(mesh.topology.index_map(tdim).size_local, dtype=np.int32)
        centers = dmesh.compute_midpoints(mesh, tdim, cells)
        crack_tip_x = 0.0 + 2.0 * epoch / steps
        values = np.exp(
            -((centers[:, 0] - crack_tip_x) / 0.025) ** 2
            -((centers[:, 1] - 0.5) / 0.025) ** 2
        )
        diameters = mesh.h(tdim, cells)
        threshold_mask = values >= 0.1
        size_mask = diameters > target_h + 64.0 * np.finfo(float).eps
        indicator = CellIndicatorResult(
            cells=cells,
            values=values,
            diameters=diameters,
            threshold_mask=threshold_mask,
            size_mask=size_mask,
            marked_mask=threshold_mask & size_mask,
        )
        selection: BufferedCellSelection = expand_refinement_buffer(
            SimpleNamespace(mesh=mesh),
            indicator,
            buffer_layers=1, target_h=target_h,
        )
        artifacts = refine_mesh_topology(mesh, selection, RefinementConfig(target_h=target_h))
        if not artifacts.report.occurred:
            continue
        artifacts = transfer_mesh_markers(marker_state, artifacts)
        mesh = artifacts.refined_mesh
        mesh_wrapper.plot_mesh(
            mesh,
            facet_tags=artifacts.facet_tags,
            boundary_markers=artifacts.boundary_markers,
            domain_markers=artifacts.domain_markers,
            annotation=(
                f"Mesh epoch {epoch}\n"
                f"Global cells: {artifacts.report.global_cells_after}"
            ),
            show_window=config.replot_on_refinement,
            movie_path=movie_path if config.save_remesh_animations else None,
        )
        marker_state = MeshMarkerState.from_refined_artifacts(artifacts)
        print(
            f"Epoch {epoch}: {artifacts.report.global_cells_before} -> "
            f"{artifacts.report.global_cells_after} cells"
        )

    if not animation_only:
        try:
            input("Refinement demo complete. Press Enter to close the mesh plot...")
        except EOFError:
            pass
    mesh_wrapper.Close_Plots()
    print(f"Remesh animation: {movie_path}")
    return movie_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=run_visual_refinement_demo.__doc__)
    parser.add_argument("--output-dir", default="Results/mesh_refinement_demo")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--animation-only",
        action="store_true",
        help="Render off-screen without opening the persistent mesh window.",
    )
    arguments = parser.parse_args()
    run_visual_refinement_demo(
        arguments.output_dir,
        steps=arguments.steps,
        animation_only=arguments.animation_only,
    )
