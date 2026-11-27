"""Interactive plotting helpers for the composite mesh class."""

from __future__ import annotations

import os as _os
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, Iterable, Optional
import warnings

import numpy as np
import pyvista as pv
from dolfinx import mesh as dmesh
from dolfinx.plot import vtk_mesh

from femcubes.core.utils.mpi_utils import comm, mprint, rank
from femcubes.core.mesh import _mesh_plotting_helpers as _mesh_plot_helpers


_INTERFACE_LINE_WIDTH = 6.0  # Line width of fields living on interfaces/boundaries
_VTK_LINE_CELL_TYPES = (3, 21, 35, 68)  # VTK line, quadratic edge, cubic line, Lagrange curve

_ICON_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(__file__)), "Icon.png")


@dataclass
class _FieldPlotState:
    """PRIVATE: Cached rendering state for one plotted solution field."""

    plotter: Any
    grid: Any
    actor: Any
    time_text: Any
    level_set_contour_actor: Any = None
    mesh_epoch: int = 0
    isosurface_actor: Any = None    # Isosurface windows (Plot_Isosurfaces) only
    isosurface_field: str | None = None
    isosurface_levels: tuple = ()
    isosurface_title: str = ""


@dataclass
class _MeasurePlotState:
    """PRIVATE: Cached chart state for one plotted scalar measure pair."""

    plotter: Any
    chart: Any
    line: Any
    x: list[float]
    y: list[float]


@dataclass(frozen=True)
class MeshPlotResult:
    """Outcome of one collective mesh-plot update."""

    visible_updated: bool
    movie_frame_written: bool


class _MeshPlotting:
    """Mix-in for interactive PyVista mesh and field visualisation.

    Provides methods for real-time plotting of the mesh geometry,
    solution fields, and time-history measures during a simulation.
    Plotting is primarily a debugging and development tool: current
    implementations gather rank-local mesh or field data to rank 0 and
    render from there.

    Attributes
    ----------
    default_plot_size : tuple of int
        Default ``(width, height)`` in pixels for plot windows.

    Public methods are part of the mesh wrapper API. Underscore-prefixed
    helpers are internal implementation details.
    """

    _icon_image = None  # lazily loaded vtkImageData (class-level cache)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, params: Any) -> None:
        self._field_plots: Dict[str, _FieldPlotState] = {}
        self._field_plots_epoch: int = int(getattr(self, "remesh_epoch", 0))
        self._measure_plots: Dict[tuple[str, str], _MeasurePlotState] = {}
        self._recording: bool = False
        self._recording_folder: str = ""
        self._recording_fps: int = 10
        self._frame_count: int = 0
        self._mesh_plotter = None
        self._mesh_plot_visible: bool = False
        self._mesh_plot_movie_path: str | None = None
        self._mesh_plot_movie_disabled: bool = False
        self._plot_mesh_data = None
        self._plot_mesh_data_epoch: int | None = None

        self.default_plot_size = (608, 512)

        super().__init__(params)

    def plot_mesh(
        self,
        mesh=None,
        *,
        facet_tags=None,
        boundary_markers=None,
        domain_markers=None,
        annotation: str | None = None,
        show_window: bool = True,
        movie_path: str | _os.PathLike | None = None,
        movie_fps: int | None = None,
    ) -> MeshPlotResult:
        """Create or update the persistent mesh plot and optional movie.

        With no arguments this plots the live mesh and its named boundaries,
        preserving the existing API. Later calls reuse the same root-rank
        plotter, rebuild its actors in place, and update the open window.

        A detached DOLFINx mesh and matching tag/name artifacts can be supplied
        without changing the live wrapper. ``movie_path`` records frames from
        this same persistent plotter; ``show_window=False`` makes its initial
        creation off-screen for animation-only use.
        """
        resolved_movie_fps = getattr(self.params, "movie_fps", 10) if movie_fps is None else movie_fps
        self._validate_mesh_plot_request(
            show_window,
            movie_path,
            resolved_movie_fps if movie_path is not None else movie_fps,
            annotation,
        )
        if not show_window and movie_path is None:
            return MeshPlotResult(False, False)

        use_live_artifacts = mesh is None
        plot_mesh = self.mesh if use_live_artifacts else mesh
        if use_live_artifacts:
            topology_data, cell_types_data, geometry_data, _ = self._build_plot_mesh_data()
            if facet_tags is None:
                facet_tags = getattr(self, "facet_tags", None)
            if boundary_markers is None:
                boundary_markers = getattr(self, "boundary_markers", None)
            if domain_markers is None:
                domain_markers = getattr(self, "domain_markers", None)
        else:
            topology_data, cell_types_data, geometry_data = vtk_mesh(
                plot_mesh,
                plot_mesh.topology.dim,
            )
            if boundary_markers is None:
                boundary_markers = {}
            if domain_markers is None:
                domain_markers = {}

        local_boundary_data = _mesh_plot_helpers.collect_boundary_plot_data(
            plot_mesh,
            facet_tags,
            boundary_markers,
        )

        all_topologies = comm.gather(topology_data, root=0)
        all_cell_types = comm.gather(cell_types_data, root=0)
        all_geometries = comm.gather(geometry_data, root=0)
        all_boundary_data = comm.gather(local_boundary_data, root=0)

        root_result = None
        if rank != 0:
            return comm.bcast(root_result, root=0)

        root_result = self._update_mesh_plotter(
            all_topologies,
            all_cell_types,
            all_geometries,
            all_boundary_data,
            dimension=plot_mesh.topology.dim,
            boundary_markers=boundary_markers,
            domain_markers=domain_markers,
            annotation=annotation,
            show_window=show_window,
            movie_path=None if movie_path is None else _os.fspath(movie_path),
            movie_fps=resolved_movie_fps,
        )
        return comm.bcast(root_result, root=0)

    def Close_Plots(self):
        """Close all open mesh, field, and measure plot windows."""
        if self._recording:
            self.StopRecording()

        for state in self._field_plots.values():
            self._close_plotter_safely(state.plotter)
        self._field_plots.clear()

        self._close_mesh_plotter_state()

        for state in self._measure_plots.values():
            self._close_plotter_safely(state.plotter)
        self._measure_plots.clear()

        close_reduced_order_plots = getattr(self, "_close_reduced_order_plots", None)
        if close_reduced_order_plots is not None:
            close_reduced_order_plots()

    def StartRecording(self, folder_path: str, fps: int = 10) -> None:
        """Begin recording field plots directly to MP4 videos.

        Existing field plots open one movie file each.  New field plots
        created while recording is active automatically open their own
        movie file as well.
        """
        if rank != 0:
            return

        _os.makedirs(folder_path, exist_ok=True)
        self._recording = True
        self._recording_folder = folder_path
        self._recording_fps = fps
        self._frame_count = 0

        for field_spec, state in self._field_plots.items():
            self._open_movie_safely(field_spec, state.plotter, folder_path, fps)

        start_reduced_order_recording = getattr(
            self,
            "_start_reduced_order_recording",
            None,
        )
        if start_reduced_order_recording is not None:
            start_reduced_order_recording(folder_path, fps)

        mprint(f"Recording started → {folder_path}")

    def StopRecording(self) -> None:
        """Stop recording and finalise all MP4 files."""
        if rank != 0:
            return
        for state in self._field_plots.values():
            self._finalize_plotter_movie_writer(state.plotter)
        for state in getattr(self, "_ro_field_plots", {}).values():
            self._finalize_plotter_movie_writer(state.plotter)
        self._recording = False
        mprint(f"Recording stopped ({self._frame_count} frames captured)")

    def SavePlotsToFolder(self, folder_path: str) -> None:
        """Save all currently open plot windows as PNG images.

        The mesh plot from :meth:`plot_mesh` is saved as ``mesh.png``; field
        and measure plots are named after their field or measure names.
        """
        if rank != 0:
            return

        _os.makedirs(folder_path, exist_ok=True)

        if self._mesh_plotter is not None:
            self._save_screenshot_safely(
                self._mesh_plotter,
                _os.path.join(folder_path, "mesh.png"),
                "mesh plot",
            )

        for field_spec, state in self._field_plots.items():
            safe_name = self._sanitize_plot_name(field_spec)
            filepath = _os.path.join(folder_path, f"{safe_name}.png")
            self._save_screenshot_safely(
                state.plotter,
                filepath,
                f"plot for '{field_spec}'",
            )

        for (x_name, y_name), state in self._measure_plots.items():
            safe_x = self._sanitize_plot_name(x_name)
            safe_y = self._sanitize_plot_name(y_name)
            filepath = _os.path.join(folder_path, f"{safe_y}_vs_{safe_x}.png")
            self._save_screenshot_safely(
                state.plotter,
                filepath,
                f"measure plot '{y_name} vs {x_name}'",
            )

        save_reduced_order_plots = getattr(self, "_save_reduced_order_plots", None)
        if save_reduced_order_plots is not None:
            save_reduced_order_plots(folder_path)

    def Plot_Fields(self):
        """Plot the fields listed in ``params.Plot_Fields``.

        Each field specification is first resolved through
        :meth:`Prepare_Field_For_Output`, optionally transformed by the
        ``LogPlot`` configuration, and then gathered to rank 0 for
        rendering. Plot windows are non-interactive by default unless
        ``params.Interactive_Plots`` is set to ``True``.

        On 3-D meshes, ``params.Plot_Isosurfaces`` (``{field: level}`` or
        ``{field: [levels]}``) adds one window per field, named
        ``<field>_isosurface``: the isosurfaces of the field at the given
        levels inside a translucent outer surface coloured by the field. The
        field does not need to be listed in ``Plot_Fields``. The entries are
        ignored on 1-D and 2-D meshes.
        """
        self._ensure_field_plot_epoch()
        self._ensure_recording_started()
        frame_count_before = self._frame_count

        if not hasattr(self.params, "Plot_Fields"):
            mprint("No fields to plot (params.Plot_Fields not defined)")
            self._plot_reduced_order_fields_from_plot_fields(frame_count_before)
            return

        plot_fields = self.params.Plot_Fields
        isosurfaces = self._get_isosurface_requests()
        if not plot_fields and not isosurfaces:
            self._plot_reduced_order_fields_from_plot_fields(frame_count_before)
            return

        V_dg1 = self._get_dg_space(1)
        gathered_fields = list(plot_fields) + [field for field in isosurfaces if field not in plot_fields]
        all_field_data = self._collect_local_field_plot_data(gathered_fields, V_dg1)
        all_rank_field_data = comm.gather(all_field_data, root=0)

        needs_mesh_data = None
        needs_outline = None
        if rank == 0:
            needs_mesh_data = self._needs_field_plot_mesh_data(
                plot_fields,
                all_rank_field_data,
            ) or self._needs_isosurface_mesh_data(isosurfaces, all_rank_field_data)
            needs_outline = self._needs_field_plot_mesh_data(
                plot_fields,
                all_rank_field_data,
                own_mesh_only=True,
            )
        needs_mesh_data, needs_outline = comm.bcast((needs_mesh_data, needs_outline), root=0)
        all_outlines = comm.gather(self._build_domain_outline_data(), root=0) if needs_outline else None

        all_topologies = None
        all_cell_types = None
        all_geometries = None
        if needs_mesh_data:
            topology_dg1, cell_types_dg1, geometry_dg1, _ = self._build_plot_mesh_data()
            all_topologies = comm.gather(topology_dg1, root=0)
            all_cell_types = comm.gather(cell_types_dg1, root=0)
            all_geometries = comm.gather(geometry_dg1, root=0)

        if rank != 0:
            self._plot_reduced_order_fields_from_plot_fields(frame_count_before)
            return

        for field_spec in plot_fields:
            self._upsert_field_plot(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_field_data,
                all_outlines,
            )
        for field_spec, levels in isosurfaces.items():
            self._upsert_isosurface_plot(
                field_spec,
                levels,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_field_data,
            )

        self._update_field_time_labels()
        self._write_recording_frames()
        self._plot_reduced_order_fields_from_plot_fields(frame_count_before)

    def Plot_Measures(self):
        """Plot scalar measures from ``params.Plot_Measures`` as line charts.

        Plot windows are non-interactive by default unless
        ``params.Interactive_Plots`` is set to ``True``.
        """
        if not hasattr(self.params, "Plot_Measures"):
            return

        plot_measures = self.params.Plot_Measures
        if not plot_measures or rank != 0:
            return

        for x_name, y_name in plot_measures:
            values = self._resolve_measure_values(x_name, y_name)
            if values is None:
                continue

            x_val, y_val = values
            key = (x_name, y_name)
            plot_state = self._measure_plots.get(key)
            if plot_state is None:
                plot_state = self._create_measure_plot_state(x_name, y_name, x_val, y_val)
                self._measure_plots[key] = plot_state
            else:
                plot_state.x.append(x_val)
                plot_state.y.append(y_val)
                self._update_measure_line(plot_state)

            plot_state.plotter.render()

    def Process_Fields_For_Plotting(self, field_spec, V_dg=None):
        """Compatibility wrapper around :meth:`Prepare_Field_For_Output`.

        This method is kept so older plotting code can still expect the
        historical ``(values, title, V_dg)`` return shape.  New code
        should prefer :meth:`Prepare_Field_For_Output`.
        """
        prepared = self.Prepare_Field_For_Plotting(field_spec, V_dg=V_dg)
        if prepared is None:
            return None
        return prepared.values, prepared.title, prepared.function_space

    # ------------------------------------------------------------------
    # Internal Helpers: Mesh Plotting
    # ------------------------------------------------------------------

    def _apply_plotter_interaction_mode(self, plotter) -> None:
        """PRIVATE: Disable user interaction unless explicitly requested."""
        if getattr(self.params, "Interactive_Plots", False):
            return
        if hasattr(plotter, "disable"):
            try:
                plotter.disable()
            except Exception:
                pass
        iren = getattr(plotter, "iren", None)
        interactor = getattr(iren, "interactor", None)
        if interactor is None:
            return
        try:
            if hasattr(interactor, "Disable"):
                interactor.Disable()
            elif hasattr(interactor, "SetEnable"):
                interactor.SetEnable(0)
        except Exception:
            pass

    def _build_plot_mesh_data(self):
        """PRIVATE: Build the DG1 mesh representation used for plotting."""
        current_epoch = getattr(self, "remesh_epoch", 0)
        if self._plot_mesh_data is None or self._plot_mesh_data_epoch != current_epoch:
            dg1_space = self._get_dg_space(1)
            topology_dg1, cell_types_dg1, geometry_dg1 = vtk_mesh(dg1_space)
            self._plot_mesh_data = (topology_dg1, cell_types_dg1, geometry_dg1, dg1_space)
            self._plot_mesh_data_epoch = current_epoch
        return self._plot_mesh_data

    def _build_space_plot_mesh_data(self, function_space):
        """PRIVATE: Return the VTK grid data of a (submesh) output space."""
        return vtk_mesh(self._get_dg_space(1, self._mesh_key(function_space.mesh)))

    def _build_domain_outline_data(self):
        """PRIVATE: Return this rank's exterior facets of the full mesh.

        Returned as ``(points, nodes_per_facet)``: lines in 2-D and boundary
        faces in 3-D, drawn around fields that are only plotted on a
        subdomain.
        """
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        facets = dmesh.exterior_facet_indices(self.mesh.topology)
        facets = facets[facets < self.mesh.topology.index_map(tdim - 1).size_local]
        vertices = dmesh.entities_to_geometry(self.mesh, tdim - 1, facets)
        return self.mesh.geometry.x[vertices.ravel()].copy(), vertices.shape[1]

    @staticmethod
    def _merge_domain_outlines(all_outlines):
        """PRIVATE: Build one outline PolyData from gathered rank facets."""
        if not all_outlines:
            return None
        points = np.vstack([points for points, _ in all_outlines if len(points)])
        nodes_per_facet = all_outlines[0][1]
        connectivity = np.arange(points.shape[0]).reshape(-1, nodes_per_facet)
        cells = np.hstack([np.full((connectivity.shape[0], 1), nodes_per_facet), connectivity]).ravel()
        if nodes_per_facet == 2:
            return pv.PolyData(points, lines=cells)
        return pv.PolyData(points, faces=cells).extract_feature_edges()

    def _invalidate_plot_mesh_cache(self) -> None:
        """PRIVATE: Discard cached VTK topology while retaining open plotters."""

        self._plot_mesh_data = None
        self._plot_mesh_data_epoch = None

    def _create_mesh_plotter(
        self,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_boundary_data,
        *,
        dimension=None,
        boundary_markers=None,
        domain_markers=None,
        annotation=None,
        off_screen=False,
    ):
        """PRIVATE: Create the root-rank plotter used by ``plot_mesh``."""
        plotter_options = {
            "window_size": self.default_plot_size,
            "title": "Mesh",
        }
        if off_screen:
            plotter_options["off_screen"] = True
        plotter = pv.Plotter(**plotter_options)
        self._populate_mesh_plotter(
            plotter,
            all_topologies,
            all_cell_types,
            all_geometries,
            all_boundary_data,
            dimension=self.mesh.topology.dim if dimension is None else dimension,
            boundary_markers=(
                getattr(self, "boundary_markers", None)
                if boundary_markers is None
                else boundary_markers
            ),
            domain_markers=(
                getattr(self, "domain_markers", None)
                if domain_markers is None
                else domain_markers
            ),
            annotation=annotation,
        )
        return plotter

    def _update_mesh_plotter(
        self,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_boundary_data,
        *,
        dimension,
        boundary_markers,
        domain_markers,
        annotation,
        show_window,
        movie_path,
        movie_fps,
    ) -> MeshPlotResult:
        """PRIVATE: Create once, then rebuild actors on the open mesh plotter."""

        is_new = self._mesh_plotter is None
        try:
            if is_new:
                self._mesh_plotter = self._create_mesh_plotter(
                    all_topologies,
                    all_cell_types,
                    all_geometries,
                    all_boundary_data,
                    dimension=dimension,
                    boundary_markers=boundary_markers,
                    domain_markers=domain_markers,
                    annotation=annotation,
                    off_screen=not show_window,
                )
            else:
                camera_position = self._capture_plotter_camera_position(
                    self._mesh_plotter
                )
                self._rebuild_mesh_plotter_actors(
                    all_topologies,
                    all_cell_types,
                    all_geometries,
                    all_boundary_data,
                    dimension=dimension,
                    boundary_markers=boundary_markers,
                    domain_markers=domain_markers,
                    annotation=annotation,
                    camera_position=camera_position,
                )

            if movie_path is not None:
                self._ensure_mesh_movie_open(movie_path, movie_fps)

            if is_new:
                if show_window:
                    self._mesh_plotter.show(interactive_update=True, auto_close=False)
                    self._apply_plotter_interaction_mode(self._mesh_plotter)
                    self._set_window_icon(self._mesh_plotter)
                    self._mesh_plot_visible = True
                else:
                    self._mesh_plotter.show(auto_close=False)
            else:
                self._mesh_plotter.update()
        except Exception as exc:
            warnings.warn(
                f"Could not update mesh plot: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            return MeshPlotResult(False, False)

        movie_frame_written = False
        if (
            movie_path is not None
            and not self._mesh_plot_movie_disabled
            and self._mesh_plot_movie_path == movie_path
        ):
            try:
                self._mesh_plotter.write_frame()
                movie_frame_written = True
            except Exception as exc:
                self._mesh_plot_movie_disabled = True
                warnings.warn(
                    f"Could not write mesh movie frame: {exc}",
                    RuntimeWarning,
                    stacklevel=3,
                )
        return MeshPlotResult(
            visible_updated=show_window and self._mesh_plot_visible,
            movie_frame_written=movie_frame_written,
        )

    def _rebuild_mesh_plotter_actors(
        self,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_boundary_data,
        *,
        dimension,
        boundary_markers,
        domain_markers,
        annotation,
        camera_position,
    ) -> None:
        """PRIVATE: Replace all mesh actors without intermediate renders.

        Once a PyVista window has been shown, actor addition renders by
        default. A gathered MPI mesh has one partition actor plus boundary
        fragments from each rank, so rebuilding an open window otherwise
        renders each contribution separately. Suppression is restored even
        when actor construction fails; the caller performs one final update.
        """

        plotter = self._mesh_plotter
        can_suppress_rendering = hasattr(plotter, "suppress_rendering")
        rendering_was_suppressed = (
            bool(plotter.suppress_rendering) if can_suppress_rendering else False
        )
        if can_suppress_rendering:
            plotter.suppress_rendering = True
        try:
            plotter.clear()
            self._populate_mesh_plotter(
                plotter,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_boundary_data,
                dimension=dimension,
                boundary_markers=boundary_markers,
                domain_markers=domain_markers,
                annotation=annotation,
            )
            self._restore_plotter_camera_position(
                plotter,
                camera_position,
            )
        finally:
            if can_suppress_rendering:
                plotter.suppress_rendering = rendering_was_suppressed

    def _populate_mesh_plotter(
        self,
        plotter,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_boundary_data,
        *,
        dimension,
        boundary_markers,
        domain_markers,
        annotation,
    ) -> None:
        """PRIVATE: Add current mesh actors to an existing generic plotter."""

        _mesh_plot_helpers.add_mesh_partitions(
            plotter,
            all_topologies,
            all_cell_types,
            all_geometries,
            dimension,
        )
        _mesh_plot_helpers.add_named_boundary_overlays(
            plotter,
            all_boundary_data,
            boundary_markers,
        )
        _mesh_plot_helpers.add_domain_label(plotter, all_geometries, domain_markers)
        if annotation is not None:
            plotter.add_text(annotation, position="upper_left", font_size=12)
        _mesh_plot_helpers.configure_mesh_camera(plotter, dimension)

    def _ensure_mesh_movie_open(self, movie_path: str, movie_fps: int) -> None:
        """PRIVATE: Open one movie on the persistent mesh plotter."""

        if self._mesh_plot_movie_disabled:
            return
        if self._mesh_plot_movie_path is not None:
            if self._mesh_plot_movie_path != movie_path:
                warnings.warn(
                    "The persistent mesh plotter is already recording to "
                    f"'{self._mesh_plot_movie_path}'; ignoring '{movie_path}'.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return

        movie_folder = _os.path.dirname(movie_path)
        if movie_folder:
            _os.makedirs(movie_folder, exist_ok=True)
        try:
            self._mesh_plotter.open_movie(movie_path, framerate=movie_fps)
            self._mesh_plot_movie_path = movie_path
        except Exception as exc:
            self._mesh_plot_movie_disabled = True
            warnings.warn(
                f"Could not open mesh movie: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )

    @staticmethod
    def _validate_mesh_plot_request(show_window, movie_path, movie_fps, annotation) -> None:
        """PRIVATE: Validate optional persistent mesh-plot controls."""

        if not isinstance(show_window, bool):
            raise TypeError("show_window must be a bool.")
        if movie_path is not None:
            try:
                normalized_path = _os.fspath(movie_path)
            except TypeError as exc:
                raise TypeError("movie_path must be path-like or None.") from exc
            if not normalized_path:
                raise ValueError("movie_path must not be empty.")
        if movie_fps is not None and (
            isinstance(movie_fps, bool) or not isinstance(movie_fps, Integral) or movie_fps < 1
        ):
            raise ValueError("movie_fps must be an integer greater than or equal to one.")
        if annotation is not None and not isinstance(annotation, str):
            raise TypeError("annotation must be a string or None.")

    def _add_mesh_partitions(self, plotter, all_topologies, all_cell_types, all_geometries) -> None:
        """PRIVATE: Add one mesh overlay per MPI partition to a plotter."""
        _mesh_plot_helpers.add_mesh_partitions(
            plotter,
            all_topologies,
            all_cell_types,
            all_geometries,
            self.mesh.topology.dim,
        )

    def _add_boundary_overlays(self, plotter, all_boundary_data) -> None:
        """PRIVATE: Overlay named boundary fragments and labels on a mesh plot."""
        _mesh_plot_helpers.add_named_boundary_overlays(
            plotter,
            all_boundary_data,
            getattr(self, "boundary_markers", None),
        )

    def _build_boundary_mesh(self, points, lines, faces):
        """PRIVATE: Build a PyVista boundary mesh from line or face connectivity."""
        return _mesh_plot_helpers.build_boundary_mesh(points, lines, faces)

    def _add_domain_label(self, plotter, all_geometries) -> None:
        """PRIVATE: Add the domain label at the center of the gathered mesh."""
        _mesh_plot_helpers.add_domain_label(
            plotter,
            all_geometries,
            getattr(self, "domain_markers", None),
        )

    def _get_domain_label_text(self) -> str:
        """PRIVATE: Return the text used for the domain label in mesh plots."""
        return _mesh_plot_helpers.domain_label_text(getattr(self, "domain_markers", None))

    def _get_boundary_color_map(self) -> Dict[str, str]:
        """PRIVATE: Assign deterministic display colors to named boundaries."""
        return _mesh_plot_helpers.boundary_color_map(self.boundary_markers)

    def _collect_boundary_plot_data(self, tdim: Optional[int] = None):
        """PRIVATE: Collect local boundary geometry fragments for plotting."""
        return _mesh_plot_helpers.collect_boundary_plot_data(
            self.mesh,
            getattr(self, "facet_tags", None),
            getattr(self, "boundary_markers", None),
            tdim=tdim,
        )

    # ------------------------------------------------------------------
    # Internal Helpers: Field Plotting
    # ------------------------------------------------------------------

    def _ensure_recording_started(self) -> None:
        """PRIVATE: Auto-start recording when requested through params."""
        if self._recording or not getattr(self.params, "record_movie", False):
            return

        movie_dir = getattr(self.params, "movie_folder", None)
        if movie_dir is None:
            movie_dir = _os.path.join(
                getattr(self.params, "output_dir", "Results"),
                "movies",
            )
        fps = getattr(self.params, "movie_fps", 10)
        self.StartRecording(movie_dir, fps=fps)

    def _ensure_field_plot_epoch(self) -> None:
        """PRIVATE: Advance the plot epoch without closing existing windows."""

        current_epoch = int(getattr(self, "remesh_epoch", 0))
        if self._field_plots_epoch == current_epoch:
            return
        self._field_plots_epoch = current_epoch

    def _prepare_plots_for_remesh(self) -> None:
        """PRIVATE: Mark the new epoch while retaining every open plot window."""

        self._ensure_field_plot_epoch()

    def _plot_state_needs_topology_rebuild(self, state) -> bool:
        """PRIVATE: Return whether a retained plot state belongs to an old mesh."""

        current_epoch = int(getattr(self, "remesh_epoch", 0))
        return int(getattr(state, "mesh_epoch", current_epoch)) != current_epoch

    def _close_mesh_plotter_state(self) -> None:
        """PRIVATE: Finalize and clear the persistent mesh plotter state."""

        if self._mesh_plotter is not None:
            self._close_plotter_safely(self._mesh_plotter)
            self._mesh_plotter = None
        self._mesh_plot_visible = False
        self._mesh_plot_movie_path = None
        self._mesh_plot_movie_disabled = False

    def _plot_reduced_order_fields_from_plot_fields(self, frame_count_before: int) -> None:
        """PRIVATE: Plot optional reduced-order fields after standard fields."""
        plot_reduced_order_fields = getattr(self, "_plot_reduced_order_fields", None)
        if plot_reduced_order_fields is None:
            return

        field_frames_written = self._frame_count != frame_count_before
        plot_reduced_order_fields(increment_frame_count=not field_frames_written)

    def _collect_local_field_plot_data(self, plot_fields, V_dg):
        """PRIVATE: Build the local field arrays gathered by ``Plot_Fields``."""
        all_field_data = {}
        log_plot = getattr(self.params, "LogPlot", None)

        for field_idx, field_spec in enumerate(plot_fields):
            prepared = self.Prepare_Field_For_Plotting(field_spec, V_dg=V_dg)
            if prepared is None:
                continue

            field_values = prepared.values
            field_title = prepared.title
            field_association = prepared.association
            field_values, field_title = self._apply_log_plot_if_requested(
                field_values,
                field_title,
                log_plot,
                field_idx,
            )
            # Fields plotted on their own submesh carry that submesh's grid.
            own_mesh_data = None
            if prepared.function_space.mesh is not self.mesh:
                own_mesh_data = self._build_space_plot_mesh_data(prepared.function_space)
            all_field_data[field_spec] = (field_values, field_title, field_association, own_mesh_data)

        if self._level_set_plotting_enabled() and "LS" in self.fields and "LS" not in all_field_data:
            prepared = self.Prepare_Field_For_Plotting("LS", V_dg=V_dg)
            if prepared is not None and prepared.association == "point":
                all_field_data["LS"] = (
                    prepared.values,
                    prepared.title,
                    prepared.association,
                )

        return all_field_data

    def _apply_log_plot_if_requested(self, field_values, field_title, log_plot, field_idx):
        """PRIVATE: Apply optional log10 scaling configured for one plotted field."""
        use_log = False
        if isinstance(log_plot, list) and field_idx < len(log_plot):
            use_log = bool(log_plot[field_idx])

        if not use_log:
            return field_values, field_title

        return (
            np.log10(np.maximum(field_values, 1.0e-300)),
            f"{field_title} (log10)",
        )

    def _upsert_field_plot(
        self,
        field_spec,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_rank_field_data,
        all_outlines=None,
    ) -> None:
        """PRIVATE: Create or refresh the plot window for one field specification."""
        field_title = self._get_field_title_from_rank_data(field_spec, all_rank_field_data)
        if field_title is None:
            return

        state = self._field_plots.get(field_spec)
        if state is None:
            plot_grid = self._build_merged_field_grid(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_field_data,
            )
            if plot_grid is None:
                return
            self._field_plots[field_spec] = self._create_field_plot_state(
                field_spec,
                field_title,
                plot_grid,
                self._merge_domain_outlines(all_outlines) if self._field_has_own_mesh(field_spec, all_rank_field_data) else None,
            )
        elif self._plot_state_needs_topology_rebuild(state):
            plot_grid = self._build_merged_field_grid(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_field_data,
            )
            if plot_grid is None:
                return
            self._rebuild_field_plot_state(
                field_spec,
                plot_grid,
            )
        else:
            field_values, field_association, level_set_values = self._collect_field_plot_values(
                field_spec,
                all_rank_field_data,
            )
            if field_values is None:
                return
            self._refresh_field_plot_state(
                field_spec,
                field_values,
                field_association,
                level_set_values,
            )

    def _get_isosurface_requests(self) -> Dict[str, tuple]:
        """PRIVATE: Return ``Plot_Isosurfaces`` as ``{field: levels}``, empty below 3-D."""
        requests = getattr(self.params, "Plot_Isosurfaces", None) or {}
        if self.mesh.topology.dim != 3:
            return {}
        return {
            field: tuple(float(level) for level in np.atleast_1d(levels))
            for field, levels in requests.items()
        }

    @staticmethod
    def _isosurface_plot_name(field_spec: str) -> str:
        """PRIVATE: Return the window (and file) name of a field's isosurface plot."""
        return f"{field_spec}_isosurface"

    def _needs_isosurface_mesh_data(self, isosurfaces, all_rank_field_data) -> bool:
        """PRIVATE: Return whether an isosurface window needs a new plot grid."""
        for field_spec in isosurfaces:
            state = self._field_plots.get(self._isosurface_plot_name(field_spec))
            if state is not None and not self._plot_state_needs_topology_rebuild(state):
                continue
            if self._get_field_title_from_rank_data(field_spec, all_rank_field_data) is not None:
                return True
        return False

    def _upsert_isosurface_plot(
        self,
        field_spec,
        levels,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_rank_field_data,
    ) -> None:
        """PRIVATE: Create or refresh the isosurface window of one field."""
        field_title = self._get_field_title_from_rank_data(field_spec, all_rank_field_data)
        if field_title is None or self._field_has_own_mesh(field_spec, all_rank_field_data):
            return

        key = self._isosurface_plot_name(field_spec)
        state = self._field_plots.get(key)
        if state is None or self._plot_state_needs_topology_rebuild(state):
            grid = self._build_merged_field_grid(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_field_data,
            )
            if grid is None:
                return
            grid = self._isosurface_point_grid(grid, field_spec)
            if state is None:
                self._field_plots[key] = self._create_isosurface_plot_state(key, field_spec, field_title, grid, levels)
                return
            state.grid = grid
            state.mesh_epoch = int(getattr(self, "remesh_epoch", 0))
        else:
            values, association, _ = self._collect_field_plot_values(field_spec, all_rank_field_data)
            if values is None:
                return
            state.grid.point_data.pop(field_spec, None)
            state.grid.cell_data.pop(field_spec, None)
            if association == "cell":
                state.grid.cell_data[field_spec] = values
            else:
                state.grid.point_data[field_spec] = values
            state.grid = self._isosurface_point_grid(state.grid, field_spec)
            self._refresh_isosurface_actors(state, shell=False)
            self._render_isosurface_window(key, state)
            return
        self._refresh_isosurface_actors(state)
        self._render_isosurface_window(key, state)

    def _render_isosurface_window(self, key, state) -> None:
        """PRIVATE: Render one isosurface window, dropping it if it was closed."""
        try:
            state.plotter.update() if hasattr(state.plotter, "update") else state.plotter.render()
        except Exception:
            mprint(f"Warning: Could not update plot for '{key}'. Window may have been closed.")
            self._field_plots.pop(key, None)

    @staticmethod
    def _isosurface_point_grid(grid, field_spec):
        """PRIVATE: Return the grid with the field as point data (cell data is averaged to the points)."""
        if field_spec in grid.cell_data:
            grid = grid.cell_data_to_point_data()
        return grid

    @staticmethod
    def _isosurface_shell(grid):
        """PRIVATE: Return the outer surface of a gathered plot grid.

        The plot grids are discontinuous (DG1, and gathered rank by rank), so
        coincident points are merged first; otherwise every cell face would
        count as a boundary face. The faces are then oriented outward
        consistently: refined cells have mixed vertex orderings, and faces of
        opposite orientation would show up as feature edges and in the shading.
        """
        bounds = np.asarray(grid.bounds, dtype=float)
        size = float(np.max(bounds[1::2] - bounds[0::2])) if grid.n_points else 1.0
        surface = grid.clean(tolerance=1.0e-9 * size).extract_surface()
        if surface.n_cells == 0:
            return surface
        return surface.compute_normals(cell_normals=True, point_normals=False, split_vertices=False,
                                       consistent_normals=True, auto_orient_normals=True)

    def _create_isosurface_plot_state(self, key, field_spec, field_title, grid, levels) -> _FieldPlotState:
        """PRIVATE: Create the window of one isosurface plot."""
        title = f"{field_title}, isosurface " + ", ".join(f"{level:g}" for level in levels)
        plotter = pv.Plotter(window_size=self.default_plot_size, title=title)
        plotter.add_text(title, position="upper_edge", font_size=14, color="black")
        state = _FieldPlotState(
            plotter=plotter,
            grid=grid,
            actor=None,
            time_text=None,
            mesh_epoch=int(getattr(self, "remesh_epoch", 0)),
            isosurface_field=field_spec,
            isosurface_levels=tuple(levels),
            isosurface_title=field_title,
        )
        self._refresh_isosurface_actors(state, field_title)
        plotter.show_axes()
        plotter.view_isometric()
        state.time_text = plotter.add_text(
            self._format_time_label(),
            position=(0.01, 0.92),
            font_size=10,
            color="black",
            name="time_label",
        )
        if self._recording:
            self._open_movie_safely(key, plotter, self._recording_folder, self._recording_fps)
        plotter.show(interactive_update=True, auto_close=False, interactive=False)
        self._apply_plotter_interaction_mode(plotter)
        self._set_window_icon(plotter)
        return state

    def _refresh_isosurface_actors(self, state, field_title=None, shell=True) -> None:
        """PRIVATE: Redraw the outer surface (with ``shell``) and the isosurfaces of one window.

        The outer surface is drawn in light, translucent grey with its feature
        edges, as a reference for where the isosurfaces lie; the isosurfaces
        are coloured by the field over its current range.
        """
        plotter = state.plotter
        if shell:
            outer = self._isosurface_shell(state.grid)
            plotter.add_mesh(outer, color="#9a9a9a", opacity=0.3, show_scalar_bar=False,
                             name="isosurface_shell", render=False)
            edges = outer.extract_feature_edges(feature_angle=30.0, boundary_edges=False,
                                                non_manifold_edges=False, manifold_edges=False)
            plotter.add_mesh(edges, color="#606060", line_width=1.0, show_scalar_bar=False,
                             name="isosurface_edges", render=False)

        field_spec = state.isosurface_field
        values = np.asarray(state.grid.point_data[field_spec], dtype=float)
        finite = values[np.isfinite(values)]
        if state.isosurface_actor is not None:
            plotter.remove_actor(state.isosurface_actor, render=False)
            state.isosurface_actor = None
        if not finite.size:
            return
        clim = [float(finite.min()), float(finite.max())]
        levels = [level for level in state.isosurface_levels if clim[0] <= level <= clim[1]]
        if not levels:
            return
        try:
            surface = state.grid.contour(isosurfaces=levels, scalars=field_spec)
        except Exception:
            return
        if getattr(surface, "n_points", 0) == 0:
            return
        # Merge the points the discontinuous cells duplicate, so the shading is smooth across cells
        surface = surface.clean(tolerance=1.0e-9 * max(state.grid.length, 1.0e-300))
        if clim[1] <= clim[0]:
            clim[1] = clim[0] + 1.0
        state.isosurface_actor = plotter.add_mesh(
            surface,
            scalars=field_spec,
            cmap=self._get_field_colormap(field_spec),
            clim=clim,
            smooth_shading=True,
            name="isosurface",
            render=False,
            scalar_bar_args={
                "title": field_title or state.isosurface_title,
                "title_font_size": 12,
                "label_font_size": 10,
                "n_labels": 5,
                "italic": False,
                "fmt": "%.2e",
                "position_x": 0.15,
                "position_y": 0.05,
            },
        )

    def _get_field_title_from_rank_data(self, field_spec, all_rank_field_data) -> Optional[str]:
        """PRIVATE: Return the first gathered field title for one field specification."""
        for rank_field_data in all_rank_field_data:
            data = rank_field_data.get(field_spec)
            if data is not None:
                return data[1]
        return None

    def _needs_field_plot_mesh_data(self, plot_fields, all_rank_field_data, own_mesh_only: bool = False) -> bool:
        """PRIVATE: Return whether any valid requested field needs a new plot grid.

        By default only fields plotted on the full mesh count; with
        ``own_mesh_only`` only fields plotted on their own submesh count.
        """
        for field_spec in plot_fields:
            state = self._field_plots.get(field_spec)
            if state is not None and not self._plot_state_needs_topology_rebuild(state):
                continue
            if self._get_field_title_from_rank_data(field_spec, all_rank_field_data) is None:
                continue
            if self._field_has_own_mesh(field_spec, all_rank_field_data) == own_mesh_only:
                return True
        return False

    @staticmethod
    def _field_has_own_mesh(field_spec, all_rank_field_data) -> bool:
        """PRIVATE: Return whether a field is plotted on its own submesh grid."""
        for rank_field_data in all_rank_field_data:
            data = rank_field_data.get(field_spec)
            if data is not None:
                return len(data) > 3 and data[3] is not None
        return False

    def _build_merged_field_grid(
        self,
        field_spec,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_rank_field_data,
    ):
        """PRIVATE: Build the gathered rank-local grid for one plotted field."""
        rank_records = list(
            self._iter_field_plot_rank_data(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_field_data,
            )
        )
        if not rank_records:
            return None

        association = rank_records[0][4]
        topologies = []
        cell_types = []
        geometries = []
        field_values = []
        level_set_values = []
        have_level_set_values = True
        point_offset = 0

        for topology, rank_cell_types, geometry, values, rank_association, rank_level_set_values in rank_records:
            if rank_association != association:
                mprint(
                    f"Warning: Cannot plot '{field_spec}' with mixed point/cell data associations."
                )
                return None

            topologies.append(self._offset_vtk_topology(topology, point_offset))
            cell_types.append(np.asarray(rank_cell_types))
            geometries.append(np.asarray(geometry))
            field_values.append(values)
            point_offset += len(geometry)
            if rank_level_set_values is None:
                have_level_set_values = False
            else:
                level_set_values.append(rank_level_set_values)

        plot_grid = pv.UnstructuredGrid(
            np.concatenate(topologies),
            np.concatenate(cell_types),
            np.vstack(geometries),
        )
        values = np.concatenate(field_values)
        if association == "cell":
            plot_grid.cell_data[field_spec] = values
        else:
            plot_grid.point_data[field_spec] = values
        if field_spec != "LS" and have_level_set_values and len(level_set_values) == len(rank_records):
            plot_grid.point_data["LS"] = np.concatenate(level_set_values)
        return plot_grid

    def _iter_field_plot_rank_data(
        self,
        field_spec,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_rank_field_data,
    ):
        """PRIVATE: Yield non-empty gathered data for one plotted field.

        The full-mesh grid data may be ``None`` when only fields plotted on
        their own submesh grids need a new plot.
        """
        no_mesh_data = [None] * len(all_rank_field_data)
        for topology, cell_types, geometry, rank_field_data in zip(
            all_topologies if all_topologies is not None else no_mesh_data,
            all_cell_types if all_cell_types is not None else no_mesh_data,
            all_geometries if all_geometries is not None else no_mesh_data,
            all_rank_field_data,
        ):
            data = rank_field_data.get(field_spec)
            if data is None:
                continue

            rank_values = np.asarray(data[0])
            if rank_values.size == 0:
                continue
            if len(data) > 3 and data[3] is not None:
                topology, cell_types, geometry = data[3]

            rank_association = data[2] if len(data) > 2 else "point"
            rank_level_set_values = None
            if self._level_set_plotting_enabled():
                if field_spec == "LS" and rank_association == "point":
                    rank_level_set_values = rank_values
                else:
                    level_set_data = rank_field_data.get("LS")
                    if level_set_data is not None:
                        level_set_values = np.asarray(level_set_data[0])
                        level_set_association = level_set_data[2] if len(level_set_data) > 2 else "point"
                        if level_set_association == "point" and len(level_set_values) == len(geometry):
                            rank_level_set_values = level_set_values
            yield (
                topology,
                cell_types,
                geometry,
                rank_values,
                rank_association,
                rank_level_set_values,
            )

    def _collect_field_plot_values(self, field_spec, all_rank_field_data):
        """PRIVATE: Concatenate gathered scalar data for an existing field plot."""
        field_values = []
        level_set_values = []
        association = None
        have_level_set_values = True

        for rank_field_data in all_rank_field_data:
            data = rank_field_data.get(field_spec)
            if data is None:
                continue

            rank_values = np.asarray(data[0])
            if rank_values.size == 0:
                continue

            rank_association = data[2] if len(data) > 2 else "point"
            if association is None:
                association = rank_association
            elif rank_association != association:
                mprint(
                    f"Warning: Cannot plot '{field_spec}' with mixed point/cell data associations."
                )
                return None, None, None

            field_values.append(rank_values)
            if self._level_set_plotting_enabled():
                if field_spec == "LS" and rank_association == "point":
                    level_set_values.append(rank_values)
                else:
                    level_set_data = rank_field_data.get("LS")
                    if level_set_data is None:
                        have_level_set_values = False
                    else:
                        rank_level_set_values = np.asarray(level_set_data[0])
                        rank_level_set_association = level_set_data[2] if len(level_set_data) > 2 else "point"
                        if rank_level_set_association == "point":
                            level_set_values.append(rank_level_set_values)
                        else:
                            have_level_set_values = False

        if not field_values:
            return None, None, None
        concatenated_level_set = None
        if have_level_set_values and len(level_set_values) == len(field_values):
            concatenated_level_set = np.concatenate(level_set_values)
        return np.concatenate(field_values), association, concatenated_level_set

    @staticmethod
    def _offset_vtk_topology(topology, point_offset: int):
        """PRIVATE: Offset point indices in a VTK cell connectivity array."""
        adjusted = np.array(topology, copy=True)
        if point_offset == 0:
            return adjusted

        if adjusted.ndim == 1:
            cursor = 0
            while cursor < adjusted.size:
                points_per_cell = int(adjusted[cursor])
                first_point = cursor + 1
                past_points = first_point + points_per_cell
                adjusted[first_point:past_points] += point_offset
                cursor = past_points
            return adjusted

        adjusted[:, 1:] += point_offset
        return adjusted

    def _build_rank_field_grids(
        self,
        field_spec,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_rank_field_data,
    ):
        """PRIVATE: Build one PyVista grid per rank for one field."""
        rank_grids = []
        for topology, cell_types, geometry, rank_field_data in zip(
            all_topologies,
            all_cell_types,
            all_geometries,
            all_rank_field_data,
        ):
            data = rank_field_data.get(field_spec)
            if data is None:
                continue
            rank_values = data[0]
            rank_association = data[2] if len(data) > 2 else "point"
            if len(rank_values) == 0:
                continue

            rank_grid = pv.UnstructuredGrid(topology, cell_types, geometry)
            if rank_association == "cell":
                rank_grid.cell_data[field_spec] = rank_values
            else:
                rank_grid.point_data[field_spec] = rank_values
            rank_grids.append(rank_grid)

        return rank_grids

    def _create_field_plot_state(self, field_spec, field_title, plot_grid, outline=None) -> _FieldPlotState:
        """PRIVATE: Create the plotter, grid, and actor state for one field plot.

        ``outline`` optionally holds the full-mesh outline drawn around a
        field that is only plotted on its subdomain.
        """
        plotter = pv.Plotter(window_size=self.default_plot_size, title=field_title)
        plotter.add_text(field_title, position="upper_edge", font_size=14, color="black")
        if outline is not None:
            plotter.add_mesh(outline, color="black", line_width=1.5)

        is_3d = self.mesh.topology.dim == 3
        actor = plotter.add_mesh(
            plot_grid,
            scalars=field_spec,
            show_edges=False,
            edge_color="black",
            # Fields on interfaces/boundaries of 2-D meshes are lines; draw
            # them thick enough to be visible next to the domain outline.
            line_width=_INTERFACE_LINE_WIDTH if self._is_line_grid(plot_grid) else 0.5,
            opacity=1.0,
            cmap=self._get_field_colormap(field_spec),
            scalar_bar_args={
                "title": field_title,
                "title_font_size": 12,
                "label_font_size": 10,
                "n_labels": 5,
                "italic": False,
                "fmt": "%.2e",
                "position_x": 0.15,
                "position_y": 0.05,
            },
        )

        plotter.show_axes()
        if is_3d:
            plotter.view_isometric()
        else:
            plotter.view_xy()
            plotter.enable_2d_style()

        time_actor = plotter.add_text(
            self._format_time_label(),
            position=(0.01, 0.92),
            font_size=10,
            color="black",
            name="time_label",
        )

        level_set_contour_actor = self._add_level_set_contour_actor(plotter, plot_grid)

        if self._recording:
            self._open_movie_safely(
                field_spec,
                plotter,
                self._recording_folder,
                self._recording_fps,
            )

        plotter.show(interactive_update=True, auto_close=False, interactive=False)
        self._apply_plotter_interaction_mode(plotter)
        self._set_window_icon(plotter)
        return _FieldPlotState(
            plotter=plotter,
            grid=plot_grid,
            actor=actor,
            time_text=time_actor,
            level_set_contour_actor=level_set_contour_actor,
            mesh_epoch=int(getattr(self, "remesh_epoch", 0)),
        )

    def _rebuild_field_plot_state(self, field_spec, plot_grid) -> None:
        """PRIVATE: Replace one actor's topology without replacing its window."""

        state = self._field_plots[field_spec]
        try:
            self._set_actor_dataset(state.actor, plot_grid)
            state.grid = plot_grid
            self._refresh_level_set_contour_actor(state)
            scalars = self._get_grid_scalars(plot_grid, field_spec)
            if np.any(np.isfinite(scalars)):
                state.actor.GetMapper().SetScalarRange(
                    [float(np.nanmin(scalars)), float(np.nanmax(scalars))]
                )
            if hasattr(state.plotter, "update"):
                state.plotter.update()
            else:
                state.plotter.render()
            state.mesh_epoch = int(getattr(self, "remesh_epoch", 0))
        except Exception:
            mprint(
                f"Warning: Could not rebuild plot topology for '{field_spec}'. "
                "The existing window was retained for a later retry."
            )

    @staticmethod
    def _is_line_grid(grid) -> bool:
        """PRIVATE: Return whether a plot grid only contains line cells."""
        celltypes = np.asarray(getattr(grid, "celltypes", []))
        return celltypes.size > 0 and bool(np.all(np.isin(celltypes, _VTK_LINE_CELL_TYPES)))

    @staticmethod
    def _get_grid_scalars(grid, field_spec: str) -> np.ndarray:
        """PRIVATE: Return the active field array from a gathered plot grid."""

        cell_data = getattr(grid, "cell_data", {})
        if field_spec in cell_data:
            return np.asarray(cell_data[field_spec])
        point_data = getattr(grid, "point_data", {})
        if field_spec in point_data:
            return np.asarray(point_data[field_spec])
        return np.asarray([], dtype=float)

    def _get_field_colormap(self, field_spec: str) -> str:
        """PRIVATE: Return the colormap used to render one field specification."""
        if "phase" in field_spec.lower():
            return "inferno"
        return "viridis"

    def _refresh_field_plot_state(self, field_spec, new_scalars, field_association, level_set_values=None) -> None:
        """PRIVATE: Refresh one existing field plot with new scalar data."""
        state = self._field_plots[field_spec]
        state_cell_data = getattr(state.grid, "cell_data", {})
        state_point_data = getattr(state.grid, "point_data", {})

        if field_association == "cell":
            if field_spec in state_cell_data:
                state_cell_data[field_spec][:] = new_scalars
            else:
                state_cell_data[field_spec] = new_scalars
        else:
            if field_spec in state_point_data:
                state_point_data[field_spec][:] = new_scalars
            else:
                state_point_data[field_spec] = new_scalars
        if level_set_values is not None:
            if "LS" in state_point_data:
                state_point_data["LS"][:] = level_set_values
            else:
                state_point_data["LS"] = level_set_values
        state.grid.Modified()
        self._refresh_level_set_contour_actor(state)

        mapper = state.actor.GetMapper()
        if np.any(np.isfinite(new_scalars)):
            mapper.SetScalarRange([np.nanmin(new_scalars), np.nanmax(new_scalars)])

        try:
            if hasattr(state.plotter, "update"):
                state.plotter.update()
            else:
                state.plotter.render()
        except Exception:
            mprint(
                f"Warning: Could not update plot for '{field_spec}'. "
                "Window may have been closed."
            )
            self._field_plots.pop(field_spec, None)

    def _add_level_set_contour_actor(self, plotter, plot_grid):
        """PRIVATE: Overlay the ``LS = 0`` contour on 2-D field plots when available."""
        if self.mesh.topology.dim != 2 or not self._level_set_plotting_enabled():
            return None
        if "LS" not in getattr(plot_grid, "point_data", {}):
            return None

        contour_grid = self._build_level_set_contour_grid(plot_grid)
        if contour_grid is None:
            return None

        return plotter.add_mesh(
            contour_grid,
            color="black",
            line_width=3,
            opacity=1.0,
            label="LS=0",
        )

    def _refresh_level_set_contour_actor(self, state) -> None:
        """PRIVATE: Rebuild the LS contour overlay for an existing plot state."""
        if self.mesh.topology.dim != 2 or not self._level_set_plotting_enabled():
            return

        old_actor = getattr(state, "level_set_contour_actor", None)
        if old_actor is not None and hasattr(state.plotter, "remove_actor"):
            try:
                state.plotter.remove_actor(old_actor)
            except Exception:
                pass

        contour_grid = self._build_level_set_contour_grid(state.grid)
        if contour_grid is None:
            state.level_set_contour_actor = None
            return

        state.level_set_contour_actor = state.plotter.add_mesh(
            contour_grid,
            color="black",
            line_width=3,
            opacity=1.0,
            label="LS=0",
        )

    def _build_level_set_contour_grid(self, plot_grid):
        """PRIVATE: Build the zero contour of gathered LS point data."""
        level_set_values = getattr(plot_grid, "point_data", {}).get("LS", None)
        if level_set_values is None:
            return None

        level_set_values = np.asarray(level_set_values, dtype=float).reshape(-1)
        if level_set_values.size == 0:
            return None
        if np.nanmin(level_set_values) > 0.0 or np.nanmax(level_set_values) < 0.0:
            return None

        try:
            contour_grid = plot_grid.contour(isosurfaces=[0.0], scalars="LS")
        except Exception:
            return None

        if getattr(contour_grid, "n_points", 0) == 0:
            return None
        self._lift_level_set_contour_grid(contour_grid, plot_grid)
        return contour_grid

    def _lift_level_set_contour_grid(self, contour_grid, plot_grid) -> None:
        """PRIVATE: Lift the LS contour above 2-D field meshes to avoid z-fighting."""
        points = np.asarray(getattr(contour_grid, "points", np.empty((0, 3))), dtype=float)
        if points.size == 0 or points.ndim != 2 or points.shape[1] < 3:
            return

        plot_points = np.asarray(getattr(plot_grid, "points", np.empty((0, 3))), dtype=float)
        if plot_points.size == 0 or plot_points.ndim != 2:
            scale = 1.0
            plane_z = 0.0
        else:
            horizontal_extent = np.ptp(plot_points[:, : min(2, plot_points.shape[1])], axis=0)
            scale = max(float(np.max(horizontal_extent)), 1.0)
            plane_z = float(np.nanmax(plot_points[:, 2])) if plot_points.shape[1] >= 3 else 0.0

        lifted_points = points.copy()
        lifted_points[:, 2] = np.maximum(lifted_points[:, 2], plane_z) + 1.0e-6 * scale
        contour_grid.points = lifted_points

    def _level_set_plotting_enabled(self) -> bool:
        """PRIVATE: Return whether params request LS-aware plotting behavior."""
        solution_steps = getattr(self.params, "Solution_Steps", {})
        field_orders = getattr(self.params, "Field_Orders", {})
        return "LS" in solution_steps or "LS" in field_orders

    def _get_level_set_transition_width(self) -> float:
        """PRIVATE: Return the positive LS transition width for plotting masks."""

        level_set_params = getattr(self.params, "Level_Set", {})
        if isinstance(level_set_params, dict) and "Transition_Width" in level_set_params:
            width = float(level_set_params["Transition_Width"])
        elif hasattr(self.params, "Level_Set_Transition_Width"):
            width = float(self.params.Level_Set_Transition_Width)
        else:
            raise ValueError("Level_Set_Transition_Width must be a positive finite number.")
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("Level_Set_Transition_Width must be a positive finite number.")
        return width

    def _smooth_level_set_heaviside_values(self, level_set_values) -> np.ndarray:
        """PRIVATE: Evaluate the smooth LS Heaviside used for plotting masks."""
        ls_values = np.asarray(level_set_values, dtype=float)
        epsilon = self._get_level_set_transition_width()
        heaviside = np.zeros_like(ls_values, dtype=float)
        heaviside[ls_values >= epsilon] = 1.0

        transition = (ls_values > -epsilon) & (ls_values < epsilon)
        xi = ls_values[transition] / epsilon
        heaviside[transition] = 0.5 * (1.0 + xi + np.sin(np.pi * xi) / np.pi)
        return heaviside

    def _format_time_label(self) -> str:
        """PRIVATE: Format the time text shown in field plot windows.

        Decomposes the current time into days, hours, and remaining seconds,
        omitting leading zero components.  Examples::

            0 s        → "t = 0.000e+00 s"
            7200 s     → "t = 2 H"
            90061 s    → "t = 1 D  1 H  6.100e+01 s"
            86400 s    → "t = 1 D"
            1e-4 s     → "t = 1.000e-04 s"
        """
        t = getattr(self.params, "t", 0.0)
        total_s = abs(t)
        sign = "-" if t < 0 else ""

        years = int(total_s // 31536000)
        remaining = total_s - years * 31536000
        days = int(remaining // 86400)
        remaining -= days * 86400
        hours = int(remaining // 3600)
        remaining -= hours * 3600
        secs = remaining

        parts = []
        if years:
            parts.append(f"{years} Y")
        if days:
            parts.append(f"{days} D")
        if hours or days:
            parts.append(f"{hours} H")
        parts.append(f"{secs:.3e} s")

        return f"t = {sign}{'  '.join(parts)}"

    def _update_field_time_labels(self) -> None:
        """PRIVATE: Refresh the per-window time label on all field plots."""
        label = self._format_time_label()
        for state in self._field_plots.values():
            state.time_text = state.plotter.add_text(
                label,
                position=(0.01, 0.92),
                font_size=10,
                color="black",
                name="time_label",
            )

    def _write_recording_frames(self) -> None:
        """PRIVATE: Write one video frame for each active field plot."""
        if not self._recording:
            return

        for state in self._field_plots.values():
            try:
                state.plotter.write_frame()
            except Exception:
                pass
        self._frame_count += 1

    # ------------------------------------------------------------------
    # Internal Helpers: Measure Plotting
    # ------------------------------------------------------------------

    def _resolve_measure_values(self, x_name: str, y_name: str) -> Optional[tuple[float, float]]:
        """PRIVATE: Resolve the ``(x, y)`` values used for one measure plot point."""
        x_val = None
        y_val = None

        if x_name == "t":
            x_val = float(getattr(self.params, "t", 0.0))
            if x_val < 0:
                return None
        elif hasattr(self.params, "Global_Measures"):
            x_val = self.params.Global_Measures.get(x_name, None)

        if hasattr(self.params, "Global_Measures"):
            y_val = self.params.Global_Measures.get(y_name, None)

        if x_val is None or y_val is None:
            return None
        return float(x_val), float(y_val)

    def _create_measure_plot_state(
        self,
        x_name: str,
        y_name: str,
        x_val: float,
        y_val: float,
    ) -> _MeasurePlotState:
        """PRIVATE: Create the chart state used for one measure plot."""
        plotter = pv.Plotter(
            window_size=self.default_plot_size,
            title=f"{y_name} vs {x_name}",
        )
        chart = pv.Chart2D()
        line = chart.line([x_val], [y_val], color="black")
        chart.x_label = x_name
        chart.y_label = y_name
        chart.title = f"{y_name} vs {x_name}"
        plotter.add_chart(chart)
        plotter.show(interactive_update=True, auto_close=False, interactive=False)
        self._apply_plotter_interaction_mode(plotter)
        self._set_window_icon(plotter)
        return _MeasurePlotState(
            plotter=plotter,
            chart=chart,
            line=line,
            x=[x_val],
            y=[y_val],
        )

    def _update_measure_line(self, plot_state: _MeasurePlotState) -> None:
        """PRIVATE: Refresh a measure chart line using the best supported API."""
        line = plot_state.line
        if hasattr(line, "update"):
            line.update(plot_state.x, plot_state.y)
            return
        if hasattr(line, "set_data"):
            line.set_data(plot_state.x, plot_state.y)
            return
        if hasattr(line, "x") and hasattr(line, "y"):
            try:
                line.x = plot_state.x
                line.y = plot_state.y
                return
            except AttributeError:
                plot_state.line = plot_state.chart.line(
                    plot_state.x,
                    plot_state.y,
                    color="black",
                )
                return

        if hasattr(plot_state.chart, "clear"):
            plot_state.chart.clear()
        plot_state.line = plot_state.chart.line(
            plot_state.x,
            plot_state.y,
            color="black",
        )

    # ------------------------------------------------------------------
    # Internal Helpers: Field Processing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal Helpers: Common Utilities
    # ------------------------------------------------------------------

    def _sanitize_plot_name(self, name: str) -> str:
        """PRIVATE: Sanitize a field or measure name for filenames."""
        return name.replace("/", "_").replace("\\", "_")

    @staticmethod
    def _set_actor_dataset(actor, grid) -> None:
        """PRIVATE: Point an existing VTK actor mapper at replacement topology."""

        mapper = actor.GetMapper()
        if hasattr(mapper, "SetInputDataObject"):
            mapper.SetInputDataObject(grid)
        elif hasattr(mapper, "SetInputData"):
            mapper.SetInputData(grid)
        elif hasattr(mapper, "dataset"):
            mapper.dataset = grid
        else:  # pragma: no cover - defensive backend compatibility guard
            raise TypeError("Plot actor mapper cannot accept replacement topology.")
        if hasattr(mapper, "Update"):
            mapper.Update()

    @staticmethod
    def _capture_plotter_camera_position(plotter):
        """PRIVATE: Snapshot a camera so an in-place actor rebuild preserves it."""

        try:
            return tuple(tuple(values) for values in plotter.camera_position)
        except Exception:
            return None

    @staticmethod
    def _restore_plotter_camera_position(plotter, camera_position) -> None:
        """PRIVATE: Restore a camera snapshot after replacing plot actors."""

        if camera_position is None:
            return
        try:
            plotter.camera_position = camera_position
            if hasattr(plotter, "reset_camera_clipping_range"):
                plotter.reset_camera_clipping_range()
        except Exception:
            pass

    @staticmethod
    def _finalize_plotter_movie_writer(plotter) -> None:
        """PRIVATE: Finalize a movie writer without closing its render window."""

        writer = getattr(plotter, "mwriter", None)
        if writer is None:
            return
        try:
            writer.close()
        except Exception:
            pass
        try:
            plotter.mwriter = None
        except Exception:
            pass

    def _open_movie_safely(self, field_spec: str, plotter, folder_path: str, fps: int) -> None:
        """PRIVATE: Open one continuous field movie and swallow backend failures."""
        movie_path = _os.path.join(
            folder_path,
            f"{self._sanitize_plot_name(field_spec)}.mp4",
        )
        try:
            plotter.open_movie(movie_path, framerate=fps)
        except Exception as exc:
            mprint(f"Warning: Could not open movie for '{field_spec}': {exc}")

    def _save_screenshot_safely(self, plotter, filepath: str, description: str) -> None:
        """PRIVATE: Save one plot screenshot and swallow backend failures."""
        try:
            plotter.screenshot(filepath)
        except Exception as exc:
            mprint(f"Warning: Could not save {description}: {exc}")

    def _close_plotter_safely(self, plotter) -> None:
        """PRIVATE: Close one plotter and ignore backend close failures."""
        try:
            plotter.close()
        except Exception:
            pass

    @staticmethod
    def _set_window_icon(plotter: pv.Plotter) -> None:
        """PRIVATE: Best-effort window icon setup for on-screen PyVista plotters."""
        if pv.OFF_SCREEN or rank != 0:
            return
        try:
            if _MeshPlotting._icon_image is None:
                import vtk

                reader = vtk.vtkPNGReader()
                reader.SetFileName(_ICON_PATH)
                reader.Update()
                _MeshPlotting._icon_image = reader.GetOutput()
            plotter.ren_win.SetIcon(_MeshPlotting._icon_image)
        except Exception:
            pass
