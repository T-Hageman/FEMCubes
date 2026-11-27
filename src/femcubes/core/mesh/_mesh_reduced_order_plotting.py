"""Reduced-order plotting helpers for two-dimensional surface models."""

from __future__ import annotations

import os as _os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from . import _mesh_plotting as _mesh_plotting


@dataclass
class _ReducedOrderFieldPlotState:
    """PRIVATE: Cached rendering state for one reduced-order field plot."""

    plotter: Any
    basal_grid: Any
    basal_fill_grid: Any
    basal_side_grid: Any
    floating_base_grid: Any
    ocean_surface_grid: Any
    field_grid: Any
    basal_actor: Any
    basal_side_actor: Any
    floating_base_actor: Any
    ocean_surface_actor: Any
    field_actor: Any
    time_text: Any
    mesh_epoch: int = 0


class _MeshReducedOrderPlotting:
    """Mix-in for 3-D visualisation of 2-D reduced-order model fields.

    The reduced-order plotter renders the registered basal geometry
    field ``z`` as a grey surface and warps the fields listed in
    ``params.Plot_RO_Fields`` onto the ice surface.  Floating regions use
    a hydrostatic ice base and render that underside in pale blue.  The
    optional ``params.RO_zscale`` factor vertically exaggerates the
    displayed geometry without changing the plotted scalar values.  This
    is intended for shallow-ice / shallow-shelf style models where the
    computational mesh is two-dimensional but the physical geometry is
    most naturally inspected as a surface embedded in three dimensions.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, params: Any) -> None:
        self._ro_field_plots: Dict[str, _ReducedOrderFieldPlotState] = {}
        super().__init__(params)

    def Plot_RO_Fields(self):
        """Plot fields listed in ``params.Plot_RO_Fields`` on the ice surface."""
        self._plot_reduced_order_fields(increment_frame_count=True)

    # ------------------------------------------------------------------
    # Internal Helpers: Driver
    # ------------------------------------------------------------------

    def _start_reduced_order_recording(self, folder_path: str, fps: int) -> None:
        """PRIVATE: Open movie files for existing reduced-order plot windows."""
        if _mesh_plotting.rank != 0 or not getattr(self, "_recording", False):
            return

        for field_spec, state in self._ro_field_plots.items():
            self._open_movie_safely(
                self._reduced_order_movie_name(field_spec),
                state.plotter,
                folder_path,
                fps,
            )

    def _save_reduced_order_plots(self, folder_path: str) -> None:
        """PRIVATE: Save reduced-order plot windows as PNG images."""
        if _mesh_plotting.rank != 0:
            return

        _os.makedirs(folder_path, exist_ok=True)
        for field_spec, state in self._ro_field_plots.items():
            safe_name = self._sanitize_plot_name(self._reduced_order_movie_name(field_spec))
            filepath = _os.path.join(folder_path, f"{safe_name}.png")
            self._save_screenshot_safely(
                state.plotter,
                filepath,
                f"reduced-order plot for '{field_spec}'",
            )

    def _close_reduced_order_plots(self) -> None:
        """PRIVATE: Close all reduced-order plot windows."""
        for state in self._ro_field_plots.values():
            self._close_plotter_safely(state.plotter)
        self._ro_field_plots.clear()

    def _plot_reduced_order_fields(self, *, increment_frame_count: bool) -> None:
        """PRIVATE: Gather and plot reduced-order field surfaces."""
        self._ensure_field_plot_epoch()
        if not hasattr(self.params, "Plot_RO_Fields"):
            return

        plot_fields = self.params.Plot_RO_Fields
        if not plot_fields:
            return

        self._ensure_recording_started()

        if self.mesh.topology.dim != 2:
            _mesh_plotting.mprint(
                "Warning: Reduced-order field plotting is only available for 2-D meshes."
            )
            return

        V_dg1 = self._get_dg_space(1)
        local_ro_data = self._collect_local_reduced_order_plot_data(plot_fields, V_dg1)
        all_rank_ro_data = _mesh_plotting.comm.gather(local_ro_data, root=0)

        needs_mesh_data = None
        if _mesh_plotting.rank == 0:
            needs_mesh_data = self._needs_reduced_order_plot_mesh_data(
                plot_fields,
                all_rank_ro_data,
            )
        needs_mesh_data = _mesh_plotting.comm.bcast(needs_mesh_data, root=0)

        all_topologies = None
        all_cell_types = None
        all_geometries = None
        if needs_mesh_data:
            topology_dg1, cell_types_dg1, geometry_dg1, _ = self._build_plot_mesh_data()
            all_topologies = _mesh_plotting.comm.gather(topology_dg1, root=0)
            all_cell_types = _mesh_plotting.comm.gather(cell_types_dg1, root=0)
            all_geometries = _mesh_plotting.comm.gather(geometry_dg1, root=0)

        if _mesh_plotting.rank != 0:
            return

        for field_spec in plot_fields:
            self._upsert_reduced_order_field_plot(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_ro_data,
            )

        self._update_reduced_order_time_labels()
        self._write_reduced_order_recording_frames(
            increment_frame_count=increment_frame_count,
        )

    def _collect_local_reduced_order_plot_data(self, plot_fields, V_dg):
        """PRIVATE: Build local z/h and reduced-order scalar plot arrays."""
        z_prepared = self.Prepare_Field_For_Plotting("z", V_dg=V_dg)
        h_prepared = self.Prepare_Field_For_Plotting("h", V_dg=V_dg)
        local_data = {"surface": None, "fields": {}}

        if z_prepared is None or h_prepared is None:
            _mesh_plotting.mprint(
                "Warning: Reduced-order plotting requires scalar fields 'z' and 'h'."
            )
            return local_data

        if z_prepared.association != "point" or h_prepared.association != "point":
            _mesh_plotting.mprint(
                "Warning: Reduced-order plotting requires point-associated z and h data."
            )
            return local_data

        # For DG h fields the surface geometry uses a CG1-smoothed version of h
        # (DG1 → CG1 → DG1 round-trip).  After the round-trip, every DG1 node
        # that is geometrically coincident with a node in an adjacent cell carries
        # the same CG1 height, so adjacent cell surfaces meet exactly in 3-D and
        # no side-view gaps appear.  Field scalar colours still use the original
        # DG1 values (collected below) to preserve per-element variation.
        if self.Is_DG_Field("h"):
            h_geom = self._project_h_for_ro_geometry(V_dg)
        else:
            h_geom = None

        local_data["surface"] = {
            "z": self._as_scalar_array(z_prepared.values),
            "h": h_geom if h_geom is not None else self._as_scalar_array(h_prepared.values),
        }
        if self._level_set_plotting_enabled() and "LS" in self.fields:
            LS_prepared = self.Prepare_Field_For_Plotting("LS", V_dg=V_dg)
            if LS_prepared is not None and LS_prepared.association == "point":
                local_data["surface"]["LS"] = self._as_scalar_array(LS_prepared.values)

        log_plot = getattr(self.params, "LogPlot_RO", None)
        for field_idx, field_spec in enumerate(plot_fields):
            prepared = self.Prepare_Field_For_Plotting(field_spec, V_dg=V_dg)
            if prepared is None:
                continue
            if prepared.association != "point":
                _mesh_plotting.mprint(
                    f"Warning: Cannot reduced-order plot '{field_spec}' "
                    "because it is cell-associated data."
                )
                continue

            values, title = self._apply_log_plot_if_requested(
                self._as_scalar_array(prepared.values),
                prepared.title,
                log_plot,
                field_idx,
            )
            local_data["fields"][field_spec] = (values, title, prepared.association)

        return local_data

    # ------------------------------------------------------------------
    # Internal Helpers: Plot Creation and Update
    # ------------------------------------------------------------------

    def _upsert_reduced_order_field_plot(
        self,
        field_spec,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_rank_ro_data,
    ) -> None:
        """PRIVATE: Create or refresh one reduced-order field plot."""
        field_title = self._get_reduced_order_field_title_from_rank_data(
            field_spec,
            all_rank_ro_data,
        )
        if field_title is None:
            return

        state = self._ro_field_plots.get(field_spec)
        if state is None:
            plot_grids = self._build_merged_reduced_order_grids(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_ro_data,
            )
            if plot_grids is None:
                return

            basal_grid, field_grid, floating_base_grid, ocean_surface_grid = plot_grids
            self._ro_field_plots[field_spec] = self._create_reduced_order_plot_state(
                field_spec,
                field_title,
                basal_grid,
                field_grid,
                floating_base_grid,
                ocean_surface_grid,
            )
            return

        if self._plot_state_needs_topology_rebuild(state):
            plot_grids = self._build_merged_reduced_order_grids(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_ro_data,
            )
            if plot_grids is None:
                return
            self._rebuild_reduced_order_plot_state(
                field_spec,
                *plot_grids,
            )
            return

        z_values, h_values, field_values, level_set_values = self._collect_reduced_order_plot_values(
            field_spec,
            all_rank_ro_data,
        )
        if field_values is None:
            return

        self._refresh_reduced_order_plot_state(
            field_spec,
            z_values,
            h_values,
            field_values,
            level_set_values=level_set_values,
        )

    def _create_reduced_order_plot_state(
        self,
        field_spec,
        field_title,
        basal_grid,
        field_grid,
        floating_base_grid,
        ocean_surface_grid,
    ) -> _ReducedOrderFieldPlotState:
        """PRIVATE: Create plotter state for one reduced-order field."""
        plot_title = field_title
        zscale_text = f"z display magnification: {self._get_reduced_order_zscale():g}x"
        plotter = _mesh_plotting.pv.Plotter(
            window_size=self.default_plot_size,
            title=plot_title,
        )
        plotter.add_text(plot_title, position="upper_edge", font_size=14, color="black")
        plotter.add_text(
            zscale_text,
            position=(0.35, 0.92),
            font_size=7,
            color="black",
            viewport=True,
        )

        basal_fill_grid = self._build_reduced_order_basal_fill_grid(basal_grid)
        basal_side_grid = self._build_reduced_order_basal_side_grid(basal_grid)
        basal_actor = plotter.add_mesh(
            basal_fill_grid,
            color="lightgray",
            show_edges=False,
            opacity=1.0,
            smooth_shading=True,
            specular=0.0,
        )
        basal_side_actor = None
        if basal_side_grid is not None:
            basal_side_actor = plotter.add_mesh(
                basal_side_grid,
                color=(0.88, 0.88, 0.88),
                show_edges=False,
                opacity=1.0,
                lighting=False,
            )
        floating_base_actor = None
        if floating_base_grid is not None:
            floating_base_actor = self._add_reduced_order_floating_base_actor(
                plotter,
                floating_base_grid,
            )
        ocean_surface_actor = None
        if ocean_surface_grid is not None:
            ocean_surface_actor = self._add_reduced_order_ocean_surface_actor(
                plotter,
                ocean_surface_grid,
            )
        field_actor = plotter.add_mesh(
            field_grid,
            scalars=field_spec,
            show_edges=False,
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
        self._set_reduced_order_camera(
            plotter,
            basal_grid,
            field_grid,
            floating_base_grid,
            ocean_surface_grid,
        )
        time_actor = plotter.add_text(
            self._format_time_label(),
            position=(0.01, 0.92),
            font_size=10,
            color="black",
            name="time_label",
        )

        if self._recording:
            self._open_movie_safely(
                self._reduced_order_movie_name(field_spec),
                plotter,
                self._recording_folder,
                self._recording_fps,
            )

        plotter.show(interactive_update=True, auto_close=False, interactive=False)
        self._apply_plotter_interaction_mode(plotter)
        self._set_window_icon(plotter)
        return _ReducedOrderFieldPlotState(
            plotter=plotter,
            basal_grid=basal_grid,
            basal_fill_grid=basal_fill_grid,
            basal_side_grid=basal_side_grid,
            floating_base_grid=floating_base_grid,
            ocean_surface_grid=ocean_surface_grid,
            field_grid=field_grid,
            basal_actor=basal_actor,
            basal_side_actor=basal_side_actor,
            floating_base_actor=floating_base_actor,
            ocean_surface_actor=ocean_surface_actor,
            field_actor=field_actor,
            time_text=time_actor,
            mesh_epoch=int(getattr(self, "remesh_epoch", 0)),
        )

    def _rebuild_reduced_order_plot_state(
        self,
        field_spec,
        basal_grid,
        field_grid,
        floating_base_grid,
        ocean_surface_grid,
    ) -> None:
        """PRIVATE: Replace RO actor topology while retaining its plot window."""

        state = self._ro_field_plots[field_spec]
        try:
            basal_fill_grid = self._build_reduced_order_basal_fill_grid(basal_grid)
            basal_side_grid = self._build_reduced_order_basal_side_grid(basal_grid)
            self._set_actor_dataset(state.basal_actor, basal_fill_grid)
            state.basal_side_actor = self._replace_reduced_order_optional_actor(
                state.plotter,
                state.basal_side_actor,
                basal_side_grid,
                lambda plotter, grid: plotter.add_mesh(
                    grid,
                    color=(0.88, 0.88, 0.88),
                    show_edges=False,
                    opacity=1.0,
                    lighting=False,
                ),
            )
            state.floating_base_actor = self._replace_reduced_order_optional_actor(
                state.plotter,
                state.floating_base_actor,
                floating_base_grid,
                self._add_reduced_order_floating_base_actor,
            )
            state.ocean_surface_actor = self._replace_reduced_order_optional_actor(
                state.plotter,
                state.ocean_surface_actor,
                ocean_surface_grid,
                self._add_reduced_order_ocean_surface_actor,
            )
            self._set_actor_dataset(state.field_actor, field_grid)

            state.basal_grid = basal_grid
            state.basal_fill_grid = basal_fill_grid
            state.basal_side_grid = basal_side_grid
            state.floating_base_grid = floating_base_grid
            state.ocean_surface_grid = ocean_surface_grid
            state.field_grid = field_grid

            scalars = self._get_grid_scalars(field_grid, field_spec)
            if scalars.size:
                state.field_actor.GetMapper().SetScalarRange(
                    [float(np.min(scalars)), float(np.max(scalars))]
                )
            if hasattr(state.plotter, "update"):
                state.plotter.update()
            else:
                state.plotter.render()
            state.mesh_epoch = int(getattr(self, "remesh_epoch", 0))
        except Exception:
            _mesh_plotting.mprint(
                f"Warning: Could not rebuild reduced-order plot topology for "
                f"'{field_spec}'. The existing window was retained for a later retry."
            )

    def _replace_reduced_order_optional_actor(
        self,
        plotter,
        actor,
        grid,
        add_actor,
    ):
        """PRIVATE: Update, add, or remove one optional RO surface actor."""

        if grid is None:
            if actor is not None and hasattr(plotter, "remove_actor"):
                plotter.remove_actor(actor)
            return None
        if actor is None:
            return add_actor(plotter, grid)
        self._set_actor_dataset(actor, grid)
        return actor

    def _refresh_reduced_order_plot_state(
        self,
        field_spec,
        z_values,
        h_values,
        field_values,
        level_set_values=None,
    ) -> None:
        """PRIVATE: Refresh the warped geometry and scalars for one RO plot."""
        state = self._ro_field_plots[field_spec]
        basal_points = np.asarray(state.basal_grid.points, dtype=float).copy()
        field_points = np.asarray(state.field_grid.points, dtype=float).copy()

        if len(z_values) != len(basal_points) or len(field_values) != len(field_points):
            _mesh_plotting.mprint(
                f"Warning: Cannot refresh reduced-order plot '{field_spec}' "
                "because gathered value sizes changed."
            )
            return

        zscale = self._get_reduced_order_zscale()
        ice_base, ice_surface, floating_mask, open_water_mask = self._compute_reduced_order_ice_geometry(
            z_values,
            h_values,
            level_set_values,
        )
        basal_points[:, 2] = zscale * z_values
        field_points[:, 2] = zscale * ice_surface
        floating_base_points = field_points.copy()
        floating_base_points[:, 2] = zscale * ice_base
        ocean_surface_points = field_points.copy()
        ocean_surface_points[:, 2] = zscale * self._get_reduced_order_sea_level()
        self._set_grid_points(state.basal_grid, basal_points)
        self._refresh_reduced_order_basal_fill_grid(state.basal_fill_grid, basal_points)
        self._refresh_reduced_order_basal_fill_grid(state.basal_side_grid, basal_points)
        self._refresh_reduced_order_floating_base_grid(
            state,
            floating_base_points,
            floating_mask,
        )
        self._refresh_reduced_order_ocean_surface_grid(
            state,
            ocean_surface_points,
            open_water_mask,
        )
        field_grid = self._build_reduced_order_field_grid(
            state.basal_grid.cells,
            state.basal_grid.celltypes,
            field_points,
            h_values,
            field_spec,
            field_values,
            level_set_values=level_set_values,
        )
        self._copy_grid_data(state.field_grid, field_grid)
        self._mark_grid_modified(state.basal_grid)
        self._mark_grid_modified(state.basal_fill_grid)
        self._mark_grid_modified(state.basal_side_grid)
        self._mark_grid_modified(state.floating_base_grid)
        self._mark_grid_modified(state.ocean_surface_grid)
        self._mark_grid_modified(state.field_grid)

        mapper = state.field_actor.GetMapper()
        mapper.SetScalarRange([float(field_values.min()), float(field_values.max())])

        try:
            if hasattr(state.plotter, "update"):
                state.plotter.update()
            else:
                state.plotter.render()
        except Exception:
            _mesh_plotting.mprint(
                f"Warning: Could not update reduced-order plot for '{field_spec}'. "
                "Window may have been closed."
            )
            self._ro_field_plots.pop(field_spec, None)

    # ------------------------------------------------------------------
    # Internal Helpers: Gathered Data Assembly
    # ------------------------------------------------------------------

    def _needs_reduced_order_plot_mesh_data(self, plot_fields, all_rank_ro_data) -> bool:
        """PRIVATE: Return whether new RO plots need gathered mesh geometry."""
        for field_spec in plot_fields:
            state = self._ro_field_plots.get(field_spec)
            if state is not None and not self._plot_state_needs_topology_rebuild(state):
                continue
            if self._get_reduced_order_field_title_from_rank_data(
                field_spec,
                all_rank_ro_data,
            ) is not None:
                return True
        return False

    def _get_reduced_order_field_title_from_rank_data(
        self,
        field_spec,
        all_rank_ro_data,
    ) -> Optional[str]:
        """PRIVATE: Return the first gathered title for one RO field."""
        for rank_ro_data in all_rank_ro_data:
            if rank_ro_data.get("surface") is None:
                continue
            data = rank_ro_data.get("fields", {}).get(field_spec)
            if data is not None:
                return data[1]
        return None

    def _build_merged_reduced_order_grids(
        self,
        field_spec,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_rank_ro_data,
    ):
        """PRIVATE: Build gathered basal and surface grids for one RO field."""
        if all_topologies is None or all_cell_types is None or all_geometries is None:
            return None

        rank_records = list(
            self._iter_reduced_order_rank_data(
                field_spec,
                all_topologies,
                all_cell_types,
                all_geometries,
                all_rank_ro_data,
            )
        )
        if not rank_records:
            return None

        topologies = []
        cell_types = []
        basal_geometries = []
        field_geometries = []
        floating_base_geometries = []
        ocean_surface_geometries = []
        floating_masks = []
        open_water_masks = []
        field_values = []
        h_values = []
        level_set_values = []
        point_offset = 0

        for (
            topology,
            rank_cell_types,
            basal_geometry,
            field_geometry,
            floating_base_geometry,
            ocean_surface_geometry,
            floating_mask,
            open_water_mask,
            rank_h_values,
            rank_level_set_values,
            values,
        ) in rank_records:
            topologies.append(self._offset_vtk_topology(topology, point_offset))
            cell_types.append(np.asarray(rank_cell_types))
            basal_geometries.append(basal_geometry)
            field_geometries.append(field_geometry)
            floating_base_geometries.append(floating_base_geometry)
            ocean_surface_geometries.append(ocean_surface_geometry)
            floating_masks.append(floating_mask)
            open_water_masks.append(open_water_mask)
            h_values.append(rank_h_values)
            if rank_level_set_values is not None:
                level_set_values.append(rank_level_set_values)
            field_values.append(values)
            point_offset += len(basal_geometry)

        merged_topology = np.concatenate(topologies)
        merged_cell_types = np.concatenate(cell_types)
        floating_base_geometry = np.vstack(floating_base_geometries)
        ocean_surface_geometry = np.vstack(ocean_surface_geometries)
        floating_mask = np.concatenate(floating_masks)
        open_water_mask = np.concatenate(open_water_masks)
        merged_field_geometry = np.vstack(field_geometries)
        merged_h_values = np.concatenate(h_values)
        merged_field_values = np.concatenate(field_values)
        merged_level_set_values = None
        if len(level_set_values) == len(rank_records):
            merged_level_set_values = np.concatenate(level_set_values)
        basal_grid = _mesh_plotting.pv.UnstructuredGrid(
            merged_topology,
            merged_cell_types,
            np.vstack(basal_geometries),
        )
        field_grid = self._build_reduced_order_field_grid(
            merged_topology,
            merged_cell_types,
            merged_field_geometry,
            merged_h_values,
            field_spec,
            merged_field_values,
            level_set_values=merged_level_set_values,
        )
        floating_base_grid = self._build_reduced_order_floating_base_grid(
            merged_topology,
            floating_base_geometry,
            floating_mask,
        )
        ocean_surface_grid = self._build_reduced_order_ocean_surface_grid(
            merged_topology,
            ocean_surface_geometry,
            open_water_mask,
        )
        return basal_grid, field_grid, floating_base_grid, ocean_surface_grid

    def _iter_reduced_order_rank_data(
        self,
        field_spec,
        all_topologies,
        all_cell_types,
        all_geometries,
        all_rank_ro_data,
    ):
        """PRIVATE: Yield valid rank-local RO data for one field."""
        zscale = self._get_reduced_order_zscale()

        for topology, cell_types, geometry, rank_ro_data in zip(
            all_topologies,
            all_cell_types,
            all_geometries,
            all_rank_ro_data,
        ):
            surface = rank_ro_data.get("surface")
            data = rank_ro_data.get("fields", {}).get(field_spec)
            if surface is None or data is None:
                continue

            base_xy = self._as_3d_plot_points(geometry)
            z_values = self._as_scalar_array(surface["z"])
            h_values = self._as_scalar_array(surface["h"])
            level_set_values = self._get_reduced_order_surface_level_set_values(surface)
            field_values = self._as_scalar_array(data[0])
            if not self._ro_value_lengths_match(
                field_spec,
                len(base_xy),
                z_values,
                h_values,
                field_values,
                level_set_values,
            ):
                continue

            ice_base, ice_surface, floating_mask, open_water_mask = self._compute_reduced_order_ice_geometry(
                z_values,
                h_values,
                level_set_values,
            )
            basal_geometry = base_xy.copy()
            field_geometry = base_xy.copy()
            floating_base_geometry = base_xy.copy()
            ocean_surface_geometry = base_xy.copy()
            basal_geometry[:, 2] = zscale * z_values
            field_geometry[:, 2] = zscale * ice_surface
            floating_base_geometry[:, 2] = zscale * ice_base
            ocean_surface_geometry[:, 2] = zscale * self._get_reduced_order_sea_level()
            yield (
                topology,
                cell_types,
                basal_geometry,
                field_geometry,
                floating_base_geometry,
                ocean_surface_geometry,
                floating_mask,
                open_water_mask,
                h_values,
                level_set_values,
                field_values,
            )

    def _collect_reduced_order_plot_values(self, field_spec, all_rank_ro_data):
        """PRIVATE: Concatenate gathered z, h, and scalar values for refresh."""
        z_values = []
        h_values = []
        level_set_values = []
        field_values = []
        have_level_set_values = True

        for rank_ro_data in all_rank_ro_data:
            surface = rank_ro_data.get("surface")
            data = rank_ro_data.get("fields", {}).get(field_spec)
            if surface is None or data is None:
                continue

            rank_z = self._as_scalar_array(surface["z"])
            rank_h = self._as_scalar_array(surface["h"])
            rank_level_set = self._get_reduced_order_surface_level_set_values(surface)
            rank_field = self._as_scalar_array(data[0])
            if rank_z.size == 0 or rank_h.size == 0 or rank_field.size == 0:
                continue
            if len(rank_z) != len(rank_h) or len(rank_z) != len(rank_field):
                _mesh_plotting.mprint(
                    f"Warning: Cannot refresh reduced-order plot '{field_spec}' "
                    "because gathered z, h, and field sizes differ."
                )
                return None, None, None, None

            if rank_level_set is None:
                have_level_set_values = False
            elif len(rank_level_set) == len(rank_z):
                level_set_values.append(rank_level_set)
            else:
                _mesh_plotting.mprint(
                    f"Warning: Cannot refresh reduced-order plot '{field_spec}' "
                    "because gathered LS and geometry sizes differ."
                )
                return None, None, None, None

            z_values.append(rank_z)
            h_values.append(rank_h)
            field_values.append(rank_field)

        if not field_values:
            return None, None, None, None
        concatenated_level_set = None
        if have_level_set_values and len(level_set_values) == len(field_values):
            concatenated_level_set = np.concatenate(level_set_values)
        return (
            np.concatenate(z_values),
            np.concatenate(h_values),
            np.concatenate(field_values),
            concatenated_level_set,
        )

    # ------------------------------------------------------------------
    # Internal Helpers: Plot Lifecycle
    # ------------------------------------------------------------------

    def _update_reduced_order_time_labels(self) -> None:
        """PRIVATE: Refresh the per-window time label on RO plots."""
        label = self._format_time_label()
        for state in self._ro_field_plots.values():
            state.time_text = state.plotter.add_text(
                label,
                position=(0.01, 0.92),
                font_size=10,
                color="black",
                name="time_label",
            )

    def _write_reduced_order_recording_frames(self, *, increment_frame_count: bool) -> None:
        """PRIVATE: Write video frames for active RO plots."""
        if not getattr(self, "_recording", False) or not self._ro_field_plots:
            return

        for state in self._ro_field_plots.values():
            try:
                state.plotter.write_frame()
            except Exception:
                pass
        if increment_frame_count:
            self._frame_count += 1

    @staticmethod
    def _reduced_order_movie_name(field_spec: str) -> str:
        """PRIVATE: Return the filename stem used for one RO plot."""
        return f"RO_{field_spec}"

    def _add_reduced_order_floating_base_actor(self, plotter, floating_base_grid):
        """PRIVATE: Add the pale-blue floating underside actor."""
        return plotter.add_mesh(
            floating_base_grid,
            color=(0.70, 0.88, 1.0),
            show_edges=False,
            opacity=1.0,
            lighting=False,
        )

    def _add_reduced_order_ocean_surface_actor(self, plotter, ocean_surface_grid):
        """PRIVATE: Add the open-water surface actor."""
        return plotter.add_mesh(
            ocean_surface_grid,
            color=(0.58, 0.80, 1.0),
            show_edges=False,
            opacity=1.0,
            lighting=False,
        )

    def _set_reduced_order_camera(self, plotter, basal_grid, field_grid, *extra_grids) -> None:
        """PRIVATE: Set an oblique RO camera with +x right and +y up."""
        points = self._collect_reduced_order_camera_points(basal_grid, field_grid, *extra_grids)
        if points.size == 0:
            plotter.view_isometric()
            return

        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        center = 0.5 * (lower + upper)
        extent = np.maximum(upper - lower, 1.0)
        distance = 1.8 * float(np.max(extent))

        horizontal_direction = np.array([1.0, -1.0, 0.0], dtype=float)
        horizontal_direction /= np.linalg.norm(horizontal_direction)
        isometric_elevation = np.arctan(1.0 / np.sqrt(2.0))
        camera_elevation = isometric_elevation - np.deg2rad(10.0)
        view_direction = (
            np.cos(camera_elevation) * horizontal_direction
            + np.sin(camera_elevation) * np.array([0.0, 0.0, 1.0], dtype=float)
        )
        position = center + distance * view_direction
        camera_position = (
            tuple(position),
            tuple(center),
            (0.0, 0.0, 1.0),
        )

        try:
            plotter.camera_position = camera_position
        except Exception:
            camera = getattr(plotter, "camera", None)
            if camera is not None:
                try:
                    if hasattr(camera, "SetPosition"):
                        camera.SetPosition(*camera_position[0])
                    else:
                        camera.position = camera_position[0]
                    if hasattr(camera, "SetFocalPoint"):
                        camera.SetFocalPoint(*camera_position[1])
                    else:
                        camera.focal_point = camera_position[1]
                    if hasattr(camera, "SetViewUp"):
                        camera.SetViewUp(*camera_position[2])
                    else:
                        camera.view_up = camera_position[2]
                except Exception:
                    plotter.view_isometric()
                    return

        if hasattr(plotter, "reset_camera_clipping_range"):
            try:
                plotter.reset_camera_clipping_range()
            except Exception:
                pass

    @staticmethod
    def _collect_reduced_order_camera_points(*grids) -> np.ndarray:
        """PRIVATE: Collect available grid points for RO camera framing."""
        point_blocks = []
        for grid in grids:
            grid_points = getattr(grid, "points", None)
            if grid_points is None:
                continue
            points = np.asarray(grid_points, dtype=float)
            if points.ndim == 2 and points.shape[0] > 0 and points.shape[1] >= 3:
                point_blocks.append(points[:, :3])

        if not point_blocks:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(point_blocks)

    def _compute_reduced_order_ice_geometry(self, z_values, h_values, level_set_values=None):
        """PRIVATE: Return ice base, surface, floating mask, and open-water mask."""
        z_values = self._as_scalar_array(z_values)
        h_values = self._as_scalar_array(h_values)
        display_h = np.maximum(h_values, 0.0)
        has_visible_ice = self._get_reduced_order_visible_ice_mask(
            h_values,
            level_set_values=level_set_values,
        )
        rho_ice, rho_ocean = self._get_reduced_order_densities()
        sea_level = self._get_reduced_order_sea_level()

        flotation = rho_ice * display_h + rho_ocean * (z_values - sea_level)
        floating = (flotation <= 0.0) & has_visible_ice
        open_water = (~has_visible_ice) & (z_values < sea_level)
        floating_base = sea_level - (rho_ice / rho_ocean) * display_h
        ice_base = np.where(floating, floating_base, z_values)
        ice_surface = ice_base + display_h
        return ice_base, ice_surface, floating, open_water

    def _build_reduced_order_field_grid(
        self,
        topology,
        cell_types,
        field_points,
        h_values,
        field_spec,
        field_values,
        *,
        level_set_values=None,
    ):
        """PRIVATE: Build the coloured top-surface grid, omitting very thin ice."""
        visible_topology, visible_cell_types = self._filter_reduced_order_surface_topology(
            topology,
            cell_types,
            h_values,
            level_set_values=level_set_values,
        )
        field_grid = _mesh_plotting.pv.UnstructuredGrid(
            visible_topology,
            visible_cell_types,
            field_points,
        )
        field_grid.point_data[field_spec] = field_values
        return field_grid

    def _filter_reduced_order_surface_topology(self, topology, cell_types, h_values, *, level_set_values=None):
        """PRIVATE: Remove top-surface cells where any vertex is outside visible ice."""
        surface_visible = self._get_reduced_order_visible_ice_mask(
            h_values,
            level_set_values=level_set_values,
        )
        cell_types = np.asarray(cell_types)
        visible_topology = []
        visible_cell_types = []

        for cell_index, cell in enumerate(self._iter_vtk_topology_cells(topology)):
            if len(cell) < 3:
                continue
            cell_points = np.asarray(cell, dtype=int)
            if np.all(surface_visible[cell_points]):
                visible_topology.extend([len(cell), *cell])
                visible_cell_types.append(cell_types[cell_index])

        return (
            np.asarray(visible_topology, dtype=np.int64),
            np.asarray(visible_cell_types, dtype=cell_types.dtype),
        )

    def _get_reduced_order_visible_ice_mask(self, h_values, *, level_set_values=None) -> np.ndarray:
        """PRIVATE: Return the point-wise mask for visible ice in RO plots."""
        h_values = self._as_scalar_array(h_values)
        thickness_visible = h_values >= self._get_reduced_order_min_surface_thickness()
        if level_set_values is None:
            return thickness_visible

        level_set_values = self._as_scalar_array(level_set_values)
        if len(level_set_values) != len(h_values):
            return thickness_visible

        level_set_active = self._smooth_level_set_heaviside_values(level_set_values) > 0.5
        return thickness_visible & level_set_active

    def _get_reduced_order_surface_level_set_values(self, surface):
        """PRIVATE: Return optional point-aligned LS values from a gathered RO surface."""
        if surface is None or "LS" not in surface:
            return None
        return self._as_scalar_array(surface["LS"])

    def _project_h_for_ro_geometry(self, V_dg):
        """PRIVATE: Return h values for RO surface geometry via a CG1 round-trip.

        DG1 h has independent nodes at every cell vertex, so adjacent cells can
        have slightly different surface heights at the same geometric point.  When
        viewed from the side this creates visible cracks between cells.

        This method applies the round-trip

            DG1 h  →  CG1 h_smooth  →  DG1 h_geom

        * ``DG1 → CG1``: ``Prepare_Field_For_Plotting`` projects the DG1 h field
          into a CG1 space.  CG1 is continuous, so coincident nodes from adjacent
          cells converge to a single value.
        * ``CG1 → DG1``: a ``fem.Expression`` re-evaluates the smooth CG1 field at
          every DG1 interpolation point.  Because CG1 is continuous, all DG1 nodes
          that share a geometric vertex receive *exactly* the same CG1 value
          regardless of which cell they belong to.

        The returned array is in DG1-topology DOF order matching ``V_dg``, so it
        slots directly into the existing surface-geometry pipeline.
        """
        from dolfinx import fem as _fem

        V_cg1 = self._get_cg_space(1)

        # Step 1: DG1 → CG1.  Re-uses the output projection cache.
        # _project_source_to_output_space uses fem.Expression for DG source fields
        # (cell-local evaluation), so each CG1 DOF is set from the cell that owns
        # it during the cell-by-cell scatter.
        h_cg1_prepared = self.Prepare_Field_For_Plotting("h", V_dg=V_cg1)
        if h_cg1_prepared is None:
            return None
        h_cg1_func = self._get_cached_output_projection("h", V_cg1)[0]

        # Step 2: CG1 → DG1.  Evaluate the smooth CG1 field at each DG1
        # interpolation point.  CG1 continuity guarantees coincident DG1 nodes
        # get the same value, eliminating inter-cell height differences.
        ro_geom_key = ("_ro_h_geom", id(V_dg))
        h_dg1_geom = self._output_projection_cache.get(ro_geom_key)
        if h_dg1_geom is None:
            h_dg1_geom = _fem.Function(V_dg)
            self._output_projection_cache[ro_geom_key] = h_dg1_geom

        expr = _fem.Expression(h_cg1_func, V_dg.element.interpolation_points)
        h_dg1_geom.interpolate(expr)
        return h_dg1_geom.x.array.copy()

    def _get_reduced_order_densities(self) -> tuple[float, float]:
        """PRIVATE: Return ice and ocean densities used for RO flotation plots."""
        materials = getattr(self.params, "Materials", {})
        ice_params = materials.get("Ice", {}) if isinstance(materials, dict) else {}
        water_params = materials.get("Water", {}) if isinstance(materials, dict) else {}
        rho_ice = float(ice_params.get("Density", 917.0))
        rho_ocean = float(water_params.get("Density", 1030.0))
        if rho_ice <= 0.0 or rho_ocean <= 0.0:
            raise ValueError("RO flotation plotting requires positive ice and ocean densities.")
        return rho_ice, rho_ocean

    def _get_reduced_order_sea_level(self) -> float:
        """PRIVATE: Return the ocean surface elevation used for RO plots."""
        if hasattr(self.params, "RO_sea_level"):
            return float(self.params.RO_sea_level)

        materials = getattr(self.params, "Materials", {})
        water_params = materials.get("Water", {}) if isinstance(materials, dict) else {}
        return float(water_params.get("WaterHeight", 0.0))

    def _get_reduced_order_min_surface_thickness(self) -> float:
        """PRIVATE: Return the minimum thickness rendered as a top ice surface."""
        min_thickness = float(getattr(self.params, "RO_min_surface_thickness", 0.01))
        if not np.isfinite(min_thickness) or min_thickness < 0.0:
            raise ValueError("RO_min_surface_thickness must be a non-negative finite number.")
        return min_thickness

    def _build_reduced_order_basal_fill_grid(self, basal_grid):
        """PRIVATE: Build top and bottom bed faces from bed surface to min(z)."""
        points = np.asarray(getattr(basal_grid, "points", []), dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
            return basal_grid

        surface_cells = self._extract_reduced_order_surface_cells(basal_grid)
        if not surface_cells:
            return basal_grid

        fill_points = self._build_reduced_order_basal_fill_points(points[:, :3])
        faces = self._build_reduced_order_basal_fill_faces(points[:, :3], surface_cells)
        if faces.size == 0:
            return basal_grid
        return _mesh_plotting.pv.PolyData(fill_points, faces=faces)

    def _build_reduced_order_basal_side_grid(self, basal_grid):
        """PRIVATE: Build the lighter vertical side walls of the bed shell."""
        points = np.asarray(getattr(basal_grid, "points", []), dtype=float)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
            return None

        surface_cells = self._extract_reduced_order_surface_cells(basal_grid)
        if not surface_cells:
            return None

        fill_points = self._build_reduced_order_basal_fill_points(points[:, :3])
        faces = self._build_reduced_order_basal_side_faces(points[:, :3], surface_cells)
        if faces.size == 0:
            return None
        return _mesh_plotting.pv.PolyData(fill_points, faces=faces)

    def _build_reduced_order_floating_base_grid(
        self,
        topology,
        floating_base_points,
        floating_mask,
    ):
        """PRIVATE: Build a pale-blue underside surface for floating ice cells."""
        points = np.asarray(floating_base_points, dtype=float)
        floating_mask = np.asarray(floating_mask, dtype=bool).reshape(-1)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
            return None
        if len(floating_mask) != len(points) or not np.any(floating_mask):
            return None

        faces = []
        for cell in self._iter_vtk_topology_cells(topology):
            if len(cell) < 3:
                continue
            if np.all(floating_mask[np.asarray(cell, dtype=int)]):
                faces.extend([len(cell), *cell])

        if not faces:
            return None
        return _mesh_plotting.pv.PolyData(points[:, :3], faces=np.asarray(faces, dtype=np.int64))

    def _build_reduced_order_ocean_surface_grid(
        self,
        topology,
        ocean_surface_points,
        open_water_mask,
    ):
        """PRIVATE: Build a water-height surface over ice-free submerged bed cells."""
        points = np.asarray(ocean_surface_points, dtype=float)
        open_water_mask = np.asarray(open_water_mask, dtype=bool).reshape(-1)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
            return None
        if len(open_water_mask) != len(points) or not np.any(open_water_mask):
            return None

        faces = []
        for cell in self._iter_vtk_topology_cells(topology):
            if len(cell) < 3:
                continue
            if np.all(open_water_mask[np.asarray(cell, dtype=int)]):
                faces.extend([len(cell), *cell])

        if not faces:
            return None
        return _mesh_plotting.pv.PolyData(points[:, :3], faces=np.asarray(faces, dtype=np.int64))

    def _refresh_reduced_order_floating_base_grid(
        self,
        state,
        floating_base_points,
        floating_mask,
    ) -> None:
        """PRIVATE: Refresh or create the floating ice underside mesh."""
        floating_base_grid = self._build_reduced_order_floating_base_grid(
            state.basal_grid.cells,
            floating_base_points,
            floating_mask,
        )
        if floating_base_grid is None:
            if state.floating_base_grid is None:
                return
            empty_grid = _mesh_plotting.pv.PolyData(
                np.asarray(floating_base_points, dtype=float)[:, :3],
                faces=np.asarray([], dtype=np.int64),
            )
            self._copy_grid_data(state.floating_base_grid, empty_grid)
            return

        if state.floating_base_grid is None:
            state.floating_base_grid = floating_base_grid
            state.floating_base_actor = self._add_reduced_order_floating_base_actor(
                state.plotter,
                state.floating_base_grid,
            )
            return

        self._copy_grid_data(state.floating_base_grid, floating_base_grid)

    def _refresh_reduced_order_ocean_surface_grid(
        self,
        state,
        ocean_surface_points,
        open_water_mask,
    ) -> None:
        """PRIVATE: Refresh or create the open-water surface mesh."""
        ocean_surface_grid = self._build_reduced_order_ocean_surface_grid(
            state.basal_grid.cells,
            ocean_surface_points,
            open_water_mask,
        )
        if ocean_surface_grid is None:
            if state.ocean_surface_grid is None:
                return
            empty_grid = _mesh_plotting.pv.PolyData(
                np.asarray(ocean_surface_points, dtype=float)[:, :3],
                faces=np.asarray([], dtype=np.int64),
            )
            self._copy_grid_data(state.ocean_surface_grid, empty_grid)
            return

        if state.ocean_surface_grid is None:
            state.ocean_surface_grid = ocean_surface_grid
            state.ocean_surface_actor = self._add_reduced_order_ocean_surface_actor(
                state.plotter,
                state.ocean_surface_grid,
            )
            return

        self._copy_grid_data(state.ocean_surface_grid, ocean_surface_grid)

    def _refresh_reduced_order_basal_fill_grid(self, basal_fill_grid, basal_points) -> None:
        """PRIVATE: Refresh the existing filled basal shell coordinates."""
        if basal_fill_grid is None:
            return
        fill_points = self._build_reduced_order_basal_fill_points(basal_points)
        if len(fill_points) != len(getattr(basal_fill_grid, "points", [])):
            return
        self._set_grid_points(basal_fill_grid, fill_points)

    @staticmethod
    def _build_reduced_order_basal_fill_points(points: np.ndarray) -> np.ndarray:
        """PRIVATE: Return top and flat-bottom points for the basal shell."""
        top_points = np.asarray(points, dtype=float)[:, :3]
        bottom_points = top_points.copy()
        bottom_points[:, 2] = float(np.min(top_points[:, 2]))
        return np.vstack((top_points, bottom_points))

    def _build_reduced_order_basal_fill_faces(self, points, surface_cells) -> np.ndarray:
        """PRIVATE: Build top and bottom faces for the basal fill."""
        point_count = len(points)
        face_entries = []

        for cell in surface_cells:
            if len(cell) < 3:
                continue

            face_entries.extend([len(cell), *cell])
            bottom_cell = [idx + point_count for idx in reversed(cell)]
            face_entries.extend([len(bottom_cell), *bottom_cell])

        return np.asarray(face_entries, dtype=np.int64)

    def _build_reduced_order_basal_side_faces(self, points, surface_cells) -> np.ndarray:
        """PRIVATE: Build only the vertical side faces for the basal shell."""
        point_count = len(points)
        face_entries = []
        for a, b in self._collect_reduced_order_boundary_edges(points, surface_cells):
            face_entries.extend([4, a, b, b + point_count, a + point_count])
        return np.asarray(face_entries, dtype=np.int64)

    def _collect_reduced_order_boundary_edges(self, points, surface_cells) -> list[tuple[int, int]]:
        """PRIVATE: Return directed edges that occur on the shell boundary."""
        boundary_edge_counts = {}
        boundary_edge_vertices = {}

        for cell in surface_cells:
            if len(cell) < 3:
                continue

            for edge in self._iter_polygon_edges(cell):
                edge_key = self._reduced_order_edge_key(points, edge)
                boundary_edge_counts[edge_key] = boundary_edge_counts.get(edge_key, 0) + 1
                boundary_edge_vertices.setdefault(edge_key, edge)

        return [
            boundary_edge_vertices[edge_key]
            for edge_key, count in boundary_edge_counts.items()
            if count == 1
        ]

    @staticmethod
    def _iter_vtk_topology_cells(topology):
        """PRIVATE: Yield VTK cell point-index lists from flat or table topology."""
        topology = np.asarray(topology, dtype=np.int64)
        if topology.ndim == 2:
            for row in topology:
                cell_size = int(row[0])
                if cell_size > 0:
                    yield row[1 : 1 + cell_size].astype(int).tolist()
            return

        cursor = 0
        while cursor < topology.size:
            cell_size = int(topology[cursor])
            first = cursor + 1
            past = first + cell_size
            if cell_size > 0 and past <= topology.size:
                yield topology[first:past].astype(int).tolist()
            cursor = past

    @staticmethod
    def _extract_reduced_order_surface_cells(grid) -> list[list[int]]:
        """PRIVATE: Extract polygon connectivity from a PyVista grid."""
        cells = getattr(grid, "cells", None)
        if cells is None:
            return []

        flat_cells = np.asarray(cells, dtype=np.int64).reshape(-1)
        surface_cells = []
        cursor = 0
        while cursor < flat_cells.size:
            cell_size = int(flat_cells[cursor])
            first = cursor + 1
            past = first + cell_size
            if cell_size >= 3 and past <= flat_cells.size:
                surface_cells.append(flat_cells[first:past].astype(int).tolist())
            cursor = past
        return surface_cells

    @staticmethod
    def _iter_polygon_edges(cell) -> list[tuple[int, int]]:
        """PRIVATE: Return directed edges for one polygon cell."""
        return [
            (int(cell[idx]), int(cell[(idx + 1) % len(cell)]))
            for idx in range(len(cell))
        ]

    @staticmethod
    def _reduced_order_edge_key(points, edge) -> tuple[tuple[float, float, float], ...]:
        """PRIVATE: Return a coordinate key for an edge, ignoring direction."""
        a, b = edge
        endpoints = (
            tuple(np.round(points[a, :3], decimals=12)),
            tuple(np.round(points[b, :3], decimals=12)),
        )
        return tuple(sorted(endpoints))

    def _get_reduced_order_zscale(self) -> float:
        """PRIVATE: Return the positive vertical display scale for RO plots."""
        zscale = float(getattr(self.params, "RO_zscale", 1.0))
        if not np.isfinite(zscale) or zscale <= 0.0:
            raise ValueError("RO_zscale must be a positive finite number.")
        return zscale

    # ------------------------------------------------------------------
    # Internal Helpers: Arrays
    # ------------------------------------------------------------------

    @staticmethod
    def _as_scalar_array(values) -> np.ndarray:
        """PRIVATE: Return values as a one-dimensional float array."""
        return np.asarray(values, dtype=float).reshape(-1)

    @staticmethod
    def _as_3d_plot_points(geometry) -> np.ndarray:
        """PRIVATE: Return plotting coordinates padded to three dimensions."""
        points = np.asarray(geometry, dtype=float)
        if points.ndim != 2 or points.shape[1] < 2:
            return np.zeros((0, 3), dtype=float)

        plot_points = np.zeros((points.shape[0], 3), dtype=float)
        cols = min(points.shape[1], 3)
        plot_points[:, :cols] = points[:, :cols]
        return plot_points

    @staticmethod
    def _set_grid_points(grid, points) -> None:
        """PRIVATE: Replace a PyVista grid's point coordinates."""
        try:
            grid.points = points
        except Exception:
            grid.points[:] = points

    @staticmethod
    def _copy_grid_data(target_grid, source_grid) -> None:
        """PRIVATE: Replace a PyVista grid's geometry/connectivity in-place."""
        if hasattr(target_grid, "copy_from"):
            target_grid.copy_from(source_grid)
            return
        target_grid.points = source_grid.points

    @staticmethod
    def _mark_grid_modified(grid) -> None:
        """PRIVATE: Notify the plotting backend that a grid changed."""
        if hasattr(grid, "Modified"):
            grid.Modified()

    @staticmethod
    def _ro_value_lengths_match(
        field_spec,
        point_count: int,
        z_values,
        h_values,
        field_values,
        level_set_values=None,
    ) -> bool:
        """PRIVATE: Validate point-aligned z, h, optional LS, and scalar arrays."""
        if point_count == 0:
            return False
        if len(z_values) == point_count and len(h_values) == point_count:
            level_set_matches = level_set_values is None or len(level_set_values) == point_count
            if len(field_values) == point_count and level_set_matches:
                return True

        _mesh_plotting.mprint(
            f"Warning: Cannot reduced-order plot '{field_spec}' because geometry, "
            "z, h, LS, and field value sizes do not match."
        )
        return False
