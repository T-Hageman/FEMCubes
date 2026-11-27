"""Tests for mesh plotting functionality."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh._mesh_dofs import FieldOutputView
from femcubes.core.mesh.refinement import (
    FieldThresholdIndicatorConfig,
    RefinementConfig,
)


@pytest.fixture
def Mesh_Test_Params():
    """Create a Params instance configured for mesh plotting tests."""

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2

    return TestParams()


@pytest.fixture
def tracked_pyvista(monkeypatch):
    """Use real pyvista but wrap Plotter to capture interactions off-screen."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    original_plotter = pv.Plotter

    class TrackingPlotter(original_plotter):  # type: ignore[misc]
        instances = []
        fail_open_movie_for_next = False

        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs.setdefault("off_screen", True)
            super().__init__(*args, **kwargs)
            self.added_meshes = []
            self.labels = []
            self.texts = []
            self.legend_count = 0
            self.axes_shown = 0
            self.viewed_xy = 0
            self.viewed_iso = 0
            self.shows = []
            self.updates = 0
            self.should_fail_update = False
            self.should_fail_open_movie = TrackingPlotter.fail_open_movie_for_next
            TrackingPlotter.fail_open_movie_for_next = False
            self.should_fail_write_frame = False
            self.should_fail_close = False
            self.should_fail_screenshot = False
            self.opened_movies = []
            self.written_frames = 0
            self.screenshots = []
            self.close_calls = 0
            self.disable_calls = 0
            self.render_suppression_states = []
            TrackingPlotter.instances.append(self)

        def render(self):
            states = getattr(self, "render_suppression_states", None)
            if states is not None:
                states.append(bool(getattr(self, "_suppress_rendering", False)))
            return super().render()

        def add_mesh(self, *args, **kwargs):
            actor = super().add_mesh(*args, **kwargs)
            self.added_meshes.append((args, kwargs, actor))
            return actor

        def add_point_labels(self, *args, **kwargs):
            result = super().add_point_labels(*args, **kwargs)
            self.labels.append((args, kwargs))
            return result

        def add_text(self, *args, **kwargs):
            result = super().add_text(*args, **kwargs)
            self.texts.append((args, kwargs))
            return result

        def show_axes(self):
            self.axes_shown += 1
            return super().show_axes()

        def view_xy(self):
            self.viewed_xy += 1
            return super().view_xy()

        def view_isometric(self):
            self.viewed_iso += 1
            return super().view_isometric()

        def add_legend(self):
            self.legend_count += 1
            return super().add_legend()

        def show(self, *args, **kwargs):
            self.shows.append((args, kwargs))
            return None

        def update(self):
            self.updates += 1
            if self.should_fail_update:
                raise RuntimeError("plotter update failed")
            return None

        def open_movie(self, path, framerate):
            if self.should_fail_open_movie:
                raise RuntimeError("cannot open movie")
            self.opened_movies.append((path, framerate))
            return None

        def write_frame(self):
            self.written_frames += 1
            if self.should_fail_write_frame:
                raise RuntimeError("cannot write frame")
            return None

        def screenshot(self, path):
            if self.should_fail_screenshot:
                raise RuntimeError("cannot save screenshot")
            self.screenshots.append(path)
            return None

        def close(self):
            self.close_calls += 1
            if self.should_fail_close:
                raise RuntimeError("cannot close")
            return super().close()

        def disable(self):
            self.disable_calls += 1
            return super().disable()

    TrackingPlotter.instances = []
    monkeypatch.setattr(pv, "Plotter", TrackingPlotter)
    yield TrackingPlotter
    TrackingPlotter.instances = []


@pytest.fixture
def mesh_for_plotting(Mesh_Test_Params):
    """Create a lightweight rectangular mesh for plotting tests."""
    from femcubes.core.mesh.mesh import Mesh

    return Mesh(Mesh_Test_Params)


@pytest.fixture
def mesh_3d_for_plotting(Mesh_Test_Params):
    """Create a 3-D mesh with scalar and vector fields for plotting tests."""
    from femcubes.core.mesh.mesh import Mesh

    params = Mesh_Test_Params
    params.Lx = 2.0
    params.Ly = 2.0
    params.Lz = 1.0
    params.dx = 0.5
    params.dim = 3

    mesh = Mesh(params)
    mesh.Add_Vector_Field("v3d", "Velocity 3D", "CG", 1, 3)
    mesh.Add_Field("theta3d", "Temperature 3D", "CG", 1)

    mesh.fields["theta3d"].x.array[:] = 1.5
    mesh.fields["v3d"].x.array[:] = np.linspace(0.0, 1.0, mesh.fields["v3d"].x.array.size)
    return mesh


@pytest.fixture
def mesh_with_plot_fields(mesh_for_plotting):
    """Create a mesh populated with scalar and vector fields for Plot_Fields."""
    mesh = mesh_for_plotting
    mesh.Add_Vector_Field("v", "Velocity", "CG", 1, 2)
    mesh.Add_Field("p", "Pressure", "CG", 1)
    mesh.Add_Field("temperature", "Temperature", "CG", 1)
    mesh.Add_Field("phasefield_Hist", "Phasefield History", "CG", 1)

    mesh.Add_Vector_TestFunction("v", nDim=2, step=0)
    mesh.Add_TestFunction("p", step=0)
    mesh.finalize_test_functions()

    mesh.fields["temperature"].x.array[:] = 2.0
    mesh.fields["p"].x.array[:] = 3.0
    mesh.fields["phasefield_Hist"].x.array[:] = 4.0
    mesh.fields["v"].x.array[:] = np.linspace(0.0, 1.0, mesh.fields["v"].x.array.size)
    return mesh


def _add_reduced_order_geometry_fields(mesh, *, z=-10.0, h=7.5):
    """Populate the z/h fields needed by reduced-order plotting tests."""
    if "z" not in mesh.fields:
        mesh.Add_Field("z", "Basal geometry", "CG", 1)
    if "h" not in mesh.fields:
        mesh.Add_Field("h", "Thickness", "CG", 1)
    mesh.fields["z"].x.array[:] = z
    mesh.fields["h"].x.array[:] = h


def _add_level_set_plotting_field(mesh, *, front_x=None):
    """Populate an LS field and params entries for LS-aware plotting tests."""
    if "LS" not in mesh.fields:
        mesh.Add_Field("LS", "Level Set", "CG", 1)

    solution_steps = dict(getattr(mesh.params, "Solution_Steps", {}))
    solution_steps["LS"] = solution_steps.get("LS", 0)
    mesh.params.Solution_Steps = solution_steps

    field_orders = dict(getattr(mesh.params, "Field_Orders", {}))
    field_orders["LS"] = 1
    mesh.params.Field_Orders = field_orders
    mesh.params.Level_Set_Transition_Width = 0.5

    field = mesh.fields["LS"]
    coords = field.function_space.tabulate_dof_coordinates()
    if front_x is None:
        field.x.array[:] = 1.0
    else:
        field.x.array[:] = np.where(coords[:, 0] <= front_x, 1.0, -1.0)
    field.x.scatter_forward()


class _FakePlotterHandle:
    """Simple fake plotter for lifecycle and file-output tests."""

    def __init__(self, fail_open_movie=False, fail_screenshot=False, fail_close=False):
        self.opened_movies = []
        self.screenshots = []
        self.close_calls = 0
        self.fail_open_movie = fail_open_movie
        self.fail_screenshot = fail_screenshot
        self.fail_close = fail_close

    def open_movie(self, path, framerate):
        if self.fail_open_movie:
            raise RuntimeError("cannot open movie")
        self.opened_movies.append((path, framerate))

    def screenshot(self, path):
        if self.fail_screenshot:
            raise RuntimeError("cannot save screenshot")
        self.screenshots.append(path)

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("cannot close")


class _FakeMeasureLine:
    """Minimal chart line object supporting the update protocol."""

    def __init__(self, x, y, color):
        self.x = list(x)
        self.y = list(y)
        self.color = color
        self.updates = []

    def update(self, x, y):
        self.x = list(x)
        self.y = list(y)
        self.updates.append((self.x.copy(), self.y.copy()))


class _FakeChart2D:
    """Small stand-in for ``pyvista.Chart2D``."""

    def __init__(self):
        self.x_label = None
        self.y_label = None
        self.title = None
        self.lines = []

    def line(self, x, y, color="black"):
        line = _FakeMeasureLine(x, y, color)
        self.lines.append(line)
        return line

    def clear(self):
        self.lines.clear()


class _FakeMeasurePlotter:
    """Fake plotter for ``Plot_Measures`` tests."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.added_charts = []
        self.shows = []
        self.render_calls = 0
        self.close_calls = 0
        self.disable_calls = 0
        _FakeMeasurePlotter.instances.append(self)

    def add_chart(self, chart):
        self.added_charts.append(chart)

    def show(self, *args, **kwargs):
        self.shows.append((args, kwargs))

    def render(self):
        self.render_calls += 1

    def close(self):
        self.close_calls += 1

    def disable(self):
        self.disable_calls += 1


def _field_plot_state(plotter, grid=None, actor=None, time_text=None):
    """Build a lightweight field-plot state object for white-box tests."""
    return SimpleNamespace(
        plotter=plotter,
        grid=grid,
        actor=actor,
        time_text=time_text,
    )


def _measure_plot_state(plotter, chart=None, line=None, x=None, y=None):
    """Build a lightweight measure-plot state object for white-box tests."""
    return SimpleNamespace(
        plotter=plotter,
        chart=chart,
        line=line,
        x=list([] if x is None else x),
        y=list([] if y is None else y),
    )


class TestMeshPlotting:
    """Tests for the mesh plotting subsystem."""

    def test_get_boundary_for_plotting_returns_named_line_data(self, mesh_for_plotting):
        """Boundary extraction should return named polyline fragments for 2-D meshes."""
        result = mesh_for_plotting._collect_boundary_plot_data(tdim=2)

        assert set(result) >= {"bottom", "right", "top", "left"}
        for name in ("bottom", "right", "top", "left"):
            data = result[name]
            assert data["points"].size > 0
            assert data["lines"] or data["faces"]

    def test_plot_mesh_records_boundary_labels(self, mesh_for_plotting, tracked_pyvista):
        """plot_mesh should drive pyvista with labeled domain and boundary data."""
        mesh_for_plotting.plot_mesh()

        assert tracked_pyvista.instances, "plot_mesh should instantiate a pyvista plotter"
        plotter = tracked_pyvista.instances[0]
        assert plotter.axes_shown > 0
        assert plotter.legend_count > 0
        assert plotter.viewed_xy > 0
        assert plotter.disable_calls == 1
        assert len(plotter.added_meshes) >= 5

        label_texts = []
        for args, _kwargs in plotter.labels:
            if len(args) >= 2:
                label_texts.extend(args[1])
        for boundary in ("bottom", "right", "top", "left"):
            assert any(boundary in text.lower() for text in label_texts), f"{boundary} label missing"
        assert any("domain" in text.lower() for text in label_texts)

    def test_plot_mesh_batches_open_window_actor_rebuild_into_one_update(
        self,
        mesh_for_plotting,
        tracked_pyvista,
    ):
        """An open MPI-style mesh plot must not render actors one by one."""

        mesh_for_plotting.plot_mesh()
        plotter = tracked_pyvista.instances[0]
        plotter.render_suppression_states.clear()
        updates_before = plotter.updates

        mesh_for_plotting.plot_mesh(annotation="Mesh epoch 001")

        assert plotter.render_suppression_states
        assert all(plotter.render_suppression_states)
        assert plotter.updates == updates_before + 1
        assert plotter.suppress_rendering is False

    def test_plot_mesh_handles_3d_boundaries(self, mesh_3d_for_plotting, tracked_pyvista):
        """3-D mesh plotting should use isometric view and add boundary surfaces."""
        mesh_3d_for_plotting.plot_mesh()

        assert tracked_pyvista.instances
        plotter = tracked_pyvista.instances[-1]
        assert plotter.viewed_iso > 0
        assert len(plotter.added_meshes) >= 7

    def test_plot_mesh_returns_on_non_root(self, mesh_for_plotting, tracked_pyvista, monkeypatch):
        """plot_mesh should return before creating a plotter on non-root ranks."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        monkeypatch.setattr(mesh_plotting, "rank", 1)
        mesh_for_plotting.plot_mesh()
        assert tracked_pyvista.instances == []

    def test_remesh_rebuilds_open_plots_without_replacing_windows(
        self,
        mesh_with_plot_fields,
        tracked_pyvista,
        tmp_path,
    ):
        """Remeshing should replace topology without moving or closing windows."""
        mesh = mesh_with_plot_fields
        mesh.params.Plot_Fields = ["temperature"]
        mesh.params.record_movie = True
        mesh.params.movie_folder = str(tmp_path)
        mesh.params.movie_fps = 6
        measure_state = _measure_plot_state(
            _FakePlotterHandle(),
            x=[0.0, 0.1],
            y=[1.0, 2.0],
        )
        mesh._measure_plots = {("t", "force"): measure_state}

        mesh.Plot_Fields()
        mesh.plot_mesh()
        old_field_state = mesh._field_plots["temperature"]
        old_field_cells = old_field_state.grid.n_cells
        old_mesh_plotter = mesh._mesh_plotter
        old_field_camera = tuple(
            tuple(values) for values in old_field_state.plotter.camera_position
        )
        old_mesh_camera = tuple(
            tuple(values) for values in old_mesh_plotter.camera_position
        )
        plotter_count = len(tracked_pyvista.instances)
        assert old_field_state.plotter.opened_movies == [
            (str(tmp_path / "temperature.mp4"), 6)
        ]

        result = mesh.Refine_From_Indicator(
            RefinementConfig(
                enabled=True,
                indicator=FieldThresholdIndicatorConfig(
                    field="temperature",
                    threshold=1.0,
                ),
                target_h=0.75,
                buffer_layers=0,
            )
        )

        assert result.installed
        assert result.visible_plot_refreshed
        assert old_field_state.plotter.close_calls == 0
        assert old_mesh_plotter.close_calls == 0
        assert mesh._field_plots["temperature"] is old_field_state
        assert mesh._mesh_plotter is old_mesh_plotter
        assert len(tracked_pyvista.instances) == plotter_count
        assert tuple(tuple(values) for values in old_mesh_plotter.camera_position) == old_mesh_camera
        assert mesh._measure_plots[("t", "force")] is measure_state
        assert measure_state.x == [0.0, 0.1]
        assert measure_state.y == [1.0, 2.0]

        mesh.Plot_Fields()

        new_field_state = mesh._field_plots["temperature"]
        assert new_field_state is old_field_state
        assert new_field_state.grid.n_cells > old_field_cells
        assert new_field_state.mesh_epoch == 1
        assert new_field_state.plotter.close_calls == 0
        assert len(tracked_pyvista.instances) == plotter_count
        assert tuple(tuple(values) for values in new_field_state.plotter.camera_position) == old_field_camera
        assert new_field_state.plotter.opened_movies == [
            (str(tmp_path / "temperature.mp4"), 6),
        ]
        assert new_field_state.plotter.written_frames == 2

    def test_remesh_rebuilds_reduced_order_plot_in_the_existing_window(
        self,
        mesh_with_plot_fields,
        tracked_pyvista,
        tmp_path,
    ):
        """Reduced-order actors should also accept new topology in place."""

        mesh = mesh_with_plot_fields
        _add_reduced_order_geometry_fields(mesh)
        mesh.params.Plot_RO_Fields = ["temperature"]
        mesh.params.record_movie = True
        mesh.params.movie_folder = str(tmp_path)
        mesh.params.movie_fps = 8

        mesh.Plot_RO_Fields()
        old_state = mesh._ro_field_plots["temperature"]
        old_cells = old_state.field_grid.n_cells
        old_camera = tuple(
            tuple(values) for values in old_state.plotter.camera_position
        )
        plotter_count = len(tracked_pyvista.instances)

        result = mesh.Refine_From_Indicator(
            RefinementConfig(
                enabled=True,
                indicator=FieldThresholdIndicatorConfig(
                    field="temperature",
                    threshold=1.0,
                ),
                target_h=0.75,
                buffer_layers=0,
            )
        )

        assert result.installed
        assert mesh._ro_field_plots["temperature"] is old_state
        assert old_state.plotter.close_calls == 0

        mesh.Plot_RO_Fields()

        new_state = mesh._ro_field_plots["temperature"]
        assert new_state is old_state
        assert new_state.field_grid.n_cells > old_cells
        assert new_state.mesh_epoch == 1
        assert new_state.plotter.close_calls == 0
        assert len(tracked_pyvista.instances) == plotter_count
        assert tuple(tuple(values) for values in new_state.plotter.camera_position) == old_camera
        assert new_state.plotter.opened_movies == [
            (str(tmp_path / "RO_temperature.mp4"), 8),
        ]
        assert new_state.plotter.written_frames == 2

    def test_get_boundary_skips_orphan_facets(self, mesh_for_plotting):
        """Boundary extraction should skip facets that are not attached to a cell."""

        class _DummyConnectivity:
            def links(self, _):
                return np.array([], dtype=np.int32)

        dummy_conn = _DummyConnectivity()
        dummy_topology = SimpleNamespace(
            create_connectivity=lambda *args, **kwargs: None,
            connectivity=lambda *args, **kwargs: dummy_conn,
        )
        dummy_geometry = SimpleNamespace(dofmap=None, x=np.zeros((0, 2)))
        mesh_for_plotting.mesh = SimpleNamespace(topology=dummy_topology, geometry=dummy_geometry)
        mesh_for_plotting.facet_tags = SimpleNamespace(find=lambda _: [0])
        mesh_for_plotting.boundary_markers = {"orphan": 1}

        result = mesh_for_plotting._collect_boundary_plot_data(tdim=2)
        assert result == {}

    def test_plot_fields_exercises_scalar_and_vector_paths(
        self, mesh_with_plot_fields, tracked_pyvista, monkeypatch
    ):
        """Plot_Fields should handle valid, invalid, and update-failure cases."""
        mesh = mesh_with_plot_fields

        class BrokenField:
            def __init__(self):
                self.name = "Broken Vec"
                self.function_space = SimpleNamespace(
                    element=SimpleNamespace(value_shape=(2,))
                )

            def sub(self, idx):
                raise RuntimeError("cannot collapse")

        broken_field = BrokenField()
        mesh.fields["broken"] = broken_field
        mesh.field_type["broken"] = "CG Vector"
        mesh.field_order["broken"] = 1

        original_get_field_for_output = mesh.Get_Field_For_Output

        def _get_field_for_output(name):
            if name == "broken":
                return FieldOutputView(
                    field_name="broken",
                    field=broken_field,
                    comp_count=2,
                )
            return original_get_field_for_output(name)

        monkeypatch.setattr(mesh, "Get_Field_For_Output", _get_field_for_output)

        mesh.params.Plot_Fields = [
            "v_x",
            "v_z",
            "v_q",
            "v",
            "missing_vec_x",
            "missing_field",
            "ghost",
            "p_x",
            "broken_x",
            "temperature",
            "p",
        ]

        mesh.Plot_Fields()

        for expected in ("v_x", "temperature", "p"):
            assert expected in mesh._field_plots
        for skipped in ("v_z", "v_q", "v", "missing_vec_x", "missing_field", "ghost", "p_x"):
            assert skipped not in mesh._field_plots

        assert len(tracked_pyvista.instances) >= 2
        assert mesh._field_plots["v_x"].plotter.disable_calls == 1
        assert mesh._field_plots["temperature"].plotter.disable_calls == 1

        mesh._field_plots["temperature"].plotter.should_fail_update = True
        mesh.Plot_Fields()
        assert "temperature" not in mesh._field_plots
        assert "v_x" in mesh._field_plots
        assert mesh._field_plots["v_x"].actor.GetMapper() is not None

    def test_plot_fields_triggers_reduced_order_surface_plots(
        self, mesh_with_plot_fields, tracked_pyvista
    ):
        """Plot_Fields should also render Plot_RO_Fields on the z + h surface."""
        mesh = mesh_with_plot_fields
        _add_reduced_order_geometry_fields(mesh, z=-10.0, h=7.5)
        mesh.params.Plot_Fields = ["temperature"]
        mesh.params.Plot_RO_Fields = ["v_x", "h"]
        mesh.params.RO_zscale = 3.0

        mesh.Plot_Fields()

        assert "temperature" in mesh._field_plots
        assert set(mesh._ro_field_plots) == {"v_x", "h"}
        floating_base = -(917.0 / 1030.0) * 7.5
        floating_surface = floating_base + 7.5
        for field_spec, state in mesh._ro_field_plots.items():
            np.testing.assert_allclose(state.basal_grid.points[:, 2], -30.0)
            assert len(state.basal_fill_grid.points) == 2 * len(state.basal_grid.points)
            np.testing.assert_allclose(
                state.basal_fill_grid.points[: len(state.basal_grid.points), 2],
                -30.0,
            )
            np.testing.assert_allclose(
                state.basal_fill_grid.points[len(state.basal_grid.points) :, 2],
                -30.0,
            )
            assert state.basal_side_grid is not None
            assert state.basal_side_actor is not None
            assert len(state.basal_side_grid.points) == 2 * len(state.basal_grid.points)
            np.testing.assert_allclose(
                state.basal_side_grid.points[: len(state.basal_grid.points), 2],
                -30.0,
            )
            assert state.floating_base_grid is not None
            assert state.floating_base_actor is not None
            np.testing.assert_allclose(
                state.floating_base_grid.points[:, 2],
                3.0 * floating_base,
            )
            title_texts = [(args[0], kwargs) for args, kwargs in state.plotter.texts if args]
            assert all("on ice surface" not in text for text, _ in title_texts)
            assert any(text == "z display magnification: 3x" for text, _ in title_texts)
            zscale_kwargs = next(
                kwargs for text, kwargs in title_texts if text == "z display magnification: 3x"
            )
            assert zscale_kwargs["font_size"] == 7
            assert zscale_kwargs["viewport"] is True
            assert zscale_kwargs["position"] == (0.35, 0.92)
            basal_kwargs = state.plotter.added_meshes[0][1]
            side_kwargs = state.plotter.added_meshes[1][1]
            floating_kwargs = state.plotter.added_meshes[2][1]
            assert basal_kwargs["show_edges"] is False
            assert basal_kwargs["opacity"] == 1.0
            assert side_kwargs["show_edges"] is False
            assert side_kwargs["opacity"] == 1.0
            assert side_kwargs["lighting"] is False
            assert floating_kwargs["show_edges"] is False
            assert floating_kwargs["opacity"] == 1.0
            assert floating_kwargs["lighting"] is False
            np.testing.assert_allclose(state.field_grid.points[:, 2], 3.0 * floating_surface)
            assert field_spec in state.field_grid.point_data
            camera = state.plotter.camera
            position = np.asarray(camera.position)
            focal_point = np.asarray(camera.focal_point)
            view_up = np.asarray(camera.GetViewUp())
            assert position[0] > focal_point[0]
            assert position[1] < focal_point[1]
            assert position[2] > focal_point[2]
            horizontal_distance = np.linalg.norm((position - focal_point)[:2])
            elevation = np.rad2deg(np.arctan2(position[2] - focal_point[2], horizontal_distance))
            expected_elevation = np.rad2deg(np.arctan(1.0 / np.sqrt(2.0))) - 10.0
            assert elevation == pytest.approx(expected_elevation)
            np.testing.assert_allclose(view_up, [0.0, 0.0, 1.0])
            self._assert_reduced_order_axis_projection(position, focal_point, view_up)
            assert state.plotter.viewed_xy == 0
            assert state.plotter.viewed_iso == 0

        mesh.fields["h"].x.array[:] = 12.0
        mesh.Plot_Fields()

        for state in mesh._ro_field_plots.values():
            np.testing.assert_allclose(state.basal_grid.points[:, 2], -30.0)
            np.testing.assert_allclose(
                state.basal_fill_grid.points[: len(state.basal_grid.points), 2],
                -30.0,
            )
            np.testing.assert_allclose(
                state.basal_side_grid.points[: len(state.basal_grid.points), 2],
                -30.0,
            )
            assert state.floating_base_grid.n_faces_strict == 0
            np.testing.assert_allclose(state.field_grid.points[:, 2], 6.0)

    def test_plot_fields_adds_level_set_zero_contour_overlay(
        self, mesh_with_plot_fields, tracked_pyvista
    ):
        """2-D field plots should overlay the LS=0 contour when LS is active."""
        mesh = mesh_with_plot_fields
        _add_level_set_plotting_field(mesh, front_x=0.5 * mesh.params.Lx)
        mesh.params.Plot_Fields = ["temperature"]

        mesh.Plot_Fields()

        state = mesh._field_plots["temperature"]
        assert "LS" in state.grid.point_data
        assert state.level_set_contour_actor is not None
        contour_meshes = [
            args[0]
            for args, kwargs, _actor in state.plotter.added_meshes
            if kwargs.get("label") == "LS=0"
        ]
        assert contour_meshes
        assert np.nanmin(contour_meshes[-1].points[:, 2]) > np.nanmax(state.grid.points[:, 2])

    def test_plot_ro_fields_public_noops_and_missing_inputs(
        self, mesh_with_plot_fields, mesh_3d_for_plotting, tracked_pyvista, monkeypatch
    ):
        """Plot_RO_Fields should return cleanly for ordinary no-op configurations."""
        mesh = mesh_with_plot_fields
        original_params = mesh.params
        tracked_pyvista.instances = []

        class _DummyParams:
            pass

        mesh.params = _DummyParams()
        mesh.Plot_RO_Fields()
        assert mesh._ro_field_plots == {}

        mesh.params = original_params
        mesh.params.Plot_RO_Fields = []
        mesh.Plot_RO_Fields()
        assert mesh._ro_field_plots == {}

        mesh.params.Plot_RO_Fields = ["temperature"]
        mesh.Plot_RO_Fields()
        assert mesh._ro_field_plots == {}

        mesh_3d_for_plotting.params.Plot_RO_Fields = ["theta3d"]
        mesh_3d_for_plotting.Plot_RO_Fields()
        assert mesh_3d_for_plotting._ro_field_plots == {}

        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        monkeypatch.setattr(mesh_plotting, "rank", 1)
        mesh.Plot_RO_Fields()
        assert mesh._ro_field_plots == {}
        assert not tracked_pyvista.instances

    def test_plot_ro_fields_skips_missing_and_cell_associated_fields(
        self, mesh_with_plot_fields, tracked_pyvista
    ):
        """Reduced-order plots should skip missing fields and cell-associated data."""
        mesh = mesh_with_plot_fields
        _add_reduced_order_geometry_fields(mesh, z=-10.0, h=7.5)
        mesh.Add_Quadrature_Field("stress_q", "Stress Q")
        mesh.params.Plot_RO_Fields = ["missing_ro", "stress_q"]

        mesh.Plot_RO_Fields()

        assert mesh._ro_field_plots == {}
        assert not tracked_pyvista.instances

    def test_plot_ro_fields_rejects_cell_associated_geometry(
        self, mesh_with_plot_fields, tracked_pyvista, monkeypatch
    ):
        """Reduced-order plotting requires point-associated z and h geometry."""
        mesh = mesh_with_plot_fields
        mesh.params.Plot_RO_Fields = ["temperature"]

        def _prepare_cell_geometry(field_spec, V_dg=None):
            return SimpleNamespace(
                values=np.array([0.0]),
                title=field_spec,
                association="cell" if field_spec in {"z", "h"} else "point",
            )

        monkeypatch.setattr(mesh, "Prepare_Field_For_Plotting", _prepare_cell_geometry)

        mesh.Plot_RO_Fields()

        assert mesh._ro_field_plots == {}
        assert not tracked_pyvista.instances

    def test_plot_ro_fields_records_new_windows_and_refresh_failures(
        self, mesh_with_plot_fields, tracked_pyvista, tmp_path
    ):
        """Direct RO plotting should participate in recording and drop failed windows."""
        mesh = mesh_with_plot_fields
        _add_reduced_order_geometry_fields(mesh, z=-10.0, h=7.5)
        mesh.params.Plot_RO_Fields = ["h"]
        mesh._recording = True
        mesh._recording_folder = str(tmp_path)
        mesh._recording_fps = 9

        mesh.Plot_RO_Fields()

        state = mesh._ro_field_plots["h"]
        assert state.plotter.opened_movies == [(str(tmp_path / "RO_h.mp4"), 9)]
        assert state.plotter.written_frames == 1
        assert mesh._frame_count == 1

        state.plotter.should_fail_write_frame = True
        mesh.Plot_RO_Fields()
        assert state.plotter.written_frames == 2
        assert mesh._frame_count == 2

        state.plotter.should_fail_update = True
        mesh.Plot_RO_Fields()
        assert "h" not in mesh._ro_field_plots

    def test_plot_ro_fields_uses_water_height_and_configured_threshold(
        self, mesh_with_plot_fields, tracked_pyvista
    ):
        """RO geometry should respect water-height and min-thickness settings."""
        mesh = mesh_with_plot_fields
        _add_reduced_order_geometry_fields(mesh, z=-10.0, h=0.005)
        mesh.params.Materials = {
            "Ice": {"Density": 917.0},
            "Water": {"Density": 1030.0, "WaterHeight": 2.5},
        }
        mesh.params.RO_zscale = 2.0
        mesh.params.RO_min_surface_thickness = 0.01
        mesh.params.Plot_RO_Fields = ["h"]

        mesh.Plot_RO_Fields()

        state = mesh._ro_field_plots["h"]
        assert state.field_grid.n_cells == 0
        assert state.ocean_surface_grid is not None
        assert state.ocean_surface_grid.n_faces_strict == state.basal_grid.n_cells
        np.testing.assert_allclose(state.ocean_surface_grid.points[:, 2], 5.0)

    def test_plot_ro_fields_uses_level_set_mask_for_visible_ice(
        self, mesh_with_plot_fields, tracked_pyvista
    ):
        """RO top surfaces should follow LS when LS is defined."""
        mesh = mesh_with_plot_fields
        _add_reduced_order_geometry_fields(mesh, z=-10.0, h=7.5)
        _add_level_set_plotting_field(mesh, front_x=0.5 * mesh.params.Lx)
        mesh.params.Materials = {
            "Ice": {"Density": 917.0},
            "Water": {"Density": 1030.0, "WaterHeight": 0.0},
        }
        mesh.params.Plot_RO_Fields = ["h"]

        mesh.Plot_RO_Fields()

        state = mesh._ro_field_plots["h"]
        assert 0 < state.field_grid.n_cells < state.basal_grid.n_cells
        assert state.ocean_surface_grid is not None
        assert state.ocean_surface_grid.n_faces_strict > 0

    def test_plot_ro_fields_smooths_dg_height_for_surface_geometry(
        self, mesh_with_plot_fields, tracked_pyvista
    ):
        """DG h fields should use the RO CG1 round-trip for surface geometry."""
        mesh = mesh_with_plot_fields
        mesh.Add_Field("z", "Basal geometry", "CG", 1)
        mesh.Add_Field("h", "Thickness", "DG", 1)
        mesh.fields["z"].x.array[:] = -10.0
        mesh.fields["h"].x.array[:] = np.linspace(2.0, 4.0, mesh.fields["h"].x.array.size)
        mesh.params.Plot_RO_Fields = ["h"]

        mesh.Plot_RO_Fields()

        state = mesh._ro_field_plots["h"]
        assert state.field_grid.n_points > 0
        assert 1 in mesh._cg_spaces
        assert any(
            isinstance(cache_key, tuple) and cache_key[0] == "_ro_h_geom"
            for cache_key in mesh._output_projection_cache
        )

    def test_reduced_order_field_grid_omits_thin_top_surface_cells(
        self, mesh_for_plotting
    ):
        """The coloured RO top surface should omit cells with very thin ice."""
        mesh = mesh_for_plotting
        topology = np.array([3, 0, 1, 2, 3, 1, 3, 2], dtype=np.int64)
        cell_types = np.array([5, 5], dtype=np.uint8)
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        )
        h_values = np.array([0.02, 0.02, 0.02, 0.005])
        field_values = np.arange(4.0)

        field_grid = mesh._build_reduced_order_field_grid(
            topology,
            cell_types,
            points,
            h_values,
            "h",
            field_values,
        )

        assert field_grid.n_cells == 1
        np.testing.assert_array_equal(field_grid.cells, [3, 0, 1, 2])
        np.testing.assert_allclose(field_grid.point_data["h"], field_values)

    def test_reduced_order_field_grid_omits_level_set_inactive_cells(
        self, mesh_for_plotting
    ):
        """The coloured RO top surface should omit cells outside LS active ice."""
        mesh = mesh_for_plotting
        mesh.params.Level_Set_Transition_Width = 0.5
        topology = np.array([3, 0, 1, 2, 3, 1, 3, 2], dtype=np.int64)
        cell_types = np.array([5, 5], dtype=np.uint8)
        points = np.zeros((4, 3))
        h_values = np.full(4, 0.02)
        level_set_values = np.array([1.0, 1.0, 1.0, -1.0])
        field_values = np.arange(4.0)

        field_grid = mesh._build_reduced_order_field_grid(
            topology,
            cell_types,
            points,
            h_values,
            "h",
            field_values,
            level_set_values=level_set_values,
        )

        assert field_grid.n_cells == 1
        np.testing.assert_array_equal(field_grid.cells, [3, 0, 1, 2])

    def test_reduced_order_floating_base_omits_mixed_removed_cells(
        self, mesh_for_plotting
    ):
        """Floating undersides should not connect removed/thin points to the bed."""
        topology = np.array([3, 0, 1, 2, 3, 1, 3, 2], dtype=np.int64)
        points = np.array(
            [
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, -1.0],
                [1.0, 1.0, -10.0],
            ]
        )
        floating_mask = np.array([True, True, True, False])

        floating_grid = mesh_for_plotting._build_reduced_order_floating_base_grid(
            topology,
            points,
            floating_mask,
        )

        assert floating_grid.n_faces_strict == 1
        np.testing.assert_array_equal(floating_grid.faces, [3, 0, 1, 2])

    def test_reduced_order_ocean_surface_shows_open_water_cells(
        self, mesh_for_plotting
    ):
        """Open-water cells should render at water height when ice is removed."""
        topology = np.array([3, 0, 1, 2, 3, 1, 3, 2], dtype=np.int64)
        points = np.array(
            [
                [0.0, 0.0, 2.5],
                [1.0, 0.0, 2.5],
                [0.0, 1.0, 2.5],
                [1.0, 1.0, 2.5],
            ]
        )
        open_water_mask = np.array([True, True, True, False])

        ocean_grid = mesh_for_plotting._build_reduced_order_ocean_surface_grid(
            topology,
            points,
            open_water_mask,
        )

        assert ocean_grid.n_faces_strict == 1
        np.testing.assert_array_equal(ocean_grid.faces, [3, 0, 1, 2])
        np.testing.assert_allclose(ocean_grid.points[:, 2], 2.5)

    def test_reduced_order_plot_refresh_removes_and_restores_thin_top_surface(
        self, mesh_with_plot_fields, tracked_pyvista
    ):
        """Plot refresh should update visible RO top-surface cells as h changes."""
        mesh = mesh_with_plot_fields
        mesh.Add_Field("z", "Basal geometry", "CG", 1)
        mesh.Add_Field("h", "Thickness", "CG", 1)
        mesh.fields["z"].x.array[:] = -10.0
        mesh.fields["h"].x.array[:] = 0.005
        mesh.params.Plot_Fields = ["temperature"]
        mesh.params.Plot_RO_Fields = ["h"]

        mesh.Plot_Fields()

        state = mesh._ro_field_plots["h"]
        assert state.basal_grid.n_cells > 0
        assert state.field_grid.n_cells == 0
        assert state.ocean_surface_grid is not None
        assert state.ocean_surface_actor is not None
        assert state.ocean_surface_grid.n_faces_strict == state.basal_grid.n_cells
        np.testing.assert_allclose(state.ocean_surface_grid.points[:, 2], 0.0)

        mesh.fields["h"].x.array[:] = 0.02
        mesh.Plot_Fields()

        assert state.field_grid.n_cells == state.basal_grid.n_cells
        assert state.ocean_surface_grid.n_faces_strict == 0

    def test_reduced_order_plot_refresh_creates_ocean_surface_actor(
        self, mesh_with_plot_fields, tracked_pyvista
    ):
        """Open water can appear after an RO plot was first created."""
        mesh = mesh_with_plot_fields
        _add_reduced_order_geometry_fields(mesh, z=-10.0, h=0.02)
        mesh.params.Plot_RO_Fields = ["h"]

        mesh.Plot_RO_Fields()

        state = mesh._ro_field_plots["h"]
        assert state.ocean_surface_grid is None
        assert state.ocean_surface_actor is None

        mesh.fields["h"].x.array[:] = 0.005
        mesh.Plot_RO_Fields()

        assert state.ocean_surface_grid is not None
        assert state.ocean_surface_actor is not None
        assert state.ocean_surface_grid.n_faces_strict == state.basal_grid.n_cells

    def test_reduced_order_basal_fill_grid_closes_to_minimum_bed(
        self, mesh_for_plotting
    ):
        """Basal fill should split top/bottom faces from lighter side walls."""
        import pyvista as pv

        topology = np.array([3, 0, 1, 2], dtype=np.int64)
        cell_types = np.array([5], dtype=np.uint8)
        points = np.array(
            [
                [0.0, 0.0, -3.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, -2.0],
            ]
        )
        basal_grid = pv.UnstructuredGrid(topology, cell_types, points)

        fill_grid = mesh_for_plotting._build_reduced_order_basal_fill_grid(basal_grid)
        side_grid = mesh_for_plotting._build_reduced_order_basal_side_grid(basal_grid)

        assert len(fill_grid.points) == 6
        np.testing.assert_allclose(fill_grid.points[:3], points)
        np.testing.assert_allclose(fill_grid.points[3:, 2], -3.0)
        assert fill_grid.n_faces_strict == 2
        assert len(side_grid.points) == 6
        np.testing.assert_allclose(side_grid.points[:3], points)
        np.testing.assert_allclose(side_grid.points[3:, 2], -3.0)
        assert side_grid.n_faces_strict == 3

    @staticmethod
    def _assert_reduced_order_axis_projection(position, focal_point, view_up):
        """Check RO camera projects x right/down, y right/up, and z up."""
        view_dir = focal_point - position
        view_dir /= np.linalg.norm(view_dir)
        screen_up = view_up - np.dot(view_up, view_dir) * view_dir
        screen_up /= np.linalg.norm(screen_up)
        screen_right = np.cross(view_dir, screen_up)
        screen_right /= np.linalg.norm(screen_right)

        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        assert np.dot(x_axis, screen_right) > 0.0
        assert np.dot(x_axis, screen_up) < 0.0
        assert np.dot(y_axis, screen_right) > 0.0
        assert np.dot(y_axis, screen_up) > 0.0
        assert np.dot(z_axis, screen_up) > 0.0

    def test_plot_fields_can_remain_interactive_when_requested(self, mesh_with_plot_fields, tracked_pyvista):
        """Plot_Fields should keep plotters interactive when explicitly enabled."""
        mesh = mesh_with_plot_fields
        mesh.params.Plot_Fields = ["temperature"]
        mesh.params.Interactive_Plots = True

        mesh.Plot_Fields()

        assert mesh._field_plots["temperature"].plotter.disable_calls == 0

    def test_plot_fields_returns_when_not_configured(self, mesh_with_plot_fields, tracked_pyvista):
        """Plot_Fields should return cleanly when Plot_Fields is absent."""
        mesh = mesh_with_plot_fields
        mesh._field_plots.clear()
        tracked_pyvista.instances = []

        class _DummyParams:
            pass

        mesh.params = _DummyParams()
        mesh.Plot_Fields()
        assert mesh._field_plots == {}
        assert not tracked_pyvista.instances

    def test_plot_fields_returns_on_empty_plot_list(self, mesh_with_plot_fields, tracked_pyvista):
        """Plot_Fields should not build plot windows when the plot list is empty."""
        mesh = mesh_with_plot_fields
        mesh._field_plots.clear()
        tracked_pyvista.instances = []
        mesh.params.Plot_Fields = []

        mesh.Plot_Fields()

        assert mesh._field_plots == {}
        assert not tracked_pyvista.instances

    def test_plot_fields_handles_3d_mesh(self, mesh_3d_for_plotting, tracked_pyvista):
        """Plot_Fields should visualize both scalar and vector data in 3-D."""
        mesh_3d_for_plotting.params.Plot_Fields = ["theta3d", "v3d_z"]
        mesh_3d_for_plotting.Plot_Fields()

        assert "theta3d" in mesh_3d_for_plotting._field_plots
        assert "v3d_z" in mesh_3d_for_plotting._field_plots
        assert tracked_pyvista.instances
        assert mesh_3d_for_plotting._field_plots["theta3d"].plotter.viewed_iso > 0

    def test_plot_isosurfaces_adds_3d_window(self, mesh_3d_for_plotting, tracked_pyvista):
        """Plot_Isosurfaces should add a shell-plus-isosurface window and refresh it."""
        mesh = mesh_3d_for_plotting
        x = mesh.fields["theta3d"].function_space.tabulate_dof_coordinates()[:, 0]
        mesh.fields["theta3d"].x.array[:] = x
        mesh.params.Plot_Fields = []
        mesh.params.Plot_Isosurfaces = {"theta3d": 1.0}
        mesh.Plot_Fields()

        state = mesh._field_plots["theta3d_isosurface"]
        assert "theta3d" not in mesh._field_plots
        assert state.isosurface_levels == (1.0,)
        assert state.isosurface_actor is not None
        # The shell is the outer surface only: a closed triangulation (F = 2V - 4) on the box faces
        shell = mesh._isosurface_shell(state.grid)
        assert shell.n_cells == 2 * shell.n_points - 4
        points = np.asarray(shell.points)
        on_face = np.isclose(points, 0.0, atol=1e-9) | np.isclose(points, [2.0, 2.0, 1.0], atol=1e-9)
        assert np.all(on_face.any(axis=1))

        # Updates keep one colour bar
        mesh.fields["theta3d"].x.array[:] = 1.5 * x
        mesh.Plot_Fields()
        assert len(mesh._field_plots["theta3d_isosurface"].plotter.scalar_bars) == 1

        # A level outside the field range draws only the shell
        mesh.fields["theta3d"].x.array[:] = 0.5
        mesh.Plot_Fields()
        assert mesh._field_plots["theta3d_isosurface"].isosurface_actor is None

    def test_isosurface_shell_is_oriented_outward_for_mixed_cell_orderings(self):
        """Refined meshes mix tetrahedron orderings; the shell must still face outward everywhere."""
        import pyvista as pv

        from femcubes.core.mesh._mesh_plotting import _MeshPlotting

        cube = pv.ImageData(dimensions=(4, 4, 4)).triangulate().cast_to_unstructured_grid()
        cells = cube.cells_dict[pv.CellType.TETRA].copy()
        cells[::2, [0, 1]] = cells[::2, [1, 0]]  # Reverse every second tetrahedron
        grid = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, cube.points)

        shell = _MeshPlotting._isosurface_shell(grid)

        outward = np.asarray(shell.cell_centers().points) - 1.5
        normals = np.asarray(shell.cell_data["Normals"])
        assert shell.n_cells == 6 * 9 * 2
        assert np.all(np.einsum("ij,ij->i", normals, outward) > 0.0)
        edges = shell.extract_feature_edges(30.0, boundary_edges=False, non_manifold_edges=False,
                                            manifold_edges=False)
        assert edges.n_cells == 12 * 3  # The cube edges only

    def test_plot_isosurfaces_ignored_in_2d(self, mesh_with_plot_fields, tracked_pyvista):
        """Plot_Isosurfaces entries should be ignored on 2-D meshes."""
        mesh = mesh_with_plot_fields
        mesh.params.Plot_Fields = []
        mesh.params.Plot_Isosurfaces = {"temperature": 0.5}
        mesh.Plot_Fields()
        assert mesh._field_plots == {}

    def test_plot_fields_auto_starts_recording_when_requested(
        self, mesh_with_plot_fields, tracked_pyvista, monkeypatch
    ):
        """Plot_Fields should auto-start recording from params when requested."""
        mesh = mesh_with_plot_fields
        mesh.params.Plot_Fields = ["temperature"]
        mesh.params.record_movie = True
        mesh.params.output_dir = "ResultsFolder"
        mesh.params.movie_fps = 12
        calls = []

        def _fake_start(folder, fps):
            calls.append((folder, fps))

        monkeypatch.setattr(mesh, "StartRecording", _fake_start)

        mesh.Plot_Fields()

        assert calls == [("ResultsFolder/movies", 12)]

    def test_plot_fields_applies_log_plot_transform(self, mesh_with_plot_fields, tracked_pyvista):
        """Plot_Fields should apply log10 scaling when requested per field."""
        mesh = mesh_with_plot_fields
        mesh.params.Plot_Fields = ["temperature"]
        mesh.params.LogPlot = [True]

        mesh.Plot_Fields()

        plotter = mesh._field_plots["temperature"].plotter
        title_texts = [args[0] for args, _kwargs in plotter.texts if args]
        assert any("(log10)" in text for text in title_texts)
        np.testing.assert_allclose(
            mesh._field_plots["temperature"].grid.point_data["temperature"],
            np.log10(2.0),
        )

    def test_plot_fields_returns_on_non_root(self, mesh_for_plotting, monkeypatch):
        """Plot_Fields should return before opening windows on non-root ranks."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        mesh_for_plotting.Add_Field("p", "Pressure", "CG", 1)
        mesh_for_plotting.fields["p"].x.array[:] = 1.0
        mesh_for_plotting.params.Plot_Fields = ["p"]

        monkeypatch.setattr(mesh_plotting, "rank", 1)
        mesh_for_plotting.Plot_Fields()
        assert mesh_for_plotting._field_plots == {}

    def test_plot_fields_merge_and_render_paths(self, mesh_for_plotting, monkeypatch):
        """Plot_Fields should update merged grids and refresh existing actors."""
        mesh = mesh_for_plotting
        mesh.Add_Field("phase", "Phase", "CG", 1)
        mesh.fields["phase"].x.array[:] = 0.25
        mesh.params.Plot_Fields = ["phase"]

        class _FakeMapper:
            def __init__(self):
                self.ranges = []

            def SetScalarRange(self, data_range):
                self.ranges.append(tuple(data_range))

        class _FakeActor:
            def __init__(self):
                self.mapper = _FakeMapper()

            def GetMapper(self):
                return self.mapper

        class _FakePlotter:
            instances = []

            def __init__(self, *args, **kwargs):
                self.render_calls = 0
                _FakePlotter.instances.append(self)

            def add_text(self, *args, **kwargs):
                return None

            def add_mesh(self, *args, **kwargs):
                return _FakeActor()

            def show_axes(self):
                return None

            def view_xy(self):
                return None

            def enable_2d_style(self):
                return None

            def show(self, *args, **kwargs):
                return None

            def render(self):
                self.render_calls += 1
                return None

        class _FakeGrid:
            create_count = 0

            def __init__(self, topology, cell_types, geometry):
                _FakeGrid.create_count += 1
                self.point_data = {}
                self.cell_data = {}

            def merge(self, grids):
                merged = _FakeGrid(None, None, None)
                merged.point_data.update(self.point_data)
                return merged

            def Modified(self):
                return None

        class _FakeComm:
            def __init__(self):
                self.gather_calls = 0

            def gather(self, data, root=0):
                self.gather_calls += 1
                return [data, data]

            def bcast(self, data, root=0):
                return data

        class _FakePV:
            OFF_SCREEN = True
            Plotter = _FakePlotter
            UnstructuredGrid = _FakeGrid

        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        fake_comm = _FakeComm()
        monkeypatch.setattr(mesh_plotting, "pv", _FakePV)
        monkeypatch.setattr(mesh_plotting, "comm", fake_comm)
        monkeypatch.setattr(mesh_plotting, "rank", 0)

        mesh.Plot_Fields()
        plotter = mesh._field_plots["phase"].plotter
        assert plotter.render_calls == 0
        grid_count_after_first_plot = _FakeGrid.create_count
        gather_count_after_first_plot = fake_comm.gather_calls

        plot_grid = mesh._field_plots["phase"].grid
        plot_grid.point_data.pop("phase", None)

        mesh.fields["phase"].x.array[:] = 0.5
        mesh.Plot_Fields()
        assert _FakeGrid.create_count == grid_count_after_first_plot
        assert fake_comm.gather_calls == gather_count_after_first_plot + 1
        assert plotter.render_calls > 0
        assert "phase" in plot_grid.point_data

    def test_plot_fields_opens_movie_when_recording(self, mesh_with_plot_fields, tracked_pyvista, tmp_path):
        """Plot_Fields should open a movie file for new field plotters while recording."""
        mesh = mesh_with_plot_fields
        mesh.params.Plot_Fields = ["temperature"]
        mesh._recording = True
        mesh._recording_folder = str(tmp_path)
        mesh._recording_fps = 15

        mesh.Plot_Fields()

        plotter = mesh._field_plots["temperature"].plotter
        assert plotter.opened_movies == [(str(tmp_path / "temperature.mp4"), 15)]
        assert plotter.written_frames == 1
        assert mesh._frame_count == 1

    def test_plot_fields_tolerates_movie_and_frame_write_failures(
        self, mesh_with_plot_fields, tracked_pyvista, tmp_path
    ):
        """Recording failures in Plot_Fields should be logged and ignored."""
        mesh = mesh_with_plot_fields
        mesh.params.Plot_Fields = ["temperature"]
        mesh._recording = True
        mesh._recording_folder = str(tmp_path)
        mesh._recording_fps = 8
        tracked_pyvista.fail_open_movie_for_next = True

        mesh.Plot_Fields()

        plotter = mesh._field_plots["temperature"].plotter
        assert plotter.opened_movies == []
        plotter.should_fail_write_frame = True

        mesh.Plot_Fields()

        assert mesh._frame_count == 2
        assert plotter.written_frames == 2

    def test_start_recording_opens_movies_for_existing_plotters(self, mesh_for_plotting, tmp_path):
        """StartRecording should create movie files for already-open field plotters."""
        plotter_a = _FakePlotterHandle()
        plotter_b = _FakePlotterHandle()
        mesh_for_plotting._field_plots = {
            "phase": _field_plot_state(plotter_a),
            r"u\x": _field_plot_state(plotter_b),
        }

        folder = tmp_path / "movies"
        mesh_for_plotting.StartRecording(str(folder), fps=24)

        assert mesh_for_plotting._recording is True
        assert mesh_for_plotting._recording_folder == str(folder)
        assert mesh_for_plotting._recording_fps == 24
        assert folder.exists()
        assert plotter_a.opened_movies == [(str(folder / "phase.mp4"), 24)]
        assert plotter_b.opened_movies == [(str(folder / "u_x.mp4"), 24)]

    def test_start_recording_returns_on_non_root(self, mesh_for_plotting, tmp_path, monkeypatch):
        """StartRecording should be a no-op on non-root ranks."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        folder = tmp_path / "movies"
        monkeypatch.setattr(mesh_plotting, "rank", 1)

        mesh_for_plotting.StartRecording(str(folder), fps=5)

        assert mesh_for_plotting._recording is False
        assert not folder.exists()

    def test_start_recording_tolerates_movie_open_failures(self, mesh_for_plotting, tmp_path):
        """StartRecording should ignore movie-open failures from existing plotters."""
        mesh_for_plotting._field_plots = {
            "phase": _field_plot_state(_FakePlotterHandle(fail_open_movie=True))
        }

        mesh_for_plotting.StartRecording(str(tmp_path), fps=11)

        assert mesh_for_plotting._recording is True

    def test_stop_recording_clears_recording_flag(self, mesh_for_plotting):
        """StopRecording should disable frame capture without touching plot state."""
        mesh_for_plotting._recording = True
        mesh_for_plotting._frame_count = 3

        mesh_for_plotting.StopRecording()

        assert mesh_for_plotting._recording is False
        assert mesh_for_plotting._frame_count == 3

    def test_stop_recording_returns_on_non_root(self, mesh_for_plotting, monkeypatch):
        """StopRecording should leave state untouched on non-root ranks."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        mesh_for_plotting._recording = True
        monkeypatch.setattr(mesh_plotting, "rank", 1)

        mesh_for_plotting.StopRecording()

        assert mesh_for_plotting._recording is True

    def test_save_plots_to_folder_saves_field_and_measure_plots(self, mesh_for_plotting, tmp_path):
        """SavePlotsToFolder should screenshot all open field and measure plotters."""
        field_plotter = _FakePlotterHandle()
        measure_plotter = _FakePlotterHandle()
        mesh_for_plotting._field_plots = {r"phase\field": _field_plot_state(field_plotter)}
        mesh_for_plotting._measure_plots = {
            ("time/axis", r"force\value"): _measure_plot_state(measure_plotter)
        }

        mesh_for_plotting.SavePlotsToFolder(str(tmp_path))

        assert field_plotter.screenshots == [str(tmp_path / "phase_field.png")]
        assert measure_plotter.screenshots == [
            str(tmp_path / "force_value_vs_time_axis.png")
        ]

    def test_save_plots_to_folder_saves_open_mesh_plot(self, mesh_for_plotting, tmp_path):
        """SavePlotsToFolder should also screenshot the plot_mesh() window."""
        mesh_plotter = _FakePlotterHandle()
        mesh_for_plotting._mesh_plotter = mesh_plotter

        mesh_for_plotting.SavePlotsToFolder(str(tmp_path))

        assert mesh_plotter.screenshots == [str(tmp_path / "mesh.png")]

    def test_reduced_order_plots_join_recording_save_and_close_lifecycle(
        self, mesh_for_plotting, tmp_path
    ):
        """RO plots should participate in the ordinary plot lifecycle methods."""
        ro_plotter = _FakePlotterHandle()
        mesh_for_plotting._ro_field_plots = {
            r"v\x": SimpleNamespace(plotter=ro_plotter)
        }

        movie_folder = tmp_path / "movies"
        mesh_for_plotting.StartRecording(str(movie_folder), fps=18)
        mesh_for_plotting.SavePlotsToFolder(str(tmp_path))
        mesh_for_plotting.Close_Plots()

        assert ro_plotter.opened_movies == [(str(movie_folder / "RO_v_x.mp4"), 18)]
        assert ro_plotter.screenshots == [str(tmp_path / "RO_v_x.png")]
        assert ro_plotter.close_calls == 1
        assert mesh_for_plotting._ro_field_plots == {}

    def test_save_plots_to_folder_returns_on_non_root(self, mesh_for_plotting, tmp_path, monkeypatch):
        """SavePlotsToFolder should be a no-op on non-root ranks."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        folder = tmp_path / "plots"
        monkeypatch.setattr(mesh_plotting, "rank", 1)

        mesh_for_plotting.SavePlotsToFolder(str(folder))

        assert not folder.exists()

    def test_save_plots_to_folder_tolerates_screenshot_failures(self, mesh_for_plotting, tmp_path):
        """SavePlotsToFolder should ignore screenshot failures from any plotter."""
        field_plotter = _FakePlotterHandle(fail_screenshot=True)
        measure_plotter = _FakePlotterHandle(fail_screenshot=True)
        mesh_for_plotting._field_plots = {"phase": _field_plot_state(field_plotter)}
        mesh_for_plotting._measure_plots = {
            ("t", "force"): _measure_plot_state(measure_plotter)
        }

        mesh_for_plotting.SavePlotsToFolder(str(tmp_path))

        assert tmp_path.exists()

    def test_close_plots_resets_all_plotting_state(self, mesh_for_plotting, monkeypatch):
        """Close_Plots should close every plotter and clear cached plotting state."""
        field_plotter = _FakePlotterHandle()
        mesh_plotter = _FakePlotterHandle()
        measure_plotter = _FakePlotterHandle()
        mesh_for_plotting._field_plots = {
            "phase": _field_plot_state(
                field_plotter,
                grid=object(),
                actor=object(),
                time_text=object(),
            )
        }
        mesh_for_plotting._mesh_plotter = mesh_plotter
        mesh_for_plotting._measure_plots = {
            ("t", "force"): _measure_plot_state(measure_plotter)
        }
        mesh_for_plotting._recording = True

        state = {"stopped": False}

        def _fake_stop_recording():
            state["stopped"] = True
            mesh_for_plotting._recording = False

        monkeypatch.setattr(mesh_for_plotting, "StopRecording", _fake_stop_recording)

        mesh_for_plotting.Close_Plots()

        assert state["stopped"] is True
        assert mesh_for_plotting._field_plots == {}
        assert mesh_for_plotting._measure_plots == {}
        assert mesh_for_plotting._mesh_plotter is None
        assert field_plotter.close_calls == 1
        assert mesh_plotter.close_calls == 1
        assert measure_plotter.close_calls == 1

    def test_close_plots_tolerates_close_failures(self, mesh_for_plotting):
        """Close_Plots should still clear state if individual plotters fail to close."""
        mesh_for_plotting._field_plots = {
            "phase": _field_plot_state(_FakePlotterHandle(fail_close=True))
        }
        mesh_for_plotting._mesh_plotter = _FakePlotterHandle(fail_close=True)
        mesh_for_plotting._measure_plots = {
            ("t", "force"): _measure_plot_state(_FakePlotterHandle(fail_close=True))
        }

        mesh_for_plotting.Close_Plots()

        assert mesh_for_plotting._field_plots == {}
        assert mesh_for_plotting._measure_plots == {}
        assert mesh_for_plotting._mesh_plotter is None

    def test_plot_measures_creates_and_updates_chart(self, mesh_for_plotting, monkeypatch):
        """Plot_Measures should create one chart and append new data on later calls."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        class _FakePV:
            OFF_SCREEN = True
            Plotter = _FakeMeasurePlotter
            Chart2D = _FakeChart2D

        _FakeMeasurePlotter.instances = []
        monkeypatch.setattr(mesh_plotting, "pv", _FakePV)
        mesh_for_plotting.params.Plot_Measures = [("t", "force")]
        mesh_for_plotting.params.Global_Measures = {"force": 1.5}
        mesh_for_plotting.params.t = 0.1

        mesh_for_plotting.Plot_Measures()

        key = ("t", "force")
        assert key in mesh_for_plotting._measure_plots
        plot_state = mesh_for_plotting._measure_plots[key]
        assert plot_state.x == [0.1]
        assert plot_state.y == [1.5]
        assert plot_state.plotter.render_calls == 1
        assert plot_state.plotter.disable_calls == 1

        mesh_for_plotting.params.t = 0.2
        mesh_for_plotting.params.Global_Measures["force"] = 2.0
        mesh_for_plotting.Plot_Measures()

        assert plot_state.x == [0.1, 0.2]
        assert plot_state.y == [1.5, 2.0]
        assert plot_state.line.updates == [([0.1, 0.2], [1.5, 2.0])]
        assert plot_state.plotter.render_calls == 2

    def test_plot_measures_returns_when_not_configured(self, mesh_for_plotting):
        """Plot_Measures should return when the params object has no plot list."""
        class _DummyParams:
            pass

        mesh_for_plotting.params = _DummyParams()

        mesh_for_plotting.Plot_Measures()

        assert mesh_for_plotting._measure_plots == {}

    def test_plot_measures_returns_on_empty_or_non_root(self, mesh_for_plotting, monkeypatch):
        """Plot_Measures should return on empty config and non-root ranks."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        mesh_for_plotting.params.Plot_Measures = []
        mesh_for_plotting.Plot_Measures()
        assert mesh_for_plotting._measure_plots == {}

        mesh_for_plotting.params.Plot_Measures = [("t", "force")]
        mesh_for_plotting.params.Global_Measures = {"force": 1.0}
        mesh_for_plotting.params.t = 0.1
        monkeypatch.setattr(mesh_plotting, "rank", 1)
        mesh_for_plotting.Plot_Measures()
        assert mesh_for_plotting._measure_plots == {}

    def test_plot_measures_uses_global_measure_x_and_skips_missing_data(
        self, mesh_for_plotting, monkeypatch
    ):
        """Plot_Measures should read x-values from Global_Measures and skip missing pairs."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        class _FakePV:
            OFF_SCREEN = True
            Plotter = _FakeMeasurePlotter
            Chart2D = _FakeChart2D

        _FakeMeasurePlotter.instances = []
        monkeypatch.setattr(mesh_plotting, "pv", _FakePV)
        mesh_for_plotting.params.Plot_Measures = [("disp", "force"), ("missing", "force")]
        mesh_for_plotting.params.Global_Measures = {"disp": 0.25, "force": 2.5}

        mesh_for_plotting.Plot_Measures()

        assert ("disp", "force") in mesh_for_plotting._measure_plots
        assert ("missing", "force") not in mesh_for_plotting._measure_plots

    def test_plot_measures_skips_negative_time(self, mesh_for_plotting, monkeypatch):
        """Plot_Measures should skip negative time values."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        class _FakePV:
            OFF_SCREEN = True
            Plotter = _FakeMeasurePlotter
            Chart2D = _FakeChart2D

        _FakeMeasurePlotter.instances = []
        monkeypatch.setattr(mesh_plotting, "pv", _FakePV)
        mesh_for_plotting.params.Plot_Measures = [("t", "force")]
        mesh_for_plotting.params.Global_Measures = {"force": 2.0}
        mesh_for_plotting.params.t = -1.0

        mesh_for_plotting.Plot_Measures()

        assert mesh_for_plotting._measure_plots == {}

    def test_plot_measures_updates_lines_with_set_data(self, mesh_for_plotting):
        """Plot_Measures should use ``set_data`` when that is the available update API."""
        class _SetDataLine:
            def __init__(self):
                self.calls = []

            def set_data(self, x, y):
                self.calls.append((list(x), list(y)))

        plotter = _FakeMeasurePlotter()
        line = _SetDataLine()
        chart = _FakeChart2D()
        mesh_for_plotting._measure_plots = {
            ("t", "force"): _measure_plot_state(plotter, chart, line, [0.1], [1.0])
        }
        mesh_for_plotting.params.Plot_Measures = [("t", "force")]
        mesh_for_plotting.params.Global_Measures = {"force": 2.0}
        mesh_for_plotting.params.t = 0.2

        mesh_for_plotting.Plot_Measures()

        assert line.calls == [([0.1, 0.2], [1.0, 2.0])]

    def test_plot_measures_recreates_line_when_xy_assignment_fails(self, mesh_for_plotting):
        """Plot_Measures should recreate a line if x/y assignment raises AttributeError."""
        class _AttrErrorLine:
            def __init__(self):
                self._x = [0.1]
                self._y = [1.0]

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                raise AttributeError("x is read-only")

            @property
            def y(self):
                return self._y

            @y.setter
            def y(self, value):
                raise AttributeError("y is read-only")

        plotter = _FakeMeasurePlotter()
        chart = _FakeChart2D()
        original_line = _AttrErrorLine()
        mesh_for_plotting._measure_plots = {
            ("t", "force"): _measure_plot_state(
                plotter,
                chart,
                original_line,
                [0.1],
                [1.0],
            )
        }
        mesh_for_plotting.params.Plot_Measures = [("t", "force")]
        mesh_for_plotting.params.Global_Measures = {"force": 2.0}
        mesh_for_plotting.params.t = 0.2

        mesh_for_plotting.Plot_Measures()

        assert mesh_for_plotting._measure_plots[("t", "force")].line is chart.lines[-1]

    def test_plot_measures_updates_lines_via_xy_assignment(self, mesh_for_plotting):
        """Plot_Measures should fall back to direct x/y assignment when available."""
        class _XYAssignableLine:
            def __init__(self):
                self._x = [0.1]
                self._y = [1.0]
                self.assignments = []

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = list(value)
                self.assignments.append(("x", self._x.copy()))

            @property
            def y(self):
                return self._y

            @y.setter
            def y(self, value):
                self._y = list(value)
                self.assignments.append(("y", self._y.copy()))

        plotter = _FakeMeasurePlotter()
        chart = _FakeChart2D()
        line = _XYAssignableLine()
        mesh_for_plotting._measure_plots = {
            ("t", "force"): _measure_plot_state(plotter, chart, line, [0.1], [1.0])
        }
        mesh_for_plotting.params.Plot_Measures = [("t", "force")]
        mesh_for_plotting.params.Global_Measures = {"force": 2.0}
        mesh_for_plotting.params.t = 0.2

        mesh_for_plotting.Plot_Measures()

        assert line.assignments == [
            ("x", [0.1, 0.2]),
            ("y", [1.0, 2.0]),
        ]

    def test_plot_measures_rebuilds_line_when_no_update_api_exists(self, mesh_for_plotting):
        """Plot_Measures should clear and rebuild a chart line as a last resort."""
        class _OpaqueLine:
            pass

        class _TrackingChart(_FakeChart2D):
            def __init__(self):
                super().__init__()
                self.clear_calls = 0

            def clear(self):
                self.clear_calls += 1
                super().clear()

        plotter = _FakeMeasurePlotter()
        chart = _TrackingChart()
        mesh_for_plotting._measure_plots = {
            ("t", "force"): _measure_plot_state(
                plotter,
                chart,
                _OpaqueLine(),
                [0.1],
                [1.0],
            )
        }
        mesh_for_plotting.params.Plot_Measures = [("t", "force")]
        mesh_for_plotting.params.Global_Measures = {"force": 2.0}
        mesh_for_plotting.params.t = 0.2

        mesh_for_plotting.Plot_Measures()

        assert chart.clear_calls == 1
        assert mesh_for_plotting._measure_plots[("t", "force")].line is chart.lines[-1]

    def test_process_fields_for_plotting_auto_selects_dg_spaces(self, mesh_with_plot_fields):
        """Process_Fields_For_Plotting should derive DG spaces for scalar and vector fields."""
        scalar_values, scalar_title, scalar_space = mesh_with_plot_fields.Process_Fields_For_Plotting(
            "temperature"
        )
        vector_values, vector_title, vector_space = mesh_with_plot_fields.Process_Fields_For_Plotting(
            "v_x"
        )

        assert scalar_title == "Temperature"
        assert "X component" in vector_title
        assert scalar_space is vector_space
        assert mesh_with_plot_fields._dg_spaces[1] is scalar_space
        np.testing.assert_allclose(scalar_values, 2.0)
        assert vector_values.size > 0

    def test_build_plot_mesh_data_reuses_cached_dg1_space(self, mesh_with_plot_fields):
        """The plotting mesh helper should reuse the mesh-side cached DG1 space."""
        first_topology, first_cell_types, first_geometry, first_space = (
            mesh_with_plot_fields._build_plot_mesh_data()
        )
        second_topology, second_cell_types, second_geometry, second_space = (
            mesh_with_plot_fields._build_plot_mesh_data()
        )

        assert first_space is second_space
        assert mesh_with_plot_fields._dg_spaces[1] is first_space
        assert first_topology is second_topology
        assert first_cell_types is second_cell_types
        assert first_geometry is second_geometry

    def test_process_fields_for_plotting_treats_non_component_underscores_as_field_names(
        self, mesh_with_plot_fields
    ):
        """Non-component suffixes should remain part of the field name."""
        field_values, field_title, field_space = mesh_with_plot_fields.Process_Fields_For_Plotting(
            "phasefield_Hist"
        )

        assert field_title == "Phasefield History"
        assert field_space is mesh_with_plot_fields._dg_spaces[1]
        np.testing.assert_allclose(field_values, 4.0)

    def test_process_fields_for_plotting_supports_tensor_component_suffixes(
        self, mesh_for_plotting
    ):
        """Tensor component suffixes should be parsed explicitly rather than as scalar names."""
        from dolfinx import fem

        tensor_element = mesh_for_plotting._create_element("CG", 1, shape=(2, 2))
        tensor_space = fem.functionspace(mesh_for_plotting.mesh, tensor_element)
        sigma = fem.Function(tensor_space, name="Stress")
        sigma.x.array[:] = np.linspace(0.0, 1.0, sigma.x.array.size)

        mesh_for_plotting.fields["sigma"] = sigma
        mesh_for_plotting.field_order["sigma"] = 1
        mesh_for_plotting.field_type["sigma"] = "CG Tensor"

        field_values, field_title, field_space = mesh_for_plotting.Process_Fields_For_Plotting(
            "sigma_xx"
        )

        assert field_title == "Stress - XX component"
        assert field_space is mesh_for_plotting._dg_spaces[1]
        assert field_values.size > 0

    def test_process_fields_for_plotting_returns_none_when_preparation_fails(
        self, mesh_for_plotting, monkeypatch
    ):
        """The plotting compatibility wrapper should propagate ``None`` from field preparation."""
        monkeypatch.setattr(mesh_for_plotting, "Prepare_Field_For_Output", lambda *args, **kwargs: None)

        assert mesh_for_plotting.Process_Fields_For_Plotting("missing") is None

    def test_get_field_dg_degree_uses_dg0_for_quadrature_fields(self, mesh_for_plotting):
        """Quadrature fields should use cell-wise DG0 for output and plotting."""
        mesh_for_plotting.Add_Quadrature_Field("stress", "Stress")

        assert mesh_for_plotting._get_field_dg_degree("stress") == 0

    def test_process_fields_for_plotting_supports_scalar_quadrature_fields(self, mesh_for_plotting):
        """Scalar quadrature fields should be projected to DG0 cell averages for plotting."""
        stress_q = mesh_for_plotting.Add_Quadrature_Field("stress_q", "Stress Q")
        stress_q.x.array[:] = 2.5

        field_values, field_title, field_space = mesh_for_plotting.Process_Fields_For_Plotting(
            "stress_q"
        )

        assert field_title == "Stress Q"
        assert field_space is mesh_for_plotting._dg_spaces[0]
        assert field_values.size > 0
        np.testing.assert_allclose(field_values, 2.5)

    def test_process_fields_for_plotting_supports_vector_quadrature_component_suffixes(
        self, mesh_for_plotting
    ):
        """Vector quadrature fields should support both axis and numeric suffixes."""
        traction_q = mesh_for_plotting.Add_Quadrature_Field("traction_q", "Traction Q", (2,))
        traction_q.x.array[0::2] = 1.25
        traction_q.x.array[1::2] = 2.5

        x_values, x_title, x_space = mesh_for_plotting.Process_Fields_For_Plotting("traction_q_x")
        one_values, one_title, one_space = mesh_for_plotting.Process_Fields_For_Plotting("traction_q_1")

        assert x_title == "Traction Q - X component"
        assert one_title == "Traction Q - 1 component"
        assert x_space is mesh_for_plotting._dg_spaces[0]
        assert one_space is x_space
        np.testing.assert_allclose(x_values, one_values)

    def test_process_fields_for_plotting_supports_tensor_quadrature_component_suffixes(
        self, mesh_for_plotting
    ):
        """Tensor quadrature fields should support both symbolic and numeric suffixes."""
        sigma_q = mesh_for_plotting.Add_Quadrature_Field("sigma_q", "Stress Q", (2, 2))
        sigma_q.x.array[0::4] = 3.0
        sigma_q.x.array[1::4] = 4.0
        sigma_q.x.array[2::4] = 5.0
        sigma_q.x.array[3::4] = 6.0

        xx_values, xx_title, xx_space = mesh_for_plotting.Process_Fields_For_Plotting("sigma_q_xx")
        oneone_values, oneone_title, oneone_space = mesh_for_plotting.Process_Fields_For_Plotting("sigma_q_11")

        assert xx_title == "Stress Q - XX component"
        assert oneone_title == "Stress Q - 11 component"
        assert xx_space is mesh_for_plotting._dg_spaces[0]
        assert oneone_space is xx_space
        np.testing.assert_allclose(xx_values, oneone_values)

    def test_process_fields_for_plotting_supports_degenerate_tensor_numeric_suffixes(
        self, mesh_for_plotting
    ):
        """Degenerate tensor quadrature fields such as ``(2, 1)`` should accept ``_1`` and ``_2``."""
        lambda_q = mesh_for_plotting.Add_Quadrature_Field("LambdaField", "Lambda field", (2, 1))
        lambda_q.x.array[0::2] = 0.75
        lambda_q.x.array[1::2] = 1.5

        first_values, first_title, first_space = mesh_for_plotting.Process_Fields_For_Plotting(
            "LambdaField_1"
        )
        second_values, second_title, second_space = mesh_for_plotting.Process_Fields_For_Plotting(
            "LambdaField_2"
        )

        assert first_title == "Lambda field - 1 component"
        assert second_title == "Lambda field - 2 component"
        assert first_space is mesh_for_plotting._dg_spaces[0]
        assert second_space is first_space
        np.testing.assert_allclose(first_values, 0.75)
        np.testing.assert_allclose(second_values, 1.5)

    def test_process_fields_for_plotting_supports_mandel_tensor_suffixes(
        self, mesh_for_plotting
    ):
        """Mandel-stored stress vectors should accept tensor suffixes and unscale shear components."""
        sigma_q = mesh_for_plotting.Add_Quadrature_Field("sigma", "Stress field", (6, 1))
        sigma_q.x.array[:] = 0.0
        sigma_q.x.array[0::6] = 11.0
        sigma_q.x.array[1::6] = 22.0
        sigma_q.x.array[5::6] = np.sqrt(2.0) * 33.0

        xx_values, xx_title, xx_space = mesh_for_plotting.Process_Fields_For_Plotting("sigma_xx")
        yy_values, yy_title, yy_space = mesh_for_plotting.Process_Fields_For_Plotting("sigma_yy")
        xy_values, xy_title, xy_space = mesh_for_plotting.Process_Fields_For_Plotting("sigma_xy")

        assert xx_title == "Stress field - XX component"
        assert yy_title == "Stress field - YY component"
        assert xy_title == "Stress field - XY component"
        assert xx_space is mesh_for_plotting._dg_spaces[0]
        assert yy_space is xx_space
        assert xy_space is xx_space
        np.testing.assert_allclose(xx_values, 11.0)
        np.testing.assert_allclose(yy_values, 22.0)
        np.testing.assert_allclose(xy_values, 33.0)

    def test_build_rank_field_grids_uses_cell_data_for_quadrature_plot_fields(self, mesh_for_plotting):
        """Quadrature plotting data should be attached as cell data on the DG1 plot grid."""
        stress_q = mesh_for_plotting.Add_Quadrature_Field("stress_q", "Stress Q")
        stress_q.x.array[:] = 2.5

        topology, cell_types, geometry, V_dg1 = mesh_for_plotting._build_plot_mesh_data()
        gathered = mesh_for_plotting._collect_local_field_plot_data(["stress_q"], V_dg1)

        rank_grids = mesh_for_plotting._build_rank_field_grids(
            "stress_q",
            [topology],
            [cell_types],
            [geometry],
            [gathered],
        )

        assert len(rank_grids) == 1
        assert "stress_q" in rank_grids[0].cell_data
        assert "stress_q" not in rank_grids[0].point_data

    def test_process_fields_for_plotting_reuses_quadrature_projection_cache(self, mesh_for_plotting):
        """Repeated quadrature plotting calls should reuse the cached DG projection field."""
        history_q = mesh_for_plotting.Add_Quadrature_Field("history_q", "History Q")
        history_q.x.array[:] = 1.0

        mesh_for_plotting.Process_Fields_For_Plotting("history_q")
        cache_size_before = len(mesh_for_plotting._output_projection_cache)

        history_q.x.array[:] = 2.0
        field_values, _, _ = mesh_for_plotting.Process_Fields_For_Plotting("history_q")

        assert len(mesh_for_plotting._output_projection_cache) == cache_size_before
        np.testing.assert_allclose(field_values, 2.0)


class TestMeshPlottingWhiteBoxEdgeCases:
    """Focused coverage for internal plotting-helper branches."""

    def test_plotting_base_hook_returns_without_reduced_order_mixin(self):
        """The base plotting hook should tolerate classes without RO plotting."""
        from femcubes.core.mesh._mesh_plotting import _MeshPlotting

        _MeshPlotting._plot_reduced_order_fields_from_plot_fields(SimpleNamespace(), 0)

    def test_reduced_order_rank_guards_and_data_edge_cases(
        self, mesh_for_plotting, monkeypatch, tmp_path
    ):
        """RO helper paths should handle non-root and malformed gathered data."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        ro_plotter = _FakePlotterHandle()
        mesh_for_plotting._recording = True
        mesh_for_plotting._ro_field_plots = {"h": SimpleNamespace(plotter=ro_plotter)}

        monkeypatch.setattr(mesh_plotting, "rank", 1)
        mesh_for_plotting._start_reduced_order_recording(str(tmp_path), fps=5)
        mesh_for_plotting._save_reduced_order_plots(str(tmp_path))
        assert ro_plotter.opened_movies == []
        assert ro_plotter.screenshots == []

        monkeypatch.setattr(mesh_plotting, "rank", 0)
        mesh_for_plotting._ro_field_plots = {}
        topology = np.array([3, 0, 1, 2], dtype=np.int64)
        cell_types = np.array([5], dtype=np.uint8)
        geometry = np.zeros((3, 2))
        field_payload = (np.array([1.0, 2.0, 3.0]), "Thickness", "point")

        mesh_for_plotting._upsert_reduced_order_field_plot(
            "h",
            None,
            None,
            None,
            [{"surface": {"z": np.zeros(3), "h": np.ones(3)}, "fields": {"h": field_payload}}],
        )
        assert "h" not in mesh_for_plotting._ro_field_plots

        no_records = list(
            mesh_for_plotting._iter_reduced_order_rank_data(
                "h",
                [topology, topology],
                [cell_types, cell_types],
                [geometry, geometry],
                [
                    {"surface": None, "fields": {"h": field_payload}},
                    {
                        "surface": {"z": np.zeros(2), "h": np.ones(2)},
                        "fields": {"h": field_payload},
                    },
                ],
            )
        )
        assert no_records == []

        merged = mesh_for_plotting._build_merged_reduced_order_grids(
            "h",
            [topology],
            [cell_types],
            [geometry],
            [
                {
                    "surface": {"z": np.zeros(2), "h": np.ones(2)},
                    "fields": {"h": field_payload},
                }
            ],
        )
        assert merged is None

        empty_values = mesh_for_plotting._collect_reduced_order_plot_values(
            "h",
            [
                {},
                {"surface": {"z": np.array([]), "h": np.array([])}, "fields": {"h": (np.array([]), "Thickness", "point")}},
            ],
        )
        assert empty_values == (None, None, None, None)

        mismatch_values = mesh_for_plotting._collect_reduced_order_plot_values(
            "h",
            [
                {
                    "surface": {"z": np.zeros(2), "h": np.ones(3)},
                    "fields": {"h": (np.ones(2), "Thickness", "point")},
                }
            ],
        )
        assert mismatch_values == (None, None, None, None)

        existing_state = SimpleNamespace()
        mesh_for_plotting._ro_field_plots = {"h": existing_state}
        mesh_for_plotting._upsert_reduced_order_field_plot(
            "h",
            None,
            None,
            None,
            [{"surface": {"z": np.array([]), "h": np.array([])}, "fields": {"h": (np.array([]), "Thickness", "point")}}],
        )
        assert mesh_for_plotting._ro_field_plots["h"] is existing_state

    def test_reduced_order_refresh_render_and_size_fallbacks(self, mesh_for_plotting):
        """RO refresh should tolerate size mismatches and render-only plotters."""
        import pyvista as pv

        state = SimpleNamespace(
            basal_grid=SimpleNamespace(points=np.zeros((3, 3))),
            field_grid=SimpleNamespace(points=np.zeros((3, 3))),
        )
        mesh_for_plotting._ro_field_plots = {"h": state}
        mesh_for_plotting._refresh_reduced_order_plot_state(
            "h",
            z_values=np.zeros(2),
            h_values=np.ones(2),
            field_values=np.ones(3),
        )

        class _FakeMapper:
            def __init__(self):
                self.ranges = []

            def SetScalarRange(self, data_range):
                self.ranges.append(tuple(data_range))

        class _FakeActor:
            def __init__(self):
                self.mapper = _FakeMapper()

            def GetMapper(self):
                return self.mapper

        class _RenderOnlyPlotter:
            def __init__(self):
                self.render_calls = 0

            def render(self):
                self.render_calls += 1

        topology = np.array([3, 0, 1, 2], dtype=np.int64)
        cell_types = np.array([5], dtype=np.uint8)
        points = np.zeros((3, 3))
        basal_grid = pv.UnstructuredGrid(topology, cell_types, points)
        field_grid = pv.UnstructuredGrid(topology, cell_types, points.copy())
        field_grid.point_data["h"] = np.ones(3)
        plotter = _RenderOnlyPlotter()
        mesh_for_plotting._ro_field_plots = {
            "h": SimpleNamespace(
                plotter=plotter,
                basal_grid=basal_grid,
                basal_fill_grid=None,
                basal_side_grid=None,
                floating_base_grid=None,
                ocean_surface_grid=None,
                field_grid=field_grid,
                field_actor=_FakeActor(),
            )
        }

        mesh_for_plotting._refresh_reduced_order_plot_state(
            "h",
            z_values=np.zeros(3),
            h_values=np.ones(3),
            field_values=np.array([1.0, 2.0, 3.0]),
        )

        assert plotter.render_calls == 1

        ocean_grid = pv.PolyData(points.copy(), faces=np.array([3, 0, 1, 2], dtype=np.int64))
        ocean_state = SimpleNamespace(
            basal_grid=basal_grid,
            ocean_surface_grid=ocean_grid,
            ocean_surface_actor=object(),
            plotter=plotter,
        )
        ocean_points = np.array(
            [
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )
        mesh_for_plotting._refresh_reduced_order_ocean_surface_grid(
            ocean_state,
            ocean_points,
            np.ones(3, dtype=bool),
        )
        np.testing.assert_allclose(ocean_state.ocean_surface_grid.points, ocean_points)

        undersized_fill_grid = SimpleNamespace(points=np.zeros((1, 3)))
        mesh_for_plotting._refresh_reduced_order_basal_fill_grid(
            undersized_fill_grid,
            np.zeros((3, 3)),
        )
        np.testing.assert_allclose(undersized_fill_grid.points, np.zeros((1, 3)))

    def test_reduced_order_camera_fallbacks(self, mesh_for_plotting):
        """Camera setup should fall back cleanly when optional backend APIs fail."""
        empty_grid = SimpleNamespace(points=None)

        class _FallbackPlotter:
            def __init__(self):
                self.viewed_iso = 0

            def view_isometric(self):
                self.viewed_iso += 1

        plotter = _FallbackPlotter()
        mesh_for_plotting._set_reduced_order_camera(plotter, empty_grid, empty_grid)
        assert plotter.viewed_iso == 1

        class _FailingCamera:
            def SetPosition(self, *args):
                raise RuntimeError("cannot set camera")

        class _FailingCameraPlotter(_FallbackPlotter):
            def __init__(self):
                super().__init__()
                self.camera = _FailingCamera()

            @property
            def camera_position(self):
                return None

            @camera_position.setter
            def camera_position(self, value):
                raise RuntimeError("camera_position unavailable")

        points_grid = SimpleNamespace(points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        failing_plotter = _FailingCameraPlotter()
        mesh_for_plotting._set_reduced_order_camera(failing_plotter, points_grid, points_grid)
        assert failing_plotter.viewed_iso == 1

        class _MethodCamera:
            def __init__(self):
                self.position = None
                self.focal_point = None
                self.view_up = None

            def SetPosition(self, *args):
                self.position = args

            def SetFocalPoint(self, *args):
                self.focal_point = args

            def SetViewUp(self, *args):
                self.view_up = args

        class _MethodCameraPlotter:
            def __init__(self):
                self.camera = _MethodCamera()

            @property
            def camera_position(self):
                return None

            @camera_position.setter
            def camera_position(self, value):
                raise RuntimeError("camera_position unavailable")

        method_plotter = _MethodCameraPlotter()
        mesh_for_plotting._set_reduced_order_camera(method_plotter, points_grid, points_grid)
        assert method_plotter.camera.position is not None
        assert method_plotter.camera.focal_point is not None
        assert method_plotter.camera.view_up == (0.0, 0.0, 1.0)

        class _AttributeCamera:
            def __init__(self):
                self.position = None
                self.focal_point = None
                self.view_up = None

        class _AttributeCameraPlotter:
            def __init__(self):
                self.camera = _AttributeCamera()

            @property
            def camera_position(self):
                return None

            @camera_position.setter
            def camera_position(self, value):
                raise RuntimeError("camera_position unavailable")

        attribute_plotter = _AttributeCameraPlotter()
        mesh_for_plotting._set_reduced_order_camera(attribute_plotter, points_grid, points_grid)
        assert attribute_plotter.camera.position is not None
        assert attribute_plotter.camera.focal_point is not None
        assert attribute_plotter.camera.view_up == (0.0, 0.0, 1.0)

        class _ClippingPlotter:
            def __init__(self):
                self.camera_position = None
                self.reset_calls = 0

            def reset_camera_clipping_range(self):
                self.reset_calls += 1
                raise RuntimeError("clip failed")

        clipping_plotter = _ClippingPlotter()
        mesh_for_plotting._set_reduced_order_camera(clipping_plotter, points_grid, points_grid)
        assert clipping_plotter.reset_calls == 1

    def test_reduced_order_geometry_and_grid_guard_branches(self, mesh_for_plotting, monkeypatch):
        """RO geometry helpers should reject invalid inputs and empty mesh fragments."""
        import pyvista as pv

        mesh_for_plotting.params.Level_Set_Transition_Width = 2.0
        np.testing.assert_allclose(
            mesh_for_plotting._smooth_level_set_heaviside_values(
                np.array([-3.0, 0.0, 3.0])
            ),
            [0.0, 0.5, 1.0],
        )
        del mesh_for_plotting.params.Level_Set_Transition_Width
        with pytest.raises(ValueError):
            mesh_for_plotting._get_level_set_transition_width()
        mesh_for_plotting.params.Level_Set_Transition_Width = -1.0
        with pytest.raises(ValueError):
            mesh_for_plotting._get_level_set_transition_width()

        mesh_for_plotting.params.Materials = {"Ice": {"Density": 0.0}, "Water": {"Density": 1030.0}}
        with pytest.raises(ValueError):
            mesh_for_plotting._get_reduced_order_densities()

        mesh_for_plotting.params.Materials = {"Ice": {"Density": 917.0}, "Water": {"Density": 1030.0, "WaterHeight": 2.5}}
        assert mesh_for_plotting._get_reduced_order_sea_level() == 2.5
        mesh_for_plotting.params.RO_sea_level = 4.0
        assert mesh_for_plotting._get_reduced_order_sea_level() == 4.0
        del mesh_for_plotting.params.RO_sea_level

        mesh_for_plotting.params.RO_min_surface_thickness = -1.0
        with pytest.raises(ValueError):
            mesh_for_plotting._get_reduced_order_min_surface_thickness()
        mesh_for_plotting.params.RO_min_surface_thickness = 0.01

        mesh_for_plotting.params.RO_zscale = 0.0
        with pytest.raises(ValueError):
            mesh_for_plotting._get_reduced_order_zscale()
        mesh_for_plotting.params.RO_zscale = 1.0

        invalid_grid = SimpleNamespace(points=np.zeros((0, 3)))
        assert mesh_for_plotting._build_reduced_order_basal_fill_grid(invalid_grid) is invalid_grid
        assert mesh_for_plotting._build_reduced_order_basal_side_grid(invalid_grid) is None

        no_cells_grid = SimpleNamespace(points=np.zeros((3, 3)), cells=None)
        assert mesh_for_plotting._build_reduced_order_basal_fill_grid(no_cells_grid) is no_cells_grid
        assert mesh_for_plotting._build_reduced_order_basal_side_grid(no_cells_grid) is None

        points = np.zeros((3, 3))
        topology_line = np.array([2, 0, 1], dtype=np.int64)
        filtered_topology, filtered_types = mesh_for_plotting._filter_reduced_order_surface_topology(
            topology_line,
            np.array([3], dtype=np.uint8),
            np.ones(3),
        )
        assert filtered_topology.size == 0
        assert filtered_types.size == 0
        assert mesh_for_plotting._build_reduced_order_basal_fill_faces(points, [[0, 1]]).size == 0

        triangle_grid = pv.UnstructuredGrid(np.array([3, 0, 1, 2], dtype=np.int64), np.array([5], dtype=np.uint8), np.zeros((3, 3)))
        monkeypatch.setattr(mesh_for_plotting, "_build_reduced_order_basal_fill_faces", lambda *_args: np.asarray([], dtype=np.int64))
        assert mesh_for_plotting._build_reduced_order_basal_fill_grid(triangle_grid) is triangle_grid
        monkeypatch.setattr(mesh_for_plotting, "_build_reduced_order_basal_side_faces", lambda *_args: np.asarray([], dtype=np.int64))
        assert mesh_for_plotting._build_reduced_order_basal_side_grid(triangle_grid) is None

        V_dg = mesh_for_plotting._get_dg_space(1)
        monkeypatch.setattr(mesh_for_plotting, "Prepare_Field_For_Plotting", lambda *_args, **_kwargs: None)
        assert mesh_for_plotting._project_h_for_ro_geometry(V_dg) is None

        assert mesh_for_plotting._build_reduced_order_floating_base_grid(np.array([], dtype=np.int64), np.zeros((0, 3)), np.array([], dtype=bool)) is None
        assert mesh_for_plotting._build_reduced_order_floating_base_grid(topology_line, points, np.ones(3, dtype=bool)) is None
        assert mesh_for_plotting._build_reduced_order_ocean_surface_grid(np.array([], dtype=np.int64), np.zeros((0, 3)), np.array([], dtype=bool)) is None
        assert mesh_for_plotting._build_reduced_order_ocean_surface_grid(topology_line, points, np.ones(3, dtype=bool)) is None

        np.testing.assert_array_equal(
            list(mesh_for_plotting._iter_vtk_topology_cells(np.array([[3, 0, 1, 2], [0, 0, 0, 0]], dtype=np.int64))),
            [[0, 1, 2]],
        )
        assert mesh_for_plotting._extract_reduced_order_surface_cells(SimpleNamespace()) == []
        assert mesh_for_plotting._collect_reduced_order_boundary_edges(points, [[0, 1]]) == []

    def test_reduced_order_array_copy_and_validation_fallbacks(self, mesh_for_plotting):
        """Array helpers should cover fallback mutation and validation warnings."""
        assert mesh_for_plotting._as_3d_plot_points(np.array([1.0, 2.0])).shape == (0, 3)

        class _SliceOnlyGrid:
            def __init__(self):
                self._points = np.zeros((2, 3))

            @property
            def points(self):
                return self._points

            @points.setter
            def points(self, value):
                raise RuntimeError("direct assignment disabled")

        grid = _SliceOnlyGrid()
        new_points = np.ones((2, 3))
        mesh_for_plotting._set_grid_points(grid, new_points)
        np.testing.assert_allclose(grid.points, new_points)

        target = SimpleNamespace(points=np.zeros((2, 3)))
        source = SimpleNamespace(points=np.ones((2, 3)))
        mesh_for_plotting._copy_grid_data(target, source)
        np.testing.assert_allclose(target.points, source.points)

        assert mesh_for_plotting._ro_value_lengths_match("h", 0, [], [], []) is False
        assert mesh_for_plotting._ro_value_lengths_match("h", 3, [1.0], [1.0], [1.0]) is False

    def test_apply_plotter_interaction_mode_disables_vtk_interactor(self, mesh_for_plotting):
        """The interaction helper should disable both plotter and VTK interactor."""

        class _FakeInteractor:
            def __init__(self):
                self.disable_calls = 0

            def Disable(self):
                self.disable_calls += 1

        class _FakeIren:
            def __init__(self):
                self.interactor = _FakeInteractor()

        class _FakePlotter:
            def __init__(self):
                self.disable_calls = 0
                self.iren = _FakeIren()

            def disable(self):
                self.disable_calls += 1

        plotter = _FakePlotter()

        mesh_for_plotting._apply_plotter_interaction_mode(plotter)

        assert plotter.disable_calls == 1
        assert plotter.iren.interactor.disable_calls == 1

    def test_apply_plotter_interaction_mode_falls_back_to_set_enable(self, mesh_for_plotting):
        """The interaction helper should use ``SetEnable(0)`` when ``Disable`` is unavailable."""

        class _FakeInteractor:
            def __init__(self):
                self.enable_values = []

            def SetEnable(self, value):
                self.enable_values.append(value)

        class _FakeIren:
            def __init__(self):
                self.interactor = _FakeInteractor()

        class _FakePlotter:
            def __init__(self):
                self.disable_calls = 0
                self.iren = _FakeIren()

            def disable(self):
                self.disable_calls += 1
                raise RuntimeError("disable failed")

        plotter = _FakePlotter()

        mesh_for_plotting._apply_plotter_interaction_mode(plotter)

        assert plotter.disable_calls == 1
        assert plotter.iren.interactor.enable_values == [0]

    def test_apply_plotter_interaction_mode_tolerates_interactor_disable_errors(self, mesh_for_plotting):
        """Interactor failures should be swallowed so plotting continues."""

        class _FakeInteractor:
            def Disable(self):
                raise RuntimeError("cannot disable interactor")

        class _FakeIren:
            def __init__(self):
                self.interactor = _FakeInteractor()

        class _FakePlotter:
            def __init__(self):
                self.disable_calls = 0
                self.iren = _FakeIren()

            def disable(self):
                self.disable_calls += 1

        plotter = _FakePlotter()

        mesh_for_plotting._apply_plotter_interaction_mode(plotter)

        assert plotter.disable_calls == 1

    def test_add_boundary_overlays_returns_without_boundary_markers(self, mesh_for_plotting):
        """Boundary overlays should no-op when no boundary markers are defined."""

        class _FakePlotter:
            def __init__(self):
                self.mesh_calls = []
                self.label_calls = []

            def add_mesh(self, *args, **kwargs):
                self.mesh_calls.append((args, kwargs))

            def add_point_labels(self, *args, **kwargs):
                self.label_calls.append((args, kwargs))

        mesh_for_plotting.boundary_markers = {}
        plotter = _FakePlotter()

        mesh_for_plotting._add_boundary_overlays(plotter, all_boundary_data=[{}])

        assert plotter.mesh_calls == []
        assert plotter.label_calls == []

    def test_add_boundary_overlays_skips_missing_and_empty_rank_data(self, mesh_for_plotting):
        """Boundary overlays should ignore missing ranks and empty boundary fragments."""

        class _FakePlotter:
            def __init__(self):
                self.mesh_calls = []
                self.label_calls = []

            def add_mesh(self, *args, **kwargs):
                self.mesh_calls.append((args, kwargs))

            def add_point_labels(self, *args, **kwargs):
                self.label_calls.append((args, kwargs))

        mesh_for_plotting.boundary_markers = {"left": 1}
        plotter = _FakePlotter()

        mesh_for_plotting._add_boundary_overlays(
            plotter,
            all_boundary_data=[
                {},
                {"left": {"points": np.zeros((0, 2)), "lines": [], "faces": []}},
            ],
        )

        assert plotter.mesh_calls == []
        assert plotter.label_calls == []

    def test_build_boundary_mesh_returns_none_without_connectivity(self, mesh_for_plotting):
        """Boundary meshes should not be created when both lines and faces are absent."""
        assert mesh_for_plotting._build_boundary_mesh(np.zeros((0, 2)), [], []) is None

    def test_get_domain_label_text_defaults_without_markers(self, mesh_for_plotting):
        """Mesh plots should use a generic domain label when no marker name is available."""
        mesh_for_plotting.domain_markers = {}

        assert mesh_for_plotting._get_domain_label_text() == "Domain"

    def test_collect_boundary_plot_data_returns_empty_without_tags_or_markers(self, mesh_for_plotting):
        """Boundary collection should no-op when tags or marker names are unavailable."""
        original_boundary_markers = mesh_for_plotting.boundary_markers
        mesh_for_plotting.facet_tags = None
        assert mesh_for_plotting._collect_boundary_plot_data() == {}

        mesh_for_plotting.facet_tags = SimpleNamespace(find=lambda _: np.array([], dtype=np.int32))
        mesh_for_plotting.boundary_markers = {}
        assert mesh_for_plotting._collect_boundary_plot_data() == {}

        mesh_for_plotting.boundary_markers = original_boundary_markers

    def test_collect_boundary_plot_data_skips_unmapped_topology_vertices(self, mesh_for_plotting):
        """Boundary collection should ignore facet vertices that do not map into the cell."""

        class _DummyConnectivity:
            def __init__(self, mapping):
                self.mapping = mapping

            def links(self, index):
                return self.mapping[index]

        class _DummyTopology:
            def __init__(self):
                self.dim = 2
                self._connectivities = {
                    (1, 2): _DummyConnectivity({0: np.array([0], dtype=np.int32)}),
                    (1, 0): _DummyConnectivity({0: np.array([99, 100], dtype=np.int32)}),
                    (2, 0): _DummyConnectivity({0: np.array([0, 1, 2], dtype=np.int32)}),
                }

            def create_connectivity(self, *args, **kwargs):
                return None

            def connectivity(self, from_dim, to_dim):
                return self._connectivities[(from_dim, to_dim)]

        mesh_for_plotting.mesh = SimpleNamespace(
            topology=_DummyTopology(),
            geometry=SimpleNamespace(
                dofmap=np.array([[0, 1, 2]], dtype=np.int32),
                x=np.zeros((3, 2)),
            ),
        )
        mesh_for_plotting.facet_tags = SimpleNamespace(find=lambda _: np.array([0], dtype=np.int32))
        mesh_for_plotting.boundary_markers = {"left": 1}

        assert mesh_for_plotting._collect_boundary_plot_data(tdim=2) == {}

    def test_field_plot_helpers_skip_missing_or_empty_rank_data(self, mesh_for_plotting):
        """Field-plot helpers should ignore ranks with no plotted data."""
        all_topologies = [np.array([], dtype=np.int32), np.array([], dtype=np.int32)]
        all_cell_types = [np.array([], dtype=np.uint8), np.array([], dtype=np.uint8)]
        all_geometries = [np.zeros((0, 2)), np.zeros((0, 2))]
        all_rank_field_data = [
            {},
            {"temperature": (np.array([]), "Temperature")},
        ]

        rank_grids = mesh_for_plotting._build_rank_field_grids(
            "temperature",
            all_topologies,
            all_cell_types,
            all_geometries,
            all_rank_field_data,
        )
        merged = mesh_for_plotting._build_merged_field_grid(
            "temperature",
            all_topologies,
            all_cell_types,
            all_geometries,
            all_rank_field_data,
        )
        mesh_for_plotting._upsert_field_plot(
            "temperature",
            all_topologies,
            all_cell_types,
            all_geometries,
            all_rank_field_data,
        )

        assert rank_grids == []
        assert merged is None
        assert "temperature" not in mesh_for_plotting._field_plots

    def test_existing_field_plot_returns_when_refresh_data_is_absent(self, mesh_for_plotting):
        """Existing field plots should ignore refreshes with no scalar payload."""
        state = _field_plot_state(plotter=object(), grid=object(), actor=object())
        mesh_for_plotting._field_plots = {"temperature": state}

        mesh_for_plotting._upsert_field_plot(
            "temperature",
            all_topologies=None,
            all_cell_types=None,
            all_geometries=None,
            all_rank_field_data=[{"temperature": (np.array([]), "Temperature", "point")}],
        )

        assert mesh_for_plotting._field_plots["temperature"] is state

    def test_field_plot_value_collection_skips_empty_data_and_rejects_mixed_associations(
        self, mesh_for_plotting
    ):
        """Field-value collection should skip empty ranks and reject inconsistent associations."""
        empty_values, empty_association, empty_level_set = mesh_for_plotting._collect_field_plot_values(
            "temperature",
            [
                {},
                {"temperature": (np.array([]), "Temperature", "point")},
            ],
        )

        mixed_values, mixed_association, mixed_level_set = mesh_for_plotting._collect_field_plot_values(
            "temperature",
            [
                {"temperature": (np.array([1.0]), "Temperature", "point")},
                {"temperature": (np.array([2.0]), "Temperature", "cell")},
            ],
        )

        assert (empty_values, empty_association, empty_level_set) == (None, None, None)
        assert (mixed_values, mixed_association, mixed_level_set) == (None, None, None)

    def test_merged_field_grid_handles_cell_data_and_2d_topology_offsets(
        self, mesh_for_plotting, monkeypatch
    ):
        """Merged plot grids should offset per-rank topology and attach cell data."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        class _FakeGrid:
            def __init__(self, topology, cell_types, geometry):
                self.topology = topology
                self.cell_types = cell_types
                self.geometry = geometry
                self.point_data = {}
                self.cell_data = {}

        class _FakePV:
            UnstructuredGrid = _FakeGrid

        monkeypatch.setattr(mesh_plotting, "pv", _FakePV)

        topology = np.array([[3, 0, 1, 2]], dtype=np.int32)
        cell_types = np.array([5], dtype=np.uint8)
        geometry = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        merged = mesh_for_plotting._build_merged_field_grid(
            "stress",
            all_topologies=[topology, topology],
            all_cell_types=[cell_types, cell_types],
            all_geometries=[geometry, geometry + 2.0],
            all_rank_field_data=[
                {"stress": (np.array([1.0]), "Stress", "cell")},
                {"stress": (np.array([2.0]), "Stress", "cell")},
            ],
        )

        np.testing.assert_array_equal(merged.topology, np.array([[3, 0, 1, 2], [3, 3, 4, 5]]))
        np.testing.assert_allclose(merged.cell_data["stress"], [1.0, 2.0])
        assert "stress" not in merged.point_data

    def test_merged_field_grid_rejects_mixed_associations(self, mesh_for_plotting):
        """Merged plot grids should not mix point and cell scalar associations."""
        topology = np.array([3, 0, 1, 2], dtype=np.int32)
        cell_types = np.array([5], dtype=np.uint8)
        geometry = np.zeros((3, 2))

        merged = mesh_for_plotting._build_merged_field_grid(
            "temperature",
            all_topologies=[topology, topology],
            all_cell_types=[cell_types, cell_types],
            all_geometries=[geometry, geometry],
            all_rank_field_data=[
                {"temperature": (np.array([1.0, 2.0, 3.0]), "Temperature", "point")},
                {"temperature": (np.array([4.0]), "Temperature", "cell")},
            ],
        )

        assert merged is None

    def test_build_rank_field_grids_attaches_point_data(self, mesh_for_plotting):
        """Rank-local plot grids should attach non-quadrature values as point data."""
        topology = np.array([3, 0, 1, 2], dtype=np.int32)
        cell_types = np.array([5], dtype=np.uint8)
        geometry = np.zeros((3, 3))

        rank_grids = mesh_for_plotting._build_rank_field_grids(
            "temperature",
            all_topologies=[topology],
            all_cell_types=[cell_types],
            all_geometries=[geometry],
            all_rank_field_data=[
                {"temperature": (np.array([1.0, 2.0, 3.0]), "Temperature", "point")}
            ],
        )

        assert len(rank_grids) == 1
        np.testing.assert_allclose(rank_grids[0].point_data["temperature"], [1.0, 2.0, 3.0])
        assert "temperature" not in rank_grids[0].cell_data

    def test_format_time_label_includes_days_and_hours(self, mesh_for_plotting):
        """Time labels should include day and hour components for long runs."""
        mesh_for_plotting.params.t = -90061.0

        assert mesh_for_plotting._format_time_label() == "t = -1 D  1 H  6.100e+01 s"
        
        mesh_for_plotting.params.t = 31536000 + 3661.0  # 1 year + 1 hour + 61 second
        assert mesh_for_plotting._format_time_label() == "t = 1 Y  1 H  6.100e+01 s"

    def test_refresh_field_plot_state_updates_cell_data(self, mesh_for_plotting):
        """Existing field plots should update cell data in-place for quadrature-style plots."""

        class _FakeMapper:
            def __init__(self):
                self.ranges = []

            def SetScalarRange(self, data_range):
                self.ranges.append(tuple(data_range))

        class _FakeActor:
            def __init__(self):
                self.mapper = _FakeMapper()

            def GetMapper(self):
                return self.mapper

        class _FakePlotter:
            def __init__(self):
                self.update_calls = 0

            def update(self):
                self.update_calls += 1

        class _FakeGrid:
            def __init__(self):
                self.point_data = {}
                self.cell_data = {}
                self.modified_calls = 0

            def Modified(self):
                self.modified_calls += 1

        plotter = _FakePlotter()
        actor = _FakeActor()
        state_grid = _FakeGrid()
        mesh_for_plotting._field_plots = {"stress_q": _field_plot_state(plotter, state_grid, actor)}

        mesh_for_plotting._refresh_field_plot_state(
            "stress_q",
            np.array([1.0, 2.0]),
            "cell",
        )

        np.testing.assert_allclose(state_grid.cell_data["stress_q"], [1.0, 2.0])

        mesh_for_plotting._refresh_field_plot_state(
            "stress_q",
            np.array([3.0, 4.0]),
            "cell",
        )

        np.testing.assert_allclose(state_grid.cell_data["stress_q"], [3.0, 4.0])
        assert actor.mapper.ranges[-1] == (3.0, 4.0)
        assert plotter.update_calls == 2
        assert state_grid.modified_calls == 2

    def test_set_window_icon_sets_cached_icon_when_available(self, monkeypatch):
        """The icon helper should cache and set the PNG icon on a plotter window."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        class _FakeReader:
            def SetFileName(self, path):
                self.path = path

            def Update(self):
                return None

            def GetOutput(self):
                return "ICON"

        class _FakeRenWin:
            def __init__(self):
                self.icons = []

            def SetIcon(self, icon):
                self.icons.append(icon)

        plotter = SimpleNamespace(ren_win=_FakeRenWin())
        monkeypatch.setitem(sys.modules, "vtk", SimpleNamespace(vtkPNGReader=_FakeReader))
        monkeypatch.setattr(mesh_plotting.pv, "OFF_SCREEN", False, raising=False)
        monkeypatch.setattr(mesh_plotting, "rank", 0)
        mesh_plotting._MeshPlotting._icon_image = None

        mesh_plotting._MeshPlotting._set_window_icon(plotter)

        assert plotter.ren_win.icons == ["ICON"]
        assert mesh_plotting._MeshPlotting._icon_image == "ICON"

    def test_set_window_icon_swallows_reader_failures(self, monkeypatch):
        """The icon helper should fail silently when icon setup raises."""
        import femcubes.core.mesh._mesh_plotting as mesh_plotting

        class _FailingReader:
            def SetFileName(self, path):
                return None

            def Update(self):
                raise RuntimeError("cannot read icon")

        plotter = SimpleNamespace(ren_win=SimpleNamespace(SetIcon=lambda icon: None))
        monkeypatch.setitem(sys.modules, "vtk", SimpleNamespace(vtkPNGReader=_FailingReader))
        monkeypatch.setattr(mesh_plotting.pv, "OFF_SCREEN", False, raising=False)
        monkeypatch.setattr(mesh_plotting, "rank", 0)
        mesh_plotting._MeshPlotting._icon_image = None

        mesh_plotting._MeshPlotting._set_window_icon(plotter)


def test_line_grids_are_detected_for_thick_interface_field_plots():
    """Fields on interfaces of 2-D meshes are line grids, drawn with thick lines."""
    import pyvista as pv

    from femcubes.core.mesh._mesh_plotting import _MeshPlotting

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lines = pv.UnstructuredGrid(np.array([2, 0, 1, 2, 1, 2]), np.array([pv.CellType.LINE] * 2), points)
    triangle = pv.UnstructuredGrid(np.array([3, 0, 1, 2]), np.array([pv.CellType.TRIANGLE]), points)

    assert _MeshPlotting._is_line_grid(lines)
    assert not _MeshPlotting._is_line_grid(triangle)
