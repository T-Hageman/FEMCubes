"""Tests for mesh file-output functionality."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.refinement import (
    FieldThresholdIndicatorConfig,
    RefinementConfig,
)


@pytest.fixture
def Output_Test_Params(tmp_path):
    """Create a Params instance configured for mesh-output tests."""

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 2.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        output_fields = ["temperature"]
        write_vtk = False
        vtk_single_file = True
        write_hdf5 = False
        output_dir = str(tmp_path / "mesh_outputs")
        output_interval = 1
        save_Mesh_every_step = False
        dt = 0.1
        t = 0.0
        Global_Measures = {}

    return TestParams()


@pytest.fixture
def mesh_with_output_field(Output_Test_Params):
    """Create a mesh with one scalar field ready for output."""
    from femcubes.core.mesh.mesh import Mesh

    mesh = Mesh(Output_Test_Params)
    mesh.Add_Field("temperature", "Temperature", "CG", 1)
    mesh.fields["temperature"].x.array[:] = 2.0
    return mesh


class _FakeVTKFile:
    """Small VTK writer stub used to test output orchestration."""

    instances = []

    def __init__(self, comm, path, mode):
        self.comm = comm
        self.path = path
        self.mode = mode
        self.function_writes = []
        self.mesh_writes = []
        self.close_calls = 0
        _FakeVTKFile.instances.append(self)

    def write_function(self, functions, time):
        self.function_writes.append(([func.name for func in functions], time))

    def write_mesh(self, mesh, time):
        self.mesh_writes.append((mesh, time))

    def close(self):
        self.close_calls += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _SequencedComm:
    """Simple communicator stub with scripted ``allreduce`` results."""

    def __init__(self, allreduce_results):
        self._allreduce_results = list(allreduce_results)

    def Barrier(self):
        return None

    def bcast(self, value, root=0):
        return value

    def allreduce(self, value, op=None):
        result = self._allreduce_results.pop(0)
        return value if result is None else result


class TestMeshOutputs:
    """Tests for the mesh output subsystem."""

    def test_output_init_cleans_existing_output_directory(self, Output_Test_Params):
        """Mesh construction should clear stale output files when output is enabled."""
        from femcubes.core.mesh.mesh import Mesh

        output_dir = Path(Output_Test_Params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        marker_file = output_dir / ".femcubes-output-dir"
        marker_file.write_text("FEMCubes output directory\n", encoding="utf-8")
        stale_file = output_dir / "stale.txt"
        stale_file.write_text("old data")
        Output_Test_Params.write_hdf5 = True

        Mesh(Output_Test_Params)

        assert output_dir.exists()
        assert marker_file.exists()
        assert not stale_file.exists()

    def test_output_init_cleans_time_data_without_field_output(self, Output_Test_Params):
        """A new run should start a fresh time_data.hdf5 even when VTK and HDF5 field output are off."""
        from femcubes.core.mesh.mesh import Mesh

        Output_Test_Params.Global_Measures = {"force": 1.0}
        Mesh(Output_Test_Params).Write_Outputs(step=0, time=0.0)

        Output_Test_Params.Global_Measures = {"disp": 0.25}
        Mesh(Output_Test_Params).Write_Outputs(step=0, time=0.0)

        if MPI.COMM_WORLD.rank == 0:
            with h5py.File(Path(Output_Test_Params.output_dir) / "time_data.hdf5", "r") as h5f:
                names = [name.decode("utf-8") for name in h5f["data_names"][()]]
                assert names == ["time", "dt", "epoch_tag", "disp"]
                assert h5f["data"].shape[0] == 1

    def test_output_init_without_field_output_leaves_other_files(self, Output_Test_Params):
        """Without field output, only a stale time_data.hdf5 is removed, and only from a marked directory."""
        from femcubes.core.mesh.mesh import Mesh

        output_dir = Path(Output_Test_Params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "time_data.hdf5").write_text("old")
        (output_dir / "keep.txt").write_text("user data")

        Mesh(Output_Test_Params)
        assert (output_dir / "time_data.hdf5").exists()  # unmarked directory: untouched

        (output_dir / ".femcubes-output-dir").write_text("FEMCubes output directory\n")
        Mesh(Output_Test_Params)
        assert not (output_dir / "time_data.hdf5").exists()
        assert (output_dir / "keep.txt").exists()

    def test_output_init_refuses_to_clean_unmarked_output_directory(self, Output_Test_Params):
        """Mesh construction should reject destructive cleanup for unmarked directories."""
        from femcubes.core.mesh.mesh import Mesh

        output_dir = Path(Output_Test_Params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stale.txt").write_text("old data")
        Output_Test_Params.write_hdf5 = True

        with pytest.raises(RuntimeError, match="Refusing to delete unmarked output directory"):
            Mesh(Output_Test_Params)

    def test_write_outputs_writes_time_data_and_hdf5_snapshot(self, mesh_with_output_field):
        """Write_Outputs should append time data and write DG field tables to HDF5."""
        mesh = mesh_with_output_field
        mesh.hdf5 = True
        mesh.output_freq = 2
        mesh.params.dt = 0.1
        mesh.params.Global_Measures = {"force": 1.5}

        mesh.Write_Outputs(step=1, time=0.1)

        mesh.params.dt = 0.2
        mesh.params.Global_Measures["force"] = 2.5
        mesh.Write_Outputs(step=2, time=0.3)

        output_dir = Path(mesh.output_dir)
        assert not (output_dir / "Outputs_1.hdf5").exists()
        assert (output_dir / "Outputs_2.hdf5").exists()
        assert (output_dir / "mesh_epoch_000.hdf5").exists()
        assert (output_dir / "time_data.hdf5").exists()
        assert (output_dir / ".femcubes-output-dir").exists()

        with h5py.File(output_dir / "time_data.hdf5", "r") as h5f:
            data_names = [name.decode("utf-8") for name in h5f["data_names"][()]]
            assert data_names == ["time", "dt", "epoch_tag", "force"]
            data = h5f["data"][()]
            np.testing.assert_allclose(data[:, 0], [0.1, 0.3])
            np.testing.assert_allclose(data[:, 1], [0.1, 0.2])
            np.testing.assert_allclose(data[:, 2], [0.0, 0.0])
            np.testing.assert_allclose(data[:, 3], [1.5, 2.5])

        with h5py.File(output_dir / "Outputs_2.hdf5", "r") as h5f:
            assert int(h5f["epoch_tag"][()]) == 0
            assert "epoch_tag" not in h5f["Fields"]
            field_data = h5f["Fields"]["temperature"]
            assert field_data.attrs["dg_degree"] == 1
            assert field_data.shape[0] == mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_global
            np.testing.assert_allclose(field_data[()], 2.0)

        with h5py.File(output_dir / "mesh_epoch_000.hdf5", "r") as h5f:
            assert int(h5f["epoch_tag"][()]) == 0
            assert "Mesh_DG1" in h5f
            assert "X" in h5f["Mesh_DG1"]
            assert "Y" in h5f["Mesh_DG1"]

    def test_write_outputs_projects_quadrature_fields_to_dg0(
        self, mesh_with_output_field
    ):
        """Quadrature output should be cell-wise DG0 rather than reconstructed DG1."""
        mesh = mesh_with_output_field
        stress_q = mesh.Add_Quadrature_Field("stress_q", "Stress Q")
        stress_q.x.array[:] = 2.5
        mesh.output_fields = ["stress_q"]
        mesh.hdf5 = True

        mesh.Write_Outputs(step=0, time=0.0)

        output_dir = Path(mesh.output_dir)
        with h5py.File(output_dir / "Outputs_0.hdf5", "r") as h5f:
            field_data = h5f["Fields"]["stress_q"]
            assert field_data.attrs["dg_degree"] == 0
            np.testing.assert_allclose(field_data[()], 2.5)

        with h5py.File(output_dir / "mesh_epoch_000.hdf5", "r") as h5f:
            assert "Mesh_DG0" in h5f

    def test_write_outputs_rejects_changed_time_data_schema(self, mesh_with_output_field):
        """Time-data output should fail if the recorded measure names change mid-run."""
        mesh = mesh_with_output_field
        mesh.hdf5 = True
        mesh.params.Global_Measures = {"force": 1.0}

        mesh.Write_Outputs(step=0, time=0.0)

        mesh.params.Global_Measures = {"disp": 0.25}
        with pytest.raises(ValueError, match="data_names mismatch"):
            mesh.Write_Outputs(step=1, time=0.1)

    def test_write_outputs_supports_scalar_field_names_with_non_component_underscores(
        self, mesh_with_output_field
    ):
        """Output field names like ``phasefield_Hist`` should not be treated as components."""
        mesh = mesh_with_output_field
        mesh.Add_Field("phasefield_Hist", "Phasefield History", "CG", 1)
        mesh.fields["phasefield_Hist"].x.array[:] = 4.0
        mesh.output_fields = ["phasefield_Hist"]
        mesh.hdf5 = True

        mesh.Write_Outputs(step=0, time=0.0)

        with h5py.File(Path(mesh.output_dir) / "Outputs_0.hdf5", "r") as h5f:
            assert "phasefield_Hist" in h5f["Fields"]
            np.testing.assert_allclose(h5f["Fields"]["phasefield_Hist"][()], 4.0)

    def test_hdf5_writes_distinct_geometry_and_tags_for_each_mesh_epoch(
        self,
        mesh_with_output_field,
    ):
        """A remesh should create immutable geometry and matching snapshot tags."""
        mesh = mesh_with_output_field
        mesh.hdf5 = True
        mesh.output_freq = 1

        mesh.Write_Outputs(step=1, time=0.1)
        result = mesh.Refine_From_Indicator(
            RefinementConfig(
                enabled=True,
                indicator=FieldThresholdIndicatorConfig(
                    field="temperature",
                    threshold=1.0,
                ),
                target_h=0.4,
                buffer_layers=0,
            )
        )
        assert result.installed
        mesh.Write_Outputs(step=2, time=0.2)

        output_dir = Path(mesh.output_dir)
        epoch_zero_path = output_dir / "mesh_epoch_000.hdf5"
        epoch_one_path = output_dir / "mesh_epoch_001.hdf5"
        with h5py.File(epoch_zero_path, "r") as epoch_zero, h5py.File(
            epoch_one_path,
            "r",
        ) as epoch_one:
            assert int(epoch_zero["epoch_tag"][()]) == 0
            assert int(epoch_one["epoch_tag"][()]) == 1
            assert epoch_one["Mesh_DG1"]["X"].shape[0] > epoch_zero["Mesh_DG1"]["X"].shape[0]

        for step, expected_epoch in ((1, 0), (2, 1)):
            with h5py.File(output_dir / f"Outputs_{step}.hdf5", "r") as h5f:
                assert int(h5f["epoch_tag"][()]) == expected_epoch
                assert "epoch_tag" not in h5f["Fields"]

        with h5py.File(output_dir / "time_data.hdf5", "r") as h5f:
            names = [name.decode("utf-8") for name in h5f["data_names"][()]]
            epoch_column = names.index("epoch_tag")
            np.testing.assert_allclose(h5f["data"][:, epoch_column], [0.0, 1.0])

    def test_time_data_records_epoch_on_steps_skipped_by_output_interval(
        self,
        mesh_with_output_field,
    ):
        """Epoch history should not depend on whether spatial output is due."""
        mesh = mesh_with_output_field
        mesh.hdf5 = True
        mesh.output_freq = 10

        mesh.Write_Outputs(step=1, time=0.1)
        mesh.remesh_epoch = 1
        mesh.Write_Outputs(step=2, time=0.2)

        output_dir = Path(mesh.output_dir)
        assert not (output_dir / "Outputs_1.hdf5").exists()
        assert not (output_dir / "Outputs_2.hdf5").exists()
        with h5py.File(output_dir / "time_data.hdf5", "r") as h5f:
            names = [name.decode("utf-8") for name in h5f["data_names"][()]]
            epoch_column = names.index("epoch_tag")
            np.testing.assert_allclose(h5f["data"][:, epoch_column], [0.0, 1.0])

    def test_hdf5_rejects_existing_mesh_epoch_with_wrong_tag(
        self,
        mesh_with_output_field,
    ):
        """An epoch filename must never be silently reused for another epoch."""
        mesh = mesh_with_output_field
        mesh.hdf5 = True
        mesh._ensure_output_directory()
        mesh_path = Path(mesh.output_dir) / "mesh_epoch_000.hdf5"
        with h5py.File(mesh_path, "w") as h5f:
            h5f.create_dataset("epoch_tag", data=np.int64(9))

        with pytest.raises(ValueError, match="epoch_tag 9, expected 0"):
            mesh._write_hdf5(step=1, time=0.1)

    def test_hdf5_rejects_existing_mesh_epoch_with_wrong_geometry_shape(
        self,
        mesh_with_output_field,
    ):
        """Existing coordinate tables must match the current epoch layout."""
        mesh = mesh_with_output_field
        mesh.hdf5 = True
        mesh._ensure_output_directory()
        mesh_path = Path(mesh.output_dir) / "mesh_epoch_000.hdf5"
        with h5py.File(mesh_path, "w") as h5f:
            h5f.create_dataset("epoch_tag", data=np.int64(0))
            group = h5f.create_group("Mesh_DG1")
            for coordinate in ("X", "Y", "Z"):
                group.create_dataset(coordinate, shape=(1, 1), dtype=float)

        with pytest.raises(ValueError, match="has shape.*expected"):
            mesh._write_hdf5(step=1, time=0.1)

    def test_write_outputs_reuses_single_vtk_file(self, mesh_with_output_field, monkeypatch):
        """Single-file VTK output should reuse one persistent writer across steps."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        _FakeVTKFile.instances = []
        monkeypatch.setattr(mesh_outputs, "VTKFile", _FakeVTKFile)
        mesh = mesh_with_output_field
        mesh.vtk = True
        mesh.output_freq = 1

        mesh.Write_Outputs(step=0, time=0.0)
        mesh.Write_Outputs(step=1, time=0.1)

        assert len(_FakeVTKFile.instances) == 1
        vtk_file = _FakeVTKFile.instances[0]
        assert vtk_file.path.endswith("Outputs_mesh_epoch_0000.pvd")
        assert vtk_file.function_writes == [
            (["temperature"], 0.0),
            (["temperature"], 0.1),
        ]

        mesh.close_outputs()

        assert vtk_file.close_calls == 1
        assert mesh._vtk_file is None

    def test_single_vtk_writer_closes_and_reopens_with_epoch_specific_name(
        self,
        mesh_with_output_field,
        monkeypatch,
    ):
        """A persistent VTK writer must never span changing topology."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        _FakeVTKFile.instances = []
        monkeypatch.setattr(mesh_outputs, "VTKFile", _FakeVTKFile)
        mesh = mesh_with_output_field
        mesh.vtk_single_file = True

        mesh._write_vtk(step=1, time=0.1)
        mesh.remesh_epoch = 1
        mesh._write_vtk(step=2, time=0.2)

        assert [Path(writer.path).name for writer in _FakeVTKFile.instances] == [
            "Outputs_mesh_epoch_0000.pvd",
            "Outputs_mesh_epoch_0001.pvd",
        ]
        assert _FakeVTKFile.instances[0].close_calls == 1
        assert _FakeVTKFile.instances[1].close_calls == 0
        assert mesh._vtk_epoch == 1

    def test_write_outputs_uses_step_specific_vtk_files_when_requested(
        self, mesh_with_output_field, monkeypatch
    ):
        """Per-step VTK output should create a fresh writer for each snapshot."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        _FakeVTKFile.instances = []
        monkeypatch.setattr(mesh_outputs, "VTKFile", _FakeVTKFile)
        mesh = mesh_with_output_field
        mesh.vtk = True
        mesh.vtk_single_file = False
        mesh.output_fields = []

        mesh.Write_Outputs(step=3, time=0.5)

        assert len(_FakeVTKFile.instances) == 1
        vtk_file = _FakeVTKFile.instances[0]
        assert vtk_file.path.endswith("Outputs_mesh_epoch_0000_step_3.pvd")
        assert vtk_file.function_writes == []
        assert vtk_file.mesh_writes[0][0] is mesh.mesh
        assert vtk_file.mesh_writes[0][1] == 0.5
        assert vtk_file.close_calls == 1

    def test_write_outputs_returns_without_output_directory(self, mesh_with_output_field, monkeypatch):
        """Write_Outputs should do nothing when no output directory is configured."""
        mesh = mesh_with_output_field
        mesh.output_dir = None

        called = {"time": False, "vtk": False, "hdf5": False}
        monkeypatch.setattr(mesh, "_append_time_data", lambda step, time: called.__setitem__("time", True))
        monkeypatch.setattr(mesh, "_write_vtk", lambda step, time: called.__setitem__("vtk", True))
        monkeypatch.setattr(mesh, "_write_hdf5", lambda step, time: called.__setitem__("hdf5", True))

        mesh.Write_Outputs(step=0, time=0.0)

        assert called == {"time": False, "vtk": False, "hdf5": False}


class TestMeshOutputsWhiteBoxEdgeCases:
    """Focused coverage for private output-helper branches."""

    def test_output_directory_marker_helpers_handle_missing_output_dir(self, mesh_with_output_field):
        """Marker helpers should no-op cleanly when no output directory is configured."""
        mesh = mesh_with_output_field
        mesh.output_dir = None

        assert mesh._output_dir_marker_path() is None
        mesh._ensure_output_directory()
        mesh._write_output_dir_marker()

    def test_require_output_dir_marker_for_cleanup_returns_for_missing_directory(
        self, mesh_with_output_field, tmp_path
    ):
        """Cleanup authorization should no-op when the target directory does not exist."""
        mesh = mesh_with_output_field
        mesh.output_dir = str(tmp_path / "does_not_exist")

        mesh._require_output_dir_marker_for_cleanup()

    def test_require_output_dir_marker_for_cleanup_rejects_non_directory_path(
        self, mesh_with_output_field, tmp_path
    ):
        """Cleanup authorization should reject file paths masquerading as output directories."""
        mesh = mesh_with_output_field
        output_file = tmp_path / "not_a_directory"
        output_file.write_text("data", encoding="utf-8")
        mesh.output_dir = str(output_file)

        with pytest.raises(RuntimeError, match="is not a directory"):
            mesh._require_output_dir_marker_for_cleanup()

    def test_close_outputs_returns_when_no_vtk_file_is_open(self, mesh_with_output_field):
        """Closing outputs should no-op when no persistent VTK writer exists."""
        mesh_with_output_field._vtk_file = None

        mesh_with_output_field.close_outputs()

        assert mesh_with_output_field._vtk_file is None

    def test_parallel_hdf5_kwargs_returns_mpio_config_for_distributed_runs(
        self, mesh_with_output_field, monkeypatch
    ):
        """Distributed HDF5 output should request the MPI driver configuration."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        sentinel_comm = object()
        monkeypatch.setattr(mesh_outputs, "size", 2)
        monkeypatch.setattr(mesh_outputs, "comm", sentinel_comm)

        assert mesh_with_output_field._parallel_hdf5_kwargs() == {
            "driver": "mpio",
            "comm": sentinel_comm,
        }

    def test_append_time_data_returns_on_non_root_rank(self, mesh_with_output_field, monkeypatch):
        """Non-root ranks should not write the shared time-data file."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        monkeypatch.setattr(mesh_outputs, "rank", 1)

        mesh_with_output_field._append_time_data(step=0, time=0.0)

        assert not (Path(mesh_with_output_field.output_dir) / "time_data.hdf5").exists()

    def test_write_hdf5_requires_mpi_enabled_h5py_for_distributed_runs(
        self, mesh_with_output_field, monkeypatch
    ):
        """Distributed HDF5 output should fail fast without MPI-enabled h5py."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        monkeypatch.setattr(mesh_outputs, "size", 2)
        monkeypatch.setattr(h5py, "get_config", lambda: SimpleNamespace(mpi=False))

        with pytest.raises(RuntimeError, match="MPI support is required"):
            mesh_with_output_field._write_hdf5(step=0, time=0.0)

    def test_write_hdf5_handles_missing_local_field_data_in_distributed_mode(
        self, mesh_with_output_field, monkeypatch
    ):
        """Distributed HDF5 output should skip globally absent fields and zero-fill remote-only ones."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        mesh = mesh_with_output_field
        mesh.output_fields = ["ghost_global", "ghost_remote"]

        monkeypatch.setattr(mesh_outputs, "size", 2)
        monkeypatch.setattr(mesh_outputs, "comm", _SequencedComm([0, 1, None]))
        monkeypatch.setattr(h5py, "get_config", lambda: SimpleNamespace(mpi=True))
        # The faked size > 1 selects collective writes, which HDF5 only
        # accepts on files opened with the MPI driver.
        monkeypatch.setattr(mesh, "_parallel_hdf5_kwargs", lambda: {"driver": "mpio", "comm": MPI.COMM_SELF})
        monkeypatch.setattr(mesh, "Prepare_Field_For_Output", lambda field_spec: None)

        mesh._write_hdf5(step=4, time=0.4)

        with h5py.File(Path(mesh.output_dir) / "Outputs_4.hdf5", "r") as h5f:
            fields = h5f["Fields"]
            assert "ghost_global" not in fields
            assert "ghost_remote" in fields
            np.testing.assert_allclose(fields["ghost_remote"][()], 0.0)
            assert fields["ghost_remote"].attrs["dg_degree"] == 1

    def test_write_hdf5_mesh_geometry_validates_existing_group_without_rewriting(
        self, mesh_with_output_field, tmp_path, monkeypatch
    ):
        """A complete compatible epoch group should remain immutable when reused."""
        mesh = mesh_with_output_field
        V_dg = mesh._get_dg_space(1)
        dofmap = V_dg.dofmap
        cell_map = mesh.mesh.topology.index_map(mesh.mesh.topology.dim)
        cell_start, cell_end = cell_map.local_range
        num_cells_local = cell_end - cell_start
        nodes_per_cell = len(dofmap.cell_dofs(0)) if num_cells_local > 0 else 0

        def _unexpected_build_cell_table(*args, **kwargs):
            raise AssertionError("_build_cell_table should not be called when write_mesh=False")

        monkeypatch.setattr(mesh, "_build_cell_table", _unexpected_build_cell_table)

        with h5py.File(tmp_path / "mesh_reuse.hdf5", "w") as h5f:
            mesh_group = h5f.create_group("Mesh_DG1")
            expected_shape = (cell_map.size_global, nodes_per_cell)
            mesh_group.create_dataset("X", shape=expected_shape, dtype=float)
            mesh_group.create_dataset("Y", shape=expected_shape, dtype=float)
            if mesh.mesh.geometry.x.shape[1] == 3:
                mesh_group.create_dataset("Z", shape=expected_shape, dtype=float)
            mesh._write_hdf5_mesh_geometry(
                h5f,
                V_dg,
                dg_degree=1,
                gdim=mesh.mesh.geometry.x.shape[1],
                num_cells_global=cell_map.size_global,
                cell_start=cell_start,
                cell_end=cell_end,
                num_cells_local=num_cells_local,
                nodes_per_cell=nodes_per_cell,
                dofmap=dofmap,
            )

            assert "Mesh_DG1" in h5f
            assert h5f["Mesh_DG1"]["X"].shape == (cell_map.size_global, nodes_per_cell)
            assert h5f["Mesh_DG1"]["Y"].shape == (cell_map.size_global, nodes_per_cell)

    def test_write_vtk_single_file_writes_mesh_when_no_field_functions_exist(
        self, mesh_with_output_field, monkeypatch
    ):
        """Single-file VTK output should fall back to mesh-only output when fields are unavailable."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        _FakeVTKFile.instances = []
        monkeypatch.setattr(mesh_outputs, "VTKFile", _FakeVTKFile)
        mesh = mesh_with_output_field
        mesh.output_fields = ["ghost"]
        monkeypatch.setattr(mesh, "Prepare_Field_For_Output", lambda field_spec: None)
        mesh._write_vtk(step=0, time=0.0)

        assert len(_FakeVTKFile.instances) == 1
        vtk_file = _FakeVTKFile.instances[0]
        assert vtk_file.function_writes == []
        assert vtk_file.mesh_writes == [(mesh.mesh, 0.0)]

    def test_write_vtk_step_file_writes_functions_when_available(
        self, mesh_with_output_field, monkeypatch
    ):
        """Per-step VTK output should write DG field functions when they are available."""
        import femcubes.core.mesh._mesh_outputs as mesh_outputs

        _FakeVTKFile.instances = []
        monkeypatch.setattr(mesh_outputs, "VTKFile", _FakeVTKFile)
        mesh = mesh_with_output_field
        mesh.vtk_single_file = False
        mesh.output_fields = ["temperature"]

        mesh._write_vtk(step=5, time=0.7)

        assert len(_FakeVTKFile.instances) == 1
        vtk_file = _FakeVTKFile.instances[0]
        assert vtk_file.path.endswith("Outputs_mesh_epoch_0000_step_5.pvd")
        assert vtk_file.function_writes == [(["temperature"], 0.7)]
