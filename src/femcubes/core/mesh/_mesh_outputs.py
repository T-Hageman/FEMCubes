"""File-output helpers for the composite mesh class."""

from __future__ import annotations

import contextlib
import os
from typing import Any

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from dolfinx.io import VTKFile

from femcubes.core.utils.mpi_utils import comm, mprint, rank, size


class _MeshOutputs:
    """Mix-in for VTK and HDF5 file output.

    Handles directory creation, epoch-segmented VTK time-series writing, and
    parallel HDF5 snapshots with immutable per-epoch geometry. Field data are
    prepared through :meth:`Prepare_Field_For_Output`, so plotting and file
    output share the same field-spec parsing and DG projection rules.

    Public methods are part of the mesh wrapper API. Underscore-prefixed
    helpers are internal implementation details.
    """

    _OUTPUT_DIR_MARKER = ".femcubes-output-dir"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, params: Any) -> None:
        self.output_fields: list[str] = list(getattr(params, "output_fields", []))
        self.vtk: bool = getattr(params, "write_vtk", False)
        self.vtk_single_file: bool = getattr(params, "vtk_single_file", True)
        self.hdf5: bool = getattr(params, "write_hdf5", False)
        self.output_dir: str | None = getattr(params, "output_dir", None)
        self.output_freq: int = getattr(params, "output_interval", 10)
        self.save_Mesh_every_step: bool = getattr(params, "save_Mesh_every_step", False)
        self._vtk_file = None
        self._vtk_epoch: int | None = None
        self._interface_vtk_files: dict = {}

        self._prepare_output_directory()

        super().__init__(params)

    def Write_Outputs(self, step: int, time: float) -> None:
        """Write configured output files for one time step.

        Parameters
        ----------
        step : int
            Integer time-step counter.
        time : float
            Physical time associated with the current step.

        Notes
        -----
        ``time_data.hdf5`` is updated on rank 0 whenever an output
        directory is configured, while field snapshots are additionally
        gated by :attr:`output_freq` and the enabled backends. In distributed
        runs, every rank must call this method so time-output errors and
        parallel file operations remain synchronized. Every history row and
        HDF5 field snapshot records the current ``remesh_epoch``.
        """
        if self.output_dir is None:
            return

        mprint(f"Writing output files for step {step}, time {time:.2f} s")

        time_data_error = None
        if rank == 0:
            try:
                self._append_time_data(step, time)
            except Exception as exc:
                time_data_error = (type(exc).__name__, str(exc))

        time_data_error = comm.bcast(time_data_error, root=0)
        if time_data_error is not None:
            error_type, error_message = time_data_error
            if error_type == "ValueError":
                raise ValueError(error_message)
            raise RuntimeError(f"Could not append time-history output ({error_type}): {error_message}")

        if not self._should_write_output_step(step):
            return

        if self.vtk:
            self._write_vtk(step, time)

        if self.hdf5:
            self._write_hdf5(step, time)

    def close_outputs(self) -> None:
        """Close any persistent output handles kept open across time steps.

        At present this mainly affects the current epoch's persistent VTK
        writers used when ``vtk_single_file`` is enabled (one for the mesh
        and one per interface with output fields).
        """
        for vtk_file in self._interface_vtk_files.values():
            vtk_file.close()
        self._interface_vtk_files = {}
        if self._vtk_file is None:
            self._vtk_epoch = None
            return

        self._vtk_file.close()
        mprint("Closed VTK time series file")
        self._vtk_file = None
        self._vtk_epoch = None

    # ------------------------------------------------------------------
    # Internal Helpers: Setup and Shared Utilities
    # ------------------------------------------------------------------

    def _outputs_enabled(self) -> bool:
        """PRIVATE: Return whether any file-output backend is enabled."""
        return self.vtk or self.hdf5

    def _prepare_output_directory(self) -> None:
        """PRIVATE: Reset the output directory before a new run starts.

        With VTK or HDF5 field output the whole directory is reset. Without
        it, only ``time_data.hdf5`` is written; a previous run's file is then
        removed, so the new run does not append to it, and the rest of the
        directory is left alone. Either way, only directories marked as
        FEMCubes output are changed.
        """
        if self.output_dir is None:
            return

        if rank == 0:
            if self._outputs_enabled():
                import shutil

                self._require_output_dir_marker_for_cleanup()
                if os.path.exists(self.output_dir):
                    shutil.rmtree(self.output_dir)
                os.makedirs(self.output_dir, exist_ok=True)
                self._write_output_dir_marker()
            elif os.path.exists(self._output_dir_marker_path()):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(os.path.join(self.output_dir, "time_data.hdf5"))

        comm.Barrier()

    def _ensure_output_directory(self) -> None:
        """PRIVATE: Collectively create the configured output directory."""
        if self.output_dir is None:
            return

        if rank == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            self._write_output_dir_marker()

        comm.Barrier()

    def _output_dir_marker_path(self) -> str | None:
        """PRIVATE: Return the marker file path for the configured output directory."""
        if self.output_dir is None:
            return None
        return os.path.join(self.output_dir, self._OUTPUT_DIR_MARKER)

    def _require_output_dir_marker_for_cleanup(self) -> None:
        """PRIVATE: Refuse destructive cleanup unless the output directory is marked."""
        if self.output_dir is None or not os.path.exists(self.output_dir):
            return
        if not os.path.isdir(self.output_dir):
            raise RuntimeError(f"Output path '{self.output_dir}' exists but is not a directory.")

        marker_path = self._output_dir_marker_path()
        if marker_path is None or not os.path.exists(marker_path):
            raise RuntimeError(
                f"Refusing to delete unmarked output directory '{self.output_dir}'. "
                f"Create '{self._OUTPUT_DIR_MARKER}' in that folder to allow cleanup."
            )

    def _write_output_dir_marker(self) -> None:
        """PRIVATE: Write the marker file that authorizes future cleanup."""
        marker_path = self._output_dir_marker_path()
        if marker_path is None:
            return

        with open(marker_path, "w", encoding="utf-8") as marker_file:
            marker_file.write("FEMCubes output directory\n")

    def _should_write_output_step(self, step: int) -> bool:
        """PRIVATE: Return whether field files should be written this step."""
        return step % self.output_freq == 0

    def _build_cell_table(self, values, dofmap, num_cells_local: int, nodes_per_cell: int):
        """PRIVATE: Reshape flat DG dof values into a per-cell table."""
        table = np.zeros((num_cells_local, nodes_per_cell), dtype=float)
        for cell in range(num_cells_local):
            dofs = dofmap.cell_dofs(cell)
            table[cell, :] = values[dofs]
        return table

    def _parallel_hdf5_kwargs(self) -> dict[str, Any]:
        """PRIVATE: Return the h5py keyword arguments needed for MPI output."""
        if size == 1:
            return {}
        return {"driver": "mpio", "comm": comm}

    def _write_hdf5_slice(self, dataset, selection, values) -> None:
        """PRIVATE: Write one rank's dataset slice, collectively under MPI."""
        if size > 1:
            if values.size:
                with dataset.collective:
                    dataset[selection] = values
                return

            # h5py skips H5Dwrite for empty high-level selections. MPI
            # collective I/O still requires zero-cell ranks to participate.
            import h5py

            file_space = dataset.id.get_space()
            file_space.select_none()
            memory_space = h5py.h5s.create_simple(values.shape)
            memory_space.select_none()
            transfer_properties = h5py.h5p.create(h5py.h5p.DATASET_XFER)
            transfer_properties.set_dxpl_mpio(h5py.h5fd.MPIO_COLLECTIVE)
            dataset.id.write(
                memory_space,
                file_space,
                values,
                dxpl=transfer_properties,
            )
            return
        dataset[selection] = values

    def _collect_vtk_functions(self):
        """PRIVATE: Convert configured output fields into DG functions for VTK."""
        functions = []
        for field_spec in self.output_fields:
            if self._interface_support_key(self._parse_output_field_spec(field_spec)[0]) is not None:
                continue  # written to the interface's own series
            prepared = self.Prepare_Field_For_Output(field_spec)
            if prepared is None:
                continue

            field_func = fem.Function(prepared.function_space, name=field_spec)
            field_func.x.array[:] = prepared.values
            functions.append(field_func)
        return functions

    def _collect_interface_vtk_functions(self) -> dict:
        """PRIVATE: Return DG functions of interface output fields, grouped by interface name."""
        groups = {}
        for field_spec in self.output_fields:
            field_name = self._parse_output_field_spec(field_spec)[0]
            support_key = self._interface_support_key(field_name)
            if support_key is None:
                continue
            V_dg = self._get_dg_space(self._get_field_dg_degree(field_name), support_key)
            prepared = self.Prepare_Field_For_Output(field_spec, V_dg=V_dg)
            if prepared is None:
                continue
            field_func = fem.Function(prepared.function_space, name=field_spec)
            field_func.x.array[:] = prepared.values
            groups.setdefault(self._interface_output_name(field_name), []).append(field_func)
        return groups

    def _interface_output_name(self, field_name: str) -> str:
        """PRIVATE: Return the file-safe name of an interface field's facet groups, e.g. ``metal_pit+metal_outer``."""
        return "+".join(self.Get_Field_Support(field_name).domain_names).replace(":", "_")

    # ------------------------------------------------------------------
    # Internal Helpers: Time and File Output
    # ------------------------------------------------------------------

    def _append_time_data(self, step: int, time: float) -> None:
        """PRIVATE: Append time-history data on rank 0 without MPI collectives."""
        _ = step
        if rank != 0:
            return

        import h5py

        os.makedirs(self.output_dir, exist_ok=True)
        self._write_output_dir_marker()
        time_file = f"{self.output_dir}/time_data.hdf5"

        epoch = self._current_mesh_epoch()
        data_names = ["time", "dt", "epoch_tag"]
        if hasattr(self.params, "Global_Measures"):
            for name in sorted(self.params.Global_Measures.keys()):
                data_names.append(name)

        data_row = [float(time), float(self.params.dt), float(epoch)]
        if hasattr(self.params, "Global_Measures"):
            for name in sorted(self.params.Global_Measures.keys()):
                value = self.params.Global_Measures.get(name, np.nan)
                data_row.append(float(value))

        data_row = np.asarray(data_row, dtype=float)

        with h5py.File(time_file, "a") as h5f:
            if "data_names" not in h5f:
                names = np.asarray(data_names, dtype="S")
                h5f.create_dataset("data_names", data=names)
                data = data_row.reshape(1, -1)
                h5f.create_dataset("data", data=data, maxshape=(None, data.shape[1]))
                return

            existing_names = [name.decode("utf-8") for name in h5f["data_names"][()]]
            if existing_names != data_names:
                raise ValueError(
                    "time_data.hdf5 data_names mismatch with current Global_Measures"
                )

            dataset = h5f["data"]
            dataset.resize((dataset.shape[0] + 1, dataset.shape[1]))
            dataset[-1, :] = data_row

    def _write_hdf5(self, step: int, time: float) -> None:
        """PRIVATE: Write one epoch-tagged field snapshot and mesh geometry."""
        _ = time
        import h5py

        self._ensure_output_directory()
        epoch = self._current_mesh_epoch()
        mesh_path = f"{self.output_dir}/mesh_epoch_{epoch:03d}.hdf5"
        outputs_path = f"{self.output_dir}/Outputs_{step}.hdf5"

        if size > 1 and not getattr(h5py.get_config(), "mpi", False):
            raise RuntimeError("h5py MPI support is required for parallel HDF5 output.")

        gdim = self.mesh.geometry.x.shape[1]
        tdim = self.mesh.topology.dim
        cell_map = self.mesh.topology.index_map(tdim)
        cell_start, cell_end = cell_map.local_range
        num_cells_local = cell_end - cell_start

        written_mesh_degrees = set()

        with (
            h5py.File(mesh_path, "a", **self._parallel_hdf5_kwargs()) as h5f_mesh,
            h5py.File(outputs_path, "a", **self._parallel_hdf5_kwargs()) as h5f_out,
        ):
            self._require_hdf5_epoch_tag(
                h5f_mesh,
                epoch,
                f"mesh epoch file '{mesh_path}'",
            )
            self._require_hdf5_epoch_tag(
                h5f_out,
                epoch,
                f"parameter output '{outputs_path}'",
            )
            fields_group = h5f_out.require_group("Fields")

            for field_spec in self.output_fields:
                interface_key = self._interface_support_key(self._parse_output_field_spec(field_spec)[0])
                if interface_key is not None:
                    self._write_hdf5_interface_field(h5f_mesh, h5f_out, field_spec, interface_key, gdim, mesh_path, outputs_path)
                    continue

                prepared = self.Prepare_Field_For_Output(field_spec)
                local_has = 1 if prepared is not None else 0
                global_has = comm.allreduce(local_has, op=MPI.SUM)
                if global_has == 0:
                    continue

                if prepared is None:
                    V_dg = self._get_dg_space(1)
                else:
                    field_values = prepared.values
                    V_dg = prepared.function_space

                dofmap = V_dg.dofmap
                nodes_per_cell = len(dofmap.cell_dofs(0)) if num_cells_local > 0 else 0
                if size > 1:
                    nodes_per_cell = comm.allreduce(nodes_per_cell, op=MPI.MAX)

                dg_degree = V_dg.element.basix_element.degree

                if dg_degree not in written_mesh_degrees:
                    written_mesh_degrees.add(dg_degree)
                    self._write_hdf5_mesh_geometry(
                        h5f_mesh,
                        V_dg,
                        dg_degree,
                        gdim,
                        cell_map.size_global,
                        cell_start,
                        cell_end,
                        num_cells_local,
                        nodes_per_cell,
                        dofmap,
                        mesh_path,
                    )

                if prepared is None:
                    dof_count = V_dg.dofmap.index_map.size_local
                    field_values = np.zeros(dof_count, dtype=float)

                cell_table = (
                    self._build_cell_table(field_values, dofmap, num_cells_local, nodes_per_cell)
                    if num_cells_local > 0
                    else np.zeros((0, nodes_per_cell))
                )

                dataset = self._require_hdf5_dataset(
                    fields_group,
                    field_spec,
                    shape=(cell_map.size_global, nodes_per_cell),
                    dtype=float,
                    description=f"field '{field_spec}' in '{outputs_path}'",
                )
                dataset.attrs["dg_degree"] = dg_degree
                self._write_hdf5_slice(
                    dataset,
                    (slice(cell_start, cell_end), slice(None)),
                    cell_table,
                )

    def _write_hdf5_interface_field(self, h5f_mesh, h5f_out, field_spec: str, support_key, gdim: int, mesh_path: str, outputs_path: str) -> None:
        """PRIVATE: Write one interface field on its own facet mesh.

        Interface fields have no parent cells, so their values go to
        ``InterfaceFields/<field_spec>`` and their DG node coordinates to the
        mesh group ``Interface_<facet groups>_DG<k>`` (named by the dataset's
        ``mesh_group`` attribute), leaving the ``Fields`` layout untouched.
        """

        field_name = self._parse_output_field_spec(field_spec)[0]
        V_dg = self._get_dg_space(self._get_field_dg_degree(field_name), support_key)
        prepared = self.Prepare_Field_For_Output(field_spec, V_dg=V_dg)
        if comm.allreduce(1 if prepared is not None else 0, op=MPI.SUM) == 0:
            return
        V_dg = V_dg if prepared is None else prepared.function_space

        cell_map = V_dg.mesh.topology.index_map(V_dg.mesh.topology.dim)
        cell_start, cell_end = cell_map.local_range
        num_cells_local = cell_end - cell_start
        dofmap = V_dg.dofmap
        nodes_per_cell = comm.allreduce(dofmap.list.shape[1] if num_cells_local > 0 else 0, op=MPI.MAX)
        dg_degree = V_dg.element.basix_element.degree
        group_name = f"Interface_{self._interface_output_name(field_name)}_DG{dg_degree}"

        self._write_hdf5_mesh_geometry(
            h5f_mesh, V_dg, dg_degree, gdim, cell_map.size_global, cell_start, cell_end,
            num_cells_local, nodes_per_cell, dofmap, mesh_path, group_name=group_name,
        )

        field_values = prepared.values if prepared is not None else np.zeros(dofmap.index_map.size_local, dtype=float)
        cell_table = (
            self._build_cell_table(field_values, dofmap, num_cells_local, nodes_per_cell)
            if num_cells_local > 0
            else np.zeros((0, nodes_per_cell))
        )
        dataset = self._require_hdf5_dataset(
            h5f_out.require_group("InterfaceFields"),
            field_spec,
            shape=(cell_map.size_global, nodes_per_cell),
            dtype=float,
            description=f"interface field '{field_spec}' in '{outputs_path}'",
        )
        dataset.attrs["dg_degree"] = dg_degree
        dataset.attrs["mesh_group"] = group_name
        self._write_hdf5_slice(dataset, (slice(cell_start, cell_end), slice(None)), cell_table)

    def _write_hdf5_mesh_geometry(
        self,
        h5f_mesh,
        V_dg,
        dg_degree: int,
        gdim: int,
        num_cells_global: int,
        cell_start: int,
        cell_end: int,
        num_cells_local: int,
        nodes_per_cell: int,
        dofmap,
        mesh_path: str | None = None,
        group_name: str | None = None,
    ) -> None:
        """PRIVATE: Create or validate immutable geometry for one DG degree.

        ``group_name`` defaults to ``Mesh_DG<dg_degree>`` (the full mesh).
        """
        group_name = group_name or f"Mesh_DG{dg_degree}"
        expected_coordinates = ("X", "Y", "Z") if gdim == 3 else ("X", "Y")
        expected_shape = (num_cells_global, nodes_per_cell)
        description = (
            f"mesh group '{group_name}'"
            if mesh_path is None
            else f"mesh group '{group_name}' in '{mesh_path}'"
        )

        if group_name in h5f_mesh:
            mesh_group = h5f_mesh[group_name]
            coordinate_names = tuple(
                name for name in ("X", "Y", "Z") if name in mesh_group
            )
            if coordinate_names != expected_coordinates:
                raise ValueError(
                    f"Existing {description} has coordinate datasets "
                    f"{coordinate_names}, expected {expected_coordinates}."
                )
            for coordinate in expected_coordinates:
                dataset = mesh_group[coordinate]
                if dataset.shape != expected_shape:
                    raise ValueError(
                        f"Existing {description}/{coordinate} has shape "
                        f"{dataset.shape}, expected {expected_shape}."
                    )
                if dataset.dtype.kind != "f":
                    raise ValueError(
                        f"Existing {description}/{coordinate} must use a floating dtype."
                    )
            return

        mesh_group = h5f_mesh.create_group(group_name)
        for coordinate in expected_coordinates:
            mesh_group.create_dataset(coordinate, shape=expected_shape, dtype=float)

        dof_coords = V_dg.tabulate_dof_coordinates().reshape(-1, gdim)
        cx = (
            self._build_cell_table(dof_coords[:, 0], dofmap, num_cells_local, nodes_per_cell)
            if num_cells_local > 0
            else np.zeros((0, nodes_per_cell))
        )
        cy = (
            self._build_cell_table(dof_coords[:, 1], dofmap, num_cells_local, nodes_per_cell)
            if num_cells_local > 0
            else np.zeros((0, nodes_per_cell))
        )
        self._write_hdf5_slice(
            mesh_group["X"],
            (slice(cell_start, cell_end), slice(None)),
            cx,
        )
        self._write_hdf5_slice(
            mesh_group["Y"],
            (slice(cell_start, cell_end), slice(None)),
            cy,
        )

        if gdim == 3:
            cz = (
                self._build_cell_table(dof_coords[:, 2], dofmap, num_cells_local, nodes_per_cell)
                if num_cells_local > 0
                else np.zeros((0, nodes_per_cell))
            )
            self._write_hdf5_slice(
                mesh_group["Z"],
                (slice(cell_start, cell_end), slice(None)),
                cz,
            )

    def _current_mesh_epoch(self) -> int:
        """PRIVATE: Return the validated epoch used by every output backend."""

        epoch = getattr(self, "remesh_epoch", 0)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("remesh_epoch must be a nonnegative integer.")
        return epoch

    def _require_hdf5_epoch_tag(self, h5f, epoch: int, description: str) -> None:
        """PRIVATE: Create or validate one top-level scalar epoch tag."""

        if "epoch_tag" not in h5f:
            h5f.create_dataset(
                "epoch_tag",
                shape=(),
                dtype=np.int64,
                fillvalue=np.int64(epoch),
            )
            return

        dataset = h5f["epoch_tag"]
        if dataset.shape != () or dataset.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"Existing {description} epoch_tag must be a scalar integer dataset."
            )
        existing_epoch = int(dataset[()])
        if existing_epoch != epoch:
            raise ValueError(
                f"Existing {description} has epoch_tag {existing_epoch}, expected {epoch}."
            )

    def _require_hdf5_dataset(
        self,
        group,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype,
        description: str,
    ):
        """PRIVATE: Create a dataset or reject an incompatible existing shape."""

        if name in group:
            dataset = group[name]
            if dataset.shape != shape:
                raise ValueError(
                    f"Existing {description} has shape {dataset.shape}, expected {shape}."
                )
            if dataset.dtype != np.dtype(dtype):
                raise ValueError(
                    f"Existing {description} has dtype {dataset.dtype}, expected "
                    f"{np.dtype(dtype)}."
                )
            return dataset
        return group.create_dataset(name, shape=shape, dtype=dtype)

    def _write_vtk(self, step: int, time: float) -> None:
        """PRIVATE: Write DG field output to an epoch-specific VTK series."""
        functions = self._collect_vtk_functions()
        epoch = self._current_mesh_epoch()

        if self.vtk_single_file:
            if self._vtk_file is not None and self._vtk_epoch != epoch:
                self.close_outputs()
            if self._vtk_file is None:
                vtk_path = f"{self.output_dir}/Outputs_mesh_epoch_{epoch:04d}.pvd"
                self._vtk_file = VTKFile(self.mesh.comm, vtk_path, "w")
                self._vtk_epoch = epoch

            if functions:
                self._vtk_file.write_function(functions, time)
            else:
                self._vtk_file.write_mesh(self.mesh, time)
            for interface, interface_functions in self._collect_interface_vtk_functions().items():
                if interface not in self._interface_vtk_files:
                    path = f"{self.output_dir}/Outputs_{interface}_mesh_epoch_{epoch:04d}.pvd"
                    self._interface_vtk_files[interface] = VTKFile(self.mesh.comm, path, "w")
                self._interface_vtk_files[interface].write_function(interface_functions, time)
            return

        vtk_path = (
            f"{self.output_dir}/Outputs_mesh_epoch_{epoch:04d}_step_{step}.pvd"
        )
        with VTKFile(self.mesh.comm, vtk_path, "w") as vtk:
            if functions:
                vtk.write_function(functions, time)
            else:
                vtk.write_mesh(self.mesh, time)
        # Interface fields live on their own facet mesh, so each interface has its own series.
        for interface, interface_functions in self._collect_interface_vtk_functions().items():
            path = f"{self.output_dir}/Outputs_{interface}_mesh_epoch_{epoch:04d}_step_{step}.pvd"
            with VTKFile(self.mesh.comm, path, "w") as vtk:
                vtk.write_function(interface_functions, time)
