"""MPI integration tests for mesh HDF5 output."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest


def _run_mpi_hdf5_output_check(output_dir: Path) -> None:
    from femcubes.core.params_template import ParamsTemplate
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.mesh.refinement import (
        FieldThresholdIndicatorConfig,
        RefinementConfig,
    )
    from femcubes.core.utils.mpi_utils import comm, rank, size

    if size < 2:
        raise RuntimeError("MPI HDF5 output checks require at least 2 ranks.")
    if not h5py.get_config().mpi:
        raise RuntimeError("MPI HDF5 output checks require MPI-enabled h5py.")

    output_dir_string = str(output_dir)

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 2.0
        Ly = 1.0
        dx = 0.5
        dim = 2

        output_fields = ["temperature"]
        write_vtk = False
        write_hdf5 = True
        output_dir = output_dir_string
        output_interval = 1
        save_Mesh_every_step = False
        dt = 0.1
        Global_Measures = {"force": 1.5}

    params = TestParams()
    mesh = Mesh(params)
    temperature = mesh.Add_Field("temperature", "Temperature", "CG", 1)
    temperature.x.array[:] = 2.0 + rank
    temperature.x.scatter_forward()

    empty_rank_path = output_dir / "empty_rank_collective.hdf5"
    with h5py.File(
        empty_rank_path,
        "w",
        driver="mpio",
        comm=comm,
    ) as h5f:
        dataset = h5f.create_dataset("values", shape=(1, 2), dtype=float)
        local_values = np.asarray([[3.0, 4.0]]) if rank == 0 else np.zeros((0, 2))
        local_start = 0 if rank == 0 else 1
        mesh._write_hdf5_slice(
            dataset,
            (slice(local_start, local_start + len(local_values)), slice(None)),
            local_values,
        )
    comm.Barrier()
    if rank == 0:
        with h5py.File(empty_rank_path, "r") as h5f:
            np.testing.assert_array_equal(h5f["values"][()], [[3.0, 4.0]])
    comm.Barrier()

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
    comm.Barrier()

    if rank == 0:
        time_path = output_dir / "time_data.hdf5"
        outputs_path = output_dir / "Outputs_1.hdf5"
        refined_outputs_path = output_dir / "Outputs_2.hdf5"
        mesh_path = output_dir / "mesh_epoch_000.hdf5"
        refined_mesh_path = output_dir / "mesh_epoch_001.hdf5"

        assert time_path.exists()
        assert outputs_path.exists()
        assert refined_outputs_path.exists()
        assert mesh_path.exists()
        assert refined_mesh_path.exists()

        with h5py.File(time_path, "r") as h5f:
            names = [name.decode("utf-8") for name in h5f["data_names"][()]]
            assert names == ["time", "dt", "epoch_tag", "force"]
            np.testing.assert_allclose(
                h5f["data"][()],
                [[0.1, 0.1, 0.0, 1.5], [0.2, 0.1, 1.0, 1.5]],
            )

        with h5py.File(outputs_path, "r") as h5f:
            assert int(h5f["epoch_tag"][()]) == 0
            field = h5f["Fields"]["temperature"]
            assert field.attrs["dg_degree"] == 1
            epoch_zero_field_shape = field.shape
            assert np.all(np.isfinite(field[()]))

        with h5py.File(refined_outputs_path, "r") as h5f:
            assert int(h5f["epoch_tag"][()]) == 1
            field = h5f["Fields"]["temperature"]
            assert np.all(np.isfinite(field[()]))

        with h5py.File(mesh_path, "r") as h5f:
            assert int(h5f["epoch_tag"][()]) == 0
            old_shape = h5f["Mesh_DG1"]["X"].shape
            assert epoch_zero_field_shape == old_shape

        with h5py.File(refined_mesh_path, "r") as h5f:
            assert int(h5f["epoch_tag"][()]) == 1
            assert h5f["Mesh_DG1"]["X"].shape[0] > old_shape[0]

    comm.Barrier()


def _run_mpi_subprocess(output_dir: Path) -> None:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")
    if not h5py.get_config().mpi:
        pytest.skip("MPI-enabled h5py not available")

    project_root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) if not pythonpath else f"{project_root}{os.pathsep}{pythonpath}"

    cmd = [mpirun, "-n", "2", sys.executable, __file__, "--mpi-output-check", str(output_dir)]
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = "\n".join(
            part for part in (stdout, stderr) if part
        ).strip()
        raise AssertionError(
            "MPI HDF5 output did not complete within 30 seconds; "
            f"the ranks are likely deadlocked.\n{output}"
        ) from exc

    if result.returncode != 0:
        output = "\n".join([result.stdout, result.stderr]).strip()
        raise AssertionError(f"MPI HDF5 output check failed:\n{output}")


@pytest.mark.mpi
def test_mpi_hdf5_output_completes_without_collective_deadlock(tmp_path):
    """Rank-zero time output and collective field output must use matching collectives."""
    _run_mpi_subprocess(tmp_path / "parallel_hdf5")


if __name__ == "__main__" and "--mpi-output-check" in sys.argv:
    output_index = sys.argv.index("--mpi-output-check") + 1
    if output_index >= len(sys.argv):
        raise SystemExit("Missing MPI output directory.")
    _run_mpi_hdf5_output_check(Path(sys.argv[output_index]))
