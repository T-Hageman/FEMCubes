"""MPI tests of the PassivatingSurface model.

The surface terms, the charge balance of the metal potential, ``Net_Current``
and the global measures are integrated over the metal facets each rank owns
and reduced over all ranks. A free-corrosion run on two ranks must reproduce
the serial run.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import femcubes.electrochemistry  # noqa: F401 - ensure model registration
from femcubes import Mesh, Physics, TimeSolver
from femcubes.core.utils.mpi_utils import comm
from femcubes.electrochemistry.tests.test_passivating_surface import REFERENCE, _build_params


def _free_corrosion_case() -> dict:
    """Run the coupled free-corrosion case; return its global measures and the net current at a trial potential."""
    passive = next(s for s in REFERENCE["states"] if s["label"].startswith("passive"))
    settings = {**REFERENCE["settings"], "Metal_Potential": "Charge_Conservation", "Initial_Metal_Potential": -0.3,
                "Initial_Coverages": {s: v for s, v in passive["coverages"].items() if s != "Fe"}}
    params = _build_params(surface=settings, end_time=5.0, dt=1.0)
    params.dx = 1.0e-4                      # several metal facets, shared out between the ranks
    params.rel_tol, params.abs_tol = 1e-13, 1e-11
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    net_current_start = physics.models["Surface"].Net_Current(-0.25)
    TimeSolver(params, physics).run()
    return {"measures": dict(params.Global_Measures), "net_current_start": net_current_start,
            "net_current_end": physics.models["Surface"].Net_Current(-0.25)}


def _run_mpi_subprocess(output: Path) -> dict:
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("mpirun not available")
    project_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(project_root) if not pythonpath else f"{project_root}{os.pathsep}{pythonpath}"
    cmd = [mpirun, "-n", "2", sys.executable, __file__, "--mpi-check", str(output)]
    if os.geteuid() == 0:
        cmd.insert(1, "--allow-run-as-root")
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise AssertionError("MPI PassivatingSurface run failed:\n" + "\n".join([result.stdout, result.stderr]).strip())
    return json.loads(output.read_text())


@pytest.mark.mpi
def test_mpi_free_corrosion_matches_serial(tmp_path):
    """Two ranks: the same metal potential, currents, coverages and net current as one rank."""
    distributed = _run_mpi_subprocess(tmp_path / "mpi.json")
    serial = _free_corrosion_case()

    assert distributed["net_current_start"] == pytest.approx(serial["net_current_start"], rel=1e-10)
    assert distributed["net_current_end"] == pytest.approx(serial["net_current_end"], rel=1e-6, abs=1e-12)
    assert set(distributed["measures"]) == set(serial["measures"])
    for name, value in serial["measures"].items():
        # The anodic current sums net rates of fast, quasi-equilibrated steps (gross rates ~1e8 times the
        # net current here): its round-off is ~1e-6 relative, independent of the partition.
        rel = 1e-4 if name == "Surface_Anodic_Current" else 1e-9
        assert distributed["measures"][name] == pytest.approx(value, rel=rel, abs=1e-14), name


if __name__ == "__main__" and "--mpi-check" in sys.argv:
    result = _free_corrosion_case()
    if comm.rank == 0:
        Path(sys.argv[sys.argv.index("--mpi-check") + 1]).write_text(json.dumps(result))
