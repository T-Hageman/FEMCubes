"""Smoke tests of the example scripts.

Each example case is run with ``--quick`` (a short, coarse run) on two MPI
ranks in a temporary folder. The test checks that the script finishes and
saves its figures; it does not check the results, which the quick settings do
not reproduce.

Run with::

    pytest examples/test_examples.py -n 8
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent
RANKS = 2

# (script, extra arguments); every case of every example is run.
RUNS = [
    ("fracture/cohesive_phasefield.py", ["--case", "full_plate"]),
    ("fracture/cohesive_phasefield.py", ["--case", "single_edge_notched"]),
    ("fracture/cohesive_phasefield.py", ["--case", "dynamic"]),
    ("fracture/viscoelastic_fracture.py", []),
    ("fracture/multidomain_tensile_bar.py", []),
    ("electrochemistry/corrosion_pit.py", []),
    ("electrochemistry/electrolyte_metal.py", []),
    ("electrochemistry/electrolyte_metal.py", ["--no-interface"]),
    *[("glaciology/ice_cliffs.py", ["--case", case]) for case in
      ("le_cohesive", "ve_cohesive", "le_split", "ve_split", "stokes", "glen")],
    *[("glaciology/ice_shelf_front.py", ["--case", case]) for case in ("profile", "uniform")],
    ("glaciology/shallow_ice.py", ["--case", "slope"]),
    ("glaciology/shallow_ice.py", ["--case", "flat"]),
    *[("glaciology/tensile_ice.py", ["--case", case]) for case in
      ("stokes", "glen", "le", "ve", "le_cohesive", "ve_cohesive")],
]


def _mpirun():
    """PRIVATE: the MPI launcher of the running Python environment, or the one on PATH."""
    local = Path(sys.executable).with_name("mpirun")
    launcher = str(local) if local.exists() else shutil.which("mpirun")
    if launcher is None:
        pytest.skip("mpirun not found")
    return launcher


@pytest.mark.examples
@pytest.mark.slow
@pytest.mark.mpi
@pytest.mark.parametrize("script, arguments", RUNS, ids=[" ".join([s, *a]) for s, a in RUNS])
def test_example_runs_and_saves_its_figures(script, arguments, tmp_path):
    output_dir = tmp_path / "output"
    environment = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}
    command = [_mpirun(), "-n", str(RANKS), sys.executable, str(EXAMPLES / script), "--quick",
               *arguments, "--output-dir", str(output_dir)]

    # Run from the temporary folder, so the examples cannot depend on the working directory.
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=1800)

    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    figures = sorted(path.name for path in output_dir.glob("*.png"))
    assert "mesh.png" in figures and len(figures) >= 3, figures
