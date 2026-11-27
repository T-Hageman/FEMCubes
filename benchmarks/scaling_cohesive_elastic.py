"""Strong-scaling study for the cohesive-elastic FullPlate example.

The default study runs the ``full_plate`` case of
``examples/fracture/cohesive_phasefield.py`` unchanged with every even MPI rank count
from 56 down to 2.  The complete sweep is run first with plotting enabled and then
repeated without plotting. Field-file and movie output are disabled in both sweeps so
the comparison isolates rendering and plot updates; final plot PNGs are still saved for
the plotting-enabled runs.

Run the complete study from the repository root with::

    python3 benchmarks/scaling_cohesive_elastic.py

For a shorter selection of rank counts, use for example::

    python3 benchmarks/scaling_cohesive_elastic.py \
        --cores 2 4 8 16 28 56

Each MPI run writes its progress directly to the terminal.  The driver
incrementally saves ``scaling_results.csv`` and
``runtime_vs_cpu_cores.png`` under
``benchmarks/results/scaling_cohesive_elastic``.  The plotted
runtime is the sum of the maximum setup, JIT warm-up, and solve wall times
over all ranks.
The companion ``runtime_breakdown_vs_cpu_cores.png`` uses stacked bars for
setup, one-time JIT warm-up, form evaluation, form-cache lookup, actual form
binding/compilation, local element assembly, PETSc assembly communication,
boundary conditions, model K/F additions, assembly bookkeeping, linear
solving, refinement, plotting, file output, and other solver work on the rank
that determines each run's solve time. JIT warm-up is completed and its
profiler data reset before the production solve, so it is not misreported as
form compilation.
An assembly-only companion, ``assembly_breakdown_vs_cpu_cores.png``, uses
the same subcategories on its own vertical scale so small assembly phases
remain visible when plotting dominates the total runtime.
The CSV also records phase invocation counts, form-cache hits, misses and hit
rate, assembly time per call, final cell/DOF counts, and the number of
refinement events so equal-work scaling can be checked directly.
MPI process startup and Python import time are retained separately in the
per-run JSON files as ``launcher_wall_seconds``.

This file only defines the case; the driver, the timing collection, the CSV
file, the figures and the reference comparison are in ``scaling_study.py``
next to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scaling_study import ScalingStudy, load_example


# Study set-up: MPI rank counts (--cores default) and the example that is timed
DEFAULT_CORE_COUNTS = tuple(range(56, 1, -2))
EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "fracture" / "cohesive_phasefield.py"
TIMING_SCHEMA_VERSION = 6  # 5: reduced return mapping; 6: linear solver recorded
SOLID_MODEL = "Solid:Cohesive_LinearElastic"  # The example's solid model (the reduced return mapping without creep)


class CohesiveElasticScaling(ScalingStudy):
    """Strong scaling of the ``full_plate`` case of the cohesive phase-field example."""

    name = "scaling_cohesive_elastic"
    case_label = "FullPlate"
    title = "Cohesive-elastic FullPlate"
    description = (
        "Run the cohesive-elastic FullPlate case as an MPI strong-scaling "
        "study and save its timings and runtime plot."
    )
    solid_model = SOLID_MODEL
    timing_schema_version = TIMING_SCHEMA_VERSION
    default_core_counts = DEFAULT_CORE_COUNTS
    default_cores_help = "every even count from 56 to 2"

    def build_params(self, plotting_enabled: bool, plotting_folder: Path) -> Any:
        """Build FullPlate from the example."""

        example = load_example(EXAMPLE_PATH)
        return example.build_params("full_plate", plots=plotting_enabled, output_dir=plotting_folder)


def main() -> None:
    """Run either the outer scaling driver or one internal MPI worker."""

    CohesiveElasticScaling().main()


if __name__ == "__main__":
    main()
