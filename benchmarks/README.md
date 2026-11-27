# Benchmarks

Scripts that measure the performance of FEMCubes, rather than show how to set
up a model (see [`../examples`](../examples) for those).

| Script | Measures |
|---|---|
| `scaling_cohesive_elastic.py` | strong scaling of the `full_plate` case of `examples/fracture/cohesive_phasefield.py`; with and without plotting, with a breakdown of the run time |
| `scaling_cohesive_elastic_half_ell.py` | the same case with half the phase-field length scale, a larger system once the crack bands are refined |

`scaling_study.py` is not a benchmark itself: it holds `ScalingStudy`, the
driver shared by the scaling benchmarks (command-line options, MPI launches,
timing collection, CSV file, figures and reference comparison). Each
benchmark script only defines its case.

```bash
# complete study: every even rank count from 56 down to 2
python benchmarks/scaling_cohesive_elastic.py

# a shorter selection, cut off after 200 s of simulated time
python benchmarks/scaling_cohesive_elastic.py --cores 2 4 8 --end-time 200
```

Results (`scaling_results.csv`, one JSON file per run, and the runtime and
breakdown figures) are written to `benchmarks/results/scaling_cohesive_elastic`,
or to the folder given by `--output-dir`. `--resume` reuses runs already in
that folder; `--help` lists the other options.

## Writing a new benchmark case

A new scaling benchmark is a script in `benchmarks/` that subclasses
`ScalingStudy`, sets the attributes that name and describe the case, and
implements `build_params`, which returns the parameters of one run:

```python
from pathlib import Path
from typing import Any

from scaling_study import ScalingStudy, load_example

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "glaciology" / "ice_cliffs.py"


class IceCliffScaling(ScalingStudy):
    name = "scaling_ice_cliff"             # results/scaling_ice_cliff, reference/scaling_ice_cliff
    case_label = "IceCliff"                # progress messages and "case" in the JSON files
    title = "Elastic cohesive ice cliff"   # start of the figure titles
    description = "Run the le_cohesive ice cliff as an MPI strong-scaling study."
    solid_model = "Solid:Cohesive_LinearElastic"  # recorded; the reference comparison filters on it
    timing_schema_version = 1              # increase when old results become incomparable
    default_core_counts = (8, 4, 2)
    default_cores_help = "8, 4 and 2"

    def build_params(self, plotting_enabled: bool, plotting_folder: Path) -> Any:
        example = load_example(EXAMPLE_PATH)
        return example.build_params("le_cohesive", plots=plotting_enabled, output_dir=plotting_folder)


if __name__ == "__main__":
    IceCliffScaling().main()
```

`build_params` returns the case as it should be timed. The settings that
every scaling study needs are then applied by `ScalingStudy.configure_params`:
the `--linear-solver` choice, no field files or movies, the per-phase
profiler on, plotting on or off for the two sweeps, and `--end-time`.
Override `configure_params` only if a case needs these done differently. Run
the script with plain Python, as above; it imports `scaling_study` from its
own folder, and the driver launches the same script under MPI for every run.

## Reference results

`reference/scaling_cohesive_elastic/` holds a stored study that serves as a
performance target. Every run of the benchmark compares itself with it:
the reference runtimes are drawn as dotted lines in
`runtime_vs_cpu_cores.png`, and the end of the run prints the runtime of
each rank count against the reference. Runs more than 10 % slower are
reported as a regression. Runs that end with a different number of cells
than the reference are reported as doing different work, since a change in
the physics or the refinement makes their runtimes incomparable. Only runs
with the same solid model and `--end-time` are compared; `--reference CSV`
compares with another study and `--no-reference` skips the comparison.

Timings depend on the machine and on what else runs on it. The stored
reference is therefore only meaningful on the machine it was made on (see
`reference/scaling_cohesive_elastic/README.md`), with the machine otherwise
idle. To replace it, run the study and copy its `scaling_results.csv`, the
per-run JSON files and the figures into that folder, together with a note of
the commit and the machine.
