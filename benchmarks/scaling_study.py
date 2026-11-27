"""Shared driver for the MPI strong-scaling benchmarks.

A benchmark case is a short script that subclasses :class:`ScalingStudy`,
sets the attributes that name and describe the case, and implements
:meth:`ScalingStudy.build_params`, which returns the parameters of one run.
Everything else is done here: the command-line options, launching one MPI
worker run per rank count (first a sweep with plotting, then one without),
the timing of set-up, JIT warm-up and solve on the slowest rank, the
breakdown of the solve into the phases recorded by the femcubes profiler,
the per-run JSON files, ``scaling_results.csv``, the runtime and breakdown
figures, and the comparison with a stored reference study.

The case script is launched again under MPI for every run, in an internal
worker mode, so it imports this module as a sibling: run it as
``python3 benchmarks/<case>.py`` from any working directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Sequence


# Repository root: working directory and PYTHONPATH entry of the MPI worker runs
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# A run slower than the reference by more than this factor is reported as a regression.
REGRESSION_FACTOR = 1.10
PLOTTING_MODES = ("with_plotting", "without_plotting")  # The two sweeps, run in this order
LINEAR_SOLVERS = ("LU", "GMRES", "GAMG", "FIELDSPLIT")  # values of femcubes LinearSolver
# Labels, categories and colours of the timing breakdown in the CSV and the plots
PLOTTING_MODE_LABELS = {
    "with_plotting": "With plotting",
    "without_plotting": "Without plotting",
}
ASSEMBLY_TIMING_CATEGORIES = (
    "form_evaluation",
    "form_cache_lookup",
    "form_compilation",
    "local_assembly",
    "assembly_communication",
    "boundary_conditions",
    "model_assembly",
    "assembly_bookkeeping",
)
PROFILED_SOLVE_CATEGORIES = (
    *ASSEMBLY_TIMING_CATEGORIES,
    "linear_solve",
    "refinement",
    "plotting",
    "output",
)
TIMING_CALL_CATEGORIES = (
    *PROFILED_SOLVE_CATEGORIES,
    "form_cache_hit",
    "form_cache_miss",
    "matrix_residual_assembly",
    "residual_only_assembly",
)
TIMING_BREAKDOWN_CATEGORIES = (
    "setup",
    "jit_warmup",
    *PROFILED_SOLVE_CATEGORIES,
    "other",
)
TIMING_CATEGORY_LABELS = {
    "setup": "Setup",
    "jit_warmup": "JIT warm-up",
    "form_evaluation": "Form evaluation",
    "form_cache_lookup": "Form cache lookup",
    "form_compilation": "Form binding/compilation",
    "local_assembly": "Local element assembly",
    "assembly_communication": "Assembly communication",
    "boundary_conditions": "Boundary conditions",
    "model_assembly": "Model K/F additions",
    "assembly_bookkeeping": "Assembly bookkeeping",
    "linear_solve": "Linear solve",
    "refinement": "Refinement",
    "plotting": "Plotting",
    "output": "File output",
    "other": "Other solve work",
}
TIMING_CATEGORY_COLORS = {
    "setup": "#4c78a8",
    "jit_warmup": "#b279a2",
    "form_evaluation": "#f2cf5b",
    "form_cache_lookup": "#59a14f",
    "form_compilation": "#ff9da6",
    "local_assembly": "#f58518",
    "assembly_communication": "#e45756",
    "boundary_conditions": "#9c755f",
    "model_assembly": "#72b7b2",
    "assembly_bookkeeping": "#bab0ac",
    "linear_solve": "#7f3c8d",
    "refinement": "#11a579",
    "plotting": "#54a24b",
    "output": "#a0cbe8",
    "other": "#9d9da1",
}


def load_example(example_path: Path) -> ModuleType:
    """Import an example script from its path and return it as a module.

    The module is named after the file, so for
    ``examples/fracture/cohesive_phasefield.py`` its ``build_params`` is
    available as ``load_example(path).build_params``.
    """

    spec = importlib.util.spec_from_file_location(example_path.stem, example_path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


def _validate_driver_arguments(arguments: argparse.Namespace) -> None:
    """PRIVATE: Reject invalid scaling-study inputs explicitly."""

    if not arguments.cores:
        raise ValueError("At least one MPI rank count is required.")
    if any(core_count < 1 for core_count in arguments.cores):
        raise ValueError("MPI rank counts must be positive integers.")
    if len(set(arguments.cores)) != len(arguments.cores):
        raise ValueError("MPI rank counts must not contain duplicates.")
    if arguments.repetitions < 1:
        raise ValueError("--repetitions must be at least one.")
    if arguments.end_time is not None and arguments.end_time <= 0.0:
        raise ValueError("--end-time must be greater than zero.")


def _resolve_mpi_launcher(name: str) -> str:
    """PRIVATE: Resolve the requested MPI launcher or raise a useful error."""

    launcher = shutil.which(name)
    if launcher is None:
        raise FileNotFoundError(
            f"Could not find MPI launcher {name!r}. "
            "Install MPI or select it with --mpiexec."
        )
    return launcher


def _available_cpu_count() -> int:
    """PRIVATE: Return the CPUs available to this process where possible."""

    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _is_mpi_child_process() -> bool:
    """PRIVATE: Detect accidental invocation of the outer driver under MPI."""

    size_variables = (
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "PMIX_SIZE",
        "MV2_COMM_WORLD_SIZE",
    )
    return any(int(os.environ.get(name, "1")) > 1 for name in size_variables)


def _worker_result_path(
    output_dir: Path,
    plotting_mode: str,
    core_count: int,
    repetition: int,
) -> Path:
    """PRIVATE: Return the stable raw-result path for one run."""

    stem = (
        f"{plotting_mode}_cores_{core_count:02d}_"
        f"repeat_{repetition:02d}"
    )
    return output_dir / f"{stem}.json"


def _worker_environment() -> dict[str, str]:
    """PRIVATE: Use one compute thread per MPI rank for a CPU scaling study."""

    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not pythonpath
        else f"{PROJECT_ROOT}{os.pathsep}{pythonpath}"
    )
    environment["PYTHONUNBUFFERED"] = "1"
    # One thread for OpenMP (which also threads MUMPS), for each BLAS library that may be linked
    # (OpenBLAS, MKL, BLIS, Accelerate) and for numba
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[variable] = "1"

    # Open MPI otherwise refuses to start when a container happens to use root.
    # Other MPI implementations simply ignore these environment variables.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        environment["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        environment["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    return environment


def _summarise_results(
    results_by_mode: dict[str, dict[int, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """PRIVATE: Reduce repetitions into one row per mode and rank count."""

    summaries: list[dict[str, Any]] = []
    for plotting_mode in PLOTTING_MODES:
        results_by_core = results_by_mode.get(plotting_mode, {})
        if not results_by_core:
            continue

        baseline_core_count = min(results_by_core)
        baseline_runtime = statistics.fmean(
            result["runtime_seconds"]
            for result in results_by_core[baseline_core_count]
        )

        for core_count in sorted(results_by_core):
            runtimes = [
                float(result["runtime_seconds"])
                for result in results_by_core[core_count]
            ]
            setup_times = [
                float(result["setup_seconds"])
                for result in results_by_core[core_count]
            ]
            solve_times = [
                float(result["solve_seconds"])
                for result in results_by_core[core_count]
            ]
            final_cells = [
                int(result["final_global_cells"])
                for result in results_by_core[core_count]
            ]
            final_dofs = [
                int(result["final_global_dofs"])
                for result in results_by_core[core_count]
            ]
            refinement_events = [
                int(result["refinement_events"])
                for result in results_by_core[core_count]
            ]
            mean_breakdown = {
                category: statistics.fmean(
                    float(result["timing_breakdown_seconds"][category])
                    for result in results_by_core[core_count]
                )
                for category in TIMING_BREAKDOWN_CATEGORIES
            }
            mean_call_counts = {
                category: statistics.fmean(
                    int(result["timing_call_counts"][category])
                    for result in results_by_core[core_count]
                )
                for category in TIMING_CALL_CATEGORIES
            }
            mean_runtime = statistics.fmean(runtimes)
            speedup = baseline_runtime / mean_runtime
            efficiency = speedup * baseline_core_count / core_count
            first = results_by_core[core_count][0]
            summary: dict[str, Any] = {
                "plotting_mode": plotting_mode,
                "cpu_cores": core_count,
                "solid_model": first.get("solid_model", ""),
                "linear_solver": first.get("linear_solver", "LU"),
                "requested_end_time": first.get("requested_end_time"),
                "repetitions": len(runtimes),
                "mean_runtime_seconds": mean_runtime,
                "min_runtime_seconds": min(runtimes),
                "max_runtime_seconds": max(runtimes),
                "std_runtime_seconds": (
                    statistics.pstdev(runtimes)
                    if len(runtimes) > 1
                    else 0.0
                ),
                "mean_setup_seconds": statistics.fmean(setup_times),
                "mean_solve_seconds": statistics.fmean(solve_times),
                "mean_final_global_cells": statistics.fmean(final_cells),
                "mean_final_global_dofs": statistics.fmean(final_dofs),
                "mean_refinement_events": statistics.fmean(
                    refinement_events
                ),
                "speedup_from_smallest_run": speedup,
                "parallel_efficiency": efficiency,
            }
            summary.update(
                {
                    f"mean_{category}_seconds": mean_breakdown[category]
                    for category in TIMING_BREAKDOWN_CATEGORIES
                    if category != "setup"
                }
            )
            summary.update(
                {
                    f"mean_{category}_calls": mean_call_counts[category]
                    for category in TIMING_CALL_CATEGORIES
                }
            )
            total_assembly_seconds = sum(
                mean_breakdown[category]
                for category in ASSEMBLY_TIMING_CATEGORIES
            )
            assembly_calls = (
                mean_call_counts["matrix_residual_assembly"]
                + mean_call_counts["residual_only_assembly"]
            )
            form_cache_lookups = (
                mean_call_counts["form_cache_hit"]
                + mean_call_counts["form_cache_miss"]
            )
            summary["mean_total_assembly_seconds"] = (
                total_assembly_seconds
            )
            summary["mean_assembly_calls"] = assembly_calls
            summary["mean_assembly_seconds_per_call"] = (
                total_assembly_seconds / assembly_calls
                if assembly_calls > 0.0
                else 0.0
            )
            summary["mean_form_cache_hit_rate"] = (
                mean_call_counts["form_cache_hit"] / form_cache_lookups
                if form_cache_lookups > 0.0
                else 0.0
            )
            summaries.append(summary)
    return summaries


def _write_summary_csv(
    summaries: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """PRIVATE: Save the aggregated scaling measurements."""

    fieldnames = list(summaries[0])
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def _load_pyplot():
    """PRIVATE: Load a headless pyplot using a writable cache directory."""

    matplotlib_cache = (
        Path(tempfile.gettempdir())
        / f"femcubes-matplotlib-{getattr(os, 'getuid', lambda: 0)()}"
    )
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("MPLCONFIGDIR"):
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _save_runtime_plot(
    summaries: Sequence[dict[str, Any]],
    output_path: Path,
    reference: dict[tuple[str, int], dict[str, Any]] | None = None,
    *,
    title: str,
) -> None:
    """PRIVATE: Save runtime curves comparing the two plotting modes, and the reference runtimes."""

    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    colors = {
        "with_plotting": "#4c78a8",
        "without_plotting": "#f58518",
    }
    markers = {"with_plotting": "o", "without_plotting": "s"}
    for plotting_mode in PLOTTING_MODES:
        mode_summaries = [
            summary
            for summary in summaries
            if summary["plotting_mode"] == plotting_mode
        ]
        if not mode_summaries:
            continue
        core_counts = [
            int(summary["cpu_cores"]) for summary in mode_summaries
        ]
        runtimes = [
            float(summary["mean_runtime_seconds"])
            for summary in mode_summaries
        ]
        errors = [
            float(summary["std_runtime_seconds"])
            for summary in mode_summaries
        ]
        plot_arguments = {
            "marker": markers[plotting_mode],
            "linewidth": 1.8,
            "color": colors[plotting_mode],
            "label": PLOTTING_MODE_LABELS[plotting_mode],
        }
        if any(error > 0.0 for error in errors):
            axis.errorbar(
                core_counts,
                runtimes,
                yerr=errors,
                capsize=3,
                **plot_arguments,
            )
        else:
            axis.plot(core_counts, runtimes, **plot_arguments)

        baseline_cores = core_counts[0]
        baseline_runtime = runtimes[0]
        ideal_runtime = [
            baseline_runtime * baseline_cores / core_count
            for core_count in core_counts
        ]
        axis.plot(
            core_counts,
            ideal_runtime,
            linestyle="--",
            linewidth=1.0,
            color=colors[plotting_mode],
            alpha=0.55,
            label=(
                f"Ideal from {baseline_cores} cores "
                f"({PLOTTING_MODE_LABELS[plotting_mode].lower()})"
            ),
        )

    for plotting_mode in PLOTTING_MODES:
        points = sorted(
            (cores, float(row["mean_runtime_seconds"]))
            for (mode, cores), row in (reference or {}).items()
            if mode == plotting_mode
        )
        if points:
            axis.plot(
                [cores for cores, _ in points],
                [runtime for _, runtime in points],
                linestyle=":",
                marker="x",
                linewidth=1.2,
                color=colors[plotting_mode],
                alpha=0.8,
                label=f"Reference ({PLOTTING_MODE_LABELS[plotting_mode].lower()})",
            )

    axis.set_xlabel("CPU cores (MPI ranks)")
    axis.set_ylabel("Runtime (s)")
    axis.set_title(f"{title} strong scaling")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _save_timing_breakdown_plot(
    summaries: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> None:
    """PRIVATE: Save comparable stacked timing bars for both plotting modes."""

    _save_stacked_breakdown_plot(
        summaries,
        output_path,
        categories=TIMING_BREAKDOWN_CATEGORIES,
        title=f"{title} runtime breakdown",
    )


def _save_assembly_breakdown_plot(
    summaries: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
) -> None:
    """PRIVATE: Save an assembly-only strong-scaling phase breakdown."""

    _save_stacked_breakdown_plot(
        summaries,
        output_path,
        categories=ASSEMBLY_TIMING_CATEGORIES,
        title=f"{title} assembly breakdown",
    )


def _save_stacked_breakdown_plot(
    summaries: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    categories: Sequence[str],
    title: str,
) -> None:
    """PRIVATE: Save one two-mode stacked timing breakdown."""

    plt = _load_pyplot()
    mode_data = [
        (
            plotting_mode,
            [
                summary
                for summary in summaries
                if summary["plotting_mode"] == plotting_mode
            ],
        )
        for plotting_mode in PLOTTING_MODES
    ]
    mode_data = [
        (plotting_mode, mode_summaries)
        for plotting_mode, mode_summaries in mode_data
        if mode_summaries
    ]
    figure, axes_grid = plt.subplots(
        len(mode_data),
        1,
        figsize=(13.0, 4.8 * len(mode_data)),
        sharey=True,
        squeeze=False,
    )
    for mode_index, (plotting_mode, mode_summaries) in enumerate(mode_data):
        axis = axes_grid[mode_index][0]
        core_counts = [
            int(summary["cpu_cores"]) for summary in mode_summaries
        ]
        # One evenly spaced bar per rank count, labelled with the count: bars at the counts
        # themselves overlap where the counts are close together (1, 2, 4, ...)
        positions = list(range(len(core_counts)))
        bottoms = [0.0] * len(mode_summaries)
        for category in categories:
            values = [
                float(summary[f"mean_{category}_seconds"])
                for summary in mode_summaries
            ]
            axis.bar(
                positions,
                values,
                bottom=bottoms,
                width=0.7,
                color=TIMING_CATEGORY_COLORS[category],
                label=TIMING_CATEGORY_LABELS[category],
            )
            bottoms = [
                bottom + value for bottom, value in zip(bottoms, values)
            ]
        axis.set_ylabel("Runtime (s)")
        axis.set_title(PLOTTING_MODE_LABELS[plotting_mode])
        axis.set_xticks(positions, [str(count) for count in core_counts])
        axis.tick_params(
            axis="x",
            rotation=90 if len(core_counts) > 14 else 0,
        )
        axis.grid(True, axis="y", alpha=0.3)

    axes_grid[-1][0].set_xlabel("CPU cores (MPI ranks)")
    figure.suptitle(title)
    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.8, 0.5),
        fontsize="small",
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.79, 0.97))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _compare_to_reference(
    summaries: Sequence[dict[str, Any]],
    reference: dict[tuple[str, int], dict[str, Any]],
) -> list[str]:
    """PRIVATE: Print runtimes against the reference and return the regressions found.

    A run is a regression when it is slower than the reference by more than
    ``REGRESSION_FACTOR``. A different final mesh size means the runs did not
    do the same work (a change in the physics or the refinement), so their
    runtimes are not comparable; this is reported as well.
    """

    regressions = []
    lines = ["", "Comparison with the reference study:",
             f"  {'mode':<18}{'cores':>6}{'runtime [s]':>14}{'reference [s]':>15}{'ratio':>8}  note"]
    for summary in summaries:
        key = (summary["plotting_mode"], int(summary["cpu_cores"]))
        if key not in reference:
            continue
        row = reference[key]
        runtime = float(summary["mean_runtime_seconds"])
        reference_runtime = float(row["mean_runtime_seconds"])
        ratio = runtime / reference_runtime
        cells, reference_cells = float(summary["mean_final_global_cells"]), float(row["mean_final_global_cells"])
        notes = []
        if abs(cells - reference_cells) > 0.01 * reference_cells:
            notes.append(f"DIFFERENT WORK: {cells:.0f} final cells, reference {reference_cells:.0f}")
        if ratio > REGRESSION_FACTOR:
            notes.append("SLOWER")
            regressions.append(f"{key[0]}, {key[1]} cores: {ratio:.2f} x the reference runtime")
        lines.append(f"  {key[0]:<18}{key[1]:>6}{runtime:>14.1f}{reference_runtime:>15.1f}{ratio:>8.2f}  {'; '.join(notes)}")
    if len(lines) > 3:
        print("\n".join(lines), flush=True)
    return regressions


class ScalingStudy:
    """Strong-scaling study of one benchmark case, with and without plotting.

    A case subclasses this class, sets the class attributes below and
    implements :meth:`build_params`; ``CaseStudy().main()`` then runs either
    the driver or, when launched by the driver under MPI, one worker run.

    Attributes
    ----------
    name : str
        Name of the study; results go to ``results/<name>`` and the stored
        reference is ``reference/<name>/scaling_results.csv``, both next to
        the case script.
    case_label : str
        Short case name, used in progress messages, in error messages and as
        ``case`` in the per-run JSON files.
    title : str
        Start of the figure titles ("<title> strong scaling" and so on).
    description : str
        Description shown by ``--help``.
    solid_model : str
        Model recorded as ``solid_model`` in the results; only reference runs
        with the same value are compared.
    timing_schema_version : int
        Version of the per-run results. ``--resume`` rejects files written
        with another version, so increase it whenever a change to the case or
        to the recorded data makes old results incomparable.
    default_core_counts : sequence of int
        Default of ``--cores``.
    default_cores_help : str
        Description of ``default_core_counts`` in the ``--cores`` help.
    """

    name: str
    case_label: str
    title: str
    description: str
    solid_model: str
    timing_schema_version: int
    default_core_counts: Sequence[int]
    default_cores_help: str

    def __init__(self) -> None:
        missing = [
            attribute
            for attribute in ScalingStudy.__annotations__
            if not hasattr(self, attribute)
        ]
        if missing:
            raise TypeError(
                f"{type(self).__name__} does not set {', '.join(missing)}."
            )

    @property
    def script_path(self) -> Path:
        """Path of the case script; the MPI workers run this file."""

        return Path(inspect.getfile(type(self))).resolve()

    @property
    def default_output_dir(self) -> Path:
        """Default of ``--output-dir``."""

        return self.script_path.parent / "results" / self.name

    @property
    def reference_csv(self) -> Path:
        """Default of ``--reference``."""

        return self.script_path.parent / "reference" / self.name / "scaling_results.csv"

    def build_params(self, plotting_enabled: bool, plotting_folder: Path) -> Any:
        """Return the parameters of one run of the case.

        Called once in every MPI worker run, after the JIT cache has been
        isolated. Return the case as it should be timed (for example an
        example's parameters with the model changed); :meth:`configure_params`
        then applies the settings shared by all scaling studies.

        Parameters
        ----------
        plotting_enabled : bool
            ``True`` in the sweep with plotting.
        plotting_folder : Path
            Folder for the figures of this run.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must implement build_params."
        )

    def configure_params(
        self,
        params: Any,
        end_time: float | None,
        plotting_enabled: bool,
        plotting_folder: Path,
        linear_solver: str = "LU",
    ) -> Any:
        """Apply the settings shared by all scaling studies and return ``params``.

        Sets the linear solver, switches off field files and movies, switches
        on the per-phase profiler, switches plotting on or off for the sweep,
        and applies ``--end-time``.
        """

        from femcubes.core.solvers.solver_enums import LinearSolver

        params.Linear_Solver = LinearSolver(linear_solver)
        # No field files or movies, and the per-phase profiler on
        params.write_vtk = False
        params.write_hdf5 = False
        params.output_dir = None
        params.output_fields = []
        params.PROFILE_TIMINGS = True
        params.record_movie = False
        # Mesh plots after each refinement only in the plotting sweep, never the refinement movie
        params.MeshRefinement = replace(
            params.MeshRefinement,
            replot_on_refinement=plotting_enabled,
            save_remesh_animations=False,
        )
        if plotting_enabled:
            params.movie_folder = str(plotting_folder)
        else:
            params.Plot_Fields = []
            params.Plot_Measures = []
        if end_time is not None:
            if end_time <= params.start_time:
                raise ValueError(
                    f"--end-time must be greater than the {self.case_label} start time."
                )
            params.end_time = end_time
        return params

    def build_parser(self) -> argparse.ArgumentParser:
        """Construct the public driver and internal worker arguments."""

        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument(
            "--cores",
            type=int,
            nargs="+",
            default=list(self.default_core_counts),
            help=f"MPI rank counts to run (default: {self.default_cores_help}).",
        )
        parser.add_argument(
            "--repetitions",
            type=int,
            default=1,
            help="Independent repetitions at each rank count (default: 1).",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=self.default_output_dir,
            help="Directory for raw timings, CSV data, and plots.",
        )
        parser.add_argument(
            "--mpiexec",
            default="mpirun",
            help="MPI launcher executable (default: mpirun).",
        )
        parser.add_argument(
            "--launcher-arg",
            action="append",
            default=[],
            help=(
                "Extra argument passed to the MPI launcher. Repeat this option for "
                "multiple arguments; use --launcher-arg=VALUE for values beginning "
                "with '-'."
            ),
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Reuse valid per-run JSON results already in the output directory.",
        )
        parser.add_argument(
            "--end-time",
            type=float,
            default=None,
            help=(
                f"Override the {self.case_label} end time. This is intended only for quick "
                "smoke studies; omit it for the complete case."
            ),
        )

        parser.add_argument(
            "--linear-solver",
            choices=LINEAR_SOLVERS,
            default="LU",
            help="Linear solver of every run (default: LU, the direct solver of the example).",
        )
        parser.add_argument(
            "--reference",
            type=Path,
            default=self.reference_csv,
            help=(
                "scaling_results.csv of a reference study to compare against "
                "(default: the stored reference in benchmarks/reference/). Only "
                "runs with the same solid model and end time are compared."
            ),
        )
        parser.add_argument(
            "--no-reference",
            dest="reference",
            action="store_const",
            const=None,
            help="Do not compare against a reference study.",
        )

        # The same file is launched under MPI in this internal mode.
        parser.add_argument(
            "--plotting",
            choices=("both", "on", "off"),
            default="both",
            help="Sweeps to run: with and without plotting (default), only with, or only without.",
        )
        parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument(
            "--result-file",
            type=Path,
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--expected-ranks",
            type=int,
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--repetition",
            type=int,
            default=1,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--plotting-mode",
            choices=PLOTTING_MODES,
            default=None,
            help=argparse.SUPPRESS,
        )
        return parser

    def _worker_command(
        self,
        arguments: argparse.Namespace,
        launcher: str,
        plotting_mode: str,
        core_count: int,
        repetition: int,
        result_file: Path,
    ) -> list[str]:
        """PRIVATE: Build one shell-free MPI worker command."""

        command = [
            launcher,
            *arguments.launcher_arg,
            "-n",
            str(core_count),
            sys.executable,
            str(self.script_path),
            "--worker",
            "--expected-ranks",
            str(core_count),
            "--repetition",
            str(repetition),
            "--result-file",
            str(result_file),
            "--plotting-mode",
            plotting_mode,
        ]
        if arguments.end_time is not None:
            command.extend(["--end-time", str(arguments.end_time)])
        command.extend(["--linear-solver", arguments.linear_solver])
        return command

    def _load_worker_result(
        self,
        result_file: Path,
        plotting_mode: str,
        core_count: int,
        repetition: int,
        requested_end_time: float | None,
        linear_solver: str,
    ) -> dict[str, Any]:
        """PRIVATE: Load and validate a worker result before using or resuming it."""

        result = json.loads(result_file.read_text(encoding="utf-8"))
        if result.get("timing_schema_version") != self.timing_schema_version:
            raise ValueError(
                f"{result_file} uses an older timing schema. "
                "Rerun it without --resume."
            )
        if result.get("plotting_mode") != plotting_mode:
            raise ValueError(
                f"{result_file} contains plotting mode "
                f"{result.get('plotting_mode')!r}; expected {plotting_mode!r}."
            )
        expected_plotting_enabled = plotting_mode == "with_plotting"
        if result.get("plotting_enabled") is not expected_plotting_enabled:
            raise ValueError(
                f"{result_file} contains an inconsistent plotting_enabled value."
            )
        if result.get("cpu_cores") != core_count:
            raise ValueError(
                f"{result_file} contains {result.get('cpu_cores')!r} ranks; "
                f"expected {core_count}."
            )
        if result.get("repetition") != repetition:
            raise ValueError(
                f"{result_file} contains repetition {result.get('repetition')!r}; "
                f"expected {repetition}."
            )
        if result.get("requested_end_time") != requested_end_time:
            raise ValueError(
                f"{result_file} was generated with end-time override "
                f"{result.get('requested_end_time')!r}; expected "
                f"{requested_end_time!r}. Remove the stale result or omit --resume."
            )
        if result.get("linear_solver") != linear_solver:
            raise ValueError(
                f"{result_file} was generated with linear solver "
                f"{result.get('linear_solver')!r}; expected {linear_solver!r}. "
                "Use another --output-dir or omit --resume."
            )
        runtime = result.get("runtime_seconds")
        if not isinstance(runtime, (int, float)) or runtime <= 0.0:
            raise ValueError(f"{result_file} contains an invalid runtime.")
        for quantity in (
            "initial_global_cells",
            "final_global_cells",
            "initial_global_dofs",
            "final_global_dofs",
            "refinement_events",
        ):
            value = result.get(quantity)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{result_file} contains an invalid {quantity!r} value."
                )

        breakdown = result.get("timing_breakdown_seconds")
        if not isinstance(breakdown, dict):
            raise ValueError(
                f"{result_file} does not contain the timing breakdown required "
                "by the current scaling study. Rerun it without --resume."
            )
        for category in TIMING_BREAKDOWN_CATEGORIES:
            value = breakdown.get(category)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(
                    f"{result_file} contains an invalid {category!r} timing."
                )
        call_counts = result.get("timing_call_counts")
        if not isinstance(call_counts, dict):
            raise ValueError(
                f"{result_file} does not contain profiler invocation counts. "
                "Rerun it without --resume."
            )
        for category in TIMING_CALL_CATEGORIES:
            value = call_counts.get(category)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(
                    f"{result_file} contains an invalid {category!r} call count."
                )
        breakdown_total = sum(
            float(breakdown[category])
            for category in TIMING_BREAKDOWN_CATEGORIES
        )
        if not math.isclose(breakdown_total, runtime, rel_tol=1.0e-9, abs_tol=1.0e-9):
            raise ValueError(
                f"{result_file} timing breakdown totals {breakdown_total:.6e} s, "
                f"but runtime_seconds is {runtime:.6e} s."
            )
        return result

    def _run_worker_subprocess(
        self,
        command: Sequence[str],
        result_file: Path,
        plotting_mode: str,
        core_count: int,
        repetition: int,
        requested_end_time: float | None,
        linear_solver: str,
    ) -> dict[str, Any]:
        """PRIVATE: Execute one MPI run and return its validated timing result."""

        print(
            f"Running {self.case_label} {PLOTTING_MODE_LABELS[plotting_mode].lower()} "
            f"with {core_count} MPI ranks "
            f"(repetition {repetition})",
            flush=True,
        )
        start = time.perf_counter()
        completed = subprocess.run(
            list(command),
            cwd=PROJECT_ROOT,
            env=_worker_environment(),
            check=False,
        )
        launcher_wall_seconds = time.perf_counter() - start

        if completed.returncode != 0:
            raise RuntimeError(
                f"MPI run with {core_count} ranks failed with exit code "
                f"{completed.returncode}."
            )
        if not result_file.exists():
            raise RuntimeError(
                f"MPI run with {core_count} ranks completed without writing "
                f"{result_file}."
            )

        result = self._load_worker_result(
            result_file,
            plotting_mode,
            core_count,
            repetition,
            requested_end_time,
            linear_solver,
        )
        result["launcher_wall_seconds"] = launcher_wall_seconds
        result_file.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"Completed {PLOTTING_MODE_LABELS[plotting_mode].lower()}, "
            f"{core_count} ranks in "
            f"{result['runtime_seconds']:.3f} s",
            flush=True,
        )
        return result

    def _save_current_summary(
        self,
        results_by_mode: dict[str, dict[int, list[dict[str, Any]]]],
        output_dir: Path,
        reference: dict[tuple[str, int], dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """PRIVATE: Preserve usable CSV and plot output after every completed size."""

        summaries = _summarise_results(results_by_mode)
        _write_summary_csv(summaries, output_dir / "scaling_results.csv")
        _save_runtime_plot(
            summaries,
            output_dir / "runtime_vs_cpu_cores.png",
            reference,
            title=self.title,
        )
        _save_timing_breakdown_plot(
            summaries,
            output_dir / "runtime_breakdown_vs_cpu_cores.png",
            title=self.title,
        )
        _save_assembly_breakdown_plot(
            summaries,
            output_dir / "assembly_breakdown_vs_cpu_cores.png",
            title=self.title,
        )
        return summaries

    def _load_reference(
        self,
        reference_csv: Path | None,
        end_time: float | None,
        linear_solver: str = "LU",
    ) -> dict[tuple[str, int], dict[str, Any]]:
        """PRIVATE: Return the reference rows for this solid model, linear solver and end time, by (mode, cores)."""

        if reference_csv is None:
            return {}
        if not reference_csv.exists():
            print(f"No reference study at {reference_csv}; skipping the comparison.", flush=True)
            return {}
        with reference_csv.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        reference = {}
        for row in rows:
            row_end_time = float(row["requested_end_time"]) if row.get("requested_end_time") else None
            same_solver = row.get("linear_solver", "LU") == linear_solver
            if row.get("solid_model") == self.solid_model and row_end_time == end_time and same_solver:
                reference[(row["plotting_mode"], int(row["cpu_cores"]))] = row
        if rows and not reference:
            print(
                f"WARNING: the reference study {reference_csv} has no runs with solid model "
                f"{self.solid_model}, linear solver {linear_solver} and end time {end_time}; skipping the comparison.",
                flush=True,
            )
        return reference

    def run_study(self, arguments: argparse.Namespace) -> None:
        """Launch the requested MPI runs and save the scaling study."""

        _validate_driver_arguments(arguments)
        if _is_mpi_child_process():
            raise RuntimeError(
                "Run the scaling driver with plain Python, not mpirun. "
                "It launches each requested MPI size itself."
            )

        launcher = _resolve_mpi_launcher(arguments.mpiexec)
        core_counts = sorted(arguments.cores, reverse=True)
        output_dir = arguments.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        available_cpus = _available_cpu_count()
        if max(core_counts) > available_cpus:
            print(
                f"WARNING: the largest run requests {max(core_counts)} ranks, but "
                f"this process currently sees {available_cpus} CPUs. The MPI "
                "launcher may reject or oversubscribe that run.",
                flush=True,
            )

        reference = self._load_reference(arguments.reference, arguments.end_time, arguments.linear_solver)
        summaries: list[dict[str, Any]] = []
        results_by_mode: dict[str, dict[int, list[dict[str, Any]]]] = {}
        selected_modes = {"both": PLOTTING_MODES, "on": ("with_plotting",), "off": ("without_plotting",)}
        for plotting_mode in selected_modes[getattr(arguments, "plotting", "both")]:
            print(
                f"\nStarting sweep "
                f"{PLOTTING_MODE_LABELS[plotting_mode].lower()}...",
                flush=True,
            )
            results_by_core: dict[int, list[dict[str, Any]]] = {}
            results_by_mode[plotting_mode] = results_by_core
            for core_count in core_counts:
                core_results: list[dict[str, Any]] = []
                for repetition in range(1, arguments.repetitions + 1):
                    result_file = _worker_result_path(
                        output_dir,
                        plotting_mode,
                        core_count,
                        repetition,
                    )
                    if arguments.resume and result_file.exists():
                        result = self._load_worker_result(
                            result_file,
                            plotting_mode,
                            core_count,
                            repetition,
                            arguments.end_time,
                            arguments.linear_solver,
                        )
                        print(
                            f"Reusing {result_file} "
                            f"({result['runtime_seconds']:.3f} s)",
                            flush=True,
                        )
                    else:
                        command = self._worker_command(
                            arguments,
                            launcher,
                            plotting_mode,
                            core_count,
                            repetition,
                            result_file,
                        )
                        result = self._run_worker_subprocess(
                            command,
                            result_file,
                            plotting_mode,
                            core_count,
                            repetition,
                            arguments.end_time,
                            arguments.linear_solver,
                        )
                    core_results.append(result)

                results_by_core[core_count] = core_results
                summaries = self._save_current_summary(results_by_mode, output_dir, reference)

        print(
            "Scaling study complete.\n"
            f"  CSV:  {output_dir / 'scaling_results.csv'}\n"
            f"  Runtime plot:   {output_dir / 'runtime_vs_cpu_cores.png'}\n"
            f"  Breakdown plot: "
            f"{output_dir / 'runtime_breakdown_vs_cpu_cores.png'}\n"
            f"  Assembly plot:  "
            f"{output_dir / 'assembly_breakdown_vs_cpu_cores.png'}",
            flush=True,
        )
        regressions = _compare_to_reference(summaries, reference)
        if regressions:
            print("PERFORMANCE REGRESSION against the reference study:\n  " + "\n  ".join(regressions), flush=True)

    def _worker_params(
        self,
        end_time: float | None,
        plotting_enabled: bool,
        plotting_folder: Path,
        linear_solver: str = "LU",
    ) -> Any:
        """PRIVATE: Build the case with the requested benchmark plotting mode."""

        from femcubes.core.utils.mpi_utils import isolate_jit_cache

        # Each run compiles its forms into its own cache, so every run includes the same JIT warm-up.
        isolate_jit_cache()
        params = self.build_params(plotting_enabled, plotting_folder)
        return self.configure_params(
            params,
            end_time,
            plotting_enabled,
            plotting_folder,
            linear_solver,
        )

    def run_worker(self, arguments: argparse.Namespace) -> None:
        """Run one MPI case and write rank-zero timing data."""

        from mpi4py import MPI

        from femcubes import Mesh, Physics, TimeSolver

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if arguments.result_file is None:
            raise ValueError("The internal worker requires --result-file.")
        if arguments.expected_ranks is None:
            raise ValueError("The internal worker requires --expected-ranks.")
        if arguments.plotting_mode is None:
            raise ValueError("The internal worker requires --plotting-mode.")
        if size != arguments.expected_ranks:
            raise RuntimeError(
                f"MPI started {size} ranks; expected {arguments.expected_ranks}."
            )

        plotting_enabled = arguments.plotting_mode == "with_plotting"
        result_file = arguments.result_file.resolve()
        plotting_folder = (
            result_file.parent
            / "plotting_artifacts"
            / result_file.stem
        )
        params = self._worker_params(
            arguments.end_time,
            plotting_enabled,
            plotting_folder,
            arguments.linear_solver,
        )
        initial_dt = params.dt

        comm.Barrier()
        setup_start = MPI.Wtime()
        mesh = Mesh(params)
        initial_plotting_seconds = 0.0
        if plotting_enabled:
            initial_plotting_start = MPI.Wtime()
            mesh.plot_mesh()
            initial_plotting_seconds = (
                MPI.Wtime() - initial_plotting_start
            )
        physics = Physics(mesh, params)
        physics.Initialize_Fields()
        solver = TimeSolver(params, physics)
        initial_global_cells = int(
            mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_global
        )
        initial_global_dofs = mesh.get_global_solution_dof_count()
        setup_local = MPI.Wtime() - setup_start
        comm.Barrier()
        setup_profiles = comm.allgather(
            {
                "rank": rank,
                "setup_seconds": setup_local,
                "plotting_seconds": initial_plotting_seconds,
            }
        )
        critical_setup_profile = max(
            setup_profiles,
            key=lambda profile: (profile["setup_seconds"], -profile["rank"]),
        )
        setup_phase_seconds = critical_setup_profile["setup_seconds"]
        setup_plotting_seconds = min(
            critical_setup_profile["plotting_seconds"],
            setup_phase_seconds,
        )
        setup_seconds = setup_phase_seconds - setup_plotting_seconds

        comm.Barrier()
        warmup_start = MPI.Wtime()
        solver.warmup_jit()
        warmup_local = MPI.Wtime() - warmup_start
        comm.Barrier()
        jit_warmup_seconds = comm.allreduce(warmup_local, op=MPI.MAX)
        solver.jit_warmup = False
        solver.reset_runtime_timings()

        comm.Barrier()
        solve_start = MPI.Wtime()
        solver.run()
        plotting_finalize_seconds = 0.0
        if plotting_enabled:
            plotting_finalize_start = MPI.Wtime()
            mesh.SavePlotsToFolder(str(plotting_folder))
            mesh.Close_Plots()
            plotting_finalize_seconds = MPI.Wtime() - plotting_finalize_start
        solve_local = MPI.Wtime() - solve_start
        final_global_cells = int(
            mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_global
        )
        final_global_dofs = mesh.get_global_solution_dof_count()
        local_profiled_timings = solver.get_runtime_timings()
        local_profiled_counts = solver.get_runtime_timing_counts()
        local_profiled_timings["plotting"] += plotting_finalize_seconds
        profiled_solve_seconds = {
            category: float(local_profiled_timings.get(category, 0.0))
            for category in PROFILED_SOLVE_CATEGORIES
        }
        other_seconds = max(
            0.0,
            solve_local - sum(profiled_solve_seconds.values()),
        )
        local_timing_profile = {
            "rank": rank,
            "solve_seconds": solve_local,
            "breakdown": {
                **profiled_solve_seconds,
                "other": other_seconds,
            },
            "call_counts": {
                category: int(local_profiled_counts.get(category, 0))
                for category in TIMING_CALL_CATEGORIES
            },
        }
        comm.Barrier()
        solve_seconds = comm.allreduce(solve_local, op=MPI.MAX)
        timing_profiles = comm.allgather(local_timing_profile)
        critical_profile = max(
            timing_profiles,
            key=lambda profile: (profile["solve_seconds"], -profile["rank"]),
        )
        timing_breakdown = {
            "setup": setup_seconds,
            "jit_warmup": jit_warmup_seconds,
            **critical_profile["breakdown"],
        }
        timing_breakdown["plotting"] += setup_plotting_seconds

        result = {
            "timing_schema_version": self.timing_schema_version,
            "case": self.case_label,
            "solid_model": self.solid_model,
            "linear_solver": arguments.linear_solver,
            "plotting_mode": arguments.plotting_mode,
            "plotting_enabled": plotting_enabled,
            "cpu_cores": size,
            "repetition": arguments.repetition,
            "requested_end_time": arguments.end_time,
            "runtime_seconds": (
                setup_phase_seconds + jit_warmup_seconds + solve_seconds
            ),
            "setup_seconds": setup_seconds,
            "setup_phase_seconds": setup_phase_seconds,
            "jit_warmup_seconds": jit_warmup_seconds,
            "solve_seconds": solve_seconds,
            "start_time": params.start_time,
            "end_time": params.end_time,
            "initial_dt": initial_dt,
            "final_dt": params.dt,
            "initial_global_cells": initial_global_cells,
            "final_global_cells": final_global_cells,
            "initial_global_dofs": initial_global_dofs,
            "final_global_dofs": final_global_dofs,
            "refinement_events": int(solver.refinement_event_count),
            "timing_rank": critical_profile["rank"],
            "timing_breakdown_seconds": timing_breakdown,
            "timing_call_counts": critical_profile["call_counts"],
        }
        if rank == 0:
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print("SCALING_RESULT=" + json.dumps(result), flush=True)
        comm.Barrier()

    def main(self) -> None:
        """Run either the outer scaling driver or one internal MPI worker."""

        arguments = self.build_parser().parse_args()
        if arguments.worker:
            self.run_worker(arguments)
        else:
            self.run_study(arguments)
