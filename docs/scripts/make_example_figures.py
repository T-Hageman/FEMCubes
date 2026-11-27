"""Make the sample-output figures of the example pages.

The figures on the example pages are the PNG files the example scripts save
themselves (``mesh.SavePlotsToFolder``). This script copies selected ones
into ``docs/_static/examples/<example>/``, trimmed of their white border so
they fit the page, and checks that each stays below the size limit.

Two commands:

``trim`` copies figures from the output folder of a run::

    python docs/scripts/make_example_figures.py trim RUN_OUTPUT_DIR EXAMPLE phasefield u_x ...

writes ``docs/_static/examples/EXAMPLE/phasefield.png`` etc. A name of the
form ``source:target`` stores ``source.png`` as ``target.png``.

``run`` runs an example with its own ``build_params`` and ``run``, with
parameter overrides and, optionally, with the field plots drawn only once at
the end (the movie is then switched off); use it under MPI::

    mpirun -n 4 python docs/scripts/make_example_figures.py run examples/glaciology/shallow_ice.py \\
        --final-plots-only -- --case slope --output-dir /tmp/shallow_ice
    mpirun -n 8 python docs/scripts/make_example_figures.py run examples/glaciology/ice_cliffs.py \\
        --set end_time=600.0 -- --case le_cohesive --output-dir /tmp/ice_cliffs

Each ``--set name=value`` replaces one attribute of the parameter object that
the example's ``build_params`` returns (the value is read as a Python
literal). Field plots of every time step, and the movie, dominate the run time
of the longer examples; ``--final-plots-only`` gives the same final figures.
"""
import argparse
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np

STATIC = Path(__file__).resolve().parents[1] / "_static" / "examples"
MAX_BYTES = 150_000
MARGIN = 8  # pixels kept around the trimmed content


def trim(source: Path, target: Path, width: int | None = None) -> int:
    """Copy ``source`` to ``target`` without its white border; return the file size.

    White rows inside the figure (between the title, the plot and the colour
    bar of a field plot) are shortened to ``2 * MARGIN`` pixels as well.
    With ``width``, wider figures are scaled down to that width in pixels.
    """
    from PIL import Image

    pixels = np.asarray(Image.open(source).convert("RGB"))
    ink = (pixels < 250).any(axis=2)
    columns = np.flatnonzero(ink.any(axis=0))
    pixels = pixels[:, max(columns[0] - MARGIN, 0):columns[-1] + MARGIN + 1]

    keep, blank_run = [], MARGIN  # start as if after a gap, so the top border shrinks to MARGIN
    for row, has_ink in enumerate(ink.any(axis=1)):
        blank_run = 0 if has_ink else blank_run + 1
        if blank_run <= 2 * MARGIN:
            keep.append(row)
    last_ink = max(i for i, row in enumerate(keep) if ink[row].any())
    keep = keep[:last_ink + MARGIN + 1]

    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(pixels[keep])
    if width is not None and image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    image.save(target, optimize=True)
    size = target.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"{target} is {size} bytes, above the limit of {MAX_BYTES}.")
    return size


def _trim_command(arguments) -> None:
    """PRIVATE: Copy the named figures of one run into the static folder of its example."""
    for name in arguments.figures:
        source, _, target = name.partition(":")
        size = trim(arguments.run_dir / f"{source}.png", STATIC / arguments.example / f"{target or source}.png",
                    arguments.width)
        print(f"{arguments.example}/{target or source}.png: {size / 1000:.0f} kB")


def _run_command(arguments, script_arguments) -> None:
    """PRIVATE: Run an example with parameter overrides and, optionally, final field plots only."""
    from femcubes.core.mesh.mesh import Mesh
    from femcubes.core.utils.example_utils import parse_example_arguments

    spec = importlib.util.spec_from_file_location(arguments.script.stem, arguments.script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    overrides = {}
    for item in arguments.set:
        name, value = item.split("=", 1)
        overrides[name] = ast.literal_eval(value)

    build_params = module.build_params

    def build_params_with_overrides(*args, **kwargs):
        params = build_params(*args, **kwargs)
        for name, value in overrides.items():
            if not hasattr(params, name):
                raise AttributeError(f"The parameters of {arguments.script.name} have no attribute {name!r}.")
            setattr(params, name, value)
        if arguments.final_plots_only:
            params._final_plot_fields = (list(params.Plot_Fields), list(getattr(params, "Plot_RO_Fields", [])))
            params.Plot_Fields, params.Plot_RO_Fields, params.record_movie = [], [], False
        return params

    module.build_params = build_params_with_overrides

    save_plots = Mesh.SavePlotsToFolder

    def save_plots_after_final_field_plot(mesh, folder_path):
        final_fields = getattr(mesh.params, "_final_plot_fields", None)
        if final_fields is not None:
            mesh.params.Plot_Fields, mesh.params.Plot_RO_Fields = final_fields
            mesh.Plot_Fields()
        save_plots(mesh, folder_path)

    Mesh.SavePlotsToFolder = save_plots_after_final_field_plot

    options = parse_example_arguments(str(arguments.script), module.__doc__, cases=getattr(module, "CASES", None),
                                      options=getattr(module, "OPTIONS", None), argv=script_arguments)
    kwargs = dict(options.options)
    if options.case is not None:
        kwargs["case"] = options.case
    module.run(**kwargs, quick=options.quick, plots=options.plots, output_dir=options.output_dir)


def main() -> None:
    argv = sys.argv[1:]
    script_arguments = []
    if "--" in argv:
        split = argv.index("--")
        argv, script_arguments = argv[:split], argv[split + 1:]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    trim_parser = commands.add_parser("trim", help="copy trimmed figures of a run into docs/_static/examples")
    trim_parser.add_argument("run_dir", type=Path, help="output folder of the example run")
    trim_parser.add_argument("example", help="folder name under docs/_static/examples")
    trim_parser.add_argument("figures", nargs="+", help="figure names without .png, or source:target")
    trim_parser.add_argument("--width", type=int, default=None, help="scale wider figures down to this width (pixels)")
    run_parser = commands.add_parser("run", help="run an example with overrides (options after --)")
    run_parser.add_argument("script", type=Path, help="example script")
    run_parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                            help="replace a parameter of the example")
    run_parser.add_argument("--final-plots-only", action="store_true",
                            help="plot the fields once at the end instead of every step, without a movie")
    arguments = parser.parse_args(argv)

    if arguments.command == "trim":
        _trim_command(arguments)
    else:
        _run_command(arguments, script_arguments)


if __name__ == "__main__":
    main()
