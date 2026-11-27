"""Command-line options shared by the example and benchmark scripts.

Every script in ``examples/`` accepts the same options, so each one can be run
from any working directory, at full size or as a short smoke run, and with its
output in a chosen folder::

    python examples/fracture/cohesive_phasefield.py --case dynamic
    mpirun -n 4 python examples/glaciology/ice_cliffs.py --quick --output-dir /tmp/cliffs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ExampleArguments:
    """Options of one example run.

    Attributes
    ----------
    case : str or None
        Selected case, or ``None`` for scripts without cases.
    quick : bool
        Run a short, coarse version of the case, intended for smoke tests: it
        checks that the script runs and produces its figures, not its results.
    plots : bool
        Create and save figures.
    output_dir : Path
        Folder for figures and result files of this case.
    options : dict
        Values of the script-specific options, by name.
    """

    case: str | None
    quick: bool
    plots: bool
    output_dir: Path
    options: dict = field(default_factory=dict)


def parse_example_arguments(script_file: str, description: str | None, cases: Sequence[str] | None = None,
                            default_case: str | None = None, options: Mapping[str, Any] | None = None,
                            argv: Sequence[str] | None = None) -> ExampleArguments:
    """Parse the standard options of an example script.

    Parameters
    ----------
    script_file : str
        ``__file__`` of the calling script. The default output folder is
        ``results/<script name>[/<case>]`` next to the script, so the output
        location does not depend on the working directory.
    description : str or None
        Help text, usually the script's docstring.
    cases : sequence of str, optional
        Names of the cases the script offers. Without cases, ``--case`` is
        not available and :attr:`ExampleArguments.case` is ``None``.
    default_case : str, optional
        Case run when ``--case`` is not given; defaults to the first case.
    options : mapping, optional
        Script-specific options as ``{name: default}``. Each becomes a
        ``--name`` option (underscores written as dashes) of the default's
        type; a boolean default ``False`` gives a flag. The values are
        returned in :attr:`ExampleArguments.options`.
    argv : sequence of str, optional
        Arguments to parse instead of ``sys.argv[1:]``.

    Returns
    -------
    ExampleArguments
        The parsed options.

    Examples
    --------
    >>> args = parse_example_arguments(__file__, __doc__, cases=["full_plate", "dynamic"])  # doctest: +SKIP
    >>> params = _build_params(args.case, quick=args.quick, plots=args.plots, output_dir=args.output_dir)  # doctest: +SKIP
    """
    cases = list(cases) if cases is not None else None
    if cases is not None and not cases:
        raise ValueError("cases must contain at least one case name.")
    if default_case is not None and (cases is None or default_case not in cases):
        raise ValueError(f"default_case {default_case!r} is not one of the cases {cases}.")

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    if cases is not None:
        parser.add_argument("--case", choices=cases, default=default_case or cases[0],
                            help="case to run (default: %(default)s)")
    parser.add_argument("--quick", action="store_true",
                        help="short, coarse run that checks the script works (not its results)")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="do not create or save figures")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="folder for figures and results (default: results/<script>[/<case>] next to the script)")
    for name, default in (options or {}).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            if default:
                raise ValueError(f"Boolean option {name!r} must default to False, so that {flag} switches it on.")
            parser.add_argument(flag, dest=name, action="store_true")
        elif isinstance(default, (int, float, str)):
            parser.add_argument(flag, dest=name, type=type(default), default=default, help="(default: %(default)s)")
        else:
            raise ValueError(f"Option {name!r} must default to a bool, int, float or str, not {type(default).__name__}.")
    parsed = parser.parse_args(argv)

    case = getattr(parsed, "case", None)
    output_dir = parsed.output_dir.resolve() if parsed.output_dir is not None else default_output_dir(script_file, case)
    values = {name: getattr(parsed, name) for name in (options or {})}
    return ExampleArguments(case=case, quick=parsed.quick, plots=parsed.plots, output_dir=output_dir, options=values)


def default_output_dir(script_file: str, case: str | None = None) -> Path:
    """Return the default output folder of an example: ``results/<script name>[/<case>]`` next to the script."""
    script = Path(script_file).resolve()
    output_dir = script.parent / "results" / script.stem
    return output_dir / case if case is not None else output_dir
