#!/usr/bin/env python
"""Run the full FEMCubes test suite.

Usage::

    python scripts/run_tests.py [--not-full-model] [-n WORKERS] [pytest arguments]

The tests are run in parallel with pytest-xdist and with a coverage report
when pytest-cov is installed. Any further argument is passed on to pytest,
for example a path or ``-k expression``.
"""
import argparse
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_tests() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=False)
    parser.add_argument("--not-full-model", action="store_true",
                        help="skip the full-model regression tests")
    parser.add_argument("-n", "--workers", default=None,
                        help="number of parallel workers (default: all cores)")
    args, extra_args = parser.parse_known_args()

    os.chdir(REPO_ROOT)
    # One thread per worker; the workers already use all cores.
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    pytest_args = [
        "-v",
        "-o", "addopts=",                 # replace the quiet defaults of pyproject.toml
        "-o", "filterwarnings=default",   # show warnings
        "--tb=short",
        "--color=yes",
        "-ra",
        "--strict-markers",
        "-W", "default",
        "-W", "ignore:.*faster on contiguous arrays:numba.core.errors.NumbaPerformanceWarning",
        "-W", "ignore::pytest.PytestAssertRewriteWarning",
    ]
    if args.not_full_model:
        pytest_args += ["-m", "not full_model"]

    if importlib.util.find_spec("xdist") is not None:
        workers = args.workers or str(os.cpu_count() or 1)
        # Work stealing lets idle workers take over tests from busy ones, so
        # the long tests do not queue behind each other.
        pytest_args += ["-n", workers, "--dist=worksteal"]
        print(f"Parallel run with {workers} workers (pytest-xdist)")
    else:
        print("Serial run (pytest-xdist is not installed)")

    if importlib.util.find_spec("pytest_cov") is not None:
        pytest_args += [
            "--cov=femcubes",
            "--cov-config=pyproject.toml",
            "--cov-report=term-missing:skip-covered",
        ]
    else:
        print("No coverage report (pytest-cov is not installed)")

    return pytest.main(pytest_args + extra_args)


if __name__ == "__main__":
    sys.exit(run_tests())
