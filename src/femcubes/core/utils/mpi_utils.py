"""MPI utility helpers shared across FEMCubes.

This module provides a small, stable wrapper around the global
``MPI.COMM_WORLD`` communicator used throughout the codebase.  It also
contains convenience helpers for root-only printing and for isolating
the DOLFINx JIT cache between independent runs.
"""

from pathlib import Path
from typing import Any
import os

from mpi4py import MPI

# Global MPI communicator used across the codebase.
comm = MPI.COMM_WORLD

# Cached communicator metadata for convenience.
rank: int = comm.Get_rank()
size: int = comm.Get_size()
is_parallel: bool = size > 1


def isolate_jit_cache(timeout: int = 120, per_process: bool = True) -> None:
    """Configure the DOLFINx JIT cache for safer concurrent execution.

    This helper is intended to be called once near the start of an
    entry-point script, before any forms are JIT-compiled.  It always
    increases the JIT lock timeout and can optionally place each
    independent OS-level run into its own cache directory.

    Parameters
    ----------
    timeout : int, optional
        Lock-wait timeout in seconds written into
        ``dolfinx.jit.DOLFINX_DEFAULT_JIT_OPTIONS``.  The default of
        ``120`` is intentionally larger than the DOLFINx default to
        reduce transient lock failures on busy systems.
    per_process : bool, optional
        If ``True`` (default), all MPI ranks in the current run share a
        cache directory named from rank 0's process ID under
        ``$XDG_CACHE_HOME/fenics`` (or ``~/.cache/fenics`` if the
        environment variable is unset).  Separate runs therefore do not
        contend for the same lock files.  If ``False``, only the timeout
        is updated and the default shared cache directory is left in
        place.
    """
    import dolfinx.jit as _jit

    _jit.DOLFINX_DEFAULT_JIT_OPTIONS["timeout"] = (
        timeout,
        _jit.DOLFINX_DEFAULT_JIT_OPTIONS["timeout"][1],
    )

    if per_process:
        pid = os.getpid() if rank == 0 else 0
        pid = comm.bcast(pid, root=0)

        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        cache_dir = base / "fenics" / f"run_{pid}"
        _jit.DOLFINX_DEFAULT_JIT_OPTIONS["cache_dir"] = (
            cache_dir,
            _jit.DOLFINX_DEFAULT_JIT_OPTIONS["cache_dir"][1],
        )


def mprint(message: Any, root: int = 0) -> None:
    """Print a message from one MPI rank only.

    Parameters
    ----------
    message : Any
        Object to print.  It is forwarded directly to ``print()`` on the
        selected rank.
    root : int, optional
        Rank that is allowed to print the message.  All other ranks
        return silently.
    """
    if rank == root:
        print(message, flush=True)
