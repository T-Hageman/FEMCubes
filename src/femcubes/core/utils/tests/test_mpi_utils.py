from pathlib import Path
import sys
import types

from mpi4py import MPI

from femcubes.core.utils import mpi_utils


class _FakeComm:
    def __init__(self):
        self.calls = []

    def bcast(self, value, root=0):
        self.calls.append((value, root))
        return value


def _install_fake_dolfinx_jit(monkeypatch, *, timeout=(10, "default"), cache_dir=("/tmp/orig", "default")):
    fake_jit = types.ModuleType("dolfinx.jit")
    fake_jit.DOLFINX_DEFAULT_JIT_OPTIONS = {
        "timeout": timeout,
        "cache_dir": cache_dir,
    }
    fake_dolfinx = types.ModuleType("dolfinx")
    fake_dolfinx.jit = fake_jit
    monkeypatch.setitem(sys.modules, "dolfinx", fake_dolfinx)
    monkeypatch.setitem(sys.modules, "dolfinx.jit", fake_jit)
    return fake_jit


def test_comm_rank_size_flags():
    assert mpi_utils.comm == MPI.COMM_WORLD
    assert mpi_utils.rank == mpi_utils.comm.Get_rank()
    assert mpi_utils.size == mpi_utils.comm.Get_size()
    assert mpi_utils.is_parallel == (mpi_utils.size > 1)


def test_mprint_respects_root(capsys):
    mpi_utils.mprint("root-only-message", root=mpi_utils.rank)
    assert "root-only-message" in capsys.readouterr().out

    mpi_utils.mprint("hidden-message", root=mpi_utils.rank + 1)
    assert "hidden-message" not in capsys.readouterr().out


def test_isolate_jit_cache_updates_timeout_only(monkeypatch):
    fake_jit = _install_fake_dolfinx_jit(monkeypatch)

    mpi_utils.isolate_jit_cache(timeout=45, per_process=False)

    assert fake_jit.DOLFINX_DEFAULT_JIT_OPTIONS["timeout"] == (45, "default")
    assert fake_jit.DOLFINX_DEFAULT_JIT_OPTIONS["cache_dir"] == ("/tmp/orig", "default")


def test_isolate_jit_cache_sets_process_specific_cache_dir(monkeypatch):
    fake_jit = _install_fake_dolfinx_jit(monkeypatch, cache_dir=(Path("/tmp/orig"), "default"))
    fake_comm = _FakeComm()

    monkeypatch.setattr(mpi_utils, "comm", fake_comm)
    monkeypatch.setattr(mpi_utils, "rank", 0)
    monkeypatch.setattr(mpi_utils.os, "getpid", lambda: 43210)
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/femcubes-cache")

    mpi_utils.isolate_jit_cache(timeout=75, per_process=True)

    assert fake_jit.DOLFINX_DEFAULT_JIT_OPTIONS["timeout"] == (75, "default")
    assert fake_jit.DOLFINX_DEFAULT_JIT_OPTIONS["cache_dir"] == (
        Path("/tmp/femcubes-cache/fenics/run_43210"),
        "default",
    )
    assert fake_comm.calls == [(43210, 0)]
