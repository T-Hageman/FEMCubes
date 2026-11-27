"""Tests for the ExternalForceFromFile boundary model."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from mpi4py import MPI

from femcubes.boundary.external_force_from_file import ExternalForceFromFile


def _write_uniform_force_table(tmp_path, value):
    """Write a lookup table with the same traction everywhere and at all times."""
    tmp_path = type(tmp_path)(MPI.COMM_WORLD.bcast(str(tmp_path), root=0))
    files = {"file_x": tmp_path / "x.csv", "file_t": tmp_path / "t.csv", "file_F": tmp_path / "F.csv"}
    if MPI.COMM_WORLD.rank == 0:
        np.savetxt(files["file_x"], [0.0, 2.0], delimiter=",")
        np.savetxt(files["file_t"], [0.0, 1.0], delimiter=",")
        np.savetxt(files["file_F"], np.full((2, 2), value), delimiter=",")
    MPI.COMM_WORLD.barrier()
    return {key: str(path) for key, path in files.items()}


def test_external_force_from_file_acts_on_a_subdomain_field(
    subdomain_boundary_mesh, SubdomainBoundary_Test_Params, tmp_path
):
    params = SubdomainBoundary_Test_Params
    files = _write_uniform_force_table(tmp_path, 2.0)
    params.Models = {
        "Load": {"boundary": "left", "field": "u_x", "coordinate": 1, "domain": "domain:left", **files},
        "Outside": {"boundary": "top", "field": "u_x", **files},
    }

    model = ExternalForceFromFile("Load", params, subdomain_boundary_mesh)
    _, residual = model.assemble_KF(model.step)

    probe = fem.Function(subdomain_boundary_mesh.Get_Field("u").function_space)
    probe.x.array[0::2] = 1.0
    local = fem.assemble_scalar(subdomain_boundary_mesh.Compile_Form(ufl.action(residual, probe)))
    assert subdomain_boundary_mesh.mesh.comm.allreduce(local, op=MPI.SUM) == pytest.approx(-2.0)

    with pytest.raises(ValueError, match="cannot act on field 'u'"):
        outside = ExternalForceFromFile("Outside", params, subdomain_boundary_mesh)
        outside.assemble_KF(outside.step)
