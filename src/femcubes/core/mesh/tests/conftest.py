"""Shared fixtures for mesh tests that need a small generated simplex mesh."""

from mpi4py import MPI
import pytest

from femcubes.core.params_template import ParamsTemplate


# The MPI launcher collects this module under two ranks; serial discovery
# would only duplicate those cases as skips.
collect_ignore = (
    ["test_mesh_refinement_distributed_contracts.py"] if MPI.COMM_WORLD.size != 2 else []
)


@pytest.fixture
def refinement_rectangle_params():
    """Return a small deterministic 2-D rectangle configuration."""

    class RefinementRectangleParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 2.0
        Ly = 1.0
        dx = 0.5
        dim = 2
        Quadrature_Degree = 3

    return RefinementRectangleParams()


@pytest.fixture
def refinement_mesh(refinement_rectangle_params):
    """Return the shared small rectangle as an FEMCubes mesh wrapper."""

    from femcubes.core.mesh.mesh import Mesh

    return Mesh(refinement_rectangle_params)
