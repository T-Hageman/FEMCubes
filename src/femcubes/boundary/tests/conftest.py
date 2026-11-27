"""Shared fixtures for boundary-model tests."""

import pytest

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh


@pytest.fixture
def BoundaryModel_Test_Params():
    """Create a simple rectangle-mesh parameter set for boundary tests."""

    class TestParams(ParamsTemplate):
        MeshType = "Rectangle"
        Lx = 4.0
        Ly = 2.0
        dx = 1.0
        dim = 2

    return TestParams()


@pytest.fixture
def boundary_model_mesh(BoundaryModel_Test_Params):
    """Create one mesh with the fields required by boundary-model tests."""
    mesh = Mesh(BoundaryModel_Test_Params)

    mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 2)
    mesh.Add_Vector_Field("uOld", "Old Displacement", "CG", 1, 2)
    mesh.Add_Field("velocity", "Velocity", "CG", 1)
    mesh.Add_Field("phasefield", "Phasefield", "CG", 1)

    mesh.Add_Vector_TestFunction("u", 2, 1)
    mesh.Add_TestFunction("phasefield", 2)
    mesh.finalize_test_functions()

    mesh.Constant_Field("u", 0.0)
    mesh.Constant_Field("uOld", 0.0)
    mesh.Constant_Field("phasefield", 0.0)

    return mesh


@pytest.fixture
def SubdomainBoundary_Test_Params():
    """Create a two-squares parameter set (``domain:left`` / ``domain:right``)."""
    from femcubes.mesh_generators.two_squared import Sample_Parameters

    params = Sample_Parameters()
    params.t = 0.0
    params.dt = 0.1
    params.Solution_Steps = {"u": 1}
    params.Field_Orders = {"u": 1}
    params.Global_Measures = {}
    return params


@pytest.fixture
def subdomain_boundary_mesh(SubdomainBoundary_Test_Params):
    """Create a two-squares mesh whose displacement only lives on ``domain:left``."""
    mesh = Mesh(SubdomainBoundary_Test_Params)

    mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 2, domain="domain:left")
    mesh.Add_Vector_TestFunction("u", 2, 1)
    mesh.finalize_test_functions()
    mesh.Constant_Field("u", 0.0)

    return mesh
