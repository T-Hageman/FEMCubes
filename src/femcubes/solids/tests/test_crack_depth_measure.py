import numpy as np
import pytest

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.two_squared import Sample_Parameters
from femcubes.solids.crack_depth_measure import CrackDepthMeasure


def _build_params():
    class Params(ParamsTemplate):
        MeshType: str = "Rectangle"
        dx: float = 0.25
        dim: int = 2
        Lx: float = 4.0
        Ly: float = 2.0

        Global_Measures: dict = {}
        Solution_Steps = {"phasefield": 0}
        Field_Orders = {"phasefield": 1}

        Models = {
            "CD": {
                "Type": "Solid:CrackDepthMeasure",
            }
        }

    return Params()


def test_crack_depth_measure_detects_initialized_phasefield_notch():
    params = _build_params()
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", params.Field_Orders["phasefield"])

    phasefield = mesh.Get_Field("phasefield", params.Solution_Steps["phasefield"])
    notch_depth = 0.5
    notch_y = params.Ly - notch_depth

    def _notch(x):
        in_x = np.logical_and(x[0] >= 1.5, x[0] <= 2.5)
        in_y = x[1] >= notch_y
        return np.where(np.logical_and(in_x, in_y), 1.0, 0.0)

    phasefield.interpolate(_notch)
    phasefield.x.scatter_forward()

    coords = mesh.mesh.geometry.x
    damaged_vertices = np.logical_and.reduce(
        (
            coords[:, 0] >= 1.5,
            coords[:, 0] <= 2.5,
            coords[:, 1] >= notch_y,
        )
    )
    expected_depth = np.max(coords[:, 1]) - np.min(coords[damaged_vertices, 1])

    model = CrackDepthMeasure("CD", params, mesh)
    model.Update_Global_Measures()

    crack_depth = params.Global_Measures["Crack_Depth"]

    assert abs(crack_depth - expected_depth) < 1.0e-12


def test_crack_depth_measure_assemble_kf_is_noop():
    params = _build_params()
    mesh = Mesh(params)
    model = CrackDepthMeasure("CD", params, mesh)

    assert model.assemble_KF(model.My_Step) is None


def test_crack_depth_measure_accepts_raw_phasefield_function():
    params = _build_params()
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Phasefield", "CG", params.Field_Orders["phasefield"])

    phasefield = mesh.Get_Field("phasefield", params.Solution_Steps["phasefield"])
    notch_y = params.Ly - 0.5

    def _notch(x):
        in_x = np.logical_and(x[0] >= 1.5, x[0] <= 2.5)
        in_y = x[1] >= notch_y
        return np.where(np.logical_and(in_x, in_y), 1.0, 0.0)

    phasefield.interpolate(_notch)
    phasefield.x.scatter_forward()

    coords = mesh.mesh.geometry.x
    damaged_vertices = np.logical_and.reduce(
        (
            coords[:, 0] >= 1.5,
            coords[:, 0] <= 2.5,
            coords[:, 1] >= notch_y,
        )
    )
    expected_depth = np.max(coords[:, 1]) - np.min(coords[damaged_vertices, 1])

    model = CrackDepthMeasure("CD", params, mesh)
    crack_depth = model.Evaluate_Crack_Depth(phasefield, None)

    assert abs(crack_depth - expected_depth) < 1.0e-12


def _two_squared_crack_depth(domain=None, phasefield_domain="domain"):
    """PRIVATE: Measure a notch at the top of the right square of a two-square mesh."""
    params = Sample_Parameters()
    params.dx, params.dx_interface = 0.1, 0.1
    params.Global_Measures = {}
    params.Solution_Steps = {"phasefield": 0}
    params.Models = {"CD": {"Type": "Solid:CrackDepthMeasure", **({"domain": domain} if domain else {})}}
    mesh = Mesh(params)
    phasefield = mesh.Add_Field("phasefield", "Phasefield", "CG", 1, domain=phasefield_domain)
    phasefield.interpolate(lambda x: np.where((np.abs(x[0] - 1.5) <= 0.1) & (x[1] >= 0.7 - 1.0e-12), 1.0, 0.0))
    phasefield.x.scatter_forward()
    model = CrackDepthMeasure("CD", params, mesh)
    model.Update_Global_Measures()
    return params.Global_Measures["Crack_Depth"]


def test_crack_depth_measure_only_considers_its_domain():
    depth = _two_squared_crack_depth()

    assert 0.2 < depth <= 0.3 + 1.0e-12
    assert _two_squared_crack_depth(domain="domain:right") == depth
    assert _two_squared_crack_depth(domain="domain:left") == 0.0


def test_crack_depth_measure_reads_a_phasefield_restricted_to_a_subdomain():
    assert _two_squared_crack_depth(phasefield_domain="domain:right") == _two_squared_crack_depth()
    with pytest.raises(ValueError, match="phase field on all of domain 'domain:left'"):
        _two_squared_crack_depth(domain="domain:left", phasefield_domain="domain:right")
