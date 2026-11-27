"""Physics-level tests of multi-domain models on a notched tensile bar."""

import numpy as np
import pytest

import femcubes.boundary  # noqa: F401 - ensure model registration
import femcubes.solids  # noqa: F401 - ensure model registration
from femcubes import LinearSolver, Mesh, ParamsTemplate, Physics, TimeSolver
from femcubes.core.utils.maths_utils import ft_to_Gc


def _build_params(case: str):
    """PRIVATE: a coarse, quickly damaging notched steel bar.

    ``case="multi"`` splits the bar into three subdomains, with the phase field
    only in ``domain:centre``; ``"single"`` uses one damaged elastic model and
    the phase field on the full mesh, which is the reference the multi-domain
    set-up must reproduce.
    """
    if case not in ("multi", "single"):
        raise ValueError(f"Unknown case: {case}")

    class Params(ParamsTemplate):
        Global_Measures: dict = {}

        MeshType: str = "NotchedTensileBar"
        dim: int = 2
        Lx: float = 0.1
        Ly: float = 0.02
        Lx_centre: float = 0.02
        Notch_Width: float = 1.0e-3
        Notch_Depth: float = 4.0e-3
        Notch_Sides: str = "both"
        dx: float = 4.0e-3
        dx_centre: float = 1.0e-3
        Quadrature_Degree: int = 3

        Materials: dict = {}
        Materials["Steel"] = {"Density": 7.85e3, "Youngs": 210.0e9, "Poisson": 0.3, "Inertia": False, "ell": 2.0e-3,
                              "f_t": 100.0e6, "kappa": 1.0e-6, "PF_formulation": "AT2", "pf_damping": 0.0}
        steel = Materials["Steel"]
        steel["Shear"] = steel["Youngs"] / (2.0 * (1.0 + steel["Poisson"]))
        steel["Lame"] = steel["Shear"] * (2.0 * steel["Poisson"]) / (1.0 - 2.0 * steel["Poisson"])
        steel["Gc"] = ft_to_Gc(steel["f_t"], steel["Youngs"], steel["ell"], steel["PF_formulation"])
        gravity: float = 0.0

        Solution_Steps = {"u": 0, "phasefield": 1}
        Field_Orders = {"u": 2, "phasefield": 2}
        boundaries = {
            "BC_Left_x": {"Type": "Boundary:PrescribedValue", "boundary": "left", "field": "u_x", "value": 0.0,
                          "domain": "domain:left"},
            "BC_Left_y": {"Type": "Boundary:PrescribedValue", "boundary": "left", "field": "u_y", "value": 0.0,
                          "domain": "domain:left"},
            "BC_Right": {"Type": "Boundary:DisplacementControl", "boundary": "right", "field": "u_x", "rate": 1.0e-6,
                         "Dummy": 1.0e16, "F_Cutoff": 0.05, "domain": "domain:right"},
        }
        pf = {"Type": "Solid:PhaseField", "Material": "Steel", "CDF_field": "ElasticEnergy", "Irreversible_Method": "Hist"}
        if case == "multi":
            Models: dict = {
                "Solid_Left": {"Type": "Solid:LinearElastic", "Material": "Steel", "domain": "domain:left"},
                "Solid_Right": {"Type": "Solid:LinearElastic", "Material": "Steel", "domain": "domain:right"},
                "Solid_Centre": {"Type": "Solid:LinearElastic", "Material": "Steel", "DamageModel": "PhaseField",
                                 "domain": "domain:centre"},
                "PF_Centre": {**pf, "domain": "domain:centre"},
                **boundaries,
            }
        else:
            Models: dict = {
                "Solid": {"Type": "Solid:LinearElastic", "Material": "Steel", "DamageModel": "PhaseField"},
                "PF": {**pf, "domain": "domain"},
                **boundaries,
            }
        ModelNames: list = list(Models)

        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 3
        max_it: int = 50
        line_search: bool = True
        line_search_lims: list = [1e-2, 1.0]
        rel_tol: float = 1e-3
        abs_tol: float = 1.0

        Newmark_Beta: float = 0.5625
        Newmark_Gamma: float = 1.0
        start_time: float = 0.0
        end_time: float = 15.0
        dt: float = 5.0
        TimeStepControl = {"Type": "Measure", "time_list": [0.0], "dt_list": [dt], "dt_min": 1.0e-8, "dt_max": dt,
                           "Measure": "Crack_Length_Change", "Target_Change": steel["ell"]}

        output_dir: str = None
        write_vtk: bool = False
        write_hdf5: bool = False
        Plot_Fields: list = []
        Plot_Measures: list = []

    return Params()


def _run(params):
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    TimeSolver(params, physics).run()
    return mesh


def test_models_reference_existing_domains_and_boundaries():
    # Every model names the subdomain it acts on; those names, and the
    # boundaries of the boundary conditions, must exist on the generated mesh,
    # with each boundary lying on its model's domain.
    params = _build_params("multi")
    mesh = Mesh(params)
    tdim = mesh.mesh.topology.dim
    mesh.mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_to_cell = mesh.mesh.topology.connectivity(tdim - 1, tdim)

    for name in params.ModelNames:
        model = params.Models[name]
        domain_cells = mesh.get_domain_cells(model["domain"])
        assert domain_cells.size > 0, name
        if "boundary" in model:
            facets = mesh.get_boundary_facets(model["boundary"])
            assert facets.size > 0, name
            cells = np.concatenate([facet_to_cell.links(facet) for facet in facets])
            assert np.all(np.isin(cells, domain_cells)), name


@pytest.mark.full_model
def test_multidomain_tensile_bar_only_creates_phasefield_dofs_in_the_centre():
    params = _build_params("multi")
    mesh = Mesh(params)
    Physics(mesh, params)

    assert mesh.Get_Field_Support("u").is_full
    for field_name in ("phasefield", "phasefield_Hist", "ElasticEnergy"):
        support = mesh.Get_Field_Support(field_name)
        assert support.domain_names == ("domain:centre",)
        assert not support.is_full
    centre_cells = mesh.get_domain_cells("domain:centre").size
    assert mesh.fields["phasefield"].function_space.mesh.topology.index_map(2).size_local <= centre_cells


@pytest.mark.full_model
def test_multidomain_tensile_bar_reproduces_the_single_domain_response():
    # Before the crack reaches the domain interfaces, restricting the phase
    # field to the centre must not change the response.
    multi = _build_params("multi")
    single = _build_params("single")
    _run(multi)
    _run(single)

    multi_force, single_force = multi.Global_Measures["BC_Right_F"], single.Global_Measures["BC_Right_F"]
    multi_crack, single_crack = multi.Global_Measures["Crack_Length"], single.Global_Measures["Crack_Length"]
    assert multi_crack > 0.0
    assert np.isclose(multi_force, single_force, rtol=1.0e-6)
    assert np.isclose(multi_crack, single_crack, rtol=1.0e-3)
