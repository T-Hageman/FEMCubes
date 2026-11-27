"""Constrained solves for consistent initial potentials (StepSolver.Solve_Constrained_Step, Electrolyte.Current_Conservation_Rows)."""

import numpy as np
import pytest

import femcubes.electrochemistry  # noqa: F401 - model registration
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.solver_enums import LinearSolver
from femcubes.core.solvers.step_solver import StepSolver

D_NA, D_CL = 1.33e-9, 2.03e-9
F_RT = 96485.3329 / (8.3145 * 298.15)


def _build_params():
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        MeshType: str = "Rectangle"
        dim: int = 2
        Lx: float = 10.0e-3
        Ly: float = 1.0e-3
        dx: float = 2.5e-4
        Quadrature_Degree: int = 2
        Materials: dict = {}
        Species: dict = {"Na+": {"Diffusivity": D_NA, "Charge": 1.0, "C0": 100.0},
                         "Cl-": {"Diffusivity": D_CL, "Charge": -1.0, "C0": 100.0}}
        C_ref: float = 1000.0
        Volume_Reactions: dict = {}
        Solution_Steps = {"C": 0, "E": 0}
        Field_Orders = {"C": 1, "E": 1}
        ModelNames = ["Electrolyte", "Bulk"]
        Models = {"Electrolyte": {"Type": "ElectroChemistry:Electrolyte", "Activity_Model": "ideal"},
                  "Bulk": {"Type": "ElectroChemistry:OpenBoundary", "boundary": "left", "E": 0.0}}
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 20
        line_search: bool = False
        rel_tol: float = 1e-12
        abs_tol: float = 1e-16
        output_dir: str = None
        start_time: float = 0.0
        end_time: float = 1.0
        dt: float = 1.0
        Plot_Fields: list = []
        Plot_Measures: list = []

    return Params()


def _salt_gradient():
    """A NaCl electrolyte whose concentration triples from the left (bulk, phi = 0) to the right edge."""
    params = _build_params()
    mesh = Mesh(params)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    for name in ("Na+", "Cl-"):
        for field_name in (name, f"{name}Old"):
            field = mesh.Get_Field(field_name)
            x = field.function_space.tabulate_dof_coordinates()[:, 0]
            field.x.array[:] = 100.0 * (1.0 + 2.0 * x / params.Lx)
            field.x.scatter_forward()
    return params, mesh, physics


def test_current_conservation_rows_weigh_the_charged_species_by_their_charge():
    _params, _mesh, physics = _salt_gradient()
    assert physics.models["Electrolyte"].Current_Conservation_Rows() == {"E": {"Na+": 1.0, "Cl-": -1.0}}


def test_frozen_salt_gradient_gives_the_diffusion_potential():
    """Without current, phi follows the liquid-junction potential (D- - D+) / (D+ + D-) ln(c / c_0) / f."""
    params, mesh, physics = _salt_gradient()
    solver = StepSolver(params, physics)
    rows = physics.models["Electrolyte"].Current_Conservation_Rows()
    solver.Solve_Constrained_Step(0, ["Na+", "Cl-"], rows)

    E = mesh.Get_Field("E")
    x = E.function_space.tabulate_dof_coordinates()[:, 0]
    expected = (D_CL - D_NA) / (D_NA + D_CL) * np.log(1.0 + 2.0 * x / params.Lx) / F_RT
    np.testing.assert_allclose(E.x.array, expected, atol=2e-3 * np.abs(expected).max())
    # The concentrations are held.
    np.testing.assert_allclose(mesh.Get_Field("Na+").x.array, mesh.Get_Field("Na+Old").x.array, rtol=0, atol=0)


def test_electroneutrality_alone_does_not_determine_the_potential():
    """Without the combined rows the potential's own equation (electroneutrality) holds already, whatever phi is."""
    params, mesh, physics = _salt_gradient()
    assert StepSolver(params, physics).Solve_Constrained_Step(0, ["Na+", "Cl-"]) == 0
    np.testing.assert_array_equal(mesh.Get_Field("E").x.array, 0.0)


def test_dirichlet_rows_are_the_owned_boundary_dofs_of_each_constrained_field():
    params, mesh, physics = _salt_gradient()
    rows = physics.Get_Dirichlet_Rows(0)
    offsets = dict(zip(mesh.step_fields[0], mesh.get_step_block_offsets(0)))
    E = mesh.Get_Field("E")
    left = np.flatnonzero(np.isclose(E.function_space.tabulate_dof_coordinates()[:, 0], 0.0))
    left = left[left < E.function_space.dofmap.index_map.size_local]
    for name in ("Na+", "Cl-", "E"):
        assert set((offsets[name] + left).tolist()) <= set(rows.tolist())
    assert rows.size == 3 * left.size


@pytest.mark.parametrize(
    ("frozen", "rows", "message"),
    [
        pytest.param(["Na+", "Cl-", "E"], None, "nothing to solve", id="all-frozen"),
        pytest.param(["Na+", "E"], {"E": {"Na+": 1.0}}, "frozen fields", id="frozen-target"),
        pytest.param(["Na+", "theta"], None, "not solved in step", id="unknown-field"),
    ],
)
def test_constrained_solve_rejects_inconsistent_fields(frozen, rows, message):
    params, _mesh, physics = _salt_gradient()
    with pytest.raises(ValueError, match=message):
        StepSolver(params, physics).Solve_Constrained_Step(0, frozen, rows)


def test_non_finite_trial_residual_halves_the_update():
    """An update into overflow (non-finite residual) is halved; the solve still reaches the diffusion potential."""
    params, mesh, physics = _salt_gradient()
    solver = StepSolver(params, physics)
    original = physics.Get_Fvec
    calls = {"n": 0}

    def overflowing_once(step):
        F = original(step)
        calls["n"] += 1
        if calls["n"] == 1:
            F.set(np.nan)
        return F

    physics.Get_Fvec = overflowing_once
    solver.Solve_Constrained_Step(0, ["Na+", "Cl-"], physics.models["Electrolyte"].Current_Conservation_Rows())
    assert calls["n"] >= 2
    E = mesh.Get_Field("E")
    x = E.function_space.tabulate_dof_coordinates()[:, 0]
    expected = (D_CL - D_NA) / (D_NA + D_CL) * np.log(1.0 + 2.0 * x / params.Lx) / F_RT
    np.testing.assert_allclose(E.x.array, expected, atol=2e-3 * np.abs(expected).max())
