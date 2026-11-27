import pytest
import numpy as np

from femcubes.core.solvers.solver_enums import LinearSolver

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.utils.mpi_utils import mprint
import femcubes.solids
import femcubes.boundary
from femcubes.core.params_template import ParamsTemplate

def _build_params(include_plots: bool = False):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        
        ## Mesh parameters
        MeshType: str = "Rectangle"
        dx: float = 0.25          # Mesh resolution in meters
        dim: int = 2               # Spatial dimension
        Lx:float = 1.0
        Ly:float = 10.0
        Quadrature_Degree: int = 2
        
        ## physics parameters
        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 917.0,        # density of ice (kg/m^3)
            "Youngs": 10.0e9,        # Young's modulus (Pa)
            "Poisson": 0.4999,        # Poisson's ratio
            "LE_Damping": 1.0e6,
            "Visc_Type": "Glen",    # Viscosity type: Newtonian, Glen, or Carreau
            "Viscosity": 1.0e12,       # dynamic viscosity (Pa.s) (used in newtonian case)
            "Glen_n": 3.0,            # Glen's flow law exponent
            "Glen_A": 1e-24,         # Glen's flow law rate factor (Pa^-n s^-1)
        }
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = Materials["Ice"]["Shear"] * (2.0 * Materials["Ice"]["Poisson"]) / (1.0 - 2.0 * Materials["Ice"]["Poisson"])
        
        gravity: float = 0.0     # gravitational acceleration (m/s^2)
        
        if include_plots:
            Plot_Fields: list = ["u_x", "u_y"]
            Plot_Measures: list = [("BC_Top_u", "BC_Top_F")]
        else:
            Plot_Fields: list = []

        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 2}
        
        Models = {}
        ModelNames = ["Solid", "BC_Bottom", "BC_Left", "BC_Top"]
        Models["Solid"] = {
            "Type": "Solid:ViscoElastic",
            "Material": "Ice",
            "DamageModel": None
        }
        Models["BC_Left"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "left",
            "field": "u_x",
            "value": 0.0
        }
        
        Models["BC_Bottom"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "u_y",
            "value": 0.0
        }
        
        Models["BC_Top"] = {
            "Type": "Boundary:DisplacementControl",
            "boundary": "top",
            "field": "u_y",
            "rate": 1.0e-6*Ly,
            "Dummy": 1.0e14
        }
        
        ## Solver Parameters
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 100
        line_search: bool = False
        line_search_lims: list = [0.01, 1.0]
        rel_tol: float = 1e-3
        abs_tol: float = 1e-2
        
        # Output parameters
        output_dir: str = None   # directory to save output files
        
        ## Time parameters
        start_time: float = 0.0      # start time (s)
        end_time: float = 100.0*50 # end time (s)
        dt: float = 100.0           # time step (s)
        Newmark_Beta: float = 0.5625  # Newmark-beta parameter
        Newmark_Gamma: float = 1.0    # Newmark-gamma parameter
    
        
    return Params()

@pytest.fixture
def InputParams():
    return _build_params()

@pytest.mark.full_model
class TestViscoElastic:
    
    def test_viscoelastic_Glen(self, InputParams):
        mesh = Mesh(InputParams)
        physics = Physics(mesh, InputParams)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams, physics)
        solver.run()

        # Compare final reaction force to the analytic plane-strain J2 creep
        # response used by the viscoelastic kernel.  In the 2-D solid model
        # e_zz = 0, and with the free lateral traction on the right edge the
        # homogeneous bar solution satisfies sigma_xx = 0. For this setting
        # the kernel's flow rule gives:
        #
        #   sigma_yy = 2 * eta * (2 * eps_dot)^(1/n)
        #
        # with eta = (1 / (2 A))^(1/n), which simplifies to:
        #
        #   sigma_yy = 2 * (eps_dot / A)^(1/n)
        #
        # The top reaction force is then sigma_yy * Lx.
        glen_a = InputParams.Materials["Ice"]["Glen_A"]
        glen_n = InputParams.Materials["Ice"]["Glen_n"]
        strain_rate = InputParams.Models["BC_Top"]["rate"] / InputParams.Ly
        sigma_yy = 2.0 * (strain_rate / glen_a) ** (1.0 / glen_n)
        force_expected = sigma_yy * InputParams.Lx
        force_measured = abs(InputParams.Global_Measures.get("BC_Top_F", 0.0))
        rel_err = abs(force_measured - force_expected) / abs(force_expected)
        mprint(
            f"Final force comparison (plane strain): "
            f"measured={force_measured:.6e}, expected={force_expected:.6e}, "
            f"rel_err={rel_err:.3e}"
        )
        assert rel_err < 0.1
        
    def test_viscoelastic_Newtonian(self, InputParams):
        # Modify InputParams to be Newtonian
        InputParams.Materials["Ice"]["Visc_Type"] = "Newtonian"
        mesh = Mesh(InputParams)
        physics = Physics(mesh, InputParams)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams, physics)
        solver.run()

        # For n = 1 the same plane-strain J2 creep solution reduces to:
        #
        #   sigma_yy = 4 * eta * eps_dot
        #
        # so the top reaction force is sigma_yy * Lx.
        viscosity = InputParams.Materials["Ice"]["Viscosity"]
        strain_rate = InputParams.Models["BC_Top"]["rate"] / InputParams.Ly
        sigma_yy = 4.0 * viscosity * strain_rate
        force_expected = sigma_yy * InputParams.Lx
        force_measured = abs(InputParams.Global_Measures.get("BC_Top_F", 0.0))
        rel_err = abs(force_measured - force_expected) / abs(force_expected)
        mprint(
            f"Final force comparison (plane strain, Newtonian): "
            f"measured={force_measured:.6e}, expected={force_expected:.6e}, "
            f"rel_err={rel_err:.3e}"
        )
        assert rel_err < 0.1
        
if __name__ == "__main__":
    params = _build_params(True)
    TestViscoElastic().test_viscoelastic_Newtonian(params)
    params = _build_params(True)
    TestViscoElastic().test_viscoelastic_Glen(params)
