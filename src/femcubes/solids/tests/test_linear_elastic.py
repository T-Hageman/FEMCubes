import pytest

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
        ## Mesh parameters
        MeshType: str = "Rectangle"
        dx: float = 10.0          # Mesh resolution in meters
        dim: int = 2               # Spatial dimension
        Lx:float = 10.0
        Ly:float = 100.0
        
        ## physics parameters
        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 917.0,        # density of ice (kg/m^3)
            "Youngs": 10.0e9,        # Young's modulus (Pa)
            "Poisson": 0.3,        # Poisson's ratio
            "LE_Damping": 1.0e6
        }
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = Materials["Ice"]["Shear"] * (2.0 * Materials["Ice"]["Poisson"]) / (1.0 - 2.0 * Materials["Ice"]["Poisson"])
        
        gravity: float = -9.81     # gravitational acceleration (m/s^2)
        
        if include_plots:
            Plot_Fields: list = ["u_x", "u_y"]
        else:
            Plot_Fields: list = []

        Solution_Steps = {"u": 0}
        Field_Orders = {"u": 2}
        
        Models = {}
        ModelNames = ["Solid", "BC_Left", "BC_Bottom", "BC_Right"]
        Models["Solid"] = {
            "Type": "Solid:LinearElastic",
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
        
        Models["BC_Right"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "right",
            "field": "u_x",
            "value": 0.0
        }
        
        ## Solver Parameters
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 100
        line_search: bool = True
        line_search_lims: list = [0.01, 1.0]
        rel_tol: float = 1e-6
        abs_tol: float = 1e-8
        
        # Output parameters
        output_dir: str = None   # directory to save output files
        
        ## Time parameters
        start_time: float = 0.0      # start time (s)
        end_time: float = 3600.0*24*10 # end time (s)
        dt: float = 3600.0*24           # time step (s)
        Newmark_Beta: float = 0.5625  # Newmark-beta parameter
        Newmark_Gamma: float = 1.0    # Newmark-gamma parameter
    
        
    return Params()

@pytest.fixture
def InputParams():
    return _build_params()

@pytest.mark.full_model
class TestLinearElastic:
    
    def test_linelastic_gravity(self, InputParams):
        mesh = Mesh(InputParams)
        physics = Physics(mesh, InputParams)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams, physics)
        solver.run()
        
        rho = InputParams.Materials["Ice"]["Density"]
        lame = InputParams.Materials["Ice"]["Lame"]
        shear = InputParams.Materials["Ice"]["Shear"]
        effective_modulus = lame + 2.0 * shear
        uy_target = rho * InputParams.gravity * InputParams.Ly * InputParams.Ly / (2.0 * effective_modulus)
        uy_func = mesh.Get_Field_For_Output("u").collapse_component(1)
        uy_computed = uy_func.x.array.min()
        
        err = abs(uy_computed - uy_target) / abs(uy_target)
        mprint(f"Computed displacement: {uy_computed}, Target displacement: {uy_target}: Error: {err}")
        assert err < 1e-2
        
if __name__ == "__main__":
    params = _build_params(True)
    TestLinearElastic().test_linelastic_gravity(params)
