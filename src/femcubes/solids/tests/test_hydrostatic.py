
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
            "Bulk": 10.0e9,        # Young's modulus (Pa)
            "Visc_Type": "Newtonian",    # Viscosity type: Newtonian, Glen, or Carreau
            "Viscosity": 1.0e13,       # dynamic viscosity (Pa.s)
            "Glen_n": 3.0,            # Glen's flow law exponent
            "Glen_A": 1e-24,         # Glen's flow law rate factor (Pa^-n s^-1)
            "visc_max": 1.0e20,       # maximum viscosity (Pa.s)
            "visc_min": 1.0e4,       # minimum viscosity (Pa.s)
        }
        gravity: float = -9.81     # gravitational acceleration (m/s^2)
        
        if include_plots:
            Plot_Fields: list = ["v_x", "v_y", "p"]
        else:
            Plot_Fields: list = []

        Solution_Steps = {"v": 0, "p": 0}
        Field_Orders = {"v": 2, "p": 1}
        
        Models = {}
        ModelNames = ["Solid", "BC_Left", "BC_Bottom", "BC_Right"]
        Models["Solid"] = {
            "Type": "Solid:Stokes",
            "Material": "Ice",
            "DamageModel": None
        }
        Models["BC_Left"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "left",
            "field": "v_x",
            "value": 0.0
        }
        
        Models["BC_Bottom"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "bottom",
            "field": "v_y",
            "value": 0.0
        }
        
        Models["BC_Right"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "right",
            "field": "v_x",
            "value": 0.0
        }
        
        ## Solver Parameters
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 20
        line_search: bool = True
        line_search_lims: list = [0.1, 1.0]
        rel_tol: float = 1e-6
        abs_tol: float = 1e3
        
        # Output parameters
        output_dir: str = None   # directory to save output files
        
        ## Time parameters
        start_time: float = 0.0      # start time (s)
        end_time: float = 3600.0*24*200 # end time (s)
        dt: float = 3600.0*24           # time step (s)
    return Params()


@pytest.fixture
def InputParams():
    return _build_params(False)
    
@pytest.mark.full_model
class TestHydroStatic:
    
    def test_hydrostatic_pressure_Newtonian(self, InputParams):
        mesh = Mesh(InputParams)
        physics = Physics(mesh, InputParams)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams, physics)
        solver.run()
        
        p_terget = -InputParams.Materials["Ice"]["Density"] * InputParams.gravity * InputParams.Ly
        p_output = mesh.Get_Field_For_Output("p")
        p_func = p_output.collapse_component(0)
        p_computed = p_func.x.array.max()
        
        err = abs(p_computed - p_terget) / abs(p_terget)
        mprint(f"Computed pressure: {p_computed}, Target pressure: {p_terget}: Error: {err}")
        assert err < 1e-2
        
    def test_hydrostatic_pressure_Glen(self, InputParams):
        InputParams.Materials["Ice"]["Visc_Type"] = "Glen"
        mesh = Mesh(InputParams)
        physics = Physics(mesh, InputParams)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams, physics)
        solver.run()
        
        p_terget = -InputParams.Materials["Ice"]["Density"] * InputParams.gravity * InputParams.Ly
        p_output = mesh.Get_Field_For_Output("p")
        p_func = p_output.collapse_component(0)
        p_computed = p_func.x.array.max()

        err = abs(p_computed - p_terget) / abs(p_terget)
        mprint(f"Computed pressure: {p_computed}, Target pressure: {p_terget}: Error: {err}")
        assert err < 1e-2
        
    def test_hydrostatic_pressure_Carreau(self, InputParams):
        InputParams.Materials["Ice"]["Visc_Type"] = "Carreau"
        mesh = Mesh(InputParams)
        physics = Physics(mesh, InputParams)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams, physics)
        solver.run()
        
        p_terget = -InputParams.Materials["Ice"]["Density"] * InputParams.gravity * InputParams.Ly
        p_output = mesh.Get_Field_For_Output("p")
        p_func = p_output.collapse_component(0)
        p_computed = p_func.x.array.max()

        err = abs(p_computed - p_terget) / abs(p_terget)
        mprint(f"Computed pressure: {p_computed}, Target pressure: {p_terget}: Error: {err}")
        
        assert err < 1e-2
        
if __name__ == "__main__":
    params = _build_params(True)
    #TestHydroStatic().test_hydrostatic_pressure_Newtonian(params)
    TestHydroStatic().test_hydrostatic_pressure_Glen(params)
    TestHydroStatic().test_hydrostatic_pressure_Carreau(params)
