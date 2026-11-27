import pytest

from femcubes.core.solvers.solver_enums import LinearSolver

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.utils.mpi_utils import mprint
import femcubes.solids
import femcubes.boundary
import femcubes.poromechanics
from femcubes.core.params_template import ParamsTemplate

def _build_params(Monolithic: bool = False, include_plots: bool = False):
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
            "Biot_n": 2,                 # Biot exponent
            "poros_crit": 0.8,            # critical porosity (–)
            "permeability0": 1.0e-12,  # reference permeability (m^2)
            "permeability_exp": 3.0    # permeability exponent (–)
        }
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = Materials["Ice"]["Shear"] * (2.0 * Materials["Ice"]["Poisson"]) / (1.0 - 2.0 * Materials["Ice"]["Poisson"])
        
        Materials["Water"] = {
            "Density": 1000.0,       # density of water (kg/m^3)
            "Bulk": 2.2e9,           # bulk modulus of water (Pa)
            "Viscosity": 1.0e-3      # dynamic viscosity of water (Pa.s)
        }
        
        gravity: float = -9.81     # gravitational acceleration (m/s^2)
        
        if include_plots:
            Plot_Fields: list = ["u_x", "u_y", "pw", "qw_x", "qw_y"]
        else:
            Plot_Fields: list = []

        if Monolithic:
            Solution_Steps = {"u": 0, "pw": 0}
        else:
            Solution_Steps = {"u": 0, "pw": 1}
        Field_Orders = {"u": 2, "pw": 1}
        
        Models = {}
        ModelNames = ["Solid", "porous", "BC_Left", "BC_Bottom", "BC_Right", "BC_Top"]
        Models["Solid"] = {
            "Type": "Solid:LinearElastic",
            "Material": "Ice",
            "DamageModel": None
        }
        Models["porous"] = {
            "Type": "PoroMechanics:PorePressure",
            "Material": "Ice",
            "Fluid": "Water",
            "Initial_Porosity": 0.1,
            "Saturated": True      # whether the material is saturated or not
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
        
        Models["BC_Top"] = {
            "Type": "Boundary:PrescribedValue",
            "boundary": "top",
            "field": "pw",
            "value": 0.0
        }
        
        ## Solver Parameters
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 100
        line_search: bool = False
        line_search_lims: list = [0.01, 1.0]
        rel_tol: float = 1e-6
        abs_tol: float = 1e-8
        
        # Output parameters
        output_dir: str = None   # directory to save output files
        
        ## Time parameters
        start_time: float = 0.0      # start time (s)
        end_time: float = 3600*1000 # end time (s)
        dt: float = 3600.0*10           # time step (s)
        Newmark_Beta: float = 0.5625  # Newmark-beta parameter
        Newmark_Gamma: float = 1.0    # Newmark-gamma parameter
    
        
    return Params()

@pytest.fixture
def InputParams_Monolithic():
    return _build_params(Monolithic = True)

@pytest.fixture
def InputParams_Staggered():
    return _build_params(Monolithic = False)

@pytest.mark.full_model
class TestPoroElastic:
    
    def test_poroelastic_saturated(self, InputParams_Monolithic, InputParams_Staggered):
        for input_params in [InputParams_Monolithic, InputParams_Staggered]:
            mesh = Mesh(input_params)
            physics = Physics(mesh, input_params)
            physics.Initialize_Fields()
            solver = TimeSolver(input_params, physics)
            solver.run()
        
            p_func = mesh.Get_Field_For_Output("pw").field
            pMax = p_func.x.array.max()
            
            u_func = mesh.Get_Field_For_Output("u").collapse_component(1)
            uy_min = u_func.x.array.min()
        
            mprint(f"Max pore pressure: {pMax:.4f} Pa")
            mprint(f"Min vertical displacement: {uy_min:.6f} m")
            
            # Hydrostatic pressure analytic solution at bottom
            p_analytic = -input_params.Materials["Water"]["Density"] * input_params.gravity * input_params.Ly
            
            # Vertical displacement at the top due to gravity of ice
            poros = input_params.Models["porous"]["Initial_Porosity"]
            Biot = 1.0-(1.0-poros/input_params.Materials["Ice"]["poros_crit"])**input_params.Materials["Ice"]["Biot_n"]
            rho_mean = input_params.Materials["Ice"]["Density"] + poros * input_params.Materials["Water"]["Density"]
            uy_analytic = (rho_mean - Biot * input_params.Materials["Water"]["Density"]) * input_params.gravity * input_params.Ly**2 / (2.0 * (input_params.Materials["Ice"]["Lame"]  + 2.0 * input_params.Materials["Ice"]["Shear"]))
            
            mprint(f"Analytic max pore pressure: {p_analytic:.4f} Pa")
            mprint(f"Analytic min vertical displacement: {uy_analytic:.6f}")
            
            assert abs(pMax - p_analytic) / abs(p_analytic) < 0.01, "Pore pressure does not match analytic solution within 10%."
            assert abs(uy_min - uy_analytic) / abs(uy_analytic) < 0.01, "Vertical displacement does not match analytic solution within 10%."
        
        
if __name__ == "__main__":
    params = _build_params(Monolithic=True, include_plots=False)
    params2= _build_params(Monolithic=False, include_plots=False)
    TestPoroElastic().test_poroelastic_saturated(params, params2)
