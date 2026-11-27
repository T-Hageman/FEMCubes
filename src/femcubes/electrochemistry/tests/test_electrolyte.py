import pytest

from ufl import Measure
from dolfinx import fem

from femcubes.core.solvers.solver_enums import LinearSolver

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.utils.mpi_utils import mprint
import femcubes.electrochemistry
from femcubes.core.params_template import ParamsTemplate

def _build_params(include_plots: bool = False):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        
        ## Mesh parameters
        MeshType: str = "Rectangle"
        dx: float = 1.0e-4          # Mesh resolution in meters
        dim: int = 2               # Spatial dimension
        Lx:float = 10.0e-3
        Ly:float = 1.0e-3
        Quadrature_Degree: int = 3
        
        ## physics parameters
        Materials: dict = {}
        Species: dict = {}

        pH = 7.0 
        Species["Na+"] = {
            "Diffusivity": 1.33e-9,    # Diffusivity of Na+ in water (m^2/s)
            "Charge": 1.0,              # Charge of Na+ ion
            "C0": "Neutrality"             # Initial concentration of Na+ (mol/m^3)
        }
        Species["Cl-"] = {
            "Diffusivity": 2.03e-9,    # Diffusivity of Cl- in water (m^2/s)
            "Charge": -1.0,             # Charge of Cl- ion
            "C0": 1000.0             # Initial concentration of Cl- (mol/m^3)
        }
        Species["H+"] = {
            "Diffusivity": 9.31e-9,    # Diffusivity of H+ in water (m^2/s)
            "Charge": 1.0,              # Charge of H+ ion
            "C0": 10**(3-pH)             # Initial concentration of H+ (mol/m^3)
        }
        Species["OH-"] = {
            "Diffusivity": 5.27e-9,    # Diffusivity of OH- in water (m^2/s)
            "Charge": -1.0,             # Charge of OH- ion
            "C0": 10**(3+pH-14)             # Initial concentration of OH- (mol/m^3)
        }
        Species["Fe2+"] = {
            "Diffusivity": 7.2e-10,    # Diffusivity of Fe2+ in water (m^2/s)
            "Charge": 2.0,              # Charge of Fe2+ ion
            "C0": 1.0e-3             # Initial concentration of Fe2+ (mol/m^3)
        }
        Species["FeOH+"] = {
            "Diffusivity": 6.5e-10,    # Diffusivity of FeOH+ in water (m^2/s)
            "Charge": 1.0,              # Charge of FeOH+ ion
            "C0": 1.0e-3             # Initial concentration of FeOH+ (mol/m^3)
        }
        
        C_ref = 1000.0  # Reference concentration (mol/m^3)
        Volume_Reactions = {}
        Volume_Reactions["pH_Buffer"] = {
            "Type": "Equilibrium",
            "K_eq": 1.0e-14,  # Equilibrium constant for water autoionization
            "k_dummy": 1.0e6,  # Dummy rate constant (convert from Equilibrium to kinetic)
            "species_in": ["H+", "OH-"],
            "stoich_in": [1.0, 1.0],
            "species_out": [],
            "stoich_out": []
        }
        Volume_Reactions["Fe_Hydrolysis"] = {
            "Type": "Kinetic",
            "k_fwd": 1.0e-3,    # Forward rate constant (1/s)
            "k_rev": 0.0,    # Reverse rate constant (1/s)
            "species_in": ["Fe2+"], # + H2O (solvent, not tracked)
            "stoich_in": [1.0],
            "species_out": ["FeOH+", "H+"],
            "stoich_out": [1.0, 1.0]
        }
        Volume_Reactions["FeOH_Hydrolysis"] = {
            "Type": "Kinetic",
            "k_fwd": 1.0e-9,    # Forward rate constant (1/s)
            "k_rev": 0.0,    # Reverse rate constant (1/s)
            "species_in": ["FeOH+"], # + H2O (solvent, not tracked)
            "stoich_in": [1.0],
            "species_out": ["H+"], # + FeOH2 (solid, not tracked)
            "stoich_out": [1.0]
        }
        
        
        
        


        gravity: float = 0.0     # gravitational acceleration (m/s^2)
        
        if include_plots:
            Plot_Fields: list = ["H+", "OH-", "Na+", "Cl-", "E", "Fe2+", "FeOH+"]
            LogPlot: list = [False, False, False, False, False, False, False]
            Plot_Measures: list = []
        else:
            Plot_Fields: list = []
            Plot_Measures: list = []

        Solution_Steps = {"C": 0, "E": 0}
        Field_Orders = {"C": 2, "E": 2}
        
        Models = {}
        ModelNames = ["Electrolyte", "BC_Left"]
        Models["Electrolyte"] = {
            "Type": "ElectroChemistry:Electrolyte",
            "Activity_Model": "ideal",
        }
        Models["BC_Left"] = {
            "Type": "ElectroChemistry:OpenBoundary",
            "boundary": "left",
            "E": 0.0
        }
        
        ## Solver Parameters
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 1
        max_it: int = 100
        line_search: bool = False
        line_search_lims: list = [0.1, 1.0]
        rel_tol: float = 1e-6
        abs_tol: float = 1e-18
        
        # Output parameters
        output_dir: str = None   # directory to save output files
        
        ## Time parameters
        start_time: float = 0.0      # start time (s)
        end_time: float = 24*3600.0 # end time (s)
        dt: float = 60.0           # time step (s)
        Newmark_Beta: float = 0.5625  # Newmark-beta parameter
        Newmark_Gamma: float = 1.0    # Newmark-gamma parameter
        TimeStepControl = {
            "Type": "None"
        }
        
    return Params()

@pytest.fixture
def InputParams():
    return _build_params()

@pytest.mark.full_model
class TestElectrolyte_Simple:
    
    def test_Electrolyte(self, InputParams):
        # Still WIP
        pass
    
        # mesh = Mesh(InputParams)
        # mesh.plot_mesh()
        # physics = Physics(mesh, InputParams)
        # physics.Initialize_Fields()
        # solver = TimeSolver(InputParams, physics)
        # solver.run()
    
        
if __name__ == "__main__":
    params = _build_params(True)
    TestElectrolyte_Simple().test_Electrolyte(params)
