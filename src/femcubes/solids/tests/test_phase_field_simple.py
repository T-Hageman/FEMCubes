import pytest

from ufl import Measure
from dolfinx import fem

from femcubes.core.solvers.solver_enums import LinearSolver

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.core.solvers.time_solver import TimeSolver
from femcubes.core.utils.mpi_utils import mprint
from femcubes.core.utils.maths_utils import ft_to_Gc
import femcubes.solids
import femcubes.boundary
from femcubes.core.params_template import ParamsTemplate

def _build_params(pf_model_in = "AT2", include_plots: bool = False):
    class Params(ParamsTemplate):
        Global_Measures: dict = {}
        
        pf_model = pf_model_in
        Cohesive = False
        if pf_model == "Cohesive_AT2":
            pf_model = "AT2"
            Cohesive = True
            
        ## Mesh parameters
        if Cohesive == True: # Cohesive-type models don't deal well with cracks along a symmetry boundary, especially in compression
            MeshType: str = "FullPlateWithHole"
            R:float = 1.25
            dx: float = 0.4          # Mesh resolution in meters
        else:
            MeshType: str = "PlateWithHole"
            R:float = 2.5
            dx: float = 0.2          # Mesh resolution in meters
        dim: int = 2               # Spatial dimension
        Lx:float = 10.0
        Ly:float = 10.0
        
        ## physics parameters
        Materials: dict = {}
        Materials["Ice"] = {
            "Density": 917.0,        # density of ice (kg/m^3)
            "Youngs": 10.0e9,        # Young's modulus (Pa)
            "Poisson": 0.3,        # Poisson's ratio
            "LE_Damping": 1.0e3,
            "ell": 3*dx,              # Length scale parameter (m)
            "f_t": 2.0e5,           # Tensile strength (Pa)
            "f_s": 4.0e5,           # Shear strength (Pa
            "Gc": 1.0,             # Fracture energy (J/m^2)
            "pf_damping": 0.0e-5,
            "CrackDrivingForce": "Stress-Based",
            "FailureType": "r1",
            "PF_formulation": pf_model,
            "Inertia": True,
            "kappa": 1.0e-6
        }
        Materials["Ice"]["Shear"] = Materials["Ice"]["Youngs"] / (2.0 * (1.0 + Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Bulk"] = Materials["Ice"]["Youngs"] / (3.0 * (1.0 - 2.0 * Materials["Ice"]["Poisson"]))
        Materials["Ice"]["Lame"] = Materials["Ice"]["Shear"] * (2.0 * Materials["Ice"]["Poisson"]) / (1.0 - 2.0 * Materials["Ice"]["Poisson"])
        
        if Cohesive:
            # The cohesive solid's damping coefficient (Pa s/m^2) is not multiplied by the density
            Materials["Ice"]["LE_Damping"] *= Materials["Ice"]["Density"]
        if Cohesive == False:
            Materials["Ice"]["Gc"] = ft_to_Gc(Materials["Ice"]["f_t"], Materials["Ice"]["Youngs"], Materials["Ice"]["ell"])

        gravity: float = 0.0     # gravitational acceleration (m/s^2)
        
        if include_plots:
            Plot_Fields: list = ["u_x", "u_y", "phasefield"]
            Plot_Measures: list = [("t", "Crack_Length"), ("BC_Top_u", "BC_Top_F")]
        else:
            Plot_Fields: list = []
            Plot_Measures: list = []

        Solution_Steps = {"u": 0, "phasefield": 1}
        Field_Orders = {"u": 2, "phasefield": 2}
        
        Models = {}
        ModelNames = ["Solid", "pf", "BC_Bottom", "BC_Top", "BC_Left"]
        
        if Cohesive:
            Models["Solid"] = {
                "Type": "Solid:Cohesive_LinearElastic",
                "Material": "Ice",
                "DamageModel": "PhaseField"
            }
            Models["pf"] = {
                "Type": "Solid:PhaseField",
                "Material": "Ice",
                "CDF_field": "PsiField",
                "Irreversible_Method": "AL"
            }
        else:
            Models["Solid"] = {
                "Type": "Solid:LinearElastic",
                "Material": "Ice",
                "DamageModel": "PhaseField"
            }
            Models["pf"] = {
                "Type": "Solid:PhaseField",
                "Material": "Ice",
                "CDF_field": "ElasticEnergy",
                "Irreversible_Method": "AL"
            }
        if pf_model == "AT2":
            Models["pf"]["Irreversible_Method"] = "Hist" # test history field irreversibility method for AT2 as well
            
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
            "rate": 1.0e-6,
            "Dummy": 1.0e10,
            "F_Cutoff": 0.05
        }
        
        if Cohesive:
            ModelNames.append("BC_Bottom2")
            Models["BC_Bottom2"] = {
                "Type": "Boundary:PrescribedValue",
                "boundary": "bottom",
                "field": "u_x",
                "value": 0.0
            }
        
        ## Solver Parameters
        Linear_Solver: LinearSolver = LinearSolver.LU
        n_passes: int = 3
        max_it: int = 5
        line_search: bool = Cohesive
        line_search_lims: list = [0.1, 1.0]
        rel_tol: float = 1e0
        abs_tol: float = 1e0
        
        # Output parameters
        output_dir: str = None   # directory to save output files
        
        ## Time parameters
        start_time: float = 0.0      # start time (s)
        end_time: float = 1200.0 # end time (s)
        dt: float = 1.0           # time step (s)
        Newmark_Beta: float = 0.5625  # Newmark-beta parameter
        Newmark_Gamma: float = 1.0    # Newmark-gamma parameter
        TimeStepControl = {
            "Type": "Measure",
            "time_list": [0.0],
            "dt_list": [1.0],
            "dt_min": 1.0e-10,
            "dt_max": 10.0,
            "Measure": "Crack_Length_Change",
            "Target_Change": dx
        }
        
    return Params()

@pytest.fixture
def InputParams_AT1():
    return _build_params(pf_model_in="AT1")

@pytest.fixture
def InputParams_AT2():
    return _build_params(pf_model_in="AT2")

@pytest.fixture
def InputParams_Cohesive_AT2():
    return _build_params(pf_model_in="Cohesive_AT2")

@pytest.fixture
def InputParams_HO_AT1():
    return _build_params(pf_model_in="HO-AT1")


@pytest.fixture
def InputParams_HO_AT2():
    return _build_params(pf_model_in="HO-AT2")


@pytest.mark.full_model
class TestPhaseField_Simple:

    def _run_tension_case(self, params):
        mesh = Mesh(params)
        physics = Physics(mesh, params)
        physics.Initialize_Fields()
        solver = TimeSolver(params, physics)
        solver.run()

        F_max = params.Global_Measures["BC_Top_F_max"]
        stress = F_max / params.Lx
        stressFactor = params.Materials["Ice"]["f_t"] / stress
        mprint(f"Max Stress: {stress*1.0e-6:.2e} MPa, Stress Factor: {stressFactor:.4f}")

        return stressFactor

    def test_linelastic_AT1_tension(self, InputParams_AT1):
        stressFactor = self._run_tension_case(InputParams_AT1)
        assert stressFactor > 2.0 and stressFactor < 4.0, "Tensile strength not reached as expected" #assuming stress factor for infinite plate, 3.0

    def test_linelastic_AT2_tension(self, InputParams_AT2):
        stressFactor = self._run_tension_case(InputParams_AT2)
        assert stressFactor > 2.0 and stressFactor < 4.0, "Tensile strength not reached as expected" 
        
    def test_linelastic_HO_AT1_tension(self, InputParams_HO_AT1):
        stressFactor = self._run_tension_case(InputParams_HO_AT1)
        assert stressFactor > 2.0 and stressFactor < 4.0, "Tensile strength not reached as expected"

    def test_linelastic_HO_AT2_tension(self, InputParams_HO_AT2):
        stressFactor = self._run_tension_case(InputParams_HO_AT2)
        assert stressFactor > 2.0 and stressFactor < 4.0, "Tensile strength not reached as expected"




    def test_cohesive_AT2_tension(self, InputParams_Cohesive_AT2):
        stressFactor = self._run_tension_case(InputParams_Cohesive_AT2) # strict range for cohesive, as it should be spot-on (mesh resolution has slight impact)
        assert stressFactor > 2.8 and stressFactor < 3.2, "Tensile strength not reached as expected" 
        

    def test_linelastic_AT1_Compression(self, InputParams_AT1):
        InputParams_AT1.Models["BC_Top"]["rate"] = -50.0*InputParams_AT1.Models["BC_Top"]["rate"]
        InputParams_AT1.end_time = 25.0
        mesh = Mesh(InputParams_AT1)
        physics = Physics(mesh, InputParams_AT1)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams_AT1, physics)
        solver.run()

        # Verify that fracture does not occur along the bottom edge
        boundary_tags = mesh.get_boundary_tag("bottom")
        ds = Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
        phasefield = mesh.Get_Field("phasefield", InputParams_AT1.Solution_Steps["phasefield"])
        dS = ds(boundary_tags)
        pf_int = fem.assemble_scalar(fem.form(phasefield * dS))
        area = fem.assemble_scalar(fem.form(1.0 * dS))
        mprint(f"Integrated phase field along bottom edge: {pf_int:.4f}, Area: {area:.4f}")
        phase_average = pf_int / area
        mprint(f"Average phase field along bottom edge: {phase_average:.4f}")
        assert phase_average < 0.1, "Unexpected fracture along the bottom edge in compression"
        
    def test_linelastic_AT2_Compression(self, InputParams_AT2):
        InputParams_AT2.Models["BC_Top"]["rate"] = -50.0*InputParams_AT2.Models["BC_Top"]["rate"]
        InputParams_AT2.end_time = 25.0
        mesh = Mesh(InputParams_AT2)
        physics = Physics(mesh, InputParams_AT2)
        physics.Initialize_Fields()
        solver = TimeSolver(InputParams_AT2, physics)
        solver.run()

        # Verify that fracture does not occur along the bottom edge
        boundary_tags = mesh.get_boundary_tag("bottom")
        ds = Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
        phasefield = mesh.Get_Field("phasefield", InputParams_AT2.Solution_Steps["phasefield"])
        dS = ds(boundary_tags)
        pf_int = fem.assemble_scalar(fem.form(phasefield * dS))
        area = fem.assemble_scalar(fem.form(1.0 * dS))
        mprint(f"Integrated phase field along bottom edge: {pf_int:.4f}, Area: {area:.4f}")
        phase_average = pf_int / area
        mprint(f"Average phase field along bottom edge: {phase_average:.4f}")
        assert phase_average < 0.1, "Unexpected fracture along the bottom edge in compression"
        
if __name__ == "__main__":
    ## Standard phase-field models
    params = _build_params("AT1", True)
    TestPhaseField_Simple().test_linelastic_AT1_tension(params)
    params = _build_params("AT2", True)
    TestPhaseField_Simple().test_linelastic_AT2_tension(params)
    params = _build_params("HO-AT1", True)
    TestPhaseField_Simple().test_linelastic_HO_AT1_tension(params)
    params = _build_params("HO-AT2", True)
    TestPhaseField_Simple().test_linelastic_HO_AT2_tension(params)
    params = _build_params("AT1", True)
    TestPhaseField_Simple().test_linelastic_AT1_Compression(params)
    params = _build_params("AT2", True)
    TestPhaseField_Simple().test_linelastic_AT2_Compression(params)
    
    ## Cohesive-type phase-field models
    params = _build_params("Cohesive_AT2", True)
    TestPhaseField_Simple().test_cohesive_AT2_tension(params)
