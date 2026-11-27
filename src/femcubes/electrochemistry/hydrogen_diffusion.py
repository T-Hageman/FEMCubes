from ufl import grad, inner
from dolfinx import fem
from mpi4py import MPI

from femcubes.core.base_model import BaseModel
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm


class HydrogenDiffusion(BaseModel):
    r"""Diffusion of interstitial lattice hydrogen in a metal (Fick's law, backward Euler).

    Hydrogen absorbed at the surface of a metal, for example during hydrogen
    evolution in an electrolyte :cite:p:`Hageman2022a`, occupies interstitial
    lattice sites and diffuses into the metal. The model describes this
    diffusion of the lattice hydrogen concentration :math:`C_L` (mol/m^3)
    with Fick's law and a constant diffusivity. Hydrogen enters through
    :doc:`metal_electrolyte_interface` (absorption reaction) or
    :doc:`prescribed_flux`. Supports 2-D, 3-D and axisymmetric meshes.

    .. note::

       Trapping of hydrogen at microstructural defects
       :cite:p:`McNabb1963,Oriani1970` and stress-driven migration are not
       yet included.

    .. rubric:: Fields

    * ``CL``: lattice hydrogen concentration :math:`C_L`, CG of order
      ``Field_Orders["CL"]``, solved in step ``Solution_Steps["CL"]``.
    * ``CLOld``: committed :math:`C_{L,n}`.

    .. rubric:: Weak form

    With :math:`\Omega` the domain group named by ``domain`` and test
    function :math:`\delta C_L`,

    .. math::

        R_{C_L} = \int_\Omega \delta C_L\,\frac{C_L - C_{L,n}}{\Delta t}\,\mathrm{d}x
        + \int_\Omega D_L\,\nabla\delta C_L\cdot\nabla C_L\,\mathrm{d}x,

    with the consistent Jacobian (consistent mass, not lumped).

    .. rubric:: Outside the weak form

    * ``initialize_fields`` sets :math:`C_L = C_{L,n}` to ``C_L0``.
    * ``commit`` copies ``CL`` to ``CLOld``; ``reset_step`` restores it.
    * Global measure ``"<model name>_Hydrogen"``:
      :math:`\int_\Omega C_L\,\mathrm{d}x` (mol; per unit thickness in
      2-D, per radian on axisymmetric meshes).
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Field_Orders = {"CL": 2}              # REQUIRED: polynomial order of CL
            Solution_Steps = {"CL": 0}            # REQUIRED: step solving CL
            Materials = {
                "Steel": {
                    "D_L": 1.0e-9,                # REQUIRED: lattice diffusivity D_L (m^2/s)
                    "C_L0": 0.0,                  # OPTIONAL (default 0.0): initial concentration (mol/m^3)
                },
            }
            Models = {
                "Hydrogen": {
                    "Type": "ElectroChemistry:HydrogenDiffusion",  # REQUIRED
                    "Material": "Steel",          # REQUIRED: key in Materials
                    "domain": "domain",           # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "ElectroChemistry:HydrogenDiffusion"
    SUPPORTS_AXISYMMETRY = True

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)

        self.My_Step = params.Solution_Steps["CL"]
        order = params.Field_Orders["CL"]
        mesh.Add_Field("CL", "Lattice Hydrogen [mol/m^3]", "CG", order, domain=self.domain)
        mesh.Add_Field("CLOld", "Old Lattice Hydrogen [mol/m^3]", "CG", order, domain=self.domain)
        mesh.Add_TestFunction("CL", self.My_Step)

        self.material = self.params.Models[self.name]["Material"]
        self.D_L = self.params.Materials[self.material]["D_L"]
        self.C_L0 = self.params.Materials[self.material].get("C_L0", 0.0)
        self.params.Global_Measures[f"{self.name}_Hydrogen"] = 0.0

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Return the backward-Euler Fick's-law tangent and residual."""
        if step != self.My_Step:
            return None
        dx = self.mesh.Get_Domain_Integrator(self.domain)

        CL = self.mesh.Get_Field("CL", step=self.My_Step)
        CLOld = self.mesh.Get_Field("CLOld")
        CL_t = self.mesh.Get_TestFunction("CL", self.My_Step)
        CL_tr = self.mesh.Get_Trial_Function("CL", self.My_Step)

        F = CL_t * (CL - CLOld) / dt_constant * dx + self.D_L * inner(grad(CL_t), grad(CL)) * dx
        K = CL_t * CL_tr / dt_constant * dx + self.D_L * inner(grad(CL_t), grad(CL_tr)) * dx
        return K, F

    def Update_Global_Measures(self) -> None:
        """Report the total amount of lattice hydrogen in the model's domain."""
        dx = self.mesh.Get_Domain_Integrator(self.domain)
        local_amount = fem.assemble_scalar(self.mesh.Compile_Form(self.mesh.Get_Field("CL") * dx))
        self.params.Global_Measures[f"{self.name}_Hydrogen"] = comm.allreduce(local_amount, op=MPI.SUM)

    def initialize_fields(self) -> None:
        """Set the lattice hydrogen concentration to its initial value."""
        self.mesh.Constant_Field("CL", self.C_L0, domain=self.domain)
        self.mesh.Constant_Field("CLOld", self.C_L0, domain=self.domain)

    def commit(self) -> None:
        """Store the converged concentration as the old state."""
        self.mesh.Copy_Field("CL", "CLOld", domain=self.domain)

    def reset_step(self) -> None:
        """Restore the last committed concentration after a rejected step."""
        self.mesh.Copy_Field("CLOld", "CL", domain=self.domain)
