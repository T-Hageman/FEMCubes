from dolfinx import fem
import numpy as np


from femcubes.core.base_model import BaseModel

class PrescribedFlux(BaseModel):
    r"""Prescribe constant inward fluxes of species on a named boundary.

    Natural boundary condition for any scalar field solved in step
    ``Solution_Steps["C"]``, typically the concentrations of
    :doc:`electrolyte` or ``CL`` of :doc:`hydrogen_diffusion` (which then
    has to be solved in that step). Supports 2-D, 3-D and axisymmetric
    meshes.

    .. rubric:: Fields

    * Defines none. Acts on the fields listed in ``"Species"``.

    .. rubric:: Weak form

    For every listed field :math:`C_k` with flux :math:`j_k` (positive into
    the domain), on :math:`\Gamma_{b}`, the boundary group named by
    ``boundary``,

    .. math::

        R_{C_k} \mathrel{+}= -\int_{\Gamma_b} j_k\,\delta C_k\,\mathrm{d}s

    (no Jacobian contribution).

    .. rubric:: Outside the weak form

    * Nothing. The boundary must lie on the support of every listed field,
      otherwise a ``ValueError`` is raised; the ``"domain"`` setting is
      only checked: the boundary must bound it.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"C": 0}             # REQUIRED: step of the listed fields
            Models = {
                "Influx": {
                    "Type": "ElectroChemistry:PrescribedFlux",  # REQUIRED
                    "boundary": "left",           # REQUIRED: boundary group
                    "Species": ["Fe2+", "Cl-"],   # REQUIRED: names of the fields
                    "flux": [1.0e-4, 2.0e-4],     # REQUIRED: inward flux per field (mol/(m^2 s)), constant
                    "domain": "domain",           # OPTIONAL (default "domain", the full mesh): domain group the boundary bounds
                },
            }
    """

    MODEL_TYPE = "ElectroChemistry:PrescribedFlux"
    SUPPORTS_AXISYMMETRY = True

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        
        self.species = self.params.Models[self.name]["Species"]
        self.flux = self.params.Models[self.name]["flux"]
            
        self.step = params.Solution_Steps["C"]
        self.u_target = fem.Constant(self.mesh.mesh, 0.0)
        self.F_max = 0.0

    def Post_Remesh(self) -> None:
        """Recreate the mesh-bound target constant on the refined mesh."""
        target_value = np.asarray(self.u_target.value).reshape(-1)[0]
        self.u_target = fem.Constant(self.mesh.mesh, target_value)
            
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if (step != self.step):
            return
        
        F = 0.0
        K = 0.0
        for i in range(len(self.species)):
            s = self.species[i]
            flux_value = self.flux[i]
            field_name = f"{s}"

            field_t = self.mesh.Get_TestFunction(field_name, step)
            ds = self.mesh.Get_Field_Boundary_Integrator(field_name, self.boundary)

            F += - flux_value * field_t * ds
            K += 0.0  # No contribution to stiffness matrix for prescribed flux
        
        return K, F

    def commit(self) -> None:

        pass
