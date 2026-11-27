from typing import List

from femcubes.core.base_model import BaseModel

class OpenBoundary(BaseModel):
    r"""Fix all species concentrations and the electrolyte potential on a named boundary.

    The model connects the simulated electrolyte to a large, well-mixed bulk
    electrolyte, whose composition is not affected by the processes inside
    the domain. Strong Dirichlet conditions hold every species of
    ``params.Species`` at its initial concentration and the potential
    :math:`\varphi` (field ``E``) at the model's ``"E"`` value. Used with
    :doc:`electrolyte`, which defines the fields. Supports 2-D, 3-D and
    axisymmetric meshes.

    .. rubric:: Fields

    * Defines none. Constrains the fields ``<species>`` and ``E`` of
      :doc:`electrolyte`.

    .. rubric:: Weak form

    No contribution.

    .. rubric:: Outside the weak form

    * Dirichlet conditions on :math:`\Gamma_{b}`, the boundary group named
      by ``boundary``: :math:`C_\pi = C_{\pi,0}` (``"C0"``) for every
      species (immobile solids included); the species with
      ``"C0": "Neutrality"`` takes :math:`-\sum_{\pi'} z_{\pi'} C_{\pi',0} / z_\pi`
      over the other species; :math:`\varphi = E`. They act at all times.
    * The fields may live on a subdomain; the boundary must lie on their
      support, otherwise building the conditions raises a ``ValueError``.
      The ``"domain"`` setting is only checked: the boundary must bound it.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Species = {...}                       # REQUIRED: as for Electrolyte ("C0", and "Charge" of dissolved species)
            Models = {
                "Bulk": {
                    "Type": "ElectroChemistry:OpenBoundary",  # REQUIRED
                    "boundary": "right",          # REQUIRED: boundary group
                    "E": 0.0,                     # REQUIRED: electrolyte potential on the boundary (V)
                    "domain": "domain",           # OPTIONAL (default "domain", the full mesh): domain group the boundary bounds
                },
            }
    """

    MODEL_TYPE = "ElectroChemistry:OpenBoundary"
    SUPPORTS_AXISYMMETRY = True

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        
        self.dofNames = {}
        self.dofVals = {}
        NeutralSpecies = None
        EN = 0.0
        for s in self.params.Species.keys():
            self.dofNames[s] = f"{s}"
            if self.params.Species[s]["C0"] == "Neutrality":
                if NeutralSpecies is not None:
                    raise ValueError("Only one species can be set to 'Neutrality' for initial concentration.")
                NeutralSpecies = s
            else:
                # Immobile solids ("Solid": True) carry no charge.
                EN += self.params.Species[s].get("Charge", 0.0) * self.params.Species[s]["C0"]
                self.dofVals[s] = self.params.Species[s]["C0"]
        if NeutralSpecies is not None:
            self.dofVals[NeutralSpecies] = -EN / self.params.Species[NeutralSpecies]["Charge"]
        self.dofNames["E"] = "E"
        self.dofVals["E"] = self.params.Models[self.name]["E"]

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Boundary-only model: contributes Dirichlet BCs but no weak forms."""
        return None
    
    def get_bcs_for_field(self, field_name: str) -> List:
        
        for f in self.dofNames.values():
            if f == field_name:
                return self.mesh.Build_Increment_Dirichlet_BCs(
                    field_name=f,
                    boundary_name=self.boundary,
                    target_value=self.dofVals[field_name],
                )

        return None
