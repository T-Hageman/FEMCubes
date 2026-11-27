"""Ice-specific phase-field fracture model."""

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.glaciology._ice_state import add_ice_state_fields, check_ice_state
from femcubes.glaciology.ice_material import IceMaterial
from femcubes.solids.phase_field import PhaseField


class IcePhaseField(PhaseField):
    r"""Phase-field fracture of ice with a toughness from the local ice state.

    The model is the phase-field model of :doc:`../solids/phase_field` (same
    fields, weak form, irreversibility methods, crack-length measures and
    settings), with the spatially varying toughness :math:`G_c(p, T)` of
    :meth:`IceMaterial.Get_Gc <femcubes.glaciology.ice_material.IceMaterial.Get_Gc>`
    instead of ``Materials[...]["Gc"]``. For ``AT2``,
    :math:`G_c = 10\,(1-p)` J/m^2, independent of :math:`T` (meant for the
    cohesive models, whose strength does not follow from :math:`G_c`); for
    ``AT1``, ``HO-AT1`` and ``HO-AT2``, the strength calibration
    :math:`G_c = c\,f_t^2\ell/E(p)` :cite:p:`Pham2011,Tanne2018`
    (:math:`c = 8/3` for the AT1 types, :math:`256/27` for ``HO-AT2``),
    which grows with porosity as :math:`(1-p)^{-4}` and, with the
    temperature-independent :math:`f_t` of
    :meth:`~femcubes.glaciology.ice_material.IceMaterial.Get_Tensile_Strength`,
    does not depend on :math:`T`. Supports 2-D and 3-D. Reads its crack
    driving force from an ice solid model (:doc:`ice_viscoelastic`:
    ``ElasticEnergy``; :doc:`ice_cohesive_viscoelastic`: ``PsiField``).

    .. rubric:: Fields

    * All fields of :doc:`../solids/phase_field` (``phasefield`` solved in
      step ``Solution_Steps["phasefield"]``, its history fields, ``Gc_eff``
      with ``Gc_Mesh_Correction``, and the auxiliary and multiplier fields of
      the chosen options).
    * ``poros`` (DG, order ``Field_Orders["Porosity"]``, else
      ``Field_Orders["pw"]``, else 1) and ``T`` (CG, order
      ``Field_Orders["Temperature"]``, else ``Field_Orders["T"]``, else 1):
      registered on ``domain`` if no other model has registered them; set by
      the user.
    * The crack driving force named by ``CDF_field``: read only.

    .. rubric:: Weak form

    The residuals of :doc:`../solids/phase_field`, with :math:`G_c` replaced by
    :math:`G_c(p,T)` in every term (crack density, gradient-auxiliary
    penalty of the higher-order formulations, and the :math:`G_c/\ell`
    scaling of the irreversibility terms).

    .. rubric:: Outside the weak form

    * As :doc:`../solids/phase_field`. With ``Gc_Mesh_Correction``,
      :math:`G_c(p,T)` is interpolated per cell into ``Gc_eff`` and
      multiplied by the correction factor.
    * Before every assembly of the phase-field step, ``poros`` is checked to
      lie in :math:`[0, 1)` and ``T`` to be at most 0 °C, on all ranks
      together (the same ``ValueError`` on every rank).
    * ``Post_Remesh`` resets the mesh correction of
      :doc:`../solids/phase_field` and rebinds :math:`G_c` to the
      transferred ``poros`` and ``T``.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Field_Orders = {
                "phasefield": 2,             # REQUIRED: order of phasefield (>= 2 with "AL")
                "Porosity": 0,               # OPTIONAL (default Field_Orders["pw"], else 1): DG order of poros
                "Temperature": 1,            # OPTIONAL (default Field_Orders["T"], else 1): CG order of T
            }
            Solution_Steps = {"phasefield": 1}  # REQUIRED: staggered step solving phasefield
            Materials = {
                "Ice": {
                    "ell": 10.0,             # REQUIRED: phase-field length scale (m)
                    "kappa": 1.0e-3,         # OPTIONAL (default 1e-3): residual stiffness (-)
                    "PF_formulation": "AT2", # OPTIONAL (default "AT2"): AT1, AT2, HO-AT1 or HO-AT2
                    "pf_damping": 0.0,       # OPTIONAL (default 0.0): viscous phase-field damping (Pa s)
                },
            }
            Models = {
                "PF": {
                    "Type": "Glaciology:PhaseField",  # REQUIRED
                    "Material": "Ice",                # REQUIRED: key in Materials
                    "CDF_field": "ElasticEnergy",     # OPTIONAL (default "ElasticEnergy"): crack driving force field
                    "Irreversible_Method": "AL",      # OPTIONAL (default "AL"): "AL", "Hist" (AT2, HO-AT2 only) or "Penalty"
                    "Negative_Time_Damage": False,    # OPTIONAL (default False): drive damage also for t <= 0
                    "Gc_Mesh_Correction": None,       # OPTIONAL (default None, off): as in Solid:PhaseField
                    "domain": "domain",               # OPTIONAL (default "domain", the full mesh): domain group
                },
            }

    The irreversibility and mesh-correction settings are those of
    :doc:`../solids/phase_field`; ``"Hist"`` is rejected for ``AT1`` and
    ``HO-AT1``.
    """

    MODEL_TYPE = "Glaciology:PhaseField"

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        # The material fields are needed before PhaseField.__init__ (which
        # sets ``self.domain``) evaluates the fracture toughness.
        add_ice_state_fields(params, mesh, params.Models[name].get("domain", "domain"))
        self.iceMat = IceMaterial()
        self._bind_ice_material_state(mesh)
        super().__init__(name, params, mesh)

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        """Check ``poros`` and ``T`` on all ranks, then update as :doc:`../solids/phase_field`."""
        if step == self.My_Step:
            check_ice_state(self.mesh, self.domain, f"{self.MODEL_TYPE} '{self.name}'")
        return super().Update_Weak_Constants(time_constant, dt_constant, step)

    def Post_Remesh(self) -> None:
        """Reset the mesh correction and rebuild the fracture toughness on the new mesh."""
        super().Post_Remesh()
        self._bind_ice_material_state()
        self.Gc = self._get_fracture_toughness()

    def _bind_ice_material_state(self, mesh=None) -> None:
        """PRIVATE: Bind material inputs to the current mesh fields."""
        mesh = self.mesh if mesh is None else mesh
        self._porosity = mesh.Get_Field("poros")
        self._temperature = mesh.Get_Field("T")

    def _get_fracture_toughness(self):
        """PRIVATE: Evaluate fracture toughness from the local ice state."""
        return self.iceMat.Get_Gc(self._porosity, self._temperature, self.ell, PF_formulation=self.PF_Version)
