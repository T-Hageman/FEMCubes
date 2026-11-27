from typing import List

from femcubes.core.base_model import BaseModel

class PrescribedValue(BaseModel):
    r"""Strong Dirichlet condition with a constant value on a named boundary.

    Prescribes a scalar field, a whole vector field, or one vector
    component on the boundary group ``boundary``, optionally only inside a
    time window. Works on 2-D, 3-D and axisymmetric meshes and with any
    model that registers the target field. For a target that grows
    linearly in time, see :doc:`prescribed_rate`.

    .. rubric:: Fields

    * ``field``: the constrained field, provided by another model (e.g.
      ``u`` of :doc:`../solids/linear_elastic`). A suffix ``_x``, ``_y`` or
      ``_z`` selects component 0, 1 or 2 of a vector field; without a
      suffix, a scalar field or all components of a vector field are
      constrained. The field may live on a subdomain; the boundary must
      then lie on its support, otherwise a ``ValueError`` is raised.

    .. rubric:: Weak form

    None.

    .. rubric:: Outside the weak form

    * Dirichlet condition :math:`u = \bar{u}` on :math:`\Gamma_\mathrm{b}`
      (the group ``boundary``), with :math:`\bar{u}` = ``value``. It is
      built as an incremental condition: the constrained Newton update is
      :math:`\bar{u} - u`, so the absolute value is reached in the first
      iteration and kept afterwards. For a whole vector field, ``value`` is
      a scalar applied to every component or a list with one entry per
      component.
    * The condition is active while :math:`t \geq t_\mathrm{min}` and
      :math:`t + \Delta t \leq t_\mathrm{max}`, with :math:`t` the start of
      the step; outside the window the field is free on the boundary.
      A change of activation triggers a rebuild of the forms.
    * DG fields are not constrained (no condition is returned).
    * The optional ``domain`` names the (sub)domain the boundary bounds and
      is checked on construction.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Models = {
                "BC_Left": {
                    "Type": "Boundary:PrescribedValue",  # REQUIRED
                    "boundary": "left",   # REQUIRED: boundary group
                    "field": "u_x",       # REQUIRED: field name, optional suffix _x, _y or _z
                    "value": 0.0,         # REQUIRED: prescribed value (field units); list for a whole vector field
                    "tmin": 0.0,          # OPTIONAL (default None, no lower limit): activation start time (s)
                    "tmax": 10.0,         # OPTIONAL (default None, no upper limit): activation end time (s)
                    "domain": "domain",   # OPTIONAL (default "domain", the full mesh): domain group the boundary bounds
                },
            }
    """

    MODEL_TYPE = "Boundary:PrescribedValue"
    SUPPORTS_AXISYMMETRY = True

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        
        dof = self.params.Models[self.name]["field"]
        self.value = self.params.Models[self.name]["value"]
        
        self.t_max = self.params.Models[self.name].get("tmax", None)
        self.t_min = self.params.Models[self.name].get("tmin", None)
        
        # Check if Vector sub-component BC
        if dof.endswith("_x"):
            self.field_name = dof[:-2]
            self.component = 0
        elif dof.endswith("_y"):
            self.field_name = dof[:-2]
            self.component = 1
        elif dof.endswith("_z"):
            self.field_name = dof[:-2]
            self.component = 2
        else:
            self.field_name = dof
            self.component = None
        self._weak_active = None

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Boundary-only model: contributes Dirichlet BCs but no weak forms."""
        return None

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        active = self._weak_time_window_active(time_constant, dt_constant, self.t_min, self.t_max)
        changed = self._weak_active is not None and active != self._weak_active
        self._weak_active = active
        return changed
    
    def get_bcs_for_field(self, field_name: str) -> List:
        """
        Get boundary conditions for a specific field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            List of BCs associated with this field
        """
        
        if (field_name != self.field_name):
            return None
        
        time = getattr(self.params, "t", getattr(self.params, "start_time", 0.0))
        active = self._weak_time_window_active(time, self.params.dt, self.t_min, self.t_max)
        if not active:
            return None

        target = self._get_target_value(self._weak_constant_value(time) + self._weak_constant_value(self.params.dt))
        if isinstance(target, (list, tuple)) and self.component is not None:
            target = target[self.component]

        return self.mesh.Build_Increment_Dirichlet_BCs(field_name=field_name, boundary_name=self.boundary, target_value=target, component=self.component)

    def _get_target_value(self, time):
        """PRIVATE: Return the absolute boundary target at the solve time."""
        return self.value
