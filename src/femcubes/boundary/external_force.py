

from femcubes.core.base_model import BaseModel

class ExternalForce(BaseModel):
    r"""Constant Neumann load on a named boundary.

    Applies a constant force per unit boundary area to one scalar field or
    one vector component on the boundary group ``boundary``, optionally
    degraded by the phase field and only inside a time window. Supports
    2-D and 3-D meshes (not axisymmetric).

    .. rubric:: Fields

    * ``field``: the loaded field, provided by another model (e.g. ``u``).
      A suffix ``_x`` ``_y`` ``_z`` selects component 0, 1, or 2; without a suffix
      the field must be scalar. The field may live on a subdomain; the
      boundary must then lie on its support, otherwise a ``ValueError`` is
      raised.
    * ``phasefield``: read only, provided by :doc:`../solids/phase_field`;
      only with ``DamagedForces=True``.

    .. rubric:: Weak form

    With :math:`u` the selected field or component, :math:`\delta u` its
    test function and :math:`\Gamma_\mathrm{b}` the group ``boundary``,

    .. math::

        R_{u} = -\int_{\Gamma_\mathrm{b}} g_\mathrm{f}(\phi)\,\bar{t}\,\delta u\,\mathrm{d}s,
        \qquad
        g_\mathrm{f}(\phi) = \begin{cases}
        (1-10^{-6})(1-\phi)^2 + 10^{-6} & \texttt{DamagedForces=True}\\
        1 & \text{otherwise}\end{cases}

    with :math:`\bar{t}` = ``value``. The term is assembled while
    :math:`t \geq t_\mathrm{min}` and :math:`t + \Delta t \leq t_\mathrm{max}`
    (:math:`t` the start of the step). It adds nothing to the Jacobian.

    .. rubric:: Outside the weak form

    * None (no Dirichlet conditions, no measures). A change of activation
      triggers a rebuild of the forms.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"u": 0, "phasefield": 1}  # REQUIRED: entry for the loaded field;
                                                        # "phasefield" only with DamagedForces
            Models = {
                "Load": {
                    "Type": "Boundary:ExternalForce",  # REQUIRED
                    "boundary": "top",        # REQUIRED: boundary group
                    "field": "u_y",           # REQUIRED: field name, optional suffix _x or _y
                    "value": -1.0e3,          # REQUIRED: force per unit area, along the component (Pa for u)
                    "tmin": 0.0,              # OPTIONAL (default None): activation start time (s)
                    "tmax": 10.0,             # OPTIONAL (default None): activation end time (s)
                    "DamagedForces": False,   # OPTIONAL (default False): degrade the load with phasefield
                    "domain": "domain",       # OPTIONAL (default "domain", the full mesh): domain group the boundary bounds
                },
            }
    """

    MODEL_TYPE = "Boundary:ExternalForce"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        
        
        self.DamagedForces = self.params.Models[self.name].get("DamagedForces", False)
        dof = self.params.Models[self.name]["field"]
        self.value = self.params.Models[self.name]["value"]
        
        self.t_min = self.params.Models[self.name].get("tmin", None)
        self.t_max = self.params.Models[self.name].get("tmax", None)
        
        # Check if Vector sub-component BC
        if "_x" in dof:
            self.field_name = dof[:-2]
            self.component = 0
        elif "_y" in dof:
            self.field_name = dof[:-2]
            self.component = 1
        elif "_z" in dof:
            self.field_name = dof[:-2]
            self.component = 2
        else:
            self.field_name = dof
            self.component = None
            
        if self.DamagedForces:
            self.step_dam = params.Solution_Steps["phasefield"]
            
        self.step = params.Solution_Steps[self.field_name]
        self._weak_active = None

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if (step != self.step):
            return
        if not self._weak_active:
            return None
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr= self.mesh.Get_Trial_Function(self.field_name, step)
        
        dam = 1.0
        if self.DamagedForces:
            phi = self.mesh.Get_Field("phasefield", step)
            dam = (1.0-1e-6)*(1.0 - phi)**2 + 1e-6 
        
        if (self.component is not None):
            field = field[self.component]
            field_t = field_t[self.component]
            field_tr = field_tr[self.component]
        
        ds = self.mesh.Get_Field_Boundary_Integrator(self.field_name, self.boundary)
        
        F = field_t * dam * -1.0*self.value * ds
        K = None
        
        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        active = self._weak_time_window_active(time_constant, dt_constant, self.t_min, self.t_max)
        changed = self._weak_active is not None and active != self._weak_active
        self._weak_active = active
        return changed
