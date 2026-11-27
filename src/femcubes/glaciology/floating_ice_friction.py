from ufl import as_vector, dot, inner, sqrt, SpatialCoordinate

from femcubes.core.base_model import BaseModel

class FloatingIceFriction(BaseModel):
    r"""Quadratic water drag on a floating ice plate in map view.

    The model represents the drag of the water under a floating ice plate of
    thickness :math:`h`, such as a sea-ice floe, described in the map plane.
    The drag acts as a body force per unit plate area that opposes the motion
    of the ice relative to a prescribed water velocity, and grows
    quadratically with this relative velocity. Prescribed water-velocity
    patterns load the plate in tension or in shear. Acts on the displacement
    ``u`` if ``"u"`` is in ``Solution_Steps`` (any solid model), otherwise on
    the velocity ``v``. 2-D only: 3-D meshes are rejected at assembly.

    .. rubric:: Fields

    * ``u`` (with ``uOld``) or ``v``: read and tested, provided by the solid
      or Stokes model; it must exist on ``domain``.
    * ``phasefield``: read only, with ``DamagedForces=True``.
    * Registers no fields.

    .. rubric:: Weak form

    With :math:`\mathbf{w}` the unknown (``u`` or ``v``), test function
    :math:`\delta\mathbf{w}` and :math:`\Omega` the domain group ``domain``,
    the contribution to the residual of :math:`\mathbf{w}` is

    .. math::

        R_{\mathbf{w}} =
        \int_\Omega g_b(\phi)\,\frac{\rho_w C}{h}\,
        \lVert\Delta\mathbf{v}\rVert_\epsilon\,\Delta\mathbf{v}\cdot\delta\mathbf{w}\,\mathrm{d}x,
        \qquad
        \Delta\mathbf{v} = \dot{\mathbf{w}} - \mathbf{v}_w(\mathbf{x}, t_{n+1}),
        \quad
        \lVert\Delta\mathbf{v}\rVert_\epsilon = \sqrt{\Delta\mathbf{v}\cdot\Delta\mathbf{v} + \epsilon^2},

    with :math:`\dot{\mathbf{w}} = (\mathbf{u}-\mathbf{u}_n)/\Delta t` for
    ``u`` and :math:`\dot{\mathbf{w}} = \mathbf{v}` for ``v``,
    :math:`\epsilon` = ``Velocity_Regularization``, and
    :math:`g_b = 10^{-6} + (1-10^{-6})(1-\phi)^2` with ``DamagedForces=True``
    (:math:`\phi` not clipped), else 1. The Jacobian is the consistent
    derivative. The water velocity :math:`\mathbf{v}_w` follows
    ``FlowPattern``, with :math:`V(t) = V_0 + \dot V t`, :math:`L_x`, :math:`L_y`
    the parameters ``Lx``, ``Ly`` of the mesh:

    * ``None``: :math:`\mathbf{v}_w = \mathbf{0}`;
    * ``"Tension_x"``: :math:`v_{w,x} = -V\,(x - L_x/2)/(L_x/2)`;
    * ``"Tension_y"``: :math:`v_{w,y} = V\,(y - L_y/2)/(L_y/2)`;
    * ``"Shear_x"``: :math:`v_{w,x} = V\,(y - L_y/2)/(L_y/2)`;
    * ``"Shear_y"``: :math:`v_{w,y} = V\,(x - L_x/2)/(L_x/2)`;

    the other component is zero.

    .. rubric:: Outside the weak form

    No Dirichlet conditions, state variables or outputs.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: must be 2 for assembly
            Lx = 100.0                       # REQUIRED for FlowPattern "Tension_x", "Shear_y" (m)
            Ly = 50.0                        # REQUIRED for FlowPattern "Tension_y", "Shear_x" (m)
            Solution_Steps = {"u": 0, "phasefield": 1}  # REQUIRED: "u" (or "v"); "phasefield" with DamagedForces
            Field_Orders = {"u": 2}          # REQUIRED: order of the unknown ("u" or "v")
            Materials = {
                "Water": {"Density": 1025.0},  # REQUIRED: water density rho_w (kg/m^3)
            }
            Models = {
                "WaterDrag": {
                    "Type": "Glaciology:FloatingIceFriction",  # REQUIRED
                    "h_ice": 1.0,                   # REQUIRED in 2-D: ice thickness h (m)
                    "Friction": 1.0e-2,             # REQUIRED: drag coefficient C (-)
                    "FlowPattern": "Tension_y",     # OPTIONAL (default None, still water): water velocity pattern
                    "V0": 1.0e-3,                   # OPTIONAL (default 0.0): pattern velocity at t = 0 (m/s)
                    "dV": 0.0,                      # OPTIONAL (default 0.0): rate of the pattern velocity (m/s^2)
                    "Velocity_Regularization": 1.0e-12,  # OPTIONAL (default 1e-12): epsilon (m/s)
                    "DamagedForces": False,         # OPTIONAL (default False): degrade the drag with phasefield
                    "domain": "domain",             # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "Glaciology:FloatingIceFriction"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.dim = params.dim
        
        if self.dim == 2:
            self.h_ice = params.Models[self.name]["h_ice"]
        else:
            self.boundary = self.params.Models[self.name]["boundary"]
            self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        
        self.DamagedForces = self.params.Models[self.name].get("DamagedForces", False)
        self.rho = self.params.Materials["Water"]["Density"]
        self.friction = self.params.Models[self.name]["Friction"]
        self.vel_pattern = self.params.Models[self.name].get("FlowPattern", None)
        self.V0 = self.params.Models[self.name].get("V0", 0.0)
        self.dV = self.params.Models[self.name].get("dV", 0.0)
        self.velocity_regularization = self.params.Models[self.name].get("Velocity_Regularization", 1.0e-12)
        
        self.StokesFormulation = False
        # check if u or v in solution steps, for now assuming no slope and only basal friction in x direction
        if "u" in self.params.Solution_Steps:
            self.field_name = "u"
            self.field_name_old = "uOld"
            self.step = self.params.Solution_Steps["u"]
            self.StokesFormulation = False
            self.order = self.params.Field_Orders["u"]
        elif "v" in self.params.Solution_Steps:
            self.field_name = "v"
            self.field_name_old = "vOld"
            self.step = self.params.Solution_Steps["v"]
            self.StokesFormulation = True
            self.order = self.params.Field_Orders["v"]
        else:
            raise ValueError("FloatingIceFriction requires either 'u' or 'v' in Solution_Steps.")
            
        if self.DamagedForces:
            self.step_dam = params.Solution_Steps["phasefield"]
            
        #mesh.Add_Vector_Field("OV", "Ocean Velocity [m/s]", "CG", self.order, self.dim)
        self.step = params.Solution_Steps[self.field_name]

    def commit(self) -> None:
        # vel_ref = self.Get_Vel_Field(3600.0)
        
        # self.mesh.Set_From_Expression("OV", vel_ref)
        pass
        
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if (step != self.step):
            return
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr = self.mesh.Get_Trial_Function(self.field_name, step)
        
        dam = 1.0
        if self.DamagedForces:
            phi = self.mesh.Get_Field("phasefield", step)
            dam = (1.0-1e-6)*(1.0 - phi)**2 + 1e-6 
        
        if self.StokesFormulation:
            vel = field
            dvel = field_tr
        else:
            fieldOld = self.mesh.Get_Field(self.field_name_old, step)
            vel = (field-fieldOld)/dt_constant
            dvel = field_tr/dt_constant
        test_f = field_t
        dx = self.mesh.Get_Domain_Integrator(self.domain)
        
        if self.dim==2:
            vel_ref = self.Get_Vel_Field(time_constant + dt_constant)
            delta_vel = vel - vel_ref
            velocity_norm = sqrt(dot(delta_vel, delta_vel) + self.velocity_regularization**2)
            f = -self.rho * self.friction / self.h_ice * velocity_norm * delta_vel 
            df= -self.rho * self.friction / self.h_ice * (velocity_norm * dvel + (dot(delta_vel, dvel) / velocity_norm) * delta_vel)

            F = -dam * inner(test_f, f) * dx
            K = -dam * inner(test_f, df) * dx
        else:
            raise NotImplementedError("FloatingIceFriction is only implemented for 2D problems.")
        
        return K, F

    def Get_Vel_Field(self, t):
        coords = SpatialCoordinate(self.mesh.mesh)
        if self.vel_pattern is None:
            vel_ref = as_vector((0.0, 0.0))
        elif self.vel_pattern == "Tension_x":
            V_target = self.V0 + self.dV * t
            
            vel_ref_x = -V_target * (coords[0] - self.params.Lx/2)/(self.params.Lx/2)
            vel_ref_y = 0.0
            vel_ref = as_vector((vel_ref_x, vel_ref_y))
        elif self.vel_pattern == "Tension_y":
            V_target = self.V0 + self.dV * t
            
            vel_ref_x = 0.0
            vel_ref_y = V_target * (coords[1] - self.params.Ly/2)/(self.params.Ly/2)
            vel_ref = as_vector((vel_ref_x, vel_ref_y))
        elif self.vel_pattern == "Shear_x":
            V_target = self.V0 + self.dV * t
            
            vel_ref_x = V_target * (coords[1] - self.params.Ly/2)/(self.params.Ly/2)
            vel_ref_y = 0.0
            vel_ref = as_vector((vel_ref_x, vel_ref_y))
        elif self.vel_pattern == "Shear_y":
            V_target = self.V0 + self.dV * t
            
            vel_ref_x = 0.0
            vel_ref_y = V_target * (coords[0] - self.params.Lx/2)/(self.params.Lx/2)
            vel_ref = as_vector((vel_ref_x, vel_ref_y))
        else:
            raise NotImplementedError("FlowPattern is not implemented for 2D.")
        
        return vel_ref