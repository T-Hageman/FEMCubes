import ufl
from ufl import inner, dot, FacetNormal, conditional

from femcubes.core.base_model import BaseModel

class BasalFriction(BaseModel):
    r"""Penalty bed contact and basal sliding friction on a boundary.

    The model represents the base of ice that slides over its bed, on one
    boundary group. A penalty on the normal motion keeps the ice in contact
    with the bed, and a tangential friction traction resists sliding. The
    friction follows either a power law of the sliding velocity
    :cite:p:`Weertman1957a`, a common choice in ice-flow models
    :cite:p:`Brondex2017`, or a
    law that is linear in the sliding velocity and proportional to the normal
    contact pressure (``Linear``). Acts on the displacement ``u`` if ``"u"``
    is in ``Solution_Steps`` (any solid model), otherwise on the velocity
    ``v`` (Stokes models). Supports 2-D; in 3-D only one tangential direction
    carries friction (see below).

    .. rubric:: Fields

    * ``u`` (with ``uOld``) or ``v``: read and tested, provided by the solid
      or Stokes model. ``boundary`` must lie on the support of this field.
    * ``phasefield``: read only, with ``DamagedForces=True``; provided by a
      phase-field model.
    * Registers no fields.

    .. rubric:: Weak form

    With :math:`\mathbf{w}` the unknown (``u`` or ``v``), test function
    :math:`\delta\mathbf{w}`, :math:`\mathbf{n}` the outward normal and
    :math:`\Gamma_b` the boundary group ``boundary``, the contribution to the
    residual of :math:`\mathbf{w}` is

    .. math::

        R_{\mathbf{w}} =
        \int_{\Gamma_b} k_n\,(\delta\mathbf{w}\cdot\mathbf{n})(\mathbf{w}\cdot\mathbf{n})\,\mathrm{d}s
        - \int_{\Gamma_b} g_b(\phi)\,\tau_b\,(\delta\mathbf{w}\cdot\mathbf{t})\,\mathrm{d}s,
        \qquad k_n = 10^{12},

    a two-sided normal penalty (it also resists lift-off), and the friction
    traction

    .. math::

        \tau_b = -C\,|v_b|^{m-1}\,v_b \quad (\texttt{Weertman}),
        \qquad
        \tau_b = -C_\mathrm{lin}\,\langle N\rangle_+\,v_b,\quad
        N = k_n\,\mathbf{w}\cdot\mathbf{n} \quad (\texttt{Linear}),

    with the sliding velocity :math:`v_b = \mathbf{t}\cdot(\mathbf{u}-\mathbf{u}_n)/\Delta t`
    for ``u`` and :math:`v_b = \mathbf{t}\cdot\mathbf{v}` for ``v``. The
    tangent is :math:`\mathbf{t} = (-n_y, n_x)` in 2-D and the unnormalized
    :math:`\mathbf{t} = \mathbf{n}\times\mathbf{e}_x = (0, n_z, -n_y)` in 3-D.
    The linear law is assembled only for :math:`C_\mathrm{lin}>0`.
    :math:`g_b = 1`, or :math:`g_b = 10^{-6} + (1-10^{-6})(1-\phi)^2` with
    ``DamagedForces=True`` (friction term only; :math:`\phi` not clipped).
    The Jacobian is the consistent derivative, including the
    :math:`\mathbf{t}\otimes\mathbf{n}` coupling of the linear law. For the
    Weertman law with :math:`m<1`, :math:`|v_b|^{m-1}` is unbounded at
    :math:`v_b = 0` (for example at rest in the first step).

    .. rubric:: Outside the weak form

    No Dirichlet conditions, state variables or outputs.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"u": 0, "phasefield": 1}  # REQUIRED: "u" (or "v" for Stokes); "phasefield" with DamagedForces
            Materials = {
                "Ice": {                     # REQUIRED: always read from Materials["Ice"]
                    # FrictionModel "Weertman":
                    "Weertman_C": 1.0e4,     # REQUIRED: friction coefficient C (Pa (s/m)^m)
                    "Weertman_m": 1.0,       # REQUIRED: sliding exponent m (-)
                    # FrictionModel "Linear":
                    "Linear_Friction_C": 0.6,  # REQUIRED: coefficient C_lin (s/m)
                },
            }
            Models = {
                "Bed": {
                    "Type": "glaciology:BasalFriction",  # REQUIRED
                    "boundary": "bottom",                # REQUIRED: boundary group of the bed
                    "FrictionModel": "Weertman",         # OPTIONAL (default "Weertman"): or "Linear"
                    "DamagedForces": False,              # OPTIONAL (default False): degrade the friction with phasefield
                    "domain": "domain",                  # OPTIONAL (default "domain"): domain the boundary must bound
                },
            }
    """

    MODEL_TYPE = "glaciology:BasalFriction"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        
        self.DamagedForces = self.params.Models[self.name].get("DamagedForces", False)
        self.FrictionModel = self.params.Models[self.name].get("FrictionModel", "Weertman")
        if self.FrictionModel == "Weertman":
            self.C_Weertman = self.params.Materials["Ice"]["Weertman_C"]
            self.m_Weertman = self.params.Materials["Ice"]["Weertman_m"]
        elif self.FrictionModel == "Linear":
            self.Linear_Friction_C = self.params.Materials["Ice"]["Linear_Friction_C"]
        
        self.StokesFormulation = False
        # check if u or v in solution steps
        if "u" in self.params.Solution_Steps:
            self.field_name = "u"
            self.step = self.params.Solution_Steps["u"]
            self.StokesFormulation = False
        elif "v" in self.params.Solution_Steps:
            self.field_name = "v"
            self.step = self.params.Solution_Steps["v"]
            self.StokesFormulation = True
        else:
            raise ValueError("BasalFriction model requires either 'u' or 'v' in Solution_Steps.")
            
        if self.DamagedForces:
            self.step_dam = params.Solution_Steps["phasefield"]
            
        self.step = params.Solution_Steps[self.field_name]
        self.dim = params.dim

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if (step != self.step):
            return
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr= self.mesh.Get_Trial_Function(self.field_name, step)
        
        normals = FacetNormal(self.mesh.mesh)
        if self.dim == 2:
            # 2D: rotate normal 90° to get the unique tangent direction
            tangents = ufl.as_vector([-normals[1], normals[0]])
        else:
            # 3D: one tangent via cross product with global x-axis (n × e_x).
            # Valid for basal boundaries where n is not parallel to e_x.    (pragma added to temporarily remove from coverage, as we don't have a 3D test case with basal friction)
            tangents = ufl.as_vector([0.0, normals[2], -normals[1]])    # pragma: no cover
        
        ds = self.mesh.Get_Field_Boundary_Integrator(self.field_name, self.boundary)
        
        # Non-Pen condition
        ky = 1e12
        n_x_n = ufl.outer(normals, normals)
        t_x_t = ufl.outer(tangents, tangents)
        t_x_n = ufl.outer(tangents, normals)

        Fy = ky * dot(field, normals)    # normal contact force (compression positive)

        F = inner(field_t, n_x_n * field) * ky * ds
        K = inner(field_t, n_x_n * field_tr) * ky * ds
        
        # Friction law
        if self.StokesFormulation == False:
            field_Old = self.mesh.Get_Field("uOld")
            v = dot(tangents, field - field_Old) / dt_constant
            dv_dfield = 1.0 / dt_constant
        else:
            v = dot(tangents,field)
            dv_dfield = 1.0
        
        dam = 1.0
        if self.DamagedForces:
            phi = self.mesh.Get_Field("phasefield", step)
            dam = (1.0-1e-6)*(1.0 - phi)**2 + 1e-6 
        
        if self.FrictionModel == "Weertman":
            Fx    = -1.0*self.C_Weertman * abs(v)**(self.m_Weertman-1) * v
            dF_dv = -1.0*self.C_Weertman * self.m_Weertman * abs(v)**(self.m_Weertman-1)

            F += -inner(field_t, tangents * Fx) * dam * ds
            K += -inner(field_t, t_x_t * field_tr) * dam * dF_dv * dv_dfield * ds
        elif self.FrictionModel == "Linear":
            if self.Linear_Friction_C > 0.0:
                FyPlus  = conditional(Fy > 0.0, Fy, 0.0)
                dFyPlus = conditional(Fy > 0.0, 1.0, 0.0)
                Fx    = -1.0*self.Linear_Friction_C * FyPlus * v

                F += -inner(field_t, tangents * Fx) * dam * ds
                # Consistent tangent: two contributions:
                #   (1) t⊗n block: friction changes because normal load changes
                #   (2) t⊗t block: friction changes because tangential velocity changes
                K += inner(field_t, (t_x_n * field_tr * (dFyPlus * ky * v)+ t_x_t * field_tr * (FyPlus * dv_dfield)) * (dam * self.Linear_Friction_C)) * ds
        else:
            raise ValueError(f"Unknown FrictionModel type: {self.FrictionModel}")
        

        
        return K, F
