from ufl import inner, as_tensor, Identity, tr, conditional, lt

from femcubes.core.base_model import BaseModel
from femcubes.core.time_discretizations import Newmark_Update as Apply_Newmark_Update
from femcubes.core.utils.maths_utils import nabla_s, Tensor_To_Mandel_UFL
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.solids._split_elastic_common import _set_fracture_driver

class LinearElastic(BaseModel):
    r"""Linear-elastic solid with Newmark time integration and optional phase-field degradation.

    The model solves the small-strain momentum balance of an isotropic
    linear-elastic solid, with optional inertia, viscous damping and gravity.
    Velocities and accelerations follow from the Newmark scheme
    :cite:p:`Newmark1959`; without inertia, the problem is quasi-static.
    Supports 2-D (plane strain, or intrinsic 2-D with ``true_2d``) and 3-D.

    With ``DamageModel="PhaseField"``, the model writes the crack driving
    force for :doc:`phase_field`, by default the principal-stress criterion
    with a threshold of :cite:t:`Miehe2015a`, and degrades the stress with
    the quadratic degradation function :math:`g(\phi)`
    :cite:p:`Bourdin2000,Miehe2010a`. The whole stress is degraded, while the
    crack driving force can be split into a damageable and a retained part
    (``CrackDrivingForce``), as in the hybrid formulation of
    :cite:t:`Ambati2015b`. The momentum balance thus remains linear in the
    displacements at fixed phase field, but damaged material has no stiffness
    left, also in compression (an inconsistent split, see
    :ref:`solids-split-consistency`). Such material cannot carry its own
    weight, so its density is removed where :math:`\phi\geq0.8`, and under
    gravity damage spreads from a crack into the surrounding material. For
    degradation of the tensile part of the stress only, see
    :doc:`split_linear_elastic`.

    .. rubric:: Fields

    * ``u``: displacement :math:`\mathbf{u}`, vector CG of order
      ``Field_Orders["u"]``, solved in step ``Solution_Steps["u"]``.
    * ``v``, ``a``: velocity and acceleration, vector CG of the same order
      (Newmark, not solved for).
    * ``uOld``, ``vOld``, ``aOld``: committed :math:`\mathbf{u}_n`,
      :math:`\dot{\mathbf{u}}_n`, :math:`\ddot{\mathbf{u}}_n`.
    * ``p``: output pressure :math:`-\lambda\,\mathrm{tr}\,\boldsymbol{\varepsilon}(\mathbf{u})`,
      DG of order ``Field_Orders["u"] - 1``.
    * ``sigma``: output Cauchy stress, quadrature field with six Mandel
      components (``sigma_xx``, ``sigma_xy``, ... for output).
    * ``ElasticEnergy``: crack driving force, scalar quadrature field;
      only with ``DamageModel``.
    * ``phasefield``: read only, provided by :doc:`phase_field`; only with
      ``DamageModel="PhaseField"``.

    .. rubric:: Weak form

    With test function :math:`\delta\mathbf{u}`,
    :math:`\boldsymbol{\varepsilon}(\mathbf{w}) = \tfrac12(\nabla\mathbf{w} + \nabla\mathbf{w}^{\top})`,
    and :math:`\Omega` the domain group named by ``domain``, the
    contribution to the residual of ``u`` is

    .. math::

        R_{\mathbf{u}} =
        \int_\Omega g(\phi)\bigl[\lambda\,\mathrm{tr}\,\boldsymbol{\varepsilon}(\delta\mathbf{u})\,
        \mathrm{tr}\,\boldsymbol{\varepsilon}(\mathbf{u})
        + 2\mu\,\boldsymbol{\varepsilon}(\delta\mathbf{u}):\boldsymbol{\varepsilon}(\mathbf{u})\bigr]\,\mathrm{d}x
        + \int_\Omega g_\rho(\phi)\,\rho\,\delta\mathbf{u}\cdot\ddot{\mathbf{u}}\,\mathrm{d}x
        + \int_\Omega c\,\delta\mathbf{u}\cdot\dot{\mathbf{u}}\,\mathrm{d}x
        - \int_\Omega g_\rho(\phi)\,\rho\,g\,\delta u_{d}\,\mathrm{d}x

    The tensors have the mesh dimension :math:`d`, so a 2-D mesh gives
    plane strain (``true_2d`` gives the same in-plane stress). The inertia
    term is assembled only with ``Inertia=True``, the damping term only for
    ``LE_Damping`` :math:`c>0`, the gravity term only for ``gravity``
    :math:`g\neq0`; :math:`g` acts along the last coordinate
    :math:`x_d`. Velocity and acceleration follow the Newmark scheme

    .. math::

        \ddot{\mathbf{u}} = \frac{\mathbf{u}-\mathbf{u}_n}{\beta\Delta t^2}
        - \frac{\dot{\mathbf{u}}_n}{\beta\Delta t}
        - \Bigl(\frac{1}{2\beta}-1\Bigr)\ddot{\mathbf{u}}_n,
        \qquad
        \dot{\mathbf{u}} = \frac{\gamma}{\beta\Delta t}(\mathbf{u}-\mathbf{u}_n)
        + \Bigl(1-\frac{\gamma}{\beta}\Bigr)\dot{\mathbf{u}}_n
        + \Delta t\Bigl(1-\frac{\gamma}{2\beta}\Bigr)\ddot{\mathbf{u}}_n,

    and the Jacobian uses :math:`\partial\ddot{\mathbf{u}}/\partial\mathbf{u} = 1/(\beta\Delta t^2)`
    and :math:`\partial\dot{\mathbf{u}}/\partial\mathbf{u} = \gamma/(\beta\Delta t)`.
    Without ``DamageModel``, :math:`g = g_\rho = 1`. With
    ``DamageModel="PhaseField"``,

    .. math::

        g(\phi) = \kappa + (1-\kappa)(1-\hat\phi)^2,\quad
        \hat\phi = \min(\max(\phi,0),1),
        \qquad
        g_\rho(\phi) = \begin{cases}1 & \phi<0.8\\ 0 & \phi\geq0.8\end{cases}

    with :math:`\phi` held fixed in this step (staggered scheme).

    .. rubric:: Outside the weak form

    * Before every assembly, :math:`\dot{\mathbf{u}}` and :math:`\ddot{\mathbf{u}}`
      are updated from the current :math:`\mathbf{u}` with the Newmark
      relations above, and ``sigma`` is recomputed. With
      ``DamageModel="PhaseField"``, ``ElasticEnergy`` is recomputed from
      :math:`\boldsymbol{\varepsilon}(\mathbf{u})` with the crack driving
      force ``CrackDrivingForce`` (see :ref:`solids-crack-driving-forces`;
      ``Stress-Based`` is clipped at zero for AT2).
    * ``commit`` copies ``u``, ``v``, ``a`` to ``uOld``, ``vOld``, ``aOld``
      and evaluates ``p`` and ``sigma``; ``reset_step`` restores the
      committed fields.
    * ``sigma`` is the plane-strain stress (with
      :math:`\sigma_{zz} = \lambda\,\mathrm{tr}\,\boldsymbol{\varepsilon}`) on 2-D meshes,
      and has :math:`\sigma_{zz}=0` with ``true_2d``.
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            Field_Orders = {"u": 2}          # REQUIRED: polynomial order of u
            Solution_Steps = {"u": 0}        # REQUIRED: staggered step solving u
            gravity = -9.81                  # REQUIRED: acceleration along the last axis (m/s^2)
            Newmark_Beta = 0.25              # REQUIRED: Newmark beta (-)
            Newmark_Gamma = 0.5              # REQUIRED: Newmark gamma (-)
            Materials = {
                "Ice": {
                    "Lame": 1.0e9,           # REQUIRED: first Lame parameter lambda (Pa)
                    "Shear": 3.5e8,          # REQUIRED: shear modulus mu (Pa)
                    "Density": 917.0,        # REQUIRED: density rho (kg/m^3)
                    "LE_Damping": 1.0e6,     # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                    "Inertia": True,         # OPTIONAL (default True): include the inertia term
                    # With DamageModel only:
                    "Gc": 1.0,               # REQUIRED: fracture toughness (J/m^2)
                    "ell": 0.1,              # REQUIRED: phase-field length scale (m)
                    "f_t": 1.0e6,            # REQUIRED: tensile strength (Pa)
                    "kappa": 1.0e-3,         # OPTIONAL (default 1e-3): residual stiffness (-)
                    "CrackDrivingForce": "Stress-Based",  # OPTIONAL (default "Stress-Based"): driving force definition
                    "PF_formulation": "AT2", # OPTIONAL (default "AT2"): AT1, AT2, HO-AT1 or HO-AT2 (Stress-Based scaling)
                },
            }
            Models = {
                "Solid": {
                    "Type": "Solid:LinearElastic",  # REQUIRED
                    "Material": "Ice",              # REQUIRED: key in Materials
                    "DamageModel": "PhaseField",    # OPTIONAL (default None): degrade the stress with phasefield
                    "true_2d": False,               # OPTIONAL (default False): intrinsic 2-D law, 2-D meshes only
                    "domain": "domain",             # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "Solid:LinearElastic"

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        self.dim = params.dim
        
        mesh.Add_Vector_Field("u", "Displacement [m]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("uOld", "Old Displacement [m]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("v", "Velocity [m/s]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("vOld", "Old Velocity [m/s]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("a", "Acceleration [m/s²]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("aOld", "Old Acceleration [m/s²]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)

        mesh.Add_Field("p", "Pressure [Pa]", "DG", params.Field_Orders["u"] - 1, domain=self.domain) # Pressure field for post-processing
        mesh.Add_Quadrature_Field("sigma", "Stress field [Pa]", (6, 1), domain=self.domain)

        self.My_Step = params.Solution_Steps["u"]
        
        # Register test functions (actual test functions will be retrieved when needed)
        mesh.Add_Vector_TestFunction("u", params.dim, self.My_Step)
        
        # material parameters
        self.material = self.params.Models[self.name]["Material"]
        self.true_2d = self.params.Models[self.name].get("true_2d", False)
        if self.true_2d and self.dim != 2:
            raise ValueError("true_2d is only supported for 2-D LinearElastic models.")
        self.lame = self.params.Materials[self.material]["Lame"]
        self.shear = self.params.Materials[self.material]["Shear"]
        self.rho = self.params.Materials[self.material]["Density"]
        self.gravity = self.params.gravity
        self.damping = self.params.Materials[self.material].get("LE_Damping", 0.0)
        self.Inertia = self.params.Materials[self.material].get("Inertia", True)
        
        self.NM_Beta = self.params.Newmark_Beta
        self.NM_Gamma = self.params.Newmark_Gamma
        
        # Check if damage model is specified
        self.damage_model_name = self.params.Models[self.name].get("DamageModel", None)
        if self.damage_model_name:
            mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", (), domain=self.domain)
            self.DrivingForce_Type = self.params.Materials[self.material].get("CrackDrivingForce", "Stress-Based")
            self.Gc = self.params.Materials[self.material]["Gc"]
            self.ell = self.params.Materials[self.material]["ell"]
            self.f_t = self.params.Materials[self.material]["f_t"]
            self.kappa = self.params.Materials[self.material].get("kappa", 1.0e-3)  # Residual stiffness for fully damaged material
        else:
            pass
        
        self.d_e = 1.0  # No damage
        self.d_rho = 1.0  # No damage
    
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """
        Return the weak form for the linear elastic model.
        """
        
        if (step != self.My_Step):
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)
        
        # Get fields
        u = self.mesh.Get_Field("u", step=self.My_Step)
        v = self.mesh.Get_Field("v")
        a = self.mesh.Get_Field("a")
        dv_du = self.NM_Gamma / (self.NM_Beta * dt_constant)
        da_du = 1.0 / (self.NM_Beta * dt_constant**2)
        
        u_t = self.mesh.Get_TestFunction("u", self.My_Step)
        u_tr = self.mesh.Get_Trial_Function("u", self.My_Step)
        
        # Momentum Balance: fully degraded elastic stresses
        F = self.d_e*self.lame * tr(nabla_s(u_t)) * tr(nabla_s(u)) * dx
        K = self.d_e*self.lame * tr(nabla_s(u_t)) * tr(nabla_s(u_tr)) * dx
        F += 2.0*self.d_e*self.shear * inner(nabla_s(u_t), nabla_s(u)) * dx
        K += 2.0*self.d_e*self.shear * inner(nabla_s(u_t), nabla_s(u_tr)) * dx
        
        # Inertia terms
        if self.Inertia:
            F += self.d_rho*self.rho * inner(u_t, a) * dx
            K += self.d_rho*self.rho * inner(u_t, da_du * u_tr) * dx
            
        # Damping terms
        if self.damping > 0.0:
            F += self.damping * inner(u_t, v) * dx
            K += self.damping * inner(u_t, dv_du * u_tr) * dx
        
        # Body forces (gravity)
        if (self.gravity != 0.0):
            F += -self.d_rho*self.rho*u_t[self.dim-1]*self.gravity*dx #gravity
        
        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        if step != self.My_Step:
            return False
        if self.damage_model_name == "PhaseField":
            self.Update_PhaseField_Fields()
        self.Newmark_Update(
            self.mesh.Get_Field("u", step=self.My_Step),
            self.mesh.Get_Field("v"),
            self.mesh.Get_Field("a"),
            self.mesh.Get_Field("uOld"),
            self.mesh.Get_Field("vOld"),
            self.mesh.Get_Field("aOld"),
            self._weak_constant_value(dt_constant),
        )
        self._update_stress_field()
        return False
    
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        for field_name in ("u", "uOld", "v", "vOld", "a", "aOld", "p", "sigma"):
            self.mesh.Zero_Field(field_name, domain=self.domain)
            
    def Newmark_Update(self, u, v, a, uOld, vOld, aOld, dt) -> None:
        """Update displacement, velocity, and acceleration using Newmark-beta method."""
        return Apply_Newmark_Update(u,v,a,uOld,vOld,aOld,dt,self.NM_Beta,self.NM_Gamma, dofs=self.mesh.Get_Field_Dofs("u", self.domain))

    def _update_stress_field(self) -> None:
        """PRIVATE: Interpolate the current Cauchy stress to quadrature points."""
        displacement = self.mesh.Get_Field("u", step=self.My_Step)
        strain = nabla_s(displacement)
        if self.dim == 2 and self.true_2d:
            stress = self.d_e * (self.lame * tr(strain) * Identity(2) + 2.0 * self.shear * strain)
            stress_3d = as_tensor([
                [stress[0, 0], stress[0, 1], 0.0],
                [stress[1, 0], stress[1, 1], 0.0],
                [0.0, 0.0, 0.0],
            ])
        elif self.dim == 2:
            strain_3d = as_tensor([
                [strain[0, 0], strain[0, 1], 0.0],
                [strain[1, 0], strain[1, 1], 0.0],
                [0.0, 0.0, 0.0],
            ])
            stress_3d = self.d_e * (self.lame * tr(strain_3d) * Identity(3) + 2.0 * self.shear * strain_3d)
        else:
            strain_3d = strain
            stress_3d = self.d_e * (self.lame * tr(strain_3d) * Identity(3) + 2.0 * self.shear * strain_3d)
        stress_mandel = Tensor_To_Mandel_UFL(stress_3d, 3)
        stress_q = as_tensor([[stress_mandel[i]] for i in range(6)])
        self.mesh.Set_From_Expression("sigma", stress_q, domain=self.domain)

    def Update_PhaseField_Fields(self) -> None:
        if self.damage_model_name != "PhaseField":
            return
        
        # Damage functions
        phi_lim = conditional(lt(self.mesh.Get_Field("phasefield"), 1.0), self.mesh.Get_Field("phasefield"), 1.0)
        phi_lim = conditional(lt(phi_lim, 0.0), 0.0, phi_lim)
        self.d_e = self.kappa + (1.0-self.kappa)*(1.0 - phi_lim)**2
        self.d_rho = conditional(lt(self.mesh.Get_Field("phasefield"), 0.8), 1.0, 0.0)
        
        # Crack driving force
        strain = nabla_s(self.mesh.Get_Field("u", step=self.My_Step))
        
        _set_fracture_driver(
            self.mesh, Tensor_To_Mandel_UFL(strain, self.dim), self.lame, self.shear,
            self.true_2d, self.DrivingForce_Type,
            {"Gc": self.Gc, "ell": self.ell, "f_t": self.f_t,
             "PF_formulation": self.params.Materials[self.material].get("PF_formulation", "AT2")},
            clip_at2=True,
            domain=self.domain,
        )
        

    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """
        
        self.Newmark_Update(
            self.mesh.Get_Field("u", step=self.My_Step),
            self.mesh.Get_Field("v"),
            self.mesh.Get_Field("a"),
            self.mesh.Get_Field("uOld"),
            self.mesh.Get_Field("vOld"),
            self.mesh.Get_Field("aOld"),
            self.params.dt,
        )
        
        # Copy current fields to old fields using mesh utility
        self.mesh.Copy_Field("u", "uOld", domain=self.domain)
        self.mesh.Copy_Field("v", "vOld", domain=self.domain)
        self.mesh.Copy_Field("a", "aOld", domain=self.domain)
        
        # Update pressure field for post-processing
        # Evaluate pressure as p = -λ * tr(ε) where ε is the strain tensor
        u = self.mesh.Get_Field("u", step=self.My_Step)
        pressure_expr = -self.lame * tr(nabla_s(u))
        self.mesh.Set_From_Expression("p", pressure_expr, domain=self.domain)
        self._update_stress_field()
        
    def reset_step(self):
        self.mesh.Copy_Field("uOld", "u", domain=self.domain)
        self.mesh.Copy_Field("vOld", "v", domain=self.domain)
        self.mesh.Copy_Field("aOld", "a", domain=self.domain)
        
