from ufl import inner, Identity, tr, conditional, lt, as_tensor

from femcubes.core.base_model import BaseModel
from femcubes.core.time_discretizations import Newmark_Update as Apply_Newmark_Update
from femcubes.core.utils.maths_utils import nabla_s, macPlus, eig, Tensor_To_Mandel_UFL, Mandel_To_Tensor_UFL
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
import ufl
from femcubes.solids._split_elastic_common import _set_fracture_driver
from femcubes.solids._domain_points import _DomainPoints

from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import (
    ViscoElasticKernels as _ViscoElasticKernels,
)

def _make_viscoelastic_kernels(viscosity, plastic_n, lame, shear, true_2d=False):
    """Wrap the C++ ViscoElasticKernels callable to match the Python kernel API."""
    if true_2d:
        return _ViscoElasticKernels(viscosity, plastic_n, lame, shear, True)
    # Keep the four-argument call for existing builds of the extension; the
    # optional fifth argument is only needed when intrinsic 2-D mode is used.
    return _ViscoElasticKernels(viscosity, plastic_n, lame, shear)

class ViscoElastic(BaseModel):
    r"""Viscoelastic (Maxwell-type J2 / Glen creep) solid with Newmark time integration.

    The model describes a solid that deforms elastically under rapid loading
    and creeps under sustained stress, as glacier ice does. The small strain
    is split additively,
    :math:`\boldsymbol\varepsilon = \boldsymbol\varepsilon^e + \boldsymbol\varepsilon^v`,
    into an elastic part, which sets the stress, and a viscous part, which
    grows at a rate set by the deviatoric stress (a Maxwell-type model). The
    viscous strain follows a J2 flow rule, either Newtonian or the power law
    of :cite:t:`Glen1955`, the standard flow law of glacier ice
    :cite:p:`CuffeyPaterson2010`. It is integrated by a backward-Euler return
    map at each quadrature point in a C++ kernel :cite:p:`SimoHughes1998`,
    and the Jacobian uses the algorithmic tangent consistent with this update
    :cite:p:`SimoTaylor1985`. Inertia, damping and gravity are included as in
    :doc:`linear_elastic`. Supports 2-D (plane strain, or intrinsic 2-D with
    ``true_2d``) and 3-D.

    With ``DamageModel="PhaseField"``, the whole stress is degraded by the
    phase field, while creep remains driven by the undegraded stress. The
    default stress-based crack driving force converts the stress into an
    energy-like quantity, so that only the stress, and not the creep
    deformation, drives the crack (see :ref:`solids-crack-driving-forces`).
    As in
    :doc:`linear_elastic`, material with :math:`\phi\geq0.8` loses its
    density (an inconsistent split, see :ref:`solids-split-consistency`):
    damaged material carries no compression, so damage spreads under gravity.
    For degradation of the tensile part only, see :doc:`split_viscoelastic`.

    .. rubric:: Fields

    * ``u``: displacement :math:`\mathbf{u}`, vector CG of order
      ``Field_Orders["u"]``, solved in step ``Solution_Steps["u"]``.
    * ``v``, ``a``, ``uOld``, ``vOld``, ``aOld``: Newmark velocity,
      acceleration and committed values, as in :doc:`linear_elastic`.
    * ``sigma``: Cauchy stress from the return map (undegraded with
      ``DamageModel="PhaseField"``), quadrature field with six Mandel
      components; ``dsigma_dStrainMandel``: its algorithmic tangent
      (:math:`6\times6`).
    * ``Plastic_Strain``, ``Plastic_Strain_Old``: viscous strain
      :math:`\boldsymbol\varepsilon^v` and its committed value
      :math:`\boldsymbol\varepsilon^v_n` (quadrature, six components).
    * ``LambdaField``, ``LambdaFieldOld``: equivalent viscous strain increment
      :math:`\Lambda` of the step (quadrature, scalar).
    * ``Total_Strain``, ``dam_elastic``: quadrature copies of
      :math:`\boldsymbol\varepsilon(\mathbf u)` and of the damage factor used
      by the kernel.
    * ``ElasticEnergy``: crack driving force (scalar quadrature); with
      ``DamageModel="PhaseField"`` or ``"poros"``.
    * ``phasefield``: read only, from :doc:`phase_field`, with
      ``DamageModel="PhaseField"``; ``poros``: porosity, DG of order
      ``Field_Orders.get("pw", 1)``, defined here and set by another model,
      with ``DamageModel="poros"``.

    .. rubric:: Weak form

    With test function :math:`\delta\mathbf{u}` and Mandel strain vectors
    (zero out-of-plane components on 2-D meshes), the contribution to the
    residual of ``u`` is

    .. math::

        R_{\mathbf{u}} =
        \int_\Omega g(\phi)\,\boldsymbol{\varepsilon}(\delta\mathbf{u}):\boldsymbol{\sigma}\,\mathrm{d}x
        + \int_\Omega g_\rho\,\rho\,\delta\mathbf{u}\cdot\ddot{\mathbf{u}}\,\mathrm{d}x
        + \int_\Omega c\,\delta\mathbf{u}\cdot\dot{\mathbf{u}}\,\mathrm{d}x
        - \int_\Omega g_\rho\,\rho\,g\,\delta u_{d}\,\mathrm{d}x .

    The inertia term is assembled only with ``Inertia=True`` (default
    ``False``), the damping term only for ``LE_Damping`` :math:`c>0`, the
    gravity term only for ``gravity`` :math:`g\neq0`; the Newmark relations
    for :math:`\dot{\mathbf u}`, :math:`\ddot{\mathbf u}` are those of
    :doc:`linear_elastic`. The Jacobian uses
    :math:`g(\phi)\,\partial\boldsymbol\sigma/\partial\boldsymbol\varepsilon`
    from ``dsigma_dStrainMandel``, consistent with the return map at fixed
    :math:`\phi`. The stress is

    .. math::

        \boldsymbol\sigma = d_\sigma\,\mathbb{C}:(\boldsymbol\varepsilon(\mathbf u) - \boldsymbol\varepsilon^v),
        \qquad
        \boldsymbol\varepsilon^v = \boldsymbol\varepsilon^v_n + \Lambda\,\frac{\partial f}{\partial\boldsymbol\sigma},
        \qquad
        f(\boldsymbol\sigma) = \eta_\mathrm{vp}\Bigl(\frac{\Lambda}{\Delta t}\Bigr)^{1/n},

    with :math:`\mathbb C` the isotropic stiffness (:math:`\lambda`, :math:`\mu`),
    :math:`f = \sqrt{\tfrac{d}{2}\,\mathbf s:\mathbf s}` (:math:`\mathbf s` the
    deviatoric stress; :math:`f=\sqrt{3J_2}` for :math:`d=3`, :math:`d=2` with
    ``true_2d``), and :math:`\Lambda = 0` while :math:`f` of the trial stress
    is below :math:`10^{-6}` Pa. ``Visc_Type="Newtonian"`` sets
    :math:`n=1`, :math:`\eta_\mathrm{vp} = d\,\eta` (so
    :math:`\dot{\boldsymbol\varepsilon}^v = \mathbf s/(2\eta)`);
    ``"Glen"`` sets :math:`n` = ``Glen_n`` and
    :math:`\eta_\mathrm{vp} = \bigl(d^{(n+1)/2}/(2A)\bigr)^{1/n}` (so
    :math:`\dot{\boldsymbol\varepsilon}^v = A\,\tau_e^{n-1}\mathbf s` with
    :math:`\tau_e^2 = \tfrac12\mathbf s:\mathbf s` in 3-D).
    The damage options are:

    * ``DamageModel`` absent: :math:`g = g_\rho = d_\sigma = 1`.
    * ``"PhaseField"``: :math:`g(\phi) = \kappa + (1-\kappa)(1-\hat\phi)^2`,
      :math:`\hat\phi = \min(\max(\phi,0),1)`, :math:`g_\rho = 1` for
      :math:`\phi<0.8` and 0 otherwise, :math:`d_\sigma = 1` (creep is driven
      by the undegraded stress). Where :math:`g < 10^{-2}` the kernel freezes
      the viscous strain (:math:`\boldsymbol\varepsilon^v = \boldsymbol\varepsilon^v_n`).
    * ``"poros"``: :math:`g = 1`, :math:`g_\rho = d_\sigma = 1-\max(\varphi,0)`
      with :math:`\varphi` the field ``poros``, so the stress is degraded
      inside the return map.

    .. rubric:: Outside the weak form

    * Before every assembly: Newmark update of :math:`\dot{\mathbf u}`,
      :math:`\ddot{\mathbf u}`; with a damage model, update of ``ElasticEnergy``;
      then the return map from :math:`\boldsymbol\varepsilon^v_n` at the
      current :math:`\boldsymbol\varepsilon(\mathbf u)` with
      :math:`\Delta t` = ``params.dt``, giving ``sigma``,
      ``dsigma_dStrainMandel``, ``Plastic_Strain`` and ``LambdaField``.
    * ``ElasticEnergy`` with ``"PhaseField"``: the driver ``CrackDrivingForce``
      (see :ref:`solids-crack-driving-forces`; ``Stress-Based`` not clipped)
      at :math:`\boldsymbol\varepsilon(\mathbf u) - \boldsymbol\varepsilon^v`,
      where :math:`\boldsymbol\varepsilon^v` is the value stored by the
      previous return map (one evaluation behind). With ``"poros"``:
      :math:`\sum_i\langle s_i\rangle_+^2/((1-\varphi) f_t)^2` from the
      principal stresses of the degraded elastic stress, an indicator only.
    * ``commit``: Newmark update, copies ``u``, ``v``, ``a``,
      ``Plastic_Strain``, ``LambdaField`` to their ``*Old`` fields;
      ``reset_step`` restores ``u``, ``v``, ``a`` and ``Plastic_Strain``.
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
            dt = 1.0                         # REQUIRED: time step, also used by the return map (s)
            Materials = {
                "Ice": {
                    "Lame": 6.5e9,           # REQUIRED: first Lame parameter lambda (Pa)
                    "Shear": 3.5e9,          # REQUIRED: shear modulus mu (Pa)
                    "Bulk": 8.8e9,           # REQUIRED: bulk modulus (Pa); read but not used
                    "Density": 917.0,        # REQUIRED: density rho (kg/m^3)
                    "Visc_Type": "Glen",     # OPTIONAL (default "Newtonian"): "Newtonian" or "Glen"
                    "Glen_A": 1.0e-24,       # REQUIRED for Glen: rate factor A (Pa^-n s^-1)
                    "Glen_n": 3.0,           # REQUIRED for Glen: exponent n (-)
                    # "Viscosity": 1.0e13,   # REQUIRED for Newtonian: dynamic viscosity eta (Pa s)
                    "LE_Damping": 0.0,       # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                    "Inertia": False,        # OPTIONAL (default False): include the inertia term
                    # With DamageModel="PhaseField" only:
                    "Gc": 1.0,               # REQUIRED: fracture toughness (J/m^2)
                    "ell": 0.1,              # REQUIRED: phase-field length scale (m)
                    "f_t": 1.0e6,            # REQUIRED: tensile strength (Pa); OPTIONAL (default 1e6) for "poros"
                    "kappa": 1.0e-3,         # OPTIONAL (default 1e-3): residual stiffness (-)
                    "CrackDrivingForce": "Stress-Based",  # OPTIONAL (default "Stress-Based"): driver definition
                    "PF_formulation": "AT2", # OPTIONAL (default "AT2"): Stress-Based scaling
                },
            }
            Models = {
                "Solid": {
                    "Type": "Solid:ViscoElastic",  # REQUIRED
                    "Material": "Ice",             # REQUIRED: key in Materials
                    "DamageModel": "PhaseField",   # OPTIONAL (default None): None, "PhaseField" or "poros"
                    "true_2d": False,              # OPTIONAL (default False): intrinsic 2-D law, 2-D meshes only
                    "domain": "domain",            # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "Solid:ViscoElastic"
    # Quadrature fields updated pointwise by the constitutive kernel.
    _POINT_FIELDS = ("Total_Strain", "dam_elastic", "sigma", "dsigma_dStrainMandel", "Plastic_Strain",
                     "Plastic_Strain_Old", "LambdaField", "LambdaFieldOld")

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        
        self.dim = params.dim
        self.true_2d = self.params.Models[self.name].get("true_2d", False)
        if self.true_2d and self.dim != 2:
            raise ValueError("true_2d is only supported for 2-D ViscoElastic models.")
        self.Create_Fields(params, mesh)
        
        # material parameters
        self.material = self.params.Models[self.name]["Material"]
        self.damage_model_name = self.params.Models[self.name].get("DamageModel", None)
        self.gravity = self.params.gravity
        self.damping = self.params.Materials[self.material].get("LE_Damping", 0.0)
        self.Inertia = self.params.Materials[self.material].get("Inertia", False)
        self._read_material(self.params.Materials[self.material])
        
        self.NM_Beta = self.params.Newmark_Beta
        self.NM_Gamma = self.params.Newmark_Gamma
        
        # Check if damage model is specified
        if self.damage_model_name == "PhaseField":
            ## Phase-field based damage
            mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", (), domain=self.domain)
            self.DrivingForce_Type = self.params.Materials[self.material].get("CrackDrivingForce", "Stress-Based")
            self.ell = self.params.Materials[self.material]["ell"]
            self.kappa = self.params.Materials[self.material].get("kappa", 1.0e-3)  # Residual stiffness for fully damaged material

            self.d_e = 1.0
            self.d_rho = 1.0
        elif self.damage_model_name == "poros":
            ## porosity-based damage
            self.d_e = 1.0
            self.d_rho = 1.0
            porosity_order = params.Field_Orders.get("pw", 1)
            
            mesh.Add_Field("poros", "Porosity [-]", "DG", porosity_order, domain=self.domain)
            
            # Optional, only for defining potential locations of cracks
            mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", (), domain=self.domain)
            self.f_t = self.params.Materials[self.material].get("f_t", 1.0e6)  # Tensile strength (Pa)
            
        else:
            self.d_e = 1.0
            self.d_rho = 1.0

        self._update_viscousCreep = self._build_creep_kernel()
        self._total_strain_expr = None
        self._damage_expr = None

    def Post_Remesh(self) -> None:
        """Clear quadrature expressions compiled on the previous mesh."""
        self._total_strain_expr = None
        self._damage_expr = None
        self._points.reset()

    def _read_material(self, material: dict) -> None:
        """PRIVATE: set ``lame``, ``shear``, ``bulk``, ``rho``, the creep law (``viscosity_type``,
        ``viscosity``, the J2 stress scale of the return map, and ``plastic_n``) and, with
        ``DamageModel="PhaseField"``, ``Gc`` and ``f_t`` from ``material``. Models with spatially
        varying properties override this."""
        self.lame = material["Lame"]
        self.shear = material["Shear"]
        self.bulk = material["Bulk"]
        self.rho = material["Density"]
        self.viscosity_type = material.get("Visc_Type", "Newtonian")
        constitutive_dim = 2.0 if self.true_2d else 3.0
        if self.viscosity_type == "Newtonian":
            # The dimension-aware J2 stress scale is d*eta for a Newtonian
            # dynamic viscosity eta, so that epsdot_v = s / (2*eta).
            self.viscosity = constitutive_dim * material["Viscosity"]
            self.plastic_n = 1.0
        elif self.viscosity_type == "Glen":
            self.Glen_A = material["Glen_A"]
            self.plastic_n = material["Glen_n"]
            self.viscosity = (constitutive_dim ** ((self.plastic_n + 1.0) / 2.0) / (2.0 * self.Glen_A)) ** (1.0 / self.plastic_n)
        else:
            raise NotImplementedError(f"Viscosity type '{self.viscosity_type}' not implemented.")
        if self.damage_model_name == "PhaseField":
            self.Gc = material["Gc"]
            self.f_t = material["f_t"]

    def _build_creep_kernel(self):
        """PRIVATE: the C++ return map of the model's constants (called with ``_update_plastic_fields``'s arrays)."""
        return _make_viscoelastic_kernels(self.viscosity, self.plastic_n, self.lame, self.shear, self.true_2d)
        
    def Create_Fields(self, params, mesh):
        mesh.Add_Vector_Field("u", "Displacement [m]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("uOld", "Old Displacement [m]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("v", "Velocity [m/s]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("vOld", "Old Velocity [m/s]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("a", "Acceleration [m/s²]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("aOld", "Old Acceleration [m/s²]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)

        mesh.Add_Quadrature_Field("sigma", "Stress field [Pa]", (6,1), domain=self.domain)
        mesh.Add_Quadrature_Field("dsigma_dStrainMandel", "dsigma/dStrainMandel field [-]", (6,6), domain=self.domain)
    
        mesh.Add_Quadrature_Field("Plastic_Strain", "Plastic Strain Tensor", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("Plastic_Strain_Old", "Old Plastic Strain Tensor", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("LambdaField", "Internal variable field [-]", (), domain=self.domain)
        mesh.Add_Quadrature_Field("LambdaFieldOld", "Old Internal variable field [-]", (), domain=self.domain)
        
        # Fields required to get ip quantities
        mesh.Add_Quadrature_Field("Total_Strain", "Total Strain Tensor", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("dam_elastic", "Elastic damage variable", (), domain=self.domain)
        
        # Register test functions (actual test functions will be retrieved when needed)
        self.My_Step = params.Solution_Steps["u"]
        mesh.Add_Vector_TestFunction("u", params.dim, self.My_Step)
        self._points = _DomainPoints(mesh, self.domain, self._POINT_FIELDS)
        

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Return the weak form for the visco-elastic model."""
        
        if (step != self.My_Step):
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)
        
        # Get fields - mesh automatically returns mixed space version if applicable
        v = self.mesh.Get_Field("v")
        a = self.mesh.Get_Field("a")
        dv_du = self.NM_Gamma / (self.NM_Beta * dt_constant)
        da_du = 1.0 / (self.NM_Beta * dt_constant**2)
        
        u_t = self.mesh.Get_TestFunction("u", self.My_Step)
        u_tr = self.mesh.Get_Trial_Function("u", self.My_Step)
        
        # Momentum Balance: Stresses
        degradation = self.d_e if self.damage_model_name == "PhaseField" else 1.0
        strain_test = Tensor_To_Mandel_UFL(nabla_s(u_t), self.dim)
        strain_trial = Tensor_To_Mandel_UFL(nabla_s(u_tr), self.dim)
        stress = self.mesh.Get_Field("sigma")
        tangent = self.mesh.Get_Field("dsigma_dStrainMandel")
        stress_vector = ufl.as_vector([stress[i, 0] for i in range(6)])
        F = degradation * inner(strain_test, stress_vector) * dx
        K = degradation * ufl.dot(strain_test, ufl.dot(tangent, strain_trial)) * dx
        
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
        u = self.mesh.Get_Field("u", step=self.My_Step)
        self.Newmark_Update(
            u,
            self.mesh.Get_Field("v"),
            self.mesh.Get_Field("a"),
            self.mesh.Get_Field("uOld"),
            self.mesh.Get_Field("vOld"),
            self.mesh.Get_Field("aOld"),
            self._weak_constant_value(dt_constant),
        )
        strain_tot = Tensor_To_Mandel_UFL(nabla_s(u), self.dim)
        if self.damage_model_name in ("PhaseField", "poros", "CrackEstimateOnly"):
            strain_p = self.mesh.Get_Field("Plastic_Strain")
            ep_vec = ufl.as_vector([strain_p[i, 0] for i in range(6)])
            self.Update_PhaseField_Fields(strain_tot - ep_vec)
        self._update_plastic_fields(strain_tot)
        return False

    def _update_plastic_fields(self, strain_expr) -> None:
        self._total_strain_expr = self._points.interpolate("Total_Strain", strain_expr, self._total_strain_expr)
        self.mesh.Get_Field("Total_Strain").x.scatter_forward()
        Damage_Creep = True
        if self.damage_model_name == "PhaseField":
            Damage_Creep = False
        self._damage_expr = self._points.interpolate("dam_elastic", self.d_e, self._damage_expr)
        self.mesh.Get_Field("dam_elastic").x.scatter_forward()
        names = ("Total_Strain", "dam_elastic", "sigma", "Plastic_Strain", "Plastic_Strain_Old",
                 "dsigma_dStrainMandel", "LambdaField", "LambdaFieldOld")
        arrays = self._points.gather(*names)
        strain, damage, *state = arrays
        self._update_viscousCreep(strain, self.params.dt, Damage_Creep, damage, *state)
        self._points.scatter(dict(zip(names, arrays)))
        self.mesh.Get_Field("Plastic_Strain").x.scatter_forward()
        self.mesh.Get_Field("LambdaField").x.scatter_forward()
        self.mesh.Get_Field("sigma").x.scatter_forward()
        self.mesh.Get_Field("dsigma_dStrainMandel").x.scatter_forward()
        
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        for field_name in ("u", "uOld", "v", "vOld", "a", "aOld", "Plastic_Strain", "Plastic_Strain_Old", "LambdaField", "LambdaFieldOld"):
            self.mesh.Zero_Field(field_name, domain=self.domain)
            
    def Newmark_Update(self, u, v, a, uOld, vOld, aOld, dt) -> None:
        """Update displacement, velocity, and acceleration using Newmark-beta method."""
        return Apply_Newmark_Update(u,v,a,uOld,vOld,aOld,dt,self.NM_Beta,self.NM_Gamma, dofs=self.mesh.Get_Field_Dofs("u", self.domain))
            

    def Update_PhaseField_Fields(self, strain) -> None:
        if self.damage_model_name == None:
            return
        
        if self.damage_model_name == "poros":
            poros = self.mesh.Get_Field("poros")
            poros = conditional(lt(poros, 0.0), 0.0, poros)
            self.d_e = 1.0 - poros
            self.d_rho = 1.0 - poros
            
            # Crack driving force for visualizing potential crack locations only!!
            strain_2d, strain_3D = self._fracture_strain_tensors(strain)
            if self.true_2d:
                sigma_2d = self.d_e * (self.lame * tr(strain_2d) * Identity(2) + 2.0 * self.shear * strain_2d)
                sigma = self._embed_2d_tensor(sigma_2d)
            else:
                sigma = self.d_e * self.lame * tr(strain_3D) * Identity(3) + 2.0 * self.d_e * self.shear * strain_3D
            eig1, eig2, eig3 = eig(sigma, 2)
            Psi_Elastic = (macPlus(eig1)**2 + macPlus(eig2)**2 + macPlus(eig3)**2) / (self.d_e * self.f_t)**2
            Psi_Elastic = conditional(lt(Psi_Elastic, 0.0), 0.0, Psi_Elastic)
            self.mesh.Set_From_Expression("ElasticEnergy", Psi_Elastic, domain=self.domain)
        
        if self.damage_model_name == "CrackEstimateOnly":
            strain_2d, strain_3D = self._fracture_strain_tensors(strain)
            if self.true_2d:
                sigma_2d = self.lame * tr(strain_2d) * Identity(2) + 2.0 * self.shear * strain_2d
                sigma = self._embed_2d_tensor(sigma_2d)
            else:
                sigma = self.lame * tr(strain_3D) * Identity(3) + 2.0 * self.shear * strain_3D
            eig1, eig2, eig3 = eig(sigma, 2)
            Psi_Elastic = (macPlus(eig1)**2 + macPlus(eig2)**2 + macPlus(eig3)**2) / (self.f_t)**2
            Psi_Elastic = conditional(lt(Psi_Elastic, 0.0), 0.0, Psi_Elastic)
            self.mesh.Set_From_Expression("ElasticEnergy", Psi_Elastic, domain=self.domain)
        
        if self.damage_model_name == "PhaseField":
            # Damage functions
            phasefield = self.mesh.Get_Field("phasefield")
            phi = conditional(lt(phasefield, 1.0), phasefield, 1.0)
            phi = conditional(lt(phi, 0.0), 0.0, phi)
            self.d_e = self.kappa + (1.0-self.kappa)*(1.0-phi)**2
            self.d_rho = conditional(lt(phasefield, 0.8), 1.0, 0.0)
            
            # Crack driving force
            _set_fracture_driver(
                self.mesh, strain, self.lame, self.shear,
                self.true_2d, self.DrivingForce_Type,
                {"Gc": self.Gc, "ell": self.ell, "f_t": self.f_t,
                 "PF_formulation": self.params.Materials[self.material].get("PF_formulation", "AT2")},
                clip_at2=False, domain=self.domain,
            )

    def _fracture_strain_tensors(self, strain):
        """PRIVATE: Return intrinsic and embedded strain tensors for fracture drivers."""
        tensor = Mandel_To_Tensor_UFL(strain, 2 if self.true_2d else 3)
        return tensor, self._embed_2d_tensor(tensor) if self.true_2d else tensor

    @staticmethod
    def _embed_2d_tensor(tensor_2d):
        """PRIVATE: Embed an intrinsic 2-D tensor with zero out-of-plane terms."""
        return as_tensor([
            [tensor_2d[0, 0], tensor_2d[0, 1], 0.0],
            [tensor_2d[1, 0], tensor_2d[1, 1], 0.0],
            [0.0, 0.0, 0.0],
        ])

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
        
        # Update plastic strain history
        self.mesh.Copy_Field("Plastic_Strain", "Plastic_Strain_Old", domain=self.domain)
        self.mesh.Copy_Field("LambdaField", "LambdaFieldOld", domain=self.domain)
        
    def reset_step(self):
        self.mesh.Copy_Field("uOld", "u", domain=self.domain)
        self.mesh.Copy_Field("vOld", "v", domain=self.domain)
        self.mesh.Copy_Field("aOld", "a", domain=self.domain)
        
        self.mesh.Copy_Field("Plastic_Strain_Old", "Plastic_Strain", domain=self.domain)
        
