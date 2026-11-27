from ufl import div, inner, dot

from femcubes.core.base_model import BaseModel
from femcubes.core.utils.maths_utils import nabla_s, macPlus, eig, Tensor_To_Mandel_UFL
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from ufl import tr, conditional, lt, as_tensor, Identity

class Stokes(BaseModel):
    r"""Compressible Stokes flow (velocity-pressure) with Newtonian, Glen or Carreau viscosity.

    The model treats a viscous solid such as ice as a slowly flowing, nearly
    incompressible fluid, and solves for its velocity and pressure. This
    purely viscous description suits flow over long time scales, on which the
    elastic response is negligible; where elastic stresses matter, see
    :doc:`viscoelastic`. A bulk-modulus term relaxes the incompressibility
    constraint, and the pressure is discretized one order lower than the
    velocity (Taylor-Hood elements, :cite:t:`Taylor1973`). The viscosity is Newtonian, follows the
    power law of :cite:t:`Glen1955` in its viscosity form
    :cite:p:`Greve2009`, or follows a Carreau-type law :cite:p:`Carreau1972`
    that bounds the Glen viscosity smoothly between ``visc_min`` and
    ``visc_max``. Optional phase-field degradation acts on the stress and on
    the bulk modulus. Supports 2-D (plane strain, or intrinsic 2-D with
    ``true_2d``) and 3-D.

    .. rubric:: Fields

    * ``v``: velocity :math:`\mathbf v`, vector CG of order
      ``Field_Orders["v"]``, solved in step ``Solution_Steps["v"]``.
    * ``p``: pressure :math:`p`, CG of order ``Field_Orders["v"] - 1``
      (Taylor-Hood, so ``Field_Orders["v"] >= 2``), solved in the same step.
    * ``vOld``, ``pOld``: committed :math:`\mathbf v_n`, :math:`p_n`.
    * ``sigma``: output Cauchy stress (degraded), quadrature field with six
      Mandel components.
    * ``ElasticEnergy``: crack driving force, scalar quadrature field; only
      with ``DamageModel``.
    * ``phasefield``, ``phasefieldOld``: read only, provided by
      :doc:`phase_field`; only with ``DamageModel="PhaseField"``.

    .. rubric:: Weak form

    With test functions :math:`\delta\mathbf v`, :math:`\delta p` and
    :math:`\dot{\boldsymbol\varepsilon} = \tfrac12(\nabla\mathbf v + \nabla\mathbf v^\top)`,
    embedded in 3-D with zero out-of-plane components on 2-D meshes, the
    contributions are

    .. math::

        R_p = \int_\Omega \delta p\,\nabla\cdot\mathbf v\,\mathrm{d}x
        + \int_\Omega \frac{1}{g(\phi)\,K\,\Delta t}\,\delta p\,(p - p_n)\,\mathrm{d}x,

    .. math::

        R_{\mathbf v} = \int_\Omega \dot{\boldsymbol\varepsilon}(\delta\mathbf v):\boldsymbol\sigma\,\mathrm{d}x
        + \int_\Omega g_\rho\,\frac{\rho}{\Delta t}\,\delta\mathbf v\cdot(\mathbf v - \mathbf v_n)\,\mathrm{d}x
        - \int_\Omega g_\rho\,\rho\,g\,\delta v_d\,\mathrm{d}x
        + \int_\Omega c\,\delta\mathbf v\cdot\mathbf v\,\mathrm{d}x,

    .. math::

        \boldsymbol\sigma = g(\phi)\Bigl[-p\,\mathbf I + 2\eta\,\dot{\boldsymbol\varepsilon}'
        + \bigl(\zeta + \tfrac23\eta\bigr)\mathrm{tr}(\dot{\boldsymbol\varepsilon})\,\mathbf I\Bigr],

    with :math:`K` = ``Bulk``, :math:`\zeta` = ``bulk_visc`` and
    :math:`\dot{\boldsymbol\varepsilon}'` the deviatoric strain rate
    (projection with :math:`1/3`, or :math:`1/2` and in-plane tensors with
    ``true_2d``). In 3-D and plane strain the bracket equals
    :math:`-p\mathbf I + 2\eta\dot{\boldsymbol\varepsilon} + \zeta\,\mathrm{tr}(\dot{\boldsymbol\varepsilon})\mathbf I`.
    The inertia term is assembled only with ``Inertia=True`` (backward
    Euler), the gravity term only for ``gravity`` :math:`g\neq0` (along the
    last axis), the damping term only for ``LE_Damping`` :math:`c>0`. The
    viscosity is

    * ``"Newtonian"``: :math:`\eta` = ``Viscosity``.
    * ``"Glen"``: :math:`\eta = \tfrac12 A^{-1/n}\dot\varepsilon_e^{(1-n)/n}`,
      clamped to [``visc_min``, ``visc_max``], with
      :math:`\dot\varepsilon_e = \sqrt{\tfrac12\dot{\boldsymbol\varepsilon}':\dot{\boldsymbol\varepsilon}'} + 10^{-20}`.
    * ``"Carreau"``: :math:`\eta = \eta_\min + (\eta_\max-\eta_\min)\bigl(1 + (\lambda_c\dot\varepsilon_e)^2\bigr)^{(1/n-1)/2}`,
      with :math:`\lambda_c` chosen so that the curve matches the clamped Glen
      viscosity at the geometric mean of the two strain rates where the Glen
      law reaches ``visc_max`` and ``visc_min``.

    The Jacobian is linear in the trial functions with :math:`\eta` frozen at
    the current velocity, so it is exact for ``"Newtonian"`` and a Picard
    linearization for ``"Glen"`` and ``"Carreau"``. Without ``DamageModel``,
    :math:`g = g_\rho = 1`. With ``DamageModel="PhaseField"``,
    :math:`g(\phi) = \kappa + (1-\kappa)(1-\hat\phi)^2`,
    :math:`\hat\phi = \min(\max(\phi,0),1)`, and :math:`g_\rho = 1` for
    :math:`\phi<0.9`, 0 otherwise.

    .. rubric:: Outside the weak form

    * Before every assembly, ``sigma`` is recomputed; with
      ``DamageModel="PhaseField"`` also ``ElasticEnergy``. Only
      ``CrackDrivingForce="Stress-Based"`` is implemented here: with the
      principal values :math:`s_i` of the undegraded stress (in-plane block
      and :math:`\sigma_{zz}`) and :math:`S = \sum_i\langle s_i\rangle_+^2/f_t^2`,
      :math:`\Psi = \langle\frac{G_c}{4\ell}(S-1)\rangle_+` for AT2/HO-AT2 and
      :math:`\Psi = \frac{3G_c}{16\ell}S` for AT1/HO-AT1.
    * ``commit`` copies ``v``, ``p`` to ``vOld``, ``pOld`` and recomputes
      ``sigma``; ``reset_step`` restores ``v`` and ``p``.
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            Field_Orders = {"v": 2}          # REQUIRED: order of v (>= 2); p uses one order less
            Solution_Steps = {"v": 0}        # REQUIRED: staggered step solving v and p
            gravity = -9.81                  # REQUIRED: acceleration along the last axis (m/s^2)
            Materials = {
                "Ice": {
                    "Bulk": 8.8e9,           # REQUIRED: bulk modulus K (Pa)
                    "Density": 917.0,        # REQUIRED: density rho (kg/m^3)
                    "Visc_Type": "Glen",     # REQUIRED: "Newtonian", "Glen" or "Carreau"
                    "Glen_A": 1.0e-24,       # REQUIRED for Glen/Carreau: rate factor A (Pa^-n s^-1)
                    "Glen_n": 3.0,           # REQUIRED for Glen/Carreau: exponent n (-)
                    "visc_min": 1.0e10,      # REQUIRED for Glen/Carreau: lower viscosity bound (Pa s)
                    "visc_max": 1.0e16,      # REQUIRED for Glen/Carreau: upper viscosity bound (Pa s)
                    # "Viscosity": 1.0e13,   # REQUIRED for Newtonian: viscosity eta (Pa s)
                    "bulk_visc": 0.0,        # OPTIONAL (default 0.0): bulk viscosity zeta (Pa s)
                    "LE_Damping": 0.0,       # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                    "Inertia": False,        # OPTIONAL (default False): include the inertia term
                    # With DamageModel only:
                    "Gc": 1.0,               # REQUIRED: fracture toughness (J/m^2)
                    "ell": 0.1,              # REQUIRED: phase-field length scale (m)
                    "f_t": 1.0e6,            # REQUIRED: tensile strength (Pa)
                    "kappa": 1.0e-3,         # OPTIONAL (default 1e-3): residual stiffness (-)
                    "CrackDrivingForce": "Stress-Based",  # OPTIONAL (default "Stress-Based"): only option implemented
                    "PF_formulation": "AT2", # OPTIONAL (default "AT2"): AT1, AT2, HO-AT1 or HO-AT2 (driver scaling)
                },
            }
            Models = {
                "Solid": {
                    "Type": "Solid:Stokes",        # REQUIRED
                    "Material": "Ice",             # REQUIRED: key in Materials
                    "DamageModel": "PhaseField",   # OPTIONAL (default None): degrade the stress with phasefield
                    "true_2d": False,              # OPTIONAL (default False): intrinsic 2-D law, 2-D meshes only
                    "domain": "domain",            # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "Solid:Stokes"
    
    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        
        mesh.Add_Vector_Field("v", "Velocity [m/s]", "CG", params.Field_Orders["v"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("vOld", "Old Velocity [m/s]", "CG", params.Field_Orders["v"], params.dim, domain=self.domain)
        
        mesh.Add_Field("p", "Pressure [Pa]", "CG", params.Field_Orders["v"]-1, domain=self.domain)
        mesh.Add_Field("pOld", "Old Pressure [Pa]", "CG", params.Field_Orders["v"]-1, domain=self.domain)
        mesh.Add_Quadrature_Field("sigma", "Stress field [Pa]", (6, 1), domain=self.domain)

        self.My_Step = params.Solution_Steps["v"]

        # Register test functions (actual test functions will be retrieved when needed)
        mesh.Add_TestFunction("p", self.My_Step)
        mesh.Add_Vector_TestFunction("v", params.dim, self.My_Step)
        
        # material parameters
        self.material = self.params.Models[self.name]["Material"]
        self.true_2d = self.params.Models[self.name].get("true_2d", False)
        if self.true_2d and params.dim != 2:
            raise ValueError("true_2d is only supported for 2-D Stokes models.")
        self.bulk = self.params.Materials[self.material]["Bulk"]
        self.rho = self.params.Materials[self.material]["Density"]
        self.Inertia = self.params.Materials[self.material].get("Inertia", False)
        self.damping = self.params.Materials[self.material].get("LE_Damping", 0.0)
        
        self.visc_type = self.params.Materials[self.material]["Visc_Type"]
        self.bulk_visc = self.params.Materials[self.material].get("bulk_visc", 0.0)
        if self.visc_type == "Newtonian":
            self.visc = self.params.Materials[self.material]["Viscosity"]
        elif self.visc_type == "Glen":
            self.Glen_A = self.params.Materials[self.material]["Glen_A"]
            self.Glen_n = self.params.Materials[self.material]["Glen_n"]
            self.visc_max = self.params.Materials[self.material]["visc_max"]
            self.visc_min = self.params.Materials[self.material]["visc_min"]
        elif self.visc_type == "Carreau":
            self.Glen_A = self.params.Materials[self.material]["Glen_A"]
            self.Glen_n = self.params.Materials[self.material]["Glen_n"]
            self.visc_max = self.params.Materials[self.material]["visc_max"]
            self.visc_min = self.params.Materials[self.material]["visc_min"]
        
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
        self.d_eOld = 1.0  # No damage
        self.d_rho = 1.0  # No damage
    
    def compute_viscosity(self, DevstrainRate):
        if self.visc_type == "Newtonian":
            return self.visc
        elif self.visc_type == "Glen":
            strain_rate_magnitude = (0.5 * inner(DevstrainRate, DevstrainRate))**0.5 + 1e-20
            eta = _glen_viscosity(strain_rate_magnitude, self.Glen_A, self.Glen_n, self.visc_min, self.visc_max, use_ufl=True)
            return eta
        elif self.visc_type == "Carreau":
            strain_rate_magnitude = (0.5 * inner(DevstrainRate, DevstrainRate))**0.5 + 1e-20
            eta = _carreau_viscosity(strain_rate_magnitude, self.Glen_A, self.Glen_n, self.visc_min, self.visc_max, use_ufl=True)
            return eta
        else:
            raise ValueError(f"Unknown viscosity type: {self.visc_type}")
    
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """
        Assemble the global stiffness matrix and force vector for the Stokes model.
        """
        
        if (step != self.My_Step):
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)
        
        # Get test functions
        p_t = self.mesh.Get_TestFunction("p", self.My_Step)
        vel_t = self.mesh.Get_TestFunction("v", self.My_Step)
        
        # Get fields - mesh automatically returns mixed space version if applicable
        v = self.mesh.Get_Field("v", step=self.My_Step)
        p = self.mesh.Get_Field("p", step=self.My_Step)
        vOld = self.mesh.Get_Field("vOld")
        pOld = self.mesh.Get_Field("pOld")
        
        v_tr = self.mesh.Get_Trial_Function("v", self.My_Step)
        p_tr = self.mesh.Get_Trial_Function("p", self.My_Step)
        
        # Mass conservation (pressure should be considered as "partially-damaged pressure",
        # further damaged later on within stress calculations). Note: This term needs the 
        # compressibility, as otherwise the pressure is lagrange-multiplier-like, 
        # and damage won't do anything. Damage allows it to go from "nearly-incompressible"
        # to highly compressible, thus allowing for crack opening.
        div_v = div(v) 
        F = p_t*div_v*dx
        K = p_t*div(v_tr)*dx
        
        F += 1.0/(self.d_e*self.bulk)/dt_constant*p_t*(p-pOld)*dx
        K += 1.0/(self.d_e*self.bulk)/dt_constant*p_t*(p_tr)*dx
        
        # Momentum conservation
        strain_rate = nabla_s(v)
        dev_strain_rate = self._deviatoric_strain_rate(strain_rate)
        visc = self.compute_viscosity(dev_strain_rate)
        stress = self._stress_tensor(p, visc, strain_rate, self.d_e)
        test_strain_rate_3d = self._as_3d_strain_rate(nabla_s(vel_t))
        stress_tr = self._stress_tensor(p_tr, visc, nabla_s(v_tr), self.d_e)

        F += inner(test_strain_rate_3d, stress) * dx
        K += inner(test_strain_rate_3d, stress_tr) * dx

        # Inertia term
        if self.Inertia:
            F += self.d_rho*self.rho/dt_constant*dot(vel_t, v-vOld)*dx
            K += self.d_rho*self.rho/dt_constant*dot(vel_t, v_tr)*dx
        
        # Gravity term
        if self.params.gravity != 0.0:
            F += -self.d_rho*self.rho*vel_t[self.params.dim-1]*self.params.gravity*dx

        # Damping terms
        if self.damping > 0.0:
            F += self.damping * inner(vel_t, v) * dx
            K += self.damping * inner(vel_t, v_tr) * dx

        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        if step != self.My_Step:
            return False
        if self.damage_model_name == "PhaseField":
            self._Update_PhaseField_Fields()
        else:
            self._update_stress_field()
        return False
    
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        for field_name in ("v", "p", "vOld", "pOld", "sigma"):
            self.mesh.Zero_Field(field_name, domain=self.domain)
            
    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """
        self.mesh.Copy_Field("v", "vOld", domain=self.domain)
        self.mesh.Copy_Field("p", "pOld", domain=self.domain)
        self._update_stress_field()

    def reset_step(self):
        self.mesh.Copy_Field("vOld", "v", domain=self.domain)
        self.mesh.Copy_Field("pOld", "p", domain=self.domain)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    
    def _Update_PhaseField_Fields(self) -> None:
        if self.damage_model_name != "PhaseField":
            return
        
        # Damage functions
        phi_lim = conditional(lt(self.mesh.Get_Field("phasefield"), 1.0), self.mesh.Get_Field("phasefield"), 1.0)
        phi_lim = conditional(lt(phi_lim, 0.0), 0.0, phi_lim)
        self.d_e = self.kappa + (1.0-self.kappa)*(1.0 - phi_lim)**2
        self.d_rho = conditional(lt(self.mesh.Get_Field("phasefield"), 0.9), 1.0, 0.0)
        
        phiOld_lim = conditional(lt(self.mesh.Get_Field("phasefieldOld"), 1.0), self.mesh.Get_Field("phasefieldOld"), 1.0)
        phiOld_lim = conditional(lt(phiOld_lim, 0.0), 0.0, phiOld_lim)
        self.d_eOld = self.kappa + (1.0-self.kappa)*(1.0 - phiOld_lim)**2
        
        # Crack driving force
        strainRate = nabla_s(self.mesh.Get_Field("v", step=self.My_Step))
        
        if self.DrivingForce_Type == "Stress-Based":
            dev_strain_rate = self._deviatoric_strain_rate(strainRate)
            p = self.mesh.Get_Field("p", step=self.My_Step)
            visc = self.compute_viscosity(dev_strain_rate)
            sigma = self._stress_tensor(p, visc, strainRate, 1.0)
            eig1, eig2, eig3 = eig(sigma, 2)
            pf_form = self.params.Materials[self.material].get("PF_formulation", "AT2")
            E_normalized = (macPlus(eig1)**2 + macPlus(eig2)**2 + macPlus(eig3)**2) / self.f_t**2
            if pf_form in ("AT2", "HO-AT2"):
                Psi_Elastic = self.Gc/(4.0*self.ell) * (E_normalized - 1.0)
            else:  # AT1
                Psi_Elastic = 3.0*self.Gc/(16.0*self.ell) * E_normalized
            Psi_Elastic = macPlus(Psi_Elastic)
        else:
            raise ValueError(f"Unknown CrackDrivingForce type: {self.DrivingForce_Type}")
        
        self.mesh.Set_From_Expression("ElasticEnergy", Psi_Elastic, domain=self.domain)
        self._update_stress_field()

    def _update_stress_field(self, stress=None) -> None:
        """PRIVATE: Interpolate the current Cauchy stress to quadrature points."""
        if stress is None:
            velocity = self.mesh.Get_Field("v", step=self.My_Step)
            pressure = self.mesh.Get_Field("p", step=self.My_Step)
            strain_rate = nabla_s(velocity)
            dev_strain_rate = self._deviatoric_strain_rate(strain_rate)
            viscosity = self.compute_viscosity(dev_strain_rate)
            stress = self._stress_tensor(pressure, viscosity, strain_rate, self.d_e)
        stress_mandel = Tensor_To_Mandel_UFL(stress, 3)
        stress_q = as_tensor([[stress_mandel[i]] for i in range(6)])
        self.mesh.Set_From_Expression("sigma", stress_q, domain=self.domain)

    def _as_3d_strain_rate(self, strain_rate):
        """PRIVATE: Embed a 2-D plane-strain rate tensor in three dimensions."""
        if self.params.dim == 2:
            return as_tensor([
                [strain_rate[0, 0], strain_rate[0, 1], 0.0],
                [strain_rate[1, 0], strain_rate[1, 1], 0.0],
                [0.0, 0.0, 0.0],
            ])
        return strain_rate

    def _deviatoric_strain_rate(self, strain_rate):
        """PRIVATE: Return the intrinsic or plane-strain deviatoric rate tensor."""
        if self.params.dim == 2 and self.true_2d:
            return strain_rate - tr(strain_rate) * Identity(2) / 2.0
        strain_rate_3d = self._as_3d_strain_rate(strain_rate)
        return strain_rate_3d - tr(strain_rate_3d) * Identity(3) / 3.0

    def _stress_tensor(self, pressure, viscosity, strain_rate, d_e=1.0):
        """PRIVATE: Build the physical stress, embedded for q-field storage."""
        strain_3D = self._as_3d_strain_rate(strain_rate)
        dev_strain = self._deviatoric_strain_rate(strain_rate)
        if self.params.dim == 2 and self.true_2d:
            stress_2d = -d_e*pressure * Identity(2) + 2.0 * d_e* viscosity * dev_strain + d_e * (self.bulk_visc + 2.0/3.0*viscosity) * tr(strain_3D) * Identity(2)
            return as_tensor([
                [stress_2d[0, 0], stress_2d[0, 1], 0.0],
                [stress_2d[1, 0], stress_2d[1, 1], 0.0],
                [0.0, 0.0, 0.0],
            ])
        return -d_e*pressure * Identity(3) + 2.0 * d_e * viscosity * dev_strain + d_e * (self.bulk_visc + 2.0/3.0*viscosity) * tr(strain_3D) * Identity(3)

def _glen_viscosity(strain_rate, glen_a, glen_n, visc_min, visc_max, use_ufl=False):
    coeff = 0.5 * (glen_a ** (-1.0 / glen_n))
    eta = coeff * (strain_rate ** ((1.0 - glen_n) / glen_n))
    if use_ufl:
        eta = conditional(lt(eta, visc_min), visc_min, eta)
        eta = conditional(lt(visc_max, eta), visc_max, eta)
    else:
        import numpy as np
        eta = np.clip(eta, visc_min, visc_max)
    return eta


def _carreau_viscosity(strain_rate, glen_a, glen_n, visc_min, visc_max, use_ufl=False):
    coeff = 0.5 * (glen_a ** (-1.0 / glen_n))
    gamma_max = (visc_max / coeff) ** (glen_n / (1.0 - glen_n))
    gamma_min = (visc_min / coeff) ** (glen_n / (1.0 - glen_n))
    gamma_ref = (gamma_max * gamma_min) ** 0.5
    n_c = 1.0 / glen_n
    m = (n_c - 1.0) / 2.0

    eta_glen_ref = _glen_viscosity(gamma_ref, glen_a, glen_n, visc_min, visc_max, use_ufl=False)
    frac_ref = (eta_glen_ref - visc_min) / (visc_max - visc_min + 1e-20)
    if use_ufl:
        frac_ref = conditional(lt(frac_ref, 1e-8), 1e-8, frac_ref)
        frac_ref = conditional(lt(1.0 - 1e-8, frac_ref), 1.0 - 1e-8, frac_ref)
        lambda_term = frac_ref ** (1.0 / m) - 1.0
        lambda_term = conditional(lt(lambda_term, 0.0), 0.0, lambda_term)
        lambda_c = (lambda_term ** 0.5) / (gamma_ref + 1e-20)
    else:
        import numpy as np
        frac_ref = np.clip(frac_ref, 1e-8, 1.0 - 1e-8)
        lambda_term = frac_ref ** (1.0 / m) - 1.0
        lambda_c = np.sqrt(max(lambda_term, 0.0)) / (gamma_ref + 1e-20)

    eta = visc_min + (visc_max - visc_min) * (1.0 + (lambda_c * strain_rate) ** 2) ** m
    return eta


def plot_viscosity_comparison(params,material="Ice", strain_rate_min=1e-20, strain_rate_max=1e5, n_samples=300, show=True):
    """
    Plot Glen and Carreau viscosities as functions of deviatoric strain rate.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    material_params = params.Materials[material]
    glen_a = material_params["Glen_A"]
    glen_n = material_params["Glen_n"]
    visc_min = material_params["visc_min"]
    visc_max = material_params["visc_max"]
    const_visc = material_params.get("Viscosity", visc_max)

    strain_rate = np.logspace(np.log10(strain_rate_min), np.log10(strain_rate_max), n_samples)

    eta_glen = _glen_viscosity(strain_rate, glen_a, glen_n, visc_min, visc_max)
    eta_carreau = _carreau_viscosity(strain_rate, glen_a, glen_n, visc_min, visc_max)
    eta_const = np.full_like(strain_rate, const_visc, dtype=float)

    fig, ax = plt.subplots()
    ax.loglog(strain_rate, eta_glen, label="Glen (clipped)")
    ax.loglog(strain_rate, eta_carreau, label="Carreau (bounded)")
    ax.loglog(strain_rate, eta_const, label="Constant")
    ax.set_xlabel("Deviatoric strain-rate magnitude")
    ax.set_ylabel("Viscosity")
    ax.legend()
    ax.grid(True, which="both", ls=":")

    if show:
        plt.show()

    return fig, ax
