from ufl import FacetNormal, Identity, Measure, conditional, jump
from ufl import grad, inner, lt, le, gt, sqrt, tr, dot
from dolfinx import fem
from dolfinx.mesh import compute_incident_entities, exterior_facet_indices
from petsc4py import PETSc
import numpy as np

from femcubes.core.base_model import BaseModel
from femcubes.core.utils.maths_utils import nabla_s
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.glaciology.ice_material import IceMaterial
from femcubes.glaciology.level_set import smooth_level_set_heaviside

# Momentum-only fictitious thickness [m] used outside the LS ice phase.
h_fict = 1e-6

class ShallowIce(BaseModel):
    r"""Shallow-shelf (SSA) ice flow with DG thickness transport in map view.

    The shallow-shelf approximation describes ice that is thin compared to its
    horizontal extent and that slides over a weak bed or floats, such as ice
    streams and ice shelves :cite:p:`Macayeal1989,Greve2009`. Vertical shear
    is neglected, so that the velocity is constant over the ice thickness and
    the momentum balance can be integrated over it. This leaves a
    two-dimensional problem in the horizontal (map) plane for the
    depth-averaged velocity :math:`\mathbf{v}`, in which the membrane
    stresses, with a Glen viscosity, balance the gravitational driving
    stress, the basal friction on grounded ice and an ocean-pressure term at
    the margin, with grounded and floating ice distinguished by the
    flotation condition. The ice thickness :math:`h` follows from the mass
    balance, discretized with upwind DG fluxes and backward Euler. Written
    for 2-D (map-plane) meshes. The ice temperature sets Glen's rate factor
    (:class:`~femcubes.glaciology.ice_material.IceMaterial`, porosity 0,
    :math:`\rho = 917` kg/m^3, :math:`n = 3`). With
    ``Solution_Steps["LS"]``, the ice extent is the level set of
    :doc:`level_set`. Boundary values come from :doc:`shallow_ice_bc`.

    .. rubric:: Fields

    * ``v``: depth-averaged velocity, vector CG of order ``Field_Orders["v"]``,
      solved in step ``Solution_Steps["v"]``; ``vOld`` its committed value.
    * ``h``: thickness, DG of order ``Field_Orders["h"]``, solved in step
      ``Solution_Steps["h"]``; ``hOld``, ``hOld2`` committed values.
    * ``z``: bed elevation, CG of order ``max(Field_Orders["h"], 1)``; set by
      the user.
    * ``T``: temperature in degrees Celsius, CG of order
      ``Field_Orders["Temperature"]`` (else ``Field_Orders["T"]``, else 1); set
      by the user.
    * With ``"LS"`` in ``Solution_Steps``: ``LS_phase`` (CG, order
      ``Field_Orders["LS"]``), ``LS_phase_h`` and ``LS_phaseOld_h`` (DG, order
      ``Field_Orders["h"]``, not used); ``LS`` is read, provided by
      :doc:`level_set`.

    .. rubric:: Weak form

    Notation: test functions :math:`\delta\mathbf{v}`, :math:`\delta h`;
    :math:`\Omega` the domain group ``domain``, :math:`\Gamma_\mathrm{int}` its
    interior facets, :math:`\partial\Omega` its outer boundary;
    :math:`\rho_o`, :math:`z_\mathrm{sl}` the water density and sea level;
    :math:`g<0` the signed ``gravity``. The active-ice weight is
    :math:`H = H_w(\psi)` (smooth Heaviside of :doc:`level_set` with
    :math:`w` = ``Level_Set["Transition_Width"]``) with a level set, else
    :math:`H = 1`. The effective bed pressure and grounding indicator are

    .. math::

        N = -g\bigl(\rho h - \rho_o\langle z_\mathrm{sl}-z\rangle_+\bigr),
        \qquad \chi_g = \begin{cases}1 & N>0\\0 & N\le0.\end{cases}

    *Mass balance* (step of ``h``):

    .. math::

        R_h = \sum_i W_i\Bigl(\frac{h_i - h_{n,i}}{\Delta t} + 10^{9}\,h_i\,[h_i\le0]\Bigr)\delta h_i
        - \int_\Omega H\,h\,\mathbf{v}\cdot\nabla\delta h\,\mathrm{d}x
        + \int_{\Gamma_\mathrm{int}} H\,h^\mathrm{up}\,(\mathbf{v}^+\cdot\mathbf{n}^+)\,
          (\delta h^+ - \delta h^-)\,\mathrm{d}S
        + \int_{\partial\Omega} H\,h\,(\mathbf{v}\cdot\mathbf{n})\,\delta h\,\mathrm{d}s,

    where the first sum is a lumped (nodal, weights :math:`W_i`) backward
    Euler capacity with a penalty on negative thickness, added through
    ``Add_KF``; :math:`h^\mathrm{up} = h^+` if
    :math:`\mathbf{v}^+\cdot\mathbf{n}^+>0`, else :math:`h^-`. The outer
    boundary term uses the interior :math:`h` on all boundary facets
    (outflow and inflow); :doc:`shallow_ice_bc` adds the inflow thickness.
    On a subdomain, facets shared with the rest of the mesh get the outflow
    term :math:`\int H^+ h^+(\mathbf{v}^+\cdot\mathbf{n}^+)\delta h^+\,\mathrm{d}S`.

    *Momentum balance* (step of ``v``):

    .. math::

        R_{\mathbf{v}} =
        \int_\Omega 2\eta\,h_m\,\boldsymbol{\varepsilon}(\delta\mathbf{v}):
          \bigl(\boldsymbol{\varepsilon}(\mathbf{v}) + \mathrm{tr}\,\boldsymbol{\varepsilon}(\mathbf{v})\,\mathbf{I}\bigr)\,\mathrm{d}x
        - \int_\Omega H\rho g\,h\,\delta\mathbf{v}\cdot\nabla s\,\mathrm{d}x
        + \int_\Omega P\,\delta\mathbf{v}\cdot\nabla H\,\mathrm{d}x
        + \int_\Omega H\chi_g\,\boldsymbol{\tau}_b\cdot\delta\mathbf{v}\,\mathrm{d}x
        + \int_\Omega 10\,(1-H)\,\mathbf{v}\cdot\delta\mathbf{v}\,\mathrm{d}x,

    with

    * :math:`h_m = H h + (1-H)\,10^{-6}` m with a level set, else
      :math:`h_m = h`;
    * :math:`\eta = \mathrm{clip}\bigl(\tfrac12 A^{-1/n}\dot\varepsilon_e^{(1-n)/n},\,
      \eta_\mathrm{min}, \eta_\mathrm{max}\bigr)`,
      :math:`\dot\varepsilon_e = \sqrt{\tfrac12(\boldsymbol{\varepsilon}:\boldsymbol{\varepsilon}
      + (\mathrm{tr}\,\boldsymbol{\varepsilon})^2)} + \dot\varepsilon_0`,
      :math:`A = A(0, T)`;
    * surface :math:`s = \chi_g(z+h) + (1-\chi_g)\bigl(z_\mathrm{sl} + (1-\rho/\rho_o)h\bigr)`;
    * front pressure (only with a level set and ``Front_Ocean_Stress``)
      :math:`P = \tfrac12|g|(\rho h^2 - \rho_o d^2)`, with water depth
      :math:`d = \min(h, \langle z_\mathrm{sl}-z\rangle_+)` on grounded and
      :math:`d = (\rho/\rho_o)h` on floating ice;
    * basal drag :math:`\boldsymbol{\tau}_b = C N\mathbf{v}` (``Budd``, a
      linear form of the law of :cite:t:`Budd1979`) or :math:`C\mathbf{v}`
      (``Weertman``, the linear case of the law of :cite:t:`Weertman1957a`),
      :math:`C` = ``C_base``;
    * the last term with :math:`(1 - [h>0])` instead of :math:`(1-H)` without
      a level set; it keeps :math:`\mathbf{v}` defined where there is no ice.

    The Jacobian of :math:`R_{\mathbf{v}}` keeps :math:`\eta` fixed
    (Picard-type viscosity linearization) and ignores the dependence of
    :math:`\chi_g` and of the clipping. If ``v`` and ``h`` share one step,
    the cross derivatives with respect to the other field are added.

    .. rubric:: Outside the weak form

    * ``commit`` copies ``v`` to ``vOld``, ``hOld`` to ``hOld2`` and ``h`` to
      ``hOld``; ``reset_step`` restores ``h`` and ``v``.
    * ``initialize_fields`` zeroes ``v``, ``vOld``, ``h``, ``hOld``, ``hOld2``;
      set ``z``, ``T``, ``h`` (and ``LS``) afterwards and call
      ``physics.commit()`` once so that the committed fields match.
    * ``Post_Remesh`` rebuilds the temperature-dependent expressions.
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: map-plane mesh
            gravity = -9.81                  # REQUIRED: signed gravitational acceleration (m/s^2)
            Solution_Steps = {"v": 0, "h": 1, "LS": 2}  # REQUIRED: "v", "h"; "LS" OPTIONAL (enables the margin)
            Field_Orders = {
                "v": 2,                      # REQUIRED: CG order of v
                "h": 1,                      # REQUIRED: DG order of h
                "LS": 1,                     # REQUIRED with "LS" in Solution_Steps
                "Temperature": 1,            # OPTIONAL (default Field_Orders["T"], else 1): CG order of T
            }
            Level_Set = {"Transition_Width": 2000.0}  # REQUIRED with "LS": width w (m), see LevelSet
            Materials = {
                "Ice": {                     # REQUIRED: always read from Materials["Ice"]
                    "C_base": 1.0e3,         # REQUIRED: basal friction coefficient C
                                             #   (Budd: s/m, Weertman: Pa s/m)
                    "Friction_Model": "Budd",  # OPTIONAL (default "Budd"): or "Weertman" (case insensitive)
                    "Front_Ocean_Stress": True,  # OPTIONAL (default True): front pressure term (with LS)
                    "visc_min": 1.0e3,       # OPTIONAL (default 1e3): lower viscosity bound (Pa s)
                    "visc_max": 1.0e15,      # OPTIONAL (default 1e20): upper viscosity bound (Pa s)
                    "strain_rate_regularization": 1.0e-20,  # OPTIONAL (default 1e-20): (1/s)
                },
                "Water": {                   # OPTIONAL
                    "Density": 1030.0,       # OPTIONAL (default 1030.0): water density (kg/m^3)
                    "WaterHeight": 0.0,      # OPTIONAL (default 0.0): sea level (m)
                },
            }
            Models = {
                "Interior": {
                    "Type": "Glaciology:ShallowIce",  # REQUIRED
                    "domain": "domain",               # OPTIONAL (default "domain", the full mesh): domain group
                },
            }

    On a subdomain, every boundary group touching it must lie entirely on
    it; its internal edge to the rest of the mesh acts as a traction-free
    outflow boundary.
    """
    
    MODEL_TYPE = "Glaciology:ShallowIce"
    
    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        BaseModel.__init__(self, name, params, mesh)
        
        self.dim = params.dim
        self.UseLumped = True
        self.TimeDiscretization = "Euler" # Euler or BDF2
        self.print_diagnostics = bool(getattr(params, "ShallowIce_Print_Diagnostics", False))

        if "LS" in params.Solution_Steps:
            self.use_level_set_mask = True
            
            self.LS_width = params.Level_Set["Transition_Width"]
            if self.LS_width <= 0.0:
                raise ValueError("Glaciology:ShallowIce Level_Set_Transition_Width must be a positive finite number.")
        else:
            self.use_level_set_mask = False
        
        self.step_V = params.Solution_Steps["v"]
        self.step_H = params.Solution_Steps["h"]
        
        # Fields being solved for
        mesh.Add_Vector_Field("v", "Velocity [m/s]", "CG", params.Field_Orders["v"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("vOld", "Velocity [m/s]", "CG", params.Field_Orders["v"], params.dim, domain=self.domain)
        mesh.Add_Field("h", "Ice Thickness [m]", "DG", params.Field_Orders["h"], domain=self.domain)
        mesh.Add_Field("hOld",  "Ice Thickness [m]", "DG", params.Field_Orders["h"], domain=self.domain)
        mesh.Add_Field("hOld2", "Ice Thickness [m]", "DG", params.Field_Orders["h"], domain=self.domain)
        # BDF2 time integration: inactive on the first step (BDF1 used instead);
        self._bdf2_active = False

        mesh.Add_Vector_TestFunction("v", params.dim, self.step_V)
        mesh.Add_TestFunction("h", self.step_H)
        
        # basal geometry
        mesh.Add_Field("z", "Basal Elevation [m]", "CG", max(params.Field_Orders.get("h", 1), 1), domain=self.domain)
        if self.use_level_set_mask:
            if "LS" not in params.Field_Orders:
                raise ValueError("Glaciology:ShallowIce LS active-domain masking requires Field_Orders['LS'].")
            mesh.Add_Field("LS_phase", "Level Set Phase Indicator [-]", "CG", params.Field_Orders["LS"], domain=self.domain)
            mesh.Add_Field("LS_phase_h", "Level Set Phase Indicator on h space [-]", "DG", params.Field_Orders["h"], domain=self.domain)
            mesh.Add_Field("LS_phaseOld_h", "Old Level Set Phase Indicator on h space [-]", "DG", params.Field_Orders["h"], domain=self.domain)
        
        # Fields for material properties
        self.porosity = 0.0
        
        Temp_order = params.Field_Orders.get("Temperature", None)
        if Temp_order is None:
            Temp_order = params.Field_Orders.get("T", 1)
        mesh.Add_Field("T", "Temperature [C]", "CG", Temp_order, domain=self.domain)
        
        # material parameters
        self.iceMat = IceMaterial()
        
        water_params = self.params.Materials.get("Water", {})
        self.rho_ocean = water_params.get("Density", 1030.0)
        self.sea_level = water_params.get("WaterHeight", 0.0)
        self.gravity = self.params.gravity
        self.gravity_magnitude = abs(float(self.gravity))
        ice_params = self.params.Materials["Ice"]
        self.basal_friction = ice_params["C_base"]
        self.apply_front_ocean_stress = bool(ice_params.get("Front_Ocean_Stress", True))
        self.friction_model = ice_params.get("Friction_Model", "Budd").lower()
        if self.friction_model not in ("budd", "weertman"):
            raise ValueError(f"Unknown ShallowIce friction model: '{self.friction_model}'. Supported values: 'Budd', 'Weertman'.")
        self.visc_min = ice_params.get("visc_min", 1.0e3)
        self.visc_max = ice_params.get("visc_max", 1.0e20)
        self.strain_rate_regularization = ice_params.get("strain_rate_regularization", 1.0e-20)
        self._bind_ice_material_state()

    def Post_Remesh(self) -> None:
        """Rebuild temperature-dependent properties on the live mesh."""
        self._bind_ice_material_state()

    def _bind_ice_material_state(self) -> None:
        """PRIVATE: Build shallow-ice material expressions from the live temperature."""
        temperature = self.mesh.Get_Field("T")
        self.rho = self.iceMat.Get_Density(self.porosity, temperature)
        self.creep_A = self.iceMat.Get_Creep_A(self.porosity, temperature)
        self.creep_n = self.iceMat.Get_Creep_n(self.porosity, temperature)

    def compute_viscosity(self, strain_rate):
        """Return the Glen-law membrane viscosity for the SSA strain rate."""
        strain_rate_eff = sqrt(0.5 * (inner(strain_rate, strain_rate) + tr(strain_rate) ** 2)) + self.strain_rate_regularization
        eta = 0.5 * self.creep_A ** (-1.0 / self.creep_n)
        eta *= strain_rate_eff ** ((1.0 - self.creep_n) / self.creep_n)
        eta = conditional(lt(eta, self.visc_min), self.visc_min, eta)
        eta = conditional(lt(self.visc_max, eta), self.visc_max, eta)
        return eta

    def _basal_friction_coefficient(self, v, h, v_tr, h_tr, z):
        """Return the basal drag coefficient beta for the configured law."""
        C = 0.0
        dC_dh = 0.0
        dC_dv = 0.0
        if self.friction_model == "budd":
            water_depth_at_bed = conditional(lt(z, self.sea_level),self.sea_level - z,0.0,)
            effective_pressure = -self.gravity * (self.rho * h - self.rho_ocean * water_depth_at_bed)
            effective_pressure_tr = -self.gravity * self.rho * h_tr
            
            C = self.basal_friction * effective_pressure * v
            dC_dh = self.basal_friction * effective_pressure_tr * v
            dC_dv = self.basal_friction * effective_pressure * v_tr
        elif self.friction_model == "weertman":
            C = self.basal_friction*v
            dC_dh = 0.0
            dC_dv = self.basal_friction*v_tr
        else:  # pragma: no cover - constructor rejects invalid friction models.
            raise ValueError(f"Unknown ShallowIce friction model: {self.friction_model}")
        
        return C, dC_dh, dC_dv

    def _get_active_domain_weight(self, old: bool = False):
        """Return the active-domain phase weight used by the shallow-ice terms."""
        if not self.use_level_set_mask:
            return 1.0
        LS = self.mesh.Get_Field("LS") if not old else self.mesh.Get_Field("LSOld")
        phase = smooth_level_set_heaviside(LS, self.LS_width)
        return phase
    
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if step != self.step_V and step != self.step_H:
            return

        # --- Fields, height-averaged velocity, ice thickness, basal geometry ---
        v    = self.mesh.Get_Field("v",    step=self.step_V)
        h    = self.mesh.Get_Field("h",    step=self.step_H)
        hOld = self.mesh.Get_Field("hOld")
        z    = self.mesh.Get_Field("z")
        active_domain = self._get_active_domain_weight(False)

        # --- Test / trial functions ---
        v_t  = self.mesh.Get_TestFunction("v", self.step_V)
        h_t  = self.mesh.Get_TestFunction("h", self.step_H)
        v_tr = self.mesh.Get_Trial_Function("v", self.step_V)
        h_tr = self.mesh.Get_Trial_Function("h", self.step_H)

        normals = FacetNormal(self.mesh.mesh)
        dx = self.mesh.Get_Domain_Integrator(self.domain)
        
        # General indicator
        hasH = conditional(le(h, 0.0), 0.0, 1.0)
        
        density_ratio = self.rho / self.rho_ocean
        water_depth_at_bed = conditional(lt(z, self.sea_level),self.sea_level - z,0.0,)
        effective_pressure = -self.gravity * (self.rho * h - self.rho_ocean * water_depth_at_bed)
        grounded = conditional(lt(0.0, effective_pressure), 1.0, 0.0)

        F = 0.0
        K = 0.0

        ## mass conservation (DG upwind transport):
        if step == self.step_H:
            # height capacity term
            if self.UseLumped == False:
                if self._bdf2_active and self.TimeDiscretization == "BDF2":
                    hOld2 = self.mesh.Get_Field("hOld2")
                    F += h_t * (3.0*h - 4.0*hOld + hOld2) / (2.0*dt_constant) * dx
                    K += h_t * 3.0*h_tr / (2.0*dt_constant) * dx
                else:
                    F += h_t * (h - hOld) / dt_constant * dx
                    K += h_t * h_tr / dt_constant * dx

                # penalty term to keep h >= 0
                F += 1.0e9 * h_t * (1.0 - hasH) * h * dx
                K += 1.0e9 * h_t * (1.0 - hasH) * h_tr * dx

            # Conservative volume flow term for the active ice amount H*h.
            F += -inner(active_domain * h * v, grad(h_t)) * dx
            K += -inner(active_domain * h_tr * v, grad(h_t)) * dx
            if step == self.step_V:
                K += -inner(active_domain * h * v_tr, grad(h_t)) * dx

            # DG Correction terms (upwind fluxes) for interior facets
            dS_int, ds, dS_edge = self._get_facet_integrators()
            n = normals  # alias for clarity
            v_n_int = dot(v('+'), n('+'))
            ice_up = conditional(gt(v_n_int, 0.0), h('+'), h('-'))
            ice_up_tr = conditional(gt(v_n_int, 0.0), h_tr('+'), h_tr('-'))

            F += ice_up * v_n_int * active_domain* jump(h_t) * dS_int
            K += ice_up_tr * v_n_int * active_domain* jump(h_t) * dS_int
            if step == self.step_V:
                K += ice_up * dot(v_tr('+'), n('+')) * active_domain* jump(h_t) * dS_int

            # Exterior natural flux: use interior h at all boundary facets.
            # ShallowIceBC adds a thickness penalty on prescribed boundaries.
            if ds is not None:
                F += active_domain *h * dot(v, normals) * h_t * ds
                K += active_domain * h_tr * dot(v, normals) * h_t * ds
                if step == self.step_V:
                    K += active_domain * h * dot(v_tr, normals) * h_t * ds
            if dS_edge is not None:
                # Facets between the model's subdomain and the rest of the
                # mesh; "+" is the subdomain side, so n("+") points outwards.
                active_edge = active_domain("+") if self.use_level_set_mask else active_domain
                outflow = active_edge * dot(v("+"), n("+")) * h_t("+")
                F += outflow * h("+") * dS_edge
                K += outflow * h_tr("+") * dS_edge
                if step == self.step_V:
                    K += active_edge * h("+") * dot(v_tr("+"), n("+")) * h_t("+") * dS_edge
        
        ## Momentum conservation:
        if step == self.step_V:
            if self.use_level_set_mask:
                inactive_domain = 1.0 - active_domain
                h_momentum = active_domain * h + inactive_domain * h_fict
                h_momentum_tr = active_domain * h_tr
            else:
                inactive_domain = 0.0
                h_momentum = h
                h_momentum_tr = h_tr
            
            # Membrane stresses
            strain_rate = nabla_s(v)
            strain_rate_tr = nabla_s(v_tr)
            
            membrane_strain_rate = strain_rate + tr(strain_rate) * Identity(self.dim)
            membrane_strain_rate_tr = strain_rate_tr + tr(strain_rate_tr) * Identity(self.dim)
            
            viscosity = self.compute_viscosity(strain_rate)
            
            F += inner(nabla_s(v_t), 2.0 * viscosity * h_momentum * membrane_strain_rate) * dx
            K += inner(nabla_s(v_t), 2.0 * viscosity * h_momentum * membrane_strain_rate_tr) * dx
            
            if step == self.step_H:
                K += inner(nabla_s(v_t), 2.0 * viscosity * h_momentum_tr * membrane_strain_rate) * dx
            
            # Gravity: 
            # grounded ice uses the bed-following surface z + h,
            # floating ice uses hydrostatic flotation.
            grounded_surface = z + h
            floating_surface = self.sea_level + (1.0 - density_ratio) * h
            surface = grounded * grounded_surface + (1.0 - grounded) * floating_surface
            surface_tr = (grounded + (1.0 - grounded) * (1.0 - density_ratio)) * h
            
            F += - active_domain * self.rho * self.gravity * h * dot(v_t, grad(surface)) * dx
            if step == self.step_H:
                K += - active_domain * self.rho * self.gravity * (h_tr * dot(v_t, grad(surface)) + h * dot(v_t, grad(surface_tr))) * dx

            # Front water-pressure
            if self.use_level_set_mask and self.apply_front_ocean_stress:
                grounded_water_depth_raw = self.sea_level - z
                grounded_water_depth = conditional(lt(grounded_water_depth_raw, 0.0),0.0,conditional(lt(h, grounded_water_depth_raw), h, grounded_water_depth_raw),)
                grounded_water_depth_tr = conditional(lt(grounded_water_depth_raw, 0.0),0.0,conditional(lt(h, grounded_water_depth_raw), h_tr, 0.0),)

                floating_water_depth = density_ratio * h
                floating_water_depth_tr = density_ratio * h_tr
                water_depth = grounded * grounded_water_depth + (1.0 - grounded) * floating_water_depth
                water_depth_tr = grounded * grounded_water_depth_tr + (1.0 - grounded) * floating_water_depth_tr

                pressure = 0.5 * self.gravity_magnitude * (self.rho * h * h - self.rho_ocean * water_depth * water_depth)
                pressure_tr = self.gravity_magnitude * (self.rho * h * h_tr - self.rho_ocean * water_depth * water_depth_tr)
                
                F += pressure * dot(v_t, grad(active_domain)) * dx
                if step == self.step_H:
                    K += pressure_tr * dot(v_t, grad(active_domain)) * dx
                
            # Basal friction
            basal_friction, dC_dh, dC_dv = self._basal_friction_coefficient(v, h, v_tr, h_tr, z)
            
            F += active_domain *  grounded * dot(v_t, basal_friction) * dx
            K += active_domain * grounded * dot(v_t, dC_dv) * dx
            if step == self.step_H:
                K += active_domain * grounded * dot(v_t, dC_dh) * dx    
                
            # preventing ill-conditioning when h<= 0 or when LS < 0 (inactive domain)
            drag_h0 = 10.0 # large enough to force v=0 when h=0 or LS < 0, but not too large to impact solution in 0<inactive domain < 1.0 zone
            if self.use_level_set_mask:
                F += drag_h0 * inactive_domain * dot(v_t, v) * dx
                K += drag_h0 * inactive_domain * dot(v_t, v_tr) * dx
            else:
                F += drag_h0 * (1.0 - hasH) * dot(v_t, v) * dx
                K += drag_h0 * (1.0 - hasH) * dot(v_t, v_tr) * dx

        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        structure_state = (self._bdf2_active, self.TimeDiscretization)
        previous_state = getattr(self, "_weak_structure_state", None)
        self._weak_structure_state = structure_state
        return previous_state is not None and previous_state != structure_state
    
    def Lumped_H(self, K, F, step: int) -> None:
        """Add lumped thickness terms; ``K`` may be absent for force-only assembly."""
        dt = self.params.dt

        dofNames = ["h"]
        OldNames = ["hOld", "hOld2"] if self._bdf2_active and self.TimeDiscretization == "BDF2" else ["hOld"]
        W, dofs, vals = self.mesh.Get_Weights_Dofs(dofNames, step=self.step_H)
        valsOld = self.mesh.Get_Old_For_Lumped(OldNames, step=self.step_H)

        n = len(W)
        k_indices = np.empty(n, dtype=np.int32)
        k_values  = np.empty(n)

        with F.localForm() as F_local:
            F_array = F_local.array

            for i in range(n):
                idx  = dofs["h"][i]
                h    = vals["h"][i]
                h_old = valsOld["hOld"][i]

                k_indices[i] = idx

                if self._bdf2_active and self.TimeDiscretization == "BDF2":
                    h_old2 = valsOld["hOld2"][i]
                    F_array[idx] += W[i] * (3.0*h - 4.0*h_old + h_old2) / (2.0*dt)
                    k_values[i]   = W[i] * 3.0 / (2.0*dt)
                else:
                    F_array[idx] += W[i] * (h - h_old) / dt
                    k_values[i]   = W[i] * 1.0 / dt

                if h <= 0.0:
                    F_array[idx] += 1.0e9 * W[i] * h
                    k_values[i]  += 1.0e9 * W[i]

        # Apply K contributions OUTSIDE the localForm() context.
        # Mixing Vec.localForm() with MatSetValuesLocal in the same block
        # can trigger PETSc MPI state conflicts that silently corrupt the
        # matrix in parallel (works on 1 rank, diverges on N > 1).
        if K is not None:
            global_indices = self.mesh.get_step_global_indices(step, k_indices)
            for i in range(n):
                idx = global_indices[i]
                K.setValues([idx], [idx], [[k_values[i]]], addv=PETSc.InsertMode.ADD)
    
    def Add_KF(self, K, F, step: int) -> None:
        """
        Add algebraic terms beyond those defined in ``Get_Weak_Form``.

        ``K`` may be ``None`` during residual-only assembly.
        """
        
        if self.UseLumped == False:
            return  # Already included in Get_Weak_Form.
        
        if (step == self.step_H):
            self.Lumped_H(K, F, step)
            return
    
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        for field_name in ("v", "vOld", "h", "hOld", "hOld2"):
            self.mesh.Zero_Field(field_name, domain=self.domain)
        self._bdf2_active = False
            
    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """

        self.mesh.Copy_Field("v", "vOld", domain=self.domain)
        # BDF2 history shift: hOld2 <- hOld <- h.
        # After the first commit two past levels are available; all subsequent
        # steps use the second-order BDF2 dh/dt stencil.
        self.mesh.Copy_Field("hOld", "hOld2", domain=self.domain)
        self.mesh.Copy_Field("h", "hOld", domain=self.domain)
        self._bdf2_active = True
        
    def reset_step(self):
        self.mesh.Copy_Field("hOld", "h", domain=self.domain)
        self.mesh.Copy_Field("vOld", "v", domain=self.domain)

    def _get_facet_integrators(self):
        """PRIVATE: Return ``(dS_interior, ds_exterior, dS_edge)`` of the model's domain.

        On the full mesh these are the plain interior/exterior facet
        measures and ``dS_edge`` is ``None``. On a subdomain, ``dS_interior``
        covers facets with both cells inside it, ``ds_exterior`` the mesh
        boundary facets of it, and ``dS_edge`` the facets shared with cells
        outside it, ordered with the subdomain cell as the "+" side.
        ``ds_exterior`` is ``None`` if the subdomain does not touch the mesh
        boundary.

        ``ds_exterior`` uses the mesh's tagged boundary measure, so that it
        can be summed with boundary-model terms (DOLFINx needs one set of
        subdomain data per integral type). This requires every boundary tag
        touching the subdomain to lie entirely on it.
        """
        full_measures = (self.mesh.Get_Interior_Facet_Integrator(), Measure("ds", domain=self.mesh.mesh), None)
        if self.domain == "domain":
            return full_measures
        tdim = self.mesh.mesh.topology.dim
        domain_cells = np.asarray(self.mesh.get_domain_cells(self.domain), dtype=np.int32)
        if set(self.mesh.get_domain_tags(self.domain)) >= set(self.mesh.domain_markers.values()):
            return full_measures
        topology = self.mesh.mesh.topology
        topology.create_connectivity(tdim, tdim - 1)
        topology.create_connectivity(tdim - 1, tdim)
        facets = np.asarray(compute_incident_entities(topology, domain_cells, tdim, tdim - 1), dtype=np.int32)
        mesh_exterior = exterior_facet_indices(topology)
        on_boundary = np.isin(facets, mesh_exterior)

        interior = fem.compute_integration_domains(fem.IntegralType.interior_facet, topology, facets[~on_boundary]).reshape(-1, 4)
        first_in, second_in = np.isin(interior[:, 0], domain_cells), np.isin(interior[:, 2], domain_cells)
        inner_entities = interior[first_in & second_in]
        edge_entities = interior[first_in != second_in]
        swap = second_in[first_in != second_in]
        edge_entities[swap] = edge_entities[swap][:, [2, 3, 0, 1]]
        dS = Measure("dS", domain=self.mesh.mesh, subdomain_data=[(1, inner_entities.ravel()), (2, edge_entities.ravel())])
        return dS(1), self._get_exterior_integrator(facets[on_boundary], mesh_exterior), dS(2)

    def _get_exterior_integrator(self, domain_exterior: np.ndarray, mesh_exterior: np.ndarray):
        """PRIVATE: Return the tagged ``ds`` over exactly the given exterior facets of the domain.

        Returns ``None`` when the domain has no exterior facets.
        """
        facet_tags = self.mesh.facet_tags
        tagged = np.isin(domain_exterior, facet_tags.indices)
        local_tags = np.unique(facet_tags.values[np.isin(facet_tags.indices, domain_exterior)])
        comm = self.mesh.mesh.comm
        tags = sorted({int(tag) for rank_tags in comm.allgather(local_tags) for tag in rank_tags})
        spills = any(
            not np.isin(np.intersect1d(facet_tags.find(tag), mesh_exterior), domain_exterior).all() for tag in tags
        )
        if comm.allreduce(int(spills or not tagged.all())):
            raise ValueError(
                f"ShallowIce '{self.name}' on '{self.domain}' needs every outer boundary facet of its domain "
                "tagged, with boundary tags that do not extend beyond the domain."
            )
        if not tags:
            return None  # the domain does not touch the mesh boundary
        return self.mesh.get_Boundary_Integrator()(tuple(tags))
