from ufl import div, inner, Identity
from dolfinx import fem
from petsc4py import PETSc
import numpy as np
from femcubes.core.base_model import BaseModel
from femcubes.core.utils.maths_utils import nabla_s
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.glaciology.ice_material import IceMaterial
from femcubes.poromechanics.shared_poroelasticity import (
    saturation_from_pressure,
    saturation_relations,
    single_saturation_from_pressure,
    validate_saturation_parameters,
)
from ufl import lt, grad
import ufl

class PorePressure(BaseModel):
    r"""Biot pore pressure with Darcy flow, saturated or unsaturated.

    The model describes a porous solid whose pores are fully or partly
    filled with a fluid, following the theory of poroelasticity of
    :cite:t:`Biot1941`. The pore pressure acts on the solid skeleton through
    the Biot coefficient, while the deformation of the skeleton changes the
    pore volume and thereby the fluid content. The fluid flows according to
    Darcy's law, with a permeability that increases with the porosity in the
    Kozeny-Carman form :cite:p:`Carman1997`. In unsaturated media, the
    saturation and the relative permeability follow from the capillary
    pressure through the relations of :cite:t:`vanGenuchten1980` and
    :cite:t:`Mualem1976`.

    The model couples the displacement ``u`` of a solid model to the pore
    pressure ``pw``: the pore pressure enters the momentum balance, and the
    mass balance of the pore fluid has Darcy flow, storage and a volumetric
    strain-rate term. Use together with a solid model on the same domain
    that solves and commits ``u`` (e.g. :doc:`../solids/linear_elastic` or
    :doc:`../glaciology/ice_viscoelastic`); fracture flow can be added with
    :doc:`fracture_flow`. Supports 2-D (plane strain) and 3-D, not
    axisymmetric meshes. ``u`` and ``pw`` may be solved in the same step
    (monolithic) or in separate staggered steps.

    .. rubric:: Fields

    * ``pw``: pore pressure :math:`p_w`, CG of order ``Field_Orders["pw"]``,
      solved in step ``Solution_Steps["pw"]``; ``pwOld`` holds :math:`p_{w,n}`.
    * ``u``, ``uOld``: displacement and its committed value, vector CG of
      order ``Field_Orders["u"]``. Also registered here, but they must be
      solved and committed by the solid model (``commit`` of this model
      does not update ``uOld``).
    * ``poros``: porosity :math:`n`, DG of order ``Field_Orders["pw"]``.
      Set to ``Initial_Porosity`` at initialization if given, otherwise it
      keeps the value set by another model or user code (zero by default).
      Held fixed by this model.
    * ``qw``: output Darcy flux :math:`\mathbf{q}_w`, vector DG of order
      ``Field_Orders["pw"] - 1``.
    * ``Sw``: output saturation :math:`S_w`, DG of order ``Field_Orders["pw"]``;
      only with ``Saturated=False``.

    .. rubric:: Weak form

    Symbols: :math:`\delta\mathbf{u}`, :math:`\delta p_w` test functions,
    :math:`\Omega` the domain group of ``domain``, :math:`\rho_w`,
    :math:`\mu_w`, :math:`K_w` density, viscosity and bulk modulus of the
    fluid (material ``Fluid``), :math:`K_s` the solid bulk modulus,
    :math:`g` = ``gravity`` along the last coordinate :math:`x_d`, and
    :math:`\mathbf{g} = g\,\mathbf{e}_d`.

    Biot coefficient and intrinsic permeability (Kozeny-Carman form):

    .. math::

        \alpha = \begin{cases}\texttt{Biot} & \text{if given}\\
        1-(1-n/n_c)^{q} & \text{otherwise}\end{cases},
        \qquad
        k = k_0\,\frac{(n/n_c)^3}{(1-n/n_c)^2},

    with :math:`n_c` = ``poros_crit``, :math:`q` = ``Biot_n`` and
    :math:`k_0` = ``permeability0``.

    Saturation: :math:`S_w = 1`, :math:`k_r = 1` with ``Saturated=True``.
    With ``Saturated=False`` (van Genuchten with the Mualem constraint
    :math:`n_\mathrm{vG} = 1/(1-m)`, capillary pressure
    :math:`p_c = \max(-p_w, 0)`, see :doc:`shared_poroelasticity`):

    .. math::

        S_w = \bigl(1 + (p_c/p_\mathrm{ref})^{n_\mathrm{vG}}\bigr)^{-m},
        \qquad
        k_r = \begin{cases}
        \hat S^{1/2}\bigl(1-(1-\hat S^{1/m})^{m}\bigr)^2 + 10^{-6} & S_{w,n} < 0.99\\
        1 & S_{w,n} \geq 0.99\end{cases},

    with :math:`\hat S = \min(\max(S_{w,n}, 10^{-10}), 0.99)`. The relative
    permeability uses the committed saturation :math:`S_{w,n} = S_w(p_{w,n})`
    in both residual and Jacobian.

    Contribution to the residual of ``u`` (step ``Solution_Steps["u"]``):

    .. math::

        R_{\mathbf{u}} = -\int_\Omega \alpha\,S_{w,n}\,p_w\,\mathrm{tr}\,\boldsymbol{\varepsilon}(\delta\mathbf{u})\,\mathrm{d}x
        - \int_\Omega S_{w,n}\,\rho_w\,n\,g\,\delta u_d\,\mathrm{d}x,

    the second term only for :math:`g \neq 0`. The pressure derivative of
    the first term enters the Jacobian only when ``u`` and ``pw`` share a
    step.

    Contribution to the residual of ``pw`` (step ``Solution_Steps["pw"]``):

    .. math::

        R_{p_w} = \int_\Omega \frac{k\,k_r}{\mu_w}\,\nabla\delta p_w\cdot(\nabla p_w - \rho_w\mathbf{g})\,\mathrm{d}x
        + \int_\Omega \frac{\alpha\,S_w}{\Delta t}\,\delta p_w\,\nabla\cdot(\mathbf{u}-\mathbf{u}_n)\,\mathrm{d}x
        + \sum_i \delta p_{w,i}\,W_i\,\frac{m_w(p_{w,i}) - m_w(p_{w,n,i})}{\Delta t},

    with the stored water content (reference density factored out)

    .. math::

        m_w(p_w) = n\,S_w\Bigl(1 + \frac{p_w}{K_w}\Bigr) + \frac{\alpha - n}{K_s}\,p_w\,S_w^2 ,

    :math:`S_w = S_w(p_w)` at the current iterate. The storage term is
    lumped (class attribute ``USE_LUMPED = True``): it is added node by node
    to the assembled vector and matrix, with lumped weights
    :math:`W_i = \int_\Omega N_i\,\mathrm{d}x` and :math:`S_w`, :math:`n`,
    :math:`\alpha` evaluated at node :math:`i`. With ``USE_LUMPED = False``
    the same term is integrated as
    :math:`\int_\Omega \delta p_w\,(m_w(p_w) - m_w(p_{w,n}))/\Delta t\,\mathrm{d}x`.
    The Jacobian is consistent in :math:`p_w` (it includes
    :math:`\partial S_w/\partial p_w` in the storage and strain-rate terms);
    the derivative with respect to ``u`` is included only when ``u`` and
    ``pw`` share a step. The flux is
    :math:`\mathbf{q}_w = -k\,k_r/\mu_w\,(\nabla p_w - \rho_w\mathbf{g})`.

    .. rubric:: Outside the weak form

    * Before every assembly, :math:`S_w`, :math:`\partial S_w/\partial p_w`
      and :math:`S_{w,n}` are rebuilt from ``pw`` and ``pwOld``, ``Sw`` is
      updated, and (in the ``pw`` step) ``qw`` is recomputed.
    * The lumped storage requires this model to be the only one registering
      ``pw``; otherwise assembly raises ``NotImplementedError``.
    * ``commit`` copies ``pw`` to ``pwOld``; ``reset_step`` restores it.
    * No Dirichlet conditions; prescribe ``pw`` with
      :doc:`../boundary/prescribed_value`.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                                # REQUIRED: 2 or 3
            Field_Orders = {"u": 2, "pw": 1}       # REQUIRED: orders of u and pw
            Solution_Steps = {"u": 0, "pw": 0}     # REQUIRED: steps of u and pw (equal: monolithic)
            gravity = -9.81                        # REQUIRED: acceleration along the last axis (m/s^2)
            Materials = {
                "Firn": {
                    "Bulk": 8.9e9,                 # REQUIRED (OPTIONAL for a material named "Ice"): solid bulk modulus K_s (Pa);
                                                   # for "Ice" the default is the non-porous ice value of IceMaterial
                    "permeability0": 1.0e-10,      # REQUIRED: permeability k_0 (m^2)
                    "poros_crit": 0.8,             # OPTIONAL (default 1.0): critical porosity n_c (-)
                    "Biot_n": 1.0,                 # OPTIONAL (default 2): exponent q of the Biot coefficient (-)
                    "Biot": 1.0,                   # OPTIONAL (default None, from porosity): constant Biot coefficient (-)
                    "Saturation_p_ref": 3.0e3,     # OPTIONAL (default 1e6), Saturated=False only: van Genuchten pressure (Pa)
                    "Saturation_m": 0.6,           # OPTIONAL (default 0.5), Saturated=False only: van Genuchten m, 0 < m < 1 (-)
                },
                "Water": {
                    "Density": 1000.0,             # REQUIRED: fluid density rho_w (kg/m^3)
                    "Bulk": 2.2e9,                 # REQUIRED: fluid bulk modulus K_w (Pa)
                    "Viscosity": 1.0e-3,           # REQUIRED: dynamic viscosity mu_w (Pa s)
                },
            }
            Models = {
                "Fluid": {
                    "Type": "PoroMechanics:PorePressure",  # REQUIRED
                    "Material": "Firn",            # REQUIRED: key in Materials (porous solid)
                    "Fluid": "Water",              # REQUIRED: key in Materials (pore fluid)
                    "Saturated": False,            # OPTIONAL (default True): unsaturated flow with van Genuchten saturation
                    "Initial_Porosity": 0.5,       # OPTIONAL (default None, poros left as set elsewhere): initial porosity (-)
                    "domain": "domain",            # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """
 
    MODEL_TYPE = "PoroMechanics:PorePressure"
    
    USE_LUMPED = True
    USE_MASS_CAPACITY = True

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        
        mesh.Add_Vector_Field("u", "Displacement [m]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)
        mesh.Add_Vector_Field("uOld", "Old Displacement [m]", "CG", params.Field_Orders["u"], params.dim, domain=self.domain)

        mesh.Add_Field("pw", "Pore Pressure [Pa]", "CG", params.Field_Orders["pw"], domain=self.domain)
        mesh.Add_Field("pwOld", "Old Pore Pressure [Pa]", "CG", params.Field_Orders["pw"], domain=self.domain)

        mesh.Add_Field("poros", "Porosity [-]", "DG", params.Field_Orders["pw"], domain=self.domain)
        mesh.Add_Vector_Field("qw", "Darcy Flux [m/s]", "DG", params.Field_Orders["pw"]-1, params.dim, domain=self.domain)

        self.Step_u = params.Solution_Steps["u"]
        self.Step_pw = params.Solution_Steps["pw"]
        
        # Register test functions (actual test functions will be retrieved when needed)
        mesh.Add_Vector_TestFunction("u", params.dim, self.Step_u)
        mesh.Add_TestFunction("pw", self.Step_pw)
        
        # material parameters
        self.material = self.params.Models[self.name]["Material"]
        self.fluid    = self.params.Models[self.name]["Fluid"]
        
        self.Saturated = self.params.Models[self.name].get("Saturated", True)
        
        
        self.bulk_s = self.params.Materials[self.material].get("Bulk", None) # default solid bulk modulus (Pa)
        if self.bulk_s is None and self.material == "Ice":
            IceMat = IceMaterial()
            self.bulk_s = IceMat.Get_Bulk_Modulus(porosity=0.0, temperature=0.0)
        elif self.bulk_s is None:
            raise ValueError(f"Bulk modulus must be specified for material '{self.material}' in poromechanics model.")
        
        
        self.bulk_f = self.params.Materials[self.fluid]["Bulk"]
        self.rho_f = self.params.Materials[self.fluid]["Density"]
        
        self.visc = self.params.Materials[self.fluid]["Viscosity"]
        self.permeability0 = self.params.Materials[self.material]["permeability0"]
        self.Biot_n = self.params.Materials[self.material].get("Biot_n", 2)
        self.poros_crit = self.params.Materials[self.material].get("poros_crit", 1.0)
        
        self._configured_biot = self.params.Materials[self.material].get("Biot", None)
        self._create_mesh_constants()

        if not self.Saturated:
            
            # additional parameters required
            self.p_ref = self.params.Materials[self.material].get("Saturation_p_ref", 1e6)  # reference pressure for saturation curve (Pa)
            self.m = self.params.Materials[self.material].get("Saturation_m", 0.5)
            validate_saturation_parameters(self.p_ref, self.m)
            
            # add field for visualization of saturation
            mesh.Add_Field("Sw", "Saturation [-]", "DG", params.Field_Orders["pw"], domain=self.domain)
        
        self.gravity = self.params.gravity
        self.gVec = ufl.as_vector([0.0, self.gravity]) if params.dim == 2 else ufl.as_vector([0.0, 0.0, self.gravity])
        
        self.initial_poros = self.params.Models[self.name].get("Initial_Porosity", None)

    def Post_Remesh(self) -> None:
        """Recreate mesh-domain constants while preserving scalar values."""
        self._create_mesh_constants(preserve_existing=True)

    def _create_mesh_constants(self, preserve_existing: bool = False) -> None:
        """PRIVATE: Create all scalar constants on the current mesh domain."""
        defaults = {
            "Sw": 1.0 if self.Saturated else 0.0,
            "SwOld": 1.0 if self.Saturated else 0.0,
            "dSw": 0.0,
            "ddSw": 0.0,
        }

        def _constant_value(name, default):
            current = getattr(self, name, None)
            if preserve_existing and isinstance(current, fem.Constant):
                return np.asarray(current.value).reshape(-1)[0]
            return default

        if self._configured_biot is None:
            self.Biot = None
        else:
            biot_value = _constant_value("Biot", self._configured_biot)
            self.Biot = fem.Constant(self.mesh.mesh, biot_value)

        for name, default in defaults.items():
            setattr(self, name, fem.Constant(self.mesh.mesh, _constant_value(name, default)))
    
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """
        Assemble the global stiffness matrix and force vector for the linear elastic model.
        """
        
        if (step != self.Step_u) and (step != self.Step_pw):
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)

        # Get fields - mesh automatically returns mixed space version if applicable
        u = self.mesh.Get_Field("u", step=self.Step_u)
        uOld = self.mesh.Get_Field("uOld")
        pw = self.mesh.Get_Field("pw", step=self.Step_pw)
        pwOld = self.mesh.Get_Field("pwOld")
        
        poros = self.mesh.Get_Field("poros")
        
        u_t = self.mesh.Get_TestFunction("u", self.Step_u)
        u_tr = self.mesh.Get_Trial_Function("u", self.Step_u)
        
        pw_t = self.mesh.Get_TestFunction("pw", self.Step_pw)
        pw_tr = self.mesh.Get_Trial_Function("pw", self.Step_pw)
        
        if self.Biot is None:
            Biot = 1.0-(1.0-poros/self.poros_crit)**self.Biot_n
        else:
            Biot = self.Biot
        
        if step == self.Step_u:
            # Fluid pressure term in Momentum Balance
            
            if False:
                Sw_u = self.Sw
                dSw_u = self.dSw
                impl_u = True
            else:
                Sw_u = self.SwOld
                dSw_u = 0.0
                impl_u = False
                
            Fu = inner(nabla_s(u_t), -Sw_u * Biot * pw * Identity(self.params.dim)) * dx
            if self.Step_pw == self.Step_u:
                # d(Sw * Biot * pw)/dpw * delta_pw = (Sw + pw * dSw) * Biot * delta_pw
                Ku = inner(nabla_s(u_t), -Sw_u * Biot * pw_tr * Identity(self.params.dim)) * dx
                if impl_u:
                    Ku += inner(nabla_s(u_t), -pw * dSw_u * Biot * pw_tr * Identity(self.params.dim)) * dx
            else:
                Ku = None
                
            if self.gravity != 0.0: # add fluid weight
                Fu -= inner(u_t[self.params.dim-1], Sw_u * self.rho_f * poros * self.gravity) * dx
                if self.Step_pw == self.Step_u and impl_u:
                    Ku -= inner(u_t[self.params.dim-1], dSw_u * self.rho_f * poros * self.gravity * pw_tr) * dx
                    
        if step == self.Step_pw:
            # Pressure-driven porous flow
            flux = self._weak_flux
            dflux_dpw = self._weak_flux_tangent
            Fp = -inner(grad(pw_t), flux) * dx
            Kp = -inner(grad(pw_t), dflux_dpw) * dx
            
            # Capacity term
            if self.USE_LUMPED == False:
                if self.USE_MASS_CAPACITY:
                    water_content = self._water_content(pw, self.Sw, poros, Biot)
                    water_content_old = self._water_content(pwOld, self.SwOld, poros, Biot)
                    dwater_content_dpw = self._water_content_pressure_derivative(pw, self.Sw, self.dSw, poros, Biot)

                    Fp += pw_t * (water_content - water_content_old) / dt_constant * dx
                    Kp += pw_t * dwater_content_dpw * pw_tr / dt_constant * dx
                else:
                        if self.Saturated == True:
                            Minv = (Biot - poros) / self.bulk_s + poros / self.bulk_f
                            dMinv_dpw = 0.0
                        else:
                            Minv = (self.Sw * (Biot - poros) / self.bulk_s * (self.Sw + pw * self.dSw)
                                    + poros * self.dSw
                                    + self.Sw * poros / self.bulk_f)
                            dMinv_dpw = (
                                self.dSw * (Biot - poros) / self.bulk_s * (self.Sw + pw * self.dSw)
                                + self.Sw * (Biot - poros) / self.bulk_s * (2.0 * self.dSw + pw * self.ddSw)
                                + poros * self.ddSw
                                + self.dSw * poros / self.bulk_f
                            )

                        Fp += pw_t * Minv * (pw - pwOld) / dt_constant * dx
                        Kp += pw_t * (Minv + dMinv_dpw * (pw - pwOld)) * pw_tr / dt_constant * dx
            
            # Compression term
            Fp += Biot * self.Sw * pw_t/dt_constant * div((u - uOld)) * dx
            Kp += Biot * self.dSw * pw_t * pw_tr / dt_constant * div((u - uOld)) * dx
            if self.Step_u == self.Step_pw:
                Kp += Biot * self.Sw * pw_t/dt_constant * div(u_tr) * dx
            
        if step == self.Step_u:
            K = Ku
            F = Fu
        if self.Step_u == self.Step_pw:
            K += Kp
            F += Fp
        elif step == self.Step_pw:
            K = Kp
            F = Fp
        
        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        if step != self.Step_u and step != self.Step_pw:
            return False
        pw = self.mesh.Get_Field("pw", step=self.Step_pw)
        pw_old = self.mesh.Get_Field("pwOld")
        self.Update_Saturation(pw, pw_old)
        if step == self.Step_pw:
            self._weak_flux, self._weak_flux_tangent = self.PorousFlux(pw, pw_old, self.mesh.Get_Trial_Function("pw", self.Step_pw), self.mesh.Get_Field("poros"))
        return False
    
    def Add_KF(self, K, F, step: int) -> None:
        """
        Add algebraic terms beyond those defined in ``Get_Weak_Form``.

        ``K`` may be ``None`` during residual-only assembly.
        """
        if (step != self.Step_pw):
            return
        
        if self.USE_LUMPED == False:
            return  # Already included in Get_Weak_Form.
        self._require_sole_pressure_domain()

        dt = self.params.dt

        dofNames = ["pw", "poros"]
        OldNames = ["pwOld", "poros"]
        W, dofs, vals = self.mesh.Get_Weights_Dofs(dofNames, step=self.Step_pw)
        valsOld = self.mesh.Get_Old_For_Lumped(OldNames, step=self.Step_pw)

        n = len(W)
        k_indices = np.empty(n, dtype=np.int32)
        k_values  = np.empty(n)

        with F.localForm() as F_local:
            F_array = F_local.array
            
            for i in range(n):
                idx   = dofs["pw"][i]
                pw    = vals["pw"][i]
                pwOld = valsOld["pwOld"][i]
                poros = vals["poros"][i]
                
                Sw, dSw, ddSw = self.Get_Single_Sw(pw, poros)
                SwOld, _, _ = self.Get_Single_Sw(pwOld, poros)
                
                if self.Biot is None:
                    Biot = 1.0-(1.0-poros/self.poros_crit)**self.Biot_n
                else:
                    Biot = float(self.Biot)
                
                if self.USE_MASS_CAPACITY:
                    storage_residual = self._water_content(pw, Sw, poros, Biot) - self._water_content(pwOld, SwOld, poros, Biot)
                    storage_tangent = self._water_content_pressure_derivative(pw, Sw, dSw, poros, Biot)
                else:
                    if self.Saturated == True:
                        Minv = (Biot - poros) / self.bulk_s + poros / self.bulk_f
                        dMinv_dpw = 0.0
                    else:
                        Minv = (Sw * (Biot - poros) / self.bulk_s * (Sw + pw * dSw)
                                + poros * dSw
                                + (Sw * poros + 1e-6) / self.bulk_f)
                        dMinv_dpw = (
                            dSw * (Biot - poros) / self.bulk_s * (Sw + pw * dSw)
                            + Sw * (Biot - poros) / self.bulk_s * (2.0 * dSw + pw * ddSw)
                            + poros * ddSw
                            + dSw * poros / self.bulk_f
                        )
                    storage_residual = Minv * (pw - pwOld)
                    storage_tangent = Minv + dMinv_dpw * (pw - pwOld)

                F_array[idx] += W[i] * storage_residual / dt
                k_indices[i] = idx
                k_values[i] = W[i] * storage_tangent / dt

        # Apply K contributions OUTSIDE the localForm() context.
        # Mixing Vec.localForm() with MatSetValuesLocal in the same block
        # can trigger PETSc MPI state conflicts that silently corrupt the
        # matrix in parallel (works on 1 rank, diverges on N > 1).
        if K is not None:
            global_indices = self.mesh.get_step_global_indices(self.Step_pw, k_indices)
            for i in range(n):
                idx = global_indices[i]
                K.setValues([idx], [idx], [[k_values[i]]], addv=PETSc.InsertMode.ADD)
    
    def _require_sole_pressure_domain(self) -> None:
        """PRIVATE: Reject lumped storage when ``pw`` is shared with other domains.

        Lumped weights cover the whole support of ``pw``, so they are only
        those of this model when no other model registers ``pw`` elsewhere.
        """
        domains = self.mesh.Get_Field_Support("pw").domain_names
        if tuple(domains) != (self.domain,):
            raise NotImplementedError(
                f"Lumped pressure storage of '{self.name}' on '{self.domain}' requires 'pw' to live on that "
                f"domain only, but it is registered on {list(domains)}; set USE_LUMPED = False."
            )

    def Get_Single_Sw(self, pw, poros):
        if self.Saturated:
            return 1.0, 0.0, 0.0
        return single_saturation_from_pressure(pw, self.p_ref, self.m)

    def _water_content(self, pw, saturation, porosity, biot):
        """PRIVATE: Return reference-density-normalized water content at fixed strain."""
        return porosity * saturation * (1.0 + pw / self.bulk_f) + (biot - porosity) / self.bulk_s * pw * saturation**2

    def _water_content_pressure_derivative(self, pw, saturation, d_saturation, porosity, biot):
        """PRIVATE: Return the pressure derivative of :meth:`_water_content`."""
        return (
            porosity * (d_saturation * (1.0 + pw / self.bulk_f) + saturation / self.bulk_f)
            + (biot - porosity) / self.bulk_s * (saturation**2 + 2.0 * pw * saturation * d_saturation)
        )

        
        
    def SatRelations(self, pc):
        return saturation_relations(pc, self.p_ref, self.m)
    
    def Update_Saturation(self, pw, pwOld):
        if self.Saturated:
            return # Saturation always at 1.0
        
        self.Sw, self.dSw, self.ddSw = saturation_from_pressure( pw, self.p_ref, self.m)
        self.SwOld, _, _ = saturation_from_pressure(pwOld, self.p_ref, self.m)
        self.mesh.Set_From_Expression("Sw", self.Sw, domain=self.domain)
    
    def PorousFlux(self, pw_field, pwOld_field, pw_tr_field, porosity_field):
        """
        Compute the Darcy flux vector given a pore pressure field and its trial function.

        Note: The effective permeability is lagged one time increment behind (using SwOld) 
        to ensure a stable convergence, even when the kr(Sw) relationship is highly nonlinear.
        """
        k = self.permeability0 * (porosity_field / self.poros_crit)**3/(1.0-porosity_field/self.poros_crit)**2

        if not self.Saturated:
            
            # Lagged relative permeability: both the residual and the Jacobian
            # use kr(SwOld) from the previous time step, giving a positive-
            # definite tangent that is robust for sharp fronts with strongly
            # nonlinear kr(Sw).
            if True: # Mualem - VG
                eps_Se = 1e-10
                Se_sat = 0.99

                # --- Residual: kr from SwOld ---
                Sw = self.SwOld
                Se_lo_r      = ufl.conditional(lt(Sw, eps_Se), eps_Se, Sw)
                Se_r         = ufl.conditional(lt(Se_lo_r, Se_sat), Se_lo_r, Se_sat)
                g_r          = 1.0 - Se_r ** (1.0 / self.m)
                f_r          = 1.0 - g_r ** self.m
                kr_r         = ufl.conditional(lt(Sw, Se_sat), Se_r ** 0.5 * f_r ** 2 + 1e-6, 1.0)

                # --- Jacobian: kr  ---
                Sw2 = self.SwOld
                Se_lo_k      = ufl.conditional(lt(Sw2, eps_Se), eps_Se, Sw2)
                Se_k         = ufl.conditional(lt(Se_lo_k, Se_sat), Se_lo_k, Se_sat)
                g_k          = 1.0 - Se_k ** (1.0 / self.m)
                f_k          = 1.0 - g_k ** self.m
                kr_k         = ufl.conditional(lt(Sw2, Se_sat), Se_k ** 0.5 * f_k ** 2 + 1e-6, 1.0)

                kr      = kr_r    # used in flux  (F)
                kr_jac  = kr_k    # used in dflux (K)
            else:
                # Simple exponential
                exp = 0.5
                kr      = (self.SwOld)**exp + 1e-8   # residual: current Sw
                kr_jac  = (self.SwOld)**exp + 1e-8   # Jacobian: lagged SwOld
        else:
            kr      = 1.0
            kr_jac  = 1.0

        driving = grad(pw_field) - self.rho_f * self.gVec
        flux = -k * kr / self.visc * driving

        # Jacobian uses kr_jac (lagged) instead of kr (current) so the
        # linearisation is stable even when kr is strongly nonlinear.
        dflux_dpw =  -k * kr_jac / self.visc * grad(pw_tr_field)

        # save to field for post-processing
        self.mesh.Set_From_Expression("qw", flux, domain=self.domain)
        return flux, dflux_dpw
    
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        if self.initial_poros is not None:
            self.mesh.Constant_Field("poros", self.initial_poros, domain=self.domain)
        for field_name in ("pw", "pwOld"):
            self.mesh.Zero_Field(field_name, domain=self.domain)

    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """
        
        # Copy current fields to old fields using mesh utility
        self.mesh.Copy_Field("pw", "pwOld", domain=self.domain)

        
    def reset_step(self):
        self.mesh.Copy_Field("pwOld", "pw", domain=self.domain)
        
