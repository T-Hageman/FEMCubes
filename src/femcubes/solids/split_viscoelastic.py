"""Viscoelasticity with selective phase-field degradation."""

from dolfinx import fem
import ufl
from ufl import conditional, lt

from femcubes.core.utils.maths_utils import Tensor_To_Mandel_UFL, nabla_s
from femcubes.solids.viscoelastic import ViscoElastic
from femcubes.solids._split_elastic_common import _fracture_driver, _validate_split_damage_material, _validate_split_settings
from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import SplitViscoElasticKernels


class SplitViscoElastic(ViscoElastic):
    r"""Viscoelastic solid with selective phase-field degradation of a split stress.

    The model combines the creep of :doc:`viscoelastic` with the selective
    degradation of :doc:`split_linear_elastic`. It has the same displacement
    fields, Newmark integration, flow law and history as :doc:`viscoelastic`,
    but only the damageable part :math:`\boldsymbol\sigma^+` of the stress is
    degraded, with the split chosen by ``CrackDrivingForce`` (see
    :ref:`solids-crack-driving-forces`). The split acts on the elastic strain,
    which depends on the viscous strain of the same step, so the stress, the
    tangent, the viscous strain and the fracture driver ``ElasticEnergy`` for
    :doc:`phase_field` are obtained from one local solve at each quadrature
    point. Which stress drives creep in damaged material is a modelling
    choice, selected by ``CreepCoupling`` (see below). Supports 3-D, 2-D plane
    strain and intrinsic 2-D (``true_2d``). Density is never degraded:
    damaged material keeps the compressive part of the stress and carries its
    weight (a consistent split, see :ref:`solids-split-consistency`).

    .. rubric:: Fields

    * ``u``, ``v``, ``a``, ``uOld``, ``vOld``, ``aOld``, ``Plastic_Strain``,
      ``Plastic_Strain_Old``, ``LambdaField``, ``LambdaFieldOld``,
      ``Total_Strain``, ``dam_elastic``, ``dsigma_dStrainMandel``: as in
      :doc:`viscoelastic`.
    * ``sigma``: mechanical stress :math:`g(\phi)\boldsymbol\sigma^+ + \boldsymbol\sigma^-`
      (quadrature, six Mandel components).
    * ``sigma_plus``, ``sigma_minus``: undegraded :math:`\boldsymbol\sigma^+`
      and :math:`\boldsymbol\sigma^-` (quadrature, six components).
    * ``ElasticEnergy``: crack driving force, scalar quadrature field (always
      defined).
    * ``phasefield``: read only, provided by :doc:`phase_field`; only with
      ``DamageModel="PhaseField"``.

    .. rubric:: Weak form

    With test function :math:`\delta\mathbf{u}` and Mandel strain vectors
    (zero out-of-plane components on 2-D meshes), the contribution to the
    residual of ``u`` is

    .. math::

        R_{\mathbf{u}} =
        \int_\Omega \boldsymbol{\varepsilon}(\delta\mathbf{u}):\boldsymbol{\sigma}\,\mathrm{d}x
        + \int_\Omega \rho\,\delta\mathbf{u}\cdot\ddot{\mathbf{u}}\,\mathrm{d}x
        + \int_\Omega c\,\delta\mathbf{u}\cdot\dot{\mathbf{u}}\,\mathrm{d}x
        - \int_\Omega \rho\,g\,\delta u_{d}\,\mathrm{d}x,
        \qquad
        \boldsymbol\sigma = g(\phi)\,\boldsymbol\sigma^+(\mathbf e) + \boldsymbol\sigma^-(\mathbf e),

    with the elastic strain :math:`\mathbf e = \boldsymbol\varepsilon(\mathbf u) - \boldsymbol\varepsilon^v`,
    :math:`g(\phi) = \kappa + (1-\kappa)(1-\hat\phi)^2`,
    :math:`\hat\phi = \min(\max(\phi,0),1)`, and :math:`g = 1` without
    ``DamageModel``. The inertia (``Inertia``, default ``False``), damping and
    gravity terms and the Newmark relations are those of :doc:`viscoelastic`,
    with the full density :math:`\rho`. The Jacobian uses the algorithmic
    tangent ``dsigma_dStrainMandel``
    :math:`= \mathrm d\boldsymbol\sigma/\mathrm d\boldsymbol\varepsilon` of the
    local solve at fixed :math:`\phi` and committed history (consistent; it
    can be nonsymmetric).

    The local solve is backward Euler in the elastic strain,

    .. math::

        \mathbf e + \Delta t\,\dot{\boldsymbol\varepsilon}^v(\boldsymbol\sigma_c(\mathbf e))
        = \boldsymbol\varepsilon(\mathbf u) - \boldsymbol\varepsilon^v_n,
        \qquad
        \dot{\boldsymbol\varepsilon}^v(\boldsymbol\sigma_c) = \frac{d}{2q}\Bigl(\frac{q}{\eta_\mathrm{vp}}\Bigr)^{n}\mathbf s_c,
        \qquad q = \sqrt{\tfrac d2\,\mathbf s_c:\mathbf s_c},

    with :math:`\mathbf s_c` the deviator of the creep stress
    :math:`\boldsymbol\sigma_c`, and :math:`n`, :math:`\eta_\mathrm{vp}`,
    :math:`d` as in :doc:`viscoelastic`; :math:`\Delta t` is the current time
    increment. There is no damage cut-off for creep. ``CreepCoupling``
    selects :math:`\boldsymbol\sigma_c`:

    * ``"EffectiveStress"`` (default): :math:`\boldsymbol\sigma_c = \mathbb C:\mathbf e`.
    * ``"DegradedStress"``: :math:`\boldsymbol\sigma_c = g\boldsymbol\sigma^+ + \boldsymbol\sigma^-`.
    * ``"FullDamagedStress"``: :math:`\boldsymbol\sigma_c = g\,\mathbb C:\mathbf e`.
    * ``"PartialExplicit"``: the ``EffectiveStress`` solve is done at the
      committed total strain :math:`\boldsymbol\varepsilon(\mathbf u_n)`,
      giving :math:`\boldsymbol\varepsilon^v` independent of the current
      iterate; the stress is then
      :math:`g\boldsymbol\sigma^+(\mathbf e) + \boldsymbol\sigma^-(\mathbf e)`
      with that :math:`\boldsymbol\varepsilon^v` held fixed, and the tangent is
      the split elastic tangent. This is a first-order splitting: creep lags
      the elastic loading by one step.

    .. rubric:: Outside the weak form

    * Before every assembly: Newmark update of :math:`\dot{\mathbf u}`,
      :math:`\ddot{\mathbf u}`, then the local solve at the current
      :math:`\boldsymbol\varepsilon(\mathbf u)` from
      :math:`\boldsymbol\varepsilon^v_n`, which sets ``sigma``,
      ``sigma_plus``, ``sigma_minus``, ``dsigma_dStrainMandel``,
      ``Plastic_Strain``, ``LambdaField``
      (:math:`\Delta t\sqrt{2/d}\,|\dot{\boldsymbol\varepsilon}^v|`) and
      ``ElasticEnergy``, all at the updated elastic strain (``Stress-Based``
      not clipped; total elastic energy without ``DamageModel``).
    * ``commit`` and ``reset_step`` as in :doc:`viscoelastic`.
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
                    "Lame": 6.5e9,           # REQUIRED: first Lame parameter lambda (Pa), >= 0
                    "Shear": 3.5e9,          # REQUIRED: shear modulus mu (Pa), > 0
                    "Bulk": 8.8e9,           # REQUIRED: bulk modulus (Pa); read but not used
                    "Density": 917.0,        # REQUIRED: density rho (kg/m^3)
                    "Visc_Type": "Glen",     # OPTIONAL (default "Newtonian"): "Newtonian" or "Glen"
                    "Glen_A": 1.0e-24,       # REQUIRED for Glen: rate factor A (Pa^-n s^-1)
                    "Glen_n": 3.0,           # REQUIRED for Glen: exponent n >= 1 (-)
                    # "Viscosity": 1.0e13,   # REQUIRED for Newtonian: dynamic viscosity eta > 0 (Pa s)
                    "LE_Damping": 0.0,       # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                    "Inertia": False,        # OPTIONAL (default False): include the inertia term
                    "CrackDrivingForce": "NoTension",  # OPTIONAL (default "Stress-Based"): stress and driver split
                    # With DamageModel="PhaseField" only:
                    "Gc": 1.0e3,             # REQUIRED: fracture toughness (J/m^2)
                    "ell": 10.0,             # REQUIRED: phase-field length scale (m)
                    "f_t": 0.4e6,            # REQUIRED: tensile strength (Pa)
                    "kappa": 1.0e-6,         # OPTIONAL (default 1e-3): residual stiffness, 0 < kappa <= 1
                    "PF_formulation": "AT2", # OPTIONAL (default "AT2"): AT1, AT2, HO-AT1 or HO-AT2 (Stress-Based scaling)
                },
            }
            Models = {
                "Solid": {
                    "Type": "Solid:SplitViscoElastic",   # REQUIRED
                    "Material": "Ice",                   # REQUIRED: key in Materials
                    "DamageModel": "PhaseField",         # OPTIONAL (default None): None or "PhaseField"
                    "CreepCoupling": "EffectiveStress",  # OPTIONAL (default "EffectiveStress"): see above
                    "true_2d": False,                    # OPTIONAL (default False): intrinsic 2-D law, 2-D meshes only
                    "domain": "domain",                  # OPTIONAL (default "domain", the full mesh): domain group
                },
            }

    Notes
    -----
    The exact splits have tangent jumps where a principal direction changes
    between tension and compression. With a small :math:`\kappa`, a Newton
    correction computed on a degraded branch can overshoot into a much
    stiffer branch, so the residual line search may need a lower limit in
    ``line_search_lims`` below 0.1. The local Newton solve raises an error
    when it does not converge.
    """

    MODEL_TYPE = "Solid:SplitViscoElastic"
    _POINT_FIELDS = ViscoElastic._POINT_FIELDS + ("sigma_plus", "sigma_minus", "ElasticEnergy")

    def __init__(self, name, params, mesh):
        settings = params.Models[name]
        split = _validate_split_settings(settings, params.Materials[settings["Material"]], type(self).__name__)
        self.creep_coupling = settings.get("CreepCoupling", "EffectiveStress")
        if self.creep_coupling not in ("EffectiveStress", "DegradedStress", "FullDamagedStress", "PartialExplicit"):
            raise ValueError("CreepCoupling must be 'EffectiveStress', 'DegradedStress', 'FullDamagedStress' or 'PartialExplicit'.")
        super().__init__(name, params, mesh)
        self.DrivingForce_Type = split
        if self.damage_model_name == "PhaseField":
            self.pf_formulation = _validate_split_damage_material(params.Materials[self.material], split,
                                                                  Gc=self.Gc, f_t=self.f_t)
        else:
            mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", (), domain=self.domain)
        mesh.Add_Quadrature_Field("sigma_plus", "Undegraded damageable stress [Pa]", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("sigma_minus", "Retained stress [Pa]", (6, 1), domain=self.domain)
        self._split_kernel = self._build_split_kernel()
        self._constitutive_dt = params.dt
        self._old_total_strain_expr = None
        self._old_total_strain_field = None

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Assemble split mechanical stress and tangent with full-density forces.

        The constitutive update stores sigma = g*sigma_plus + sigma_minus and
        its algorithmic tangent in quadrature fields. Use those fields directly;
        inertia and gravity retain the full density at every damage level.
        """
        if step != self.My_Step:
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)

        v = self.mesh.Get_Field("v")
        a = self.mesh.Get_Field("a")
        u_t = self.mesh.Get_TestFunction("u", self.My_Step)
        u_tr = self.mesh.Get_Trial_Function("u", self.My_Step)
        dv_du = self.NM_Gamma / (self.NM_Beta * dt_constant)
        da_du = 1.0 / (self.NM_Beta * dt_constant**2)

        # The stored stress and tangent already contain the consistent split.
        strain_test = Tensor_To_Mandel_UFL(nabla_s(u_t), self.dim)
        strain_trial = Tensor_To_Mandel_UFL(nabla_s(u_tr), self.dim)
        stress = self.mesh.Get_Field("sigma")
        tangent = self.mesh.Get_Field("dsigma_dStrainMandel")
        stress_vector = ufl.as_vector([stress[i, 0] for i in range(6)])
        F = ufl.inner(strain_test, stress_vector) * dx
        K = ufl.dot(strain_test, ufl.dot(tangent, strain_trial)) * dx

        if self.Inertia:
            F += self.rho * ufl.inner(u_t, a) * dx
            K += self.rho * ufl.inner(u_t, da_du * u_tr) * dx

        if self.damping > 0.0:
            F += self.damping * ufl.inner(u_t, v) * dx
            K += self.damping * ufl.inner(u_t, dv_du * u_tr) * dx

        if self.gravity != 0.0:
            F -= self.rho * u_t[self.dim-1] * self.gravity * dx

        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step):
        """Update mechanics and fracture driving force from one local state."""
        self._constitutive_dt = self._weak_constant_value(dt_constant)
        return super().Update_Weak_Constants(time_constant, dt_constant, step)

    def Update_PhaseField_Fields(self, strain):
        """Bind stiffness degradation while retaining the full material density."""
        self.d_rho = 1.0
        if self.damage_model_name == "PhaseField":
            phasefield = self.mesh.Get_Field("phasefield")
            phi = conditional(lt(phasefield, 1.0), phasefield, 1.0)
            phi = conditional(lt(phi, 0.0), 0.0, phi)
            self.d_e = self.kappa + (1.0-self.kappa)*(1.0-phi)**2

    def Post_Remesh(self) -> None:
        """Rebuild current and committed-strain sampling on the refined mesh."""
        super().Post_Remesh()
        self._old_total_strain_expr = None
        self._old_total_strain_field = None

    # Internal helpers

    def _build_split_kernel(self):
        """PRIVATE: the C++ split return map of the model's constants."""
        return SplitViscoElasticKernels(
            self.viscosity, self.plastic_n, self.lame, self.shear,
            self.DrivingForce_Type, self.creep_coupling, self.true_2d,
        )

    def _previous_total_strains(self):
        """PRIVATE: Sample committed displacement strain for PartialExplicit."""
        if self.creep_coupling != "PartialExplicit":
            return None
        old_strain = Tensor_To_Mandel_UFL(nabla_s(self.mesh.Get_Field("uOld")), self.dim)
        if not self._points.can_compile("Total_Strain", old_strain):
            return self._points.evaluate("Total_Strain", old_strain).reshape(-1, 6)
        if self._old_total_strain_expr is None:
            space = self.mesh.Get_Field("Total_Strain").function_space
            self._old_total_strain_field = fem.Function(space)
            self._old_total_strain_expr = fem.Expression(old_strain, space.element.interpolation_points)
        self._old_total_strain_field.interpolate(self._old_total_strain_expr)
        self._old_total_strain_field.x.scatter_forward()
        return self._old_total_strain_field.x.array.reshape(-1, 6)

    def _update_plastic_fields(self, strain_expr):
        """PRIVATE: Evaluate both stress and driver at the converged elastic strain."""
        self._total_strain_expr = self._points.interpolate("Total_Strain", strain_expr, self._total_strain_expr)
        self.mesh.Get_Field("Total_Strain").x.scatter_forward()
        self._damage_expr = self._points.interpolate("dam_elastic", self.d_e, self._damage_expr)
        self.mesh.Get_Field("dam_elastic").x.scatter_forward()
        strain, old_plastic, damage = self._points.gather("Total_Strain", "Plastic_Strain_Old", "dam_elastic")
        result = self._split_kernel(
            strain.reshape(-1, 6),
            old_plastic.reshape(-1, 6),
            damage, self._constitutive_dt,
            old_total_strains=self._previous_total_strains(),
        )
        driver = _fracture_driver(
            result["driver"], self.DrivingForce_Type,
            {"Gc": self.Gc, "ell": self.ell, "f_t": self.f_t, "PF_formulation": self.pf_formulation}
            if self.damage_model_name == "PhaseField" else None,
            clip_at2=False,
            elastic_strain=strain.reshape(-1, 6) - result["strain_p"], stress=result["stress"],
        )
        self._points.scatter({
            "sigma": result["stress"],
            "sigma_plus": result["stress_plus"],
            "sigma_minus": result["stress_minus"],
            "Plastic_Strain": result["strain_p"],
            "dsigma_dStrainMandel": result["tangent"],
            "LambdaField": result["lambda"],
            "ElasticEnergy": driver,
        })
        for name in ("sigma", "sigma_plus", "sigma_minus", "Plastic_Strain", "dsigma_dStrainMandel", "LambdaField", "ElasticEnergy"):
            self.mesh.Get_Field(name).x.scatter_forward()
