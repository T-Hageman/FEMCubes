"""Consistently split viscoelastic ice with local material properties."""

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.glaciology._ice_state import add_ice_state_fields, check_ice_state, warn_temperature_not_set
from femcubes.glaciology.ice_material import IceMaterial
from femcubes.solids._split_elastic_common import _fracture_driver
from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import SplitViscoElasticKernelsVariableProps
from femcubes.solids.split_viscoelastic import SplitViscoElastic


class IceViscoElastic(SplitViscoElastic):
    r"""Viscoelastic ice with selective phase-field degradation and local material properties.

    The model applies the formulation of :doc:`../solids/split_viscoelastic`
    (momentum balance, Newmark integration, Glen creep by return mapping,
    stress :math:`g(\phi)\boldsymbol{\sigma}^+ + \boldsymbol{\sigma}^-`) to
    glacier ice whose properties vary in space. Ice deforms elastically under
    rapid loading and creeps under sustained stress following Glen's flow law
    :cite:p:`Glen1955`, with a rate factor that depends strongly on
    temperature :cite:p:`Paterson1982`, while porous firn is softer and
    creeps faster than solid ice. The density, elastic moduli, creep stress
    scale, Glen exponent, tensile strength and fracture toughness are
    therefore evaluated pointwise from the porosity and temperature fields
    with :class:`~femcubes.glaciology.ice_material.IceMaterial`, and scalar
    material values in ``Materials`` (``Lame``, ``Shear``, ``Density``,
    ``Glen_A``, ``Gc``, ``f_t``, ...) are not read. As the split is
    consistent, damaged ice keeps its compressive stresses and its density,
    and continues to carry its own weight (see
    :ref:`solids-split-consistency`). Supports 2-D (plane strain, or
    intrinsic 2-D with ``true_2d``) and 3-D. Couples to
    :doc:`ice_phasefield` (or :doc:`../solids/phase_field`) through
    ``ElasticEnergy``.

    .. rubric:: Fields

    * ``u``, ``v``, ``a``, ``uOld``, ``vOld``, ``aOld``, ``sigma``,
      ``dsigma_dStrainMandel``, ``Plastic_Strain``, ``Plastic_Strain_Old``,
      ``LambdaField``, ``LambdaFieldOld``, ``Total_Strain``,
      ``dam_elastic``: as in :doc:`../solids/viscoelastic` (``u`` solved in
      step ``Solution_Steps["u"]``).
    * ``ElasticEnergy``, ``sigma_plus``, ``sigma_minus``: as in
      :doc:`../solids/split_viscoelastic`.
    * ``poros``: porosity :math:`p`, DG of order ``Field_Orders["Porosity"]``
      (else ``Field_Orders["pw"]``, else 1). Set by the user; not changed by
      the model.
    * ``T``: temperature in degrees Celsius, CG of order
      ``Field_Orders["Temperature"]`` (else ``Field_Orders["T"]``, else 1).
      Set by the user; not changed by the model.
    * ``Lame``, ``Shear``, ``Creep_Stress_Scale``: scalar quadrature fields
      holding :math:`\lambda(p)`, :math:`\mu(p)` and the creep stress scale
      :math:`\eta_\mathrm{vp}(p,T)` of
      :meth:`~femcubes.glaciology.ice_material.IceMaterial.Get_ViscoPlastic_viscosity`
      (with :math:`d = 2` for ``true_2d``, 3 otherwise) passed to the
      constitutive kernel.
    * ``TensileStrength``, ``FractureToughness``: scalar quadrature fields
      with :math:`f_t` and :math:`G_c(p,T)` of the ``Stress-Based`` driver;
      only with ``DamageModel="PhaseField"`` and
      ``CrackDrivingForce="Stress-Based"``.
    * ``phasefield``: read only, provided by :doc:`ice_phasefield`; only with
      ``DamageModel="PhaseField"``.

    .. rubric:: Weak form

    The residual of ``u`` is the one of :doc:`../solids/split_viscoelastic`,
    with the ice density :math:`\rho(p) = 917(1-p)`:

    .. math::

        R_{\mathbf{u}} =
        \int_\Omega \boldsymbol{\sigma}:\boldsymbol{\varepsilon}(\delta\mathbf{u})\,\mathrm{d}x
        + \int_\Omega \rho(p)\,\delta\mathbf{u}\cdot\ddot{\mathbf{u}}\,\mathrm{d}x
        + \int_\Omega c\,\delta\mathbf{u}\cdot\dot{\mathbf{u}}\,\mathrm{d}x
        - \int_\Omega \rho(p)\,g\,\delta u_{d}\,\mathrm{d}x,
        \qquad
        \boldsymbol{\sigma} = g(\phi)\,\boldsymbol{\sigma}^+ + \boldsymbol{\sigma}^-,

    with :math:`\ddot{\mathbf{u}}`, :math:`\dot{\mathbf{u}}` from the Newmark
    relations, the inertia term only with ``Inertia=True`` (default
    ``False``), the damping term only for ``LE_Damping`` :math:`c>0` and the
    gravity term only for :math:`g\neq0`. The Jacobian uses the algorithmic
    tangent ``dsigma_dStrainMandel`` of the local update. Density is never
    degraded. :math:`g(\phi) = \kappa + (1-\kappa)(1-\hat\phi)^2` with
    :math:`\hat\phi` the phase field clipped to :math:`[0,1]`, and
    :math:`g = 1` without ``DamageModel``.

    .. rubric:: Outside the weak form

    * Before every assembly: Newmark update of ``v`` and ``a``; ``Lame``,
      ``Shear``, ``Creep_Stress_Scale`` (and ``TensileStrength``,
      ``FractureToughness``) are interpolated from the current ``poros`` and
      ``T``; the variable-property split kernel then returns ``sigma``,
      ``sigma_plus``, ``sigma_minus``, ``Plastic_Strain``, ``LambdaField``,
      the tangent and ``ElasticEnergy`` (``CrackDrivingForce``, with the
      local :math:`G_c(p,T)` and :math:`f_t` for ``Stress-Based``; without
      ``DamageModel``, ``Stress-Based`` stores the elastic energy
      :math:`\tfrac12\boldsymbol{\varepsilon}^e:\boldsymbol{\sigma}`).
    * **Checks**, on all ranks together before every assembly, raising the
      same ``ValueError`` on every rank: ``poros`` in :math:`[0, 1)`, ``T``
      at most 0 °C and, for ``Stress-Based`` with a phase field, positive
      :math:`f_t` and :math:`G_c`. At the first assembly a warning is given
      if ``T`` is 0 °C everywhere (probably never set).
    * The Glen exponent is :math:`n = 3`; ``CreepCoupling`` selects the stress
      in the flow rule as in :doc:`../solids/split_viscoelastic`.
    * ``commit`` copies ``u``, ``v``, ``a``, ``Plastic_Strain``,
      ``LambdaField`` to their ``Old`` fields; ``reset_step`` restores them.
    * ``Post_Remesh`` resets the quadrature data of the base model and
      rebuilds the material expressions on the new mesh.
    * No Dirichlet conditions. ``poros`` and ``T`` must be set after
      ``Physics.Initialize_Fields`` and before the first solve.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            Field_Orders = {
                "u": 2,                      # REQUIRED: order of u
                "Porosity": 0,               # OPTIONAL (default Field_Orders["pw"], else 1): DG order of poros
                "Temperature": 1,            # OPTIONAL (default Field_Orders["T"], else 1): CG order of T
            }
            Solution_Steps = {"u": 0}        # REQUIRED: staggered step solving u
            gravity = -9.81                  # REQUIRED: acceleration along the last axis (m/s^2)
            Newmark_Beta = 0.25              # REQUIRED: Newmark beta (-)
            Newmark_Gamma = 0.5              # REQUIRED: Newmark gamma (-)
            Materials = {
                "Ice": {                     # REQUIRED entry (may be empty)
                    "LE_Damping": 0.0,       # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                    "Inertia": False,        # OPTIONAL (default False): include the inertia term
                    "CrackDrivingForce": "Stress-Based",  # OPTIONAL (default "Stress-Based"): split and driver
                    # With DamageModel="PhaseField" only:
                    "ell": 10.0,             # REQUIRED: phase-field length scale (m)
                    "kappa": 1.0e-3,         # OPTIONAL (default 1e-3): residual stiffness (-)
                    "PF_formulation": "AT2", # OPTIONAL (default "AT2"): selects Gc(p, T) and the driver scaling
                },
            }
            Models = {
                "Solid": {
                    "Type": "Glaciology:ViscoElastic",   # REQUIRED
                    "Material": "Ice",                   # REQUIRED: key in Materials
                    "DamageModel": "PhaseField",         # OPTIONAL (default None): None or "PhaseField"
                    "CreepCoupling": "EffectiveStress",  # OPTIONAL (default "EffectiveStress"): or "DegradedStress",
                                                         #   "FullDamagedStress", "PartialExplicit"
                    "true_2d": False,                    # OPTIONAL (default False): intrinsic 2-D law
                    "domain": "domain",                  # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "Glaciology:ViscoElastic"
    # Per-point properties of the split kernel
    _PROPERTY_FIELDS = ("Lame", "Shear", "Creep_Stress_Scale")
    # Per-point properties of the Stress-Based driver (DamageModel="PhaseField" only)
    _STRENGTH_FIELDS = ("TensileStrength", "FractureToughness")
    _FIELD_LABELS = {"Lame": "Lame Constant [Pa]", "Shear": "Shear Modulus [Pa]",
                     "Creep_Stress_Scale": "Creep Stress Scale [Pa s^(1/n)]", "TensileStrength": "Tensile Strength [Pa]",
                     "FractureToughness": "Fracture Toughness [J/m^2]"}

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        settings = params.Models[name]
        domain = settings.get("domain", "domain")
        material = params.Materials[settings["Material"]]
        self._stress_based_damage = (settings.get("DamageModel") == "PhaseField"
                                     and material.get("CrackDrivingForce", "Stress-Based") == "Stress-Based")
        add_ice_state_fields(params, mesh, domain)
        self._sampled_fields = self._PROPERTY_FIELDS + (self._STRENGTH_FIELDS if self._stress_based_damage else ())
        for field_name in self._sampled_fields:
            mesh.Add_Quadrature_Field(field_name, self._FIELD_LABELS[field_name], (), domain=domain)
        self._POINT_FIELDS = SplitViscoElastic._POINT_FIELDS + self._sampled_fields
        self.iceMat = IceMaterial()
        self._property_expressions = {}
        self._temperature_checked = False
        super().__init__(name, params, mesh)

    def Post_Remesh(self) -> None:
        """Reset the quadrature data and rebuild the material expressions on the new mesh."""
        super().Post_Remesh()
        self._bind_material()
        self._property_expressions.clear()

    # Internal helpers

    def _read_material(self, material: dict) -> None:
        """PRIVATE: ice properties from ``poros`` and ``T`` (``ell`` and ``PF_formulation`` from
        ``material`` for the toughness)."""
        self.viscosity_type = "Glen"
        if self.damage_model_name == "PhaseField":
            self.ell = material["ell"]
            self.pf_formulation = material.get("PF_formulation", "AT2")
        self._bind_material()

    def _bind_material(self) -> None:
        """PRIVATE: material expressions of the live ``poros`` and ``T`` fields."""
        porosity, temperature = self.mesh.Get_Field("poros"), self.mesh.Get_Field("T")
        ice = self.iceMat
        self.rho = ice.Get_Density(porosity, temperature)
        self.lame = ice.Get_Lame_Parameter(porosity, temperature)
        self.shear = ice.Get_Shear_Modulus(porosity, temperature)
        self.bulk = self.lame + self.shear if self.true_2d else ice.Get_Bulk_Modulus(porosity, temperature)
        self.plastic_n = ice.Get_Creep_n(porosity, temperature)
        self.viscosity = ice.Get_ViscoPlastic_viscosity(porosity, temperature, dim=2 if self.true_2d else 3)
        if self.damage_model_name == "PhaseField":
            self.f_t = ice.Get_Tensile_Strength(porosity, temperature)
            self.Gc = ice.Get_Gc(porosity, temperature, self.ell, PF_formulation=self.pf_formulation)

    def _build_creep_kernel(self):
        """PRIVATE: none; the variable-property split kernel does the return map."""
        return None

    def _build_split_kernel(self):
        """PRIVATE: the split return map with per-point properties."""
        return SplitViscoElasticKernelsVariableProps(self.plastic_n, self.DrivingForce_Type, self.creep_coupling,
                                                     self.true_2d)

    def _update_plastic_fields(self, strain_expr) -> None:
        """PRIVATE: Update split stresses, tangent and driver with local ice properties."""
        self._total_strain_expr = self._points.interpolate("Total_Strain", strain_expr, self._total_strain_expr)
        self._damage_expr = self._points.interpolate("dam_elastic", self.d_e, self._damage_expr)
        values = {"Lame": self.lame, "Shear": self.shear, "Creep_Stress_Scale": self.viscosity}
        if self._stress_based_damage:
            values.update(TensileStrength=self.f_t, FractureToughness=self.Gc)
        for name, value in values.items():
            self._property_expressions[name] = self._points.interpolate(name, value, self._property_expressions.get(name))
        for name in ("Total_Strain", "dam_elastic") + self._sampled_fields:
            self.mesh.Get_Field(name).x.scatter_forward()

        strength = {}
        if self._stress_based_damage:
            strength = dict(zip(("f_t", "Gc"), self._points.gather(*self._STRENGTH_FIELDS)))
        label = f"{self.MODEL_TYPE} '{self.name}'"
        temperature_unset = check_ice_state(self.mesh, self.domain, label, positive=strength)
        if temperature_unset and not self._temperature_checked:
            warn_temperature_not_set(self.mesh, label)
        self._temperature_checked = True

        strain, old_plastic, damage, lame, shear, viscosity = self._points.gather(
            "Total_Strain", "Plastic_Strain_Old", "dam_elastic", *self._PROPERTY_FIELDS)
        result = self._split_kernel(
            strain.reshape(-1, 6),
            old_plastic.reshape(-1, 6),
            damage, self._constitutive_dt,
            lame, shear, viscosity,
            old_total_strains=self._previous_total_strains(),
        )
        material = None
        if self.damage_model_name == "PhaseField":
            material = {"Gc": self.Gc, "ell": self.ell, "f_t": self.f_t, "PF_formulation": self.pf_formulation,
                        **strength}
        driver = _fracture_driver(
            result["driver"], self.DrivingForce_Type, material,
            elastic_strain=strain.reshape(-1, 6) - result["strain_p"],
            stress=result["stress"],
        )
        values = {
            "sigma": result["stress"],
            "sigma_plus": result["stress_plus"],
            "sigma_minus": result["stress_minus"],
            "Plastic_Strain": result["strain_p"],
            "dsigma_dStrainMandel": result["tangent"],
            "LambdaField": result["lambda"],
            "ElasticEnergy": driver,
        }
        self._points.scatter(values)
        for name in values:
            self.mesh.Get_Field(name).x.scatter_forward()
