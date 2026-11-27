"""Ice-specific cohesive visco-elastic solid model."""

import numpy as np

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.glaciology._ice_state import (
    add_ice_state_fields,
    admissible_tensile_strength,
    check_ice_state,
    warn_temperature_not_set,
)
from femcubes.glaciology.ice_material import IceMaterial
from femcubes.solids._cohesive_viscoelastic_base import elastic_stiffness
from femcubes.solids._principal_cve import PrincipalCVEUpdate, opening_strength_scales
from femcubes.solids._reduced_cve import ReducedCVEMaterial, ReducedCVEUpdate
from femcubes.solids.cohesive_viscoelastic import Cohesive_ViscoElastic
from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    CohesiveViscoElasticKernelsVariableProps as _CVEVariablePropsKernels,
)


class IceCohesiveViscoElastic(Cohesive_ViscoElastic):
    r"""Cohesive viscoelastic ice with local porosity and temperature.

    The model applies the strength-based cohesive phase-field model with
    creep of :cite:t:`Hageman2026CMAME` to glacier ice whose properties vary
    in space, as :doc:`ice_viscoelastic` does for the standard phase-field
    model. It is :doc:`../solids/cohesive_viscoelastic` (momentum balance,
    Glen creep and the cohesive fracture strain solved together in one local
    problem per quadrature point, phase-field degradation of the strength,
    the same failure types, fields, history and global measures), with the
    density, elastic moduli, creep stress scale and tensile strength
    evaluated pointwise from the porosity and temperature fields with
    :class:`~femcubes.glaciology.ice_material.IceMaterial`. ``Lame``,
    ``Shear``, ``Bulk``, ``Density``, ``Visc_Type``, ``Glen_A``, ``Glen_n``,
    ``Viscosity`` and ``f_t`` in ``Materials`` are not read; the cohesive
    inputs ``f_s``, ``FailureType``, ``DP_phi``, ``DP_q`` and ``kappa`` are
    material constants as in :doc:`../solids/cohesive_viscoelastic`. Supports
    2-D (plane strain, ``plane_stress`` or ``true_2d``) and 3-D, with every
    failure type of :doc:`../solids/cohesive_viscoelastic`, including the
    opening-only criterion ``r2_2L_PSD`` (not with ``true_2d``). Couples to
    :doc:`ice_phasefield` (or :doc:`../solids/phase_field`) through
    ``PsiField``; a model providing ``phasefield`` is required.

    .. rubric:: Fields

    * All fields of :doc:`../solids/cohesive_viscoelastic` (``u`` solved in
      step ``Solution_Steps["u"]``, the Newmark fields, ``sigma``,
      ``dsigma_dStrainMandel``, the viscous and cohesive history fields,
      ``eta``, ``etaNorm``, ``ElasticEnergy``, ``PsiField``, and
      ``StrengthRatio`` with ``Strength_Ratio_Field=True``).
    * ``poros``: porosity :math:`p`, DG of order ``Field_Orders["Porosity"]``
      (else ``Field_Orders["pw"]``, else 1); set by the user.
    * ``T``: temperature in degrees Celsius, CG of order
      ``Field_Orders["Temperature"]`` (else ``Field_Orders["T"]``, else 1);
      set by the user.
    * ``Lame``, ``Shear``, ``Creep_Stress_Scale``, ``TensileStrength``:
      scalar quadrature fields with :math:`\lambda(p)`, :math:`\mu(p)`, the
      creep stress scale :math:`\eta_\mathrm{vp}(p,T)` of
      :meth:`~femcubes.glaciology.ice_material.IceMaterial.Get_ViscoPlastic_viscosity`
      (with :math:`d = 2` for ``true_2d``, 3 otherwise) and :math:`f_t`:
      the properties of the local update.
    * ``phasefield``: read only.

    .. rubric:: Weak form

    The residual and Jacobian of ``u`` are those of
    :doc:`../solids/cohesive_viscoelastic`, with the ice density
    :math:`\rho(p) = 917(1-p)` in the inertia and gravity terms:

    .. math::

        R_{\mathbf{u}} =
        \int_\Omega \boldsymbol{\varepsilon}(\delta\mathbf{u}):\boldsymbol{\sigma}\,\mathrm{d}x
        + \int_\Omega \rho(p)\,\delta\mathbf{u}\cdot\ddot{\mathbf{u}}\,\mathrm{d}x
        + \int_\Omega \frac{c}{\Delta t}\,\delta\mathbf{u}\cdot(\mathbf{u}-\mathbf{u}_n)\,\mathrm{d}x
        - \int_\Omega \rho(p)\,g\,\delta u_{d}\,\mathrm{d}x,

    with the inertia term only for ``Inertia=True`` (default ``False``),
    the damping term only for ``LE_Damping`` :math:`c>0` and the gravity
    term only for :math:`g\neq0`. With ``consistent_stiffness=False`` the
    Jacobian uses the elastic stiffness :math:`\mathbb{D}` of
    :math:`\lambda(p)`, :math:`\mu(p)` (plane strain / 3-D, condensed plane
    stress, or intrinsic 2-D).

    .. rubric:: Outside the weak form

    * Before every assembly: the Newmark update of ``v`` and ``a``; ``Lame``,
      ``Shear``, ``Creep_Stress_Scale`` and ``TensileStrength`` are
      interpolated from the current ``poros`` and ``T``; then the local
      update of :doc:`../solids/cohesive_viscoelastic` (reduced form, and
      principal space for ``r2_2L_PSD``) with these properties at every
      point. Points at which the reduced update does not converge are
      recomputed by the C++ kernel of
      :doc:`../solids/cohesive_viscoelastic_kernel` with the same per-point
      properties. The update writes ``sigma``, the tangent, ``eta``,
      ``etaNorm``, ``Plastic_Strain``, the multipliers, ``ElasticEnergy``,
      ``PsiField`` and, if kept, ``StrengthRatio``. The bulk modulus of the
      local problem is :math:`\lambda+2\mu/3`, or :math:`\lambda+\mu` for
      ``true_2d``. The Glen exponent is :math:`n = 3`.
    * **Checks**, on all ranks together before every assembly, raising the
      same ``ValueError`` on every rank: ``poros`` in :math:`[0, 1)`, ``T``
      at most 0 °C, and :math:`f_t` positive, below ``f_s`` for ``r2_2L``
      and ``r2_2L_DP``, and at most :math:`\sqrt{2/3}\,f_s` for
      ``r2_2L_PSD``. At the first assembly a warning is given if ``T`` is
      0 °C everywhere (probably never set). A point with inadmissible
      properties gets a non-finite stress instead of an exception.
    * ``Gc`` and ``GcII`` are not the toughness, which is :math:`G_c(p,T)`
      of the phase-field model: only their ratio is used, in the crack
      driving force of ``r1_DP``, as in :doc:`../solids/cohesive_viscoelastic`.
    * ``commit``, ``reset_step`` and the global measures ``Eta_Max`` and
      ``Eta_Active_Area`` (with ``Eta_Active_Threshold``) are those of
      :doc:`../solids/cohesive_viscoelastic`.
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
                "Ice": {
                    "ell": 10.0,             # REQUIRED: phase-field length scale (m); used by the phase-field model
                    "f_s": 1.5e6,            # REQUIRED: shear / compressive strength (Pa)
                    "FailureType": "r2_2L_PSD",  # OPTIONAL (default "r1"): strength surface, see Cohesive_ViscoElastic
                    "DP_phi": 0.0,           # OPTIONAL (default 0.0): friction angle of the DP types (deg)
                    "DP_q": 1.5e6,           # OPTIONAL (default f_s): confined strength cap of the DP types (Pa)
                    "kappa": 1.0e-6,         # OPTIONAL (default 1e-3): residual degradation (-)
                    "Gc": 1.0,               # OPTIONAL (default 1.0): reference toughness; only GcII/Gc is used, r1_DP only
                    "GcII": 1.0,             # OPTIONAL (default Gc): reference mode-II toughness (J/m^2)
                    "LE_Damping": 0.0,       # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                    "Inertia": False,        # OPTIONAL (default False): include the inertia term
                },
            }
            Models = {
                "Solid": {
                    "Type": "Glaciology:Cohesive_ViscoElastic",  # REQUIRED
                    "Material": "Ice",                # REQUIRED: key in Materials
                    "plane_stress": False,            # OPTIONAL (default False): 2-D plane stress
                    "true_2d": False,                 # OPTIONAL (default False): intrinsic 2-D law, not with plane_stress
                    "consistent_stiffness": True,     # OPTIONAL (default True): algorithmic tangent, else elastic stiffness
                    "Strength_Ratio_Field": False,    # OPTIONAL (default False): keep StrengthRatio; r2_2L / r2_2L_PSD only
                    "Eta_Active_Threshold": 1.0e-8,   # OPTIONAL (default 1e-8): threshold of Eta_Active_Area (-)
                    "domain": "domain",               # OPTIONAL (default "domain", the full mesh): domain group
                },
                "PF": {
                    "Type": "Glaciology:PhaseField",  # provides phasefield, with Gc(p, T)
                    "Material": "Ice",
                    "CDF_field": "PsiField",
                },
            }
    """

    MODEL_TYPE = "Glaciology:Cohesive_ViscoElastic"
    # Per-point properties of the local update, in the order of _IceCohesiveUpdate.set_properties
    _PROPERTY_FIELDS = ("Lame", "Shear", "Creep_Stress_Scale", "TensileStrength")
    _POINT_FIELDS = Cohesive_ViscoElastic._POINT_FIELDS + _PROPERTY_FIELDS

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        domain = params.Models[name].get("domain", "domain")
        add_ice_state_fields(params, mesh, domain)
        labels = ("Lame Constant [Pa]", "Shear Modulus [Pa]", "Creep Stress Scale [Pa s^(1/n)]",
                  "Tensile Strength [Pa]")
        for field_name, label in zip(self._PROPERTY_FIELDS, labels):
            mesh.Add_Quadrature_Field(field_name, label, (), domain=domain)
        self.iceMat = IceMaterial()
        self._property_expressions = {}
        self._temperature_checked = False
        super().__init__(name, params, mesh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Post_Remesh(self) -> None:
        """Reset the quadrature data and rebuild the material expressions on the new mesh."""
        super().Post_Remesh()
        self._bind_material()
        self._property_expressions.clear()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _read_material(self, material: dict) -> None:
        """PRIVATE: ice properties from ``poros`` and ``T``; the ratio GcII/Gc from ``material``."""
        if "GcII_over_Gc" in material:
            raise ValueError("GcII_over_Gc is not supported; give the reference toughnesses Gc and GcII instead.")
        if "GcII" in material and "Gc" not in material:
            raise ValueError("GcII needs the reference toughness Gc (only GcII/Gc is used).")
        self.Gc = material.get("Gc", 1.0)
        self.GcII = material.get("GcII", self.Gc)
        if not (self.Gc > 0.0 and self.GcII > 0.0):
            raise ValueError("The reference toughnesses Gc and GcII must be positive.")
        self.viscosity_type = "Glen"
        self._bind_material()

    def _bind_material(self) -> None:
        """PRIVATE: density, moduli, creep law, tensile strength and elastic stiffness from the
        live ``poros`` and ``T`` fields."""
        porosity, temperature = self.mesh.Get_Field("poros"), self.mesh.Get_Field("T")
        ice = self.iceMat
        self.rho = ice.Get_Density(porosity, temperature)
        self.lame = ice.Get_Lame_Parameter(porosity, temperature)
        self.shear = ice.Get_Shear_Modulus(porosity, temperature)
        self.bulk = self.lame + self.shear if self.true_2d else ice.Get_Bulk_Modulus(porosity, temperature)
        self.plastic_n = ice.Get_Creep_n(porosity, temperature)
        self.viscosity = ice.Get_ViscoPlastic_viscosity(porosity, temperature, dim=2 if self.true_2d else 3)
        self.f_t = ice.Get_Tensile_Strength(porosity, temperature)
        self.D = elastic_stiffness(self.lame, self.shear, true_2d=self.true_2d, plane_stress=self.plane_stress)

    def _build_update(self, mesh: Mesh):
        """PRIVATE: the update of Solid:Cohesive_ViscoElastic with per-point properties."""
        return _IceCohesiveUpdate(
            failure_type=self.failureType, f_s=self.f_s, kappa=self.kappa, n_exp=self.plastic_n,
            G2_over_G1=self.GcII / self.Gc, DP_phi=self.DP_phi, DP_q=self.DP_q,
            plane_stress=self.plane_stress, true_2d=self.true_2d)

    def _update_ip_fields(self, strain_expr) -> None:
        """PRIVATE: sample the ice properties at the points, check them on all ranks, run the update."""
        values = (self.lame, self.shear, self.viscosity, self.f_t)
        for field_name, value in zip(self._PROPERTY_FIELDS, values):
            self._property_expressions[field_name] = self._points.interpolate(
                field_name, value, self._property_expressions.get(field_name))
        properties = self._points.gather(*self._PROPERTY_FIELDS)
        label = f"{self.MODEL_TYPE} '{self.name}'"
        temperature_unset = check_ice_state(self.mesh, self.domain, label,
                                            tensile_strength=(properties[3], self.failureType, self.f_s))
        if temperature_unset and not self._temperature_checked:
            warn_temperature_not_set(self.mesh, label)
        self._temperature_checked = True
        self._update_cve_state.set_properties(*properties)
        super()._update_ip_fields(strain_expr)

    def _strength_surface_scales(self):
        """PRIVATE: per-point (A, B) of the strength surface of ``StrengthRatio``."""
        return opening_strength_scales(self._points.gather("TensileStrength")[0], self.f_s)


class _IceCohesiveUpdate:
    """PRIVATE: integration-point update of Solid:Cohesive_ViscoElastic with per-point Lame
    parameter, shear modulus, creep stress scale and tensile strength.

    Called like the C++ kernel, after ``set_properties``. Builds the reduced update (or, for
    "r2_2L_PSD", the principal-space update) of these properties; the points the reduced update
    does not converge at are recomputed by the variable-property C++ kernel. Points with
    inadmissible properties are computed with admissible stand-ins and get a non-finite stress,
    so that no exception is raised on one rank only."""

    def __init__(self, *, failure_type, f_s, kappa, n_exp, G2_over_G1, DP_phi, DP_q, plane_stress, true_2d):
        if failure_type == "r2_2L_PSD" and true_2d:
            raise NotImplementedError("r2_2L_PSD is not implemented for true_2d.")
        self.failure_type, self.f_s, self.kappa, self.n_exp = failure_type, float(f_s), kappa, n_exp
        self.G2_over_G1, self.DP_phi, self.DP_q = G2_over_G1, DP_phi, DP_q
        self.plane_stress, self.true_2d = plane_stress, true_2d
        self.fallback = None
        if failure_type != "r2_2L_PSD":
            self.fallback = _CVEVariablePropsKernels(n_exp, f_s, G2_over_G1, DP_phi, DP_q, kappa, failure_type,
                                                     plane_stress, true_2d)
        # reject inadmissible constants (f_s, kappa, DP parameters) now, with admissible stand-in properties
        self._update_for(np.ones(1), np.ones(1), np.ones(1), np.full(1, 0.5 * self.f_s))
        self.properties = None

    def set_properties(self, lame, shear, viscosity, f_t) -> None:
        """PRIVATE: the point values of the properties for the next calls."""
        self.properties = tuple(np.array(v, dtype=float).reshape(-1) for v in (lame, shear, viscosity, f_t))

    def __call__(self, strain, dt, damage, stress, *state):
        n = damage.size
        if n == 0:
            return 0
        lame, shear, viscosity, f_t = self.properties
        ok = (admissible_tensile_strength(f_t, self.failure_type, self.f_s) & np.isfinite(lame) & np.isfinite(shear)
              & (shear > 0.0) & (3.0 * lame + 2.0 * shear > 0.0) & np.isfinite(viscosity) & (viscosity >= 0.0))
        if not ok.all():
            lame, shear, viscosity = (np.where(ok, v, 1.0) for v in (lame, shear, viscosity))
            f_t = np.where(ok, f_t, 0.5 * self.f_s)
        recomputed = self._update_for(lame, shear, viscosity, f_t)(strain, dt, damage, stress, *state)
        stress.reshape(n, 6)[~ok] = np.nan
        return recomputed

    def _update_for(self, lame, shear, viscosity, f_t):
        """PRIVATE: the update of these per-point properties."""
        if self.failure_type == "r2_2L_PSD":
            return PrincipalCVEUpdate(lame=lame, shear=shear, f_t=f_t, f_s=self.f_s, kappa=self.kappa,
                                      viscosity=viscosity, n_exp=self.n_exp, plane_stress=self.plane_stress)
        material = ReducedCVEMaterial.from_model_constants(
            true_2d=self.true_2d, lame=lame, shear=shear, bulk=lame + 2.0 * shear / 3.0, viscosity=viscosity,
            n_exp=self.n_exp, f_t=f_t, f_s=self.f_s, kappa=self.kappa, failure_type=self.failure_type,
            G2_over_G1=self.G2_over_G1, DP_phi=self.DP_phi, DP_q=self.DP_q)
        return _VariablePropsReducedUpdate(material, plane_stress=self.plane_stress, fallback=self.fallback)


class _VariablePropsReducedUpdate(ReducedCVEUpdate):
    """PRIVATE: the reduced update of per-point properties; its fallback is the variable-property
    C++ kernel, called with the properties of the points it recomputes."""

    def _call_fallback(self, idx, sub, dt, damage):
        m = self.material
        self.fallback(sub[0], dt, damage, m.lame[idx], m.shear[idx], m.viscosity[idx], m.f_t[idx], *sub[1:])
