"""Cohesive visco-elastic solid with the local problem solved in reduced (invariant) form."""

import numpy as np

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.solids._cohesive_viscoelastic_base import _CohesiveViscoElasticBase
from femcubes.solids._domain_points import _DomainPoints
from femcubes.solids._principal_cve import PrincipalCVEUpdate, opening_strength_scales, strength_ratio
from femcubes.solids._reduced_cve import ReducedCVEMaterial, ReducedCVEUpdate


class Cohesive_ViscoElastic(_CohesiveViscoElasticBase):
    r"""Cohesive visco-elastic solid with Newmark time integration.

    The model describes fracture through an explicit strength surface,
    rather than through the degradation of the elastic stiffness. Following
    :cite:t:`Vicentini2026` and :cite:t:`Bourdin2025`, cracks are represented
    by fracture eigenstrains: once the stress reaches the strength surface,
    the eigenstrain develops such that the stress remains on it, and the
    phase field degrades the strength potential, describing the loss of
    cohesion as the crack opens. Crack nucleation is therefore set by the
    strength of the material, independent of the phase-field length scale,
    while :math:`G_c` sets the energy dissipated as cracks grow. As the
    eigenstrains need no spatial gradients, they are resolved at each
    integration point by a return-mapping scheme :cite:p:`Hageman2026b`.
    Creep enters the same local problem :cite:p:`Hageman2026CMAME`. Creep and
    fracture interact only through the stress, so that no choice of a
    damaged or undamaged stress to drive creep is needed.

    The small-strain momentum balance has a strain that splits additively
    into elastic, viscous (creep) and cohesive parts,
    :math:`\boldsymbol\varepsilon = \boldsymbol\varepsilon^e + \boldsymbol\varepsilon^v + \boldsymbol\eta`.
    Creep, Newtonian or following the power law of :cite:t:`Glen1955`, and
    the cohesive eigenstrain are solved together in one local problem per
    quadrature point, in reduced (invariant) form; points where that does not converge
    are recomputed by the C++ kernel of :doc:`cohesive_viscoelastic_kernel`,
    which solves the same local problem. The phase field degrades only the
    failure potential. Supports 2-D (plane strain, plane stress with
    ``plane_stress``, or intrinsic 2-D with ``true_2d``) and 3-D, with the
    failure types listed below. A ``phasefield`` field must exist, provided
    by :doc:`phase_field` (with ``CDF_field="PsiField"``) or
    :doc:`no_phase_field`; the ``DamageModel`` setting is not read.
    :doc:`cohesive_linear_elastic` is this model without creep.

    .. rubric:: Fields

    * ``u``: displacement :math:`\mathbf{u}`, vector CG of order
      ``Field_Orders["u"]``, solved in step ``Solution_Steps["u"]``.
    * ``v``, ``a``, ``uOld``, ``vOld``, ``aOld``: velocity, acceleration
      and committed values, vector CG of the same order.
    * ``sigma`` (6 Mandel components), ``dsigma_dStrainMandel``
      (:math:`6\times6`): stress and algorithmic tangent, quadrature fields.
    * ``Plastic_Strain``, ``Plastic_Strain_Old``: viscous strain
      :math:`\boldsymbol\varepsilon^v` and :math:`\boldsymbol\varepsilon^v_n`;
      ``LambdaField_visc``, ``LambdaFieldOld_visc``: viscous multiplier
      :math:`\Lambda_v`.
    * ``eta``, ``etaOld``: cohesive strain :math:`\boldsymbol\eta`;
      ``etaNorm``: :math:`\|\boldsymbol\eta\|` (output, e.g. for mesh
      refinement); ``LambdaField_coh``, ``LambdaFieldOld_coh``: cohesive
      multipliers (2 components; for ``r2_2L_PSD`` the first holds the
      undegraded potential :math:`\psi` instead).
    * ``ElasticEnergy``: elastic energy density of
      :math:`\boldsymbol\varepsilon^e` (output); ``PsiField``: crack
      driving force :math:`\psi` for :doc:`phase_field`.
    * ``StrengthRatio`` (with ``Strength_Ratio_Field=True`` only): stress
      relative to the degraded strength surface (output, see below).
    * ``Total_Strain``, ``dam_elastic``: strain and :math:`d_e` at the
      quadrature points (update input; with ``plane_stress`` the update
      writes :math:`\varepsilon_{zz}` back into ``Total_Strain``).
    * ``phasefield``: read only.

    .. rubric:: Weak form

    With test function :math:`\delta\mathbf{u}` and :math:`\Omega` the
    domain group named by ``domain``, the contribution to the residual of
    ``u`` is

    .. math::

        R_{\mathbf{u}} =
        \int_\Omega \boldsymbol\varepsilon(\delta\mathbf{u}):\boldsymbol\sigma\,\mathrm{d}x
        + \int_\Omega \rho\,\delta\mathbf{u}\cdot\ddot{\mathbf{u}}\,\mathrm{d}x
        + \int_\Omega \frac{c}{\Delta t}\,\delta\mathbf{u}\cdot(\mathbf{u}-\mathbf{u}_n)\,\mathrm{d}x
        - \int_\Omega \rho\,g\,\delta u_d\,\mathrm{d}x,

    with :math:`\boldsymbol\sigma` the quadrature field ``sigma``. The
    inertia term is assembled for ``Inertia=True`` (default ``False``),
    with :math:`\ddot{\mathbf{u}}` from the Newmark relation of
    :doc:`linear_elastic`; the damping term (a backward difference, not
    the Newmark velocity) for ``LE_Damping`` :math:`c>0`; the gravity term
    for ``gravity`` :math:`g\neq0`, along the last coordinate. Density is
    not degraded by the phase field. The Jacobian is

    .. math::

        K_{\mathbf{u}} =
        \int_\Omega \boldsymbol\varepsilon(\delta\mathbf{u}):\mathbb{K}:\boldsymbol\varepsilon(\Delta\mathbf{u})\,\mathrm{d}x
        + \int_\Omega \frac{\rho}{\beta\Delta t^2}\,\delta\mathbf{u}\cdot\Delta\mathbf{u}\,\mathrm{d}x
        + \int_\Omega \frac{c}{\Delta t}\,\delta\mathbf{u}\cdot\Delta\mathbf{u}\,\mathrm{d}x,

    where :math:`\mathbb{K} = \mathbb{C}_\mathrm{alg}` (``dsigma_dStrainMandel``,
    consistent at fixed :math:`\phi`) for ``consistent_stiffness=True``
    (default), and the constant elastic stiffness :math:`\mathbb{D}`
    (plane strain / 3-D, condensed plane stress, or intrinsic 2-D) for
    ``consistent_stiffness=False``, which is then not the derivative of
    :math:`R_{\mathbf{u}}`.

    .. rubric:: Outside the weak form

    * **Local problem.** Before every assembly, the strain
      :math:`\boldsymbol\varepsilon(\mathbf{u})` and
      :math:`d_e = \kappa + (1-\kappa)(1-\hat\phi)^2`
      (:math:`\hat\phi = \min(\max(\phi,0),1)`, clamped to
      :math:`[\kappa,1]`) are sampled at the quadrature points, and
      :math:`(\boldsymbol\sigma, \Lambda_v, \Lambda_i)` are found with
      backward Euler in time:

      .. math::

          \boldsymbol\sigma = \mathbb{D}:(\boldsymbol\varepsilon - \boldsymbol\varepsilon^v - \boldsymbol\eta),
          \qquad
          \boldsymbol\varepsilon^v = \boldsymbol\varepsilon^v_n + \Lambda_v\frac{\partial f}{\partial\boldsymbol\sigma},
          \qquad
          \Lambda_v = \Delta t\Bigl(\frac{f(\boldsymbol\sigma)}{\eta_v}\Bigr)^{n},

      with :math:`f = \sqrt{\tfrac{d_c}{2}\mathbf{s}:\mathbf{s}}`
      (:math:`\mathbf{s}` the stress deviator, :math:`d_c = 3`, or 2 for
      ``true_2d``), and the cohesive strain
      :math:`\boldsymbol\eta = \sum_i\Lambda_i\mathbf{G}_i(\mathbf{e})` on
      :math:`\mathbf{e} = \boldsymbol\varepsilon - \boldsymbol\varepsilon^v`
      with the directions and potential :math:`F` of the failure type
      (list below; branches selected by :math:`\mathrm{tr}\,\mathbf{e}`),
      subject to the complementarity conditions

      .. math::

          \Lambda_i\geq0,\quad
          r_i = \frac{\partial F}{\partial\Lambda_i} - \boldsymbol\sigma:\mathbf{G}_i + \eta_c\Lambda_i \geq 0,\quad
          \Lambda_i r_i = 0,

      :math:`\eta_c = 10^{-8}K\kappa`. The creep law gives
      :math:`\dot{\boldsymbol\varepsilon}^v = \mathbf{s}/(2\eta_N)` for
      ``Visc_Type="Newtonian"`` (:math:`\eta_v = d_c\eta_N`, :math:`n=1`)
      and :math:`\dot{\boldsymbol\varepsilon}^v = A\tau_e^{n-1}\mathbf{s}`,
      :math:`\tau_e = \sqrt{\mathbf{s}:\mathbf{s}/2}`, for
      ``Visc_Type="Glen"``
      (:math:`\eta_v = (d_c^{(n+1)/2}/(2A))^{1/n}`); ``"None"`` disables
      creep. :math:`\mathbb{D}` is built from ``Lame`` and ``Shear``;
      :math:`K` = ``Bulk`` (:math:`\lambda+\mu` for ``true_2d``) enters
      the failure potential, the regularization and ``ElasticEnergy``.
      With ``plane_stress``, :math:`\varepsilon_{zz}` is an extra local
      unknown with :math:`\sigma_{zz}=0`, and the tangent is condensed.
      The update returns :math:`\boldsymbol\sigma`,
      :math:`\boldsymbol\varepsilon^v`, :math:`\boldsymbol\eta`, the
      multipliers, :math:`\mathbb{C}_\mathrm{alg}` (implicit
      differentiation of the converged local system), ``ElasticEnergy`` and
      ``PsiField``.
    * **Reduced local update.** The local problem is isotropic, so the
      creep strain, the cohesive strain and the stress lie in the plane of
      the volumetric and deviatoric directions of the trial strain
      :math:`\boldsymbol\varepsilon - \boldsymbol\varepsilon^v_n`, and the
      unknowns reduce to the creep multiplier and the one or two cohesive
      multipliers (algorithm and accuracy at the top of this page). The
      potentials, branches, penalties and regularizations are those of the
      kernel, so the update reproduces the kernel to its tolerance; inside
      the split r2 surfaces the multipliers are exactly zero (the cohesive
      strain, and so ``etaNorm``, is exactly zero below the strength
      surface). Points where the reduced Newton method does not converge
      are recomputed by the C++ kernel.
    * ``r2_2L_PSD`` is solved in principal space: for each of the seven
      sets of opening principal directions the stationarity condition
      reduces to one scalar equation, and the admissible candidate of
      lowest energy is kept; with creep, the principal creep increment is
      found by Newton's method around it. Non-converged points get a NaN
      stress (no exception) so that the global solver rejects the step on
      all ranks. Not available with ``true_2d``.
    * **History.** Only :math:`\boldsymbol\varepsilon^v_n` is history; the
      state depends only on the strain, :math:`\boldsymbol\varepsilon^v_n`,
      :math:`\Delta t` and :math:`d_e` (no warm start). If the local
      problem fails, ``sigma`` and the tangent are NaN so that the global
      solver rejects the step.
    * **Crack driving force.** ``PsiField`` is
      :math:`\psi = \partial F/\partial d_e` at the converged state, as
      listed below.
    * ``StrengthRatio`` (``r2_2L`` and ``r2_2L_PSD`` only) is the gauge of
      the stress on the degraded strength surface,

      .. math::

          \frac{1}{d_e}\sqrt{\frac{\langle\mathrm{tr}\,\boldsymbol\sigma\rangle_+^2}{9A}
          + \frac{\|\mathrm{dev}\,\boldsymbol\sigma\|^2}{B}},
          \qquad A = \Bigl(\frac{\tilde f_t}{3}\Bigr)^2,\quad B = \frac{2f_s^2}{3},

      below 1 inside the surface and 1 on it; for ``r2_2L_PSD`` it is
      evaluated for the tensile part of the stress (an upper bound).
      Unlike ``etaNorm`` it is non-zero below the strength surface, so a
      refinement indicator on it (threshold below 1) refines the mesh
      before the first cohesive strain forms.
    * **Global measures**, written every step: ``Eta_Max``, the largest
      ``etaNorm``, and ``Eta_Active_Area``,
      :math:`\int_\Omega [\|\boldsymbol\eta\| > \eta_\mathrm{act}]\,\mathrm{d}x`
      with :math:`\eta_\mathrm{act}` = ``Eta_Active_Threshold``. The first
      non-zero value marks the onset of the cohesive strain.
    * ``commit`` runs the Newmark update and copies ``u``, ``v``, ``a``,
      ``Plastic_Strain``, ``LambdaField_visc``, ``eta`` and
      ``LambdaField_coh`` to their ``Old`` fields; ``reset_step`` copies
      them back.
    * No Dirichlet conditions.

    .. rubric:: Failure types

    ``FailureType`` is one of ``r1``, ``r1_DP``, ``r2``, ``r2_DP``,
    ``r2_2L``, ``r2_2L_DP`` and ``r2_2L_PSD``; other values raise a
    ``ValueError``. The ``r1`` and ``r2`` potentials are those of
    :cite:t:`Vicentini2026`. The Drucker-Prager-like variants (``r1_DP``,
    ``r2_DP``, ``r2_2L_DP``) increase the shear strength under compression,
    in the manner of the Drucker-Prager-like criterion of
    :cite:t:`Hageman2026b`. The two-multiplier criterion ``r2_2L`` and the
    opening-only criterion ``r2_2L_PSD`` share their potential but differ in
    the admissible eigenstrains :cite:p:`Hageman2026CMAME`: once the cohesion
    is lost, ``r2_2L`` admits only hydrostatic compression, so that fully
    damaged material responds to compression as a fluid, whereas
    ``r2_2L_PSD`` restricts the eigenstrain to opening, so that fully damaged
    material is a no-tension material that still transfers compressive
    stresses. In return, ``r2_2L_PSD`` does not describe compressive
    failure. Notation: :math:`\mathrm{tr}` and :math:`\mathrm{dev}`
    are 3-D (2-D for ``true_2d``, with :math:`\sqrt3` replaced by
    :math:`\sqrt2`); :math:`\|\cdot\|` is the Mandel (Frobenius) norm;
    :math:`\mathbf{n}_e = \mathbf{e}/\|\mathbf{e}\|`,
    :math:`\mathbf{n}_d = \mathrm{dev}\,\mathbf{e}/\|\mathrm{dev}\,\mathbf{e}\|`
    (uniaxial deviator if :math:`\mathrm{dev}\,\mathbf{e}\approx\mathbf{0}`),
    :math:`s_e = \mathrm{sign}(\mathrm{tr}\,\mathbf{e})`, and

    .. math::

        F_1 = \frac{f_t}{3}\,\mathrm{tr}\,\boldsymbol\eta,
        \qquad
        F_2 = \frac{f_s}{\sqrt2}\,\|\mathrm{dev}\,\boldsymbol\eta\|

    (:math:`\|\mathrm{dev}\,\boldsymbol\eta\|` regularized by
    :math:`10^{-10}`). The branch "compression" means
    :math:`\mathrm{tr}\,\mathbf{e}<0` (``r2_DP``:
    :math:`\leq0`). :math:`\phi_{DP}` = ``DP_phi`` (degrees),
    :math:`q_{DP}` = ``DP_q``, :math:`K` = ``Bulk``.

    Cohesive strain directions:

    * two multipliers (``r1``, ``r1_DP``, ``r2_2L``, ``r2_2L_DP``):
      :math:`\boldsymbol\eta = \Lambda_1 s_e\mathbf{I}/\sqrt3 + \Lambda_2\mathbf{n}_d`;
    * one multiplier (``r2``, ``r2_DP``): :math:`\boldsymbol\eta = \Lambda\mathbf{n}_e`
      for :math:`\mathrm{tr}\,\mathbf{e}\geq0`, :math:`\Lambda\mathbf{n}_d`
      otherwise;
    * ``r2_2L_PSD``: :math:`\boldsymbol\eta = \sum_{i=1}^3\Lambda_i\,\mathbf{n}_i\otimes\mathbf{n}_i`
      with :math:`\mathbf{n}_i` the principal directions of
      :math:`\mathbf{e}` (positive semidefinite, opening only).

    Potentials :math:`F(\boldsymbol\eta)` and driving forces :math:`\psi`:

    * ``r1``: :math:`d_e(F_1+F_2)`, :math:`\psi = F_1+F_2`; in
      compression :math:`-10^6F_1 + d_eF_2`, :math:`\psi = F_2`.
    * ``r1_DP``: :math:`d_eF_1` (compression: :math:`-10^6F_1`) plus
      :math:`d_e\,q\,\|\mathrm{dev}\,\boldsymbol\eta\|/\sqrt2`
      with :math:`q = q_0 = \max(f_s - \tfrac{f_t}{3}\tan\phi_{DP},0)` for
      :math:`K\,\mathrm{tr}\,\mathbf{e}\geq f_t/3` and
      :math:`q = \min(f_s - K\tan\phi_{DP}\,\mathrm{tr}\,\mathbf{e},\,q_{DP})`
      otherwise; :math:`\psi = F_1` (0 in compression)
      :math:`+\,(G_c/G_{c,II})\,q_0\|\mathrm{dev}\,\boldsymbol\eta\|/\sqrt2`.
      This is the only use of ``GcII``.
    * ``r2``, ``r2_2L``, ``r2_2L_DP``, ``r2_2L_PSD``:
      :math:`F = d_e\bigl(\sqrt{\tilde F_1^2 + \tilde F_2^2 + \epsilon_F^2} - \epsilon_F\bigr)`,
      :math:`\psi = F/d_e`, with :math:`\epsilon_F = 10^{-6}` Pa,
      :math:`\tilde F_1 = \tfrac{\tilde f_t}{3}\mathrm{tr}\,\boldsymbol\eta`,
      :math:`\tilde F_2 = \tfrac{\tilde f_s}{\sqrt2}\|\mathrm{dev}\,\boldsymbol\eta\|`,
      :math:`\tilde f_s = 2f_s/\sqrt3` (:math:`f_s` for ``true_2d``),
      :math:`\tilde f_t = f_t` for ``r2`` and
      :math:`\tilde f_t = f_t/\sqrt{1-(f_t/f_s)^2}` for the others, so that
      ``f_t`` and ``f_s`` are the uniaxial tensile and compressive
      strengths (requires :math:`f_s>f_t`). In compression, ``r2``,
      ``r2_2L`` and ``r2_2L_DP`` drop :math:`\tilde F_1` from the root and
      add the penalty :math:`-10^6\tilde F_1`; ``r2_2L_DP`` then also
      replaces :math:`f_s` in :math:`\tilde f_s` by
      :math:`\min(f_s + p\tan\phi_{DP},\,q_{DP})` with
      :math:`p = \max(-K\,\mathrm{tr}(\mathbf{e}-\boldsymbol\eta),0)`.
      ``r2_2L_PSD`` has no penalty (its directions keep
      :math:`\mathrm{tr}\,\boldsymbol\eta\geq0`) and requires
      :math:`f_t\leq\sqrt{2/3}f_s`.
    * ``r2_DP``: :math:`d_e\sqrt{F_1^2+F_2^2}` for
      :math:`\mathrm{tr}\,\mathbf{e}>0`, otherwise
      :math:`-10^{12}F_1 + d_e\,q\,\|\mathrm{dev}\,\boldsymbol\eta\|/\sqrt2`,
      :math:`q = \min(f_s - K\tan\phi_{DP}\,\mathrm{tr}\,\mathbf{e},\,q_{DP})`;
      :math:`\psi = F/d_e` without the penalty.

    ``r1_DP``, ``r2_DP`` and ``r2_2L_DP`` require :math:`\phi_{DP}\geq0`,
    :math:`q_{DP}\geq f_s`, and :math:`\phi_{DP}>0` when
    :math:`q_{DP}>f_s`.
    :func:`~femcubes.solids.failure_surface_plot.plot_FailureSurface`
    plots the resulting strength surfaces.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            Field_Orders = {"u": 2}          # REQUIRED: polynomial order of u
            Solution_Steps = {"u": 0}        # REQUIRED: staggered step solving u
            gravity = 0.0                    # REQUIRED: acceleration along the last axis (m/s^2)
            Newmark_Beta = 0.25              # REQUIRED: Newmark beta (-), also without inertia
            Newmark_Gamma = 0.5              # REQUIRED: Newmark gamma (-)
            Materials = {
                "Ice": {
                    "Lame": 6.5e9,           # REQUIRED: first Lame parameter (Pa)
                    "Shear": 3.5e9,          # REQUIRED: shear modulus (Pa)
                    "Bulk": 8.8e9,           # REQUIRED: bulk modulus (Pa), consistent with Lame and Shear
                    "Density": 917.0,        # REQUIRED: density (kg/m^3)
                    "f_t": 0.4e6,            # REQUIRED: tensile strength (Pa)
                    "f_s": 1.5e6,            # REQUIRED: shear / compressive strength (Pa)
                    "Gc": 1.0,               # REQUIRED: fracture toughness (J/m^2); here only GcII/Gc is used
                    "ell": 0.1,              # REQUIRED: length scale (m); read, used by Solid:PhaseField only
                    "Visc_Type": "Glen",     # OPTIONAL (default "Newtonian"): "Newtonian", "Glen" or "None"
                    "Glen_A": 1.0e-24,       # REQUIRED for "Glen": rate factor A (Pa^-n s^-1)
                    "Glen_n": 3.0,           # REQUIRED for "Glen": exponent n (-)
                    # "Viscosity": 1.0e13,   # REQUIRED for "Newtonian": shear viscosity eta_N (Pa s)
                    "FailureType": "r2_2L",  # OPTIONAL (default "r1"): cohesive potential, see the list above
                    "GcII": 1.0,             # OPTIONAL (default Gc): mode-II toughness, r1_DP only (J/m^2)
                    "kappa": 1.0e-3,         # OPTIONAL (default 1e-3): residual degradation (-)
                    "DP_phi": 0.0,           # OPTIONAL (default 0.0): friction angle of the DP types (deg)
                    "DP_q": 1.5e6,           # OPTIONAL (default f_s): confined strength cap of the DP types (Pa)
                    "Inertia": False,        # OPTIONAL (default False): include the inertia term
                    "LE_Damping": 0.0,       # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                },
            }
            Models = {
                "Solid": {
                    "Type": "Solid:Cohesive_ViscoElastic",  # REQUIRED
                    "Material": "Ice",                  # REQUIRED: key in Materials
                    "plane_stress": False,              # OPTIONAL (default False): 2-D plane stress
                    "true_2d": False,                   # OPTIONAL (default False): intrinsic 2-D law; not with plane_stress
                    "consistent_stiffness": True,       # OPTIONAL (default True): algorithmic tangent, else elastic stiffness
                    "Strength_Ratio_Field": False,      # OPTIONAL (default False): keep StrengthRatio; r2_2L / r2_2L_PSD only
                    "Eta_Active_Threshold": 1.0e-8,     # OPTIONAL (default 1e-8): threshold of Eta_Active_Area (-)
                    "domain": "domain",                 # OPTIONAL (default "domain", the full mesh): domain group
                },
                "pf": {
                    "Type": "Solid:PhaseField",         # provides phasefield
                    "Material": "Ice",
                    "CDF_field": "PsiField",
                },
            }

    Notes
    -----
    For monolithic L-BFGS solves through crack initiation,
    ``consistent_stiffness=False`` gives a positive definite base matrix;
    the algorithmic tangent can become indefinite during cohesive
    softening.
    """

    MODEL_TYPE = "Solid:Cohesive_ViscoElastic"

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        material = params.Materials[params.Models[name]["Material"]]
        self.strength_ratio_field = params.Models[name].get("Strength_Ratio_Field", False)
        if not isinstance(self.strength_ratio_field, bool):
            raise ValueError("Strength_Ratio_Field must be True or False.")
        if self.strength_ratio_field and material.get("FailureType", "r1") not in ("r2_2L", "r2_2L_PSD"):
            raise ValueError('Strength_Ratio_Field is available for FailureType "r2_2L" and "r2_2L_PSD" only.')
        super().__init__(name, params, mesh)
        if self.strength_ratio_field:
            mesh.Add_Quadrature_Field("StrengthRatio", "Stress relative to the degraded strength surface [-]",
                                      (), domain=self.domain)
            self._points = _DomainPoints(mesh, self.domain, self._POINT_FIELDS + ("StrengthRatio",))

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_update(self, mesh: Mesh):
        """PRIVATE: the reduced return mapping with the C++ kernel as fallback for the points it
        does not converge at; for "r2_2L_PSD" the principal-space update (no fallback)."""
        if self.failureType == "r2_2L_PSD":
            self.reduced_material = None
            return PrincipalCVEUpdate(
                lame=self.lame, shear=self.shear, f_t=self.f_t, f_s=self.f_s, kappa=self.kappa,
                viscosity=self.viscosity, n_exp=self.plastic_n, plane_stress=self.plane_stress,
                true_2d=self.true_2d)
        self.reduced_material = ReducedCVEMaterial.from_model_constants(
            true_2d=self.true_2d, lame=self.lame, shear=self.shear, bulk=self.bulk, viscosity=self.viscosity,
            n_exp=self.plastic_n, f_t=self.f_t, f_s=self.f_s, kappa=self.kappa, failure_type=self.failureType,
            G2_over_G1=self.GcII / self.Gc, DP_phi=self.DP_phi, DP_q=self.DP_q,
        )
        return ReducedCVEUpdate(self.reduced_material, plane_stress=self.plane_stress,
                                fallback=self._build_kernel())

    def _multiplier_slots(self, failure_type: str) -> int:
        """PRIVATE: two components of ``LambdaField_coh`` for every type (for "r2_2L_PSD" the first
        holds the undegraded potential)."""
        return 2

    def _update_ip_fields(self, strain_expr) -> None:
        super()._update_ip_fields(strain_expr)
        if not self.strength_ratio_field:
            return
        sigma, damage = self._points.gather("sigma", "dam_elastic")
        ratio = strength_ratio(sigma, np.clip(damage, self.kappa, 1.0), *self._strength_surface_scales(),
                               opening_only=self.failureType == "r2_2L_PSD")
        self._points.scatter({"StrengthRatio": ratio})

    def _strength_surface_scales(self):
        """PRIVATE: (A, B) of the strength surface of ``StrengthRatio``, numbers or per-point arrays."""
        return opening_strength_scales(self.f_t, self.f_s)
