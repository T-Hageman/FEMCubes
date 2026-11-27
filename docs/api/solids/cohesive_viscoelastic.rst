Cohesive_ViscoElastic
=====================

``Solid:Cohesive_ViscoElastic`` is the strength-based cohesive phase-field model with creep
of :cite:t:`Hageman2026CMAME`. It combines the fracture eigenstrains and explicit strength
surfaces of :cite:t:`Vicentini2026` with power-law creep in one local constitutive problem per
integration point, extending the return mapping of :cite:t:`Hageman2026b`. Here, this local
problem is solved in a reduced (invariant) form. The same local problem is solved by the C++
kernel of :doc:`cohesive_viscoelastic_kernel`, by one Newton iteration over the stress and all
multipliers; both models have the same fields, material parameters, weak form, phase-field
coupling and history handling. This page first describes the reduced update; the model
settings are in the class documentation below. The reduced update reproduces the kernel to the
kernel's tolerance and is typically 5-7 times faster in fracture simulations, as the local
problem is then no longer the dominant cost. The derivation is written out in a technical
note, :download:`reduced_return_mapping.tex <../../notes/reduced_return_mapping.tex>` (LaTeX
source in ``docs/notes``; compile it with ``pdflatex``).

Reduced local problem
---------------------

The local problem is isotropic. Write the trial strain as
:math:`\boldsymbol\varepsilon-\boldsymbol\varepsilon^v_{old}=e_v\,\hat{\mathbf i}+e_q\,\mathbf n`,
with :math:`\hat{\mathbf i}` the unit volumetric direction and :math:`\mathbf n` the unit
deviatoric direction (in-plane components for intrinsic 2-D, :math:`n_d=2`; :math:`n_d=3` for
plane strain, plane stress and 3-D). The J2 creep strain is along :math:`\mathbf n`, and the
cohesive strain directions of every failure type lie in the plane of :math:`\hat{\mathbf i}`
and :math:`\mathbf n`:

* split types (``r1``, ``r1_DP``, ``r2_2L``, ``r2_2L_DP``): one multiplier along
  :math:`\pm\hat{\mathbf i}` and one along :math:`\mathbf n`;
* single-multiplier types (``r2``, ``r2_DP``): one multiplier along the full pre-cohesive
  strain direction in tension and along :math:`\mathbf n` in compression.

``r2_2L_PSD`` (opening along the principal directions) does not lie in this plane; it is
solved in principal space instead, as described in the class documentation.

Hence the stress is :math:`a\,\hat{\mathbf i}+b\,\mathbf n` and the cohesive strain
:math:`\eta_v\hat{\mathbf i}+\eta_q\mathbf n`, with :math:`a=k_{vol}(e_v-\eta_v)` and
:math:`b=2\mu(e_q-\gamma\Lambda_v-\eta_q)`, :math:`k_{vol}=n_d\lambda+2\mu`,
:math:`\gamma=\sqrt{n_d/2}`. The unknowns reduce to the creep multiplier :math:`\Lambda_v`
and the one or two cohesive multipliers, and every failure potential is a function of
:math:`(\eta_v,\eta_q)` and the pre-cohesive volumetric strain (which selects the tensile or
compressive branch and the Drucker-Prager confinement). The potentials are written in these
coordinates with the kernel's branches, penalties and regularizations.

Algorithm
---------

* **Creep-only state.** Without cohesive strain, the creep multiplier solves the scalar equation
  :math:`x=c(1-x)^n` for :math:`x=\gamma\Lambda_v/e_q` in the single number
  :math:`c=\gamma\Lambda_{v,trial}/e_q`. It is solved once per creep exponent by bisection and
  tabulated (cubic Hermite in :math:`\log_{10}c`, exact values and derivatives); most points
  (intact, creeping, or below the failure surface) need no iteration at all.
* **Activity.** The cohesive modes are tested at the creep-only state as in the kernel: the
  drive of each mode at a small probe multiplier and, for the split r2 criteria in tension,
  the dual norm of the volumetric and deviatoric stress projections.
* **Return.** Only points with an active mode solve the reduced complementarity system: the creep
  equation and, for each cohesive mode, the Fischer-Burmeister form :cite:p:`Fischer1992` of
  "multiplier at its bound, or drive
  :math:`\partial F/\partial\lambda_i-\boldsymbol\sigma\cdot\mathbf d_i+\eta_c\lambda_i`
  zero" (the kernel's residuals), by a semi-smooth Newton method :cite:p:`QiSun1993` with a
  backtracking line search in two or three unknowns, from the creep-only state and an elastic
  estimate of the active multipliers.
* **Tangent.** The consistent tangent follows by implicit differentiation of the reduced system,
  plus the rotation of :math:`\mathbf n` with the strain.
* **Plane stress.** :math:`\sigma_{zz}(\varepsilon_{zz})=0` is solved per point by a safeguarded
  Newton method from the elastic plane-stress strain; the solved :math:`\varepsilon_{zz}` is
  written to ``Total_Strain`` and the tangent is condensed, as in the kernel.
* **Fallback.** A point at which the reduced Newton method does not converge (rare, extreme
  states) is recomputed by the C++ kernel, with the inputs the kernel would have received.

The returned state depends only on the strain, the committed viscous strain, the time step and
the damage (no warm start).

Accuracy
--------

The reduced system is solved to round-off, and the stress agrees with the kernel to the
kernel's tolerance. Two differences are by design: inside the split r2 surfaces the multipliers
are exactly zero, where the kernel's smoothed potential may return tiny "regularized"
multipliers (about :math:`10^{-7}f_t/E`); and at strongly damaged points with large openings,
where the kernel accepts states with relative residuals of its large stress scale, the reduced
update is the more accurate (it satisfies the creep equation and, in plane stress,
:math:`\sigma_{zz}=0` together with the 3-D response to round-off).

.. automodule:: femcubes.solids.cohesive_viscoelastic
   :members:
   :show-inheritance:

References
----------

.. bibliography::
   :filter: docname in docnames
