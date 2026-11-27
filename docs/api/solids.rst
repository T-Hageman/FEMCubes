``femcubes.solids`` -- Solid Mechanics Models
=============================================

The solid models describe the deformation and fracture of solids through the
small-strain momentum balance, with linear-elastic, viscoelastic (creep) or
purely viscous (Stokes) material behaviour. Fracture is described with the
phase-field method :cite:p:`Bourdin2000,Miehe2010a`, in two forms. In the
standard phase-field solids (``LinearElastic``, ``SplitLinearElastic``,
``ViscoElastic``, ``SplitViscoElastic`` and ``Stokes``), the phase field
degrades the stiffness of the material. In the cohesive solids, cracks open
through fracture eigenstrains that are limited by an explicit strength
surface, and the phase field degrades only the strength of the material
:cite:p:`Vicentini2026,Hageman2026b`. The package also contains the
phase-field model itself and two auxiliary models, a phase-field constraint
and a crack-depth measure.

Each model page states the fields the model defines or reads, the exact
terms it adds to the weak form, what it does outside the weak form, and an
example input. Common notation: :math:`\Omega` is the domain group named by
the model setting ``domain`` (the full mesh by default), :math:`\delta(\cdot)`
is the test function of a field, unknowns without subscript are at
:math:`t_{n+1}`, the subscript :math:`n` marks committed values and
:math:`\Delta t` is the time increment. On axisymmetric meshes ``dx``
carries the radius weight.

.. toctree::
   :maxdepth: 1

   solids/linear_elastic
   solids/split_linear_elastic
   solids/cohesive_linear_elastic
   solids/cohesive_viscoelastic
   solids/cohesive_viscoelastic_kernel
   solids/failure_surface_plot
   solids/phase_field
   solids/no_phase_field
   solids/viscoelastic
   solids/split_viscoelastic
   solids/stokes
   solids/crack_depth_measure

.. _solids-which-cohesive-model:

Which cohesive model
--------------------

The three cohesive solids share one formulation, with cohesive eigenstrains
limited by the strength surface and a phase field that acts on the failure
potential only, and the same failure types. They differ in how the local
problem is solved and in whether creep is included:

* :doc:`solids/cohesive_viscoelastic` (``Solid:Cohesive_ViscoElastic``) is the
  default. Its local problem, in which creep and the cohesive eigenstrain are
  resolved together :cite:p:`Hageman2026CMAME`, is solved in reduced
  (invariant) form, with the C++ kernel as fallback for points at which this
  does not converge.
* :doc:`solids/cohesive_viscoelastic_kernel`
  (``Solid:Cohesive_ViscoElastic_Kernel``) solves the same local problem with
  the C++ kernel of the paper that introduced the model; the results agree to
  the kernel's tolerance, at a higher cost.
* :doc:`solids/cohesive_linear_elastic` (``Solid:Cohesive_LinearElastic``) is
  the default model without creep, the cohesive formulation of
  :cite:t:`Hageman2026b`.

:func:`~femcubes.solids.failure_surface_plot.plot_FailureSurface` plots
the strength surface of a material.

.. _solids-split-consistency:

Consistent and inconsistent splits
----------------------------------

The standard phase-field solids come in two kinds, which behave differently
once the material is damaged:

.. list-table::
   :header-rows: 1

   * - Kind
     - Models
     - Degraded stress
     - Density of damaged material
   * - inconsistent split
     - :doc:`solids/linear_elastic`, :doc:`solids/viscoelastic`
     - the whole stress, :math:`g(\phi)\boldsymbol\sigma`; only the driving
       force is split
     - removed where :math:`\phi \geq 0.8` (gravity and inertia)
   * - consistent split
     - :doc:`solids/split_linear_elastic`, :doc:`solids/split_viscoelastic`
     - only the damageable part, :math:`g(\phi)\boldsymbol\sigma^+ + \boldsymbol\sigma^-`,
       with the same split as the driving force
     - full density at every damage level

The inconsistent split degrades the stress isotropically while splitting the
driving force, as in the hybrid formulation of :cite:t:`Ambati2015b`, which
keeps the momentum balance linear in the displacements. Damaged material then
has no stiffness left, also in compression, so it cannot carry its own
weight, and its density is therefore removed. Under gravity or other
compressive loads, damage then spreads from a crack into the surrounding
material. A consistent-split model keeps the compressive part of the stress
in damaged material, so that cracked material still carries its weight and
keeps its full density. Use the consistent split (e.g.
``CrackDrivingForce: "Spectral"``) for problems loaded by gravity or in
compression.

The cohesive solids (:doc:`solids/cohesive_viscoelastic`) are consistent by
construction: the phase field degrades only the failure potential, and the
density is not degraded. Gravity can therefore be included directly, without
the degraded body force and the decomposition of the elastic energy that
phase-field models of crevasses need when the phase field degrades the
elastic energy :cite:p:`Sun2021,Clayton2022,Clayton2025,Hageman2026`. Which
compressive stress states fully damaged material still carries follows from
the failure type: only hydrostatic compression for the two-multiplier
criterion ``r2_2L``, and every stress state without a tensile principal
stress for the opening-only criterion ``r2_2L_PSD``, which therefore suits
problems in which damaged material is loaded in compression
:cite:p:`Hageman2026CMAME`.

.. _solids-crack-driving-forces:

Crack driving forces
--------------------

The material entry ``CrackDrivingForce`` selects the quantity a solid
writes to the quadrature field ``ElasticEnergy``, which
:doc:`solids/phase_field` reads as its driving force :math:`\Psi`. All
options share one implementation (``elastic_split`` in
``cpp_kernels/shared/phasefield_splits.h``, converted to ``ElasticEnergy``
in ``_split_elastic_common.py``). The same split defines the damageable
stress :math:`\boldsymbol\sigma^+` and the retained stress
:math:`\boldsymbol\sigma^-` of the selectively degraded models
:doc:`solids/split_linear_elastic` and :doc:`solids/split_viscoelastic`,
whose mechanical stress is :math:`g(\phi)\boldsymbol\sigma^+ + \boldsymbol\sigma^-`.
``LinearElastic`` and ``ViscoElastic`` degrade the whole stress and use
the split only for the driving force (``Stokes`` has its own
``Stress-Based`` driver). Under multi-axial stress states, the split
determines the strength surface implied by the model and the stresses that
fractured material still transfers :cite:p:`DeLorenzis2021,Vicentini2024`.

The split acts on the elastic strain :math:`\mathbf e` (the total strain for
elastic models, :math:`\boldsymbol\varepsilon - \boldsymbol\varepsilon^v` for
viscoelastic ones) with

.. math::

   \theta = \mathrm{tr}\,\mathbf e,\quad
   \mathbf e_\mathrm{dev} = \mathbf e - \tfrac{\theta}{d}\mathbf I,\quad
   K_d = \lambda + \tfrac{2\mu}{d},\quad
   \psi_0 = \tfrac12\lambda\theta^2 + \mu\,\mathbf e:\mathbf e,\quad
   \boldsymbol\sigma_0 = \lambda\theta\mathbf I + 2\mu\mathbf e,

:math:`\langle x\rangle_+ = \max(x, 0)`, :math:`\langle x\rangle_- = \min(x, 0)`,
:math:`d=2` with ``true_2d`` and :math:`d=3` otherwise (3-D, and 2-D plane
strain with its out-of-plane components). For the energy options
:math:`\psi^- = \psi_0 - \psi^+`,
:math:`\boldsymbol\sigma^\pm = \partial\psi^\pm/\partial\mathbf e`, and
``ElasticEnergy`` :math:`=\psi^+`:

* ``FullEnergy``: :math:`\psi^+ = \psi_0`; no tension/compression
  distinction, as in the original formulation of :cite:t:`Bourdin2000`.
* ``VolDev``: :math:`\psi^+ = \tfrac12 K_d\langle\theta\rangle_+^2 + \mu\,\mathbf e_\mathrm{dev}:\mathbf e_\mathrm{dev}`;
  only volumetric compression is retained, the volumetric-deviatoric split
  of :cite:t:`Amor2009`.
* ``VolOnly``: :math:`\psi^+ = \tfrac12 K_d\langle\theta\rangle_+^2`;
  volumetric compression and all shear are retained.
* ``Spectral``: :math:`\psi^+ = \tfrac12\lambda\langle\theta\rangle_+^2 + \mu\,\mathbf e^+:\mathbf e^+`,
  with :math:`\mathbf e^+ = \sum_i\langle e_i\rangle_+\mathbf n_i\otimes\mathbf n_i`
  built from the principal strains :math:`e_i`, the spectral split of
  :cite:t:`Miehe2010a`.
* ``NoTension``: :math:`\psi^-(\mathbf e) = \min_{\mathbf q\succeq 0}\psi_0(\mathbf e-\mathbf q)`,
  the energy of a no-tension material :cite:p:`DelPiero1989,Freddi2010`.
  With sorted principal strains :math:`e_1\le\dots\le e_d` and :math:`m`
  compressive directions,
  :math:`\psi^- = \mu\sum_{i\le m}e_i^2 + \tfrac12\lambda_m\bigl(\sum_{i\le m}e_i\bigr)^2`
  and :math:`\lambda_m = 2\mu\lambda/(2\mu+(d-m)\lambda)`, where :math:`m`
  grows from 0 while :math:`2\mu e_{m+1} + \lambda_m\sum_{i\le m}e_i < 0`.
* ``Stress-Based`` (default): not an energy split. The solid converts the
  stress into an energy-like quantity, so that only the stress, and no
  plastic or creep deformation, drives the crack, with a threshold at the
  tensile strength following the failure criterion of :cite:t:`Miehe2015a`.
  With the principal stresses :math:`s_i` of :math:`\boldsymbol\sigma_0`,
  :math:`\boldsymbol\sigma^+ = \sum_i\langle s_i\rangle_+\mathbf n_i\otimes\mathbf n_i`
  and :math:`\boldsymbol\sigma^- = \boldsymbol\sigma_0 - \boldsymbol\sigma^+`
  (the tangent can be nonsymmetric). With
  :math:`S = \sum_i\langle s_i\rangle_+^2/f_t^2`, the driving force is

  .. math::

     \Psi = \frac{G_c}{4\ell}(S-1)\ \ \text{(AT2, HO-AT2)},\qquad
     \Psi = \frac{3G_c}{16\ell}\,S\ \ \text{(AT1, HO-AT1)},

  selected by ``PF_formulation``. ``LinearElastic`` and
  ``SplitLinearElastic`` clip the AT2 value at zero,
  :math:`\Psi = \frac{G_c}{4\ell}\max(S-1,0)`; ``ViscoElastic`` and
  ``SplitViscoElastic`` do not. Without a damage model, the split models
  store the total elastic energy :math:`\tfrac12\mathbf e:\boldsymbol\sigma`
  for this option instead.

The splits require finite :math:`\lambda\ge0` and :math:`\mu>0`. Stresses
and tangents are evaluated at fixed phase field; the tension/compression
switches are not smoothed.

References
----------

.. bibliography::
   :filter: docname in docnames
