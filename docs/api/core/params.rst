Parameters
==========

The :class:`femcubes.core.params_template.ParamsTemplate` class provides
the shared configuration defaults for FEMCubes simulations.

Overview
--------

Most examples define a small subclass that sets class attributes for:

* mesh geometry and ``MeshType``
* material data in ``Materials``
* configured models in ``Models`` and ``ModelNames``
* staggered solve structure in ``Solution_Steps`` and ``Field_Orders``
* time-integration settings such as ``start_time``, ``end_time``, and
  ``dt``

Gravity Direction
-----------------

``gravity`` is a signed scalar acceleration along the last spatial coordinate:
y in two dimensions and z in three dimensions. Solid body forces and the
pore-fluid weight correction use this same direction. Negative values act
toward decreasing y or z.

Mutable Defaults
----------------

``ParamsTemplate`` deliberately copies mutable configuration attributes
onto each instance.  This prevents cross-test or cross-run leakage for
fields such as:

* ``Global_Measures``
* ``line_search_lims``
* ``output_fields``
* ``Plot_Fields``
* ``Plot_Isosurfaces``
* ``Plot_RO_Fields``
* ``Plot_Measures``

Solver Settings
---------------

The settings of the solvers are described with the solvers: ``Step_Solvers``
(the nonlinear method of each staggered step), the tolerances
(``rel_tol``, ``abs_tol``, ``max_it``), the staggered passes, the early-abort
and stagnation settings and the l-BFGS settings in
:doc:`solvers/nonlinear`; ``TimeStepControl`` and the retry factors in
:doc:`solvers/time_stepping`; and ``Linear_Solver`` with the GMRES settings in
:doc:`solvers/linear`.

Runtime Profiling
-----------------

``PROFILE_TIMINGS`` defaults to ``False``.  When enabled, the solver
accumulates exclusive rank-local wall times for form evaluation,
``fem.form`` compilation/cache lookup, local element assembly, PETSc
assembly communication, boundary-condition application, model-specific
K/F additions, assembly bookkeeping, linear solving, mesh refinement,
plotting, and file output.  The public
``solver.get_runtime_timings()`` method returns a detached dictionary of
these cumulative values, while ``solver.get_runtime_timing_counts()``
returns how many regions contributed to each value.  This distinguishes a
slow operation from a frequently repeated one.  Assembly event counts
additionally separate ``matrix_residual_assembly`` from
``residual_only_assembly`` events.  Call
``solver.reset_runtime_timings()`` after an explicit JIT warm-up when only
production solve work should be profiled.  No MPI collectives are performed
while timing; applications can reduce or gather the values once after the
solve according to their reporting needs.

``RO_zscale`` may be set on shallow/reduced-order examples to vertically
exaggerate the displayed ``z`` and ``h`` geometry in reduced-order plots
without changing the plotted scalar values.  ``RO_sea_level`` may be set
to override the default sea level of ``0.0`` used for reduced-order
floating-ice display.

Elastic Helper
--------------

The convenience method :meth:`ParamsTemplate._SetElasticParams` derives
``Bulk``, ``Lame``, and ``Shear`` from a material's ``Youngs`` and
``Poisson`` entries.  It is mainly useful in examples and tests where
the user wants to define isotropic elasticity in the more familiar
Young's-modulus / Poisson-ratio form.

.. automodule:: femcubes.core.params_template
   :members:
   :show-inheritance:
