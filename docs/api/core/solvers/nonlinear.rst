Nonlinear solvers
=================

Within one time increment, :class:`~femcubes.core.solvers.step_solver.StepSolver`
solves the nonlinear equations of all fields at the new time level. The fields
are grouped into staggered steps, which are solved one after another and, when
they are coupled, repeated over several passes. Each step is solved by one of
four nonlinear methods, chosen per step: Newton's method with a backtracking
line search, a Newton trust-region method, a Newton continuation method for
solves that stall, and limited-memory BFGS. All methods share the same
residual norm, convergence tests and failure handling, described first.

``StepSolver`` updates the fields in place; it neither advances the time nor
commits history variables, which is the task of
:class:`~femcubes.core.solvers.time_solver.TimeSolver` (see
:doc:`time_stepping`). It can also be used on its own for a single nonlinear
solve. The nonlinear methods are implemented as internal mixins of
``femcubes.core.solvers``, one per method (``_Newton`` in ``_newton.py``,
``_NewtonTrustRegion``, ``_NewtonContinuation``, ``_LBFGS`` and
``_ConstrainedStep``, the latter providing ``Solve_Constrained_Step``), on top
of the residual norms, convergence tests, update limiting and line-search
helpers of ``_StepSolverBase``; ``StepSolver`` combines them with the
staggered loop.

Staggered scheme
----------------

``Solution_Steps`` maps every field to the number of the step that solves it.
Fields of the same step are solved monolithically, while the fields of the
other steps are held at their current values. This is the alternating
minimization, or staggered scheme, that is common for phase-field fracture
:cite:p:`Bourdin2000,Miehe2010a`: the displacement is solved for a fixed
phase field, and the phase field for the resulting displacement.

A pass solves the steps once, in the order of their numbers. After every pass,
the residuals of all steps are assembled again at the final state of the pass
(the post-pass check), as solving a later step changes the residual of the
earlier ones. If every step then satisfies the convergence criterion below,
the staggered iteration ends; otherwise the next pass starts from the current
fields, up to ``n_passes`` passes (default 1). With ``n_passes_force = True``
all ``n_passes`` passes are performed even when an earlier pass has
converged. A problem with a single step always uses one pass. For example, the
ice-cliff example solves the displacement in step 0 and the phase field in
step 1, with up to ten passes:

.. code-block:: python

   class Params(ParamsTemplate):
       Solution_Steps = {"u": 0, "phasefield": 1}
       Step_Solvers = {0: "Newton", 1: "Newton"}
       n_passes = 10                  # stops early once all steps have converged
       max_it = 20                    # Newton iterations per step and pass
       line_search = True
       line_search_lims = [1.0e-1, 1.0]
       rel_tol = 1e-5
       abs_tol = 1e-1

Reaching ``n_passes`` with post-pass residuals above tolerance emits the
warning ``Reached ... staggered passes with residuals above tolerance`` but
retains the iterate, which ``TimeSolver`` then treats as a converged solve. A
time-step controller accepting the crack-length change does therefore not
establish coupled mechanics/phase-field convergence. If these residuals remain
above tolerance, the pass budget should be increased; the choice of the inner
nonlinear solver alone does not resolve an incomplete outer coupling.

Convergence and tolerances
--------------------------

All nonlinear methods measure convergence with the global RMS force residual
``||F||2 / sqrt(N)``, where ``N`` is the global number of degrees of freedom
of the step. This retains an L2-based error estimator while making the
absolute tolerance independent of the mesh size and the MPI partitioning. On
axisymmetric meshes, each residual row is first divided by the mean radius of
its basis function (see :doc:`../mesh`), so that ``abs_tol`` keeps the meaning
it has on plane meshes. A step has converged when either

* the RMS residual is below ``abs_tol`` (default ``1e-8``), or
* the RMS residual is below ``rel_tol`` (default ``1e-6``) times a reference
  residual, and also below ``rel_tol_max_residual`` (default: no limit).

The reference residual is the RMS residual after the first nonlinear update of
the step in the first pass, and it is kept for all later passes of the same
increment. Taking it after the first update, rather than before, means that a
large penalty-force load increment cannot by itself make an unconverged
interior residual pass ``rel_tol``. Keeping it over the passes means that a
later pass, which starts from a smaller residual, still has to reach the same
level. The converse case, a first update that overshoots (for example a
concentration driven negative inside an exponential rate law), makes that
reference large and lets the next residual pass ``rel_tol`` while it is still
large; ``rel_tol_max_residual`` accepts the relative criterion only while the
RMS residual is also below its value, and leaves ``abs_tol`` unaffected. The
constant normalization by ``sqrt(N)`` cancels from the relative ratio, so it
does not change the meaning of ``rel_tol``. The unscaled force vector is still
used in the l-BFGS curvature and line-search tests.

The convergence test follows every update, so each inner solve performs at
least one linear solve, also when its residual already satisfies the
tolerance. ``max_it`` (default 100) limits the iterations of one inner solve,
that is, of one step in one pass. A solve that reaches ``max_it`` without
converging raises a ``RuntimeError``, as does a solve whose relative residual
exceeds ``1e4`` after its first iteration (divergence), or whose assembled
force vector or tangent contains non-finite values. ``TimeSolver`` catches
these errors and repeats the increment with a smaller time step (see
:doc:`time_stepping`).

To find the fields that keep a step from converging, set
``DEBUG_FIELD_CONVERGENCE = True``: after every Newton iteration, the solver
then prints, for each field of the step, the RMS residual over its rows, the
largest change of the field by the last update, and the node where that change
occurs. A field whose update is negligible while its residual stays above
``abs_tol`` indicates a round-off floor; updates that stay large, at a few
nodes, indicate a cycle or a non-smooth term there.

Selecting the method of each step
---------------------------------

``Step_Solvers`` selects the nonlinear method independently for each
staggered step: ``"Newton"``, ``"NewtonTrustRegion"``,
``"NewtonContinuation"`` or ``"LBFGS"``. Steps that are not listed use Newton,
and unknown step numbers or method names are rejected:

.. code-block:: python

   Solution_Steps = {"u": 0, "phasefield": 1}
   Step_Solvers = {0: "NewtonTrustRegion", 1: "Newton"}

``line_search = False`` turns the line search off for every method: Newton and
l-BFGS then take the full step, and both Newton variants fall back to
full-step Newton. The linear systems of all methods are solved by the backend
selected with ``Linear_Solver`` (see :doc:`linear`).

Newton's method and its line search
-----------------------------------

The default method is Newton--Raphson: every iteration assembles the
consistent tangent ``K`` and the residual ``F``, solves ``K du = -F``, and
updates the fields. With ``line_search = True`` (the default), the update is
damped by a backtracking line search on the merit function
``0.5 ||F||2**2`` :cite:p:`Nocedal2006`. The search starts from the full
Newton step, ``alpha = 1``, and halves ``alpha`` until the merit decreases,
down to the lower limit ``line_search_lims[0]`` (default ``[0.01, 1.0]``); the
upper limit is used only by l-BFGS. The first decreasing trial is accepted.
If no trial decreases the merit, the trial with the smallest merit is kept, or
the smallest step when no trial improves on the starting point. Every trial
costs one residual assembly, so the lower limit also bounds the cost of an
iteration: the default of 0.01 allows up to seven trials, a limit of 0.1 up to
four.

Bounded fields and update caps
------------------------------

``Positive_Fields`` lists fields whose positive values must stay positive,
such as concentrations and surface coverages. The Newton and Newton
trust-region solvers take each decrease of such a value in log space, smoothly
capped (the trust region before its dogleg step; the Newton line search on
every trial step ``limit(alpha du)``, as the capped full step, scaled, need not
decrease the residual norm even when the Newton step does): with
``r = -du / u`` and ``L = -ln(Positive_Update_Factor)``,

.. math::

   u_\mathrm{new} = u \exp\left(-L \tanh(r / L)\right),

which is the Newton step for ``r << 1``, is never negative, and lowers a value
by at most the factor ``Positive_Update_Factor`` (default 1e-4, in (0, 1)) per
iteration. A trace species whose equilibrium value falls by decades between
time steps therefore needs only a few iterations, and there is no switch
between limited and unlimited updates (a hard limit can make Newton cycle,
with a trace concentration limited in one iteration and overshooting in the
next):

.. code-block:: python

   class Params(ParamsTemplate):
       Positive_Fields = ["Fe2+", "OH-"]
       Positive_Update_Factor = 1.0e-4
       rel_tol_max_residual = 1.0e-6

Increases, and entries that are zero or negative, are not changed. Near
convergence the relative changes are small, so quadratic convergence is kept.

``Fraction_Fields`` lists fields bounded to (0, 1), such as surface
coverages: decreases are limited as for ``Positive_Fields``, and increases the
same way with the distance ``1 - u`` to the upper bound, so a coverage far
from its equilibrium cannot overshoot 1 in one Newton step:

.. code-block:: python

   class Params(ParamsTemplate):
       Positive_Fields = ["Fe2+", "OH-"]
       Fraction_Fields = ["theta_Fe", "theta_FeOH"]

``Max_Field_Update`` maps fields to the largest change per iteration. When a
listed field would change by more than its cap ``c``, the whole Newton update is
scaled by ``tanh(m) / m``, with ``m`` the largest change relative to the cap
over all listed fields and ranks: a damped Newton step that keeps the
direction (capping the potential alone, while the concentrations take the full
step, mixes inconsistent updates), and the Newton step itself for small
changes. It suits potentials entering exponential (Butler--Volmer) rates: a
burst of current makes the linearized ohmic response overshoot by volts, while
a cap of 0.2 V changes such a rate by at most about :math:`e^4` per iteration:

.. code-block:: python

   class Params(ParamsTemplate):
       Max_Field_Update = {"E": 0.2, "E_m": 0.2}

A field may be listed in ``Positive_Fields`` or in ``Fraction_Fields``, not in
both, and a field with a ``Max_Field_Update`` in neither; unknown field names
are rejected. These limits act in the Newton and Newton trust-region solvers
and in ``Solve_Constrained_Step``, not in l-BFGS or Newton continuation.

Stagnating Newton solves
------------------------

Newton can also cycle between a few states without converging, e.g. where
the bounded updates of ``Positive_Fields`` differ from the Newton step at a
few nodes of a precipitation front. ``Stagnation_Iterations`` (default 0: off)
ends a Newton or Newton trust-region solve once its RMS residual has not
fallen below half of its smallest earlier value for that many iterations, so
the time step is cut without spending ``max_it`` iterations first. When such a
solve ends without converging (stagnated or at ``max_it``),
``Stagnation_Accept_Factor`` (default: off, at least 1) restores the iterate
with the smallest RMS residual and accepts it if that residual is below the
factor times ``abs_tol``:

.. code-block:: python

   class Params(ParamsTemplate):
       abs_tol = 1.0e-9
       Stagnation_Iterations = 8
       Stagnation_Accept_Factor = 10.0       # accept a best iterate below 1e-8

Newton trust region
-------------------

For a mechanics solve that stalls with heavily shortened Newton corrections,
select the optional residual-based dogleg method :cite:p:`Nocedal2006`::

   Step_Solvers = {0: "NewtonTrustRegion", 1: "Newton"}

It uses the exact Jacobian and the linear solver of Newton's method, with a fixed variable
scale obtained from the square root of the absolute initial tangent diagonal.
The Cauchy direction uses ``J.T*F`` for the merit function ``0.5*||F||2**2``;
it therefore supports nonsymmetric tangents such as the stress-spectral split
without assuming an elastic-energy potential. A trial lies on the path from
the Cauchy correction to the full Newton correction. The trust radius changes
according to the agreement between predicted and actual residual reduction.
Unlike backtracking along a single direction, this can change the direction
when a full Newton correction crosses many tensile/compressive branches.

The first radius admits the full Newton correction. Subsequent radii are
adapted automatically; ``line_search_lims`` does not bound dogleg corrections.
A failed set of 20 trust-region trials retains the smallest tried correction
for recovery instead of rejecting the time increment immediately. The outer
``max_it`` limit and the convergence checks of Newton's method apply as well. If no usable
Cauchy direction exists, ordinary Newton backtracking supplies the fallback.
Setting ``line_search=False`` uses ordinary full-step Newton.

This solver changes neither the constitutive law nor the physical time step.
It is not a guarantee of convergence, and small residual-minimizing corrections
can still stagnate. Its benefit is best assessed on the simulation at hand, by
comparing the number of retries and of residual evaluations.

Newton continuation
-------------------

For residual minimization that stagnates with the exact stress-spectral law,
select::

   Step_Solvers = {0: "NewtonContinuation", 1: "Newton"}
   max_it = 500

After 20 unconverged ordinary trust-region iterations, this method solves
auxiliary equilibria :math:`F(u)+\eta W(u-u_a)=0`, where :math:`W` is the
absolute initial tangent diagonal with a positive numerical floor and
:math:`u_a` is the last accepted auxiliary equilibrium. Damping starts at one.
The artificial term changes the solver path only: every convergence decision
for the physical problem uses the original residual :math:`F(u)`, with the
usual absolute and relative tolerances and reference norm. Constitutive
stresses, tangents, damage drivers and the physical time increment are unchanged.

Each auxiliary stage uses trust-region corrections. On auxiliary convergence,
the anchor advances; damping halves if the stage needed at most five iterations.
A stage retries from its anchor with ten times the damping after 35 iterations,
or after at least ten iterations when the residual exceeds 90 percent of the
best of the preceding five values. Auxiliary convergence uses the same tolerance
values, with its own first-update reference norm.

All stages share one ``max_it`` budget. Raising this budget can avoid expensive
time-step retries, but no auxiliary convergence can substitute for physical
residual convergence. Exhausting the budget triggers the usual time-step
retry. The smallest-trial recovery fallback is retained, and
``line_search=False`` selects ordinary full-step Newton. This method does not
guarantee convergence or resolve an exhausted outer staggered-pass budget.

Limited-memory BFGS
-------------------

The l-BFGS method :cite:p:`Liu1989,Nocedal2006` replaces most tangent
assemblies by residual-only assemblies. It is configured by:

.. code-block:: python

   Solution_Steps = {"u": 0, "phasefield": 1}
   Step_Solvers = {0: "LBFGS", 1: "Newton"}
   LBFGS_History = 10
   LBFGS_LineSearch_MaxIter = 20
   LBFGS_LineSearch_Tolerance = 0.9
   LBFGS_LineSearch_MaxResidualGrowth = 2.0
   LBFGS_StagnationIterations = 5
   LBFGS_RecoveryLineSearchTolerance = 0.1
   LBFGS_PowellDamping = 0.2

The values shown are the defaults. l-BFGS
assembles a tangent at the start of the staggered step, uses it as the
initial inverse-Hessian preconditioner, and normally performs
limited-memory updates, with at most ``LBFGS_History`` correction pairs, using
residual-only assembly. Once a correction
pair is available, the preconditioner is curvature-scaled by
``(s.y) / (y.K0^-1.y)`` before applying the two-loop recursion.  This
preserves the tangent's field scaling while adapting its magnitude to the
latest secant stiffness.  If a correction
pair fails the positive-curvature check, the base tangent is refreshed
at the accepted state while previously valid correction pairs remain
active.  The same refresh occurs after
``LBFGS_StagnationIterations`` consecutive iterations fail to improve the
best RMS force-vector L2 norm.  That refresh occurs at the current iterate:
it neither rolls the solution back nor clears the limited-memory history.
``LBFGS_RecoveryLineSearchTolerance`` tightens the directional curvature
search for the first direction from the refreshed tangent.  Among the
positive-curvature trials evaluated by that recovery search, the trial
with the smallest force-vector L2 norm is retained rather than
automatically accepting the trial closest to the directional stationary
point.  Subsequent directions return to the normal tolerance while the
new correction history is being built.

For a non-positive or small secant curvature, below ``LBFGS_PowellDamping``
times the curvature of the base tangent, Powell damping
:cite:p:`Powell1978` blends the measured
residual difference with the base-tangent action and retains a positive
correction pair.  This is particularly important while a fracture active
set is changing.  If the full tangent or its retained corrections produce
a non-descent direction, the solver first retries the refreshed base
tangent and then uses ``-abs(diag(K))^-1 F`` as a last resort.  The
absolute diagonal supplies a positive displacement/phase-field scale even
when a softening consistent tangent is indefinite.

With ``line_search`` enabled, l-BFGS uses a safeguarded
secant search for a point satisfying the strong-Wolfe curvature condition
``abs(F(x + alpha*p).p) <= tolerance * abs(F(x).p)``, rather than the
residual backtracking of Newton.  The default
``tolerance=0.9`` (``LBFGS_LineSearch_Tolerance``) is the usual curvature
scale :cite:p:`Nocedal2006`; substantially
smaller values unnecessarily force the search close to a directional
stationary point and can skip earlier trials with much smaller residuals.
The search evaluates at most ``LBFGS_LineSearch_MaxIter`` trials, between the
bounds ``line_search_lims``, and accepts a directionally converged trial only
when its force-vector L2 norm is at most ``LBFGS_LineSearch_MaxResidualGrowth``
times that at the start of the search.
If the upper search bound does not meet the directional tolerance,
l-BFGS accepts it only when it reduces the force-vector L2 norm; otherwise
it backtracks.

Early abort of unrecoverable steps
----------------------------------

By default a failing step is only handed back to the time-step controller
once an inner solve exhausts ``max_it``, its relative residual exceeds
``1e4``, or the assembled force vector becomes non-finite. When a step is far
too large for the current state, for example at the dynamic nucleation of a
crack in a structure that stores much more energy than the fracture can
dissipate, this can cost many expensive passes before ``TimeSolver`` reduces
``dt``. Three optional checks abort such a step earlier. All are disabled by
default, because a few diverging Newton iterations or slowly converging
staggered passes are normal in phase-field fracture and do not by themselves
indicate an unrecoverable step.

* ``staggered_divergence_passes`` (default 0) with
  ``staggered_divergence_factor`` (default 1.5): abort when the post-pass
  residual of any step grows by more than the factor for that many
  consecutive passes.
* ``staggered_stagnation_passes`` (default 0) with
  ``staggered_stagnation_factor`` (default 0.9): abort when no step reduces
  its post-pass residual to below the factor times its previous value for
  that many consecutive passes.
* ``line_search_nonfinite_abort`` (default 0): abort a Newton solve when the
  residual backtracking meets a non-finite trial residual and still fails to
  reduce the merit, for that many consecutive iterations. A non-finite trial
  alone is not counted; an isolated overshoot that the backtracking recovers
  from is accepted as usual. ``StepSolver.last_line_search_outcome`` exposes
  the accepted step length, the number of non-finite trials and whether the
  merit decreased for the most recent Newton line search.

Each abort raises ``RuntimeError`` from ``Solve_Step``, which ``TimeSolver``
treats like any other failed solve: the step is reset and ``dt`` reduced by
``reset_dt_factor`` and ``reset_dt_factor_repeated`` (see
:doc:`time_stepping`).

Constrained solves for initial states
-------------------------------------

``StepSolver.Solve_Constrained_Step(step, frozen_fields, combined_rows)``
solves one step with the listed fields held at their current values and,
optionally, the equations of some free fields replaced by combinations of
other rows at the same node. It serves to make initial states consistent. An
electrolyte potential has no time derivative and its own equation
(electroneutrality) does not contain it, so a new simulation starts it at an
arbitrary value; with the concentrations frozen, the charge-weighted sum of
the species balances (current conservation) gives the potential that belongs
to the initial concentrations, including diffusion potentials and ohmic drops:

.. code-block:: python

   solver = StepSolver(params, physics)
   rows = physics.models["Electrolyte"].Current_Conservation_Rows()   # {"E": {species: charge}}
   frozen = list(params.Species) + ["theta_Fe", ...]                  # everything except E (and E_m)
   solver.Solve_Constrained_Step(0, frozen, rows)

Frozen rows become identities; Dirichlet rows of a combined field keep their
condition, and Dirichlet source rows are left out of the sums
(``Physics.Get_Dirichlet_Rows``). The linear systems use the LU factorization of MUMPS
:cite:p:`Amestoy2001,Amestoy2019` whatever
``Linear_Solver`` is, and convergence uses ``abs_tol``, ``rel_tol`` and
``max_it``. Unlike the other solves, the relative criterion of this solve
refers to the residual before its first update. An update that makes the
residual non-finite (overflow of an exponential rate law) is halved, up to ten
times.

API reference
-------------

.. automodule:: femcubes.core.solvers.step_solver
   :members:
   :show-inheritance:

References
----------

.. bibliography::
   :filter: docname in docnames
