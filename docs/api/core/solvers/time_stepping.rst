Time stepping
=============

:class:`~femcubes.core.solvers.time_solver.TimeSolver` owns the simulation
clock. It advances the time from ``start_time`` to ``end_time`` in increments
``dt``, solves every increment with the staggered nonlinear solver of
:doc:`nonlinear` (``TimeSolver`` inherits from ``StepSolver``), and commits the
converged state. The time step can be kept constant, follow a user table, or
adapt to a monitored global measure such as the crack growth per step. In all
cases, an increment whose solve fails is repeated with a smaller time step,
so that the same input can pass both slow loading and sudden events such as
crack nucleation.

.. code-block:: python

   mesh = Mesh(params)
   physics = Physics(mesh, params)
   physics.Initialize_Fields()
   solver = TimeSolver(params, physics)
   solver.run()

The time loop
-------------

Before the loop, ``run()`` assembles every staggered step once, so that the
compilation of the forms is not part of the first increment
(``JIT_WARMUP``, default ``True``). Every increment then proceeds as follows:

1. ``Update_TimeIncrements()`` sets ``dt`` for the increment from the policy
   in ``TimeStepControl``, and shortens it where needed to land on ``t = 0``
   or on ``end_time``.
2. ``Solve_Step()`` solves all staggered steps at ``t + dt``. If it raises an
   error on any rank, the models are returned to their last committed state
   (``Physics.Reset_Step``), ``dt`` is reduced, and the increment is solved
   again (see `Failed solves`_ below).
3. The global measures are updated, and ``Check_Step_Validity()`` decides
   whether the result is acceptable for the policy. A rejected increment is
   reset and solved again with the ``dt`` set by the check.
4. If mesh refinement is enabled and due, the refinement indicator is
   evaluated; a refined mesh leads to a new solve of the same increment on
   the refined mesh (see `Mesh refinement`_ below).
5. The state is committed, ``t`` advances by ``dt``, and the fields, measures
   and output files are written.

The loop runs while ``t < end_time``. A failure on any rank counts as a
failure on all ranks, and the accept, retry and remesh decisions are checked
to agree on all ranks, so that they always follow the same path.

Time-step policies
------------------

``TimeStepControl`` is a dictionary whose ``"Type"`` selects the policy;
without it, the policy is ``"None"``. The keys and their defaults are:

``"Type"``
   ``"None"``, ``"StepWise"`` or ``"Measure"``.
``"time_list"``, ``"dt_list"``
   a piecewise-constant table of the largest time step: ``dt_list[i]``
   applies from ``time_list[i]`` up to ``time_list[i + 1]``, the last
   entry from the last time on, and the first entry before ``time_list[0]``.
   Required for ``"StepWise"``; optional for ``"Measure"``.
``"dt_grow_factor"`` (default 1.2)
   the factor by which ``dt`` grows per increment towards the table value
   (``"StepWise"`` and ``"Measure"``).
``"dt_max"`` (default 600)
   the largest time step of ``"Measure"`` without ``time_list``; it is not
   used when a table is given.
``"Measure"``, ``"Target_Change"``
   the name of the global measure (an entry of ``params.Global_Measures``)
   and its largest accepted value per increment (``"Measure"``).
``"dt_cut_factor"`` (default 2)
   the factor by which ``"Measure"`` divides ``dt`` when an increment exceeds
   ``Target_Change``; it must be larger than one.
``"dt_min"`` (default ``1e-8``)
   the smallest time step, for all policies (see `Failed solves`_).

Constant time step (``"None"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The policy ``"None"`` keeps ``dt`` as it is. The input ``dt`` is used for every
increment, apart from the last one, which is shortened to end at
``end_time``. A reduction after a failed solve is, however, kept for the rest
of the run, as this policy contains no growth. The shallow-ice example uses it,
by simply not defining ``TimeStepControl``:

.. code-block:: python

   class Params(ParamsTemplate):
       start_time = -50 * YEAR
       end_time = 50 * YEAR
       dt = 0.1 * YEAR             # without TimeStepControl, dt changes only after a failed step

Tabulated time step (``"StepWise"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The policy ``"StepWise"`` takes the largest time step from the table and lets
``dt`` grow towards it: every increment uses
``min(dt_grow_factor * dt, dt_list[i])``, with ``dt`` the time step of the
previous increment and ``i`` the table interval that contains the current
time. A time step reduced after a failed solve therefore recovers
gradually, while a smaller table value at the start of a new interval applies
at once. As the growth is also applied to the first increment, the run starts
with ``dt_grow_factor`` times the input ``dt`` when this is below the table
value. Before the first entry of ``time_list``, the first entry of
``dt_list`` applies. The corrosion
pit grows its 300 s step by 5 % per increment up to one hour:

.. code-block:: python

   class Params(ParamsTemplate):
       start_time = 0.0
       dt = 300.0
       TimeStepControl = {"Type": "StepWise",
                          "time_list": [0.0],
                          "dt_list": [3600.0],
                          "dt_grow_factor": 1.05
       }

This policy does not reject converged increments.

Measure-controlled time step (``"Measure"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The policy ``"Measure"`` adapts ``dt`` to the change of a global measure per
increment, typically ``Crack_Length_Change``, the change of the crack length
written by the phase-field model (``Solid:PhaseField``). The time step grows
as for ``"StepWise"``, towards ``dt_list[i]`` if a table is given and towards
``dt_max`` otherwise. After every solve, ``Check_Step_Validity`` compares the
measure with ``Target_Change``: an increment whose measure exceeds it is
rejected, reset, and solved again with ``dt`` divided by ``dt_cut_factor``,
but not below ``dt_min``. An increment whose time step is already within 5 %
of ``dt_min`` is always accepted. Every increment is checked, from the first
on, and a measure that none of the models writes is rejected with an error.

In this way, the crack advances by at most about ``Target_Change`` per
increment, and the time step follows the pace of the fracture: long steps
while the cracks are stationary, and steps that are cut, possibly by orders of
magnitude, when they grow. A cut factor larger than two shortens the search
for a short enough step when the crack growth jumps; the ice-shelf front uses
a factor of ten:

.. code-block:: python

   class Params(ParamsTemplate):
       start_time = 0.0
       dt = 3 * 3600.0
       TimeStepControl = {"Type": "Measure",
                          "time_list": [0.0],
                          "dt_list": [3 * 3600.0],
                          "dt_min": 1.0e-3,
                          "dt_cut_factor": 10.0,
                          "Measure": "Crack_Length_Change",
                          "Target_Change": 2.0 * ell
       }

At an increment that starts exactly at ``t = 0``, ``"Measure"`` sets ``dt`` to
the table value of the entry ``time_list[i] == 0`` if there is one, instead of
growing it. With ``start_time = 0`` the run therefore starts with that value,
and a run with an initialization period at negative times starts the main
period with it, rather than with a step grown from the short step that ended
the initialization. The ice-cliff example uses hourly steps for a 100-hour
creep spin-up, steps of up to 1 s for the first 10 s after the water level
starts to fall, and up to 30 s afterwards:

.. code-block:: python

   class Params(ParamsTemplate):
       start_time = -100 * 3600.0
       end_time = lowering_time
       dt = 3600.0
       TimeStepControl = {"Type": "Measure",
                          "time_list": [-1e99, 0.0, 10.0],
                          "dt_list": [3600.0, 1.0, 30.0],
                          "dt_min": 1.0e-6,
                          "Measure": "Crack_Length_Change",
                          "Target_Change": 2.0 * ell
       }

Landing on t = 0 and on the end time
------------------------------------

Simulations with an initialization period, such as a spin-up under gravity,
start at a negative time and apply the actual loading from ``t = 0``. For the
policies ``"StepWise"`` and ``"Measure"``, an increment from ``t < 0`` that
would pass ``t = 0`` is shortened to end at ``t = 0`` exactly, so that the
loading starts at its intended time. With ``"StepWise"``, and with
``"Measure"`` when its table has no entry at ``t = 0``, the time step then
grows again from this shortened step at ``dt_grow_factor`` per increment.

For every policy, the increment that would pass ``end_time`` is shortened to
end there, so the final state is that at ``end_time``; an increment that
would stop short of it by less than ``1e-9`` of its length is extended to it.
After the last increment, a time within this round-off of ``end_time`` is set
to ``end_time``, so that no extra increment of round-off length follows.

Failed solves
-------------

Any error raised by ``Solve_Step`` counts as a failed solve: a nonlinear solve
that reaches ``max_it`` or diverges, a non-finite residual or tangent, and
the optional early aborts of :doc:`nonlinear`. ``TimeSolver`` then resets the
models to their committed state and solves the increment again with a smaller
time step. The reduction uses ``reset_dt_factor`` (default 4) at the first
failure of an increment and ``reset_dt_factor_repeated`` (default equal to
``reset_dt_factor``) at every further consecutive failure of the same
increment, so that a run can be configured to search for a stable increment
in decades once the first reduction has not been enough. Both factors must be
larger than one:

.. code-block:: python

   class Params(ParamsTemplate):
       reset_dt_factor = 4.0              # first failure of an increment: dt / 4
       reset_dt_factor_repeated = 10.0    # every further failure: dt / 10, down to dt_min

The reduced time step is not smaller than ``dt_min``. A solve that fails
while ``dt`` is already at ``dt_min`` aborts the simulation on all ranks
(``MPI.Abort``), as no smaller step is available. After a successful
increment, the policy determines how the time step recovers: gradually for
``"StepWise"`` and ``"Measure"``, not at all for ``"None"``.

Mesh refinement
---------------

With adaptive mesh refinement enabled (``MeshRefinement``, see
:doc:`../mesh`), the refinement indicator is evaluated after an increment has
been solved and accepted by ``Check_Step_Validity``, from
``start_step`` on and every ``check_interval`` increments. If cells are
marked, the refined mesh is installed, the models and the cached linear
solvers are rebuilt for it, and the same increment is solved again at the
unchanged ``t`` and ``dt``, starting from the fields transferred to the new
mesh. This new solve again passes through the failure handling and the
validity check, and may refine the mesh once more, up to
``max_levels_per_step`` refinements per increment and ``max_events`` in
total. Only a solve on the final mesh of the increment is committed. As the
refinement never coarsens, a later time-step reduction keeps the refined mesh.

API reference
-------------

.. automodule:: femcubes.core.solvers.time_solver
   :members:
   :show-inheritance:
