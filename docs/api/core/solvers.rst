Solvers
=======

The solver layer turns the assembled physics into a converged solution at
every time level. It consists of three nested parts. On the outside,
:class:`~femcubes.core.solvers.time_solver.TimeSolver` advances the time from
``start_time`` to ``end_time``, chooses the time step of every increment from
the ``TimeStepControl`` policy, and repeats an increment with a smaller time
step when its solve fails or when its change of a monitored measure is too
large. Within one increment,
:class:`~femcubes.core.solvers.step_solver.StepSolver` solves the fields in the
staggered steps of ``Solution_Steps``, one after another, and repeats these
passes until all steps are converged together or ``n_passes`` is reached. Each
step in turn is solved by the nonlinear method chosen for it in
``Step_Solvers`` (Newton with a line search by default), and every iteration
of that method solves a linear system with the backend chosen in
``Linear_Solver``.

Failures travel up through these levels: a nonlinear solve that does not
converge, or a residual or tangent with non-finite values, raises an error,
and ``TimeSolver`` answers it by resetting the increment and reducing the time
step. A typical transient simulation only needs the outermost level:

.. code-block:: python

   mesh = Mesh(params)
   physics = Physics(mesh, params)
   physics.Initialize_Fields()
   solver = TimeSolver(params, physics)
   solver.run()

The pages below describe the three parts and the input parameters that
control them.

.. toctree::
   :maxdepth: 1

   solvers/nonlinear
   solvers/time_stepping
   solvers/linear
