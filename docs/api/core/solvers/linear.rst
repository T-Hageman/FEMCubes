Linear solvers
==============

Every iteration of the nonlinear solvers of :doc:`nonlinear` solves a linear
system ``K du = -F`` with the tangent of the current staggered step, and the
l-BFGS method also uses these solves for the action of its initial inverse
Hessian. These systems are solved with PETSc :cite:p:`Balay1997`, through the
internal mixin :class:`femcubes.core.solvers._lin_solvers._LinSolvers`. The
parameter ``Linear_Solver`` selects the solver for all steps of a simulation,
as a :class:`~femcubes.core.solvers.solver_enums.LinearSolver` value (or its
name as a string):

.. code-block:: python

   from femcubes.core.solvers.solver_enums import LinearSolver

   class Params(ParamsTemplate):
       Linear_Solver = LinearSolver.LU    # direct solver (MUMPS), the default

The choice affects the cost of a solve and its robustness, but not the
nonlinear iterations: all options solve the same system, and an iterative
solve that converges gives the same Newton update as a direct one to within its
tolerance.

Available solvers
-----------------

``LU`` (default)
   A direct LU factorization with MUMPS :cite:p:`Amestoy2001,Amestoy2019`,
   used as a PETSc ``preonly`` KSP. The analysis is distributed, with a
   parallel ParMETIS ordering (MUMPS ``ICNTL(28) = 2``, ``ICNTL(29) = 2``);
   further settings are an automatic row and column scaling
   (``ICNTL(8) = 77``), a memory relaxation of 100 % (``ICNTL(14) = 100``),
   null-pivot detection (``ICNTL(24) = 1``) and a relative pivot threshold of
   0.01 (``CNTL(1)``). The system is factorized again whenever the tangent
   has been reassembled, while the ordering is reused for as long as the
   cached solver of the step remains valid.
``GMRES``
   Restarted GMRES :cite:p:`Saad1986` with an additive Schwarz (ASM)
   preconditioner :cite:p:`Toselli2005`, in the restricted form that is the
   PETSc default :cite:p:`Cai1999`. Each rank's subdomain, extended by
   ``ASM_Overlap`` layers of overlap, is solved with a sequential MUMPS
   factorization (with the local AMF ordering instead of ParMETIS, and
   otherwise the settings of ``LU``).
``GAMG``
   GMRES with PETSc's algebraic multigrid preconditioner GAMG, in its
   smoothed-aggregation form :cite:p:`Vanek1996` (``AGG`` with one smoothing
   step of the prolongator). For a step that solves only the displacement
   ``u``, in two or three dimensions, the rigid-body modes of ``u`` are given
   to GAMG as its near-null space, and they are rebuilt after every remesh.
   Steps with a mixed space (several fields solved together) are passed on
   to ``FIELDSPLIT`` automatically, as a single multigrid hierarchy is not
   suited to a mixed operator.
``FIELDSPLIT``
   GMRES with a PETSc field-split block preconditioner :cite:p:`Brown2012`,
   with one block per field of the step. When a field is a Lagrange
   multiplier (a field name starting with ``LMultiplier``), whose diagonal
   block is zero, the split is a full Schur-complement factorization with the
   ``selfp`` approximation of the Schur complement, preconditioned by Jacobi;
   otherwise the blocks are combined additively. Every other block is
   preconditioned by one GAMG cycle. A step with a single field uses plain
   ``GAMG``.

The three iterative solvers share the settings ``GMRES_Restart`` (default
50), ``GMRES_Max_Iter`` (default 200) and ``GMRES_RelTol`` (default
``1e-6``, the relative tolerance of the Krylov solve); ``ASM_Overlap``
(default 1) applies to ``GMRES`` only:

.. code-block:: python

   class Params(ParamsTemplate):
       Linear_Solver = LinearSolver.GMRES
       GMRES_Restart = 50
       GMRES_Max_Iter = 200
       GMRES_RelTol = 1e-6
       ASM_Overlap = 1

Solver workspaces and failures
------------------------------

Each staggered step keeps its own PETSc solver, which is configured at its
first solve and reused by all later iterations and time increments, so that
the solver set-up, the MUMPS settings and the ordering are not repeated. The
solver is rebuilt when the size, the parallel layout or the matrix type of
the step changes, and all solvers are destroyed after a remesh.

Before every solve, the tangent and the residual are checked for non-finite
values; either raises a ``RuntimeError``, which ``TimeSolver`` treats as a
failed solve and answers with a smaller time step (see :doc:`time_stepping`).
An iterative solve that does not converge, by contrast, only prints a warning
with its PETSc reason, iteration count and residual: the nonlinear iteration
continues with the inexact update, and its own convergence checks then decide
whether the step succeeds.

Choosing a solver
-----------------

For problems of the size of the examples, we use the direct solver, which is
the default. The strong-scaling benchmark of the
cohesive plate (:doc:`../../../examples/benchmarks/scaling_cohesive_elastic`)
shows both its strength and its limit. Factorizing the full system in every
Newton iteration remains affordable up to the few hundred thousand degrees of
freedom of that problem, but the parallel efficiency of MUMPS is limited by
the dense separator blocks at the top of the nested-dissection elimination
tree :cite:p:`George1973`, which are shared by all ranks: the problem with
about 194 000 degrees of freedom is solved most efficiently on four to eight
ranks, and one with about 400 000 degrees of freedom still gains up to 24
ranks. The iterative solvers did not improve on this for the fractured plate.
Algebraic multigrid (``GAMG``) does not converge once the cracks have formed,
and GMRES with additive Schwarz preconditioning (``GMRES``) converges but
takes about 2.5 times longer than the direct solver on the same ranks.

The iterative solvers are therefore intended for larger problems, in which the
memory and the factorization time of the direct solver become prohibitive.
``GAMG`` suits a single elliptic field, such as an elasticity step without
fracture or a pressure step, whose operator multigrid can coarsen;
``FIELDSPLIT`` applies the same multigrid to each field of a monolithic
multi-field step, such as a coupled Biot step; and ``GMRES`` with ASM, whose
subdomain solves are direct, is the more robust but slower option when the
operator is not elliptic, for example once a material softens. Having no
coarse level, ASM needs more iterations as the number of subdomains grows
:cite:p:`Toselli2005`, so that it is less suited to large elasticity problems,
in which the coupling is long-range. As the
nonlinear solver continues after a linear solve that has not converged, the
number of Newton iterations and the solver warnings in the log show whether an
iterative solver is adequate for a problem.

Threads per rank
----------------

FEMCubes parallelizes with MPI, and each rank should run a single thread.
Importing the solvers limits the BLAS and OpenMP thread pools of the process
to one thread (``threadpoolctl``), and ``StepSolver`` sets
``OMP_NUM_THREADS=1`` in its environment. As a thread library reads its
settings when it is first loaded, the variables are best also set before the
run starts, for example with ``source scripts/limit_omp.sh`` (which sets
``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and ``MKL_NUM_THREADS`` to one).
When every core already runs a rank, additional threads only compete for the
same cores and memory bandwidth. The benchmark above therefore limits OpenMP,
the BLAS library and numba (``NUMBA_NUM_THREADS``) to one thread per rank.

API reference
-------------

.. automodule:: femcubes.core.solvers.solver_enums
   :members:
   :show-inheritance:

.. automodule:: femcubes.core.solvers._lin_solvers
   :members:
   :show-inheritance:

References
----------

.. bibliography::
   :filter: docname in docnames
