"""Internal PETSc linear-solver helpers used by the nonlinear solvers."""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

# Limit the BLAS/OpenMP thread pools already loaded to one thread per MPI rank. Libraries that read their
# thread count when first loaded (OpenMP of MUMPS) need OMP_NUM_THREADS=1 set before the run starts.
from threadpoolctl import threadpool_limits
threadpool_limits(limits=1)

from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

from femcubes.core.utils.mpi_utils import mprint, rank
from femcubes.core.utils.runtime_profiler import RuntimeProfiler
from femcubes.core.solvers.solver_enums import LinearSolver


@dataclass
class SolverWorkspace:
    """Per-step PETSc solver workspace cached across nonlinear iterations."""

    signature: tuple[Any, ...]
    ksp: PETSc.KSP
    du: PETSc.Vec
    rhs: PETSc.Vec
    mumps_configured: bool = False
    asm_configured: bool = False
    gamg_configured: bool = False
    fieldsplit_configured: bool = False
    fieldsplit_uses_schur: bool = False
    _fieldsplit_is: Any = None  # List[PETSc.IS] kept alive while PC is active
    near_nullspace: Any = None  # PETSc.NullSpace of the rigid-body modes, or None
    near_nullspace_epoch: Any = None  # mesh epoch for which near_nullspace was built


class _LinSolvers:
    """Internal mix-in providing PETSc linear-solver management.

    Creates, configures, and destroys KSP solvers for each staggered
    step. Supports direct LU factorisation via MUMPS and iterative GMRES with
    ASM, GAMG, or field-aware GAMG/Schur preconditioning.

    Public API
    ----------
    This mix-in does not define user-facing solver methods.  It exists
    to support ``StepSolver`` and ``TimeSolver`` through underscored
    helper methods.

    Notes
    -----
    The mix-in is responsible for:

    * caching one PETSc ``KSP`` workspace per staggered step when sizes match
    * selecting LU/MUMPS, GMRES/ASM, GMRES/GAMG or GMRES/field split from ``params.Linear_Solver``
      (unknown choices are rejected); the GMRES settings are ``GMRES_Restart`` (default 50),
      ``GMRES_Max_Iter`` (200), ``GMRES_RelTol`` (1e-6) and ``ASM_Overlap`` (1)
    * validating matrix/vector norms before solve
    * rebuilding cached workspaces automatically when the operator layout changes

    """
    def __init__(self, params: Any, physics: Any) -> None:
        self.params = params
        self.physics = physics
        self._runtime_profiler = getattr(physics, "_runtime_profiler", None)
        if self._runtime_profiler is None:
            self._runtime_profiler = RuntimeProfiler(
                enabled=getattr(params, "PROFILE_TIMINGS", False)
            )

        # ------------------------------------------------------------------
        # Generic Solver State
        # ------------------------------------------------------------------
        self._solver_cache: dict[int, SolverWorkspace] = {}
        self.linear_solver = self._normalize_solver_choice(
            getattr(params, "Linear_Solver", LinearSolver.LU)
        )
        self.DEBUG_PRINT_TIMING = getattr(params, "DEBUG_PRINT_TIMING", False)

        # ------------------------------------------------------------------
        # GMRES / ASM Settings
        # ------------------------------------------------------------------
        self.gmres_restart  = getattr(params, "GMRES_Restart",  50)
        self.gmres_max_it   = getattr(params, "GMRES_Max_Iter", 200)
        self.gmres_rtol     = getattr(params, "GMRES_RelTol",   1e-6)
        self.asm_overlap    = getattr(params, "ASM_Overlap",    1)

    # ------------------------------------------------------------------
    # Internal Helpers: Generic Solver Management
    # ------------------------------------------------------------------

    def _normalize_solver_choice(self, choice):
        """PRIVATE: Normalize enum or string solver choices to upper-case text; unknown choices are rejected."""
        if isinstance(choice, LinearSolver):
            return choice.value
        value = str(choice).upper()
        known = [solver.value for solver in LinearSolver]
        if value not in known:
            raise ValueError(f"Unknown Linear_Solver {choice!r}; choose from {known}.")
        return value

    def _initialize_solver(self, F_vec, step):
        """PRIVATE: Optional hook for per-step setup before Newton iterations.

        The current implementation does not need any persistent
        preallocation or setup at this stage, but the hook is kept so
        ``StepSolver`` can call into a stable interface.
        """
        pass

    def _positive_tangent_diagonal(self, matrix, prototype):
        """PRIVATE: Return a positive diagonal scale for ordinary or ghosted vectors."""
        diagonal = prototype.duplicate()
        matrix.getDiagonal(diagonal)
        floor = np.finfo(float).eps * max(diagonal.norm(PETSc.NormType.NORM_INFINITY), 1.0)
        owned = diagonal.getArray()
        owned[:] = np.maximum(np.abs(owned), floor)
        return diagonal

    def _cleanup_solver(self):
        """PRIVATE: Destroy all cached PETSc solver objects and work vectors."""
        for workspace in self._solver_cache.values():
            self._destroy_solver_workspace(workspace)
        self._solver_cache.clear()

    def _solve_linear_system(self, K_mat, F_vec, step):
        """PRIVATE: Solve ``K * du = -F`` for one Newton iteration.

        The routine validates the assembled matrix/vector, chooses the
        configured backend, and returns the updated solver workspace.
        """
        with self._runtime_profiler.measure("linear_solve"):
            return self._solve_linear_system_profiled(K_mat, F_vec, step)

    def _solve_linear_system_profiled(self, K_mat, F_vec, step):
        """PRIVATE: Execute one linear solve inside the profiling boundary."""
        t_start = time.time()

        workspace = self._get_or_create_solver_workspace(step, K_mat, F_vec)
        if hasattr(workspace.du, "set"):
            workspace.du.set(0.0)

        K_norm = K_mat.norm()
        F_norm = F_vec.norm()

        if not np.isfinite(F_norm):
            mprint(f"\t\t\tERROR: Force vector has NaN/Inf values! Vector norm = {F_norm}")
            raise RuntimeError("Aborting due to invalid force vector.")

        if not np.isfinite(K_norm):
            mprint(f"\t\t\tERROR: Matrix has NaN/Inf values! Matrix norm = {K_norm}")
            raise RuntimeError("Aborting due to invalid matrix.")

        if self.linear_solver == LinearSolver.GMRES.value:
            self._solve_asm(workspace, K_mat, F_vec)
        elif self.linear_solver == LinearSolver.GAMG.value:
            # A single GAMG hierarchy is appropriate for one elliptic field,
            # but not for a mixed saddle-point operator.  Route mixed steps
            # through field-aware block preconditioning automatically while
            # retaining GAMG on the elliptic diagonal blocks.
            if self._step_has_mixed_space(step):
                self._solve_fieldsplit(workspace, K_mat, F_vec, step)
            else:
                self._solve_gamg(workspace, K_mat, F_vec, step)
        elif self.linear_solver == LinearSolver.FIELDSPLIT.value:
            self._solve_fieldsplit(workspace, K_mat, F_vec, step)
        else:
            self._solve_mumps(workspace, K_mat, F_vec)

        converged_reason = workspace.ksp.getConvergedReason()
        if converged_reason < 0:
            mprint(f"\t\t\t\tWarning: Solver did not converge (reason: {converged_reason})")
            mprint(
                f"\t\t\t\tIterations: {workspace.ksp.getIterationNumber()}, "
                f"Final residual: {workspace.ksp.getResidualNorm():.6e}"
            )

        t_end = time.time()
        if rank == 0 and self.DEBUG_PRINT_TIMING:
            mprint(f"\t\t\tSolver step duration: {t_end - t_start:.4f} s")

        return workspace

    def _prepare_rhs(self, workspace: SolverWorkspace, F_vec):
        """PRIVATE: Build the negated right-hand side vector for PETSc solves."""
        F_vec.copy(result=workspace.rhs)
        workspace.rhs.scale(-1.0)

    def _destroy_solver_workspace(self, workspace: SolverWorkspace):
        """PRIVATE: Destroy one cached solver workspace."""
        if workspace.du is not None:
            workspace.du.destroy()
            workspace.du = None
        if workspace.rhs is not None:
            workspace.rhs.destroy()
            workspace.rhs = None
        if workspace.ksp is not None:
            workspace.ksp.destroy()
            workspace.ksp = None
        if workspace._fieldsplit_is is not None:
            for is_obj in workspace._fieldsplit_is:
                if is_obj is not None:
                    is_obj.destroy()
            workspace._fieldsplit_is = None
        if getattr(workspace, "near_nullspace", None) is not None:
            workspace.near_nullspace.destroy()
            workspace.near_nullspace = None

    def _build_solver_signature(self, K_mat, F_vec):
        """PRIVATE: Build a signature describing whether a workspace is reusable."""
        matrix_size = tuple(K_mat.getSize()) if hasattr(K_mat, "getSize") else None
        vector_size = F_vec.getSize() if hasattr(F_vec, "getSize") else None
        vector_local_size = (
            F_vec.getLocalSize() if hasattr(F_vec, "getLocalSize") else None
        )
        matrix_type = K_mat.getType() if hasattr(K_mat, "getType") else None
        return (
            self.linear_solver,
            matrix_size,
            vector_size,
            vector_local_size,
            matrix_type,
        )

    def _create_solver_workspace(self, K_mat, F_vec) -> SolverWorkspace:
        """PRIVATE: Create a fresh solver workspace for one staggered step."""
        return SolverWorkspace(
            signature=self._build_solver_signature(K_mat, F_vec),
            ksp=PETSc.KSP().create(F_vec.getComm()),
            du=F_vec.copy(),
            rhs=F_vec.copy(),
        )

    def _get_or_create_solver_workspace(self, step, K_mat, F_vec):
        """PRIVATE: Reuse or rebuild the cached solver workspace for ``step``."""
        signature = self._build_solver_signature(K_mat, F_vec)
        workspace = self._solver_cache.get(step)
        if workspace is not None and workspace.signature != signature:
            self._destroy_solver_workspace(workspace)
            workspace = None

        if workspace is None:
            workspace = self._create_solver_workspace(K_mat, F_vec)
            self._solver_cache[step] = workspace

        return workspace

    # ------------------------------------------------------------------
    # Internal Helpers: LU / MUMPS
    # ------------------------------------------------------------------

    def _configure_mumps_factor(self, pc, *, parallel_analysis: bool = True):
        """PRIVATE: Apply the standard LU / MUMPS factorization settings.

        Global factorizations use MUMPS' distributed analysis with ParMETIS.
        ASM subdomain factorizations are sequential, so they retain the local
        AMF ordering instead.  Both paths reuse their first ordering while the
        cached solver workspace remains valid.
        """
        pc.setFactorOrdering(reuse=True)
        pc.setFactorSetUpSolverType()
        factor = pc.getFactorMatrix()
        factor.setMumpsIcntl(4, 1)    # Print level (1 = errors only)
        if parallel_analysis:
            factor.setMumpsIcntl(28, 2)  # Parallel analysis
            factor.setMumpsIcntl(29, 2)  # ParMETIS parallel ordering
        else:
            factor.setMumpsIcntl(7, 2)   # AMF ordering for sequential subdomains
        factor.setMumpsIcntl(8, 77)   # Scaling: automatic choice (row/col equilibration)
        factor.setMumpsIcntl(14, 100) # Memory relaxation (100% = more memory, less compression)
        factor.setMumpsIcntl(24, 1)   # Null pivot detection
        factor.setMumpsCntl(1, 0.01)  # Relative pivot threshold (larger = more stable)

    def _solve_mumps(self, workspace: SolverWorkspace, K_mat, F_vec):
        """PRIVATE: Direct solve using LU factorization with MUMPS."""
        if not workspace.mumps_configured:
            # One-time setup: solver type, PC type, MUMPS options.
            workspace.ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = workspace.ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType("mumps")
            workspace.ksp.setOperators(K_mat)
            self._configure_mumps_factor(pc)
            workspace.mumps_configured = True
        else:
            # Subsequent iterations: only update the operator so MUMPS
            # re-factorizes numerically while retaining its ICNTL settings
            # and, for an unchanged sparsity pattern, its symbolic analysis.
            workspace.ksp.setOperators(K_mat)

        self._prepare_rhs(workspace, F_vec)
        workspace.ksp.solve(workspace.rhs, workspace.du)

    # ------------------------------------------------------------------
    # Internal Helpers: GMRES / ASM
    # ------------------------------------------------------------------

    def _solve_asm(self, workspace: SolverWorkspace, K_mat, F_vec):
        """PRIVATE: GMRES solve with ASM/MUMPS subdomain preconditioning.

        Each ASM subdomain solve is configured as a local direct MUMPS
        factorization.  The global Krylov method is PETSc GMRES with
        restart, tolerance, and overlap configured once and then reused
        across Newton iterations within the same time step.

        Notes
        -----
        Calling ``ksp.setType``, ``pc.setType``, or ``pc.setASMOverlap``
        on an already-configured workspace can reset the PC and destroy
        the subdomain KSP configuration.  All one-time setup is therefore
        kept inside the ``asm_configured`` guard; subsequent calls only
        update the operator via ``setOperators``.
        """
        if not workspace.asm_configured:
            # One-time setup: KSP type, GMRES parameters, PC type, overlap,
            # subdomain MUMPS factorizations.
            workspace.ksp.setType(PETSc.KSP.Type.GMRES)
            workspace.ksp.setGMRESRestart(self.gmres_restart)
            workspace.ksp.setTolerances(
                rtol=self.gmres_rtol,
                max_it=self.gmres_max_it,
            )

            pc = workspace.ksp.getPC()
            pc.setType(PETSc.PC.Type.ASM)
            pc.setASMOverlap(self.asm_overlap)

            workspace.ksp.setOperators(K_mat)
            pc.setUp()

            sub_ksps = pc.getASMSubKSP()
            for sub_ksp in sub_ksps:
                sub_ksp.setType(PETSc.KSP.Type.PREONLY)
                sub_pc = sub_ksp.getPC()
                sub_pc.setType(PETSc.PC.Type.LU)
                sub_pc.setFactorSolverType("mumps")
                self._configure_mumps_factor(sub_pc, parallel_analysis=False)

            workspace.asm_configured = True
        else:
            # Subsequent iterations: update the operator only.  PETSc will
            # call PCSetUp internally on the next solve, re-factorizing the
            # subdomain matrices while reusing their type and ICNTL settings.
            workspace.ksp.setOperators(K_mat)

        self._prepare_rhs(workspace, F_vec)
        workspace.ksp.solve(workspace.rhs, workspace.du)

    # ------------------------------------------------------------------
    # Internal Helpers: GMRES / GAMG
    # ------------------------------------------------------------------

    def _build_fieldsplit_is(self, step: int, K_mat):
        """PRIVATE: Build one PETSc IS per logical field for a mixed solution step.

        Each entry in the returned list is a ``(field_name, PETSc.IS)`` pair
        whose IS contains the globally owned DOF indices for that field in the
        assembled PETSc matrix.  Returns ``None`` when the step has only a
        single logical field so that callers can fall back to plain GAMG.

        Every field is one contiguous block of the owned rows of the
        monolithic matrix (see :meth:`_MeshDOFs.get_step_block_offsets`),
        so each IS is a contiguous range starting at this rank's ownership
        offset.
        """
        mesh = self.physics.mesh
        if not mesh.mixed_spaces.get(step):
            return None  # single-field step — caller falls back to GAMG

        field_names = mesh.step_fields[step]
        offsets = mesh.get_step_block_offsets(step)
        global_offset = int(K_mat.getOwnershipRange()[0])
        comm = K_mat.getComm()

        field_iss = []
        for fname, start, end in zip(field_names, offsets, offsets[1:]):
            global_dofs = np.arange(global_offset + start, global_offset + end, dtype=PETSc.IntType)
            field_iss.append((fname, PETSc.IS().createGeneral(global_dofs, comm=comm)))

        return field_iss

    def _step_has_mixed_space(self, step: int) -> bool:
        """PRIVATE: Return whether ``step`` has a populated mixed space."""
        physics = getattr(self, "physics", None)
        mesh = getattr(physics, "mesh", None)
        mixed_spaces = getattr(mesh, "mixed_spaces", {})
        return bool(mixed_spaces.get(step))

    @staticmethod
    def _fieldsplit_requires_schur(field_names) -> bool:
        """PRIVATE: Identify multiplier blocks with no coercive diagonal."""
        return any(name.startswith("LMultiplier") for name in field_names)

    def _solve_fieldsplit(self, workspace: SolverWorkspace, K_mat, F_vec, step: int):
        """PRIVATE: GMRES solve with a FieldSplit/GAMG block preconditioner.

        The monolithic assembled matrix is split into one block per logical
        field registered for ``step`` (e.g. displacement and pressure for a
        coupled Biot step). Coercive diagonal blocks are preconditioned by
        GAMG. General mixed systems use an additive block approximation;
        multiplier systems use a full Schur factorization so a zero
        multiplier diagonal is never passed to GAMG.

        For single-field steps (no mixed space) the method falls back
        transparently to plain GAMG on the full system.

        Notes
        -----
        PETSc IS objects describing the owned global DOF range for each field
        are built once via :meth:`_build_fieldsplit_is` and stored in
        ``workspace._fieldsplit_is`` to prevent premature garbage collection.
        They are released in :meth:`_destroy_solver_workspace`.

        All KSP/PC setup is done inside the ``fieldsplit_configured`` guard;
        subsequent Newton iterations only call ``setOperators``.
        """
        if not workspace.fieldsplit_configured:
            field_iss = self._build_fieldsplit_is(step, K_mat)

            if field_iss is None:
                # Single-field step: FieldSplit is a no-op — use plain GAMG.
                # Mark fieldsplit as configured so subsequent calls skip the
                # IS-building check and route directly through _solve_gamg.
                workspace.fieldsplit_configured = True
                workspace._fieldsplit_is = None
                self._solve_gamg(workspace, K_mat, F_vec, step)
                return

            workspace.ksp.setType(PETSc.KSP.Type.GMRES)
            workspace.ksp.setGMRESRestart(self.gmres_restart)
            workspace.ksp.setTolerances(
                rtol=self.gmres_rtol,
                max_it=self.gmres_max_it,
            )

            pc = workspace.ksp.getPC()
            pc.setType(PETSc.PC.Type.FIELDSPLIT)
            field_names = [name for name, _ in field_iss]
            use_schur = self._fieldsplit_requires_schur(field_names)
            workspace.fieldsplit_uses_schur = use_schur
            if use_schur:
                # Augmented-Lagrangian phase-field systems have a zero
                # multiplier--multiplier block.  Plain GAMG, or an additive
                # split that tries to invert that block, is therefore not a
                # valid preconditioner.  Use the phase-field block as A00 and
                # approximate S = A11 - A10 A00^{-1} A01 with PETSc's
                # diagonal-A00 (selfp) construction.
                pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
                pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
                pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
            else:
                # Independent GAMG corrections are effective for mixed
                # systems whose diagonal blocks are themselves coercive.
                pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

            for fname, is_field in field_iss:
                pc.setFieldSplitIS((fname, is_field))

            # Keep IS objects alive until the workspace (and its PC) is destroyed.
            workspace._fieldsplit_is = [is_f for _, is_f in field_iss]

            workspace.ksp.setOperators(K_mat)
            pc.setUp()

            # Apply GAMG to elliptic field blocks.  In a Schur split the final
            # sub-KSP acts on the multiplier Schur complement; its selfp
            # preconditioning matrix is mass-like and is handled robustly by
            # Jacobi without asking GAMG to coarsen a zero A11 block.
            sub_ksps = pc.getFieldSplitSubKSP()
            for index, sub_ksp in enumerate(sub_ksps):
                sub_ksp.setType(PETSc.KSP.Type.PREONLY)
                sub_pc = sub_ksp.getPC()
                if use_schur and index == len(sub_ksps) - 1:
                    sub_pc.setType(PETSc.PC.Type.JACOBI)
                else:
                    sub_pc.setType(PETSc.PC.Type.GAMG)
                    sub_pc.setGAMGType(PETSc.PC.GAMGType.AGG)
                    sub_pc.setGAMGSmooths(1)

            workspace.fieldsplit_configured = True
        else:
            # Single-field fallback: delegate to the GAMG workspace (which
            # handles its own operator update via gamg_configured).
            if workspace._fieldsplit_is is None:
                self._solve_gamg(workspace, K_mat, F_vec, step)
                return
            workspace.ksp.setOperators(K_mat)

        self._prepare_rhs(workspace, F_vec)
        workspace.ksp.solve(workspace.rhs, workspace.du)

    def _solve_gamg(self, workspace: SolverWorkspace, K_mat, F_vec, step=None):
        """PRIVATE: GMRES solve with PETSc algebraic multigrid (GAMG) preconditioning.

        GAMG is an aggregation-based algebraic multigrid preconditioner
        that is well-suited to elliptic problems such as elasticity and
        Darcy flow.  It captures long-range coupling across MPI ranks and
        typically converges in 10-20 GMRES iterations where block-Jacobi
        (ASM) needs 100-200.

        Notes
        -----
        All KSP/PC setup is done once (inside the ``gamg_configured``
        guard); subsequent Newton iterations only update the operator via
        ``setOperators``.  PETSc regenerates the multigrid hierarchy on
        the next ``solve`` call.
        """
        if not workspace.gamg_configured:
            workspace.ksp.setType(PETSc.KSP.Type.GMRES)
            workspace.ksp.setGMRESRestart(self.gmres_restart)
            workspace.ksp.setTolerances(
                rtol=self.gmres_rtol,
                max_it=self.gmres_max_it,
            )

            pc = workspace.ksp.getPC()
            pc.setType(PETSc.PC.Type.GAMG)
            pc.setGAMGType(PETSc.PC.GAMGType.AGG)
            pc.setGAMGSmooths(1)

            workspace.ksp.setOperators(K_mat)
            workspace.gamg_configured = True
        else:
            workspace.ksp.setOperators(K_mat)
        self._attach_rigid_body_modes(workspace, K_mat, step)

        self._prepare_rhs(workspace, F_vec)
        workspace.ksp.solve(workspace.rhs, workspace.du)

    def _attach_rigid_body_modes(self, workspace: SolverWorkspace, K_mat, step) -> None:
        """PRIVATE: Give GAMG the rigid-body modes of the displacement as its near-null space.

        Without them, aggregation treats the displacement components as
        unrelated scalars, and GAMG converges poorly for elasticity. The modes
        are attached only when ``step`` solves the displacement field ``u``
        alone; any other step (no ``u``, or ``u`` together with other fields)
        is left unchanged. They are rebuilt after a remesh, and attached to
        every new operator, as the near-null space belongs to the matrix.
        """
        if step is None:
            return
        epoch = getattr(getattr(self.physics, "mesh", None), "remesh_epoch", 0)
        if workspace.near_nullspace_epoch != epoch:
            if workspace.near_nullspace is not None:
                workspace.near_nullspace.destroy()
            workspace.near_nullspace = self._rigid_body_near_nullspace(step, K_mat)
            workspace.near_nullspace_epoch = epoch
        if workspace.near_nullspace is not None:
            K_mat.setNearNullSpace(workspace.near_nullspace)

    def _rigid_body_near_nullspace(self, step, K_mat):
        """PRIVATE: Orthonormal rigid-body modes of ``u`` in the layout of ``K_mat``, or ``None``.

        Returns ``None`` unless ``step`` solves only the field ``u``, as a
        vector field with one component per spatial dimension (2-D or 3-D)
        whose owned degrees of freedom are the owned rows of ``K_mat``. The
        decision is taken collectively, so all ranks attach the modes or none.
        """
        mesh = getattr(self.physics, "mesh", None)
        if mesh is None or list(getattr(mesh, "step_fields", {}).get(step, [])) != ["u"]:
            return None
        space = mesh.fields["u"].function_space
        dim = mesh.mesh.geometry.dim
        block_size = space.dofmap.index_map_bs
        n_owned = space.dofmap.index_map.size_local
        comm = K_mat.getComm().tompi4py()
        layout_matches = block_size == dim and dim in (2, 3) and K_mat.getLocalSize()[0] == n_owned * block_size
        if not comm.allreduce(layout_matches, op=MPI.LAND):
            return None

        # The modes are interpolated, as the degrees of freedom need not be point values (FEMCubes
        # uses Bernstein elements); rigid-body motions are linear, so any space of degree >= 1
        # represents them exactly. Centring the rotations improves the orthonormalisation.
        points = mesh.mesh.geometry.x[:, :dim]
        centre = comm.allreduce(points.sum(axis=0), op=MPI.SUM) / max(comm.allreduce(len(points), op=MPI.SUM), 1)
        motions = [lambda x, k=k: np.ones_like(x[0]) * (np.arange(dim) == k)[:, None] for k in range(dim)]
        for a, b in ([(0, 1)] if dim == 2 else [(0, 1), (1, 2), (2, 0)]):  # rotations in the (a, b) plane
            def rotation(x, a=a, b=b):
                values = np.zeros((dim, x.shape[1]))
                values[a], values[b] = -(x[b] - centre[b]), x[a] - centre[a]
                return values
            motions.append(rotation)
        mode_function = fem.Function(space)
        modes = []
        for motion in motions:
            mode_function.interpolate(motion)
            modes.append(mode_function.x.array[:n_owned * block_size].copy())

        vectors = []
        for mode in modes:  # Gram-Schmidt orthonormalisation
            vector = K_mat.createVecRight()
            vector.setArray(mode)
            for previous in vectors:
                vector.axpy(-vector.dot(previous), previous)
            vector.normalize()
            vectors.append(vector)
        nullspace = PETSc.NullSpace().create(vectors=vectors, comm=K_mat.getComm())
        for vector in vectors:
            vector.destroy()
        return nullspace
