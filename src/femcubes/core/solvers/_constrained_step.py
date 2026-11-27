"""Internal constrained Newton solve of one staggered step (frozen fields, combined equations)."""

from mpi4py import MPI
import numpy as np
from petsc4py import PETSc

from femcubes.core.utils.mpi_utils import comm, mprint
from femcubes.core.solvers._step_solver_base import _StepSolverBase


class _ConstrainedStep(_StepSolverBase):
    """PRIVATE: Newton solve of a step with some fields held fixed and some equations replaced by combinations."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Solve_Constrained_Step(self, step: int, frozen_fields, combined_rows=None) -> int:
        """Solve one step with some of its fields held fixed and some equations replaced by combinations.

        Newton's method on the fields of ``step`` that are not listed in
        ``frozen_fields`` (their increments are zero; their rows are replaced
        by identities). ``combined_rows`` maps a free field to
        ``{source_field: coefficient}``: at every node the residual row of the
        free field is replaced by the weighted sum of the source fields' rows
        at the same node (all fields of the same nodal layout), and so is its
        tangent row. Rows of the free field under a Dirichlet condition keep
        their condition, and source rows under a Dirichlet condition are left
        out of the sum.

        This gives consistent initial states. For example, with the
        concentrations frozen, the charge-weighted sum of the species balances
        (current conservation) determines the electrolyte potential, whose
        own equation (electroneutrality) does not contain it; see
        :meth:`femcubes.electrochemistry.electrolyte.Electrolyte.Current_Conservation_Rows`.
        Time-step terms vanish when the frozen fields equal their old values.

        The linear systems are solved with a direct LU factorisation (MUMPS)
        whatever ``Linear_Solver`` is. Convergence uses ``abs_tol`` and
        ``rel_tol`` on the RMS of the combined residual over the free rows, and
        ``max_it`` iterations; an update that makes the residual non-finite
        (overflow of exponential rate laws) is halved, up to ten times.

        Returns the number of iterations; raises ``RuntimeError`` without
        convergence and ``ValueError`` for unknown or inconsistent fields.
        """
        mesh = self.physics.mesh
        field_names = list(mesh.step_fields.get(step, []))
        frozen = list(frozen_fields)
        combined_rows = {target: dict(sources) for target, sources in (combined_rows or {}).items()}
        unknown = sorted((set(frozen) | set(combined_rows) | {s for v in combined_rows.values() for s in v}) - set(field_names))
        if unknown:
            raise ValueError(f"Fields not solved in step {step}: {unknown}.")
        if set(frozen) & set(combined_rows):
            raise ValueError(f"Combined rows cannot replace the equations of frozen fields: {sorted(set(frozen) & set(combined_rows))}.")
        if set(frozen) >= set(field_names):
            raise ValueError("Every field of the step is frozen; nothing to solve.")

        offsets = mesh.get_step_block_offsets(step)
        frozen_local = np.concatenate([np.arange(start, end) for name, start, end in zip(field_names, offsets, offsets[1:])
                                       if name in frozen] or [np.empty(0, dtype=np.int64)]).astype(np.int64)
        frozen_global = mesh.get_step_global_indices(step, frozen_local).astype(PETSc.IntType)
        n_free = comm.allreduce(offsets[-1] - frozen_local.size, op=MPI.SUM)
        P = self._row_combination_matrix(step, combined_rows)

        ksp = PETSc.KSP().create(comm)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        reference = None
        try:
            for iteration in range(self.max_iterations + 1):
                K, F = self.physics.Get_Kmat_Fvec(step)
                K_c = P.matMult(K)
                F_c = F.duplicate()
                P.mult(F, F_c)
                K_c.zeroRows(frozen_global, diag=1.0)
                F_c.setValues(frozen_global, np.zeros(frozen_global.size))
                F_c.assemble()
                norm = self._scaled_norm(F_c, step) / np.sqrt(max(n_free, 1))
                reference = norm if reference is None else reference
                mprint(f"\t\tConstrained step {step}, iteration {iteration}: RMS residual = {norm:.6e}")
                if norm < self.absolute_tolerance or (iteration > 0 and norm < self.relative_tolerance * reference):
                    K_c.destroy()
                    F_c.destroy()
                    return iteration
                if iteration == self.max_iterations:
                    K_c.destroy()
                    F_c.destroy()
                    break
                du = F_c.duplicate()
                F_c.scale(-1.0)
                ksp.setOperators(K_c)
                ksp.solve(F_c, du)
                self._limit_positive_update(du, step)
                self._update_solution(du, step)
                # A step into overflow (exponential rate laws) is halved until the residual is finite again.
                for _halving in range(10):
                    F_trial = P.getVecLeft()
                    P.mult(self.physics.Get_Fvec(step), F_trial)
                    F_trial.setValues(frozen_global, np.zeros(frozen_global.size))
                    F_trial.assemble()
                    finite = np.isfinite(F_trial.norm())
                    F_trial.destroy()
                    if finite:
                        break
                    du.scale(-0.5)
                    self._update_solution(du, step)
                    du.scale(-1.0)
                for obj in (du, K_c, F_c):
                    obj.destroy()
        finally:
            ksp.destroy()
            P.destroy()
        raise RuntimeError(f"Constrained solve of step {step} did not converge in {self.max_iterations} iterations "
                           f"(RMS residual {norm:.3e}).")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _row_combination_matrix(self, step, combined_rows):
        """PRIVATE: Return the matrix P (layout of the step's tangent) whose product with the residual replaces rows by combinations.

        P is the identity except for the owned rows of each target field of
        ``combined_rows`` (see :meth:`Solve_Constrained_Step`): such a row takes
        the coefficients of the source rows at the same node, unless the
        target row is a Dirichlet row; Dirichlet source rows are left out.
        """
        mesh = self.physics.mesh
        K, _F = self.physics.Get_Kmat_Fvec(step)
        start, end = K.getOwnershipRange()
        dirichlet = set(self.physics.Get_Dirichlet_Rows(step).tolist())
        rows = {row: {row: 1.0} for row in range(start, end)}
        for target, sources in combined_rows.items():
            _weights, dofs, _vals = mesh.Get_Weights_Dofs([target] + list(sources), step)
            target_rows = mesh.get_step_global_indices(step, dofs[target])
            source_rows = {name: mesh.get_step_global_indices(step, dofs[name]) for name in sources}
            for node, (local, row) in enumerate(zip(dofs[target], target_rows)):
                if int(local) in dirichlet:
                    continue
                rows[int(row)] = {int(source_rows[name][node]): float(coefficient)
                                  for name, coefficient in sources.items()
                                  if coefficient != 0.0 and int(dofs[name][node]) not in dirichlet}
        n_entries = max((len(entries) for entries in rows.values()), default=1)
        P = PETSc.Mat().createAIJ(K.getSizes(), nnz=(n_entries, n_entries), comm=comm)
        P.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        for row, entries in rows.items():
            if entries:
                P.setValues([row], list(entries), list(entries.values()))
        P.assemble()
        return P
