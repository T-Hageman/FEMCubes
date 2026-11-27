"""Enumerations used by the FEMCubes solver layer."""

from enum import Enum


class LinearSolver(Enum):
    """Available linear solver backends.

    Members
    -------
    LU
        Direct LU factorisation via MUMPS.
    GMRES
        Iterative GMRES with ASM / MUMPS sub-domain preconditioner.
        Effective for small-to-medium problems; poor for large elasticity
        problems where long-range coupling is important.
    GAMG
        Iterative GMRES with PETSc algebraic multigrid (GAMG) preconditioner.
        Intended for large elasticity or poromechanics problems where
        ASM/block-Jacobi converges too slowly; it does not converge for
        phase-field fracture once cracks have formed.
    FIELDSPLIT
        Iterative GMRES with PETSc FieldSplit block preconditioner.
        Each logical field (e.g. displacement, pressure) is assigned its
        own sub-KSP preconditioned by GAMG.  Recommended for monolithic
        multi-field steps (e.g. coupled Biot) where a single GAMG applied
        to the full mixed system may coarsen incorrectly.  Falls back to
        plain GAMG for single-field steps.

    Notes
    -----
    The chosen backend is read from
    :attr:`femcubes.core.params_template.ParamsTemplate.Linear_Solver` and
    normalized inside :class:`femcubes.core.solvers._lin_solvers._LinSolvers`.
    """
    LU         = "LU"
    GMRES      = "GMRES"
    GAMG       = "GAMG"
    FIELDSPLIT = "FIELDSPLIT"
