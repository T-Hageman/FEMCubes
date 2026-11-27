from typing import Any

from dolfinx import fem
from petsc4py import PETSc

from femcubes.core.utils.runtime_profiler import RuntimeProfiler


class _PhysicsBase:
    """Internal root mix-in storing the shared physics context.

    ``Physics`` combines several underscored mix-ins.  This base layer
    stores the common mesh / parameter references that the other mix-ins
    use during model creation and global assembly.

    Public API
    ----------
    Only ``__init__`` is intended to be called from the composite
    ``Physics`` class.  The mix-in itself is not constructed directly.

    Attributes
    ----------
    mesh : Mesh
        The computational mesh.
    params : ParamsTemplate
        Simulation parameter object.
    DEBUG_PRINT_TIMING : bool
        If ``True``, each assembly / solve phase prints elapsed time.
    """

    def __init__(self, mesh: Any, params: Any) -> None:
        """Store shared mesh and parameter references for later mix-ins."""
        self.mesh = mesh
        self.params = params
        self.DEBUG_PRINT_TIMING = params.DEBUG_PRINT_TIMING if hasattr(params, 'DEBUG_PRINT_TIMING') else False
        self._runtime_profiler = RuntimeProfiler(
            enabled=getattr(params, "PROFILE_TIMINGS", False)
        )
        self._create_weak_constants()
        
        super().__init__(mesh, params)

    def _create_weak_constants(self) -> None:
        """PRIVATE: Create mesh-bound time constants used by weak forms."""
        current_time = getattr(
            self.params,
            "t",
            getattr(self.params, "start_time", 0.0),
        )
        self.time_constant = fem.Constant(
            self.mesh.mesh,
            PETSc.ScalarType(current_time),
        )
        self.dt_constant = fem.Constant(
            self.mesh.mesh,
            PETSc.ScalarType(self.params.dt),
        )

    def _sync_weak_constants(self) -> None:
        """PRIVATE: Synchronize time constants with mutable parameters."""
        self.time_constant.value = PETSc.ScalarType(
            getattr(
                self.params,
                "t",
                getattr(self.params, "start_time", 0.0),
            )
        )
        self.dt_constant.value = PETSc.ScalarType(self.params.dt)
