from typing import Any

from femcubes.core.base_model import BaseModel
from femcubes.core.utils.mpi_utils import mprint

class _PhysicsModels:
    """Internal mix-in responsible for model lifecycle management.

    During ``__init__``, iterates over ``params.ModelNames``, looks up
    each model type in :attr:`BaseModel._registry`, and constructs the
    model instance.  It also exposes the public lifecycle helpers used
    by the time solver to initialize fields, revert rejected steps, and
    commit converged ones.

    Public API
    ----------
    ``Reset_Step()``
        Revert all models to the last committed state.
    ``Initialize_Fields()``
        Initialize all model-owned fields before the solve loop.
    ``Post_Remesh()``
        Clear assembled workspaces and refresh model-local mesh state.
    ``commit()``
        Commit the converged state for all models.

    Attributes
    ----------
    models : dict
        ``{model_name: BaseModel}`` instances created from the params.
    """

    def __init__(self, mesh: Any, params: Any) -> None:
        self._initialize_models()
        super().__init__(mesh, params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Reset_Step(self) -> None:
        """Reset all models to their previously committed state."""
        for model in self.models.values():
            model.reset_step()
    
    def Initialize_Fields(self) -> None:
        """Initialize all fields owned by the configured models."""
        for model in self.models.values():
            model.initialize_fields()

    def Post_Remesh(self) -> None:
        """Refresh physics state after a new live mesh epoch is installed.

        The mesh installation must be complete before this method is called.
        Existing model instances and scalar runtime history are retained.
        """
        self._destroy_assembled_objects()
        self._create_weak_constants()
        for model_name in self.params.ModelNames:
            self.models[model_name].Post_Remesh()
            
    def commit(self) -> None:
        """Commit the current state for all configured models."""
        for model in self.models.values():
            model.commit()

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _initialize_models(self) -> None:
        """PRIVATE: Instantiate all configured physics and boundary models."""
        mprint("Initializing physics models...")
        self.models = {}

        for model_name in self.params.ModelNames:
            mprint(f"\tSetting up model: {model_name}")

            model_info = self.params.Models[model_name]
            model_cls = BaseModel.get_model_class(model_info["Type"])
            if self.mesh.axisymmetric and not model_cls.SUPPORTS_AXISYMMETRY:
                raise ValueError(f"Model '{model_name}' ({model_info['Type']}) does not support axisymmetric meshes.")
            self.models[model_name] = model_cls(model_name, self.params, self.mesh)

        self.mesh.finalize_test_functions()
        self.mesh.print_dofs_info()
