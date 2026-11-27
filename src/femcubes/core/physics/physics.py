"""Composite public physics object for FEMCubes simulations."""

from typing import Any

from ._physics_base import _PhysicsBase
from ._physics_models import _PhysicsModels
from ._physics_kf import _PhysicsKF

class Physics(_PhysicsBase, _PhysicsModels, _PhysicsKF):
    """Composite class managing all physics models in a simulation.

    Assembles functionality from:

    * `_PhysicsBase` -- shared mesh / parameter references
    * `_PhysicsModels` -- model instantiation, field initialization,
      commit / reset helpers
    * `_PhysicsKF` -- global stiffness-matrix and force-vector assembly,
      boundary-condition application

    ``Physics`` is the object solvers interact with directly.  It hides
    the details of individual constitutive and boundary-condition
    models, presenting instead a small unified interface for:

    * field initialization
    * global residual / tangent assembly
    * boundary-condition lookup
    * history-variable commit / reset
    * global-measure updates

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    params : ParamsTemplate
        Simulation parameter object.

    Public API
    ----------
    The public methods are provided by the mix-ins above and include:

    ``Initialize_Fields()``
        Initialize all model-owned fields.
    ``Reset_Step()``
        Restore the last committed state after a rejected solve step.
    ``Post_Remesh()``
        Destroy assembled workspaces and refresh model-local mesh state.
    ``commit()``
        Commit the converged state.
    ``Get_Kmat_Fvec(step)``
        Assemble the global stiffness matrix and residual vector.
    ``Get_Fvec(step)``
        Assemble the global residual vector without assembling a matrix.
    ``Update_Global_Measures()``
        Update model-provided global measures.
    ``get_fields_for_step(step)``
        Report which fields are solved during a staggered step.
    ``get_boundary_conditions(step=None)``
        Collect active boundary conditions.
    """
    
    def __init__(self, mesh: Any, params: Any) -> None:
        """Construct the physics stack for one mesh / parameter pair."""
        super().__init__(mesh, params)
