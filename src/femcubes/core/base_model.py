"""Base abstractions and registration utilities for FEMCubes models.

Concrete constitutive, transport, and boundary-condition models all
derive from :class:`BaseModel`.  The base class provides two essential
services:

* a registry keyed by ``MODEL_TYPE`` strings so user input can refer to
  models symbolically
* a small lifecycle interface used by :class:`femcubes.core.physics.physics.Physics`
  and the solver stack
"""

from typing import Dict, Type
import numpy as np

from femcubes.core._discover import load_plugins
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate

class BaseModel:
    """Abstract base class for all physics and boundary-condition models.

    Every concrete model must subclass ``BaseModel`` and set
    :attr:`MODEL_TYPE` to a unique ``"Domain:Name"`` string (e.g.
    ``"Solid:LinearElastic"``).  Registration into the global
    :attr:`_registry` happens automatically via ``__init_subclass__``.

    The physics layer only depends on this base interface.  As a result,
    any model that derives from ``BaseModel`` and implements the
    lifecycle hooks below can participate in the framework without any
    further manual registration step.

    Typical responsibilities of a subclass include:

    * registering mesh fields or test functions during ``__init__``
    * returning stable UFL weak-form contributions from
      :meth:`Get_Weak_Form`
    * updating form coefficients and constitutive state through
      :meth:`Update_Weak_Constants`
    * optionally contributing Dirichlet BCs through
      :meth:`get_bcs_for_field`
    * updating history variables in :meth:`commit` and
      :meth:`reset_step`

    Attributes
    ----------
    REQUIRED_PARAMETERS : dict
        Parameter names that the model expects on the ``params`` object.
    REQUIRED_MODEL_SETTINGS : dict
        Keys that must appear in the model's entry inside ``params.Models``.
    MODEL_TYPE : str
        Unique identifier used to look up this model class at runtime.
    SUPPORTS_AXISYMMETRY : bool
        Whether the model is valid on axisymmetric (r-z) meshes. Only
        models whose weak forms are scalar and integrate through the mesh
        integrators (which carry the radius weight) may set this.
    _registry : dict
        Class-level mapping ``{MODEL_TYPE: class}`` populated automatically.
    """

    REQUIRED_PARAMETERS = {}
    REQUIRED_MODEL_SETTINGS = {}
    MODEL_TYPE: str = ""  # Override in subclasses, e.g. "Solid:Stokes"
    SUPPORTS_AXISYMMETRY: bool = False

    # Auto-populated registry: model type string -> class
    _registry: Dict[str, Type["BaseModel"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Register subclasses automatically when they define ``MODEL_TYPE``."""
        super().__init_subclass__(**kwargs)
        model_type = getattr(cls, "MODEL_TYPE", "")
        if model_type:
            BaseModel._registry[model_type] = cls

    @classmethod
    def get_model_class(cls, type_string: str) -> Type["BaseModel"]:
        """Look up a registered model class by its ``MODEL_TYPE`` string.

        Parameters
        ----------
        type_string : str
            Registry key such as ``"Solid:LinearElastic"`` or
            ``"Boundary:PrescribedValue"``.

        Returns
        -------
        type[BaseModel]
            The registered subclass corresponding to ``type_string``.

        Raises
        ------
        ValueError
            If no model has been registered under the requested type.
        """
        if type_string not in cls._registry:
            # The type may be provided by an installed plugin.
            load_plugins()
        if type_string not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown model type: {type_string}. Available: {available}"
            )
        return cls._registry[type_string]

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        """Instantiate a model.

        Parameters
        ----------
        name : str
            Human-readable name for this model instance (key in
            ``params.Models``).
        params : ParamsTemplate
            Simulation parameter object.
        mesh : Mesh
            The computational mesh.

        Notes
        -----
        Subclasses typically extend ``__init__`` to register fields,
        cache material parameters, and store the solution step(s) they
        participate in.
        """
        self.name = name
        self.params = params
        self.mesh = mesh
        self.type = self.MODEL_TYPE
        # (Sub)domain this model acts on, e.g. "domain:centre"; the full mesh by default.
        self.domain = getattr(params, "Models", {}).get(name, {}).get("domain", "domain")
    
    def initialize_fields(self) -> None:
        """Set initial conditions for all fields owned by this model.

        Called once after all models are constructed and before the
        time-stepping loop begins.  Override in subclasses that own
        fields or history variables requiring explicit initialization.
        """
        return None

    def Post_Remesh(self) -> None:
        """Refresh model-local state tied to the previous mesh epoch.

        Called after the live mesh, transferred fields, markers, test
        functions, and mixed spaces have all been installed for a new mesh
        epoch. Models that cache mesh-bound constants, expressions, function
        spaces, or direct field references should recreate or clear them in
        this hook. Runtime history that is independent of the mesh should be
        preserved.
        """
        return None
    
    def get_bcs_for_field(self, field_name: str) -> None:
        """Return Dirichlet BCs this model contributes for *field_name*.

        Parameters
        ----------
        field_name : str
            The name of the solution field (e.g. ``"u_x"``).

        Returns
        -------
        list of dolfinx.fem.DirichletBC or None
            Any Dirichlet conditions contributed by the model for this
            field, or ``None`` if the model does not constrain it.
        
        Notes
        -----
        This hook is optional.  Models that do not impose Dirichlet
        constraints should keep the default ``None`` return.
        """
        return None
    
    def Get_Weak_Form(self, time_constant, dt_constant, step):
        """Return stable UFL weak-form contributions for a staggered step.

        Parameters
        ----------
        time_constant, dt_constant
            Physics-owned scalar constants containing the current time and
            time increment. Forms should use these instead of embedding
            ``params.t`` or ``params.dt`` as Python literals.
        step : int
            Index of the current staggered solver step.

        Returns
        -------
        tuple of (K_form, F_form) or None
            UFL bilinear and linear forms, or ``None`` if this model
            does not contribute to *step*.  The physics layer sums these
            contributions across all active models.
        
        Notes
        -----
        Physics caches the returned objects until this model requests a
        refresh or the mesh changes. Boundary-only models should explicitly
        return ``None``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement "
            "Get_Weak_Form(time_constant, dt_constant, step)."
        )

    def Update_Weak_Constants(self, time_constant, dt_constant, step) -> bool:
        """Update mutable form coefficients and constitutive state.

        This hook is called before every assembly. Return ``True`` only when
        the UFL expression graph itself must be rebuilt; changes to values in
        existing ``Function`` and ``Constant`` objects do not require it.
        """
        return False

    def assemble_KF(self, step):
        """Compatibility entry point for standalone model callers."""
        time = getattr(self.params, "t", getattr(self.params, "start_time", 0.0))
        dt = getattr(self.params, "dt", 0.0)
        self.Update_Weak_Constants(time, dt, step)
        return self.Get_Weak_Form(time, dt, step)

    @staticmethod
    def _weak_constant_value(value) -> float:
        """PRIVATE: Read a scalar from a Python value or DOLFINx Constant."""
        raw_value = getattr(value, "value", value)
        return float(np.real(np.asarray(raw_value).reshape(-1)[0]))

    def _weak_time_window_active(self, time_constant, dt_constant, t_min, t_max) -> bool:
        """PRIVATE: Evaluate the standard model activation time window."""
        time = self._weak_constant_value(time_constant)
        dt = self._weak_constant_value(dt_constant)
        if t_max is not None and time + dt > t_max:
            return False
        if t_min is not None and time < t_min:
            return False
        return True
    
    def commit(self) -> None:
        """Commit the converged solution (e.g. copy fields to ``*_old``).

        Called once per accepted time step, after the staggered passes (converged, or when ``n_passes``
        is reached without convergence).
        Override in subclasses that maintain history fields or derived
        output quantities.
        """
        return None
    
    def reset_step(self) -> None:
        """Revert fields to the last committed state.

        Called when adaptive time-stepping rejects a step.  Override in
        subclasses that keep a committed / trial state split.
        """
        return None
