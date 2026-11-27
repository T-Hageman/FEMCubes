"""Strong Dirichlet conditions with a constant prescribed rate."""

import numpy as np

from femcubes.boundary.prescribed_value import PrescribedValue


class PrescribedRate(PrescribedValue):
    r"""Strong Dirichlet condition growing linearly in time on a named boundary.

    Prescribes :math:`\bar{u}(t) = u_0 + \dot{u}\,t` for a scalar field, a
    whole vector field, or one vector component. All other behaviour
    (component suffixes, incremental conditions, ``tmin`` / ``tmax``
    window, subdomains, axisymmetric meshes) is that of
    :doc:`prescribed_value`.

    .. rubric:: Fields

    * ``field``: the constrained field, provided by another model, with the
      optional suffix ``_x``, ``_y`` or ``_z`` as in :doc:`prescribed_value`.

    .. rubric:: Weak form

    None.

    .. rubric:: Outside the weak form

    * Dirichlet condition :math:`u = u_0 + \dot{u}\,(t + \Delta t)` on
      :math:`\Gamma_\mathrm{b}`, evaluated at the end of the step being
      solved. :math:`u_0` (``value``) is the intercept at physical time
      zero: the ramp does not restart at ``start_time`` or ``tmin``.
      Repeated iterations and retried steps recompute the absolute target.
    * ``value`` and ``rate`` are finite scalars or lists; a list needs one
      entry per component of the underlying field (also when a single
      component is selected), a scalar applies to every component.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Models = {
                "BC_Top": {
                    "Type": "Boundary:PrescribedRate",  # REQUIRED
                    "boundary": "top",    # REQUIRED: boundary group
                    "field": "u_y",       # REQUIRED: field name, optional suffix _x, _y or _z
                    "value": 0.0,         # REQUIRED: intercept u_0 at t = 0 (field units)
                    "rate": 1.0e-6,       # REQUIRED: slope (field units/s)
                    "tmin": 0.0,          # OPTIONAL (default None): activation start time (s)
                    "tmax": 10.0,         # OPTIONAL (default None): activation end time (s)
                    "domain": "domain",   # OPTIONAL (default "domain", the full mesh): domain group the boundary bounds
                },
            }
    """

    MODEL_TYPE = "Boundary:PrescribedRate"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        self.value = np.asarray(self.value, dtype=float)
        self.rate = np.asarray(params.Models[name]["rate"], dtype=float)
        for label, values in (("value", self.value), ("rate", self.rate)):
            if values.ndim > 1 or values.size == 0 or not np.all(np.isfinite(values)):
                raise ValueError(f"PrescribedRate {label} must be a finite scalar or nonempty component vector.")
        if self.value.ndim == self.rate.ndim == 1 and self.value.shape != self.rate.shape:
            raise ValueError("PrescribedRate value and rate vectors must have matching lengths.")

    def _get_target_value(self, time):
        """PRIVATE: Evaluate the absolute ramp at the end of the current step."""
        component_count = self.mesh.Get_Field_For_Output(self.field_name).comp_count
        for label, values in (("value", self.value), ("rate", self.rate)):
            if values.ndim == 1 and values.size != component_count:
                raise ValueError(f"PrescribedRate {label} vector must have {component_count} component entries.")
        return (self.value + self.rate*time).tolist()
