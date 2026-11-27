"""Boundary condition model for shallow-ice thickness and level-set transport."""

import math
from typing import List, Optional

from femcubes.core.base_model import BaseModel


class ShallowIceBC(BaseModel):
    r"""Thickness, velocity and level-set boundary conditions for :doc:`shallow_ice`.

    The model sets the boundary values of the shallow-ice problem on one
    boundary group. The velocity components and the level-set value are
    imposed strongly, whereas the thickness, a discontinuous (DG) field, is
    driven towards a prescribed inflow value by a penalty. Every entry is
    optional; a model without any contributes nothing. 2-D (``v_x``,
    ``v_y``).

    .. rubric:: Fields

    * ``h``, ``v``: provided by :doc:`shallow_ice`; ``LS``: provided by
      :doc:`level_set`. ``boundary`` must lie on the support of ``h`` for the
      penalty term.
    * Registers no fields.

    .. rubric:: Weak form

    With ``h`` set, the contribution to the residual of ``h`` (step of
    ``h``), with test function :math:`\delta h` and :math:`\Gamma_b` the
    boundary group ``boundary``, is

    .. math::

        R_h = \int_{\Gamma_b} k_h\,(h - h_\mathrm{in})\,\delta h\,\mathrm{d}s,

    with :math:`k_h` = ``h_penalty`` and :math:`h_\mathrm{in}` = ``h``. It acts
    on the whole boundary group, independent of the flow direction, and
    drives :math:`h` towards :math:`h_\mathrm{in}` also at zero velocity. The
    Jacobian is exact.

    .. rubric:: Outside the weak form

    * Dirichlet conditions on ``v``: component 0 to ``v_x`` and component 1
      to ``v_y``, each only if given.
    * Dirichlet condition on ``LS``: ``LS`` if given; otherwise, if ``h`` is
      given and ``Solution_Steps`` contains ``"LS"``,
      :math:`3w` with :math:`w` = ``Level_Set["Transition_Width"]``
      (fully inside ice). None for velocity-only boundaries.
    * No state variables or outputs.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"v": 0, "h": 1, "LS": 2}  # REQUIRED: "v" and "h"; "LS" OPTIONAL
            Level_Set = {"Transition_Width": 2000.0}    # REQUIRED with "LS" in Solution_Steps (m)
            Models = {
                "BC_Inlet": {
                    "Type": "Glaciology:ShallowIceBC",  # REQUIRED
                    "boundary": "left",                 # REQUIRED: boundary group
                    "h": 100.0,                         # OPTIONAL (default None): inflow thickness h_in (m)
                    "h_penalty": 1.0,                   # OPTIONAL (default 1.0): penalty k_h (m/s)
                    "LS": 6000.0,                       # OPTIONAL (default 3 w if "h" is given, else None): LS value (m)
                    "v_x": 1.0e-5,                      # OPTIONAL (default None): x velocity (m/s)
                    "v_y": 0.0,                         # OPTIONAL (default None): y velocity (m/s)
                },
            }

    Increase ``h_penalty`` if the boundary thickness converges too slowly on
    coarse meshes or with large time steps.
    """

    MODEL_TYPE = "Glaciology:ShallowIceBC"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)

        model_cfg = params.Models[name]
        self.boundary = model_cfg["boundary"]
        self.h_inlet: Optional[float] = model_cfg.get("h", None)
        self.LS: Optional[float] = model_cfg.get("LS", None)
        self.h_penalty: float = model_cfg.get("h_penalty", 1.0)
        self.v_x: Optional[float] = model_cfg.get("v_x", None)
        self.v_y: Optional[float] = model_cfg.get("v_y", None)

        self.step_V = params.Solution_Steps["v"]
        self.step_H = params.Solution_Steps["h"]
        self.step_LS = params.Solution_Steps.get("LS", None)
        self.default_LS = self._get_default_level_set_value(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Add the penalty BC for h on the prescribed inflow boundary.

        Adds ``h_penalty * (h - h_inlet) * q`` to the residual, which drives
        ``h`` toward ``h_inlet`` independently of the flow velocity.  Returns
        ``None`` when the step does not include the thickness field or when no
        inlet thickness is configured.
        """
        if step != self.step_H or self.h_inlet is None:
            return None

        h = self.mesh.Get_Field("h", step=self.step_H)
        h_t = self.mesh.Get_TestFunction("h", self.step_H)
        h_tr = self.mesh.Get_Trial_Function("h", self.step_H)

        ds_boundary = self.mesh.Get_Field_Boundary_Integrator("h", self.boundary)

        # Penalty enforcement: κ·(h − h_inlet)·q on the inflow boundary.
        # Works even when v = 0 (Weertman start-up), vanishes at steady state.
        F_h = self.h_penalty * (h - self.h_inlet) * h_t * ds_boundary
        K_h = self.h_penalty * h_tr * h_t * ds_boundary

        return K_h, F_h

    def get_bcs_for_field(self, field_name: str) -> Optional[List]:
        """Return strong Dirichlet BCs contributed by this boundary model."""
        if field_name == "v":
            bcs = []
            if self.v_x is not None:
                bcs += self.mesh.Build_Increment_Dirichlet_BCs("v", self.boundary, self.v_x, component=0)
            if self.v_y is not None:
                bcs += self.mesh.Build_Increment_Dirichlet_BCs("v", self.boundary, self.v_y, component=1)
            return bcs if bcs else None

        if field_name == "LS":
            if self.LS is not None:
                return self.mesh.Build_Increment_Dirichlet_BCs("LS", self.boundary, self.LS)
            if self.h_inlet is not None:
                return self.mesh.Build_Increment_Dirichlet_BCs("LS", self.boundary, self.default_LS)
            return None

        return None

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_default_level_set_value(self, params) -> Optional[float]:
        """PRIVATE: Return the default LS inflow value for prescribed-height boundaries."""
        if self.step_LS is None:
            return None

        width = float(params.Level_Set["Transition_Width"])
        if not math.isfinite(width) or width <= 0.0:
            raise ValueError("Glaciology:ShallowIceBC Level_Set_Transition_Width must be a positive finite number.")
        return 3.0 * width
