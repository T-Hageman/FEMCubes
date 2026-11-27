from dolfinx import fem
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from basix.ufl import element
from basix import LagrangeVariant

from femcubes.core.base_model import BaseModel


class ExternalForceFromFile(BaseModel):
    r"""Neumann load varying along one coordinate and in time, read from CSV files.

    Applies a force per unit boundary area :math:`\bar{t}(x_k, t)` to one
    scalar field or vector component on the boundary group ``boundary``.
    The load is a table over one spatial coordinate :math:`x_k` and time,
    interpolated bilinearly. Supports 2-D and 3-D meshes (not
    axisymmetric).

    .. rubric:: Fields

    * ``field``: the loaded field, provided by another model (e.g. ``u``).
      A suffix ``_x`` or ``_y`` selects component 0 or 1; without a suffix
      the field must be scalar. The field may live on a subdomain; the
      boundary must then lie on its support, otherwise a ``ValueError`` is
      raised.
    * ``phasefield``: read only, provided by :doc:`../solids/phase_field`;
      only with ``DamagedForces=True``.
    * An internal scalar Lagrange function holds :math:`\bar{t}` (order
      ``Field_Orders[field]``, default 1); it is not a registered field.

    .. rubric:: Weak form

    With :math:`u` the selected field or component, :math:`\delta u` its
    test function and :math:`\Gamma_\mathrm{b}` the group ``boundary``,

    .. math::

        R_{u} = -\int_{\Gamma_\mathrm{b}} g_\mathrm{f}(\phi)\,\bar{t}(x_k, t+\Delta t)\,\delta u\,\mathrm{d}s,

    with :math:`g_\mathrm{f}` as in :doc:`external_force`
    (:math:`(1-10^{-6})(1-\phi)^2+10^{-6}` with ``DamagedForces``, else 1).
    The term is assembled while :math:`t \geq t_\mathrm{min}` and
    :math:`t + \Delta t \leq t_\mathrm{max}` (:math:`t` the start of the
    step). It adds nothing to the Jacobian.

    .. rubric:: Outside the weak form

    * On construction, the files ``file_x`` (:math:`N_x` coordinates),
      ``file_t`` (:math:`N_t` times) and ``file_F`` (an
      :math:`N_x \times N_t` table; an :math:`N_t \times N_x` table is
      transposed) are read with ``numpy.genfromtxt`` (comma separated) and
      multiplied by ``scale_x``, ``scale_t`` and ``scale_F``. The
      coordinates and times must be strictly monotonic.
    * Before every assembly inside the time window, :math:`\bar{t}` is
      interpolated linearly in :math:`x_k` and :math:`t` at
      :math:`t+\Delta t` and interpolated into the internal function.
      Outside the table, values are extrapolated linearly from the nearest
      two entries.
    * No Dirichlet conditions, no measures.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"u": 0}         # REQUIRED: entry for the loaded field (and "phasefield" with DamagedForces)
            Field_Orders = {"u": 2}           # OPTIONAL (default 1): order of the internal load function
            Models = {
                "Load": {
                    "Type": "Boundary:ExternalForceFromFile",  # REQUIRED
                    "boundary": "top",          # REQUIRED: boundary group
                    "field": "u_y",             # REQUIRED: field name, optional suffix _x or _y
                    "file_x": "load_x.csv",     # REQUIRED: coordinates x_k (m before scaling)
                    "file_t": "load_t.csv",     # REQUIRED: times (s before scaling)
                    "file_F": "load_F.csv",     # REQUIRED: load table, N_x rows by N_t columns (Pa for u)
                    "coordinate": 0,            # OPTIONAL (default 0): index k of the coordinate x_k (0 = x, 1 = y, 2 = z)
                    "scale_x": 1.0,             # OPTIONAL (default 1.0): factor on file_x
                    "scale_t": 1.0,             # OPTIONAL (default 1.0): factor on file_t
                    "scale_F": 1.0,             # OPTIONAL (default 1.0): factor on file_F
                    "tmin": 0.0,                # OPTIONAL (default None): activation start time (s)
                    "tmax": 10.0,               # OPTIONAL (default None): activation end time (s)
                    "DamagedForces": False,     # OPTIONAL (default False): degrade the load with phasefield
                    "domain": "domain",         # OPTIONAL (default "domain", the full mesh): domain group the boundary bounds
                },
            }
    """

    MODEL_TYPE = "Boundary:ExternalForceFromFile"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)

        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)

        # Optional phase-field degradation
        self.DamagedForces = self.params.Models[self.name].get("DamagedForces", False)

        # Parse field name and optional vector component
        dof = self.params.Models[self.name]["field"]
        if "_x" in dof:
            self.field_name = dof[:-2]
            self.component = 0
        elif "_y" in dof:
            self.field_name = dof[:-2]
            self.component = 1
        else:
            self.field_name = dof
            self.component = None

        if self.DamagedForces:
            self.step_dam = params.Solution_Steps["phasefield"]

        self.step = params.Solution_Steps[self.field_name]

        self.t_min = self.params.Models[self.name].get("tmin", None)
        self.t_max = self.params.Models[self.name].get("tmax", None)

        # Spatial axis used for the 1-D profile (0 = x, 1 = y)
        self.coordinate_num = self.params.Models[self.name].get("coordinate", 0)

        # ------------------------------------------------------------------ #
        # Load and scale CSV data
        # ------------------------------------------------------------------ #
        scale_x = self.params.Models[self.name].get("scale_x", 1.0)
        scale_t = self.params.Models[self.name].get("scale_t", 1.0)
        scale_F = self.params.Models[self.name].get("scale_F", 1.0)

        x_data = np.genfromtxt(self.params.Models[self.name]["file_x"], delimiter=",") * scale_x
        t_data = np.genfromtxt(self.params.Models[self.name]["file_t"], delimiter=",") * scale_t
        F_data = np.genfromtxt(self.params.Models[self.name]["file_F"], delimiter=",") * scale_F

        # print(f"ExternalForceFromFile '{name}': Loaded CSV data shapes: x {x_data.shape}, t {t_data.shape}, F {F_data.shape}")
        # # For debugging, print x t and first row of F
        # print(f"x_data: {x_data}")
        # print(f"t_data: {t_data}")
        # print(f"F_data (first row): {F_data[0, :]}")

        # F_data must be (N_x, N_t) — check and transpose if necessary
        if F_data.shape != (len(x_data), len(t_data)):
            if F_data.shape == (len(t_data), len(x_data)):
                F_data = F_data.T
            else:
                raise ValueError(
                    f"ExternalForceFromFile '{name}': F_data shape {F_data.shape} "
                    f"is incompatible with x ({len(x_data)},) and t ({len(t_data)},)."
                )

        # 2-D bilinear interpolant over (spatial_coord, time).
        # Out-of-bounds values are extrapolated linearly from the edge cells.
        self._interpolant = RegularGridInterpolator(
            (x_data, t_data),
            F_data,
            method="linear",
            bounds_error=False,
            fill_value=None,  # linear extrapolation
        )

        # ------------------------------------------------------------------ #
        # fem.Function to hold the spatially varying force at the current step.
        # Created directly (not via mesh.Add_Field) to avoid polluting the
        # public field registry with an internal helper.
        # ------------------------------------------------------------------ #
        self._field_order = params.Field_Orders.get(self.field_name, 1)
        self._create_force_function()
        self._weak_active = None

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def Post_Remesh(self) -> None:
        """Recreate the internal traction function on the refined mesh."""
        self._create_force_function()
        self._update_force()

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if step != self.step:
            return
        if not self._weak_active:
            return None

        field_t  = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr = self.mesh.Get_Trial_Function(self.field_name, step)

        dam = 1.0
        if self.DamagedForces:
            phi = self.mesh.Get_Field("phasefield", step)
            dam = (1.0 - 1e-6) * (1.0 - phi) ** 2 + 1e-6

        if self.component is not None:
            field_t  = field_t[self.component]
            field_tr = field_tr[self.component]

        ds = self.mesh.Get_Field_Boundary_Integrator(self.field_name, self.boundary)

        F = -1.0*field_t * dam * self._force_func * ds   #minus sign to define as external force (positive=internal, negative=external)
        K = None

        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        active = self._weak_time_window_active(
            time_constant,
            dt_constant,
            self.t_min,
            self.t_max,
        )
        changed = self._weak_active is not None and active != self._weak_active
        self._weak_active = active
        if active and step == self.step:
            time = self._weak_constant_value(time_constant)
            dt = self._weak_constant_value(dt_constant)
            self._update_force(time + dt)
        return changed

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _create_force_function(self) -> None:
        """PRIVATE: Create the internal scalar force function on the live mesh."""
        elem = element(
            "Lagrange",
            self.mesh.mesh.basix_cell(),
            degree=self._field_order,
            lagrange_variant=LagrangeVariant.bernstein,
        )
        V_scalar = fem.functionspace(self.mesh.mesh, elem)
        self._force_func = fem.Function(V_scalar, name=f"ext_force_{self.name}")

    def _update_force(self, current_time=None):
        """PRIVATE: Interpolate the traction to the current time and refresh ``_force_func``."""
        t_current = (
            self.params.t + self.params.dt
            if current_time is None
            else current_time
        )
        coord_num = self.coordinate_num
        interp = self._interpolant

        def _eval(x):
            # x has shape (3, n_points) in DOLFINx's interpolation interface.
            xi = x[coord_num]
            ti = np.full(xi.shape, t_current)
            # RegularGridInterpolator expects input shape (n_points, 2).
            pts = np.column_stack([xi, ti])
            return interp(pts)

        self._force_func.interpolate(_eval)
