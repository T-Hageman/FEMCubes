from ufl import FacetNormal, SpatialCoordinate, conditional, dot

from femcubes.core.base_model import BaseModel

class OceanPressure(BaseModel):
    r"""Hydrostatic ocean pressure on a boundary, following the deformed height.

    The model applies the water pressure below a fixed sea level as a normal
    traction on one boundary group in contact with the ocean, such as an ice
    front or the base of an ice shelf. The pressure increases hydrostatically
    with depth below the sea level and vanishes above it. It is evaluated at
    the current height of each boundary point, so that a boundary that moves
    down is loaded by a higher pressure, which provides the restoring force
    of buoyancy. Acts on the displacement ``u`` if ``"u"`` is in
    ``Solution_Steps`` (any solid model), otherwise on the velocity ``v``
    (Stokes models). Supports 2-D and 3-D; the vertical axis is the last
    coordinate.

    .. rubric:: Fields

    * ``u`` or ``v``: read and tested, provided by the solid or Stokes model.
      ``boundary`` must lie on the support of this field.
    * ``<name>_OceanDisplacement`` (``<name>`` the model key): vector CG of
      order ``Field_Orders["v"]`` on ``domain``, the accumulated displacement
      :math:`\mathbf{u}_n`; only for the velocity formulation.
    * ``phasefield``: read only, with ``DamagedForces=True``.

    .. rubric:: Weak form

    With :math:`\mathbf{w}` the unknown (``u`` or ``v``), test function
    :math:`\delta\mathbf{w}`, :math:`\mathbf{n}` the outward normal and
    :math:`\Gamma_b` the boundary group ``boundary``, the contribution to the
    residual of :math:`\mathbf{w}` is

    .. math::

        R_{\mathbf{w}} =
        \int_{\Gamma_b} g_b(\phi)\,p\,(\delta\mathbf{w}\cdot\mathbf{n})\,\mathrm{d}s,
        \qquad
        p = \bigl\langle -\rho_w\,g\,(h_w - z)\bigr\rangle_+ ,

    with :math:`g<0` the signed ``gravity``, :math:`h_w` = ``WaterHeight`` and
    the current height :math:`z = x_d + u_d` for ``u`` or
    :math:`z = x_d + u_{n,d} + \Delta t\,v_d` for ``v`` (backward Euler).
    :math:`g_b = 10^{-6} + (1-10^{-6})(1-\phi)^2` with ``DamagedForces=True``
    (:math:`\phi` not clipped), else 1. The normal is the one of the
    undeformed mesh. The Jacobian is the consistent derivative
    :math:`\partial p/\partial w_d = \rho_w g` (times :math:`\Delta t` for
    ``v``) where :math:`p>0`.

    .. rubric:: Outside the weak form

    * Velocity formulation only: ``initialize_fields`` zeroes
      ``<name>_OceanDisplacement`` and ``commit`` adds
      :math:`\Delta t\,\mathbf{v}` to it.
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            gravity = -9.81                  # REQUIRED: acceleration along the last axis (m/s^2)
            Solution_Steps = {"u": 0, "phasefield": 1}  # REQUIRED: "u" (or "v"); "phasefield" with DamagedForces
            Field_Orders = {"u": 2}          # REQUIRED for "v" only: Field_Orders["v"]
            Materials = {
                "Water": {                   # REQUIRED: always read from Materials["Water"]
                    "Density": 1025.0,       # REQUIRED: water density rho_w (kg/m^3)
                    "WaterHeight": 30.0,     # OPTIONAL (default 0.0): sea level h_w on the last axis (m)
                },
            }
            Models = {
                "Ocean": {
                    "Type": "glaciology:OceanPressure",  # REQUIRED
                    "boundary": "right",                 # REQUIRED: boundary group in contact with water
                    "DamagedForces": False,              # OPTIONAL (default False): degrade the pressure with phasefield
                    "domain": "domain",                  # OPTIONAL (default "domain"): domain of the ice (and of v)
                },
            }
    """

    MODEL_TYPE = "glaciology:OceanPressure"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.dim = params.dim
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        
        self.DamagedForces = self.params.Models[self.name].get("DamagedForces", False)
        self.hw = self.params.Materials["Water"].get("WaterHeight", 0.0)
        self.rho = self.params.Materials["Water"]["Density"]
        self.g = self.params.gravity
        
        self.StokesFormulation = False
        # check if u or v in solution steps, for now assuming no slope and only basal friction in x direction
        if "u" in self.params.Solution_Steps:
            self.field_name = "u"
            self.step = self.params.Solution_Steps["u"]
            self.StokesFormulation = False
        elif "v" in self.params.Solution_Steps:
            self.field_name = "v"
            self.step = self.params.Solution_Steps["v"]
            self.StokesFormulation = True
            self.displacement_field_name = f"{self.name}_OceanDisplacement"
            mesh.Add_Vector_Field(
                self.displacement_field_name,
                f"Accumulated displacement for {self.name} ocean pressure [m]",
                "CG",
                params.Field_Orders["v"],
                params.dim,
                domain=self.domain,
            )
        else:
            raise ValueError("OceanPressure model requires either 'u' or 'v' in Solution_Steps.")
            
        if self.DamagedForces:
            self.step_dam = params.Solution_Steps["phasefield"]
            
        self.step = params.Solution_Steps[self.field_name]

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if (step != self.step):
            return
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr= self.mesh.Get_Trial_Function(self.field_name, step)
        
        dam = 1.0
        if self.DamagedForces:
            phi = self.mesh.Get_Field("phasefield", step)
            dam = (1.0-1e-6)*(1.0 - phi)**2 + 1e-6 
        
        # Integrator, spatial coordinate, and surface normals
        ds = self.mesh.Get_Field_Boundary_Integrator(self.field_name, self.boundary)
        x = SpatialCoordinate(self.mesh.mesh)
        normals = FacetNormal(self.mesh.mesh)
        
        if self.StokesFormulation:
            displacement = self.mesh.Get_Field(self.displacement_field_name)
            vertical_displacement = displacement[self.dim-1] + dt_constant * field[self.dim-1]
            vertical_displacement_tr = dt_constant * field_tr[self.dim-1]
        else:
            vertical_displacement = field[self.dim-1]
            vertical_displacement_tr = field_tr[self.dim-1]

        current_height = x[self.dim-1] + vertical_displacement
        pressure_raw = -self.rho * self.g * (self._water_height(time_constant, dt_constant) - current_height)
        p = conditional(pressure_raw > 0.0, pressure_raw, 0.0)
        p_tr = conditional(pressure_raw > 0.0, self.rho * self.g * vertical_displacement_tr, 0.0)
        
        F = dam * dot(field_t, p * normals) * ds
        K = dam * dot(field_t, p_tr * normals) * ds
        
        return K, F

    def _water_height(self, time_constant, dt_constant):
        """PRIVATE: sea level h_w during the step (constant here; time-dependent in subclasses)."""
        return self.hw

    def initialize_fields(self) -> None:
        """Initialize the velocity formulation's displacement history."""
        if self.StokesFormulation:
            self.mesh.Zero_Field(self.displacement_field_name, domain=self.domain)

    def commit(self) -> None:
        """Advance velocity-based displacement history with backward Euler."""
        if not self.StokesFormulation:
            return

        displacement = self.mesh.Get_Field(self.displacement_field_name)
        velocity = self.mesh.Get_Field_For_Output("v").field
        if velocity.function_space.mesh is not displacement.function_space.mesh:
            raise ValueError(f"OceanPressure '{self.name}': its 'domain' ({self.domain!r}) must match the domain of the velocity field 'v'.")
        dofs = self.mesh.Get_Field_Dofs(self.displacement_field_name, self.domain)
        dofs = slice(None) if dofs is None else dofs
        displacement.x.array[dofs] += self.params.dt * velocity.x.array[dofs]
        displacement.x.scatter_forward()
