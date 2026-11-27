"""Pressure- and gravity-driven fluid flow through phase-field fractures."""

from ufl import grad, inner, conditional, lt, as_vector

from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.poromechanics.shared_poroelasticity import saturation_from_pressure, validate_saturation_parameters

class FractureFlow(BaseModel):
    r"""Additional Darcy flow through phase-field fractures.

    Open cracks conduct fluid far more readily than the surrounding porous
    material. As a phase-field crack is diffuse, the flow along it is
    represented by an additional Darcy flux with a fracture permeability,
    scaled by the phase field, in the region that the phase field marks as
    cracked. The model adds this fracture flux to the pore-fluid mass balance
    of :doc:`pore_pressure`, which keeps providing the porous flow, storage
    and solid coupling. The flux is active where the phase field is at
    least 0.1 and the porosity is below 0.2. Supports 2-D and 3-D, not
    axisymmetric meshes.

    .. rubric:: Fields

    * ``pw``: read, provided by :doc:`pore_pressure`; this model adds to its
      residual.
    * ``poros``: read, provided by :doc:`pore_pressure` (or a glaciology
      solid model).
    * ``phasefield``: read, provided by a phase-field model (e.g.
      :doc:`../solids/phase_field`), taken from step
      ``Solution_Steps["phasefield"]``.

    All three must live on the whole domain group of ``domain``, otherwise
    assembly raises a ``ValueError``.

    .. rubric:: Weak form

    With the symbols of :doc:`pore_pressure`, :math:`k_f` =
    ``fracture_permeability`` and

    .. math::

        \hat\phi = \begin{cases}\phi & \phi \geq 0.1 \text{ and } n < 0.2\\
        0 & \text{otherwise}\end{cases},
        \qquad
        \mathbf{q}_f = -\frac{k_f}{\mu_w}\,\hat\phi\,S_w\,(\nabla p_w - \rho_w\mathbf{g}),

    the contribution to the residual of ``pw`` is

    .. math::

        R_{p_w} = -\int_\Omega \nabla\delta p_w\cdot\mathbf{q}_f\,\mathrm{d}x.

    :math:`S_w = 1` with ``Saturated=True``; otherwise :math:`S_w(p_w)` at
    the current iterate, with the van Genuchten relation of
    :doc:`pore_pressure`. The Jacobian keeps only the
    :math:`\nabla p_w` derivative,
    :math:`\int_\Omega (k_f/\mu_w)\,\hat\phi\,S_w\,\nabla\delta p_w\cdot\nabla\Delta p_w\,\mathrm{d}x`;
    the saturation derivative is dropped, so for unsaturated flow it is
    not the consistent tangent.

    .. rubric:: Outside the weak form

    * None (no state, no Dirichlet conditions, no outputs).

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                                    # REQUIRED: 2 or 3
            Field_Orders = {"pw": 1}                   # REQUIRED: entry for pw
            Solution_Steps = {"pw": 0, "phasefield": 1}  # REQUIRED: entries for pw and phasefield
            gravity = -9.81                            # REQUIRED: acceleration along the last axis (m/s^2)
            Materials = {
                "Firn": {
                    "fracture_permeability": 1.0e-8,   # REQUIRED: fracture permeability k_f >= 0 (m^2)
                    "Saturation_p_ref": 3.0e3,         # OPTIONAL (default 1e6), Saturated=False only: van Genuchten pressure (Pa)
                    "Saturation_m": 0.6,               # OPTIONAL (default 0.5), Saturated=False only: van Genuchten m (-)
                },
                "Water": {
                    "Density": 1000.0,                 # REQUIRED: fluid density rho_w (kg/m^3)
                    "Viscosity": 1.0e-3,               # REQUIRED: dynamic viscosity mu_w > 0 (Pa s)
                },
            }
            Models = {
                "FractureFlow": {
                    "Type": "PoroMechanics:FractureFlow",  # REQUIRED
                    "Material": "Firn",                # REQUIRED: key in Materials (same as PorePressure)
                    "Fluid": "Water",                  # REQUIRED: key in Materials (same as PorePressure)
                    "Saturated": False,                # OPTIONAL (default True): as in PorePressure
                    "domain": "domain",                # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "PoroMechanics:FractureFlow"

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)

        if "pw" not in params.Solution_Steps:
            raise ValueError("FractureFlow requires 'pw' in Solution_Steps.")
        if "phasefield" not in params.Solution_Steps:
            raise ValueError("FractureFlow requires 'phasefield' in Solution_Steps.")
        if "pw" not in params.Field_Orders:
            raise ValueError("FractureFlow requires 'pw' in Field_Orders.")

        self.Step_pw = params.Solution_Steps["pw"]
        self.Step_phasefield = params.Solution_Steps["phasefield"]

        model_settings = params.Models[self.name]
        self.material = model_settings["Material"]
        self.fluid = model_settings["Fluid"]
        self.Saturated = model_settings.get("Saturated", True)

        material_settings = params.Materials[self.material]
        fluid_settings = params.Materials[self.fluid]
        self.permeability = material_settings["fracture_permeability"]
        self.viscosity = fluid_settings["Viscosity"]
        self.rho_f = fluid_settings["Density"]
        self.gravity = params.gravity
        self.gVec = as_vector([0.0, self.gravity]) if params.dim == 2 else as_vector([0.0, 0.0, self.gravity])
        if self.permeability < 0.0:
            raise ValueError("fracture_permeability must be non-negative.")
        if self.viscosity <= 0.0:
            raise ValueError("Fluid Viscosity must be strictly positive.")

        if not self.Saturated:
            self.p_ref = material_settings.get("Saturation_p_ref", 1.0e6)
            self.m = material_settings.get("Saturation_m", 0.5)
            validate_saturation_parameters(self.p_ref, self.m)

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Return the fracture-flow contribution to the pressure equation."""
        if step != self.Step_pw:
            return None
        self._require_fields_on_domain(("pw", "poros", "phasefield"))
        dx = self.mesh.Get_Domain_Integrator(self.domain)

        pw = self.mesh.Get_Field("pw", step=self.Step_pw)
        phasefield = self.mesh.Get_Field("phasefield", step=self.Step_phasefield)
        poros = self.mesh.Get_Field("poros")
        pw_t = self.mesh.Get_TestFunction("pw", self.Step_pw)
        pw_tr = self.mesh.Get_Trial_Function("pw", self.Step_pw)

        saturation, d_saturation = self._saturation(pw)
        phasefield_capped = conditional(lt(phasefield, 0.1), 0.0, phasefield)
        phasefield_capped = conditional(lt(poros, 0.2), phasefield_capped, 0.0)  # No enhancement for already-porous material
        mobility = self.permeability / self.viscosity
        driving = grad(pw) - self.rho_f * self.gVec
        flux  = -mobility * phasefield_capped * saturation * driving
        dflux = -mobility * phasefield_capped * (saturation * grad(pw_tr) + 0.0*d_saturation * pw_tr * driving)

        F = -inner(grad(pw_t), flux) * dx
        K = -inner(grad(pw_t), dflux) * dx
        
        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        """Report that the form uses no mutable model-local coefficients."""

        return False

    def _require_fields_on_domain(self, field_names) -> None:
        """PRIVATE: Reject coupled fields that do not cover this model's domain."""
        domain_tags = set(self.mesh.get_domain_tags(self.domain))
        for field_name in field_names:
            support = self.mesh.Get_Field_Support(field_name)
            if support.is_interface or not domain_tags <= set(support.tags):
                raise ValueError(
                    f"FractureFlow '{self.name}' on '{self.domain}' requires field '{field_name}' on that domain, "
                    f"but it lives on {list(support.domain_names)}."
                )

    def _saturation(self, pore_pressure):
        """PRIVATE: Return saturation and its pressure derivative."""
        if self.Saturated:
            return 1.0, 0.0
        saturation, d_saturation, _ = saturation_from_pressure( pore_pressure, self.p_ref, self.m)
        return saturation, d_saturation
