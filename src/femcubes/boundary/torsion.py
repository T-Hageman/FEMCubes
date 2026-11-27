from dolfinx import fem
from mpi4py import MPI
import ufl
import numpy as np

from femcubes.core.base_model import BaseModel
from femcubes.core.utils.mpi_utils import comm

class AppliedTorsion(BaseModel):
    r"""Penalty-enforced rigid rotation of a named boundary.

    Drives the displacement ``u`` on the boundary group ``boundary``
    towards a rigid rotation about the coordinate origin by the angle
    :math:`\omega = \dot{\omega}\,(t+\Delta t)` in the plane ``plane``.
    ``"xy"`` works on 2-D and 3-D meshes, ``"xz"`` and ``"yz"`` need 3-D.
    Not for axisymmetric meshes.

    .. rubric:: Fields

    * ``u``: displacement, provided by a solid model (e.g.
      :doc:`../solids/linear_elastic`). It may live on a subdomain; the
      boundary must then lie on its support, otherwise a ``ValueError`` is
      raised.

    .. rubric:: Weak form

    With :math:`(i, j)` the in-plane coordinate indices of ``plane``
    (``"xy"``: (0, 1), ``"xz"``: (0, 2), any other value: (1, 2)),
    :math:`D` = ``Dummy`` and :math:`\Gamma_\mathrm{b}` the group ``boundary``,

    .. math::

        R_{\mathbf{u}} = \int_{\Gamma_\mathrm{b}} D\bigl[(u_i - \bar{u}_i)\,\delta u_i
        + (u_j - \bar{u}_j)\,\delta u_j\bigr]\,\mathrm{d}s,

    .. math::

        \bar{u}_i = x_i(\cos\omega - 1) - x_j\sin\omega,
        \qquad
        \bar{u}_j = x_i\sin\omega + x_j(\cos\omega - 1),

    with :math:`\dot{\omega}` = ``rate``. The Jacobian is
    :math:`\int_{\Gamma_\mathrm{b}} D(\Delta u_i\,\delta u_i + \Delta u_j\,\delta u_j)\,\mathrm{d}s`;
    the out-of-plane component is not constrained.

    .. rubric:: Outside the weak form

    * Global measures, updated after each converged step:
      ``<name>_Angle`` :math:`= \omega` and ``<name>_Torque``
      :math:`= \int_{\Gamma_\mathrm{b}} (x_i f_j - x_j f_i)\,\mathrm{d}s`,
      the torque of the penalty traction
      :math:`f_k = -D(u_k - \bar{u}_k)` on the body about the rotation axis.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"u": 0}         # REQUIRED: step of the displacement u
            Models = {
                "Twist": {
                    "Type": "Boundary:Torsion",  # REQUIRED
                    "boundary": "top",        # REQUIRED: boundary group
                    "plane": "xy",            # REQUIRED: rotation plane, "xy", "xz" or "yz"
                    "rate": 1.0e-3,           # REQUIRED: rotation rate (rad/s)
                    "Dummy": 1.0e12,          # REQUIRED: penalty stiffness D (Pa/m)
                    "domain": "domain",       # OPTIONAL (default "domain", the full mesh): domain group the boundary bounds
                },
            }
    """

    MODEL_TYPE = "Boundary:Torsion"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        self.plane = self.params.Models[self.name]["plane"]
        self.rate = self.params.Models[self.name]["rate"]
        self.dummy = self.params.Models[self.name]["Dummy"]
        
        # Select vector components based on plane
        if self.plane == "xy":
            self.field_name = "u"
            self.component = [0, 1]
        elif self.plane == "xz":
            self.field_name = "u"
            self.component = [0, 2]
        else:
            self.field_name = "u"
            self.component = [1,2]
	
        self.step = params.Solution_Steps[self.field_name]
        self.cos_omega = fem.Constant(self.mesh.mesh, 1.0)
        self.sin_omega = fem.Constant(self.mesh.mesh, 0.0)

    def Post_Remesh(self) -> None:
        """Recreate rotation constants on the live mesh domain."""
        cosine = np.asarray(self.cos_omega.value).reshape(-1)[0]
        sine = np.asarray(self.sin_omega.value).reshape(-1)[0]
        self.cos_omega = fem.Constant(self.mesh.mesh, cosine)
        self.sin_omega = fem.Constant(self.mesh.mesh, sine)
        
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if (step != self.step):
            return
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr = self.mesh.Get_Trial_Function(self.field_name, step)

        ds = self.mesh.Get_Field_Boundary_Integrator(self.field_name, self.boundary)

        # Spatial coordinates on the boundary
        x = ufl.SpatialCoordinate(self.mesh.mesh)
        c0 = x[self.component[0]]  # first in-plane coordinate
        c1 = x[self.component[1]]  # second in-plane coordinate

        # Target displacement from rigid-body rotation by omega
        #   u_target_0 = c0*(cos(w)-1) - c1*sin(w)
        #   u_target_1 = c0*sin(w)     + c1*(cos(w)-1)
        u_target_0 = c0 * (self.cos_omega - 1.0) - c1 * self.sin_omega
        u_target_1 = c0 * self.sin_omega + c1 * (self.cos_omega - 1.0)

        # Current displacement components
        u0 = field[self.component[0]]
        u1 = field[self.component[1]]

        # Test / trial components
        v0 = field_t[self.component[0]]
        v1 = field_t[self.component[1]]
        du0 = field_tr[self.component[0]]
        du1 = field_tr[self.component[1]]

        # Penalty residual  R = dummy * (u - u_target)
        F = (v0 * self.dummy * (u0 - u_target_0) + v1 * self.dummy * (u1 - u_target_1)) * ds

        # Stiffness (Jacobian w.r.t. u; target is independent of u)
        K = (v0 * self.dummy * du0 + v1 * self.dummy * du1) * ds

        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        if step != self.step:
            return False
        time = self._weak_constant_value(time_constant)
        dt = self._weak_constant_value(dt_constant)
        omega = self.rate * (time + dt)
        self.cos_omega.value = np.cos(omega)
        self.sin_omega.value = np.sin(omega)
        return False

    def Update_Global_Measures(self) -> None:
        field = self.mesh.Get_Field(self.field_name, self.step)
        ds = self.mesh.Get_Field_Boundary_Integrator(self.field_name, self.boundary)
        force_0, force_1 = self.PenaltyForces(self.params.t, self.params.dt, field)
        x = ufl.SpatialCoordinate(self.mesh.mesh)
        torque = x[self.component[0]] * force_1 - x[self.component[1]] * force_0
        self.params.Global_Measures[f"{self.name}_Torque"] = comm.allreduce(
            fem.assemble_scalar(self.mesh.Compile_Form(torque * ds)), op=MPI.SUM)
        self.params.Global_Measures[f"{self.name}_Angle"] = self.rate * (self.params.t + self.params.dt)

    def PenaltyForces(self, time, dt, field):
        """Return the two in-plane components of the penalty traction on the body."""
        omega = self.rate * (time + dt)
        self.cos_omega.value = np.cos(omega)
        self.sin_omega.value = np.sin(omega)

        x = ufl.SpatialCoordinate(self.mesh.mesh)
        c0 = x[self.component[0]]
        c1 = x[self.component[1]]

        u_target_0 = c0 * (self.cos_omega - 1.0) - c1 * self.sin_omega
        u_target_1 = c0 * self.sin_omega + c1 * (self.cos_omega - 1.0)

        u0 = field[self.component[0]]
        u1 = field[self.component[1]]

        force_0 = -self.dummy * (u0 - u_target_0)
        force_1 = -self.dummy * (u1 - u_target_1)
        return force_0, force_1
