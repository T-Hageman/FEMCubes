from dolfinx import fem
import numpy as np

from mpi4py import MPI
from femcubes.core.utils.mpi_utils import comm

from femcubes.core.base_model import BaseModel

class DisplacementControl(BaseModel):
    r"""Penalty displacement control on a named boundary.

    The model loads a specimen as in a displacement-controlled test: it
    drives one scalar field or vector component on the boundary group
    ``boundary`` towards :math:`\bar{u} = \dot{u}\,(t+\Delta t)` with a
    penalty stiffness, and reports the reaction force, which follows
    directly from the penalty term. The recorded force-displacement curve
    then shows the response of the specimen, and the optional ``F_Cutoff``
    ends the simulation once the force has dropped after failure. Supports
    2-D and 3-D meshes (not axisymmetric); only the ``_x`` and ``_y``
    components can be selected.

    .. rubric:: Fields

    * ``field``: the controlled field, provided by another model (e.g.
      ``u``). A suffix ``_x`` or ``_y`` selects component 0 or 1; without
      a suffix the field must be scalar. The field may live on a subdomain;
      the boundary must then lie on its support, otherwise a ``ValueError``
      is raised.

    .. rubric:: Weak form

    With :math:`u` the selected field or component, :math:`\delta u` its
    test function, :math:`D` = ``Dummy`` and :math:`\Gamma_\mathrm{b}` the
    group ``boundary``,

    .. math::

        R_{u} = \int_{\Gamma_\mathrm{b}} D\,(u - \bar{u})\,\delta u\,\mathrm{d}s,
        \qquad
        \bar{u} = \begin{cases}
        0 & t+\Delta t < 0 \text{ and not } \texttt{negativeTime}\\
        \dot{u}\,(t+\Delta t) & \text{otherwise}\end{cases}

    with :math:`\dot{u}` = ``rate``; the Jacobian is
    :math:`\int_{\Gamma_\mathrm{b}} D\,\Delta u\,\delta u\,\mathrm{d}s`.
    The constraint is satisfied up to an error of order :math:`1/D`.

    .. rubric:: Outside the weak form

    * Global measures, updated after each converged step:
      ``<name>_F`` :math:`= -\int_{\Gamma_\mathrm{b}} D\,(u - \dot{u}(t+\Delta t))\,\mathrm{d}s`
      (the penalty force on the body), ``<name>_u`` :math:`= \dot{u}(t+\Delta t)`
      and ``<name>_ArcLen``
      :math:`= \sqrt{(F-F_n)^2/F_0^2 + (\bar{u}-\bar{u}_n)^2/\dot{u}^2}`,
      with :math:`F_0` set from the force and displacement increments of the
      first committed step. ``commit`` adds ``<name>_F_max``, the force of
      largest magnitude so far.
    * With ``F_Cutoff``: once :math:`t+\Delta t > 100\,\Delta t` and
      :math:`|F/F_\mathrm{max}| <` ``F_Cutoff``, ``end_time`` is reduced to
      :math:`t + 51\,\Delta t` (50 steps after the end of the current step);
      it is restored if the force ratio rises above ``F_Cutoff`` again.
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"u": 0}         # REQUIRED: entry for the controlled field
            Models = {
                "BC_Top": {
                    "Type": "Boundary:DisplacementControl",  # REQUIRED
                    "boundary": "top",        # REQUIRED: boundary group
                    "field": "u_y",           # REQUIRED: field name, optional suffix _x or _y
                    "rate": 1.0e-6,           # REQUIRED: prescribed rate (field units/s, m/s for u)
                    "Dummy": 1.0e12,          # REQUIRED: penalty stiffness D (Pa/m for u)
                    "F_Cutoff": 0.05,         # OPTIONAL (default None, off): end the run once |F/F_max| drops below this
                    "negativeTime": False,    # OPTIONAL (default False): apply the ramp also for t + dt < 0
                    "domain": "domain",       # OPTIONAL (default "domain", the full mesh): domain group the boundary bounds
                },
            }
    """

    MODEL_TYPE = "Boundary:DisplacementControl"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        
        self.boundary = self.params.Models[self.name]["boundary"]
        self.mesh.Require_Boundary_On_Domain(self.boundary, self.domain)
        
        dof = self.params.Models[self.name]["field"]
        self.rate = self.params.Models[self.name]["rate"]
        self.dummy = self.params.Models[self.name]["Dummy"]
        self.F_Cutoff = self.params.Models[self.name].get("F_Cutoff", None)
        self.negativeTime = self.params.Models[self.name].get("negativeTime", False)
        
        # Check if Vector sub-component BC
        if "_x" in dof:
            self.field_name = dof[:-2]
            self.component = 0
        elif "_y" in dof:
            self.field_name = dof[:-2]
            self.component = 1
        else:
            self.field_name = dof
            self.component = None
            
        self.step = params.Solution_Steps[self.field_name]
        self.u_target = fem.Constant(self.mesh.mesh, 0.0)
        self.F_max = 0.0
        
        self.FOld = 0.0
        self.uOld = 0.0
        
        self.F0 = 1.00e6
        self.changed_tmax = False

    def Post_Remesh(self) -> None:
        """Recreate the mesh-bound target constant without losing history."""
        target_value = np.asarray(self.u_target.value).reshape(-1)[0]
        self.u_target = fem.Constant(self.mesh.mesh, target_value)
        
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        if (step != self.step):
            return
        
        field = self.mesh.Get_Field(self.field_name, step)
        field_t = self.mesh.Get_TestFunction(self.field_name, step)
        field_tr= self.mesh.Get_Trial_Function(self.field_name, step)
        
        if (self.component is not None):
            field = field[self.component]
            field_t = field_t[self.component]
            field_tr = field_tr[self.component]
            
        ds = self.mesh.Get_Field_Boundary_Integrator(self.field_name, self.boundary)
        force = -self.dummy * (field - self.u_target)
        
        F = field_t * -1.0*force * ds
        K = field_t * self.dummy * field_tr * ds
        
        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        if step != self.step:
            return False
        time = self._weak_constant_value(time_constant)
        dt = self._weak_constant_value(dt_constant)
        if time + dt < 0.0 and not self.negativeTime:
            time = 0.0
            dt = 0.0
        self.u_target.value = self.rate * (time + dt)
        return False

    def PenaltyForces(self, time, dt, u):
        u_target = self.rate * (time + dt)
        self.u_target.value = u_target
        penalty_force = -self.dummy * (u - self.u_target)
        return penalty_force

    def Update_Global_Measures(self) -> None:
        field = self.mesh.Get_Field(self.field_name, self.step)
        if (self.component is not None):
            field = field[self.component]
            
        time = self.params.t
        dt = self.params.dt
        ds = self.mesh.Get_Field_Boundary_Integrator(self.field_name, self.boundary)
        force = self.PenaltyForces(time, dt, field)
        Fext = comm.allreduce(fem.assemble_scalar(self.mesh.Compile_Form(force*ds)), op=MPI.SUM)
        uExt = self.rate * (time + dt)
        self.params.Global_Measures[f"{self.name}_F"] = Fext
        self.params.Global_Measures[f"{self.name}_u"] = uExt
        
        self.ArcLen = np.sqrt((Fext-self.FOld)**2/(self.F0)**2 + (uExt - self.uOld)**2/(self.rate)**2)
        self.params.Global_Measures[f"{self.name}_ArcLen"] = self.ArcLen
        
        

    def commit(self) -> None:
        if abs(self.params.Global_Measures[f"{self.name}_F"]) > abs(self.F_max):
            self.F_max = self.params.Global_Measures[f"{self.name}_F"]
        self.params.Global_Measures[f"{self.name}_F_max"] = self.F_max
        #mprint(f"{self.name} - Current Force: {self.params.Global_Measures[f'{self.name}_F']:.6e}, Max Force: {self.F_max:.6e}")
        
        if abs(self.F0 - 1.00e6) < 1.0e-8: # base on lin-el stress changes
            self.F0 = abs(self.params.Global_Measures[f"{self.name}_F"])/np.abs((self.params.Global_Measures[f"{self.name}_u"] - self.uOld)/(self.rate))
        
        self.FOld = self.params.Global_Measures[f"{self.name}_F"]
        self.uOld = self.params.Global_Measures[f"{self.name}_u"]
        
        t = self.params.t + self.params.dt
        dt = self.params.dt
        
        if self.F_Cutoff != None:
            if self.changed_tmax and abs(self.FOld/self.F_max)>self.F_Cutoff:
                self.params.end_time = self.original_tmax
                self.changed_tmax = False
            
            if t>100.0*dt and abs(self.FOld/self.F_max)<self.F_Cutoff and self.changed_tmax == False:
                self.original_tmax = self.params.end_time
                self.params.end_time = t + 50.0*dt
                self.changed_tmax = True
        
