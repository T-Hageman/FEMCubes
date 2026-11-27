import numpy as np
import ufl
from ufl import div, inner, dot, dx, TrialFunction, max_value
from dolfinx import fem

from femcubes.core.base_model import BaseModel
from femcubes.core.utils.maths_utils import nabla_s, macPlus, macMinus
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from ufl import derivative, grad, conditional, le, lt
from mpi4py import MPI
from femcubes.core.utils.mpi_utils import mprint, comm, rank

class PhaseField(BaseModel):
    r"""Phase-field fracture model (AT1, AT2 and their higher-order variants).

    The phase-field method describes a crack by a damage field
    :math:`\phi\in[0,1]` (0 intact, 1 broken) that varies smoothly over a
    width set by the length scale :math:`\ell`, instead of by an explicit
    discontinuity. It regularizes the variational formulation of brittle
    fracture of :cite:t:`FrancfortMarigo1998`, in which cracks grow such that
    the sum of the elastic energy and the fracture energy, :math:`G_c` per
    unit crack area, is minimized, with the crack area approximated by the
    volume integral of a crack density :math:`\gamma(\phi)`
    :cite:p:`Ambrosio1990,Bourdin2000`. Crack nucleation, growth, branching
    and merging then follow from one partial differential equation for
    :math:`\phi`, without tracking crack faces.

    This model solves that equation. It is driven by a crack driving force
    field written by a solid model (``ElasticEnergy`` by default, see
    :ref:`solids-crack-driving-forces`), with the degradation function
    :math:`g(\phi) = \kappa + (1-\kappa)(1-\phi)^2` of the solids giving the
    driving term. The crack density selects the formulation. The quadratic
    ``AT2`` density :cite:p:`Bourdin2000,Miehe2010a` gives damage as soon as
    an energy-based driving force is non-zero, whereas with the linear
    ``AT1`` density and :math:`\phi\geq0` the material remains elastic until
    the driving force reaches a threshold :cite:p:`Pham2011`. For both, the
    stress at which a homogeneous bar fails scales as
    :math:`\sqrt{EG_c/\ell}`, so that :math:`\ell` can be chosen to give a
    prescribed tensile strength :cite:p:`Pham2011,Tanne2018` (see
    :func:`~femcubes.core.utils.maths_utils.ft_to_Gc`). ``HO-AT2`` uses the
    fourth-order crack density of :cite:t:`Borden2014`, which adds a
    second-gradient term, and ``HO-AT1`` applies the same gradient terms to
    the ``AT1`` density. The second gradient is evaluated through an
    auxiliary gradient field, tied to :math:`\nabla\phi` by a penalty, so
    that continuous Lagrange elements can be used. With the cohesive solids,
    the driving force is their strength potential, and the phase field
    degrades the strength of the material rather than its stiffness
    :cite:p:`Hageman2026b`.

    Crack growth is irreversible, so :math:`\phi` may not decrease. This is
    enforced by the history field of :cite:t:`Miehe2010a`, which replaces the
    driving force by its maximum over the loading history; by an augmented
    Lagrangian :cite:p:`Wheeler2014`, with the multiplier solved as an
    additional field (the default); or by a smooth penalty, with the penalty
    coefficient following from a tolerance on the violation as proposed by
    :cite:t:`Gerasimov2019`. An optional viscous term regularizes the
    evolution in time :cite:p:`Miehe2010a`. The optional mesh-size correction
    compensates the first-order element-size error
    :math:`G_c^\mathrm{num} = G_c(1 + c\,h/\ell)` of cohesive phase-field
    formulations whose inelastic strain localizes in one element layer
    :cite:p:`Feng2026AMR`, by reducing :math:`G_c` near the crack. Supports
    2-D, 3-D and axisymmetric meshes through the domain integrator.

    .. rubric:: Fields

    * ``phasefield``: :math:`\phi`, CG of order ``Field_Orders["phasefield"]``,
      solved in step ``Solution_Steps["phasefield"]``.
    * ``phasefieldOld``, ``phasefieldOldOld``: :math:`\phi_n` and
      :math:`\phi_{n-1}`.
    * ``phasefield_grad``, ``phasefield_gradOld``: auxiliary gradient
      :math:`\mathbf q\approx\nabla\phi` (vector CG, same order, solved in the
      same step) and its committed value; only for ``HO-AT1``/``HO-AT2``.
    * ``phasefield_Hist``, ``phasefield_HistOld``: history :math:`\mathcal H`
      and :math:`\mathcal H_n` (scalar quadrature); only for
      ``Irreversible_Method="Hist"``, which is available for ``AT2`` and
      ``HO-AT2`` only (it does not keep :math:`\phi\geq0`, which ``AT1`` needs).
    * ``LMultiplier_PF``, ``LMultiplier_PFOld``: normalized multiplier
      :math:`\hat\lambda` (CG of order ``Field_Orders["phasefield"] - 1``,
      so the order must be at least 2, solved in the same step) and its
      committed value; only for ``Irreversible_Method="AL"`` (default).
    * ``Gc_eff``: corrected toughness, DG0; only with ``Gc_Mesh_Correction``.
    * ``CDF_field`` (default ``ElasticEnergy``): read only, the driving force
      :math:`\Psi_\mathrm{CDF}` written by a solid model; with the
      correction also its ``Source_Field`` (default ``etaNorm``).

    .. rubric:: Weak form

    With test function :math:`\delta\phi`, the contribution to the residual
    of ``phasefield`` is

    .. math::

        R_\phi = \int_\Omega G_c\,\gamma'(\phi;\delta\phi)\,\mathrm{d}x
        - \int_\Omega 2(1-\kappa)(1-\phi)\,\Psi\,\delta\phi\,\mathrm{d}x
        + \int_\Omega \eta_d\,\delta\phi\,\frac{\phi-\phi_n}{\Delta t}\,\mathrm{d}x
        + R_\mathrm{irr},

    with the crack-density derivative

    .. math::

        \gamma'(\phi;\delta\phi) =
        \begin{cases}
        \frac{3}{8\ell}\bigl(\delta\phi + 2\ell^2\nabla\delta\phi\cdot\nabla\phi\bigr) & \text{AT1},\\
        \frac{1}{2\ell}\bigl(2\phi\,\delta\phi + 2\ell^2\nabla\delta\phi\cdot\nabla\phi\bigr) & \text{AT2},
        \end{cases}

    the viscous term only for ``pf_damping`` :math:`\eta_d>0`, and
    :math:`G_c` replaced by the field ``Gc_eff`` with ``Gc_Mesh_Correction``.
    The driving force is :math:`\Psi = \mathcal H` for ``"Hist"`` and
    :math:`\Psi = \Psi_\mathrm{CDF}` otherwise; with
    ``Negative_Time_Damage=False`` (default) :math:`\Psi = 0` while
    :math:`t+\Delta t\le10^{-9}`. For ``HO-AT1`` and ``HO-AT2``, with
    :math:`c_w = 3/(8\ell)` and :math:`1/(2\ell)` respectively and test
    function :math:`\delta\mathbf q`, the first integral is replaced by

    .. math::

        R_{\phi,\mathbf q} = \int_\Omega G_c\,c_w\Bigl[
        \delta\phi\ (\text{HO-AT1}) \text{ or } 2\phi\,\delta\phi\ (\text{HO-AT2})
        + \ell^2\,\delta\mathbf q\cdot\mathbf q
        + \tfrac{\ell^4}{8}\nabla\delta\mathbf q:\nabla\mathbf q
        + \ell^2(\delta\mathbf q-\nabla\delta\phi)\cdot(\mathbf q-\nabla\phi)\Bigr]\mathrm{d}x

    where the last term is the penalty that ties :math:`\mathbf q` to
    :math:`\nabla\phi`. The irreversibility term is

    * ``"Hist"``: :math:`R_\mathrm{irr} = 0`; :math:`\mathcal H` is updated
      outside the weak form.
    * ``"AL"``: with :math:`A = \hat\lambda + c_\mathrm{AS}(\phi - \tilde\phi_n)`,
      :math:`\tilde\phi_n = \min(\phi_n,1)` set to 0 where
      :math:`\phi_n<0.1`, and :math:`c_\mathrm{AS}` =
      ``Irreversibility_Active_Set_Parameter``,

      .. math::

          R_\mathrm{irr} = \int_\Omega \frac{G_c}{\ell}\langle A\rangle_-\,\delta\phi\,\mathrm{d}x,
          \qquad
          R_{\hat\lambda} = \int_\Omega \frac{G_c}{\ell\,c_\mathrm{AS}}
          \bigl(\langle A\rangle_- - \hat\lambda\bigr)\,\delta\hat\lambda\,\mathrm{d}x .

    * ``"Penalty"``: with the violation :math:`w = \phi_n - \phi` and the
      :math:`C^2` positive part :math:`P(w)` (0 for :math:`w<0`,
      :math:`w^3/s^2 - w^4/(2s^3)` for :math:`0\le w<s`, :math:`w - s/2`
      beyond, :math:`s` = ``Irreversibility_Smoothing``),

      .. math::

          R_\mathrm{irr} = -\int_\Omega c_\mathrm{pen}\frac{G_c}{\ell}\,P(w)\,P'(w)\,\delta\phi\,\mathrm{d}x,
          \qquad
          c_\mathrm{pen} =
          \begin{cases}
          27/(64\,\mathrm{TOL}^2) & \text{AT1, HO-AT1},\\
          1/\mathrm{TOL}^2 - 1 & \text{AT2, HO-AT2},
          \end{cases}

      with :math:`\mathrm{TOL}` = ``Irreversibility_Tolerance`` unless
      ``Irreversibility_Penalty_Factor`` sets :math:`c_\mathrm{pen}`.

    The Jacobian is consistent at fixed :math:`\Psi` (the active sets of the
    AL and penalty terms are differentiated piecewise).

    .. rubric:: Outside the weak form

    * ``"Hist"``: before every assembly
      :math:`\mathcal H = \max(\Psi_\mathrm{CDF}, \mathcal H_n)` (only
      :math:`\mathcal H_n` while :math:`t+\Delta t\le10^{-9}` and
      ``Negative_Time_Damage=False``).
    * ``Gc_Mesh_Correction``: once per time step and after every remesh,
      :math:`G_c^\mathrm{eff} = G_c/(1 + c\,\bar h/\ell)` per cell, where
      :math:`\bar h` is the mean or minimum diameter of the source cells
      (cells whose maximum of ``Source_Field`` exceeds ``Source_Threshold``)
      within ``Search_Radius`` :math:`\cdot\,\ell` of the cell midpoint; cells
      without a source cell nearby keep :math:`G_c`. ``Element_Size`` replaces
      :math:`\bar h` by one value everywhere. A spatially varying
      :math:`G_c` (:doc:`../glaciology/ice_phasefield`) is interpolated per
      cell first. Full domain only.
    * ``commit`` copies :math:`\phi_n\to\phi_{n-1}`, :math:`\phi\to\phi_n`,
      and ``phasefield_grad``, ``phasefield_Hist`` or ``LMultiplier_PF`` to
      their ``*Old`` fields; ``reset_step`` restores them.
    * Global measures: ``Crack_Length`` :math:`=\int_\Omega\gamma(\phi)\,\mathrm dx`
      with :math:`\gamma = \frac{3}{8\ell}(\phi + \ell^2|\nabla\phi|^2)` (AT1),
      :math:`\frac{1}{2\ell}(\phi^2 + \ell^2|\nabla\phi|^2)` (AT2), and
      :math:`\frac{\ell^2}{2}|\mathbf q|^2 + \frac{\ell^4}{16}|\nabla\mathbf q|^2`
      in place of :math:`\ell^2|\nabla\phi|^2` for the HO variants;
      ``Crack_Length_Change`` is :math:`|L - L_n|`.
    * No Dirichlet conditions (see :doc:`no_phase_field`).

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Field_Orders = {"phasefield": 2}      # REQUIRED: order of phasefield (>= 2 for "AL")
            Solution_Steps = {"phasefield": 1}    # REQUIRED: staggered step solving phasefield
            Materials = {
                "Ice": {
                    "Gc": 1.0,               # REQUIRED: fracture toughness G_c (J/m^2)
                    "ell": 0.1,              # REQUIRED: phase-field length scale (m)
                    "kappa": 1.0e-3,         # OPTIONAL (default 1e-3): residual stiffness in the driving term (-)
                    "PF_formulation": "AT2", # OPTIONAL (default "AT2"): "AT1", "AT2", "HO-AT1" or "HO-AT2"
                    "pf_damping": 0.0,       # OPTIONAL (default 0.0): viscous damping eta_d (J s/m^3)
                },
            }
            Models = {
                "PF": {
                    "Type": "Solid:PhaseField",       # REQUIRED
                    "Material": "Ice",                # REQUIRED: key in Materials
                    "CDF_field": "ElasticEnergy",     # OPTIONAL (default "ElasticEnergy"): driving force field
                    "Irreversible_Method": "AL",      # OPTIONAL (default "AL"): "AL", "Hist" (AT2, HO-AT2 only) or "Penalty"
                    "Irreversibility_Active_Set_Parameter": 1.0,  # OPTIONAL (default 1.0), AL only: c_AS > 0 (-)
                    # "Irreversibility_Tolerance": 1.0e-2,        # OPTIONAL (default 1e-2), Penalty only: 0 < TOL < 1 (-)
                    # "Irreversibility_Smoothing": 1.0e-3,        # OPTIONAL (default min(1e-3, 0.1 TOL)), Penalty only: 0 < s <= TOL (-)
                    # "Irreversibility_Penalty_Factor": 1.0e4,    # OPTIONAL (default from TOL), Penalty only: c_pen > 0 (-)
                    "Negative_Time_Damage": False,    # OPTIONAL (default False): drive damage also for t + dt <= 1e-9
                    "domain": "domain",               # OPTIONAL (default "domain", the full mesh): domain group
                    "Gc_Mesh_Correction": {           # OPTIONAL (default None, off); full domain only
                        "Coefficient": 0.5,           # REQUIRED: c > 0 (-)
                        "Source_Field": "etaNorm",    # OPTIONAL (default "etaNorm"): scalar field marking the crack core
                        "Source_Threshold": 1.0e-8,   # OPTIONAL (default 1e-8): cell maximum above this marks a source cell
                        "Search_Radius": 2.0,         # OPTIONAL (default 2.0): search radius in units of ell (-)
                        "Reduction": "mean",          # OPTIONAL (default "mean"): "mean" or "min"
                        # "Element_Size": 0.005,      # OPTIONAL (default None): fixed h instead of the search (m)
                    },
                },
            }
    """

    MODEL_TYPE = "Solid:PhaseField"
    
    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        
        self.My_Step = params.Solution_Steps["phasefield"]
        self.Irreversible_Method = self.params.Models[self.name].get("Irreversible_Method", "AL")
        valid_irreversibility_methods = {"AL", "Hist", "Penalty"}
        if self.Irreversible_Method not in valid_irreversibility_methods:
            raise ValueError(
                f"Unknown Irreversible_Method: {self.Irreversible_Method}. "
                f"Expected one of {sorted(valid_irreversibility_methods)}."
            )
        self.material = self.params.Models[self.name]["Material"]
        self.PF_Version = self.params.Materials[self.material].get("PF_formulation", "AT2")
        valid_formulations = {"AT1", "AT2", "HO-AT1", "HO-AT2"}
        if self.PF_Version not in valid_formulations:
            raise ValueError(f"Unknown PF_formulation: {self.PF_Version}")
        if self.PF_Version in ("AT1", "HO-AT1") and self.Irreversible_Method == "Hist":
            # The history field only makes the driving force non-decreasing; with the linear AT1 density
            # nothing then keeps phi >= 0, which the "AL" and "Penalty" methods enforce.
            raise ValueError(
                f"Irreversible_Method 'Hist' cannot be used with PF_formulation '{self.PF_Version}': it does not "
                "keep the phase field non-negative. Use 'AL' or 'Penalty'."
            )
        self.Higher_Order = self.PF_Version in ("HO-AT1", "HO-AT2")
        self.irreversibility_active_set_parameter = None
        self.irreversibility_tolerance = None
        self.irreversibility_smoothing = None
        self.irreversibility_penalty_factor = None
        if self.Irreversible_Method == "AL":
            self._configure_augmented_lagrangian()
        elif self.Irreversible_Method == "Penalty":
            self._configure_irreversibility_penalty()
        
        mesh.Add_Field("phasefield", "Phasefield [-]", "CG", params.Field_Orders["phasefield"], domain=self.domain)
        mesh.Add_Field("phasefieldOld", "Old Phasefield [-]", "CG", params.Field_Orders["phasefield"], domain=self.domain)
        mesh.Add_Field("phasefieldOldOld", "Old Old Phasefield [-]", "CG", params.Field_Orders["phasefield"], domain=self.domain)
        if self.Higher_Order:
            mesh.Add_Vector_Field("phasefield_grad","Phasefield Gradient Auxiliary [1/m]","CG",params.Field_Orders["phasefield"],params.dim, domain=self.domain)
            mesh.Add_Vector_Field("phasefield_gradOld","Old Phasefield Gradient Auxiliary [1/m]","CG",params.Field_Orders["phasefield"],params.dim, domain=self.domain)

        if self.Irreversible_Method == "Hist":
            mesh.Add_Quadrature_Field("phasefield_Hist", "Phasefield History [-]", (), domain=self.domain)
            mesh.Add_Quadrature_Field("phasefield_HistOld", "Old Phasefield History [-]", (), domain=self.domain)
        elif self.Irreversible_Method == "AL":
            mesh.Add_Field("LMultiplier_PF", "Normalized Lagrange Multiplier for Phasefield [-]", "CG", params.Field_Orders["phasefield"]-1, domain=self.domain)
            mesh.Add_Field("LMultiplier_PFOld", "Old Normalized Lagrange Multiplier for Phasefield [-]", "CG", params.Field_Orders["phasefield"]-1, domain=self.domain)
        
        # Register test functions (actual test functions will be retrieved when needed)
        mesh.Add_TestFunction("phasefield", self.My_Step)
        if self.Higher_Order:
            mesh.Add_Vector_TestFunction("phasefield_grad", params.dim, self.My_Step)
        if self.Irreversible_Method == "AL":
            mesh.Add_TestFunction("LMultiplier_PF", self.My_Step)
        
        # crack driving force field
        self.CDF_Name = self.params.Models[self.name].get("CDF_field", "ElasticEnergy")
        
        # material parameters
        self.damage_zero = self.params.Models[self.name].get("Negative_Time_Damage", False)
        self.ell = self.params.Materials[self.material]["ell"]
        self.Gc = self._get_fracture_toughness()
        self.damping = self.params.Materials[self.material].get("pf_damping", 0.0)
        self.kappa = self.params.Materials[self.material].get("kappa", 1e-3)
        self.gc_correction = self._configure_gc_mesh_correction()
        if self.gc_correction is not None:
            mesh.Add_Field("Gc_eff", "Mesh-corrected fracture toughness [J/m^2]", "DG", 0, domain=self.domain)
            self._set_uncorrected_toughness()
        self._gc_correction_key = None

        self.params.Global_Measures[f"Crack_Length"] = 0.0
        self.params.Global_Measures[f"Crack_Length_Change"] = 0.0
    
    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """
        Assemble the global stiffness matrix and force vector for phase-field fractures.
        """
        
        if (step != self.My_Step):
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)

        # Get fields
        phi = self.mesh.Get_Field("phasefield", step=self.My_Step)
        phiOld = self.mesh.Get_Field("phasefieldOld", step=self.My_Step)
        phi_t = self.mesh.Get_TestFunction("phasefield", self.My_Step)
        phi_tr = self.mesh.Get_Trial_Function("phasefield", self.My_Step)

        if self.Higher_Order:
            GradPhi = self.mesh.Get_Field("phasefield_grad", step=self.My_Step)
            GradPhi_t = self.mesh.Get_TestFunction("phasefield_grad", self.My_Step)
            GradPhi_tr = self.mesh.Get_Trial_Function("phasefield_grad", self.My_Step)
        
        if self.Irreversible_Method == "AL":
            L = self.mesh.Get_Field("LMultiplier_PF", step=self.My_Step)
            L_t = self.mesh.Get_TestFunction("LMultiplier_PF", self.My_Step)
            L_tr = self.mesh.Get_Trial_Function("LMultiplier_PF", self.My_Step)
        
        if (self.Irreversible_Method == "Hist"):
            Psi_Elastic = self.mesh.Get_Field("phasefield_Hist")
        else:
            Psi_Elastic = self.mesh.Get_Field(self.CDF_Name)
        if not self.damage_zero:
            # Keep activation symbolic so cached forms follow physics' time constants.
            Psi_Elastic = conditional(le(time_constant + dt_constant, 1.0e-9), 0.0, Psi_Elastic)

        # Crack capacity terms
        if self.Higher_Order:
            _, dGamma, ddGamma = GetCrackDensity(self.ell,phi,dp=GradPhi,model=self.PF_Version,p_t=phi_t,p_tr=phi_tr,dp_t=GradPhi_t,dp_tr=GradPhi_tr)
        else:
            _, dGamma, ddGamma = GetCrackDensity(self.ell,phi,model=self.PF_Version,p_t=phi_t,p_tr=phi_tr)

        Gc = self._fracture_toughness_coefficient()
        F = Gc * dGamma * dx
        K = Gc * ddGamma * dx

        ## enforce gradient-auxiliary constraint for higher-order formulations
        if self.Higher_Order:
            if self.PF_Version == "HO-AT1":
                pre_fac = 3.0 / (8.0 * self.ell)
            else: # HO-AT2
                pre_fac = 1.0 / (2.0 * self.ell)

            # Symmetric penalty on q - grad(phi).
            F += Gc * pre_fac * self.ell**2 * inner(GradPhi_t - grad(phi_t), GradPhi - grad(phi)) * dx
            K += Gc * pre_fac * self.ell**2 * inner(GradPhi_t - grad(phi_t), GradPhi_tr - grad(phi_tr)) * dx
        
        # Crack driving force term
        F += -2.0*(1-self.kappa)*phi_t*(1-phi)*Psi_Elastic*dx
        K += -2.0*(1-self.kappa)*phi_t*(-phi_tr)*Psi_Elastic*dx
            
        # Damping term
        if self.damping > 0.0:
            F += self.damping*inner(phi_t, (phi - phiOld)/dt_constant)*dx
            K += self.damping*inner(phi_t, phi_tr/dt_constant)*dx

        if self.Irreversible_Method == "AL":
            phiOldLim = conditional(lt(phiOld, 1.0), phiOld, 1.0)
            phiOldLim = conditional(lt(phiOldLim, 0.1), 0.0, phiOldLim)
            irrCondition = phi - phiOldLim
            active_set_parameter = self.irreversibility_active_set_parameter
            phase_scale = Gc / self.ell

            AugLagr = L + active_set_parameter*irrCondition
            AugMinus = macMinus(AugLagr)

            active_branch = conditional(le(AugLagr, 0.0), 1.0, 0.0)
            dAug_dL = active_branch
            dAug_dphi = active_set_parameter*active_branch
            
            # Damage equation terms
            F += phase_scale/active_set_parameter*inner(phi_t, dAug_dphi*AugMinus)*dx
            K += phase_scale/active_set_parameter*inner(phi_t, dAug_dphi*dAug_dphi*phi_tr + dAug_dphi*dAug_dL*L_tr)*dx
            
            # Lagrange multiplier equation terms
            F += phase_scale/active_set_parameter*inner(L_t, dAug_dL*AugMinus - L)*dx
            K += phase_scale/active_set_parameter*inner(L_t, dAug_dL*dAug_dphi*phi_tr + dAug_dL*dAug_dL*L_tr - L_tr)*dx

        elif self.Irreversible_Method == "Penalty":
            violation = phiOld - phi
            penalty, dpenalty, ddpenalty = _smooth_positive_part(violation, self.irreversibility_smoothing)
            penalty_coefficient = self.irreversibility_penalty_factor * Gc / self.ell

            F += -penalty_coefficient * penalty * dpenalty * phi_t * dx
            K += penalty_coefficient * (dpenalty**2 + penalty * ddpenalty) * phi_t * phi_tr * dx

        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        if step != self.My_Step:
            return False
        time = self._weak_constant_value(time_constant)
        dt = self._weak_constant_value(dt_constant)
        if self.gc_correction is not None:
            self._update_gc_mesh_correction(time, dt)
        if self.Irreversible_Method != "Hist":
            return False
        if time + dt > 1.0e-9 or self.damage_zero:
            history = max_value(
                self.mesh.Get_Field(self.CDF_Name),
                self.mesh.Get_Field("phasefield_HistOld"),
            )
        else:
            history = self.mesh.Get_Field("phasefield_HistOld")
        self.mesh.Set_From_Expression("phasefield_Hist", history, domain=self.domain)
        return False
    
    def Evaluate_Crack_Length(self, phasefield, Oldphasefield) -> float:
        """
        Evaluate the total crack length in the domain based on the phase-field variable.
        
        Args:
            phasefield: The phase-field variable representing damage.
            
        Returns:
            Total crack length (float).
        """
        # Crack density function
        if self.Higher_Order:
            q = self.mesh.Get_Field("phasefield_grad", step=self.My_Step)
            qOld = self.mesh.Get_Field("phasefield_gradOld")
            gamma, _, _ = GetCrackDensity(self.ell, phasefield, dp=q, model=self.PF_Version)
            gamma_old, _, _ = GetCrackDensity(self.ell, Oldphasefield, dp=qOld, model=self.PF_Version)
        else:
            gamma, _, _ = GetCrackDensity(self.ell, phasefield, model=self.PF_Version)
            gamma_old, _, _ = GetCrackDensity(self.ell, Oldphasefield, model=self.PF_Version)
        
        dx = self.mesh.Get_Domain_Integrator(self.domain)
        total_crack_length = fem.assemble_scalar(self.mesh.Compile_Form(gamma*dx))
        old_crack_length = fem.assemble_scalar(self.mesh.Compile_Form(gamma_old*dx))
        
        # sync across processes
        total_crack_length = comm.allreduce(total_crack_length, op=MPI.SUM)
        old_crack_length = comm.allreduce(old_crack_length, op=MPI.SUM)
        
        self.params.Global_Measures[f"Crack_Length"] = total_crack_length
        self.params.Global_Measures[f"Crack_Length_Change"] = abs(total_crack_length - old_crack_length)
        
        #mprint(f"Total Crack Length: {total_crack_length:.6f}, Change in Crack Length: {total_crack_length - old_crack_length:.6f}")
        return total_crack_length
    
    def Update_Global_Measures(self) -> None:
        phi = self.mesh.Get_Field("phasefield", step=self.My_Step)
        phiOld = self.mesh.Get_Field("phasefieldOld", step=self.My_Step)
        self.Evaluate_Crack_Length(phi, phiOld)
    
    def initialize_fields(self) -> None:
        """
        Set all fields for this model to zero.
        """
        fields_to_zero = ["phasefield", "phasefieldOld", "phasefieldOldOld"]
        if self.Higher_Order:
            fields_to_zero.extend(["phasefield_grad", "phasefield_gradOld"])

        for field_name in fields_to_zero:
            self.mesh.Zero_Field(field_name, domain=self.domain)
            
        if self.Irreversible_Method == "Hist":
            self.mesh.Zero_Field("phasefield_Hist", domain=self.domain)
            self.mesh.Zero_Field("phasefield_HistOld", domain=self.domain)
        elif self.Irreversible_Method == "AL":
            self.mesh.Zero_Field("LMultiplier_PF", domain=self.domain)
            self.mesh.Zero_Field("LMultiplier_PFOld", domain=self.domain)
        if self.gc_correction is not None:
            self._set_uncorrected_toughness()
            self._gc_correction_key = None

    def Post_Remesh(self) -> None:
        """Recompute the mesh-corrected toughness on the new mesh before the next assembly."""
        self._gc_correction_key = None

    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """
        # Copy current fields to old fields
        self.mesh.Copy_Field("phasefieldOld", "phasefieldOldOld", domain=self.domain)
        self.mesh.Copy_Field("phasefield", "phasefieldOld", domain=self.domain)
        if self.Higher_Order:
            self.mesh.Copy_Field("phasefield_grad", "phasefield_gradOld", domain=self.domain)
        if self.Irreversible_Method == "Hist":
            self.mesh.Copy_Field("phasefield_Hist", "phasefield_HistOld", domain=self.domain)
        elif self.Irreversible_Method == "AL":
            self.mesh.Copy_Field("LMultiplier_PF", "LMultiplier_PFOld", domain=self.domain)
    
    def reset_step(self):
        """Restore the last committed phase-field state after a rejected step."""
        self.mesh.Copy_Field("phasefieldOld", "phasefield", domain=self.domain)
        if self.Higher_Order:
            self.mesh.Copy_Field("phasefield_gradOld", "phasefield_grad", domain=self.domain)
        if self.Irreversible_Method == "Hist":
            self.mesh.Copy_Field("phasefield_HistOld", "phasefield_Hist", domain=self.domain)
        elif self.Irreversible_Method == "AL":
            self.mesh.Copy_Field("LMultiplier_PFOld", "LMultiplier_PF", domain=self.domain)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_fracture_toughness(self): # Note: Declared as method, as it is overwritten by ice-specific models
        """PRIVATE: Return the material fracture toughness used by the model."""
        return self.params.Materials[self.material]["Gc"]

    def _fracture_toughness_coefficient(self):
        """PRIVATE: Return the toughness used in the weak form: a float, or the ``Gc_eff`` field."""
        if self.gc_correction is None:
            return self.Gc
        return self.mesh.Get_Field("Gc_eff")

    def _set_uncorrected_toughness(self) -> None:
        """PRIVATE: set ``Gc_eff`` to G_c: a constant, or a spatially varying G_c (UFL) per cell."""
        if isinstance(self.Gc, ufl.core.expr.Expr):
            self.mesh.Set_From_Expression("Gc_eff", self.Gc, domain=self.domain)
        else:
            self.mesh.Constant_Field("Gc_eff", self.Gc, domain=self.domain)

    def _configure_gc_mesh_correction(self):
        """PRIVATE: Validate the ``Gc_Mesh_Correction`` model setting; ``None`` when disabled."""
        settings = self.params.Models[self.name].get("Gc_Mesh_Correction", None)
        if settings is None or settings is False:
            return None
        if not isinstance(settings, dict) or "Coefficient" not in settings:
            raise ValueError("Gc_Mesh_Correction must be a dict with at least a 'Coefficient' entry.")
        known = {"Coefficient", "Source_Field", "Source_Threshold", "Search_Radius", "Reduction", "Element_Size"}
        unknown = set(settings) - known
        if unknown:
            raise ValueError(f"Unknown Gc_Mesh_Correction setting(s): {sorted(unknown)}. Expected a subset of {sorted(known)}.")
        config = {
            "Coefficient": self._positive_finite_setting("Gc_Mesh_Correction['Coefficient']", settings["Coefficient"]),
            "Source_Field": settings.get("Source_Field", "etaNorm"),
            "Source_Threshold": float(settings.get("Source_Threshold", 1.0e-8)),
            "Search_Radius": self._positive_finite_setting("Gc_Mesh_Correction['Search_Radius']", settings.get("Search_Radius", 2.0)),
            "Reduction": settings.get("Reduction", "mean"),
            "Element_Size": settings.get("Element_Size", None),
        }
        if config["Reduction"] not in ("mean", "min"):
            raise ValueError("Gc_Mesh_Correction['Reduction'] must be 'mean' or 'min'.")
        if config["Element_Size"] is not None:
            config["Element_Size"] = self._positive_finite_setting("Gc_Mesh_Correction['Element_Size']", config["Element_Size"])
        if not isinstance(config["Source_Field"], str) or not config["Source_Field"]:
            raise ValueError("Gc_Mesh_Correction['Source_Field'] must be a field name.")
        if self.domain != "domain":
            raise NotImplementedError("Gc_Mesh_Correction is only implemented for phase fields on the full domain.")
        if config["Element_Size"] is None:
            mprint(f"\tGc mesh correction: Gc/(1 + {config['Coefficient']} h/ell), h = {config['Reduction']} diameter "
                   f"of cells with {config['Source_Field']} > {config['Source_Threshold']:.1e} within {config['Search_Radius']} ell")
        else:
            factor = 1.0 / (1.0 + config["Coefficient"] * config["Element_Size"] / self.ell)
            mprint(f"\tGc mesh correction: Gc/(1 + {config['Coefficient']} h/ell) with h = {config['Element_Size']:.4e} "
                   f"everywhere (factor {factor:.4f})")
        return config

    def _update_gc_mesh_correction(self, time: float, dt: float) -> None:
        """PRIVATE: Refresh ``Gc_eff`` once per (time, dt, mesh epoch), following Feng et al. (2026)."""
        key = (time, dt, getattr(self.mesh, "remesh_epoch", 0))
        if key == self._gc_correction_key:
            return
        self._gc_correction_key = key
        config = self.gc_correction
        field = self.mesh.Get_Field("Gc_eff")
        cell_dofs = field.function_space.dofmap.list[:, 0]
        msh = field.function_space.mesh
        tdim = msh.topology.dim
        index_map = msh.topology.index_map(tdim)
        n_local = index_map.size_local + index_map.num_ghosts
        cells = np.arange(n_local, dtype=np.int32)
        factor = np.ones(n_local)

        if config["Element_Size"] is not None:
            factor[:] = 1.0 / (1.0 + config["Coefficient"] * config["Element_Size"] / self.ell)
        else:
            from dolfinx import cpp as _cpp
            from dolfinx.mesh import compute_midpoints
            from scipy.spatial import cKDTree

            diameters = np.asarray(_cpp.mesh.h(msh._cpp_object, tdim, cells))
            midpoints = np.asarray(compute_midpoints(msh, tdim, cells))[:, :tdim]
            # Source (crack-core) cells: owned cells whose source field exceeds the threshold.
            source_max = self._cell_max_of_field(config["Source_Field"], n_local)
            owned_sources = np.flatnonzero(source_max[: index_map.size_local] > config["Source_Threshold"])
            source_points = np.concatenate(comm.allgather(midpoints[owned_sources]), axis=0)
            source_sizes = np.concatenate(comm.allgather(diameters[owned_sources]), axis=0)
            if len(source_sizes) > 0:
                radius = config["Search_Radius"] * self.ell
                tree = cKDTree(source_points)
                neighbours = tree.query_ball_point(midpoints, r=radius)
                reduce = np.mean if config["Reduction"] == "mean" else np.min
                h_bar = np.array([reduce(source_sizes[idx]) if idx else np.nan for idx in neighbours])
                near = np.isfinite(h_bar)
                factor[near] = 1.0 / (1.0 + config["Coefficient"] * h_bar[near] / self.ell)
        self._set_uncorrected_toughness()
        field.x.array[cell_dofs[cells]] *= factor

    def _cell_max_of_field(self, field_name: str, n_cells: int) -> np.ndarray:
        """PRIVATE: Cell-wise maximum of a scalar (CG, DG or quadrature) field over its cell DOFs."""
        source = self.mesh.Get_Field(field_name)
        space = source.function_space
        block_size = space.dofmap.index_map_bs
        if block_size != 1 or space.value_shape not in ((), (1,)):
            raise ValueError(f"Gc_Mesh_Correction source field '{field_name}' must be scalar.")
        dofs = space.dofmap.list[:n_cells]
        return source.x.array[dofs].max(axis=1)

    def _configure_augmented_lagrangian(self) -> None:
        """PRIVATE: Validate and store the normalized active-set parameter."""
        model_settings = self.params.Models[self.name]
        self.irreversibility_active_set_parameter = self._positive_finite_setting(
            "Irreversibility_Active_Set_Parameter",
            model_settings.get("Irreversibility_Active_Set_Parameter", 1.0),
        )

    def _configure_irreversibility_penalty(self) -> None:
        """PRIVATE: Validate and store the smooth primal-penalty parameters."""
        model_settings = self.params.Models[self.name]
        tolerance = self._positive_finite_setting("Irreversibility_Tolerance",model_settings.get("Irreversibility_Tolerance", 1.0e-2))
        if tolerance >= 1.0:
            raise ValueError("Irreversibility_Tolerance must be smaller than 1.")

        default_smoothing = min(1.0e-3, 0.1 * tolerance)
        smoothing = self._positive_finite_setting("Irreversibility_Smoothing", model_settings.get("Irreversibility_Smoothing", default_smoothing))
        if smoothing > tolerance:
            raise ValueError("Irreversibility_Smoothing must not exceed Irreversibility_Tolerance.")

        penalty_factor = model_settings.get("Irreversibility_Penalty_Factor", None)
        if penalty_factor is None:
            if self.PF_Version in ("AT1", "HO-AT1"):
                penalty_factor = 27.0 / (64.0 * tolerance**2)
            else:
                penalty_factor = 1.0 / tolerance**2 - 1.0
        penalty_factor = self._positive_finite_setting("Irreversibility_Penalty_Factor", penalty_factor)

        self.irreversibility_tolerance = tolerance
        self.irreversibility_smoothing = smoothing
        self.irreversibility_penalty_factor = penalty_factor

    @staticmethod
    def _positive_finite_setting(name, value) -> float:
        """PRIVATE: Return one positive finite scalar model setting."""
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite scalar.")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite scalar.") from exc
        if not 0.0 < value < float("inf"):
            raise ValueError(f"{name} must be a positive finite scalar.")
        return value


def _smooth_positive_part(value, smoothing):
    """PRIVATE: Return a C2 positive-part approximation and its derivatives."""
    transition_value = value**3 / smoothing**2 - value**4 / (2.0 * smoothing**3)

    transition_derivative = 3.0 * value**2 / smoothing**2 - 2.0 * value**3 / smoothing**3

    transition_second_derivative = 6.0 * value / smoothing**2 - 6.0 * value**2 / smoothing**3

    positive_part = conditional(lt(value, 0.0), 0.0, conditional(lt(value, smoothing), transition_value, value - smoothing / 2.0))
    first_derivative = conditional(lt(value, 0.0), 0.0, conditional(lt(value, smoothing), transition_derivative, 1.0))
    second_derivative = conditional(lt(value, 0.0), 0.0, conditional(lt(value, smoothing), transition_second_derivative, 0.0))
    return positive_part, first_derivative, second_derivative


def GetCrackDensity(ell, p, dp=None, model="AT2", p_t=None, p_tr=None, dp_t=None, dp_tr=None):
    """Return crack-density and optional weak-form derivatives.

    Parameters
    ----------
    ell : float
        Regularisation length.
    p : UFL expression
        Phase-field variable.
    dp : UFL expression, optional
        Auxiliary gradient variable used by higher-order formulations.
    model : {"AT1", "AT2", "HO-AT1", "HO-AT2"}
        Phase-field formulation.
    p_t, p_tr : UFL expression, optional
        Test and trial functions for ``p``.
    dp_t, dp_tr : UFL expression, optional
        Test and trial functions for ``dp``.

    Returns
    -------
    tuple
        ``(Gamma, F, K)``, where ``Gamma`` is the crack
        density, ``F``/``K`` are the residual and tangent terms
    """
    Gamma = None
    F = None
    K = None
    
    Include_FK = (p_t is not None) and (p_tr is not None)
    if model in ("HO-AT1", "HO-AT2"):
        if dp is None:
            raise ValueError("Higher-order crack density requires the auxiliary gradient field 'dp'.")
        Include_FK = Include_FK and (dp_t is not None) and (dp_tr is not None)

    if model == "AT1":
        pre_fac = 3 / (8.0 * ell)
        Gamma = pre_fac * (p + ell**2 * inner(grad(p), grad(p)))
        if Include_FK:
            F = pre_fac * (p_t + 2.0 * ell**2 * inner(grad(p_t), grad(p)))
            K = pre_fac * (2.0 * ell**2 * inner(grad(p_t), grad(p_tr)))

    elif model == "AT2":
        pre_fac = 1 / (2.0 * ell)
        Gamma = pre_fac * (p**2 + ell**2 * inner(grad(p), grad(p)))
        if Include_FK:
            F = pre_fac * (2.0 * p_t * p + 2.0 * ell**2 * inner(grad(p_t), grad(p)))
            K = pre_fac * (2.0 * p_t * p_tr + 2.0 * ell**2 * inner(grad(p_t), grad(p_tr)))

    elif model == "HO-AT1":
        pre_fac = 3 / (8.0 * ell)
        Gamma = pre_fac * (p + ell**2 / 2 * inner(dp, dp) + ell**4 / 16 * inner(grad(dp), grad(dp)))
        if Include_FK:
            F = pre_fac * p_t
            F += pre_fac * (ell**2 * inner(dp_t, dp) + ell**4 / 8 * inner(grad(dp_t), grad(dp)))
            K = pre_fac * (ell**2 * inner(dp_t, dp_tr) + ell**4 / 8 * inner(grad(dp_t), grad(dp_tr)))

    elif model == "HO-AT2":
        pre_fac = 1 / (2.0 * ell)
        Gamma = pre_fac * (p**2 + ell**2 / 2 * inner(dp, dp) + ell**4 / 16 * inner(grad(dp), grad(dp)))
        if Include_FK:
            F = pre_fac * (2.0 * p_t * p)
            F += pre_fac * (ell**2 * inner(dp_t, dp) + ell**4 / 8 * inner(grad(dp_t), grad(dp)))
            K = pre_fac * (2.0 * p_t * p_tr)
            K += pre_fac * (ell**2 * inner(dp_t, dp_tr) + ell**4 / 8 * inner(grad(dp_t), grad(dp_tr)))
            
    else:
        raise ValueError(f"Unknown phase-field formulation: {model}")

    return Gamma, F, K
    
    
