import math
from dataclasses import dataclass, field

import numpy as np
from ufl import inner, ln, max_value, derivative
from dolfinx import fem

from femcubes.core.base_model import BaseModel
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.electrochemistry._activity import ActivityModel
from ufl import grad
from petsc4py import PETSc


_REACTION_TYPES = ("Kinetic", "Equilibrium", "Precipitation", "Degassing")


@dataclass
class _VolumeReaction:
    """PRIVATE: Parsed volume reaction.

    ``stoichiometry`` is the net production of each field per unit
    (forward) rate; ``orders_in``/``orders_out`` are the activity orders of
    the forward and backward rates (Kinetic, Equilibrium) or the exponents
    of the saturation ratio (Precipitation, in ``orders_in``).
    """

    name: str
    type: str
    stoichiometry: dict
    orders_in: dict = field(default_factory=dict)
    orders_out: dict = field(default_factory=dict)
    constants: dict = field(default_factory=dict)


class Electrolyte(BaseModel):
    r"""Ionic transport in an electrolyte (Nernst-Planck with electroneutrality), with volume reactions.

    The model describes the transport of dissolved species in an electrolyte
    and the reactions between them. Each dissolved species is transported by
    diffusion and by migration in the electric field, following the
    Nernst-Planck equation, with an additional flux along gradients of its
    activity coefficient in non-ideal solutions :cite:p:`Newman2021`. The
    electrolyte potential follows from electroneutrality, the condition that
    the net charge of the solution vanishes at every point, which holds
    outside the thin double layers at charged surfaces :cite:p:`Newman2021`.
    The capacity, reaction and electroneutrality terms are integrated with a
    lumped scheme; lumped integration stabilizes simulations of fast
    metal-electrolyte reactions :cite:p:`Hageman2023a`.

    Every species :math:`\pi` of ``params.Species`` has a concentration
    :math:`C_\pi` (mol/m^3) and the electrolyte potential is :math:`\varphi`
    (V). Immobile solids (``"Solid": True``, e.g. precipitates, amount per
    electrolyte volume) are not transported and only take part in
    precipitation reactions. Supports 2-D, 3-D and
    axisymmetric meshes. Boundary and interface models acting on these
    fields: :doc:`open_boundary`, :doc:`prescribed_flux`,
    :doc:`metal_electrolyte_interface` and :doc:`passivating_surface`.

    Symbols used by all electrochemistry models: :math:`F` Faraday constant,
    :math:`R` gas constant, :math:`T` temperature, :math:`f = F/(RT)`,
    :math:`z_\pi` charge number, :math:`D_\pi` diffusivity,
    :math:`a_\pi = \gamma_\pi C_\pi / C_{ref}` activity with activity
    coefficient :math:`\gamma_\pi` (see :doc:`activity`), :math:`E_m` metal
    potential, and lumped weights :math:`W_i = \int_\Omega N_i\,\mathrm{d}x`
    of node :math:`i` (shape function :math:`N_i`; the radius-weighted
    integral on axisymmetric meshes). Nodal sums
    :math:`\sum_i \delta C_{\pi,i} W_i(\dots)_i` evaluate the bracket with
    the nodal values of node :math:`i`.

    .. rubric:: Fields

    * ``<species>`` (one per entry of ``params.Species``, named after it):
      concentration :math:`C_\pi`, CG of order ``Field_Orders["C"]``,
      solved in step ``Solution_Steps["C"]``.
    * ``E``: electrolyte potential :math:`\varphi`, CG of order
      ``Field_Orders["E"]``, solved in the same step
      (``Solution_Steps["E"]`` must equal ``Solution_Steps["C"]``; the
      lumped terms also need ``Field_Orders["E"] == Field_Orders["C"]``).
    * ``<species>Old``, ``EOld``: committed values :math:`C_{\pi,n}`,
      :math:`\varphi_n`.
    * ``pH``: output, CG of order ``Field_Orders["C"]``, only when ``"H+"``
      is a species: :math:`\mathrm{pH} = 3 - \log_{10}(\max(C_{H^+}, 10^{-14})) - \log_{10}\gamma_{H^+}`
      (concentrations in mol/m^3).

    .. rubric:: Weak form

    With :math:`\Omega` the domain group named by ``domain``, the residual
    of each dissolved species :math:`\pi` is

    .. math::

        R_{C_\pi} = \int_\Omega \nabla\delta C_\pi \cdot D_\pi\bigl(\nabla C_\pi
        + C_\pi\nabla\ln\gamma_\pi + z_\pi f\,C_\pi\nabla\varphi\bigr)\,\mathrm{d}x
        + \sum_i \delta C_{\pi,i}\,W_i\Bigl(\frac{C_{\pi,i}-C_{\pi,n,i}}{\Delta t} - P_{\pi,i}\Bigr),

    that of an immobile solid has only the nodal sum, and the potential
    has the lumped electroneutrality condition

    .. math::

        R_{\varphi} = \sum_i \delta\varphi_i\,W_i \sum_{\pi\ \mathrm{dissolved}} z_\pi \max(C_{\pi,i}, 0).

    The integral is assembled from UFL, the nodal sums directly
    (:meth:`Add_KF`, with :math:`\Delta t` = ``params.dt``). The term
    :math:`C_\pi\nabla\ln\gamma_\pi` exists only for non-ideal activity
    models and species with :math:`\gamma_\pi \not\equiv 1`; with
    ``"Activity_Tangent": "lagged"`` its Jacobian keeps only the derivative
    of the factor :math:`C_\pi` (the residual is unchanged; the full
    derivative couples every pair of species and dominates the assembly
    time with many species and SIT). All other Jacobian terms are the
    consistent derivatives. :math:`P_\pi` is the net production by the
    volume reactions (mol/(m^3 s)),
    :math:`P_\pi = \sum_r \nu_{\pi r}\,r_r` with stoichiometry
    :math:`\nu_{\pi r}` (positive for products), evaluated with
    activities of concentrations floored at ``Concentration_Floor``
    :math:`C_{floor}`. Below the floor a rate does not change with that
    concentration, so a Newton update that takes a trace species below it
    (e.g. OH- where a pit acidifies) loses the restoring effect of a fast
    reaction, and the next update diverges; for species that fall by orders
    of magnitude, choose the floor far below their smallest value and list
    them in the solver option ``Positive_Fields``. The reaction types (``params.Volume_Reactions[name]["Type"]``):

    * ``"Kinetic"`` (default):
      :math:`r = k_{fwd}\prod_{in} a^{m} - k_{rev}\prod_{out} a^{m}`, orders
      :math:`m` from ``orders_in``/``orders_out``, default the stoichiometry.
    * ``"Equilibrium"``: penalty
      :math:`r = k_{dummy}\,G` with :math:`G = \prod_{in} a^\nu - K_{eq}\prod_{out} a^\nu`;
      with ``"Normalised": True``,
      :math:`r = k_{eq}\,C_{ref}\,G / \sum_\pi |\partial G/\partial a_\pi|`,
      which relaxes the equilibrium at the rate :math:`k_{eq}` (1/s)
      whatever the size of :math:`K_{eq}` and the concentrations.
    * ``"Precipitation"`` of the solid ``"solid"``, :math:`r` positive for
      precipitation:
      :math:`r = -k_p\bigl(A + B - \sqrt{A^2 + B^2 + 2\delta^2}\bigr)`,
      :math:`A = C_s`, :math:`B = -C_T\ln\Omega`, :math:`\delta = \varepsilon C_T`,
      the smoothed Fischer-Burmeister function of
      :math:`C_s \geq 0`, :math:`\Omega \leq 1`, :math:`C_s\ln\Omega = 0`.
      The saturation ratio is :math:`\Omega = \prod a^\nu / K_s` over the
      dissolution products (``species``/``stoich``, negative for consumed
      species such as OH-) and :math:`C_T` the concentration of the first
      listed species. With ``"forms"`` (equivalent dissolution reactions,
      one per dissolved complex, each ``{"species", "stoich", "K_s"}``, the
      complex first), :math:`\Omega = \sum_f w_f\Omega_f`,
      :math:`w_f = C_f/C_T`, :math:`C_T = \sum_g C_g`, and each form takes
      the share :math:`w_f` of :math:`r`. The weighted ratio is well
      conditioned for ions that exist mostly as complexes (free Fe3+ is
      about 1e-35 mol/m^3 at pH 13), and equals every :math:`\Omega_f` at
      the hydrolysis equilibria. When both OH- and H+ are species, the
      exchange of OH- equivalents of a precipitation is split between them,
      OH- receiving the share :math:`C_{OH^-}/(C_{OH^-} + C_{H^+})`, so it
      never consumes the trace one.
    * ``"Degassing"`` of ``"species"``:
      :math:`r = k_g\max(C - C_{sat}, 0)`, consuming that species.

    .. rubric:: Outside the weak form

    * ``initialize_fields`` sets :math:`C_\pi` to ``"C0"``; the one species
      with ``"C0": "Neutrality"`` takes the value that makes the initial
      state electroneutral; :math:`\varphi = 0`.
    * ``commit`` copies the fields to ``<species>Old`` and ``EOld`` and
      updates ``pH``; ``reset_step`` restores the committed fields.
    * :meth:`Current_Conservation_Rows` gives the row combination for a
      consistent initial potential through
      :meth:`femcubes.core.solvers.step_solver.StepSolver.Solve_Constrained_Step`.
    * No Dirichlet conditions, no global measures.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Field_Orders = {"C": 1, "E": 1}      # REQUIRED: polynomial order of the concentrations and of E (equal)
            Solution_Steps = {"C": 0, "E": 0}    # REQUIRED: step solving the concentrations and E (equal)
            Species = {                          # REQUIRED: one field per species
                "Na+": {"Diffusivity": 1.33e-9,  # REQUIRED for dissolved species: D (m^2/s)
                        "Charge": 1.0,           # REQUIRED for dissolved species: z (-)
                        "C0": "Neutrality"},     # REQUIRED: initial concentration (mol/m^3), or "Neutrality" (one species)
                "Cl-": {"Diffusivity": 2.03e-9, "Charge": -1.0, "C0": 600.0},
                "H+": {"Diffusivity": 9.31e-9, "Charge": 1.0, "C0": 1.0e-4},
                "OH-": {"Diffusivity": 5.27e-9, "Charge": -1.0, "C0": 1.0e-4},
                "Fe2+": {"Diffusivity": 7.19e-10, "Charge": 2.0, "C0": 0.0},
                "FeOH+": {"Diffusivity": 7.19e-10, "Charge": 1.0, "C0": 0.0},
                "Fe(OH)2(s)": {"Solid": True,    # OPTIONAL (default False): immobile solid, needs only "C0"
                               "C0": 0.0},
            }
            C_ref = 1000.0                       # OPTIONAL (default 1.0): reference concentration of the activities (mol/m^3)
            Activity_Parameters = {              # REQUIRED unless Activity_Model is "ideal", see activity
                "A": 0.509, "B_a": 1.5, "SIT_epsilon": {"Na+|Cl-": 0.03},
            }
            Volume_Reactions = {                 # OPTIONAL (default none)
                "Water": {"Type": "Equilibrium",         # OPTIONAL (default "Kinetic")
                          "Normalised": True,            # OPTIONAL (default False)
                          "K_eq": 1.0e-14,               # Equilibrium: REQUIRED if Normalised, else OPTIONAL (default 0.0)
                          "k_eq": 1.0e3,                 # REQUIRED if Normalised: relaxation rate (1/s)
                          # "k_dummy": 1.0e6,            # OPTIONAL (default 0.0) if not Normalised: penalty rate (mol/(m^3 s))
                          "species_in": ["H+", "OH-"],   # OPTIONAL (default []): reactants
                          "stoich_in": [1.0, 1.0],       # OPTIONAL (default []): their stoichiometry
                          "species_out": [], "stoich_out": []},  # OPTIONAL (default []): products
                "Hydrolysis": {"Type": "Kinetic",
                               "k_fwd": 0.1,             # OPTIONAL (default 0.0): forward rate (mol/(m^3 s))
                               "k_rev": 1.0e-3,          # OPTIONAL (default 0.0): backward rate (mol/(m^3 s))
                               # "orders_in": [1.0], "orders_out": [...],  # OPTIONAL (default the stoichiometry)
                               "species_in": ["Fe2+"], "stoich_in": [1.0],
                               "species_out": ["FeOH+", "H+"], "stoich_out": [1.0, 1.0]},
                "Fe(OH)2": {"Type": "Precipitation",
                            "solid": "Fe(OH)2(s)",       # REQUIRED: the solid species
                            "k_p": 1.0,                  # REQUIRED: rate constant (1/s)
                            "species": ["Fe2+", "OH-"],  # REQUIRED unless "forms": dissolution products
                            "stoich": [1.0, 2.0],        # REQUIRED unless "forms"
                            "K_s": 10**-15.68,           # REQUIRED unless "forms": solubility product (-)
                            "epsilon": 1.0e-3},          # OPTIONAL (default 1e-3): smoothing (-)
                # "H2 out": {"Type": "Degassing", "species": "H2",  # REQUIRED: species, C_sat (mol/m^3), k_g (1/s)
                #            "C_sat": 0.78, "k_g": 1.0},
            }
            Models = {
                "Electrolyte": {
                    "Type": "ElectroChemistry:Electrolyte",  # REQUIRED
                    "Activity_Model": "SIT",           # REQUIRED: "ideal", "Davies" or "SIT"
                    "Activity_Tangent": "full",        # OPTIONAL (default "full"): or "lagged"
                    "Temperature": 298.15,             # OPTIONAL (default 298.15): T (K)
                    "Concentration_Floor": 1.0e-12,    # OPTIONAL (default 1e-12): floor in the reaction rates (mol/m^3)
                    "domain": "domain",                # OPTIONAL (default "domain", the full mesh): domain group
                },
            }
    """

    MODEL_TYPE = "ElectroChemistry:Electrolyte"
    SUPPORTS_AXISYMMETRY = True

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        settings = self.params.Models[self.name]

        self.F = 96485.3329  # Faraday's constant (A/mol)
        self.R = 8.3145     # Universal gas constant (J/(mol·K))
        self.T = settings.get("Temperature", 298.15)     # Temperature (K)

        self.SpeciesNames = list(self.params.Species.keys())
        self.SolidNames = [s for s in self.SpeciesNames if self.params.Species[s].get("Solid", False)]
        self.DissolvedNames = [s for s in self.SpeciesNames if s not in self.SolidNames]
        self.NumSpecies = len(self.SpeciesNames)
        field_order = self.params.Field_Orders["C"]
        self.C_Step = self.params.Solution_Steps["C"]
        self.D = {}
        self.z = {}
        for s in self.SpeciesNames:
            field_name = f"{s}"
            self.mesh.Add_Field(field_name, f"Concentration of {s} [mol/m^3]", "CG", field_order, domain=self.domain)
            field_name_old = f"{s}Old"
            self.mesh.Add_Field(field_name_old, f"Old concentration of {s} [mol/m^3]", "CG", field_order, domain=self.domain)
            self.mesh.Add_TestFunction(field_name, self.C_Step)

            if s in self.SolidNames:
                self.D[s] = 0.0
                self.z[s] = 0.0
            else:
                self.D[s] = self.params.Species[s]["Diffusivity"]
                self.z[s] = self.params.Species[s]["Charge"]

        E_Step = self.params.Solution_Steps["E"]
        if E_Step != self.C_Step:
            raise ValueError("Electrolyte model requires that concentration and electric potential be solved in the same step.")
        field_order_E = self.params.Field_Orders["E"]
        self.mesh.Add_Field("E", "Electric Potential [V]", "CG", field_order_E, domain=self.domain)
        self.mesh.Add_Field("EOld", "Old Electric Potential [V]", "CG", field_order_E, domain=self.domain)
        self.mesh.Add_TestFunction("E", self.C_Step)
        if "H+" in self.SpeciesNames:
            self.mesh.Add_Field("pH", "pH [-]", "CG", field_order, domain=self.domain)

        if "Activity_Model" not in settings:
            raise ValueError(f"Electrolyte model '{name}' requires the setting 'Activity_Model' ('ideal', 'Davies' or 'SIT').")
        self.activity = ActivityModel(settings["Activity_Model"], getattr(self.params, "Activity_Parameters", None),
                                      {s: self.z[s] for s in self.DissolvedNames})
        self.C_Ref = getattr(self.params, "C_ref", 1.0)
        # Floor of the concentrations in the reaction rates (mol/m^3). Trace
        # species of fast equilibria (e.g. free Fe3+ in alkaline solution,
        # ~1e-33 mol/m^3) need a floor far below their true value.
        self.C_Floor = settings.get("Concentration_Floor", 1.0e-12)
        self.activity_tangent = settings.get("Activity_Tangent", "full")
        if self.activity_tangent not in ("full", "lagged"):
            raise ValueError(f"Activity_Tangent must be 'full' or 'lagged', not '{self.activity_tangent}'.")
        self.reactions = []
        for rname, rdata in getattr(self.params, "Volume_Reactions", {}).items():
            parsed = self._parse_reaction(rname, rdata)
            self.reactions += parsed if isinstance(parsed, list) else [parsed]

    def Current_Conservation_Rows(self) -> dict:
        """Return the row combination that makes current conservation the equation of the potential ``"E"``.

        ``{"E": {species: charge}}`` over the charged dissolved species, for
        :meth:`femcubes.core.solvers.step_solver.StepSolver.Solve_Constrained_Step`.
        The charge-weighted sum of the species balances is the conservation
        of charge, :math:`\\nabla \\cdot \\sum_\\pi z_\\pi \\mathbf{J}_\\pi` plus the
        surface currents (charge-balanced volume reactions cancel, and so
        does the capacity term of an electroneutral state); with the
        concentrations frozen it determines the potential, which
        electroneutrality does not contain. A consistent initial potential
        (with its diffusion potential and ohmic drop) follows from
        ``solver.Solve_Constrained_Step(step, frozen_fields=<all species, coverages, ...>,
        combined_rows=electrolyte.Current_Conservation_Rows())``.
        """
        return {"E": {s: self.z[s] for s in self.DissolvedNames if self.z[s] != 0.0}}

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """
        Return the Nernst-Planck flux and electromigration terms of the electrolyte.

        Capacity, electroneutrality, and volume reactions are added as lumped
        terms in :meth:`Add_KF`. With a non-ideal activity model the flux
        includes :math:`C \\nabla \\ln\\gamma`, linearized through UFL with
        respect to every concentration it depends on.
        """

        if (step != self.C_Step):
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)

        E    = self.mesh.Get_Field("E", step=self.C_Step)
        E_t  = self.mesh.Get_TestFunction("E", step=self.C_Step)
        E_tr = self.mesh.Get_Trial_Function("E", step=self.C_Step)

        C_fields = {s: self.mesh.Get_Field(f"{s}", step=self.C_Step) for s in self.DissolvedNames}
        ln_gamma = self.activity.Ln_Gamma_UFL(C_fields)

        F = 0
        K = 0
        for s in self.DissolvedNames:
            C    = C_fields[s]
            C_t  = self.mesh.Get_TestFunction(f"{s}", step=self.C_Step)
            C_tr = self.mesh.Get_Trial_Function(f"{s}", step=self.C_Step)

            ## Nernst-Planck equation,
            # capacity within lumped integration

            # flux
            F += inner(grad(C_t), self.D[s]*grad(C))*dx
            K += inner(grad(C_t), self.D[s]*grad(C_tr))*dx

            # electromigration
            F += inner(grad(C_t), C*self.D[s]*self.z[s]*self.F/(self.R*self.T)*grad(E))*dx
            K += inner(grad(C_t), C*self.D[s]*self.z[s]*self.F/(self.R*self.T)*grad(E_tr))*dx
            K += inner(grad(C_t), self.D[s]*self.z[s]*self.F/(self.R*self.T)*grad(E)*C_tr)*dx

            # activity-coefficient gradient (non-ideal electrolytes only)
            if ln_gamma is not None and s in ln_gamma:
                F_activity = inner(grad(C_t), self.D[s]*C*grad(ln_gamma[s]))*dx
                F += F_activity
                if self.activity_tangent == "full":
                    for k in self.DissolvedNames:
                        K += derivative(F_activity, C_fields[k], self.mesh.Get_Trial_Function(f"{k}", step=self.C_Step))
                else:   # "lagged": the derivative of the C factor only (the residual is unchanged)
                    K += inner(grad(C_t), self.D[s]*C_tr*grad(ln_gamma[s]))*dx

            # Electroneutrality
            # Electroneutrality within lumped integration

        # The lumped capacity, electroneutrality, and reaction terms of
        # Add_KF couple every field at a node. Zero-valued terms reserve
        # those couplings in the block matrix layout of the step to prevent 
        # allocation issues when using large numbers of species.
        zero = fem.Constant(self.mesh.mesh, 0.0)
        fields = [(self.mesh.Get_TestFunction(name, step=self.C_Step), self.mesh.Get_Trial_Function(name, step=self.C_Step))
                  for name in self.SpeciesNames + ["E"]]
        F += zero * E_t * dx
        for s in self.SolidNames:
            F += zero * self.mesh.Get_TestFunction(f"{s}", step=self.C_Step) * dx
        for test, _ in fields:
            for _, trial in fields:
                K += zero * test * trial * dx

        return K, F

    def Add_KF(self, K, F, step: int) -> None:
        """
        Add the lumped capacity, electroneutrality and volume-reaction terms.

        ``K`` may be ``None`` during residual-only assembly.
        """
        if (step != self.C_Step):
            return

        dt = self.params.dt

        dofNames = self.SpeciesNames + ["E"]
        OldNames = [f"{s}Old" for s in self.SpeciesNames] + ["EOld"]
        W, dofs, vals = self.mesh.Get_Weights_Dofs(dofNames, step=self.C_Step)
        valsOld = self.mesh.Get_Old_For_Lumped(OldNames, step=self.C_Step)
        n_nodes, n_fields = len(W), self.NumSpecies + 1
        Floc = np.zeros((n_nodes, n_fields))
        Kloc = np.zeros((n_nodes, n_fields, n_fields))

        # Capacity
        for fi, f in enumerate(self.SpeciesNames):
            Floc[:, fi] += W * (vals[f] - valsOld[f + "Old"]) / dt
            Kloc[:, fi, fi] += W / dt

        # Electroneutrality (dissolved species, non-negative concentrations)
        for fi, f in enumerate(self.SpeciesNames):
            if f in self.SolidNames:
                continue
            positive = vals[f] > 0.0
            Floc[:, -1] += W * self.z[f] * np.where(positive, vals[f], 0.0)
            Kloc[:, -1, fi] += W * self.z[f] * positive

        # Volume reactions
        production, derivatives = self.Production_Rates(vals)
        for fi, f in enumerate(self.SpeciesNames):
            Floc[:, fi] -= W * production[f]
            for fj, g in enumerate(self.SpeciesNames):
                if g in derivatives[f]:
                    Kloc[:, fi, fj] -= W * derivatives[f][g]

        with F.localForm() as F_local:
            F_array = F_local.array
            for fi, f in enumerate(dofNames):
                F_array[dofs[f]] += Floc[:, fi]

        if K is not None:
            global_dofs = np.stack([self.mesh.get_step_global_indices(self.C_Step, dofs[name]) for name in dofNames], axis=1)
            for indices, block in zip(global_dofs, Kloc):
                K.setValues(indices, indices, block, addv=PETSc.InsertMode.ADD)

    def Production_Rates(self, vals):
        """Return the nodal production rates of the volume reactions and their derivatives.

        Parameters
        ----------
        vals : dict
            Nodal concentrations (mol/m^3) of all species.

        Returns
        -------
        tuple
            ``(production, derivatives)``: per species the production rate
            in mol/(m^3 s), and per species ``derivatives[s][k]``, the
            derivative of that production with respect to :math:`C_k`
            (listed only where it can be non-zero).
        """
        n_nodes = len(vals[self.SpeciesNames[0]])
        production = {s: np.zeros(n_nodes) for s in self.SpeciesNames}
        derivatives = {s: {} for s in self.SpeciesNames}
        if not self.reactions:
            return production, derivatives
        activity, d_activity = self._activities(vals)

        split = "OH-" in self.DissolvedNames and "H+" in self.DissolvedNames
        if split:
            # Abundance share of OH- among OH- and H+ (see the OH-/H+ split below).
            C_OH = np.maximum(vals["OH-"], self.C_Floor)
            C_H = np.maximum(vals["H+"], self.C_Floor)
            w = C_OH / (C_OH + C_H)
            d_w = {"OH-": (1.0 - w) / (C_OH + C_H) * (vals["OH-"] > self.C_Floor),
                   "H+": -w / (C_OH + C_H) * (vals["H+"] > self.C_Floor)}
        for reaction in self.reactions:
            rate, d_rate = self._reaction_rate(reaction, vals, activity, d_activity)
            stoichiometry = dict(reaction.stoichiometry)
            nu_OH = stoichiometry.get("OH-", 0.0) - stoichiometry.get("H+", 0.0)
            if split and reaction.type == "Precipitation" and nu_OH != 0.0:
                # Precipitation is not self-limiting in OH- (or H+): split its exchange of OH- equivalents
                # between OH- and H+ by their abundance (equivalent through the water equilibrium), so it
                # does not consume the trace one (OH- of a neutralising pit solution).
                stoichiometry.pop("OH-", None)
                stoichiometry.pop("H+", None)
                for s, share, d_share in (("OH-", w, d_w), ("H+", w - 1.0, d_w)):
                    production[s] += nu_OH * share * rate
                    for k in set(d_rate) | set(d_share):
                        derivatives[s][k] = (derivatives[s].get(k, 0.0) + nu_OH * share * d_rate.get(k, 0.0)
                                             + nu_OH * rate * d_share.get(k, 0.0))
            for s, nu in stoichiometry.items():
                production[s] += nu * rate
                for k, d in d_rate.items():
                    derivatives[s][k] = derivatives[s].get(k, 0.0) + nu * d
        return production, derivatives

    def initialize_fields(self) -> None:
        """
        Set the concentrations to their initial values and the potential to zero.
        """

        NeutralSpecies = None
        EN = 0.0
        for s in self.SpeciesNames:
            field_name = f"{s}"
            field_name_old = f"{s}Old"
            C = self.params.Species[s]["C0"]
            if C == "Neutrality":
                if NeutralSpecies is not None:
                    raise ValueError("Only one species can be set to 'Neutrality' for initial concentration.")
                NeutralSpecies = s
            else:
                self.mesh.Constant_Field(field_name, C, domain=self.domain)
                self.mesh.Constant_Field(field_name_old, C, domain=self.domain)
                EN += C * self.z[s]
        if NeutralSpecies is not None:
            C_neutral = -EN / self.z[NeutralSpecies]
            field_name = f"{NeutralSpecies}"
            field_name_old = f"{NeutralSpecies}Old"
            self.mesh.Constant_Field(field_name, C_neutral, domain=self.domain)
            self.mesh.Constant_Field(field_name_old, C_neutral, domain=self.domain)

        self.mesh.Constant_Field("E", 0.0, domain=self.domain)
        self.mesh.Constant_Field("EOld", 0.0, domain=self.domain)
        self._update_pH()

    def commit(self) -> None:
        """
        Commit the current solution by updating old fields.
        """
        # Copy current fields to old fields
        for s in self.SpeciesNames:
            field_name = f"{s}"
            field_name_old = f"{s}Old"
            self.mesh.Copy_Field(field_name, field_name_old, domain=self.domain)

        self.mesh.Copy_Field("E", "EOld", domain=self.domain)
        self._update_pH()

    def reset_step(self):
        self.mesh.Copy_Field("EOld", "E", domain=self.domain)
        for s in self.SpeciesNames:
            field_name = f"{s}"
            field_name_old = f"{s}Old"
            self.mesh.Copy_Field(field_name_old, field_name, domain=self.domain)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _update_pH(self) -> None:
        """PRIVATE: Set the pH field from the H+ activity (concentrations in mol/m^3, i.e. 1e-3 mol/L)."""
        if "H+" not in self.SpeciesNames:
            return
        # Concentrations may undershoot zero slightly; cap the pH at 17.
        C_H = max_value(self.mesh.Get_Field("H+"), 1.0e-14)
        pH = 3.0 - ln(C_H) / math.log(10.0)
        ln_gamma = self.activity.Ln_Gamma_UFL({s: self.mesh.Get_Field(s) for s in self.DissolvedNames})
        if ln_gamma is not None:
            pH = pH - ln_gamma["H+"] / math.log(10.0)
        self.mesh.Set_From_Expression("pH", pH, domain=self.domain)

    def _parse_reaction(self, rname: str, rdata: dict) -> _VolumeReaction:
        """PRIVATE: Validate one entry of ``params.Volume_Reactions`` and return its parsed form (a list for Precipitation, one per form)."""
        rtype = rdata.get("Type", "Kinetic")
        if rtype not in _REACTION_TYPES:
            raise ValueError(f"Volume reaction '{rname}': unknown Type '{rtype}'; expected one of {_REACTION_TYPES}.")

        def check_species(names, solid_allowed=False):
            for s in names:
                if s not in self.SpeciesNames:
                    raise ValueError(f"Reaction {rname} references unknown species {s}.")
                if s in self.SolidNames and not solid_allowed:
                    raise ValueError(f"Reaction {rname}: solid {s} can only take part in a Precipitation reaction.")

        if rtype in ("Kinetic", "Equilibrium"):
            species_in, stoich_in = rdata.get("species_in", []), rdata.get("stoich_in", [])
            species_out, stoich_out = rdata.get("species_out", []), rdata.get("stoich_out", [])
            check_species(species_in + species_out)
            stoichiometry = {}
            for s, nu in zip(species_in, stoich_in):
                stoichiometry[s] = stoichiometry.get(s, 0.0) - nu
            for s, nu in zip(species_out, stoich_out):
                stoichiometry[s] = stoichiometry.get(s, 0.0) + nu
            if rtype == "Kinetic":
                orders_in = dict(zip(species_in, rdata.get("orders_in", stoich_in)))
                orders_out = dict(zip(species_out, rdata.get("orders_out", stoich_out)))
                constants = {"k_fwd": rdata.get("k_fwd", 0.0), "k_rev": rdata.get("k_rev", 0.0)}
            else:
                orders_in, orders_out = dict(zip(species_in, stoich_in)), dict(zip(species_out, stoich_out))
                normalised = rdata.get("Normalised", False)
                if normalised:
                    constants = {"normalised": True, "K_eq": rdata["K_eq"], "k_eq": rdata["k_eq"]}
                else:
                    constants = {"normalised": False, "K_eq": rdata.get("K_eq", 0.0), "k_dummy": rdata.get("k_dummy", 0.0)}
            return _VolumeReaction(rname, rtype, stoichiometry, orders_in, orders_out, constants)

        if rtype == "Precipitation":
            solid = rdata["solid"]
            if solid not in self.SolidNames:
                raise ValueError(f"Precipitation '{rname}': '{solid}' must be a species with 'Solid': True.")
            if "width" in rdata:
                raise ValueError(f"Precipitation '{rname}': 'width' is no longer supported; the rate is the smoothed "
                                 "Fischer-Burmeister form with the smoothing 'epsilon'.")
            epsilon = float(rdata.get("epsilon", 1.0e-3))
            if epsilon <= 0.0:
                raise ValueError(f"Precipitation '{rname}': 'epsilon' must be positive, not {epsilon}.")
            forms = rdata.get("forms", [{"species": rdata.get("species"), "stoich": rdata.get("stoich"), "K_s": rdata.get("K_s")}])
            group = [(form["species"][0], dict(zip(form["species"], form["stoich"])), form["K_s"]) for form in forms]
            parsed = []
            for index, form in enumerate(forms):
                check_species(form["species"])
                # Positive rate = precipitation: the solid forms, the dissolution products are consumed.
                stoichiometry = {solid: 1.0}
                for s, nu in zip(form["species"], form["stoich"]):
                    stoichiometry[s] = stoichiometry.get(s, 0.0) - nu
                orders = dict(zip(form["species"], form["stoich"]))
                parsed.append(_VolumeReaction(rname, rtype, stoichiometry, orders, {},
                                              {"solid": solid, "k_p": rdata["k_p"], "epsilon": epsilon,
                                               "form": index, "group": group}))
            return parsed

        # Degassing
        species = rdata["species"]
        check_species([species])
        return _VolumeReaction(rname, rtype, {species: -1.0}, {}, {},
                               {"species": species, "C_sat": rdata["C_sat"], "k_g": rdata["k_g"]})

    def _activities(self, vals):
        """PRIVATE: Return nodal activities ``a[s]`` of the dissolved species and ``d_activity[s][k] = da_s/dC_k``.

        Concentrations below the floor are clipped, with a zero derivative of
        the clipped concentration itself.
        """
        ln_gamma, d_ln_gamma = self.activity.Ln_Gamma({s: vals[s] for s in self.DissolvedNames})
        activity, d_activity = {}, {}
        for s in self.DissolvedNames:
            clipped = np.maximum(vals[s], self.C_Floor)
            active = (vals[s] > self.C_Floor).astype(float)
            a = np.exp(ln_gamma[s]) * clipped / self.C_Ref
            activity[s] = a
            d = {k: a * d_ln for k, d_ln in d_ln_gamma[s].items()}
            d[s] = d.get(s, 0.0) + a * active / clipped
            d_activity[s] = d
        return activity, d_activity

    def _reaction_rate(self, reaction: _VolumeReaction, vals, activity, d_activity):
        """PRIVATE: Return ``(rate, {species: d rate / d C})`` of one volume reaction at the nodes."""
        c = reaction.constants

        if reaction.type == "Degassing":
            excess = vals[c["species"]] - c["C_sat"]
            return c["k_g"] * np.maximum(excess, 0.0), {c["species"]: c["k_g"] * (excess > 0.0)}

        # Rate and its derivatives with respect to the activities, then the chain rule to concentrations.
        products_in = _monomial(activity, reaction.orders_in)
        products_out = _monomial(activity, reaction.orders_out)
        d_rate_da = {}
        d_rate_dC = {}

        if reaction.type == "Kinetic":
            rate = c["k_fwd"] * products_in - c["k_rev"] * products_out
            for s, m in reaction.orders_in.items():
                d_rate_da[s] = d_rate_da.get(s, 0.0) + c["k_fwd"] * m * products_in / activity[s]
            for s, m in reaction.orders_out.items():
                d_rate_da[s] = d_rate_da.get(s, 0.0) - c["k_rev"] * m * products_out / activity[s]

        elif reaction.type == "Equilibrium" and not c["normalised"]:
            rate = c["k_dummy"] * (products_in - c["K_eq"] * products_out)
            for s, m in reaction.orders_in.items():
                d_rate_da[s] = d_rate_da.get(s, 0.0) + c["k_dummy"] * m * products_in / activity[s]
            for s, m in reaction.orders_out.items():
                d_rate_da[s] = d_rate_da.get(s, 0.0) - c["k_dummy"] * c["K_eq"] * m * products_out / activity[s]

        elif reaction.type == "Equilibrium":
            # G = P_in - K P_out, normalised by N = sum_s |dG/da_s|:
            # rate = k_eq C_ref G / N, d rate = k_eq C_ref (dG / N - G dN / N^2).
            terms = [(1.0, products_in, reaction.orders_in), (-c["K_eq"], products_out, reaction.orders_out)]
            participants = list(dict.fromkeys(list(reaction.orders_in) + list(reaction.orders_out)))
            G = products_in - c["K_eq"] * products_out
            dG = {s: sum(sign * m.get(s, 0.0) * P / activity[s] for sign, P, m in terms) for s in participants}
            N = sum(np.abs(dG[s]) for s in participants) + 1.0e-300
            dN = {t: 0.0 for t in participants}
            for s in participants:
                sign_s = np.sign(dG[s])
                for t in participants:
                    # second derivatives of the monomials
                    d2G = 0.0
                    for sign, P, m in terms:
                        m_s, m_t = m.get(s, 0.0), m.get(t, 0.0)
                        if m_s == 0.0 or m_t == 0.0:
                            continue
                        factor = m_s * (m_t - 1.0) if s == t else m_s * m_t
                        d2G = d2G + sign * factor * P / (activity[s] * activity[t])
                    dN[t] = dN[t] + sign_s * d2G
            scale = c["k_eq"] * self.C_Ref
            rate = scale * G / N
            for s in participants:
                d_rate_da[s] = scale * (dG[s] / N - G * dN[s] / N ** 2)

        else:  # Precipitation: the share of one form in the precipitation of its solid
            # Omega = sum_f w_f Omega_f with abundance weights w_f = C_f / C_T of the complexes, C_T = sum_g C_g
            # (all Omega_f are equal at the hydrolysis equilibria); this form takes the share w_f of
            # R = -k_p FB(a, b), the smoothed Fischer-Burmeister function of the complementarity
            # a = C_s >= 0, b = -C_T ln Omega >= 0, a b = 0: FB = a + b - sqrt(a^2 + b^2 + 2 delta^2), delta = eps C_T.
            keys = [key for key, _orders, _K in c["group"]]
            clipped = {key: np.maximum(vals[key], self.C_Floor) for key in keys}
            above = {key: (vals[key] > self.C_Floor).astype(float) for key in keys}
            C_T = sum(clipped.values())
            weights = {key: clipped[key] / C_T for key in keys}
            omega = 0.0
            d_omega_da, d_omega_dC = {}, {}
            for key, orders, K_s in c["group"]:
                omega_f = _monomial(activity, orders) / K_s
                omega = omega + weights[key] * omega_f
                for s, m in orders.items():
                    d_omega_da[s] = d_omega_da.get(s, 0.0) + weights[key] * m * omega_f / activity[s]
                for other in keys:
                    d_w = ((1.0 if other == key else 0.0) - weights[key]) / C_T * above[other]
                    d_omega_dC[other] = d_omega_dC.get(other, 0.0) + omega_f * d_w
            ln_omega = np.log(omega)
            C_solid = vals[c["solid"]]
            a, b, delta = C_solid, -C_T * ln_omega, c["epsilon"] * C_T
            root = np.sqrt(a**2 + b**2 + 2.0 * delta**2)
            total_rate = -c["k_p"] * (a + b - root)
            # partial derivatives of FB, then d total / d ln(Omega), d C_T (through b and delta), d C_s
            FB_a, FB_b, FB_delta = 1.0 - a / root, 1.0 - b / root, -2.0 * delta / root
            d_total_d_ln_omega = c["k_p"] * FB_b * C_T
            d_total_d_C_T = c["k_p"] * (FB_b * ln_omega - FB_delta * c["epsilon"])
            key = keys[c["form"]]
            share = weights[key]
            rate = share * total_rate
            for s, d in d_omega_da.items():
                d_rate_da[s] = share * d_total_d_ln_omega * d / omega
            for other in keys:
                d_share = ((1.0 if other == key else 0.0) - share) / C_T * above[other]
                d_rate_dC[other] = (share * d_total_d_ln_omega * d_omega_dC.get(other, 0.0) / omega
                                    + share * d_total_d_C_T * above[other]
                                    + total_rate * d_share)
            d_rate_dC[c["solid"]] = d_rate_dC.get(c["solid"], 0.0) - share * c["k_p"] * FB_a

        for s, d_a in d_rate_da.items():
            for k, d_ak in d_activity[s].items():
                d_rate_dC[k] = d_rate_dC.get(k, 0.0) + d_a * d_ak
        return rate, d_rate_dC


def _monomial(activity: dict, orders: dict):
    """PRIVATE: Return :math:`\\prod_s a_s^{m_s}` (1 for an empty product)."""
    product = 1.0
    for s, m in orders.items():
        if m != 0.0:
            product = product * activity[s] ** m
    return product
