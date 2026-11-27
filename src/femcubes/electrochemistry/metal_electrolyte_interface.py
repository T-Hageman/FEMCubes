from dataclasses import dataclass

import numpy as np
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc
from ufl import Measure

from femcubes.core.base_model import BaseModel
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.utils.mpi_utils import comm


_FARADAY = 96485.3329   # Faraday constant (C/mol)
_GAS_CONSTANT = 8.3145  # Universal gas constant (J/(mol K))
_CHARGE_CONSERVATION = "Charge_Conservation"


@dataclass(frozen=True)
class _Reaction:
    """PRIVATE: Static description of one interface reaction.

    ``stoichiometry`` gives the production of each balanced field per unit
    (forward) reaction rate; ``electrons`` the electrons consumed from the
    metal per unit rate (cathodic reactions positive).
    """

    fields: tuple
    stoichiometry: dict
    electrons: int
    backward: bool
    electrochemical: bool = True


_REACTIONS = {
    # H+ + e- <-> H_ads
    "Volmer_acid": _Reaction(("H+", "theta_ads"), {"H+": -1.0, "theta_ads": 1.0}, 1, True),
    # H+ + e- + H_ads -> H2
    "Heyrovsky_acid": _Reaction(("H+", "theta_ads"), {"H+": -1.0, "theta_ads": -1.0}, 1, False),
    # H2O + e- <-> H_ads + OH-
    "Volmer_base": _Reaction(("OH-", "theta_ads"), {"OH-": 1.0, "theta_ads": 1.0}, 1, True),
    # H2O + e- + H_ads -> H2 + OH-
    "Heyrovsky_base": _Reaction(("theta_ads",), {"OH-": 1.0, "theta_ads": -1.0}, 1, False),
    # 2 H_ads -> H2
    "Tafel": _Reaction(("theta_ads",), {"theta_ads": -2.0}, 0, False, electrochemical=False),
    # H_ads <-> H_abs
    "Absorption": _Reaction(("theta_ads", "CL"), {"theta_ads": -1.0, "CL": 1.0}, 0, True, electrochemical=False),
    # Fe2+ + 2e- <-> Fe (backward: corrosion)
    "Corrosion": _Reaction(("Fe2+",), {"Fe2+": -1.0}, 2, True),
    # O2 + 2 H2O + 4e- <-> 4 OH-
    "Oxygen": _Reaction(("O2", "OH-"), {"O2": -1.0, "OH-": 4.0}, 4, True),
}

# Solution-step keys of the fields entering the reactions.
_FIELD_STEP_KEYS = {"theta_ads": "theta", "CL": "CL"}


class MetalElectrolyteInterface(BaseModel):
    r"""Reactions at a metal-electrolyte interface: hydrogen evolution and absorption, corrosion and oxygen reduction.

    The model couples an electrolyte to a metal through the electrochemical
    reactions at their common surface. The charge-transfer reactions follow
    Butler-Volmer kinetics, with forward and backward rates that depend
    exponentially on the overpotential :cite:p:`Newman2021`. Hydrogen
    evolution proceeds through adsorbed hydrogen: the Volmer step adsorbs
    hydrogen on the surface, the Heyrovsky and Tafel steps recombine it into
    hydrogen gas, and part of the adsorbed hydrogen is absorbed into the
    metal, which couples the hydrogen evolution to the hydrogen uptake of the
    metal :cite:p:`Hageman2022a`. The metal potential is either imposed, or
    follows from charge conservation, which balances the anodic and cathodic
    currents over the whole metal surface with the applied current (zero for
    an isolated metal) :cite:p:`Hageman2023`.

    The reactions are lumped surface reactions on an interior facet group
    between an :doc:`electrolyte` and a metal (optionally with
    :doc:`hydrogen_diffusion` for absorbed hydrogen), with an optional
    adsorbed-hydrogen coverage on the interface and an optional
    charge-conserving metal potential. The hydrogen reactions follow
    :cite:t:`Hageman2023b`, Eqs. (30)-(41); corrosion, oxygen reduction and
    the charge-conserving metal potential follow :cite:t:`Hageman2023`,
    Eqs. (8)-(17). Several models may share the coverage field and
    the metal potential (e.g. a pit and the outer surface of one metal);
    each lumps its terms over the nodes of its own facets. Supports 2-D,
    3-D and axisymmetric meshes. Symbols as in :doc:`electrolyte`; here
    :math:`W_i = \int_\Gamma N_i\,\mathrm{d}\Gamma` are the lumped weights
    of the interface nodes.

    The reactions (only those listed in ``"Reactions"`` are active), their
    net forward rates :math:`r` (mol/(m^2 s)) with the overpotential
    :math:`\eta = E_m - \varphi - E_{eq}`, concentrations clipped at zero,
    and their electrons consumed from the metal :math:`n_r`:

    * ``Volmer_acid``, H+ + e- <-> H_ads, :math:`n=1`:
      :math:`k C_{H^+}(1-\theta)e^{-\alpha f\eta} - k'\theta e^{(1-\alpha)f\eta}`
    * ``Heyrovsky_acid``, H+ + e- + H_ads -> H2, :math:`n=1`:
      :math:`k C_{H^+}\theta e^{-\alpha f\eta}`
    * ``Volmer_base``, H2O + e- <-> H_ads + OH-, :math:`n=1`:
      :math:`k(1-\theta)e^{-\alpha f\eta} - k' C_{OH^-}\theta e^{(1-\alpha)f\eta}`
    * ``Heyrovsky_base``, H2O + e- + H_ads -> H2 + OH-, :math:`n=1`:
      :math:`k\theta e^{-\alpha f\eta}`
    * ``Tafel``, 2 H_ads -> H2, :math:`n=0`: :math:`k|\theta|\theta`
    * ``Absorption``, H_ads <-> H_abs, :math:`n=0`:
      :math:`k(N_L - C_L)\theta - k' C_L(1-\theta)`
    * ``Corrosion``, Fe2+ + 2e- <-> Fe, :math:`n=2`:
      :math:`k C_{Fe^{2+}}e^{-\alpha f\eta} - k' e^{(1-\alpha)f\eta}`
    * ``Oxygen``, O2 + 2H2O + 4e- <-> 4OH-, :math:`n=4`:
      :math:`k C_{O_2}e^{-\alpha f\eta} - k' C_{OH^-}e^{(1-\alpha)f\eta}`

    with :math:`k` = ``"k"``, :math:`k'` = ``"k_rev"`` (reversible
    reactions only), :math:`\theta = \theta_{ads}`. The reactions produce
    :math:`P_X = \sum_r \nu_{Xr}\,r_r` of each balanced field
    :math:`X \in \{C_{H^+}, C_{OH^-}, C_{Fe^{2+}}, C_{O_2}, \theta_{ads}, C_L\}`
    with the stoichiometry of the reaction equations (H_ads counts for
    :math:`\theta_{ads}`, H_abs for :math:`C_L`). Fe2+ hydrolysis is a
    volume reaction of :doc:`electrolyte`.

    .. rubric:: Fields

    * ``theta_ads``: adsorbed hydrogen coverage :math:`\theta_{ads}`, CG of
      order ``Field_Orders["theta"]`` on the facet group only, solved in
      step ``Solution_Steps["theta"]``; defined when a hydrogen reaction
      (Volmer, Heyrovsky, Tafel, Absorption) is active. ``theta_adsOld``:
      committed value.
    * ``E_m`` (name set by ``"Metal_Potential_Field"``): metal potential
      :math:`E_m`, a global field (one unknown), with ``E_mOld``; only
      with ``"Metal_Potential": "Charge_Conservation"``.
    * Uses ``E`` and the species fields of the active reactions from
      :doc:`electrolyte`, and ``CL`` from :doc:`hydrogen_diffusion` with
      ``Absorption``. All coupled fields are solved in step
      ``Solution_Steps["C"]`` (checked) and must be scalar CG fields of one
      polynomial degree, at most 2.

    .. rubric:: Weak form

    On :math:`\Gamma`, the facet group named by ``domain``, the nodal
    (lumped) contributions are, for each balanced field other than the
    coverage,

    .. math::

        R_{X} \mathrel{+}= -\sum_i \delta X_i\,W_i\,P_{X,i},

    for the coverage

    .. math::

        R_{\theta_{ads}} = \sum_i \delta\theta_i\,W_i\Bigl(N_{ads}\frac{\theta_i - \theta_{n,i}}{\Delta t} - P_{\theta,i}\Bigr),

    and, with charge conservation, for the global unknown :math:`E_m`

    .. math::

        R_{E_m} \mathrel{+}= F\sum_i W_i \sum_r n_r\,r_{r,i} - I_{applied},

    where several models sharing :math:`E_m` add their terms to one balance
    (and their ``"Applied_Current"`` values add up). The Jacobian is the
    consistent derivative (zero for clipped negative concentrations); the
    reactions couple to :math:`\varphi` and :math:`E_m` through
    :math:`\eta`. The potential :math:`\varphi` itself gets no interface
    term. A zero UFL block over the facet mesh reserves the matrix layout
    of ``theta_ads``.

    .. rubric:: Outside the weak form

    * ``initialize_fields`` sets :math:`\theta_{ads}` to ``"theta0"`` and
      :math:`E_m` to ``"Initial_Metal_Potential"``; ``commit`` copies them
      to the old fields, ``reset_step`` restores them.
    * Global measures (after every converged step):
      ``"<model name>_Coverage"`` (average :math:`\theta_{ads}`, with
      hydrogen reactions), ``"<model name>_Current"``
      (:math:`F\int_\Gamma\sum_r n_r r_r`, net cathodic, A per
      m^(dim-1)), per active group ``"<model name>_Hydrogen_Current"``
      and ``"<model name>_Oxygen_Current"`` (cathodic) and
      ``"<model name>_Corrosion_Current"`` (anodic, positive when
      corroding), and ``"<model name>_Metal_Potential"``.
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"C": 0, "E": 0, "theta": 0, "CL": 0}  # REQUIRED: one step for all coupled fields
            Field_Orders = {"C": 2, "E": 2, "theta": 2, "CL": 2}    # REQUIRED: theta order (equal to the others, at most 2)
            Models = {
                "Interface": {
                    "Type": "ElectroChemistry:MetalElectrolyteInterface",  # REQUIRED
                    "domain": "interface",                # REQUIRED: facet group of the interface
                    "Metal_Potential": "Charge_Conservation",  # REQUIRED: fixed E_m (V_SHE) or "Charge_Conservation"
                    "Metal_Potential_Field": "E_m",       # OPTIONAL (default "E_m"): shared by models on one metal
                    "Initial_Metal_Potential": -0.4,      # OPTIONAL (default 0.0): initial E_m (V_SHE)
                    "Applied_Current": 0.0,               # OPTIONAL (default 0.0): current into the metal (A per m^(dim-1))
                    "Temperature": 293.15,                # OPTIONAL (default 293.15): T (K)
                    "N_ads": 1.0e-3,                      # REQUIRED with hydrogen reactions: adsorption sites (mol/m^2)
                    "theta0": 0.0,                        # OPTIONAL (default 0.0): initial coverage (-)
                    "N_L": 1.0e6,                         # REQUIRED with Absorption: lattice sites (mol/m^3)
                    "Reactions": {                        # REQUIRED: at least one. Per reaction, all OPTIONAL:
                        # "k" (default 0.0), "k_rev" (default 0.0, reversible reactions only),
                        # "alpha" (default 0.5, -), "E_eq" (default 0.0, V_SHE; not used by Tafel, Absorption).
                        # Units of k and k_rev make r a rate in mol/(m^2 s) (m/s when multiplied by a concentration).
                        "Volmer_acid": {"k": 1.0e-4, "k_rev": 1.0e-10, "alpha": 0.5, "E_eq": 0.0},
                        "Heyrovsky_acid": {"k": 1.0e-10, "alpha": 0.3, "E_eq": 0.0},
                        "Volmer_base": {"k": 1.0e-8, "k_rev": 1.0e-13, "alpha": 0.5, "E_eq": 0.0},
                        "Heyrovsky_base": {"k": 1.0e-10, "alpha": 0.3, "E_eq": 0.0},
                        "Tafel": {"k": 1.0e-6},
                        "Absorption": {"k": 1.0e1, "k_rev": 7.0e5},
                        "Corrosion": {"k": 1.0e-6, "k_rev": 1.0e-6, "alpha": 0.5, "E_eq": -0.4},
                        "Oxygen": {"k": 1.0e-9, "k_rev": 1.0e-11, "alpha": 0.5, "E_eq": 0.4},
                    },
                },
            }
    """

    MODEL_TYPE = "ElectroChemistry:MetalElectrolyteInterface"
    SUPPORTS_AXISYMMETRY = True

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        settings = self.params.Models[self.name]

        reactions = settings["Reactions"]
        unknown = set(reactions) - set(_REACTIONS)
        if unknown:
            raise ValueError(f"Unknown interface reactions {sorted(unknown)}; expected {list(_REACTIONS)}.")
        if not reactions:
            raise ValueError(f"Interface model '{name}' needs at least one reaction.")
        self.reactions = {
            reaction: {
                "k": reactions[reaction].get("k", 0.0),
                "k_rev": reactions[reaction].get("k_rev", 0.0) if _REACTIONS[reaction].backward else 0.0,
                "alpha": reactions[reaction].get("alpha", 0.5),
                "E_eq": reactions[reaction].get("E_eq", 0.0),
            }
            for reaction in _REACTIONS if reaction in reactions
        }
        balanced = {field for reaction in self.reactions for field in _REACTIONS[reaction].stoichiometry}
        inputs = {field for reaction in self.reactions for field in _REACTIONS[reaction].fields}
        self._rows = [field for field in _FIELD_ORDER if field in balanced]
        self._fields = [field for field in _FIELD_ORDER if field in balanced | inputs or field == "E"]
        self.has_coverage = "theta_ads" in self._fields

        self.My_Step = params.Solution_Steps["C"]
        for field in self._fields:
            key = _FIELD_STEP_KEYS.get(field, "C")
            if params.Solution_Steps[key] != self.My_Step:
                raise ValueError(
                    "MetalElectrolyteInterface requires all fields it couples to be solved in one step; "
                    f"'{key}' is solved in step {params.Solution_Steps[key]}, the electrolyte in step {self.My_Step}."
                )
        if self.has_coverage:
            order = params.Field_Orders["theta"]
            mesh.Add_Field("theta_ads", "Surface Coverage [-]", "CG", order, domain=self.domain)
            mesh.Add_Field("theta_adsOld", "Old Surface Coverage [-]", "CG", order, domain=self.domain)
            mesh.Add_TestFunction("theta_ads", self.My_Step)
            self.N_ads = settings["N_ads"]
            self.theta0 = settings.get("theta0", 0.0)
        if "Absorption" in self.reactions:
            self.N_L = settings["N_L"]

        metal_potential = settings["Metal_Potential"]
        self.charge_conservation = metal_potential == _CHARGE_CONSERVATION
        if self.charge_conservation:
            self.E_m_field = settings.get("Metal_Potential_Field", "E_m")
            mesh.Add_Global_Field(self.E_m_field, "Metal Potential [V]")
            mesh.Add_Global_Field(f"{self.E_m_field}Old", "Old Metal Potential [V]")
            mesh.Add_TestFunction(self.E_m_field, self.My_Step)
            self.E_m0 = settings.get("Initial_Metal_Potential", 0.0)
            self.applied_current = settings.get("Applied_Current", 0.0)
        elif isinstance(metal_potential, str):
            raise ValueError(
                f"Metal_Potential must be a number or '{_CHARGE_CONSERVATION}', not '{metal_potential}'."
            )
        else:
            self.E_m = float(metal_potential)
        self.f = _FARADAY / (_GAS_CONSTANT * settings.get("Temperature", 293.15))

        for measure in self._measure_names():
            self.params.Global_Measures[measure] = 0.0

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Return a zero coverage block; the coverage capacity and all reactions are added in :meth:`Add_KF`.

        The capacity is lumped over this model's own interface nodes, so
        several models (e.g. on the pit and on the outer surface of one
        metal) can share the coverage field without counting it twice.
        """
        self._dt = self._weak_constant_value(dt_constant)
        if step != self.My_Step or not self.has_coverage:
            return None
        theta_t = self.mesh.Get_TestFunction("theta_ads", self.My_Step)
        theta_tr = self.mesh.Get_Trial_Function("theta_ads", self.My_Step)
        facet_mesh = self.mesh.Get_Field("theta_ads").function_space.mesh
        zero = fem.Constant(facet_mesh, 0.0)
        dGamma = Measure("dx", domain=facet_mesh)
        return zero * theta_t * theta_tr * dGamma, zero * theta_t * dGamma

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        """Store the time step used by the lumped coverage capacity."""
        self._dt = self._weak_constant_value(dt_constant)
        return False

    def Add_KF(self, K, F, step: int) -> None:
        """Add the lumped interface reaction terms (and the charge balance) to the residual and tangent."""
        if step != self.My_Step:
            return
        weights, dofs, vals = self._interface_state(step)
        production, derivatives = self.Production_Rates(vals)
        if self.has_coverage:
            # Lumped capacity N_ads d(theta)/dt, as a negative production.
            capacity = self.N_ads / self._dt
            production["theta_ads"] = production["theta_ads"] - capacity * (vals["theta_ads"] - vals["theta_adsOld"])
            derivatives["theta_ads"]["theta_ads"] = derivatives["theta_ads"]["theta_ads"] - capacity
        columns = list(self._fields)
        if self.charge_conservation:
            columns.append(self.E_m_field)
            charge_dof = self.mesh.Get_Global_Field_Dof(self.E_m_field, step)

        with F.localForm() as F_local:
            for row in self._rows:
                np.add.at(F_local.array, dofs[row], -weights * production[row])
        if self.charge_conservation:
            # Electrons consumed by the reactions minus the applied current.
            # The applied current is added once, by the rank owning the row.
            owner_term = -self.applied_current if comm.rank == 0 else 0.0
            charge_residual = _FARADAY * np.sum(weights * production["charge"]) + owner_term
            F.setValues([charge_dof], [charge_residual], addv=PETSc.InsertMode.ADD_VALUES)
        if K is None:
            return

        global_dofs = {name: self.mesh.get_step_global_indices(step, dofs[name]) for name in self._fields}
        rows = np.stack([global_dofs[row] for row in self._rows], axis=1)
        node_columns = np.stack([global_dofs[column] for column in self._fields], axis=1)
        blocks = -weights[:, None, None] * np.stack(
            [np.stack([derivatives[row][column] for column in columns], axis=1) for row in self._rows], axis=1
        )
        if self.charge_conservation:
            n_nodes = weights.size
            node_columns = np.hstack([node_columns, np.full((n_nodes, 1), charge_dof, dtype=node_columns.dtype)])
            rows = np.hstack([rows, np.full((n_nodes, 1), charge_dof, dtype=rows.dtype)])
            charge_block = _FARADAY * weights[:, None] * np.stack([derivatives["charge"][column] for column in columns], axis=1)
            blocks = np.concatenate([blocks, charge_block[:, None, :]], axis=1)
        for node_rows, columns_of_node, block in zip(rows, node_columns, blocks):
            K.setValues(node_rows, columns_of_node, block, addv=PETSc.InsertMode.ADD)

    def Production_Rates(self, vals):
        """Return nodal production rates and their derivatives.

        Parameters
        ----------
        vals : dict
            Nodal values of the fields the active reactions couple (e.g.
            ``"H+"``, ``"OH-"``, ``"E"``, ``"theta_ads"``, ``"CL"``,
            ``"Fe2+"``, ``"O2"``), and, with charge conservation, of the
            metal potential field.

        Returns
        -------
        tuple
            ``(production, derivatives)``: per balanced field the production
            rate in mol/(m^2 s) (for the coverage: ``N_ads`` times its rate of
            change), plus ``"charge"``, the electrons consumed from the metal
            (mol/(m^2 s)); and per balanced entry and per input field the
            derivative of that production.
        """
        rates = self._reaction_rates(vals)
        columns = list(self._fields) + ([self.E_m_field] if self.charge_conservation else [])
        zero = np.zeros_like(vals["E"])
        production, derivatives = {}, {}
        rows = {row: {name: _REACTIONS[name].stoichiometry.get(row, 0.0) for name in rates} for row in self._rows}
        rows["charge"] = {name: float(_REACTIONS[name].electrons) for name in rates}
        for row, coefficients in rows.items():
            terms = [(coefficient, rates[name]) for name, coefficient in coefficients.items() if coefficient]
            production[row] = sum((coefficient * rate for coefficient, (rate, _d) in terms), zero.copy())
            derivatives[row] = {
                column: sum((coefficient * d.get(column, zero) for coefficient, (_rate, d) in terms), zero.copy())
                for column in columns
            }
        return production, derivatives

    def Update_Global_Measures(self) -> None:
        """Report the coverage, currents and metal potential along the interface."""
        weights, _dofs, vals = self._interface_state(self.My_Step)
        rates = self._reaction_rates(vals)
        length = comm.allreduce(weights.sum(), op=MPI.SUM)

        def integral(values):
            return comm.allreduce(float(np.sum(weights * values)), op=MPI.SUM)

        measures = self.params.Global_Measures
        if self.has_coverage:
            measures[f"{self.name}_Coverage"] = integral(vals["theta_ads"]) / length
        measures[f"{self.name}_Current"] = _FARADAY * integral(
            sum(_REACTIONS[name].electrons * rate for name, (rate, _d) in rates.items()))
        for group, reactions in _CURRENT_GROUPS.items():
            active = [name for name in reactions if name in rates]
            if active:
                # Corrosion is reported as an anodic (positive when corroding) current.
                sign = -1.0 if group == "Corrosion" else 1.0
                measures[f"{self.name}_{group}_Current"] = sign * _FARADAY * integral(
                    sum(_REACTIONS[name].electrons * rates[name][0] for name in active))
        measures[f"{self.name}_Metal_Potential"] = self._metal_potential()

    def initialize_fields(self) -> None:
        """Set the surface coverage and the metal potential to their initial values."""
        if self.has_coverage:
            self.mesh.Constant_Field("theta_ads", self.theta0, domain=self.domain)
            self.mesh.Constant_Field("theta_adsOld", self.theta0, domain=self.domain)
        if self.charge_conservation:
            self.mesh.Constant_Field(self.E_m_field, self.E_m0)
            self.mesh.Constant_Field(f"{self.E_m_field}Old", self.E_m0)

    def commit(self) -> None:
        """Store the converged coverage and metal potential as the old state."""
        if self.has_coverage:
            self.mesh.Copy_Field("theta_ads", "theta_adsOld", domain=self.domain)
        if self.charge_conservation:
            self.mesh.Copy_Field(self.E_m_field, f"{self.E_m_field}Old")

    def reset_step(self) -> None:
        """Restore the last committed coverage and metal potential after a rejected step."""
        if self.has_coverage:
            self.mesh.Copy_Field("theta_adsOld", "theta_ads", domain=self.domain)
        if self.charge_conservation:
            self.mesh.Copy_Field(f"{self.E_m_field}Old", self.E_m_field)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _interface_state(self, step: int):
        """PRIVATE: Return lumped weights, step DOFs and nodal values, including the metal potential."""
        names = self._fields + (["theta_adsOld"] if self.has_coverage else [])
        weights, dofs, vals = self.mesh.Get_Interface_Weights_Dofs(self.domain, names, step)
        if self.charge_conservation:
            vals = {**vals, self.E_m_field: np.full(weights.size, self.mesh.Get_Global_Value(self.E_m_field))}
        return weights, dofs, vals

    def _metal_potential(self) -> float:
        """PRIVATE: Return the current metal potential."""
        return self.mesh.Get_Global_Value(self.E_m_field) if self.charge_conservation else self.E_m

    def _reaction_rates(self, vals) -> dict:
        """PRIVATE: Return per active reaction ``(net forward rate, {field: derivative})``."""
        phi = vals["E"]
        E_m = vals[self.E_m_field] if self.charge_conservation else self.E_m
        clipped = {name: np.maximum(vals[name], 0.0) for name in ("H+", "OH-", "Fe2+", "O2") if name in vals}
        active = {name: (vals[name] > 0.0).astype(float) for name in clipped}
        theta = vals.get("theta_ads")
        rates = {}
        for name, r in self.reactions.items():
            eta = E_m - phi - r["E_eq"]
            fe = np.exp(-r["alpha"] * self.f * eta)
            be = np.exp((1.0 - r["alpha"]) * self.f * eta)
            k, k_rev = r["k"], r["k_rev"]
            # Forward (cathodic) and backward rates, d/d(eta) of each, and the
            # concentration/coverage derivatives of the net rate.
            if name == "Volmer_acid":
                forward, backward = k * clipped["H+"] * (1.0 - theta) * fe, k_rev * theta * be
                d = {"H+": k * (1.0 - theta) * fe * active["H+"], "theta_ads": -k * clipped["H+"] * fe - k_rev * be}
            elif name == "Heyrovsky_acid":
                forward, backward = k * clipped["H+"] * theta * fe, 0.0
                d = {"H+": k * theta * fe * active["H+"], "theta_ads": k * clipped["H+"] * fe}
            elif name == "Volmer_base":
                forward, backward = k * (1.0 - theta) * fe, k_rev * clipped["OH-"] * theta * be
                d = {"OH-": -k_rev * theta * be * active["OH-"], "theta_ads": -k * fe - k_rev * clipped["OH-"] * be}
            elif name == "Heyrovsky_base":
                forward, backward = k * theta * fe, 0.0
                d = {"theta_ads": k * fe}
            elif name == "Tafel":
                rates[name] = (k * np.abs(theta) * theta, {"theta_ads": 2.0 * k * np.abs(theta)})
                continue
            elif name == "Absorption":
                C_L = vals["CL"]
                rates[name] = (k * (self.N_L - C_L) * theta - k_rev * C_L * (1.0 - theta), {
                    "CL": -k * theta - k_rev * (1.0 - theta),
                    "theta_ads": k * (self.N_L - C_L) + k_rev * C_L,
                })
                continue
            elif name == "Corrosion":
                forward, backward = k * clipped["Fe2+"] * fe, k_rev * be
                d = {"Fe2+": k * fe * active["Fe2+"]}
            else:  # Oxygen
                forward, backward = k * clipped["O2"] * fe, k_rev * clipped["OH-"] * be
                d = {"O2": k * fe * active["O2"], "OH-": -k_rev * be * active["OH-"]}
            # d(net)/d(eta) = -alpha f forward - (1 - alpha) f backward; eta = E_m - phi - E_eq.
            d_eta = -r["alpha"] * self.f * forward - (1.0 - r["alpha"]) * self.f * backward
            d["E"] = -d_eta
            if self.charge_conservation:
                d[self.E_m_field] = d_eta
            rates[name] = (forward - backward, d)
        return rates

    def _measure_names(self) -> list:
        """PRIVATE: Return the global measure names this model reports."""
        names = [f"{self.name}_Current", f"{self.name}_Metal_Potential"]
        if self.has_coverage:
            names.insert(0, f"{self.name}_Coverage")
        names += [f"{self.name}_{group}_Current" for group, reactions in _CURRENT_GROUPS.items()
                  if any(name in self.reactions for name in reactions)]
        return names


# Reactions reported together as one current measure.
_CURRENT_GROUPS = {
    "Hydrogen": ("Volmer_acid", "Heyrovsky_acid", "Volmer_base", "Heyrovsky_base"),
    "Corrosion": ("Corrosion",),
    "Oxygen": ("Oxygen",),
}

# Fields the reactions may couple, in block order.
_FIELD_ORDER = ["H+", "OH-", "Fe2+", "O2", "E", "theta_ads", "CL"]
