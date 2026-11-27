from collections import deque
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
from femcubes.electrochemistry._activity import ActivityModel


_FARADAY = 96485.33212       # Faraday constant (C/mol), CODATA 2018
_GAS_CONSTANT = 8.314462618  # Universal gas constant (J/(mol K)), CODATA 2018
_CHARGE_CONSERVATION = "Charge_Conservation"
_LN10 = np.log(10.0)
_FE_MOLAR_VOLUME = 55.845e-3 / 7874.0   # m^3/mol of iron, for the dissolved depth
_CURRENT_DENSITY_FIELDS = {"Anodic_Current_Density": "Anodic current density [A/m^2]",
                           "Dissolution_Current_Density": "Dissolution current density [A/m^2]",
                           "ORR_Current_Density": "ORR current density [A/m^2]",
                           "HER_Current_Density": "HER current density [A/m^2]"}
_MAX_EXPONENT = 600.0                   # cap of the Butler-Volmer exponents (overflow guard)


@dataclass(frozen=True)
class _Step:
    """PRIVATE: One surface step: reactant and product state, electrons released, solutes consumed and released."""

    reactant: str
    product: str
    electrons: int
    consumed: dict
    released: dict


# Surface states; the bare site "Fe" follows from the site balance.
STATES = ("Fe", "FeOH", "FeOH2", "FeCl", "FeOHCl", "FeOOH", "FeOCl")
_COVERED = STATES[1:]
_PROTECTED = ("FeOH2", "FeOOH")       # film-covered states (cathodic exchange current weighting)

STEPS = {
    "R1": _Step("Fe", "FeOH", 1, {"OH-": 1}, {}),
    "R2": _Step("FeOH", "FeOH2", 1, {"OH-": 1}, {}),
    "R3": _Step("FeOH", "Fe", 1, {}, {"Fe2+": 1, "OH-": 1}),
    "R4": _Step("Fe", "FeCl", 1, {"Cl-": 1}, {}),
    "R5": _Step("FeCl", "Fe", 1, {}, {"Fe2+": 1, "Cl-": 1}),
    "R6": _Step("FeOH", "FeOHCl", 1, {"Cl-": 1}, {}),
    "R7": _Step("FeCl", "FeOHCl", 1, {"OH-": 1}, {}),
    "R8": _Step("FeOHCl", "Fe", 0, {"H+": 1}, {"Fe2+": 1, "Cl-": 1, "H2O": 1}),
    "R9": _Step("FeOH2", "FeOOH", 1, {"OH-": 1}, {"H2O": 1}),
    "R10": _Step("FeOOH", "FeOCl", 0, {"Cl-": 1}, {"OH-": 1}),
    "R11": _Step("FeOHCl", "FeOCl", 1, {"OH-": 1}, {"H2O": 1}),
    "R12": _Step("FeOOH", "Fe", 0, {"H2O": 2}, {"Fe(OH)4-": 1, "H+": 1}),
    "R13": _Step("FeOCl", "Fe", 0, {"H2O": 3}, {"Fe(OH)4-": 1, "Cl-": 1, "H+": 2}),
    "R14": _Step("FeOH2", "Fe", 0, {}, {"Fe2+": 1, "OH-": 2}),
}
_NAMES = tuple(STEPS)
_FE2_STEPS = ("R3", "R5", "R8", "R14")
_FE3_STEPS = ("R12", "R13")
_ROUTES = {"Hydroxide": ("R3", "R12", "R14"), "Chloride": ("R5", "R8", "R13")}     # dissolution routes
_WATER_FORMS = ("R1", "R7", "R9", "R11")      # electrochemical steps consuming one OH-
_PROTON_FORMS = ("R12", "R13", "R14")         # chemical dissolution steps
# Solutes of the surface steps (H2O: solvent, unit activity) and of the cathodic reactions.
_SURFACE_SOLUTES = ("OH-", "H+", "Cl-", "Fe2+", "Fe(OH)4-")
_CATHODIC_SOLUTES = ("O2", "H2")
_CATHODIC = ("ORR", "HER_acid", "HER_base")


class PassivatingSurface(BaseModel):
    r"""Passivation and breakdown of an iron surface: surface states, their reactions, and the cathodic reactions.

    Iron in alkaline electrolytes is protected by a thin passive film of iron
    hydroxides and oxyhydroxides, which chloride ions can break down
    locally, initiating localized corrosion :cite:p:`Frankel2017`. The model
    describes this competition at the scale of the surface sites: hydroxide
    ions convert bare sites into the film-forming states, while chloride ions
    form chloride-containing states that dissolve as Fe2+ or Fe(OH)4- and
    return bare sites.

    Lumped surface model on a facet group (the metal surface bounding an
    :doc:`electrolyte`). Each surface site is in one of seven states with
    coverages :math:`\theta_j`, :math:`\sum_j\theta_j = 1`: bare ``Fe``,
    ``FeOH``, ``FeCl`` (Fe(I)), ``FeOH2``, ``FeOHCl`` (Fe(II)) and
    ``FeOOH``, ``FeOCl`` (Fe(III)). Fourteen surface steps convert the
    states and exchange solutes with the electrolyte; three cathodic
    reactions act on the whole surface; the metal potential is fixed or
    follows from charge conservation. Several models may share the
    coverage fields and the metal potential (e.g. a defect and the
    surrounding surface, each with its own ``"Initial_Coverages"``); each
    lumps over its own facets. Supports 2-D, 3-D and axisymmetric meshes.
    Symbols as in :doc:`electrolyte`, with
    :math:`W_i = \int_\Gamma N_i\,\mathrm{d}\Gamma` the lumped weights of
    the interface nodes, :math:`V = E_m - \varphi`, and activities
    :math:`a = \gamma\max(C, C_{floor})/C_{ref}` (H2O has unit activity).

    **Surface steps** (oxidations, with the electrons :math:`n_\kappa` released):

    * R1: Fe + OH- <-> FeOH + e-
    * R2: FeOH + OH- <-> FeOH2 + e-
    * R3: FeOH <-> Fe + Fe2+ + OH- + e-
    * R4: Fe + Cl- <-> FeCl + e-
    * R5: FeCl <-> Fe + Fe2+ + Cl- + e-
    * R6: FeOH + Cl- <-> FeOHCl + e-
    * R7: FeCl + OH- <-> FeOHCl + e-
    * R8: FeOHCl + H+ <-> Fe + Fe2+ + Cl- + H2O
    * R9: FeOH2 + OH- <-> FeOOH + H2O + e-
    * R10: FeOOH + Cl- <-> FeOCl + OH-
    * R11: FeOHCl + OH- <-> FeOCl + H2O + e-
    * R12: FeOOH + 2H2O <-> Fe + Fe(OH)4- + H+
    * R13: FeOCl + 3H2O <-> Fe + Fe(OH)4- + Cl- + 2H+
    * R14: FeOH2 <-> Fe + Fe2+ + 2OH-

    With reactant state :math:`r`, product state :math:`p`, consumed and
    released solutes :math:`s^c`, :math:`s^r`, the forward and backward
    rates (mol/(m^2 s)) are, for electrochemical steps
    (:math:`\eta = V - E^\circ - \Delta u`),

    .. math::

        \nu = \frac{\hat{i}_0}{F}\theta_r\prod a^{s^c}e^{\alpha f\eta}, \qquad
        \nu' = \frac{\hat{i}_0}{F}\theta_p\prod a^{s^r}e^{-(1-\alpha)f\eta},

    and for chemical steps
    :math:`\nu = \hat{k}\theta_r\prod a^{s^c}`,
    :math:`\nu' = \hat{k}\theta_p\prod a^{s^r}\,10^{-\log K}e^{f\Delta u}`;
    exponents are capped at 600. The state shift
    :math:`u_j = \sum_g g_g c_{gj}\sum_l c_{gl}\theta_l - \lambda_j(V - E_{ref,j})`
    holds the Frumkin interactions (``"Frumkin"`` groups :math:`g`) and the
    electrosorption of selected states (``"Electrosorption"``), and
    :math:`\Delta u = u_p - u_r`. Without parallel forms,
    :math:`\hat{i}_0 = \zeta i_0` and :math:`\hat{k} = \zeta k`, with
    the ``"Slow_Down"`` factor :math:`\zeta \leq 1` for the listed steps
    (1 otherwise), which leaves every equilibrium unchanged. Parallel
    forms multiply both directions by the same factor:

    * water form of R1, R7, R9, R11 (``"Water_Pathway"``, reported as
      ``"<step>_water"``): rate prefactor
      :math:`\zeta_w i_0^w K_w^\alpha / a_{OH^-}` (:math:`K_w = 10^{\log_{10}K_w}`),
      added as a separate form that releases H+ instead of consuming OH-;
    * proton-promoted form of R12, R13, R14 (``"Proton_Pathway"``):
      :math:`\hat{k} = \zeta(k + k^H a_{H^+}^m)`.

    **Cathodic reactions** (reductions, net rate positive when cathodic,
    :math:`\eta = V - E^\circ`), with the exchange current weighted by the
    film-covered fraction :math:`\theta_P = \theta_{FeOH2} + \theta_{FeOOH}`,
    :math:`i_{0,j} = i_{0,j}^P(\theta_P + r_j(1-\theta_P))`:

    * ``ORR``, O2 + 2H2O + 4e- <-> 4OH-:
      :math:`\frac{i_0}{4F}(a_{O_2}e^{-\alpha f\eta} - a_{OH^-}^4 e^{(4-\alpha)f\eta})`
    * ``HER_acid``, 2H+ + 2e- <-> H2:
      :math:`\frac{i_0}{2F}(a_{H^+}e^{-\alpha f\eta} - a_{H_2}a_{H^+}^{-1}e^{(2-\alpha)f\eta})`
    * ``HER_base``, 2H2O + 2e- <-> H2 + 2OH-:
      :math:`\frac{i_0}{2F}(e^{-\alpha f\eta} - a_{H_2}a_{OH^-}^2 e^{(2-\alpha)f\eta})`

    **Productions.** Per node, the coverage production is
    :math:`P_{\theta_j} = \sum_\kappa(\delta_{jp(\kappa)} - \delta_{jr(\kappa)})(\nu_\kappa - \nu'_\kappa)`
    over all forms, the solute productions :math:`P_s` follow from the
    stoichiometry of the forms and cathodic reactions (H2O excluded), the
    electrons consumed from the metal are
    :math:`P_e = -\sum_\kappa n_\kappa(\nu_\kappa - \nu'_\kappa) + 4\nu_{ORR} + 2\nu_{HER,a} + 2\nu_{HER,b}`.
    The net exchange of OH- equivalents
    :math:`X = P_{OH^-} - P_{H^+}` is then split by abundance:
    :math:`P_{OH^-} = wX`, :math:`P_{H^+} = -(1-w)X`,
    :math:`w = C_{OH^-}/(C_{OH^-} + C_{H^+})` (floored concentrations),
    equivalent through the water equilibrium, so that no trace species is
    consumed at a rate that does not vanish with its concentration.

    .. rubric:: Fields

    * ``theta_<state>`` for the seven states (``theta_Fe``, ``theta_FeOH``,
      ...): coverages :math:`\theta_j`, CG of order ``Field_Orders["theta"]``
      on the facet group only, solved in step ``Solution_Steps["theta"]``
      (must equal ``Solution_Steps["C"]``); ``theta_<state>Old``: committed
      values.
    * ``E_m`` (name set by ``"Metal_Potential_Field"``): metal potential, a
      global field with ``E_mOld``; only with
      ``"Metal_Potential": "Charge_Conservation"``.
    * Outputs on the facet group (same element as the coverages):
      ``Dissolved_Depth`` (m), ``Anodic_Current_Density``
      (:math:`F\sum_\kappa n_\kappa(\nu_\kappa - \nu'_\kappa)`),
      ``Dissolution_Current_Density``
      (:math:`2F\nu_{Fe^{2+}} + 3F\nu_{Fe^{3+}}`, net rates of R3, R5, R8,
      R14 and of R12, R13), ``ORR_Current_Density`` and
      ``HER_Current_Density`` (cathodic, positive; zero without cathodic
      reactions), all in A/m^2.
    * Uses ``E`` and all dissolved species of :doc:`electrolyte`; the
      species OH-, H+, Cl-, Fe2+ and Fe(OH)4- (and O2, H2 with
      ``"Cathodic"``) must exist. All coupled fields must be scalar CG
      fields of one degree, at most 2.

    .. rubric:: Weak form

    On :math:`\Gamma`, the facet group named by ``domain``, the nodal
    (lumped) contributions are, for every coverage,

    .. math::

        R_{\theta_j} = \sum_i \delta\theta_{j,i}\,W_i\Bigl(N_s\frac{\theta_{j,i} - \theta_{j,n,i}}{\Delta t} - P_{\theta_j,i}\Bigr),

    except that at each node the row of the state with the largest
    coverage is replaced by the site balance
    :math:`W_i N_s/\Delta t\,(\sum_j\theta_{j,i} - 1)` (eliminating one
    coverage instead would lose trace coverages, such as the bare site of
    a passive surface, about 1e-17, to round-off); for every solute
    :math:`s`

    .. math::

        R_{C_s} \mathrel{+}= -\sum_i \delta C_{s,i}\,W_i\,P_{s,i};

    and, with charge conservation, for the global unknown :math:`E_m`

    .. math::

        R_{E_m} \mathrel{+}= F\sum_i W_i P_{e,i} - I_{applied}.

    The Jacobian is the consistent derivative (the bare-site and floor
    clipping give zero derivatives where active). Options:

    * ``"Coverage_Hold": t_hold``: for every step starting at
      :math:`t < t_{hold}`, :math:`P_{\theta_j}` is dropped from the
      coverage rows (the coverages keep their committed values) while the
      reactions still exchange solutes and carry the current.
    * ``"Potential_Hold": t_hold`` (charge conservation only): for steps
      starting at :math:`t < t_{hold}`, the row of :math:`E_m` is
      :math:`E_m - E_{m,n}` instead (a potentiostatic hold; all models
      sharing the metal potential need the same value).

    A zero UFL block over the facet mesh reserves the matrix layout of
    the coverages.

    .. rubric:: Outside the weak form

    * ``initialize_fields`` sets the coverages on this model's nodes to
      ``"Initial_Coverages"`` (bare site: the remainder), ``Dissolved_Depth``
      to zero and :math:`E_m` to ``"Initial_Metal_Potential"``; it raises a
      ``ValueError`` for an empty facet group.
    * ``commit`` adds :math:`\Delta t\,V_{Fe}(\nu_{Fe^{2+}} + \nu_{Fe^{3+}})`
      to ``Dissolved_Depth`` (:math:`V_{Fe}` the molar volume of iron; the
      surface does not move) and copies the coverages and :math:`E_m` to
      the old fields; ``reset_step`` restores them.
    * Global measures (per model name, after every converged step), which
      also update the current-density fields: ``_Anodic_Current``,
      ``_Fe2_Rate``, ``_Fe3_Rate`` (mol/s), ``_Dissolution_Current``,
      ``_Hydroxide_Route_Rate`` (R3 + R12 + R14) and
      ``_Chloride_Route_Rate`` (R5 + R8 + R13), ``_ORR_Current``,
      ``_HER_Current`` (with cathodic reactions), ``_Coverage_<state>``
      (averages) and ``_Metal_Potential``; currents in A per m^(dim-1).
    * :meth:`Net_Current` gives the net cathodic current at a trial metal
      potential, to start from a balanced :math:`E_m`.
    * With ``"Solute_Gibbs"``, construction checks that the constants close
      every reaction cycle (tolerance 1e-6 V).
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            Solution_Steps = {"C": 0, "E": 0, "theta": 0}  # REQUIRED: coverages in the electrolyte step
            Field_Orders = {"C": 1, "E": 1, "theta": 1}    # REQUIRED: theta order (equal to the others, at most 2)
            C_ref = 1000.0                        # OPTIONAL (default 1.0): reference concentration (mol/m^3)
            Activity_Parameters = {...}           # REQUIRED unless Activity_Model is "ideal", see activity
            Models = {
                "Surface": {
                    "Type": "ElectroChemistry:PassivatingSurface",  # REQUIRED
                    "domain": "bottom",           # REQUIRED: facet group of the metal surface
                    "Activity_Model": "ideal",    # REQUIRED: "ideal", "Davies" or "SIT"
                    "Metal_Potential": "Charge_Conservation",  # REQUIRED: fixed E_m (V_SHE) or "Charge_Conservation"
                    "Metal_Potential_Field": "E_m",  # OPTIONAL (default "E_m"): shared by models on one metal
                    "Initial_Metal_Potential": -0.4,  # OPTIONAL (default 0.0): initial E_m (V_SHE)
                    "Applied_Current": 0.0,       # OPTIONAL (default 0.0): current into the metal (A per m^(dim-1))
                    "Potential_Hold": 0.0,        # OPTIONAL (default no hold): hold E_m until this time (s)
                    "Coverage_Hold": 0.0,         # OPTIONAL (default no hold): hold the coverages until this time (s)
                    "Site_Density": 2.0e-5,       # REQUIRED: N_s (mol/m^2)
                    "Initial_Coverages": {"FeOH2": 0.1, "FeOOH": 0.9},  # OPTIONAL (default {}): covered states, rest bare
                    "Temperature": 298.15,        # OPTIONAL (default 298.15): T (K)
                    "Concentration_Floor": 1.0e-60,  # OPTIONAL (default 1e-60): floor of the activities (mol/m^3)
                    "log10_Kw": -14.00137823281,  # OPTIONAL (default -14.0): water constant of the water forms
                    "Reactions": {                # REQUIRED: all of R1-R14.
                        # Electrochemical steps: E0 (V_SHE), i0 (A/m^2), alpha (-), all REQUIRED.
                        # Chemical steps (R8, R10, R12, R13, R14): logK (-), k (mol/(m^2 s)), both REQUIRED.
                        "R1": {"E0": -0.91, "i0": 96485332.12, "alpha": 0.5},
                        "R2": {"E0": -0.95, "i0": 96485.33212, "alpha": 0.7},
                        "R3": {"E0": -0.0302361789787038, "i0": 4824.266606, "alpha": 0.3},
                        "R4": {"E0": -0.32, "i0": 0.09648533212, "alpha": 0.5},
                        "R5": {"E0": -0.6202361789787038, "i0": 48242666060.0, "alpha": 0.3},
                        "R6": {"E0": 0.085, "i0": 964853321.2, "alpha": 0.5},
                        "R7": {"E0": -0.505, "i0": 964853321.2, "alpha": 0.5},
                        "R8": {"logK": 15.949272852864448, "k": 3.0},
                        "R9": {"E0": -0.325, "i0": 964853321.2, "alpha": 0.5},
                        "R10": {"logK": -1.65, "k": 1.0e4},
                        "R11": {"E0": -1.2623870730185591, "i0": 964853321.2, "alpha": 0.5},
                        "R12": {"logK": -13.652298695197363, "k": 7.0e-10},
                        "R13": {"logK": -26.0036769280079, "k": 1.95e-6},
                        "R14": {"logK": -15.547226700554607, "k": 0.0},
                    },
                    "Water_Pathway": {            # OPTIONAL (default none): R1, R7, R9, R11
                        "R1": {"i0_w": 964853321.2},  # REQUIRED per entry: i0_w (A/m^2)
                    },
                    "Proton_Pathway": {           # OPTIONAL (default none): R12, R13, R14
                        "R14": {"k_H": 50.0, "m": 1.0},  # REQUIRED per entry: k_H >= 0 (mol/(m^2 s)), m > 0 (-)
                    },
                    "Frumkin": {                  # OPTIONAL (default none): interaction groups
                        "Cl": {"g": 0.2,          # REQUIRED per group: g (V), states with weights c (-)
                               "states": {"FeCl": 1, "FeOHCl": 1, "FeOCl": 1}},
                    },
                    "Electrosorption": {          # OPTIONAL (default none)
                        "FeOCl": {"gamma": 0.19, "E_ref": 0.241},  # REQUIRED per state: lambda (-), E_ref (V_SHE)
                    },
                    "Cathodic": {                 # OPTIONAL (default none): all three or none; needs species O2, H2
                        # i0 (A/m^2, passive surface), alpha (-), E0 (V_SHE): REQUIRED; ratio r_j: OPTIONAL (default 1.0)
                        "ORR": {"i0": 2.84e-3, "alpha": 0.4695, "E0": 0.40057902222827524, "ratio": 10.0},
                        "HER_acid": {"i0": 0.044, "alpha": 0.51, "E0": 0.0},
                        "HER_base": {"i0": 5.51e-5, "alpha": 0.51, "E0": -0.8283124309568889},
                    },
                    "Slow_Down": {"factor": 1.0e-3,  # OPTIONAL (default no slow-down): zeta in (0, 1]
                                  "steps": ["R5", "R6", "R1_water"]},  # steps or water forms it applies to
                    "Solute_Gibbs": {             # OPTIONAL (default no check): J/mol, all six solutes
                        "OH-": -157220.0, "H+": 0.0, "Cl-": -131217.0, "H2O": -237140.0,
                        "Fe2+": -90719.0, "Fe(OH)4-": -841693.0,
                    },
                },
            }
    """

    MODEL_TYPE = "ElectroChemistry:PassivatingSurface"
    SUPPORTS_AXISYMMETRY = True

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        settings = self.params.Models[self.name]

        self.f = _FARADAY / (_GAS_CONSTANT * settings.get("Temperature", 298.15))
        self.site_density = settings["Site_Density"]
        self.C_Ref = getattr(self.params, "C_ref", 1.0)
        self.C_Floor = settings.get("Concentration_Floor", 1.0e-60)
        self._read_reactions(settings)
        self._read_shifts(settings)
        self._read_cathodic(settings)
        if "Solute_Gibbs" in settings:
            self._check_cycles(settings["Solute_Gibbs"])
        self.initial_coverages = self._read_initial_coverages(settings.get("Initial_Coverages", {}))
        self.coverage_hold = float(settings.get("Coverage_Hold", -np.inf))
        if np.isnan(self.coverage_hold):
            raise ValueError(f"PassivatingSurface '{name}': Coverage_Hold must be a time (s), not NaN.")

        # Dissolved species of the electrolyte (all enter the activity coefficients).
        species = self.params.Species
        self.dissolved = [s for s in species if not species[s].get("Solid", False)]
        missing = [s for s in _SURFACE_SOLUTES + (_CATHODIC_SOLUTES if self.cathodic else ()) if s not in self.dissolved]
        if missing:
            raise ValueError(f"PassivatingSurface '{name}' needs the electrolyte species {missing}.")
        if "Activity_Model" not in settings:
            raise ValueError(f"PassivatingSurface '{name}' requires the setting 'Activity_Model' ('ideal', 'Davies' or 'SIT').")
        self.activity = ActivityModel(settings["Activity_Model"], getattr(self.params, "Activity_Parameters", None),
                                      {s: species[s]["Charge"] for s in self.dissolved})

        self.coverage_fields = [f"theta_{state}" for state in STATES]
        self._rows = self.coverage_fields + list(_SURFACE_SOLUTES) + (list(_CATHODIC_SOLUTES) if self.cathodic else [])
        self._fields = self.coverage_fields + self.dissolved + ["E"]

        self.My_Step = params.Solution_Steps["C"]
        if params.Solution_Steps["theta"] != self.My_Step:
            raise ValueError("PassivatingSurface requires the coverages to be solved in one step with the electrolyte.")
        order = params.Field_Orders["theta"]
        for field_name in self.coverage_fields:
            mesh.Add_Field(field_name, f"Coverage {field_name[6:]} [-]", "CG", order, domain=self.domain)
            mesh.Add_Field(f"{field_name}Old", f"Old coverage {field_name[6:]} [-]", "CG", order, domain=self.domain)
            mesh.Add_TestFunction(field_name, self.My_Step)
        mesh.Add_Field("Dissolved_Depth", "Dissolved depth [m]", "CG", order, domain=self.domain)
        for field_name, description in _CURRENT_DENSITY_FIELDS.items():
            mesh.Add_Field(field_name, description, "CG", order, domain=self.domain)

        metal_potential = settings["Metal_Potential"]
        self.charge_conservation = metal_potential == _CHARGE_CONSERVATION
        if self.charge_conservation:
            self.E_m_field = settings.get("Metal_Potential_Field", "E_m")
            mesh.Add_Global_Field(self.E_m_field, "Metal Potential [V]")
            mesh.Add_Global_Field(f"{self.E_m_field}Old", "Old Metal Potential [V]")
            mesh.Add_TestFunction(self.E_m_field, self.My_Step)
            self.E_m0 = settings.get("Initial_Metal_Potential", 0.0)
            self.applied_current = settings.get("Applied_Current", 0.0)
            self.potential_hold = float(settings.get("Potential_Hold", -np.inf))
        elif isinstance(metal_potential, str):
            raise ValueError(f"Metal_Potential must be a number or '{_CHARGE_CONSERVATION}', not '{metal_potential}'.")
        else:
            self.E_m = float(metal_potential)

        for measure in self._measure_names():
            self.params.Global_Measures[measure] = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Return zero coverage blocks; the capacity and all reactions are added in :meth:`Add_KF`."""
        self._dt = self._weak_constant_value(dt_constant)
        if step != self.My_Step:
            return None
        facet_mesh = self.mesh.Get_Field(self.coverage_fields[0]).function_space.mesh
        zero = fem.Constant(facet_mesh, 0.0)
        dGamma = Measure("dx", domain=facet_mesh)
        K, F = 0, 0
        for row in self.coverage_fields:
            test = self.mesh.Get_TestFunction(row, self.My_Step)
            F += zero * test * dGamma
            for column in self.coverage_fields:
                K += zero * test * self.mesh.Get_Trial_Function(column, self.My_Step) * dGamma
        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        """Store the time step used by the lumped coverage capacity."""
        self._dt = self._weak_constant_value(dt_constant)
        return False

    def Add_KF(self, K, F, step: int) -> None:
        """Add the lumped surface terms (and the charge balance) to the residual and tangent."""
        if step != self.My_Step:
            return
        weights, dofs, vals = self._interface_state(step)
        production, derivatives = self.Production_Rates(vals)
        # Lumped capacity N_s d(theta)/dt, as a negative production.
        capacity = self.site_density / self._dt
        holding_coverages = getattr(self.params, "t", 0.0) < self.coverage_hold
        for row in self.coverage_fields:
            if holding_coverages:
                # Coverage hold: the states keep their committed values; the reactions still exchange species
                # with the electrolyte and carry the current.
                production[row] = -capacity * (vals[row] - vals[f"{row}Old"])
                derivatives[row] = {column: np.zeros_like(d) for column, d in derivatives[row].items()}
                derivatives[row][row] = derivatives[row][row] - capacity
                continue
            production[row] = production[row] - capacity * (vals[row] - vals[f"{row}Old"])
            derivatives[row][row] = derivatives[row][row] - capacity
        # Site balance in place of the balance of the dominant state of each node, scaled like the capacity.
        theta = np.stack([vals[row] for row in self.coverage_fields])
        dominant = np.argmax(theta, axis=0)
        site_balance = capacity * (1.0 - theta.sum(axis=0))
        for i, row in enumerate(self.coverage_fields):
            replaced = dominant == i
            production[row] = np.where(replaced, site_balance, production[row])
            for column in derivatives[row]:
                if column in self.coverage_fields:
                    derivatives[row][column] = np.where(replaced, -capacity, derivatives[row][column])
                else:
                    derivatives[row][column] = np.where(replaced, 0.0, derivatives[row][column])
        columns = list(self._fields)
        if self.charge_conservation:
            columns.append(self.E_m_field)
            charge_dof = self.mesh.Get_Global_Field_Dof(self.E_m_field, step)

        with F.localForm() as F_local:
            for row in self._rows:
                np.add.at(F_local.array, dofs[row], -weights * production[row])
        holding = self.charge_conservation and getattr(self.params, "t", 0.0) < self.potential_hold
        if holding:
            # Potentiostatic hold: E_m stays at its committed value (one row entry per model, owner rank).
            if comm.rank == 0:
                E_m_old = self.mesh.Get_Global_Value(f"{self.E_m_field}Old")
                F.setValues([charge_dof], [self.mesh.Get_Global_Value(self.E_m_field) - E_m_old], addv=PETSc.InsertMode.ADD_VALUES)
        elif self.charge_conservation:
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
            if holding:
                # Same sparsity as without the hold (zeros), so the matrix layout is kept at the release;
                # the hold row d(E_m - E_m,old)/dE_m = 1 is added once.
                charge_block = np.zeros_like(charge_block)
                if comm.rank == 0:
                    K.setValues([charge_dof], [charge_dof], [[1.0]], addv=PETSc.InsertMode.ADD)
            blocks = np.concatenate([blocks, charge_block[:, None, :]], axis=1)
        for node_rows, columns_of_node, block in zip(rows, node_columns, blocks):
            K.setValues(node_rows, columns_of_node, block, addv=PETSc.InsertMode.ADD)

    def Production_Rates(self, vals):
        """Return nodal production rates and their derivatives.

        Parameters
        ----------
        vals : dict
            Nodal values of the coverage fields ``"theta_<state>"``, of all
            dissolved species, of ``"E"`` and, with charge
            conservation, of the metal potential field.

        Returns
        -------
        tuple
            ``(production, derivatives)``: per coverage field
            :math:`N_s \\dot\\theta` from the reactions (mol/(m^2 s); the
            site balance is applied in :meth:`Add_KF`), per solute its release
            into the electrolyte (mol/(m^2 s)), and ``"charge"``, the
            electrons consumed from the metal (mol/(m^2 s)); and per entry
            and per input field the derivative of that production.
        """
        rates = self.Reaction_Rates(vals)
        columns = list(self._fields) + ([self.E_m_field] if self.charge_conservation else [])
        zero = np.zeros_like(vals["E"])
        production = {row: zero.copy() for row in self._rows + ["charge"]}
        derivatives = {row: {column: zero.copy() for column in columns} for row in self._rows + ["charge"]}

        def add(row, coefficient, rate, d_rate):
            production[row] += coefficient * rate
            for column, d in d_rate.items():
                derivatives[row][column] += coefficient * d

        for key, step in self._forms():
            rate, d_rate = rates[key]
            add(f"theta_{step.reactant}", -1.0, rate, d_rate)
            add(f"theta_{step.product}", 1.0, rate, d_rate)
            for solute, nu in step.released.items():
                if solute != "H2O":
                    add(solute, nu, rate, d_rate)
            for solute, nu in step.consumed.items():
                if solute != "H2O":
                    add(solute, -nu, rate, d_rate)
            add("charge", -float(step.electrons), rate, d_rate)

        if self.cathodic:
            for reaction, coefficients in (("ORR", {"O2": -1.0, "OH-": 4.0, "charge": 4.0}),
                                           ("HER_acid", {"H+": -2.0, "H2": 1.0, "charge": 2.0}),
                                           ("HER_base", {"OH-": 2.0, "H2": 1.0, "charge": 2.0})):
                rate, d_rate = rates[reaction]
                for row, coefficient in coefficients.items():
                    add(row, coefficient, rate, d_rate)

        # Split the net exchange of OH- equivalents (OH- released minus H+ released) between OH- and H+
        # by their abundance, w = C_OH / (C_OH + C_H): equivalent through the water equilibrium, but no
        # trace species (OH- in acid, H+ in base) is consumed at a rate the water equilibrium must resupply.
        C_OH = np.maximum(vals["OH-"], self.C_Floor)
        C_H = np.maximum(vals["H+"], self.C_Floor)
        w = C_OH / (C_OH + C_H)
        d_w = {"OH-": (1.0 - w) / (C_OH + C_H) * (vals["OH-"] > self.C_Floor),
               "H+": -w / (C_OH + C_H) * (vals["H+"] > self.C_Floor)}
        exchange = production["OH-"] - production["H+"]
        d_exchange = {column: derivatives["OH-"][column] - derivatives["H+"][column] for column in columns}
        production["OH-"] = w * exchange
        production["H+"] = -(1.0 - w) * exchange
        for column in columns:
            derivatives["OH-"][column] = w * d_exchange[column] + exchange * d_w.get(column, 0.0)
            derivatives["H+"][column] = -(1.0 - w) * d_exchange[column] + exchange * d_w.get(column, 0.0)
        return production, derivatives

    def Reaction_Rates(self, vals) -> dict:
        """Return per reaction form (``"R1"`` ... ``"R14"``, ``"<step>_water"``) and cathodic reaction ``(net forward rate, {field: derivative})``.

        Net forward rates in mol/(m^2 s) at the nodes; surface steps are
        written as oxidations (positive: anodic), cathodic reactions as
        reductions (positive: cathodic).
        """
        theta = self._full_coverages(vals)
        V = self._metal_potential_values(vals) - vals["E"]
        ln_a, d_ln_a = self._log_activities(vals)
        n_nodes = V.size

        # State shifts u (Frumkin + electrosorption) and their differences Delta u = u_p - u_r.
        u = self.interaction_matrix @ theta - self.lambdas[:, None] * (V[None, :] - self.E_ref[:, None])
        rates = {}
        for name, step in STEPS.items():
            r, p = STATES.index(step.reactant), STATES.index(step.product)
            constants = self.reactions[name]
            du = u[p] - u[r]
            d_du_d_theta = self.interaction_matrix[p] - self.interaction_matrix[r]       # over the 7 states
            d_du_dV = -(self.lambdas[p] - self.lambdas[r])
            ln_A_c = sum(nu * ln_a[s] for s, nu in step.consumed.items() if s != "H2O")
            ln_A_r = sum(nu * ln_a[s] for s, nu in step.released.items() if s != "H2O")

            # Pre-factor(s): k_hat (mol/(m^2 s)) and d k_hat / d ln a, per form of the step. The water
            # form of an electrochemical step is reported separately (key "<step>_water"): it has the
            # same states and electrons but releases H+ instead of consuming OH-.
            if step.electrons:
                alpha = constants["alpha"]
                forms = [(name, np.full(n_nodes, constants["i0"] / _FARADAY * constants["zeta"]), {})]
                if name in self.water_pathway:
                    i0_w, zeta_w = self.water_pathway[name]
                    water = i0_w / _FARADAY * zeta_w * 10.0 ** (alpha * self.log10_Kw) * np.exp(-ln_a["OH-"])
                    forms.append((f"{name}_water", water, {"OH-": -water}))
                eta = V - constants["E0"] - du
                exponent_fw = np.minimum(alpha * self.f * eta, _MAX_EXPONENT)
                exponent_bw = np.minimum(-(1.0 - alpha) * self.f * eta, _MAX_EXPONENT)
                d_ln_fw_d_theta = -alpha * self.f * d_du_d_theta
                d_ln_bw_d_theta = (1.0 - alpha) * self.f * d_du_d_theta
                d_ln_fw_dV = alpha * self.f * (1.0 - d_du_dV)
                d_ln_bw_dV = -(1.0 - alpha) * self.f * (1.0 - d_du_dV)
            else:
                k_hat = np.full(n_nodes, constants["k"])
                d_khat = {}
                if name in self.proton_pathway:
                    k_H, m = self.proton_pathway[name]
                    proton = k_H * np.exp(m * ln_a["H+"])
                    k_hat = k_hat + proton
                    d_khat["H+"] = m * proton * constants["zeta"]
                forms = [(name, k_hat * constants["zeta"], d_khat)]
                exponent_fw = np.zeros(n_nodes)
                exponent_bw = np.minimum(-constants["logK"] * _LN10 + self.f * du, _MAX_EXPONENT)
                d_ln_fw_d_theta = np.zeros(len(STATES))
                d_ln_bw_d_theta = self.f * d_du_d_theta
                d_ln_fw_dV = 0.0
                d_ln_bw_dV = self.f * d_du_dV

            base_fw = np.exp(ln_A_c + exponent_fw)
            base_bw = np.exp(ln_A_r + exponent_bw)
            for key, k_hat, d_khat in forms:
                forward = k_hat * base_fw * theta[r]
                backward = k_hat * base_bw * theta[p]

                # Derivatives: full coverages (bare site through the site balance), potential, log-activities.
                d_full = (forward[None, :] * d_ln_fw_d_theta[:, None] - backward[None, :] * d_ln_bw_d_theta[:, None])
                d_full[r] += k_hat * base_fw
                d_full[p] -= k_hat * base_bw
                d = {f"theta_{state}": d_full[i] for i, state in enumerate(STATES)}
                d_dV = forward * d_ln_fw_dV - backward * d_ln_bw_dV
                d_ln_a_rate = {}
                for s, nu in step.consumed.items():
                    if s != "H2O":
                        d_ln_a_rate[s] = d_ln_a_rate.get(s, 0.0) + nu * forward
                for s, nu in step.released.items():
                    if s != "H2O":
                        d_ln_a_rate[s] = d_ln_a_rate.get(s, 0.0) - nu * backward
                for s, dk in d_khat.items():
                    d_ln_a_rate[s] = d_ln_a_rate.get(s, 0.0) + dk * (base_fw * theta[r] - base_bw * theta[p])
                self._add_potential_and_activity_derivatives(d, d_dV, d_ln_a_rate, d_ln_a)
                rates[key] = (forward - backward, d)

        if self.cathodic:
            theta_P = sum(theta[STATES.index(state)] for state in _PROTECTED)
            for reaction in _CATHODIC:
                rates[reaction] = self._cathodic_rate(reaction, theta_P, V, ln_a, d_ln_a)
        return rates

    def Net_Current(self, E_m: float) -> float:
        """Return the net cathodic current (A per m^(dim-1)) of this surface at the current fields for a trial metal potential.

        :math:`F \\int_\\Gamma (\\text{electrons consumed}) \\, d\\Gamma`, positive when
        the cathodic reactions dominate. Useful to start a charge-conserving
        simulation from a balanced metal potential (the sum over all
        surface models sharing it plus the applied current is zero): the
        metal potential has no capacitance, so an initial state far from
        the balance (e.g. an active defect on a passive plate) puts the
        first Newton step far outside the convergence radius.
        """
        weights, _dofs, vals = self._interface_state(self.My_Step)
        if self.charge_conservation:
            vals = {**vals, self.E_m_field: np.full(weights.size, float(E_m))}
        else:
            raise ValueError("Net_Current needs a charge-conserving metal potential (Metal_Potential = 'Charge_Conservation').")
        production, _derivatives = self.Production_Rates(vals)
        return comm.allreduce(_FARADAY * float(np.sum(weights * production["charge"])), op=MPI.SUM)

    def Update_Global_Measures(self) -> None:
        """Report the integrated currents and rates, the average coverages and the metal potential."""
        weights, _dofs, vals = self._interface_state(self.My_Step)
        rates = self.Reaction_Rates(vals)
        length = comm.allreduce(weights.sum(), op=MPI.SUM)

        def integral(values):
            return comm.allreduce(float(np.sum(weights * values)), op=MPI.SUM)

        measures = self.params.Global_Measures
        net = {name: rate for name, (rate, _d) in rates.items()}
        fe2 = sum(net[name] for name in _FE2_STEPS)
        fe3 = sum(net[name] for name in _FE3_STEPS)
        measures[f"{self.name}_Anodic_Current"] = _FARADAY * integral(sum(step.electrons * net[key] for key, step in self._forms()))
        measures[f"{self.name}_Fe2_Rate"] = integral(fe2)
        measures[f"{self.name}_Fe3_Rate"] = integral(fe3)
        measures[f"{self.name}_Dissolution_Current"] = _FARADAY * integral(2.0 * fe2 + 3.0 * fe3)
        for route, steps in _ROUTES.items():
            measures[f"{self.name}_{route}_Route_Rate"] = integral(sum(net[name] for name in steps))
        if self.cathodic:
            measures[f"{self.name}_ORR_Current"] = 4.0 * _FARADAY * integral(net["ORR"])
            measures[f"{self.name}_HER_Current"] = 2.0 * _FARADAY * integral(net["HER_acid"] + net["HER_base"])
        theta = self._full_coverages(vals)
        for i, state in enumerate(STATES):
            measures[f"{self.name}_Coverage_{state}"] = integral(theta[i]) / length
        measures[f"{self.name}_Metal_Potential"] = self._metal_potential()

        # Local current densities (output fields), at the nodes of this model's facets.
        densities = {"Anodic_Current_Density": _FARADAY * sum(step.electrons * net[key] for key, step in self._forms()),
                     "Dissolution_Current_Density": _FARADAY * (2.0 * fe2 + 3.0 * fe3),
                     "ORR_Current_Density": 4.0 * _FARADAY * net["ORR"] if self.cathodic else np.zeros(weights.size),
                     "HER_Current_Density": (2.0 * _FARADAY * (net["HER_acid"] + net["HER_base"]) if self.cathodic
                                             else np.zeros(weights.size))}
        for field_name, values in densities.items():
            self.mesh.Set_Interface_Values(self.domain, field_name, values)

    def initialize_fields(self) -> None:
        """Set the coverages and the metal potential to their initial values.

        Raises ``ValueError`` if the model's facet group is empty (on all ranks).
        """
        weights, _dofs, _vals = self.mesh.Get_Interface_Weights_Dofs(self.domain, [self.coverage_fields[0]], None)
        if comm.allreduce(weights.size, op=MPI.SUM) == 0:
            raise ValueError(f"PassivatingSurface '{self.name}': the facet group '{self.domain}' has no facets.")
        # Set on this model's own interface nodes: models sharing the coverage fields (e.g. a defect and
        # the surrounding surface) each keep their own initial state (the nodes on their common edge take
        # the values of the model initialised last).
        for state in STATES:
            values = np.full(weights.size, self.initial_coverages[state])
            self.mesh.Set_Interface_Values(self.domain, f"theta_{state}", values)
            self.mesh.Set_Interface_Values(self.domain, f"theta_{state}Old", values)
        self.mesh.Set_Interface_Values(self.domain, "Dissolved_Depth", np.zeros(weights.size))
        if self.charge_conservation:
            self.mesh.Constant_Field(self.E_m_field, self.E_m0)
            self.mesh.Constant_Field(f"{self.E_m_field}Old", self.E_m0)

    def commit(self) -> None:
        """Store the converged state and update the dissolved depth."""
        _weights, _dofs, vals = self._interface_state(self.My_Step)
        rates = self.Reaction_Rates(vals)
        dissolution = sum(rates[name][0] for name in _FE2_STEPS + _FE3_STEPS)
        _weights, _dofs, depth = self.mesh.Get_Interface_Weights_Dofs(self.domain, ["Dissolved_Depth"], None)
        self.mesh.Set_Interface_Values(self.domain, "Dissolved_Depth", depth["Dissolved_Depth"] + self._dt * _FE_MOLAR_VOLUME * dissolution)
        for field_name in self.coverage_fields:
            self.mesh.Copy_Field(field_name, f"{field_name}Old", domain=self.domain)
        if self.charge_conservation:
            self.mesh.Copy_Field(self.E_m_field, f"{self.E_m_field}Old")

    def reset_step(self) -> None:
        """Restore the last committed coverages and metal potential after a rejected step."""
        for field_name in self.coverage_fields:
            self.mesh.Copy_Field(f"{field_name}Old", field_name, domain=self.domain)
        if self.charge_conservation:
            self.mesh.Copy_Field(f"{self.E_m_field}Old", self.E_m_field)

    # ------------------------------------------------------------------
    # Internal Helpers: settings
    # ------------------------------------------------------------------

    def _read_reactions(self, settings) -> None:
        """PRIVATE: Read and validate the constants of R1-R14, the parallel forms and the slow-down factor."""
        reactions = settings["Reactions"]
        unknown = set(reactions) - set(STEPS)
        missing = set(STEPS) - set(reactions)
        if unknown or missing:
            raise ValueError(f"PassivatingSurface 'Reactions' must list exactly {list(STEPS)}; "
                             f"unknown {sorted(unknown)}, missing {sorted(missing)}.")
        slow = settings.get("Slow_Down", {"factor": 1.0, "steps": []})
        zeta = float(slow["factor"])
        if not 0.0 < zeta <= 1.0:
            raise ValueError("Slow_Down factor must lie in (0, 1].")
        slow_steps = set(slow["steps"])
        valid_slow = set(STEPS) | {f"{name}_water" for name in _WATER_FORMS}
        if slow_steps - valid_slow:
            raise ValueError(f"Slow_Down steps {sorted(slow_steps - valid_slow)} are not steps or water forms.")

        self.reactions = {}
        for name, step in STEPS.items():
            entry = reactions[name]
            if step.electrons:
                keys = ("E0", "i0", "alpha")
            else:
                keys = ("logK", "k")
            absent = [key for key in keys if key not in entry]
            if absent:
                raise ValueError(f"PassivatingSurface step {name} needs {list(keys)}; missing {absent}.")
            constants = {key: float(entry[key]) for key in keys}
            constants["zeta"] = zeta if name in slow_steps else 1.0
            self.reactions[name] = constants

        self.log10_Kw = float(settings.get("log10_Kw", -14.0))
        self.water_pathway = {}
        for name, entry in settings.get("Water_Pathway", {}).items():
            if name not in _WATER_FORMS:
                raise ValueError(f"Water_Pathway: {name} has no water form; expected one of {_WATER_FORMS}.")
            # (i0_w, slow-down factor of the water form)
            self.water_pathway[name] = (float(entry["i0_w"]), zeta if f"{name}_water" in slow_steps else 1.0)
        self.proton_pathway = {}
        for name, entry in settings.get("Proton_Pathway", {}).items():
            if name not in _PROTON_FORMS:
                raise ValueError(f"Proton_Pathway: {name} has no proton-promoted form; expected one of {_PROTON_FORMS}.")
            if entry["k_H"] < 0.0 or entry["m"] <= 0.0:
                raise ValueError(f"Proton_Pathway {name}: needs k_H >= 0 and m > 0.")
            self.proton_pathway[name] = (float(entry["k_H"]), float(entry["m"]))

    def _read_shifts(self, settings) -> None:
        """PRIVATE: Build the Frumkin interaction matrix and the electrosorption valencies."""
        n = len(STATES)
        matrix = np.zeros((n, n))
        for group, spec in settings.get("Frumkin", {}).items():
            unknown = set(spec["states"]) - set(STATES)
            if unknown:
                raise ValueError(f"Frumkin group '{group}': unknown states {sorted(unknown)}.")
            c = np.array([spec["states"].get(state, 0.0) for state in STATES], dtype=float)
            matrix += float(spec["g"]) * np.outer(c, c)
        self.interaction_matrix = matrix
        self.lambdas = np.zeros(n)
        self.E_ref = np.zeros(n)
        for state, spec in settings.get("Electrosorption", {}).items():
            if state not in STATES:
                raise ValueError(f"Electrosorption: unknown state '{state}'.")
            self.lambdas[STATES.index(state)] = float(spec["gamma"])
            self.E_ref[STATES.index(state)] = float(spec["E_ref"])

    def _read_cathodic(self, settings) -> None:
        """PRIVATE: Read the cathodic reactions (all three, or none)."""
        cathodic = settings.get("Cathodic", {})
        if cathodic and set(cathodic) != set(_CATHODIC):
            raise ValueError(f"Cathodic must list all of {_CATHODIC} (or be omitted).")
        self.cathodic = {}
        for reaction, entry in cathodic.items():
            self.cathodic[reaction] = {key: float(entry[key]) for key in ("i0", "alpha", "E0")}
            self.cathodic[reaction]["ratio"] = float(entry.get("ratio", 1.0))

    def _read_initial_coverages(self, coverages: dict) -> dict:
        """PRIVATE: Validate the initial coverages of the covered states; the bare site takes the remainder."""
        unknown = set(coverages) - set(_COVERED)
        if unknown:
            raise ValueError(f"Initial_Coverages: unknown or bare states {sorted(unknown)}; give the covered states only.")
        values = {state: float(coverages.get(state, 0.0)) for state in _COVERED}
        if any(v < 0.0 for v in values.values()) or sum(values.values()) > 1.0 + 1e-12:
            raise ValueError("Initial_Coverages must be non-negative and sum to at most one.")
        values["Fe"] = 1.0 - sum(values.values())
        return values

    def _check_cycles(self, gibbs: dict) -> None:
        """PRIVATE: Reject step constants whose reaction cycles do not close (tolerance 1e-6 V)."""
        RT = _FARADAY / self.f

        def step_energy(name):
            step, c = STEPS[name], self.reactions[name]
            return step.electrons * _FARADAY * c["E0"] if step.electrons else -RT * _LN10 * c["logK"]

        def solutes(amounts):
            return sum(nu * gibbs[s] for s, nu in amounts.items())

        # State energies along a spanning tree from the bare site, then every step checked.
        G = {"Fe": 0.0}
        queue = deque(["Fe"])
        while queue:
            state = queue.popleft()
            for name, step in STEPS.items():
                if step.reactant == state and step.product not in G:
                    G[step.product] = G[state] + step_energy(name) + solutes(step.consumed) - solutes(step.released)
                    queue.append(step.product)
                elif step.product == state and step.reactant not in G:
                    G[step.reactant] = G[state] - step_energy(name) - solutes(step.consumed) + solutes(step.released)
                    queue.append(step.reactant)
        for name, step in STEPS.items():
            mismatch = (G[step.product] + solutes(step.released) - G[step.reactant] - solutes(step.consumed)
                        - step_energy(name)) / _FARADAY
            if abs(mismatch) > 1.0e-6:
                raise ValueError(f"PassivatingSurface: the constants of {name} do not close the reaction cycles "
                                 f"({mismatch:+.2e} V); use a consistent set of E0 / logK.")

    # ------------------------------------------------------------------
    # Internal Helpers: rates
    # ------------------------------------------------------------------

    def _forms(self):
        """PRIVATE: Return ``(rate key, step)`` of every reaction form: R1-R14, plus the water forms (releasing H+)."""
        forms = []
        for name, step in STEPS.items():
            forms.append((name, step))
            if name in self.water_pathway:
                released = {"H+": 1}
                forms.append((f"{name}_water", _Step(step.reactant, step.product, step.electrons, {}, released)))
        return forms

    def _full_coverages(self, vals) -> np.ndarray:
        """PRIVATE: Return the coverages of all seven states, states x nodes."""
        return np.stack([vals[f"theta_{state}"] for state in STATES])

    def _log_activities(self, vals):
        """PRIVATE: Return ``ln a`` of the solutes used by the reactions and ``d ln a_s / d C_k``."""
        ln_gamma, d_ln_gamma = self.activity.Ln_Gamma({s: vals[s] for s in self.dissolved})
        ln_a, d_ln_a = {}, {}
        for s in _SURFACE_SOLUTES + (_CATHODIC_SOLUTES if self.cathodic else ()):
            clipped = np.maximum(vals[s], self.C_Floor)
            ln_a[s] = np.log(clipped / self.C_Ref) + ln_gamma[s]
            d = dict(d_ln_gamma[s])
            d[s] = d.get(s, 0.0) + (vals[s] > self.C_Floor) / clipped
            d_ln_a[s] = d
        return ln_a, d_ln_a

    def _add_potential_and_activity_derivatives(self, d, d_dV, d_ln_a_rate, d_ln_a) -> None:
        """PRIVATE: Add d/dE (= -d/dV), d/dE_m (= d/dV) and the concentration derivatives (chain rule) to ``d``."""
        d["E"] = -d_dV
        if self.charge_conservation:
            d[self.E_m_field] = d_dV
        for s, d_rate in d_ln_a_rate.items():
            for k, d_ln in d_ln_a[s].items():
                d[k] = d.get(k, 0.0) + d_rate * d_ln

    def _cathodic_rate(self, reaction, theta_P, V, ln_a, d_ln_a):
        """PRIVATE: Return ``(net cathodic rate, derivatives)`` of one cathodic reaction (mol/(m^2 s))."""
        c = self.cathodic[reaction]
        weight = theta_P + c["ratio"] * (1.0 - theta_P)
        alpha, eta = c["alpha"], V - c["E0"]
        if reaction == "ORR":
            n, ln_fw_a, ln_bw_a = 4.0, {"O2": 1.0}, {"OH-": 4.0}
        elif reaction == "HER_acid":
            n, ln_fw_a, ln_bw_a = 2.0, {"H+": 1.0}, {"H2": 1.0, "H+": -1.0}
        else:
            n, ln_fw_a, ln_bw_a = 2.0, {}, {"H2": 1.0, "OH-": 2.0}
        scale = c["i0"] / (n * _FARADAY)
        exponent_fw = np.minimum(-alpha * self.f * eta, _MAX_EXPONENT)
        exponent_bw = np.minimum((n - alpha) * self.f * eta, _MAX_EXPONENT)
        forward = scale * weight * np.exp(sum((nu * ln_a[s] for s, nu in ln_fw_a.items()), np.zeros_like(V)) + exponent_fw)
        backward = scale * weight * np.exp(sum((nu * ln_a[s] for s, nu in ln_bw_a.items()), np.zeros_like(V)) + exponent_bw)
        rate = forward - backward

        d = {f"theta_{state}": np.zeros_like(V) for state in STATES}
        d_weight = (1.0 - c["ratio"]) * rate / weight
        for state in _PROTECTED:
            d[f"theta_{state}"] = d[f"theta_{state}"] + d_weight
        d_dV = -alpha * self.f * forward - (n - alpha) * self.f * backward
        d_ln_a_rate = {}
        for s, nu in ln_fw_a.items():
            d_ln_a_rate[s] = d_ln_a_rate.get(s, 0.0) + nu * forward
        for s, nu in ln_bw_a.items():
            d_ln_a_rate[s] = d_ln_a_rate.get(s, 0.0) - nu * backward
        self._add_potential_and_activity_derivatives(d, d_dV, d_ln_a_rate, d_ln_a)
        return rate, d

    # ------------------------------------------------------------------
    # Internal Helpers: fields
    # ------------------------------------------------------------------

    def _interface_state(self, step: int):
        """PRIVATE: Return lumped weights, step DOFs and nodal values, including the metal potential."""
        names = self._fields + [f"{name}Old" for name in self.coverage_fields]
        weights, dofs, vals = self.mesh.Get_Interface_Weights_Dofs(self.domain, names, step)
        if self.charge_conservation:
            vals = {**vals, self.E_m_field: np.full(weights.size, self.mesh.Get_Global_Value(self.E_m_field))}
        return weights, dofs, vals

    def _metal_potential(self) -> float:
        """PRIVATE: Return the current metal potential."""
        return self.mesh.Get_Global_Value(self.E_m_field) if self.charge_conservation else self.E_m

    def _metal_potential_values(self, vals) -> np.ndarray:
        """PRIVATE: Return the metal potential at the nodes."""
        return vals[self.E_m_field] if self.charge_conservation else np.full_like(vals["E"], self.E_m)

    def _measure_names(self) -> list:
        """PRIVATE: Return the global measure names this model reports."""
        names = [f"{self.name}_{key}" for key in ("Anodic_Current", "Fe2_Rate", "Fe3_Rate", "Dissolution_Current",
                                                  "Hydroxide_Route_Rate", "Chloride_Route_Rate")]
        if self.cathodic:
            names += [f"{self.name}_ORR_Current", f"{self.name}_HER_Current"]
        names += [f"{self.name}_Coverage_{state}" for state in STATES]
        names.append(f"{self.name}_Metal_Potential")
        return names
