r"""Activity coefficients of dissolved species (ideal, Davies, SIT), shared by the electrochemistry models.

In concentrated solutions, the interactions between ions make their
effective concentrations, the activities, differ from their concentrations,
which is described by activity coefficients :math:`\gamma_\pi`. The ideal
model neglects these interactions. The equation of :cite:t:`Davies1938` extends the Debye-Hueckel description to
moderate ionic strengths, and the specific ion interaction theory (SIT) adds
interaction coefficients for pairs of oppositely charged ions, which extends
the description to higher ionic strengths.

The activity coefficients are used by :doc:`electrolyte` (flux and volume reactions) and
:doc:`passivating_surface` (surface reactions), selected per model with the
required setting ``"Activity_Model"``. Concentrations are in mol/m^3 (the
field unit); the activity models use molalities, taken equal to molarities,
:math:`m_\pi = C_\pi / 1000` (the dilute-water approximation, within a few
percent up to about 1 M). The ionic strength is
:math:`I = \max(\tfrac{1}{2} \sum_\pi z_\pi^2 m_\pi, 10^{-12})` over the
dissolved species, and

* ``"ideal"``: :math:`\log_{10} \gamma_\pi = 0`;
* ``"Davies"``: :math:`\log_{10} \gamma_\pi = -A z_\pi^2 \left(\frac{\sqrt{I}}{1 + \sqrt{I}} - b I\right)`
  (for I up to about 0.5 M);
* ``"SIT"``: :math:`\log_{10} \gamma_\pi = -z_\pi^2 \frac{A \sqrt{I}}{1 + B_a \sqrt{I}} + \sum_k \varepsilon(\pi, k)\, m_k`,

with the SIT sum over the oppositely charged species :math:`k` with a listed
interaction coefficient (unlisted pairs are zero). The activity of a
dissolved species is :math:`a_\pi = \gamma_\pi C_\pi / C_{ref}`. The
constants come from ``params.Activity_Parameters``:

.. code-block:: python

    class Params(ParamsTemplate):
        Activity_Parameters = {   # REQUIRED unless every Activity_Model is "ideal"
            "A": 0.509,           # REQUIRED (Davies, SIT): Debye-Hueckel constant at 25 C (kg^0.5 mol^-0.5)
            "B_a": 1.5,           # REQUIRED (SIT): B a_j (kg^0.5 mol^-0.5), NEA convention
            "b_Davies": 0.3,      # REQUIRED (Davies): Davies constant (kg/mol)
            "SIT_epsilon": {"Na+|Cl-": 0.03, "H+|Cl-": 0.12},  # OPTIONAL (SIT, default {}): "cation|anion": eps (kg/mol);
                                                             # pairs with absent species are ignored
        }
"""

import math

import numpy as np
from ufl import conditional, ge, sqrt


ACTIVITY_MODELS = ("ideal", "Davies", "SIT")

_MOLAR = 1.0e-3                # (mol/L) per (mol/m^3)
_LN10 = math.log(10.0)
_I_MIN = 1.0e-12               # floor of the ionic strength (mol/L), keeps d sqrt(I)/dI finite


class ActivityModel:
    """PRIVATE: Activity coefficients of the dissolved species, as NumPy arrays (nodal) or UFL expressions.

    Parameters
    ----------
    model : str
        ``"ideal"``, ``"Davies"`` or ``"SIT"``; any other value is rejected.
    parameters : dict or None
        ``params.Activity_Parameters`` (see the module docstring); not
        needed for ``"ideal"``.
    charges : dict
        Charge number of every dissolved species (immobile solids excluded).
    """

    def __init__(self, model, parameters, charges: dict):
        if model not in ACTIVITY_MODELS:
            raise ValueError(f"Activity_Model must be one of {ACTIVITY_MODELS}, not {model!r}.")
        self.model = model
        self.charges = dict(charges)
        self.species = list(charges)
        self.ideal = model == "ideal"
        if self.ideal:
            return
        if parameters is None:
            raise ValueError(f"Activity_Model '{model}' requires params.Activity_Parameters.")
        self.A = float(parameters["A"])
        if model == "Davies":
            self.b = float(parameters["b_Davies"])
            self.epsilon = {}
        else:
            self.B_a = float(parameters["B_a"])
            self.epsilon = {}
            for pair, value in parameters.get("SIT_epsilon", {}).items():
                cation, anion = pair.split("|")
                if cation not in self.charges or anion not in self.charges:
                    continue        # pair of species not present in this electrolyte
                if not (self.charges[cation] > 0 and self.charges[anion] < 0):
                    raise ValueError(f"SIT_epsilon pair '{pair}' must be written as 'cation|anion'.")
                self.epsilon[(cation, anion)] = float(value)
                self.epsilon[(anion, cation)] = float(value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Ln_Gamma(self, C: dict):
        """Return ``(ln_gamma, d_ln_gamma)`` at the nodes, from nodal concentrations (mol/m^3).

        ``ln_gamma[s]`` is an array over the nodes; ``d_ln_gamma[s][k]`` is
        :math:`\\partial \\ln\\gamma_s / \\partial C_k` (m^3/mol), listed only
        for the species ``k`` it depends on (empty for the ideal model).
        """
        n = len(next(iter(C.values())))
        if self.ideal:
            return {s: np.zeros(n) for s in self.species}, {s: {} for s in self.species}

        I = 0.5 * _MOLAR * sum(self.charges[s] ** 2 * np.maximum(C[s], 0.0) for s in self.species)
        I = np.maximum(I, _I_MIN)
        root = np.sqrt(I)
        if self.model == "Davies":
            debye = self.A * (root / (1.0 + root) - self.b * I)
            d_debye_dI = self.A * (0.5 / (root * (1.0 + root) ** 2) - self.b)
        else:
            debye = self.A * root / (1.0 + self.B_a * root)
            d_debye_dI = 0.5 * self.A / (root * (1.0 + self.B_a * root) ** 2)

        ln_gamma, d_ln_gamma = {}, {}
        for s in self.species:
            z2 = self.charges[s] ** 2
            log_gamma = -z2 * debye
            derivatives = {}
            for k in self.species:
                # dI/dC_k = z_k^2 / 2 * 1e-3
                d = -z2 * d_debye_dI * 0.5 * _MOLAR * self.charges[k] ** 2
                if (s, k) in self.epsilon:
                    log_gamma = log_gamma + self.epsilon[(s, k)] * _MOLAR * C[k]
                    d = d + self.epsilon[(s, k)] * _MOLAR
                if np.any(d != 0.0):
                    derivatives[k] = _LN10 * d
            ln_gamma[s] = _LN10 * log_gamma
            d_ln_gamma[s] = derivatives
        return ln_gamma, d_ln_gamma

    def Ln_Gamma_UFL(self, C: dict) -> dict:
        """Return :math:`\\ln\\gamma_s` as UFL expressions of the concentration fields (``None`` if ideal).

        Species whose coefficient is identically one (neutral, without SIT
        coefficients) are omitted.
        """
        if self.ideal:
            return None
        I = 0.5 * _MOLAR * sum(self.charges[s] ** 2 * C[s] for s in self.species)
        I = conditional(ge(I, _I_MIN), I, _I_MIN)
        root = sqrt(I)
        if self.model == "Davies":
            debye = self.A * (root / (1.0 + root) - self.b * I)
        else:
            debye = self.A * root / (1.0 + self.B_a * root)
        ln_gamma = {}
        for s in self.species:
            if self.charges[s] == 0.0 and not any((s, k) in self.epsilon for k in self.species):
                continue
            log_gamma = -self.charges[s] ** 2 * debye
            for k in self.species:
                if (s, k) in self.epsilon:
                    log_gamma = log_gamma + self.epsilon[(s, k)] * _MOLAR * C[k]
            ln_gamma[s] = _LN10 * log_gamma
        return ln_gamma

