"""Cohesive linear-elastic solid: the cohesive visco-elastic solid without creep."""

from femcubes.solids.cohesive_viscoelastic import Cohesive_ViscoElastic


class Cohesive_LinearElastic(Cohesive_ViscoElastic):
    r"""Cohesive linear-elastic solid with Newmark time integration.

    The model is the strength-based cohesive phase-field model of
    :cite:t:`Hageman2026b`, in which fracture eigenstrains limited by an
    explicit strength surface :cite:p:`Vicentini2026` represent the opening
    of cracks, and the phase field degrades the strength potential. Crack
    nucleation is governed by the strength surface and crack growth by the
    fracture energy, both independent of the phase-field length scale
    :cite:p:`Hageman2026b`. It is :doc:`cohesive_viscoelastic` with creep
    switched off: the material's ``Visc_Type`` is taken as ``"None"``
    whatever it is set to, and the creep entries (``Viscosity``,
    ``Glen_A``, ``Glen_n``) are not read. The strain splits into an elastic
    part and the cohesive eigenstrain,
    :math:`\boldsymbol\varepsilon = \boldsymbol\varepsilon^e + \boldsymbol\eta`,
    and everything else (fields, failure types, options, phase-field
    coupling, global measures) is that of :doc:`cohesive_viscoelastic`.

    .. rubric:: Weak form

    With test function :math:`\delta\mathbf{u}` and :math:`\Omega` the
    domain group named by ``domain``, the residual and Jacobian of ``u``
    are those of :doc:`cohesive_viscoelastic`:

    .. math::

        R_{\mathbf{u}} =
        \int_\Omega \boldsymbol\varepsilon(\delta\mathbf{u}):\boldsymbol\sigma\,\mathrm{d}x
        + \int_\Omega \rho\,\delta\mathbf{u}\cdot\ddot{\mathbf{u}}\,\mathrm{d}x
        + \int_\Omega \frac{c}{\Delta t}\,\delta\mathbf{u}\cdot(\mathbf{u}-\mathbf{u}_n)\,\mathrm{d}x
        - \int_\Omega \rho\,g\,\delta u_d\,\mathrm{d}x,

    with :math:`\boldsymbol\sigma = \mathbb{D}:(\boldsymbol\varepsilon - \boldsymbol\eta)`
    from the local cohesive problem, inertia for ``Inertia=True`` (default
    ``False``), damping for ``LE_Damping`` :math:`c>0` (a coefficient in
    Pa s/m\ :sup:`2`, not multiplied by the density, also without inertia)
    and gravity for ``gravity`` :math:`g\neq0`. The density is the full
    density at every damage level.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            Field_Orders = {"u": 2}          # REQUIRED: polynomial order of u
            Solution_Steps = {"u": 0}        # REQUIRED: staggered step solving u
            gravity = 0.0                    # REQUIRED: acceleration along the last axis (m/s^2)
            Newmark_Beta = 0.25              # REQUIRED: Newmark beta (-)
            Newmark_Gamma = 0.5              # REQUIRED: Newmark gamma (-)
            Materials = {
                "Iron": {
                    "Lame": 1.15e11,         # REQUIRED: first Lame parameter (Pa)
                    "Shear": 7.7e10,         # REQUIRED: shear modulus (Pa)
                    "Bulk": 1.67e11,         # REQUIRED: bulk modulus (Pa), consistent with Lame and Shear
                    "Density": 8.0e3,        # REQUIRED: density (kg/m^3)
                    "f_t": 150e6,            # REQUIRED: tensile strength (Pa)
                    "f_s": 300e6,            # REQUIRED: shear / compressive strength (Pa)
                    "Gc": 1.0e4,             # REQUIRED: fracture toughness (J/m^2); here only GcII/Gc is used
                    "ell": 0.01,             # REQUIRED: length scale (m); read, used by Solid:PhaseField only
                    "FailureType": "r1",     # OPTIONAL (default "r1"): cohesive potential, see Cohesive_ViscoElastic
                    "GcII": 1.0e4,           # OPTIONAL (default Gc): mode-II toughness, r1_DP only (J/m^2)
                    "kappa": 1.0e-6,         # OPTIONAL (default 1e-3): residual degradation (-)
                    "DP_phi": 0.0,           # OPTIONAL (default 0.0): friction angle of the DP types (deg)
                    "DP_q": 300e6,           # OPTIONAL (default f_s): confined strength cap of the DP types (Pa)
                    "Inertia": True,         # OPTIONAL (default False): include the inertia term
                    "LE_Damping": 0.0,       # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                },
            }
            Models = {
                "Solid": {
                    "Type": "Solid:Cohesive_LinearElastic",  # REQUIRED
                    "Material": "Iron",                 # REQUIRED: key in Materials
                    "plane_stress": False,              # OPTIONAL (default False): 2-D plane stress
                    "true_2d": False,                   # OPTIONAL (default False): intrinsic 2-D law; not with plane_stress
                    "consistent_stiffness": True,       # OPTIONAL (default True): algorithmic tangent, else elastic stiffness
                    "Eta_Active_Threshold": 1.0e-8,     # OPTIONAL (default 1e-8): threshold of Eta_Active_Area (-)
                    "domain": "domain",                 # OPTIONAL (default "domain", the full mesh): domain group
                },
                "pf": {
                    "Type": "Solid:PhaseField",         # provides phasefield
                    "Material": "Iron",
                    "CDF_field": "PsiField",
                },
            }
    """

    MODEL_TYPE = "Solid:Cohesive_LinearElastic"

    @staticmethod
    def _creep_law(material: dict, constitutive_dim: float):
        """PRIVATE: no creep, whatever the material's ``Visc_Type``."""
        return "None", 0.0, 1.0
