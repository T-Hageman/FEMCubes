"""Cohesive visco-elastic solid with the local problem solved by the C++ kernel."""

from femcubes.core.mesh.mesh import Mesh
from femcubes.solids._cohesive_viscoelastic_base import _CohesiveViscoElasticBase


class Cohesive_ViscoElastic_Kernel(_CohesiveViscoElasticBase):
    r"""Cohesive visco-elastic solid, local problem solved by the C++ kernel.

    The model is the same as :doc:`cohesive_viscoelastic` (fields, material
    parameters, failure types, weak form, phase-field coupling, history and
    global measures); only the integration-point update differs. The
    combined creep and cohesive local problem is solved by one Newton
    iteration over the stress and all multipliers in C++, as in the paper
    that introduced the model, :cite:t:`Hageman2026CMAME` (see
    :doc:`../../publications`), which extends the return mapping of
    :cite:t:`Hageman2026b` with creep. The results agree with
    :doc:`cohesive_viscoelastic` to the kernel's tolerance; that model is
    typically several times faster in fracture simulations.

    .. rubric:: Fields

    As :doc:`cohesive_viscoelastic`, except that ``LambdaField_coh`` and
    ``LambdaFieldOld_coh`` have three components for ``r2_2L_PSD`` (the
    multipliers of the three principal directions), and that there is no
    ``StrengthRatio`` option.

    .. rubric:: Weak form

    Identical to :doc:`cohesive_viscoelastic`, including the choice of
    Jacobian by ``consistent_stiffness``.

    .. rubric:: Outside the weak form

    * **Local problem.** The equations of :doc:`cohesive_viscoelastic`,
      solved for :math:`(\boldsymbol\sigma, \Lambda_v, \Lambda_i)` (and
      :math:`\varepsilon_{zz}` with ``plane_stress``) in one Newton
      iteration; the complementarity conditions are imposed with a
      Fischer-Burmeister function :cite:p:`Fischer1992`, with the active set of the cohesive
      modes updated in every iteration (onset from the drive of each mode
      at a probe multiplier, the dual norm of the volumetric and deviatoric
      stress projections for the split r2 criteria in tension, and the
      opening gauge on the principal directions for ``r2_2L_PSD``).
      :math:`\mathbb{C}_\mathrm{alg}` follows by implicit differentiation
      of the converged local system.
    * **History.** Only :math:`\boldsymbol\varepsilon^v_n` is history; the
      cohesive strain is recomputed from the current state at every call,
      and the multipliers of the previous call serve as the initial guess.
      The smoothed potentials can leave tiny "regularized" multipliers
      (about :math:`10^{-7}f_t/E`) below the strength surface, so
      ``etaNorm`` is small but non-zero there. If the local problem fails,
      also after the built-in continuation, ``sigma`` and the tangent are
      NaN so that the global solver rejects the step.

    Example model settings
    ----------------------

    The global and material entries are those of
    :doc:`cohesive_viscoelastic`; the model block is:

    .. code-block:: python

        Models = {
            "Solid": {
                "Type": "Solid:Cohesive_ViscoElastic_Kernel",  # REQUIRED
                "Material": "Ice",              # REQUIRED: key in Materials
                "plane_stress": False,          # OPTIONAL (default False): 2-D plane stress
                "true_2d": False,               # OPTIONAL (default False): intrinsic 2-D law; not with plane_stress
                "consistent_stiffness": True,   # OPTIONAL (default True): algorithmic tangent, else elastic stiffness
                "Eta_Active_Threshold": 1.0e-8, # OPTIONAL (default 1e-8): threshold of Eta_Active_Area (-)
                "domain": "domain",             # OPTIONAL (default "domain", the full mesh): domain group
            },
            "pf": {
                "Type": "Solid:PhaseField",     # provides phasefield
                "Material": "Ice",
                "CDF_field": "PsiField",
            },
        }
    """

    MODEL_TYPE = "Solid:Cohesive_ViscoElastic_Kernel"

    def _build_update(self, mesh: Mesh):
        """PRIVATE: the C++ kernel."""
        return self._build_kernel()
