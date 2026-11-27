"""Linear elasticity with selective phase-field degradation."""

import ufl
from ufl import conditional, lt

from femcubes.core.utils.maths_utils import Tensor_To_Mandel_UFL, nabla_s
from femcubes.solids.linear_elastic import LinearElastic
from femcubes.solids._split_elastic_common import _fracture_driver, _validate_split_damage_material, _validate_split_settings
from femcubes.solids._domain_points import _DomainPoints
from femcubes.solids.cpp_kernels.viscoelastic._viscoelastic_kernels_cpp import SplitLinearElasticKernels


class SplitLinearElastic(LinearElastic):
    r"""Linear-elastic solid with selective phase-field degradation of a split stress.

    The model has the same displacement fields, Newmark integration and
    lifecycle as :doc:`linear_elastic`, but degrades only the damageable part
    :math:`\boldsymbol\sigma^+` of the stress, while damaged material
    transfers the retained part :math:`\boldsymbol\sigma^-` as intact
    material does. The split is chosen by ``CrackDrivingForce`` (see
    :ref:`solids-crack-driving-forces`), and the fracture driver
    ``ElasticEnergy`` for :doc:`phase_field` is evaluated from the same split
    at the same strain. The energy splits follow established decompositions of
    the elastic energy: the volumetric-deviatoric split of :cite:t:`Amor2009`
    (``VolDev``), the spectral split of :cite:t:`Miehe2010a` (``Spectral``)
    and the no-tension split of :cite:t:`DelPiero1989` and
    :cite:t:`Freddi2010` (``NoTension``). The default ``Stress-Based`` option
    splits the principal stresses instead, with the driving force following
    the failure criterion of :cite:t:`Miehe2015a`.
    Since the degraded and retained parts depend on the strain, the momentum
    balance is nonlinear in the displacements, also at fixed phase field.

    Density is never degraded: damaged material keeps the compressive part of
    the stress and carries its weight (a consistent split, see
    :ref:`solids-split-consistency`), which makes this model suited to
    problems loaded by gravity or in compression. Supports 3-D, 2-D plane
    strain and intrinsic 2-D (``true_2d``).

    .. rubric:: Fields

    * ``u``, ``v``, ``a``, ``uOld``, ``vOld``, ``aOld``, ``p``: as in
      :doc:`linear_elastic`.
    * ``sigma``: mechanical stress :math:`g(\phi)\boldsymbol\sigma^+ + \boldsymbol\sigma^-`,
      quadrature field with six Mandel components.
    * ``sigma_plus``, ``sigma_minus``: undegraded :math:`\boldsymbol\sigma^+`
      and :math:`\boldsymbol\sigma^-` (quadrature, six components).
    * ``dsigma_dStrainMandel``: stress tangent, quadrature :math:`6\times6`.
    * ``Total_Strain`` (six components) and ``dam_elastic`` (scalar
      :math:`g(\phi)`): quadrature copies used by the point kernel.
    * ``ElasticEnergy``: crack driving force, scalar quadrature field (always
      defined).
    * ``phasefield``: read only, provided by :doc:`phase_field`; only with
      ``DamageModel="PhaseField"``.

    .. rubric:: Weak form

    With test function :math:`\delta\mathbf{u}` and the strains written as
    six-component Mandel vectors (zero out-of-plane components on 2-D
    meshes), the contribution to the residual of ``u`` is

    .. math::

        R_{\mathbf{u}} =
        \int_\Omega \boldsymbol{\varepsilon}(\delta\mathbf{u}):\boldsymbol{\sigma}\,\mathrm{d}x
        + \int_\Omega \rho\,\delta\mathbf{u}\cdot\ddot{\mathbf{u}}\,\mathrm{d}x
        + \int_\Omega c\,\delta\mathbf{u}\cdot\dot{\mathbf{u}}\,\mathrm{d}x
        - \int_\Omega \rho\,g\,\delta u_{d}\,\mathrm{d}x,
        \qquad
        \boldsymbol{\sigma} = g(\phi)\,\boldsymbol\sigma^+(\boldsymbol\varepsilon(\mathbf u))
        + \boldsymbol\sigma^-(\boldsymbol\varepsilon(\mathbf u)),

    with :math:`g(\phi) = \kappa + (1-\kappa)(1-\hat\phi)^2`,
    :math:`\hat\phi = \min(\max(\phi,0),1)`, and :math:`g = 1` without
    ``DamageModel``. The inertia, damping and gravity terms, their options
    and the Newmark relations are those of :doc:`linear_elastic`, but with
    the full density :math:`\rho` at every damage level. The stress and the
    tangent :math:`g\,\partial\boldsymbol\sigma^+/\partial\boldsymbol\varepsilon + \partial\boldsymbol\sigma^-/\partial\boldsymbol\varepsilon`
    are quadrature fields, so the Jacobian is consistent at fixed
    :math:`\phi`; for ``Stress-Based`` it can be nonsymmetric. With
    ``true_2d`` the split is intrinsic 2-D, otherwise plane strain (:math:`\varepsilon_{zz}=0`,
    out-of-plane stress retained).

    .. rubric:: Outside the weak form

    * Before every assembly, :math:`\dot{\mathbf u}` and :math:`\ddot{\mathbf u}`
      follow the Newmark update, and the point kernel evaluates ``sigma``,
      ``sigma_plus``, ``sigma_minus``, ``dsigma_dStrainMandel`` and
      ``ElasticEnergy`` at the current :math:`\boldsymbol\varepsilon(\mathbf u)`
      (``Stress-Based`` clipped at zero for AT2; total elastic energy without
      ``DamageModel``).
    * ``commit`` copies ``u``, ``v``, ``a`` to the ``*Old`` fields and
      evaluates ``p`` :math:`= -\lambda\,\mathrm{tr}\,\boldsymbol\varepsilon`
      and the stress fields; ``reset_step`` restores ``u``, ``v``, ``a``.
    * No Dirichlet conditions.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            Field_Orders = {"u": 2}          # REQUIRED: polynomial order of u
            Solution_Steps = {"u": 0}        # REQUIRED: staggered step solving u
            gravity = -9.81                  # REQUIRED: acceleration along the last axis (m/s^2)
            Newmark_Beta = 0.25              # REQUIRED: Newmark beta (-)
            Newmark_Gamma = 0.5              # REQUIRED: Newmark gamma (-)
            Materials = {
                "Ice": {
                    "Lame": 1.0e9,           # REQUIRED: first Lame parameter lambda (Pa), finite, >= 0
                    "Shear": 3.5e8,          # REQUIRED: shear modulus mu (Pa), finite, > 0
                    "Density": 917.0,        # REQUIRED: density rho (kg/m^3)
                    "LE_Damping": 0.0,       # OPTIONAL (default 0.0): damping coefficient c (Pa s/m^2)
                    "Inertia": True,         # OPTIONAL (default True): include the inertia term
                    "CrackDrivingForce": "NoTension",  # OPTIONAL (default "Stress-Based"): stress and driver split
                    # With DamageModel="PhaseField" only:
                    "Gc": 1.0e3,             # REQUIRED: fracture toughness (J/m^2)
                    "ell": 10.0,             # REQUIRED: phase-field length scale (m)
                    "f_t": 0.4e6,            # REQUIRED: tensile strength (Pa)
                    "kappa": 1.0e-6,         # OPTIONAL (default 1e-3): residual stiffness, 0 < kappa <= 1
                    "PF_formulation": "AT2", # OPTIONAL (default "AT2"): AT1, AT2, HO-AT1 or HO-AT2 (Stress-Based scaling)
                },
            }
            Models = {
                "Solid": {
                    "Type": "Solid:SplitLinearElastic",  # REQUIRED
                    "Material": "Ice",                   # REQUIRED: key in Materials
                    "DamageModel": "PhaseField",         # OPTIONAL (default None): None or "PhaseField"
                    "true_2d": False,                    # OPTIONAL (default False): intrinsic 2-D law, 2-D meshes only
                    "domain": "domain",                  # OPTIONAL (default "domain", the full mesh): domain group
                },
            }

    ``CreepCoupling`` is rejected. ``p`` remains the diagnostic
    :math:`-\lambda\,\mathrm{tr}\,\boldsymbol\varepsilon`, not the mean split
    stress.
    """

    MODEL_TYPE = "Solid:SplitLinearElastic"

    def __init__(self, name, params, mesh):
        settings = params.Models[name]
        split = _validate_split_settings(settings, params.Materials[settings["Material"]], type(self).__name__)
        if "CreepCoupling" in settings:
            raise ValueError("SplitLinearElastic has no creep; omit CreepCoupling.")
        super().__init__(name, params, mesh)
        self.DrivingForce_Type = split
        if self.damage_model_name == "PhaseField":
            self.pf_formulation = _validate_split_damage_material(params.Materials[self.material], split)
        else:
            mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", (), domain=self.domain)
        mesh.Add_Quadrature_Field("sigma_plus", "Undegraded damageable stress [Pa]", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("sigma_minus", "Retained stress [Pa]", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("dsigma_dStrainMandel", "Stress tangent [Pa]", (6, 6), domain=self.domain)
        mesh.Add_Quadrature_Field("Total_Strain", "Strain Tensor", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("dam_elastic", "Elastic damage variable", (), domain=self.domain)
        self._split_kernel = SplitLinearElasticKernels(self.lame, self.shear, split, self.true_2d)
        self._total_strain_expr = None
        self._damage_expr = None
        self._points = _DomainPoints(mesh, self.domain, (
            "Total_Strain", "dam_elastic", "sigma", "sigma_plus", "sigma_minus", "dsigma_dStrainMandel", "ElasticEnergy",
        ))

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Assemble split mechanical stress and tangent with full-density forces.

        The constitutive update stores sigma = g*sigma_plus + sigma_minus and
        its algorithmic tangent in quadrature fields. Use those fields directly;
        inertia and gravity retain the full density at every damage level.
        """
        if step != self.My_Step:
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)

        v = self.mesh.Get_Field("v")
        a = self.mesh.Get_Field("a")
        u_t = self.mesh.Get_TestFunction("u", self.My_Step)
        u_tr = self.mesh.Get_Trial_Function("u", self.My_Step)
        dv_du = self.NM_Gamma / (self.NM_Beta * dt_constant)
        da_du = 1.0 / (self.NM_Beta * dt_constant**2)

        # The stored stress and tangent already contain the consistent split.
        strain_test = Tensor_To_Mandel_UFL(nabla_s(u_t), self.dim)
        strain_trial = Tensor_To_Mandel_UFL(nabla_s(u_tr), self.dim)
        stress = self.mesh.Get_Field("sigma")
        tangent = self.mesh.Get_Field("dsigma_dStrainMandel")
        stress_vector = ufl.as_vector([stress[i, 0] for i in range(6)])
        F = ufl.inner(strain_test, stress_vector) * dx
        K = ufl.dot(strain_test, ufl.dot(tangent, strain_trial)) * dx

        if self.Inertia:
            F += self.rho * ufl.inner(u_t, a) * dx
            K += self.rho * ufl.inner(u_t, da_du * u_tr) * dx

        if self.damping > 0.0:
            F += self.damping * ufl.inner(u_t, v) * dx
            K += self.damping * ufl.inner(u_t, dv_du * u_tr) * dx

        if self.gravity != 0.0:
            F -= self.rho * u_t[self.dim-1] * self.gravity * dx

        return K, F

    def Post_Remesh(self):
        """Clear expressions compiled on the previous mesh."""
        self._total_strain_expr = None
        self._damage_expr = None
        self._points.reset()

    def Update_PhaseField_Fields(self):
        """Bind stiffness degradation; the stress update supplies the driver."""
        self.d_rho = 1.0
        if self.damage_model_name == "PhaseField":
            phasefield = self.mesh.Get_Field("phasefield")
            phi = conditional(lt(phasefield, 1.0), phasefield, 1.0)
            phi = conditional(lt(phi, 0.0), 0.0, phi)
            self.d_e = self.kappa + (1.0-self.kappa)*(1.0-phi)**2

    # Internal helpers

    def _update_stress_field(self):
        """PRIVATE: Evaluate split stress, tangent and driver at the current strain."""
        self.Update_PhaseField_Fields()
        displacement = self.mesh.Get_Field("u", step=self.My_Step)
        expression = Tensor_To_Mandel_UFL(nabla_s(displacement), self.dim)
        self._total_strain_expr = self._points.interpolate("Total_Strain", expression, self._total_strain_expr)
        self.mesh.Get_Field("Total_Strain").x.scatter_forward()
        self._damage_expr = self._points.interpolate("dam_elastic", self.d_e, self._damage_expr)
        self.mesh.Get_Field("dam_elastic").x.scatter_forward()
        strain, damage = self._points.gather("Total_Strain", "dam_elastic")
        result = self._split_kernel(strain.reshape(-1, 6), damage)
        driver = _fracture_driver(
            result["driver"], self.DrivingForce_Type,
            {"Gc": self.Gc, "ell": self.ell, "f_t": self.f_t, "PF_formulation": self.pf_formulation}
            if self.damage_model_name == "PhaseField" else None,
            clip_at2=True,
            elastic_strain=strain.reshape(-1, 6), stress=result["stress"],
        )
        self._points.scatter({
            "sigma": result["stress"], "sigma_plus": result["stress_plus"],
            "sigma_minus": result["stress_minus"], "dsigma_dStrainMandel": result["tangent"],
            "ElasticEnergy": driver,
        })
        for name in ("sigma", "sigma_plus", "sigma_minus", "dsigma_dStrainMandel", "ElasticEnergy"):
            self.mesh.Get_Field(name).x.scatter_forward()
