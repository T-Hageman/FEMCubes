"""PRIVATE: shared base of the cohesive visco-elastic solid models.

``Solid:Cohesive_ViscoElastic`` (reduced local update) and
``Solid:Cohesive_ViscoElastic_Kernel`` (C++ kernel) share the fields, the material
parameters, the weak form, the phase-field coupling, the history and the global measures
defined here; they differ only in the integration-point update returned by
``_build_update``. The base class has no ``MODEL_TYPE`` and is not registered.
"""

from ufl import inner, dot
import numpy as np
from petsc4py import PETSc

from femcubes.core.base_model import BaseModel
from femcubes.core.time_discretizations import Newmark_Update as Apply_Newmark_Update
from femcubes.core.utils.maths_utils import nabla_s, Tensor_To_Mandel_UFL
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.params_template import ParamsTemplate
from ufl import conditional, lt, gt
import ufl
from dolfinx import fem
from mpi4py import MPI
from femcubes.solids.cpp_kernels.cohesive_viscoelastic._cohesive_viscoelastic_kernels_cpp import (
    CohesiveViscoElasticKernels as _CVEKernels,
    cohesive_multiplier_slots as _cohesive_multiplier_slots,
)
from femcubes.solids._domain_points import _DomainPoints

#: Supported values of the material entry ``FailureType``.
FAILURE_TYPES = ("r1", "r1_DP", "r2", "r2_DP", "r2_2L", "r2_2L_DP", "r2_2L_PSD")


def check_failure_type(failure_type: str) -> str:
    """Return ``failure_type`` if it is one of :data:`FAILURE_TYPES`, else raise a ``ValueError``."""
    if failure_type not in FAILURE_TYPES:
        raise ValueError(f"Unknown FailureType {failure_type!r}; supported: {', '.join(FAILURE_TYPES)}.")
    return failure_type


def elastic_stiffness(lame, shear, *, true_2d=False, plane_stress=False):
    """Isotropic elastic stiffness (6 x 6 Mandel matrix, UFL) of plane strain and 3-D, the
    condensed plane-stress stiffness (zero zz row and column), or the intrinsic 2-D stiffness
    (in-plane components only). ``lame`` and ``shear`` are numbers or UFL expressions."""
    D = [[0.0] * 6 for _ in range(6)]
    if true_2d:
        D[0][0] = D[1][1] = lame + 2.0 * shear
        D[0][1] = D[1][0] = lame
        D[5][5] = 2.0 * shear
    elif plane_stress:
        D[0][0] = D[1][1] = lame + 2.0 * shear - lame**2/(lame + 2.0 * shear)
        D[0][1] = D[1][0] = lame - lame**2/(lame + 2.0 * shear)
        D[3][3] = D[4][4] = D[5][5] = 2.0*shear
    else:
        D[0][0] = D[1][1] = D[2][2] = lame + 2.0 * shear
        D[0][1] = D[0][2] = D[1][0] = D[1][2] = D[2][0] = D[2][1] = lame
        D[3][3] = D[4][4] = D[5][5] = 2.0*shear
    return ufl.as_matrix(D)


class _CohesiveViscoElasticBase(BaseModel):
    """PRIVATE: fields, material parameters, weak form and history of the cohesive
    visco-elastic solids; subclasses provide the integration-point update
    (``_build_update``)."""

    # Quadrature fields updated pointwise by the constitutive update.
    _POINT_FIELDS = (
        "Total_Strain", "dam_elastic", "sigma", "eta", "etaOld", "Plastic_Strain", "Plastic_Strain_Old",
        "dsigma_dStrainMandel", "ElasticEnergy", "PsiField", "LambdaField_visc", "LambdaField_coh",
        "LambdaFieldOld_coh", "etaNorm",
    )

    def __init__(self, name, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)
        self.dim = params.dim

        self.LumpedMass = False
        self.consistent_stiffness = self.params.Models[self.name].get(
            "consistent_stiffness", True
        )

        self._create_fields(params, mesh)

        # --- Material parameters ---
        self.material = self.params.Models[self.name]["Material"]
        material = self.params.Materials[self.material]
        self.plane_stress = self.params.Models[self.name].get("plane_stress", False)
        self.true_2d = self.params.Models[self.name].get("true_2d", False)
        if self.plane_stress and self.dim != 2:
            raise ValueError(f"plane_stress is only supported for 2-D {type(self).__name__} models.")
        if self.true_2d and self.dim != 2:
            raise ValueError(f"true_2d is only supported for 2-D {type(self).__name__} models.")
        if self.plane_stress and self.true_2d:
            raise ValueError("plane_stress and true_2d cannot both be enabled.")
        self.gravity = self.params.gravity
        self.damping = material.get("LE_Damping", 0.0)
        self.Inertia = material.get("Inertia", False)

        # Elastic, density, creep and tensile-strength parameters
        self._read_material(material)
        self.D = elastic_stiffness(self.lame, self.shear, true_2d=self.true_2d, plane_stress=self.plane_stress)

        self.NM_Beta = self.params.Newmark_Beta
        self.NM_Gamma = self.params.Newmark_Gamma

        # Cohesive / phase-field parameters
        self.ell = material["ell"]
        self.f_s = material["f_s"]
        self.kappa = material.get("kappa", 1e-3)
        self.failureType = material.get("FailureType", "r1")
        self.DP_phi = material.get("DP_phi", 0.0)
        self.DP_q = material.get("DP_q", self.f_s)

        self.d_e = None

        # Expression caches
        self._strain_expr = None
        self._damage_expr = None

        # Global measures of the cohesive fracture strain (time series)
        self.eta_active_threshold = float(
            self.params.Models[self.name].get("Eta_Active_Threshold", 1.0e-8)
        )
        if self.eta_active_threshold < 0.0:
            raise ValueError("Eta_Active_Threshold must be non-negative.")
        self._eta_area_form = None
        self.params.Global_Measures["Eta_Max"] = 0.0
        self.params.Global_Measures["Eta_Active_Area"] = 0.0

        # Integration-point update, called with the arrays of _update_ip_fields
        self._update_cve_state = self._build_update(mesh)

    # -----------------------------------------------------------------
    #  Public API
    # -----------------------------------------------------------------
    def Post_Remesh(self) -> None:
        """Clear quadrature expressions compiled on the previous mesh."""
        self._strain_expr = None
        self._damage_expr = None
        self._eta_area_form = None
        self._points.reset()

        self.Update_PhaseField_Fields()

    def Update_Global_Measures(self) -> None:
        """Record the maximum and the active measure of the cohesive fracture strain."""
        eta_norm = self.mesh.Get_Field("etaNorm")
        index_map = eta_norm.function_space.dofmap.index_map
        owned = eta_norm.x.array[: index_map.size_local * eta_norm.function_space.dofmap.index_map_bs]
        local_max = float(owned.max()) if owned.size else 0.0
        comm = eta_norm.function_space.mesh.comm
        self.params.Global_Measures["Eta_Max"] = comm.allreduce(local_max, op=MPI.MAX)

        if self._eta_area_form is None:
            dx = self.mesh.Get_Domain_Integrator(self.domain)
            active = conditional(gt(eta_norm, self.eta_active_threshold), 1.0, 0.0)
            self._eta_area_form = self.mesh.Compile_Form(active * dx)
        local_area = fem.assemble_scalar(self._eta_area_form)
        self.params.Global_Measures["Eta_Active_Area"] = comm.allreduce(local_area, op=MPI.SUM)

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Return the Jacobian and residual of the momentum balance."""
        if step != self.Step_u:
            return
        dx = self.mesh.Get_Domain_Integrator(self.domain)

        u = self.mesh.Get_Field("u", step=self.Step_u)
        a = self.mesh.Get_Field("a")
        uOld = self.mesh.Get_Field("uOld")

        u_t = self.mesh.Get_TestFunction("u", self.Step_u)
        u_tr = self.mesh.Get_Trial_Function("u", self.Step_u)

        strain_t = Tensor_To_Mandel_UFL(nabla_s(u_t), self.dim)
        strain_tr = Tensor_To_Mandel_UFL(nabla_s(u_tr), self.dim)

        # Stresses
        stress = self.mesh.Get_Field("sigma")
        stress_vec = ufl.as_vector([stress[i, 0] for i in range(6)])
        dsigma_dStrainMandel_q = self.mesh.Get_Field("dsigma_dStrainMandel")

        F = inner(strain_t, stress_vec) * dx
        if self.consistent_stiffness:
            K = dot(strain_t, dot(dsigma_dStrainMandel_q*1.0 + self.D*0.0, strain_tr)) * dx
        else:
            K = inner(strain_t, dot(self.D, strain_tr)) * dx

        # Inertia
        if self.Inertia and self.LumpedMass==False:
            F += self.rho * inner(u_t, a) * dx
            da_du = 1.0 / (self.NM_Beta * dt_constant**2)
            K += self.rho * inner(u_t, da_du * u_tr) * dx

        # Damping
        if self.damping > 0.0:
            F += self.damping/dt_constant * inner(u_t, u-uOld) * dx
            K += self.damping/dt_constant * inner(u_t, u_tr) * dx

        # Gravity
        if self.gravity != 0.0:
            F += -self.rho * u_t[self.dim-1] * self.gravity * dx

        return K, F

    def Update_Weak_Constants(self, time_constant, dt_constant, step: int) -> bool:
        """Run the Newmark update and the integration-point update of the current displacement."""
        if step != self.Step_u:
            return False

        _, self.da_du = self.Newmark_Update(
            self.mesh.fields["u"],
            self.mesh.Get_Field("v"),
            self.mesh.Get_Field("a"),
            self.mesh.Get_Field("uOld"),
            self.mesh.Get_Field("vOld"),
            self.mesh.Get_Field("aOld"),
            self._weak_constant_value(dt_constant),
        )
        strain_tot = Tensor_To_Mandel_UFL(nabla_s(self.mesh.Get_Field("u", step=self.Step_u)),self.dim)
        self._update_ip_fields(strain_tot)
        return False

    def Update_PhaseField_Fields(self) -> None:
        """Rebuild the degradation d_e of the failure potential from the phase field."""
        phi_lim = conditional(lt(self.mesh.Get_Field("phasefield"), 1.0),self.mesh.Get_Field("phasefield"), 1.0)
        phi_lim = conditional(lt(phi_lim, 0.0), 0.0, phi_lim)
        self.d_e = (1 - self.kappa) * (1.0 - phi_lim) ** 2 + self.kappa

    def Add_KF(self, K, F, step: int) -> None:
        """
        Add algebraic terms beyond those defined in ``Get_Weak_Form``.

        ``K`` may be ``None`` during residual-only assembly.
        """

        if self.LumpedMass == False or self.Step_u != step or self.Inertia == False:
            return  # Already included in Get_Weak_Form.

        W, dofs, _vals = self.mesh.Get_Weights_Dofs(["u"], step=self.Step_u)
        vals = self.mesh.Get_Old_For_Lumped(["a"], step=self.Step_u)
        u_dofs = dofs["u"]
        a_vals = vals["a"]

        if len(u_dofs) != len(W) or len(a_vals) != len(W):
            raise ValueError(
                "Lumped displacement mass expects one weight and acceleration value "
                "per displacement component DOF."
            )

        n = len(u_dofs)
        k_indices = np.empty(n, dtype=np.int32)
        k_values  = np.empty(n)

        with F.localForm() as F_local:
            F_array = F_local.array

            for i in range(n):
                idx = u_dofs[i]
                a = a_vals[i]

                k_indices[i] = idx
                F_array[idx] += W[i] * self.rho * a
                k_values[i]   = W[i] * self.rho * self.da_du

        # Apply K contributions outside the localForm() context.  This mirrors
        # the scalar lumped operators and avoids PETSc vector/matrix state issues.
        if K is not None:
            global_indices = self.mesh.get_step_global_indices(self.Step_u, k_indices)
            for i in range(n):
                idx = global_indices[i]
                K.setValues([idx], [idx], [[k_values[i]]], addv=PETSc.InsertMode.ADD)

    def Newmark_Update(self, u, v, a, uOld, vOld, aOld, dt):
        """Update velocity and acceleration from the displacement with the Newmark relations."""
        return Apply_Newmark_Update(u,v,a,uOld,vOld,aOld,dt,self.NM_Beta,self.NM_Gamma, False, dofs=self.mesh.Get_Field_Dofs("u", self.domain))

    def initialize_fields(self) -> None:
        """Zero the model's fields and build the degradation of the failure potential."""
        for field_name in (
            "u", "uOld", "v", "vOld", "a", "aOld",
            "Plastic_Strain", "Plastic_Strain_Old",
            "LambdaField_visc", "LambdaFieldOld_visc",
            "eta", "etaOld",
            "LambdaField_coh", "LambdaFieldOld_coh",
            "ElasticEnergy", "PsiField",
        ):
            self.mesh.Zero_Field(field_name, domain=self.domain)

        self.Update_PhaseField_Fields()

    def commit(self) -> None:
        """Run the Newmark update and copy the current state to the committed fields."""
        self.mesh.Copy_Field("u", "u")
        self.Newmark_Update(
            self.mesh.fields["u"],
            self.mesh.Get_Field("v"),
            self.mesh.Get_Field("a"),
            self.mesh.Get_Field("uOld"),
            self.mesh.Get_Field("vOld"),
            self.mesh.Get_Field("aOld"),
            self.params.dt,
        )
        self.mesh.Copy_Field("u", "uOld", domain=self.domain)
        self.mesh.Copy_Field("v", "vOld", domain=self.domain)
        self.mesh.Copy_Field("a", "aOld", domain=self.domain)
        # Viscous history
        self.mesh.Copy_Field("Plastic_Strain", "Plastic_Strain_Old", domain=self.domain)
        self.mesh.Copy_Field("LambdaField_visc", "LambdaFieldOld_visc", domain=self.domain)
        # Cohesive history
        self.mesh.Copy_Field("eta", "etaOld", domain=self.domain)
        self.mesh.Copy_Field("LambdaField_coh", "LambdaFieldOld_coh", domain=self.domain)

    def reset_step(self):
        """Restore the committed state."""
        self.mesh.Copy_Field("uOld", "u", domain=self.domain)
        self.mesh.Copy_Field("vOld", "v", domain=self.domain)
        self.mesh.Copy_Field("aOld", "a", domain=self.domain)
        self.mesh.Copy_Field("Plastic_Strain_Old", "Plastic_Strain", domain=self.domain)
        self.mesh.Copy_Field("LambdaFieldOld_visc", "LambdaField_visc", domain=self.domain)
        self.mesh.Copy_Field("etaOld", "eta", domain=self.domain)
        self.mesh.Copy_Field("LambdaFieldOld_coh", "LambdaField_coh", domain=self.domain)

    # -----------------------------------------------------------------
    #  Internal Helpers
    # -----------------------------------------------------------------
    def _build_update(self, mesh: Mesh):
        """PRIVATE: the integration-point update, called as the C++ kernel is (see ``_update_ip_fields``)."""
        raise NotImplementedError

    def _read_material(self, material: dict) -> None:
        """PRIVATE: set ``lame``, ``shear``, ``bulk``, ``rho``, the creep law (``viscosity_type``,
        ``viscosity``, the J2 stress scale of the local update, 0 disables creep, and
        ``plastic_n``), ``f_t``, ``Gc`` and ``GcII`` from ``material``. Models with spatially
        varying properties override this."""
        self.lame = material["Lame"]
        self.shear = material["Shear"]
        self.bulk = material["Bulk"]
        self.rho = material["Density"]
        self.viscosity_type, self.viscosity, self.plastic_n = self._creep_law(
            material, 2.0 if self.true_2d else 3.0)
        self.Gc = material["Gc"]
        self.GcII = material.get("GcII", self.Gc)
        self.f_t = material["f_t"]

    def _build_kernel(self):
        """PRIVATE: the C++ kernel of the local problem for the model's constants."""
        return _CVEKernels(
            self.bulk, self.lame, self.shear,
            self.viscosity, self.plastic_n,
            self.f_t, self.f_s, self.GcII / self.Gc, self.DP_phi, self.DP_q,
            self.kappa, self.failureType, self.plane_stress, self.true_2d,
        )

    def _multiplier_slots(self, failure_type: str) -> int:
        """PRIVATE: components of ``LambdaField_coh`` per point: 3 for "r2_2L_PSD", 2 otherwise."""
        return _cohesive_multiplier_slots(failure_type)

    @staticmethod
    def _creep_law(material: dict, constitutive_dim: float):
        """PRIVATE: (Visc_Type, J2 stress scale, exponent n) of the creep law of ``material``."""
        viscosity_type = material.get("Visc_Type", "Newtonian")
        if viscosity_type == "Newtonian":
            return viscosity_type, constitutive_dim * material["Viscosity"], 1.0
        if viscosity_type == "Glen":
            n_exp = material["Glen_n"]
            viscosity = (constitutive_dim ** ((n_exp + 1.0) / 2.0) / (2.0 * material["Glen_A"])) ** (1.0 / n_exp)
            return viscosity_type, viscosity, n_exp
        if viscosity_type == "None":
            return viscosity_type, 0.0, 1.0
        raise NotImplementedError(f"Viscosity type '{viscosity_type}' not implemented.")

    def _create_fields(self, params, mesh):
        """PRIVATE: Register fields shared by cohesive visco-elastic models."""

        # --- Displacement / velocity / acceleration fields ---
        mesh.Add_Vector_Field("u", "Displacement [m]", "CG", params.Field_Orders["u"], self.dim, domain=self.domain)
        mesh.Add_Vector_Field("uOld", "Old Displacement [m]", "CG", params.Field_Orders["u"], self.dim, domain=self.domain)
        mesh.Add_Vector_Field("v", "Velocity [m/s]", "CG", params.Field_Orders["u"], self.dim, domain=self.domain)
        mesh.Add_Vector_Field("vOld", "Old Velocity [m/s]", "CG", params.Field_Orders["u"], self.dim, domain=self.domain)
        mesh.Add_Vector_Field("a", "Acceleration [m/s²]", "CG", params.Field_Orders["u"], self.dim, domain=self.domain)
        mesh.Add_Vector_Field("aOld", "Old Acceleration [m/s²]", "CG", params.Field_Orders["u"], self.dim, domain=self.domain)

        # --- Stress / tangent ---
        mesh.Add_Quadrature_Field("sigma", "Stress field [Pa]", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("dsigma_dStrainMandel", "dsigma/dStrainMandel field [-]", (6, 6), domain=self.domain)

        # --- Viscous (plastic) strain fields ---
        mesh.Add_Quadrature_Field("Plastic_Strain", "Viscous Strain Tensor", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("Plastic_Strain_Old", "Old Viscous Strain Tensor", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("LambdaField_visc", "Viscous internal variable field [-]", (), domain=self.domain)
        mesh.Add_Quadrature_Field("LambdaFieldOld_visc", "Old viscous internal variable field [-]", (), domain=self.domain)

        # --- Cohesive fracture strain fields ---
        mesh.Add_Quadrature_Field("eta", "Cohesive fracture strain [-]", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("etaOld", "Old cohesive fracture strain [-]", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("etaNorm", "Norm of cohesive fracture strain [-]", (), domain=self.domain)
        material = params.Materials[params.Models[self.name]["Material"]]
        slots = self._multiplier_slots(check_failure_type(material.get("FailureType", "r1")))
        mesh.Add_Quadrature_Field("LambdaField_coh", "Cohesive internal variable field [-]", (slots, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("LambdaFieldOld_coh", "Old cohesive internal variable field [-]", (slots, 1), domain=self.domain)

        # --- Auxiliary IP fields ---
        mesh.Add_Quadrature_Field("Total_Strain", "Total Strain Tensor", (6, 1), domain=self.domain)
        mesh.Add_Quadrature_Field("dam_elastic", "Elastic damage variable", (), domain=self.domain)
        mesh.Add_Quadrature_Field("ElasticEnergy", "Elastic Energy Density [J/m³]", (), domain=self.domain)
        mesh.Add_Quadrature_Field("PsiField", "Crack_Damage_Field [J/m³]", (), domain=self.domain)

        self.Step_u = params.Solution_Steps["u"]
        mesh.Add_Vector_TestFunction("u", params.dim, self.Step_u)
        self._points = _DomainPoints(mesh, self.domain, self._POINT_FIELDS)

    def _update_ip_fields(self, strain_expr) -> None:
        """PRIVATE: sample strain and d_e at the points, run the local update, write the results."""
        self._strain_expr = self._points.interpolate("Total_Strain", strain_expr, self._strain_expr)
        self._damage_expr = self._points.interpolate("dam_elastic", self.d_e, self._damage_expr)

        names = ("Total_Strain", "dam_elastic", "sigma", "eta", "etaOld", "Plastic_Strain", "Plastic_Strain_Old",
                 "dsigma_dStrainMandel", "ElasticEnergy", "PsiField", "LambdaField_visc", "LambdaField_coh",
                 "LambdaFieldOld_coh")
        arrays = self._points.gather(*names)
        total_arr, damage, *state = arrays  # the update writes e_zz back for plane-stress post-processing
        if self.true_2d:
            total_arr.reshape(-1, 6)[:, (2, 3, 4)] = 0.0

        # Creep and cohesive strain solved together in one local problem per point
        self._update_cve_state(total_arr, self.params.dt, damage, *state)
        values = dict(zip(names, arrays))
        values["etaNorm"] = np.linalg.norm(values["eta"].reshape(-1, 6), axis=1)
        self._points.scatter(values)
