"""PRIVATE: reduced (invariant) integration-point update of ``Solid:Cohesive_ViscoElastic``.

The local problem of the C++ kernel of ``Solid:Cohesive_ViscoElastic_Kernel`` (stress,
J2 creep multiplier and one or two cohesive multipliers, Fischer-Burmeister complementarity)
is isotropic: with the trial strain ``eps - eps_v_old = e_v i_hat + e_q n`` (``i_hat`` the unit
volumetric direction, ``n`` the unit deviatoric direction; in-plane components for intrinsic
2-D), the creep strain, the cohesive strain directions of every ``FailureType`` and the stress
all lie in span(i_hat, n). The stress is ``a i_hat + b n`` and the unknowns reduce to the creep
multiplier and the cohesive multipliers, two or three scalars per point:

* the creep-only state (no cohesive strain) solves the scalar equation ``x = c (1 - x)^n``
  in one dimensionless number; it is solved once per creep exponent and tabulated;
* the cohesive modes are tested at the creep-only state exactly as in the kernel (the drive
  of each mode at a probe multiplier, and the dual-norm onset test of the split r2 criteria);
* only if a mode is active, the reduced complementarity system is solved by a semi-smooth
  Newton method (Fischer-Burmeister, backtracking line search) in two or three unknowns;
* the consistent tangent follows by implicit differentiation of the reduced system, plus the
  rotation of ``n`` with the strain;
* plane stress solves sigma_zz(e_zz) = 0 per point (safeguarded Newton), with the condensed
  tangent of the kernel.

The failure potentials F(eta) of all failure types are written in the reduced coordinates
(eta = eta_v i_hat + eta_q n) with the kernel's branch selection, penalties and
regularizations, so the update reproduces the kernel to its tolerance. Points at which the
reduced Newton method does not converge are recomputed by the kernel (``fallback``).
"""

import functools
import math
from dataclasses import dataclass

import numba
import numpy as np

# Failure potentials (surface codes) and the number of cohesive multipliers of each type.
# ("r2_2L_PSD" is solved in principal space, femcubes.solids._principal_cve.)
S_R1, S_R2, S_R2DP, S_R1DP = 0, 1, 2, 3
FAILURE_TYPES = {
    "r1": (S_R1, 2), "r1_DP": (S_R1DP, 2),
    "r2_2L": (S_R2, 2), "r2_2L_DP": (S_R2, 2), "r2": (S_R2, 1),
    "r2_DP": (S_R2DP, 1),
}
DP_TYPES = ("r1_DP", "r2_DP", "r2_2L_DP")

# Creep-only table: log10 of the creep number c from -30 to 14 (44001 points, 0.001 decades).
CREEP_TABLE = (-30.0, 14.0, 44001)

# Kernel constants (cohesive_viscoelastic_kernels_internal.cpp, fracture_relations.h).
LAMBDA_PROBE = 1.0e-9       # probe multiplier of the activity test
ETA_DEV_EPS = 1.0e-10       # regularisation of |dev eta|
NORM_EPS = 1.0e-6           # smoothing of the r2 norm
SQRT_HALF = math.sqrt(0.5)


@dataclass(frozen=True)
class ReducedCVEMaterial:
    """Material constants of the reduced local problem, as used by the C++ kernel.

    ``nd`` is 2 for intrinsic 2-D (``true_2d``) and 3 for plane strain, plane stress and 3-D.
    ``bulk`` is the bulk modulus the kernel uses (cohesive viscosity, elastic energy and the
    DP confinement): ``lame + shear`` in intrinsic 2-D, the material's ``Bulk`` otherwise
    (:meth:`from_model_constants` applies this). ``viscosity`` is the J2 stress scale (0
    disables creep), ``DP_phi`` in degrees.

    ``lame``, ``shear``, ``bulk``, ``viscosity`` and ``f_t`` are numbers, or arrays with one
    value per point of the update for spatially varying properties (``constants`` then has one
    row per point); the other constants, and the creep exponent, are the same for all points.
    """

    nd: int
    lame: float
    shear: float
    bulk: float
    viscosity: float
    n_exp: float
    f_t: float
    f_s: float
    kappa: float
    failure_type: str
    G2_over_G1: float = 1.0
    DP_phi: float = 0.0
    DP_q: float = 0.0

    def __post_init__(self):
        if self.nd not in (2, 3):
            raise ValueError(f"nd must be 2 (intrinsic 2-D) or 3, got {self.nd}.")
        if self.failure_type not in FAILURE_TYPES:
            raise ValueError(f"Unknown FailureType {self.failure_type!r}; supported: {sorted(FAILURE_TYPES)}.")
        if not np.all((self.shear > 0.0) & (self.bulk > 0.0) & (self.lame + 2.0 * self.shear / self.nd > 0.0)):
            raise ValueError("The reduced update needs positive shear and bulk moduli.")
        if not (np.all(self.f_t > 0.0) and self.f_s > 0.0):
            raise ValueError("The reduced update needs positive strengths f_t and f_s.")
        if self.failure_type in ("r2_2L", "r2_2L_DP") and not np.all(self.f_t < self.f_s):
            raise ValueError("Split r2 criteria need f_s > f_t.")
        if not (0.0 < self.kappa <= 1.0):
            raise ValueError("kappa must satisfy 0 < kappa <= 1.")
        if np.any(self.viscosity < 0.0) or (np.any(self.viscosity > 0.0) and self.n_exp < 1.0):
            raise ValueError("The reduced update needs viscosity >= 0 and a creep exponent n >= 1.")
        if not self.G2_over_G1 > 0.0:
            raise ValueError("G2_over_G1 must be positive.")
        if self.failure_type in DP_TYPES:
            if self.DP_phi < 0.0 or self.DP_q < self.f_s or (self.DP_phi <= 0.0 and self.DP_q > self.f_s):
                raise ValueError("DP criteria need DP_phi >= 0, DP_q >= f_s, and DP_phi > 0 when DP_q > f_s.")

    @classmethod
    def from_model_constants(cls, *, true_2d, lame, shear, bulk, viscosity, n_exp, f_t, f_s, kappa, failure_type,
                             G2_over_G1=1.0, DP_phi=0.0, DP_q=None):
        """Constants as ``Solid:Cohesive_ViscoElastic`` passes them to its kernel (``DP_q`` defaults to f_s)."""
        return cls(nd=2 if true_2d else 3, lame=lame, shear=shear, bulk=(lame + shear) if true_2d else bulk,
                   viscosity=viscosity, n_exp=n_exp, f_t=f_t, f_s=f_s, kappa=kappa, failure_type=failure_type,
                   G2_over_G1=G2_over_G1, DP_phi=DP_phi, DP_q=f_s if DP_q is None else DP_q)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_multipliers(self) -> int:
        """Number of cohesive multipliers (2 for the split types, 1 otherwise)."""
        return FAILURE_TYPES[self.failure_type][1]

    @property
    def k_vol(self) -> float:
        """Stiffness along the unit volumetric direction, nd lame + 2 shear."""
        return self.nd * self.lame + 2.0 * self.shear

    @property
    def k_dev(self) -> float:
        """Stiffness along unit deviatoric directions, 2 shear."""
        return 2.0 * self.shear

    @property
    def gamma(self) -> float:
        """Norm of the J2 flow direction, sqrt(nd / 2)."""
        return math.sqrt(self.nd / 2.0)

    @property
    def youngs(self) -> float:
        """Young's modulus of the nd-dimensional material."""
        lame, mu = self.lame, self.shear
        if self.nd == 2:
            return 4.0 * mu * (lame + mu) / (lame + 2.0 * mu)
        return mu * (3.0 * lame + 2.0 * mu) / (lame + mu)

    def constants(self) -> np.ndarray:
        """Constants array of the compiled update (layout documented in ``_constant_index``): one
        row, or one row per point for per-point properties."""
        code, nl = FAILURE_TYPES[self.failure_type]
        f_t_vol, f_s_dev = self.f_t, self.f_s
        if code == S_R2:
            f_s_dev = self.f_s if self.nd == 2 else 2.0 * self.f_s / math.sqrt(3.0)
            if nl == 2:
                f_t_vol = self.f_t / np.sqrt(1.0 - (self.f_t / self.f_s) ** 2)
        points = np.broadcast(self.lame, self.shear, self.bulk, self.viscosity, self.f_t).shape
        c = np.zeros(points + (len(_constant_index),))
        values = {
            "k_vol": self.k_vol, "k_dev": self.k_dev, "gamma": self.gamma, "sd": math.sqrt(self.nd),
            "nd": float(self.nd), "visc_coh": 1.0e-8 * self.bulk * self.kappa, "viscosity": self.viscosity,
            "n_exp": self.n_exp, "f_t": self.f_t, "f_s": self.f_s, "f_t_vol": f_t_vol, "f_s_dev": f_s_dev,
            "bulk": self.bulk, "tan_phi": math.tan(math.radians(self.DP_phi)), "DP_q": self.DP_q,
            "G2_over_G1": self.G2_over_G1, "kappa": self.kappa, "code": float(code), "nl": float(nl),
            "dp_split": 1.0 if self.failure_type == "r2_2L_DP" else 0.0,
        }
        for key, value in values.items():
            c[..., _constant_index[key]] = value
        return c


_constant_index = {key: i for i, key in enumerate((
    "k_vol", "k_dev", "gamma", "sd", "nd", "visc_coh", "viscosity", "n_exp", "f_t", "f_s", "f_t_vol", "f_s_dev",
    "bulk", "tan_phi", "DP_q", "G2_over_G1", "kappa", "code", "nl", "dp_split"))}


@functools.lru_cache(maxsize=8)
def creep_only_table(n_exp):
    """Creep-only solution s_v = -log(1 - x) of x = c (1 - x)^n on the table grid (bisection to
    round-off), with ds_v / dlog10(c): (axis (lo, step), s_v, ds_v)."""
    lo, hi, n = CREEP_TABLE
    log_c = np.linspace(lo, hi, n)
    c = 10.0 ** log_c
    x_lo, x_hi = np.zeros(n), np.ones(n)
    for _ in range(200):
        mid = 0.5 * (x_lo + x_hi)
        below = mid - c * (1.0 - mid) ** n_exp < 0.0
        x_lo, x_hi = np.where(below, mid, x_lo), np.where(below, x_hi, mid)
    x = 0.5 * (x_lo + x_hi)
    s_v = -np.log1p(-x)
    dx_dc = (1.0 - x) ** n_exp / (1.0 + n_exp * c * (1.0 - x) ** (n_exp - 1.0))
    ds_v = c * math.log(10.0) * dx_dc / (1.0 - x)
    return np.array([lo, (hi - lo) / (n - 1)]), s_v, ds_v


class ReducedCVEUpdate:
    """Integration-point update of ``Solid:Cohesive_ViscoElastic`` in reduced form (every FailureType but "r2_2L_PSD").

    Called like the C++ kernel, ``update(strain, dt, damage, stress, eta, eta_old, strain_v,
    strain_v_old, tangent, elastic_energy, psi, lambda_v, lambda_c, lambda_c_old)`` with flat
    arrays (Mandel vectors of 6, tangents of 36, two cohesive multipliers per point); writes
    ``stress``, ``eta``, ``strain_v``, ``tangent``, ``elastic_energy``, ``psi``, ``lambda_v``
    and ``lambda_c`` and returns the number of points recomputed by ``fallback``. The reduced
    solution depends only on the strain, the committed viscous strain, the time step and the
    damage; ``eta_old`` and ``lambda_c_old`` (the kernel's warm start) are passed on to the
    fallback only.

    ``plane_stress=True`` (``material.nd == 3``): e_zz is solved from sigma_zz = 0 and written
    back into ``strain``; the tangent is condensed, with zero zz row and column, as in the
    kernel. ``fallback`` is a kernel-signature update (the C++ kernel of the same material) for
    points at which the reduced Newton method does not converge; without it such points raise.
    With per-point properties in ``material``, its arrays hold one value per point of the call.
    """

    TOL = 1.0e-12                   # reduced residual, relative to the stress scale
    MAX_ITER = 60
    PLANE_STRESS_TOL = 1.0e-9       # |sigma_zz| relative to max(d f, |sigma_xx| + |sigma_yy|)
    PLANE_STRESS_MAX_ITER = 60

    def __init__(self, material: ReducedCVEMaterial, plane_stress=False, fallback=None):
        if plane_stress and material.nd != 3:
            raise ValueError("Plane stress needs the 3-D (nd = 3) reduced material.")
        self.material = material
        self.plane_stress = bool(plane_stress)
        self.fallback = fallback
        self.const = np.atleast_2d(material.constants())
        self.table = creep_only_table(float(material.n_exp))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, strain, dt, damage, stress, eta, eta_old, strain_v, strain_v_old, tangent,
                 elastic_energy, psi, lambda_v, lambda_c, lambda_c_old):
        n = damage.size
        if n == 0:
            return 0
        eps, eps_v_old = strain.reshape(n, 6), strain_v_old.reshape(n, 6)
        failed = np.zeros(n, dtype=np.bool_)
        if self.plane_stress:
            eps[:, 2] = _plane_stress_e_zz(np.ascontiguousarray(eps, dtype=np.float64),
                                           np.ascontiguousarray(eps_v_old, dtype=np.float64),
                                           np.ascontiguousarray(damage.reshape(n), dtype=np.float64), float(dt),
                                           self.const, *self.table, np.atleast_1d(self._creep_coefficient(dt)), self.TOL,
                                           self.MAX_ITER, self.PLANE_STRESS_TOL, self.PLANE_STRESS_MAX_ITER, failed)
        _update_points(np.ascontiguousarray(eps, dtype=np.float64), np.ascontiguousarray(eps_v_old, dtype=np.float64),
                       np.ascontiguousarray(damage.reshape(n), dtype=np.float64), float(dt), self.const, *self.table,
                       np.atleast_1d(self._creep_coefficient(dt)), self.TOL, self.MAX_ITER, stress.reshape(n, 6), eta.reshape(n, 6),
                       strain_v.reshape(n, 6), tangent.reshape(n, 36), elastic_energy.reshape(n), psi.reshape(n),
                       lambda_v.reshape(n), lambda_c.reshape(n, 2), failed)
        if self.plane_stress:
            _condense_zz(tangent.reshape(n, 36))
        return self._recompute_failed(failed, strain, dt, damage, stress, eta, eta_old, strain_v, strain_v_old,
                                      tangent, elastic_energy, psi, lambda_v, lambda_c, lambda_c_old)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _creep_coefficient(self, dt):
        """PRIVATE: gamma dt (gamma 2 mu / eta)^n (creep-only number per e_q^(n-1)); 0 without creep.
        A number, or an array with one value per point for per-point properties."""
        m = self.material
        if np.ndim(m.viscosity) == 0:
            if m.viscosity <= 0.0:
                return 0.0
            return float(m.gamma * dt * (m.gamma * m.k_dev / m.viscosity) ** m.n_exp)
        creep = m.viscosity > 0.0
        return np.where(creep, m.gamma * dt * (m.gamma * m.k_dev / np.where(creep, m.viscosity, 1.0)) ** m.n_exp, 0.0)

    def _recompute_failed(self, failed, strain, dt, damage, stress, eta, eta_old, strain_v, strain_v_old, tangent,
                          elastic_energy, psi, lambda_v, lambda_c, lambda_c_old):
        """PRIVATE: recompute the points the reduced Newton method did not converge at with the fallback."""
        idx = np.flatnonzero(failed)
        if idx.size == 0:
            return 0
        if self.fallback is None:
            raise RuntimeError(f"Reduced CVE update did not converge at {idx.size} points and has no fallback.")
        n = damage.size
        views = [a.reshape(n, -1) for a in (strain, stress, eta, eta_old, strain_v, strain_v_old, tangent,
                                            elastic_energy, psi, lambda_v, lambda_c, lambda_c_old)]
        sub = [np.ascontiguousarray(v[idx]).ravel() for v in views]
        dam = np.ascontiguousarray(damage.reshape(n)[idx])
        self._call_fallback(idx, sub, dt, dam)
        for v, s in zip(views, sub):
            v[idx] = s.reshape(idx.size, -1)
        return int(idx.size)

    def _call_fallback(self, idx, sub, dt, damage):
        """PRIVATE: run the fallback on the points ``idx``; ``sub`` holds their arrays in the
        kernel's order (strain, stress, ..., lambda_c_old), ``damage`` their damage."""
        self.fallback(sub[0], dt, damage, *sub[1:])


# ----------------------------------------------------------------------
# Compiled helpers
# ----------------------------------------------------------------------

@numba.njit(cache=True)
def _point_invariants(e, nd, nvec):
    """PRIVATE: (e_v, e_q) of one trial strain e (Mandel, 6; in-plane components for nd = 2); the
    unit deviatoric direction is written into ``nvec`` (no temporary arrays)."""
    if nd == 2:
        e_v = (e[0] + e[1]) / math.sqrt(2.0)
        h = 0.5 * (e[0] - e[1])
        nvec[0] = h
        nvec[1] = -h
        nvec[2] = 0.0
        nvec[3] = 0.0
        nvec[4] = 0.0
        nvec[5] = e[5]
    else:
        tr = e[0] + e[1] + e[2]
        e_v = tr / math.sqrt(3.0)
        for k in range(6):
            nvec[k] = e[k]
        for k in range(3):
            nvec[k] -= tr / 3.0
    q = 0.0
    for k in range(6):
        q += nvec[k] * nvec[k]
    q = math.sqrt(q)
    if q > 1e-30:
        for k in range(6):
            nvec[k] = nvec[k] / q
    else:
        for k in range(6):
            nvec[k] = 0.0
        if nd == 2:
            nvec[0], nvec[1] = 1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0)
        else:
            nvec[0], nvec[1], nvec[2] = 2.0 / math.sqrt(6.0), -1.0 / math.sqrt(6.0), -1.0 / math.sqrt(6.0)
    return e_v, q


@numba.njit(cache=True)
def _invariants(eps, eps_v_old, nd):
    """PRIVATE: (e_v, e_q, n) of the trial strains eps - eps_v_old (in-plane components for nd = 2)."""
    n = eps.shape[0]
    e_v = np.empty(n)
    e_q = np.empty(n)
    nvec = np.zeros((n, 6))
    for i in range(n):
        e_v[i], e_q[i] = _point_invariants(eps[i] - eps_v_old[i], nd, nvec[i])
    return e_v, e_q, nvec


@numba.njit(cache=True)
def _point_creep(c, e_q, n_exp, axis, s_tab, ds_tab):
    """PRIVATE: s_v(log10 c) of the creep-only state by cubic Hermite interpolation, and ds_v/de_q.

    Below the table s_v = c (exact small-creep limit); above it the large-c asymptote
    s_v ~ log(c) / n continues the last table value.
    """
    if c <= 1e-30:
        return c, (n_exp - 1.0) * c / e_q
    lo, h = axis[0], axis[1]
    m = s_tab.shape[0]
    ln10 = math.log(10.0)
    L = math.log10(c)
    x = (L - lo) / h
    if x >= m - 1:
        return s_tab[m - 1] + (L - (lo + (m - 1) * h)) * ln10 / n_exp, (n_exp - 1.0) / (n_exp * e_q)
    j = max(int(math.floor(x)), 0)
    u = x - j
    y0, y1, d0, d1 = s_tab[j], s_tab[j + 1], ds_tab[j] * h, ds_tab[j + 1] * h
    s_v = ((2 * u ** 3 - 3 * u ** 2 + 1) * y0 + (u ** 3 - 2 * u ** 2 + u) * d0
           + (-2 * u ** 3 + 3 * u ** 2) * y1 + (u ** 3 - u ** 2) * d1)
    dsdx = ((6 * u ** 2 - 6 * u) * y0 + (3 * u ** 2 - 4 * u + 1) * d0
            + (-6 * u ** 2 + 6 * u) * y1 + (3 * u ** 2 - 2 * u) * d1)
    return s_v, dsdx / h * (n_exp - 1.0) / (e_q * ln10)


@numba.njit(cache=True)
def _creep_from_table(c, e_q, n_exp, axis, s_tab, ds_tab):
    """PRIVATE: s_v and ds_v/de_q of the creep-only state for all points (see _point_creep)."""
    n = c.shape[0]
    s_v = np.empty(n)
    ds_q = np.empty(n)
    for i in range(n):
        s_v[i], ds_q[i] = _point_creep(c[i], e_q[i], n_exp, axis, s_tab, ds_tab)
    return s_v, ds_q


@numba.njit(cache=True)
def _condense_zz(tangent):
    """PRIVATE: plane-stress condensation of the (n, 36) tangents in place: C - C[:, zz] C[zz, :] / C[zz, zz],
    with zero zz row and column (as in the C++ kernel)."""
    for i in range(tangent.shape[0]):
        czz = tangent[i, 14]
        for r in range(6):
            for s in range(6):
                if r != 2 and s != 2:
                    tangent[i, 6 * r + s] -= tangent[i, 6 * r + 2] * tangent[i, 12 + s] / czz
        for k in range(6):
            tangent[i, 6 * k + 2] = 0.0
            tangent[i, 12 + k] = 0.0


@numba.njit(cache=True)
def _r2_dev_scale(strength, nd):
    """PRIVATE: deviatoric scale of the r2 potentials (2 f / sqrt(3) in 3-D)."""
    return strength if nd == 2 else 2.0 * strength / math.sqrt(3.0)


@numba.njit(cache=True)
def _surface(eta_v, eta_q, e_v, d, c):
    """PRIVATE: failure potential in reduced coordinates, eta = eta_v i_hat + eta_q n, at the
    pre-cohesive volumetric strain e_v and damage d.

    Returns (F, F_v, F_q, H_vv, H_vq, H_qq, G_v, G_q, psi): F and its gradient and Hessian with
    respect to (eta_v, eta_q), the derivatives G of the gradient with respect to e_v (strain-
    dependent DP strengths) and psi, the undamaged potential the kernel outputs (F_damage).
    Branches, penalties and regularizations as in shared/fracture_relations.h.
    """
    sd, f_t, f_s, bulk, tan_phi, dp_q = c[3], c[8], c[9], c[12], c[13], c[14]
    code = int(c[17])
    nd = int(c[4])
    tension = e_v >= 0.0
    trace_s = sd * e_v
    rq = math.sqrt(eta_q * eta_q + ETA_DEV_EPS * ETA_DEV_EPS)
    f2s = SQRT_HALF * (rq - ETA_DEV_EPS)
    df2s = SQRT_HALF * eta_q / rq
    ddf2s = SQRT_HALF * ETA_DEV_EPS * ETA_DEV_EPS / (rq * rq * rq)
    F = F_v = F_q = H_vv = H_vq = H_qq = G_v = G_q = psi = 0.0
    if code == S_R1:
        if tension:
            F = d * (f_t * eta_v / sd + f_s * f2s)
            F_v = d * f_t / sd
            psi = f_t * eta_v / sd + f_s * f2s
        else:
            F = -1.0e6 * f_t * eta_v / sd + d * f_s * f2s
            F_v = -1.0e6 * f_t / sd
            psi = f_s * f2s
        F_q = d * f_s * df2s
        H_qq = d * f_s * ddf2s
    elif code == S_R2:
        f_t_vol, f_s_dev = c[10], c[11]
        f1 = f_t_vol * eta_v / sd if tension else 0.0
        df1 = f_t_vol / sd if tension else 0.0
        f_dev = f_s_dev
        df_eta = 0.0     # d f_dev / d eta_v
        df_e = 0.0       # d f_dev / d e_v
        if c[19] > 0.0 and not tension:
            # r2_2L_DP: confinement from the elastic pressure of the pre-cohesive strain minus eta
            elastic_trace = sd * (e_v - eta_v)
            pressure = max(-bulk * elastic_trace, 0.0)
            confined = min(f_s + pressure * tan_phi, dp_q)
            f_dev = _r2_dev_scale(confined, nd)
            if elastic_trace < 0.0 and confined < dp_q:
                scale = _r2_dev_scale(bulk * tan_phi, nd)
                df_eta = scale * sd
                df_e = -scale * sd
        w = 0.5 * f_dev * f_dev
        q2 = eta_q * eta_q
        g_v = f1 * df1 + 0.5 * f_dev * q2 * df_eta
        g_q = w * eta_q
        h_vv = df1 * df1 + 0.5 * q2 * df_eta * df_eta
        h_vq = f_dev * eta_q * df_eta
        norm = math.sqrt(f1 * f1 + w * q2 + NORM_EPS * NORM_EPS)
        n3 = norm * norm * norm
        F = d * (norm - NORM_EPS)
        psi = norm - NORM_EPS
        F_v = d * g_v / norm
        F_q = d * g_q / norm
        H_vv = d * h_vv / norm - d * g_v * g_v / n3
        H_vq = d * h_vq / norm - d * g_v * g_q / n3
        H_qq = d * w / norm - d * g_q * g_q / n3
        if df_e != 0.0:
            # derivative of the gradient with respect to the confined strength, times d f_dev / d e_v
            G_v = d * (0.5 * q2 * df_eta / norm - 0.5 * f_dev * q2 * g_v / n3) * df_e
            G_q = d * (f_dev * eta_q / norm - 0.5 * f_dev * q2 * g_q / n3) * df_e
        if not tension:
            F += -1.0e6 * f_t_vol * eta_v / sd
            F_v += -1.0e6 * f_t_vol / sd
    elif code == S_R2DP:
        if e_v > 0.0:
            f1 = f_t * eta_v / sd
            df1 = f_t / sd
            f2 = f_s * f2s
            df2 = f_s * df2s
            ddf2 = f_s * ddf2s
            norm = math.sqrt(f1 * f1 + f2 * f2)
            reg = math.sqrt(f1 * f1 + f2 * f2 + 1.0e-20)
            g_v = f1 * df1
            g_q = f2 * df2
            F = d * norm
            psi = norm
            F_v = d * g_v / reg
            F_q = d * g_q / reg
            H_vv = d / reg * (df1 * df1 - g_v * g_v / (reg * reg))
            H_vq = d / reg * (-g_v * g_q / (reg * reg))
            H_qq = d / reg * (df2 * df2 + f2 * ddf2 - g_q * g_q / (reg * reg))
        else:
            q_dp = min(f_s - bulk * tan_phi * trace_s, dp_q)
            F = -1.0e12 * f_t * eta_v / sd + d * q_dp * f2s
            psi = q_dp * f2s
            F_v = -1.0e12 * f_t / sd
            F_q = d * q_dp * df2s
            H_qq = d * q_dp * ddf2s
            if q_dp < dp_q:
                G_q = -d * bulk * tan_phi * sd * df2s
    else:   # S_R1DP
        if tension:
            F = d * f_t * eta_v / sd
            F_v = d * f_t / sd
            psi = f_t * eta_v / sd
        else:
            F = -1.0e6 * f_t * eta_v / sd
            F_v = -1.0e6 * f_t / sd
        q0 = max(f_s - f_t / 3.0 * tan_phi, 0.0)
        if trace_s * bulk >= f_t / 3.0:
            q = q0
        else:
            q = min(f_s - trace_s * bulk * tan_phi, dp_q)
            if q < dp_q:
                G_q = -d * bulk * tan_phi * sd * df2s
        F += d * q * f2s
        psi += q0 * f2s / c[15]
        F_q = d * q * df2s
        H_qq = d * q * ddf2s
    return F, F_v, F_q, H_vv, H_vq, H_qq, G_v, G_q, psi


@numba.njit(cache=True)
def _direction(nl, i, e_v, s_q):
    """PRIVATE: cohesive direction of mode i in reduced coordinates (dir_v, dir_q) at the
    pre-cohesive strain (e_v, s_q), and its derivatives with respect to e_v and s_q:
    (dir_v, dir_q, d dir_v/d e_v, d dir_q/d e_v, d dir_v/d s_q, d dir_q/d s_q)."""
    if nl == 2:
        if i == 0:
            return (1.0 if e_v >= 0.0 else -1.0), 0.0, 0.0, 0.0, 0.0, 0.0
        return 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    if e_v < 0.0:
        return 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    r2 = e_v * e_v + s_q * s_q + 1.0e-20
    r = math.sqrt(r2)
    r3 = r2 * r
    return (e_v / r, s_q / r, (s_q * s_q + 1.0e-20) / r3, -e_v * s_q / r3,
            -e_v * s_q / r3, (e_v * e_v + 1.0e-20) / r3)


@numba.njit(cache=True)
def _fischer_burmeister(x, y, scale):
    """PRIVATE: phi(x, y) = sqrt(x^2 + y^2) - x - y (stable form) and its partial derivatives."""
    r = math.sqrt(x * x + y * y)
    den = r + x + y
    phi = -2.0 * x * y / den if den > 0.0 else r - x - y
    if r > 1.0e-14 * scale:
        return phi, x / r - 1.0, y / r - 1.0
    return phi, 1.0 / math.sqrt(2.0) - 1.0, 1.0 / math.sqrt(2.0) - 1.0


@numba.njit(cache=True)
def _system(u, nl, e_v, e_q, d, creep_rate, c, R, J, JE):
    """PRIVATE: reduced residual R (nl + 1), its Jacobian J with respect to u = (Lambda_v,
    lambda_1[, lambda_2]) and with respect to (e_v, e_q) (JE), and the stress coordinates
    (a, b), their derivatives (da/du, db/du via return, da/de, db/de at fixed u), eta and psi.

    R_0 = Lambda_v - creep_rate * b^n (J2 creep); R_i = FB(k_dev lambda_i, drive_i) with the
    cohesive drive dF/dlambda_i - sigma . dir_i + visc_coh lambda_i (kernel's residual)."""
    k_vol, k_dev, gamma, visc, n_exp = c[0], c[1], c[2], c[5], c[7]
    lam_v = u[0]
    s_q = e_q - gamma * lam_v
    dirs = np.empty((2, 6))
    eta_v = eta_q = 0.0
    dev_dsv = deq_dsv = dev_dsq = deq_dsq = 0.0
    for i in range(nl):
        dv, dq, dv_ev, dq_ev, dv_sq, dq_sq = _direction(nl, i, e_v, s_q)
        dirs[i, 0], dirs[i, 1], dirs[i, 2], dirs[i, 3], dirs[i, 4], dirs[i, 5] = dv, dq, dv_ev, dq_ev, dv_sq, dq_sq
        lam = u[1 + i]
        eta_v += lam * dv
        eta_q += lam * dq
        dev_dsv += lam * dv_ev
        deq_dsv += lam * dq_ev
        dev_dsq += lam * dv_sq
        deq_dsq += lam * dq_sq
    a = k_vol * (e_v - eta_v)
    b = k_dev * (s_q - eta_q)
    F, F_v, F_q, H_vv, H_vq, H_qq, G_v, G_q, psi = _surface(eta_v, eta_q, e_v, d, c)
    w_v = F_v - a
    w_q = F_q - b
    # derivatives at fixed (e_v, e_q)
    dev_dL = -gamma * dev_dsq
    deq_dL = -gamma * deq_dsq
    da_dL = -k_vol * dev_dL
    db_dL = -k_dev * (gamma + deq_dL)
    # derivatives at fixed u
    da_dev = k_vol * (1.0 - dev_dsv)
    da_deq = -k_vol * dev_dsq
    db_dev = -k_dev * deq_dsv
    db_deq = k_dev * (1.0 - deq_dsq)
    bp = max(b, 0.0)
    creep = creep_rate * bp ** n_exp
    dcreep = n_exp * creep_rate * bp ** (n_exp - 1.0) if bp > 0.0 else 0.0
    R[0] = lam_v - creep
    J[0, 0] = 1.0 - dcreep * db_dL
    JE[0, 0] = -dcreep * db_dev
    JE[0, 1] = -dcreep * db_deq
    da_dl = np.empty(2)
    db_dl = np.empty(2)
    for j in range(nl):
        da_dl[j] = -k_vol * dirs[j, 0]
        db_dl[j] = -k_dev * dirs[j, 1]
        J[0, 1 + j] = -dcreep * db_dl[j]
    scale = max(d * max(c[8], c[9]), 1.0e-300)
    for i in range(nl):
        dv, dq = dirs[i, 0], dirs[i, 1]
        drive = w_v * dv + w_q * dq + visc * u[1 + i]
        phi, dphi_x, dphi_y = _fischer_burmeister(k_dev * u[1 + i], drive, scale)
        R[1 + i] = phi
        dwv_dL = H_vv * dev_dL + H_vq * deq_dL - da_dL
        dwq_dL = H_vq * dev_dL + H_qq * deq_dL - db_dL
        ddrive_dL = dwv_dL * dv + dwq_dL * dq - gamma * (w_v * dirs[i, 4] + w_q * dirs[i, 5])
        J[1 + i, 0] = dphi_y * ddrive_dL
        for j in range(nl):
            dwv = H_vv * dirs[j, 0] + H_vq * dirs[j, 1] - da_dl[j]
            dwq = H_vq * dirs[j, 0] + H_qq * dirs[j, 1] - db_dl[j]
            J[1 + i, 1 + j] = dphi_y * (dwv * dv + dwq * dq + (visc if i == j else 0.0))
        J[1 + i, 1 + i] += dphi_x * k_dev
        dwv_dev = H_vv * dev_dsv + H_vq * deq_dsv + G_v - da_dev
        dwq_dev = H_vq * dev_dsv + H_qq * deq_dsv + G_q - db_dev
        dwv_deq = H_vv * dev_dsq + H_vq * deq_dsq - da_deq
        dwq_deq = H_vq * dev_dsq + H_qq * deq_dsq - db_deq
        JE[1 + i, 0] = dphi_y * (dwv_dev * dv + dwq_dev * dq + w_v * dirs[i, 2] + w_q * dirs[i, 3])
        JE[1 + i, 1] = dphi_y * (dwv_deq * dv + dwq_deq * dq + w_v * dirs[i, 4] + w_q * dirs[i, 5])
    return a, b, da_dL, db_dL, da_dl, db_dl, da_dev, da_deq, db_dev, db_deq, eta_v, eta_q, psi


@numba.njit(cache=True)
def _solve_small(J, rhs, m):
    """PRIVATE: solve J x = rhs (m <= 3) by Gaussian elimination with partial pivoting; rhs is
    overwritten with x. Returns False for a singular matrix."""
    A = J[:m, :m].copy()
    for k in range(m):
        p = k
        for r in range(k + 1, m):
            if abs(A[r, k]) > abs(A[p, k]):
                p = r
        if A[p, k] == 0.0:
            return False
        if p != k:
            for s in range(m):
                A[k, s], A[p, s] = A[p, s], A[k, s]
            rhs[k], rhs[p] = rhs[p], rhs[k]
        for r in range(k + 1, m):
            f = A[r, k] / A[k, k]
            for s in range(k, m):
                A[r, s] -= f * A[k, s]
            rhs[r] -= f * rhs[k]
    for k in range(m - 1, -1, -1):
        acc = rhs[k]
        for s in range(k + 1, m):
            acc -= A[k, s] * rhs[s]
        rhs[k] = acc / A[k, k]
    return True


@numba.njit(cache=True)
def _merit(R, m, k_dev):
    """PRIVATE: squared residual norm in stress units."""
    acc = (k_dev * R[0]) ** 2
    for i in range(1, m):
        acc += R[i] * R[i]
    return acc


@numba.njit(cache=True)
def _point_state(e_v, e_q, d, s_v, ds_v, dt, creep_rate, c, tol, max_iter):
    """PRIVATE: reduced state of one point. Returns (a, da/de_v, da/de_q, b, db/de_v, db/de_q,
    Lambda_v, eta_v, eta_q, lambda_1, lambda_2, psi, converged).

    Creep-only state from the table (s_v, ds_v); if a cohesive mode is active there (kernel
    tests), the reduced complementarity system is solved by semi-smooth Newton with a
    backtracking line search; the derivatives follow by implicit differentiation."""
    k_vol, k_dev, gamma, sd = c[0], c[1], c[2], c[3]
    nl = int(c[18])
    ex = math.exp(-s_v)
    lam_creep = e_q * (1.0 - ex) / gamma
    a0 = k_vol * e_v
    b0 = k_dev * e_q * ex
    # activity test at the creep-only state, with probe multipliers (kernel)
    s_q0 = e_q - gamma * lam_creep
    eta_v = eta_q = 0.0
    for i in range(nl):
        dv, dq, _, _, _, _ = _direction(nl, i, e_v, s_q0)
        eta_v += LAMBDA_PROBE * dv
        eta_q += LAMBDA_PROBE * dq
    F, F_v, F_q, H_vv, H_vq, H_qq, G_v, G_q, psi0 = _surface(eta_v, eta_q, e_v, d, c)
    min_exceed = 1.0e-6 * d * max(c[8], c[9])
    active = np.zeros(2, dtype=np.bool_)
    drives = np.zeros(2)
    any_active = False
    for i in range(nl):
        dv, dq, _, _, _, _ = _direction(nl, i, e_v, s_q0)
        drives[i] = (F_v - a0) * dv + (F_q - b0) * dq
        if drives[i] < -min_exceed:
            active[i] = True
            any_active = True
    if nl == 2 and int(c[17]) == S_R2 and e_v >= 0.0:
        # Split r2 in tension: the onset is the dual norm of the volumetric and deviatoric stress
        # projections. The probe drives are not a valid test here: at probe multipliers the
        # smoothed norm's gradient points along (1, 1), and inside the surface the kernel then
        # finds "regularised roots" (multipliers ~1e-7 f_t/E) where the sharp potential has none.
        ratio = math.hypot(max(a0, 0.0) / (d * c[10] / sd), max(b0, 0.0) / (d * c[11] / math.sqrt(2.0)))
        active[0] = ratio > 1.0 + 1.0e-10 and a0 > 0.0
        active[1] = ratio > 1.0 + 1.0e-10 and b0 > 0.0
        any_active = active[0] or active[1]
    if not any_active:
        _, _, _, _, _, _, _, _, psi = _surface(0.0, 0.0, e_v, d, c)
        db_deq = k_dev * (ex - e_q * ex * ds_v)
        return a0, k_vol, 0.0, b0, 0.0, db_deq, lam_creep, 0.0, 0.0, 0.0, 0.0, psi, True
    # semi-smooth Newton on (Lambda_v, lambda_i) from the creep-only state; elastic estimate
    # of the active multipliers
    m = nl + 1
    u = np.zeros(3)
    u[0] = lam_creep
    for i in range(nl):
        if active[i]:
            dv, dq, _, _, _, _ = _direction(nl, i, e_v, s_q0)
            u[1 + i] = max(-drives[i] / (k_vol * dv * dv + k_dev * dq * dq), LAMBDA_PROBE)
    R = np.zeros(3)
    J = np.zeros((3, 3))
    JE = np.zeros((3, 2))
    trial = np.zeros(3)
    R_t = np.zeros(3)
    J_t = np.zeros((3, 3))
    JE_t = np.zeros((3, 2))
    step = np.zeros(3)
    stress_scale = max(d * max(c[8], c[9]), abs(a0), abs(b0))
    _system(u, nl, e_v, e_q, d, creep_rate, c, R, J, JE)
    merit = _merit(R, m, k_dev)
    converged = False
    for it in range(max_iter):
        if math.sqrt(merit) <= tol * stress_scale:
            converged = True
            break
        for k in range(m):
            step[k] = -R[k]
        if not _solve_small(J, step, m):
            break
        alpha = 1.0
        accepted = False
        for _ in range(30):
            for k in range(m):
                trial[k] = u[k] + alpha * step[k]
            trial[0] = max(trial[0], 0.0)
            for i in range(nl):
                trial[1 + i] = max(trial[1 + i], 0.0)
            _system(trial, nl, e_v, e_q, d, creep_rate, c, R_t, J_t, JE_t)
            merit_t = _merit(R_t, m, k_dev)
            if merit_t <= (1.0 - 1.0e-4 * alpha) * merit:
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            # stagnation: converged only if already at round-off
            converged = math.sqrt(merit) <= 1.0e3 * tol * stress_scale
            break
        for k in range(m):
            u[k] = trial[k]
            R[k] = R_t[k]
            for s in range(m):
                J[k, s] = J_t[k, s]
        merit = merit_t
    a, b, da_dL, db_dL, da_dl, db_dl, da_dev, da_deq, db_dev, db_deq, eta_v, eta_q, psi = _system(
        u, nl, e_v, e_q, d, creep_rate, c, R, J, JE)
    if not converged:
        converged = math.sqrt(_merit(R, m, k_dev)) <= tol * stress_scale
    # implicit differentiation: du/de = -J^-1 JE
    du_dev = np.zeros(3)
    du_deq = np.zeros(3)
    for k in range(m):
        du_dev[k] = -JE[k, 0]
        du_deq[k] = -JE[k, 1]
    ok_v = _solve_small(J, du_dev, m)
    ok_q = _solve_small(J, du_deq, m)
    if not (ok_v and ok_q):
        converged = False
    da_v = da_dev + da_dL * du_dev[0]
    da_q = da_deq + da_dL * du_deq[0]
    db_v = db_dev + db_dL * du_dev[0]
    db_q = db_deq + db_dL * du_deq[0]
    for j in range(nl):
        da_v += da_dl[j] * du_dev[1 + j]
        da_q += da_dl[j] * du_deq[1 + j]
        db_v += db_dl[j] * du_dev[1 + j]
        db_q += db_dl[j] * du_deq[1 + j]
    lam2 = u[2] if nl == 2 else 0.0
    return a, da_v, da_q, b, db_v, db_q, u[0], eta_v, eta_q, u[1], lam2, psi, converged


@numba.njit(cache=True)
def _creep_only(e_q, creep_coef, n_exp, axis, s_tab, ds_tab):
    """PRIVATE: creep-only (s_v, ds_v/de_q) of one point from the table."""
    safe_eq = e_q if e_q > 1.0e-300 else 1.0
    cnum = creep_coef * safe_eq ** (n_exp - 1.0) if e_q > 1.0e-300 else 0.0
    return _point_creep(cnum, safe_eq, n_exp, axis, s_tab, ds_tab)


@numba.njit(cache=True)
def _update_points(eps, eps_v_old, dam, dt, c, axis, s_tab, ds_tab, creep_coef, tol, max_iter,
                   stress, eta_out, strain_v, tangent, energy, psi_out, lam_v_out, lam_c, failed):
    """PRIVATE: reduced update of all points into the output arrays. Points marked ``failed`` (by
    the plane-stress solve) and points it does not converge at are marked and left untouched.
    ``c`` (constants) and ``creep_coef`` have one row for all points or one row per point."""
    nd = c[0, 4]
    n = eps.shape[0]
    nd_i = int(nd)
    i_hat = np.zeros(6)
    for k in range(nd_i):
        i_hat[k] = 1.0 / math.sqrt(nd)
    p_dev = np.zeros((6, 6))
    if nd_i == 2:
        p_dev[0, 0] = 0.5
        p_dev[1, 1] = 0.5
        p_dev[0, 1] = -0.5
        p_dev[1, 0] = -0.5
        p_dev[5, 5] = 1.0
    else:
        for r in range(3):
            for s in range(3):
                p_dev[r, s] = (1.0 if r == s else 0.0) - 1.0 / 3.0
        p_dev[3, 3] = 1.0
        p_dev[4, 4] = 1.0
        p_dev[5, 5] = 1.0
    e = np.zeros(6)
    nv = np.zeros(6)
    for i in range(n):
        if failed[i]:
            continue
        row = i if c.shape[0] > 1 else 0
        ci = c[row]
        k_vol, k_dev, gamma, viscosity, n_exp, bulk, kappa = ci[0], ci[1], ci[2], ci[6], ci[7], ci[12], ci[16]
        creep_rate = dt * (gamma / viscosity) ** n_exp if viscosity > 0.0 else 0.0
        for k in range(6):
            e[k] = eps[i, k] - eps_v_old[i, k]
        e_v, e_q = _point_invariants(e, nd_i, nv)
        d = min(max(dam[i], kappa), 1.0)
        s_v, ds_v = _creep_only(e_q, creep_coef[row], n_exp, axis, s_tab, ds_tab)
        a, da_v, da_q, b, db_v, db_q, lam_v, eta_v, eta_q, lam1, lam2, psi, ok = _point_state(
            e_v, e_q, d, s_v, ds_v, dt, creep_rate, ci, tol, max_iter)
        if not ok:
            # leave the point's outputs (the kernel's warm-start inputs) untouched for the fallback
            failed[i] = True
            continue
        ratio = b / e_q if e_q > 1e-30 else k_dev
        for r in range(6):
            stress[i, r] = a * i_hat[r] + b * nv[r]
            strain_v[i, r] = eps_v_old[i, r] + gamma * lam_v * nv[r]
            eta_out[i, r] = eta_v * i_hat[r] + eta_q * nv[r]
            for s in range(6):
                tangent[i, 6 * r + s] = (i_hat[r] * (da_v * i_hat[s] + da_q * nv[s])
                                         + nv[r] * (db_v * i_hat[s] + db_q * nv[s])
                                         + ratio * (p_dev[r, s] - nv[r] * nv[s]))
        # the kernel's elastic energy, K tr(e_el)^2 / 2 + mu |dev e_el|^2, tr(e_el) = sqrt(nd) a / k_vol
        energy[i] = 0.5 * (bulk * nd * (a / k_vol) ** 2 + b * b / k_dev)
        psi_out[i] = psi
        lam_v_out[i] = lam_v
        lam_c[i, 0] = lam1
        lam_c[i, 1] = lam2


@numba.njit(cache=True)
def _plane_stress_e_zz(eps, eps_v_old, dam, dt, c, axis, s_tab, ds_tab, creep_coef, tol, max_iter, ps_tol,
                       ps_max_iter, failed):
    """PRIVATE: e_zz with sigma_zz = 0 for every point (plane stress, 3-D reduced problem):
    Newton steps on sigma_zz(e_zz) with dsigma_zz/de_zz inside the bracket of known signs,
    bisection when a step leaves it, the bracket grown geometrically until it holds a sign
    change; starts from the elastic plane-stress strain. Non-converged points are marked and keep
    their input e_zz. ``c`` and ``creep_coef`` have one row for all points or one row per point."""
    n = eps.shape[0]
    out = np.empty(n)
    i_hat = 1.0 / math.sqrt(3.0)
    e = np.zeros(6)
    nvec = np.zeros(6)
    for i in range(n):
        row = i if c.shape[0] > 1 else 0
        ci = c[row]
        k_vol, k_dev, gamma, viscosity, n_exp, kappa = ci[0], ci[1], ci[2], ci[6], ci[7], ci[16]
        lame = (k_vol - k_dev) / 3.0
        shear = 0.5 * k_dev
        youngs = shear * (3.0 * lame + 2.0 * shear) / (lame + shear)
        s0 = max(ci[8], ci[9]) / youngs
        creep_rate = dt * (gamma / viscosity) ** n_exp if viscosity > 0.0 else 0.0
        for k in range(6):
            e[k] = eps[i, k] - eps_v_old[i, k]
        e[3] = -eps_v_old[i, 3]
        e[4] = -eps_v_old[i, 4]
        z = eps_v_old[i, 2] - lame / (lame + 2.0 * shear) * (e[0] + e[1])
        lo = -math.inf
        hi = math.inf
        scale = max(max(abs(e[0]), abs(e[1])), 1.0e-6 * s0)
        d = min(max(dam[i], kappa), 1.0)
        strength = d * max(ci[8], ci[9])
        done = False
        for it in range(ps_max_iter):
            e[2] = z - eps_v_old[i, 2]
            ev, eq = _point_invariants(e, 3, nvec)
            s_v, ds_v = _creep_only(eq, creep_coef[row], n_exp, axis, s_tab, ds_tab)
            a, da_v, da_q, b, db_v, db_q, _, _, _, _, _, _, ok = _point_state(
                ev, eq, d, s_v, ds_v, dt, creep_rate, ci, tol, max_iter)
            if not ok:
                break
            r = a * i_hat + b * nvec[2]
            ratio = b / eq if eq > 1e-30 else k_dev
            slope = (i_hat * (da_v * i_hat + da_q * nvec[2]) + nvec[2] * (db_v * i_hat + db_q * nvec[2])
                     + ratio * (2.0 / 3.0 - nvec[2] * nvec[2]))
            if abs(r) < ps_tol * max(strength, abs(a * i_hat + b * nvec[0]) + abs(a * i_hat + b * nvec[1])):
                done = True
                break
            if r < 0.0:
                lo = max(lo, z)
            elif r > 0.0:
                hi = min(hi, z)
            newton = z - r / slope if slope > 0.0 else math.nan
            if math.isfinite(newton) and lo < newton < hi:
                z = newton
            elif math.isfinite(lo) and math.isfinite(hi):
                z = 0.5 * (lo + hi)
            elif r > 0.0:
                z = z - scale * 2.0 ** it
            else:
                z = z + scale * 2.0 ** it
        if not done:
            failed[i] = True
            z = eps[i, 2]   # keep the input e_zz (the kernel's starting value) for the fallback
        out[i] = z
    return out
