"""PRIVATE: principal-space integration-point update of the opening-only r2 criterion.

Used by ``Solid:Cohesive_ViscoElastic`` with ``FailureType`` "r2_2L_PSD"; the C++ kernel of
``Solid:Cohesive_ViscoElastic_Kernel`` has its own implementation of this criterion. The
failure potential is that of "r2_2L",

    F(eta) = d sqrt(A tr(eta)^2 + B |dev eta|^2),   A = (f_t_vol / 3)^2,   B = 2 f_s^2 / 3,

but the cohesive strain is restricted to positive semidefinite tensors (no principal cohesive
strain is negative: cracks open, they do not close or slide without opening) instead of
tr(eta) >= 0. The damaged material then carries every stress state without a tensile
principal stress like the intact material (a no-tension material at full damage).

The local problem, min over eta >= 0 of (e - eta) : C : (e - eta) / 2 + F(eta) with the trial
strain e = eps - eps_v_old, is isotropic and convex, so eta is coaxial with e and the unknowns
are its three principal values eta_i >= 0. With the support S of eta fixed, the elastic
stiffness (lame 1 1^T + 2 mu I) and the quadratic form of F ((A - B/3) 1 1^T + B I) restricted
to S share their eigenvectors (1_S and its complement), and stationarity reduces to one scalar
equation in the value of the potential, with a monotone Newton iteration. The seven supports
are enumerated; the feasible candidate (eta_S >= 0) of lowest energy is the minimizer. The
consistent tangent follows from the derivative of the principal stresses (implicit
differentiation on the support) and the rotation of the principal axes.

With J2 creep (creep strain increment dt gamma (gamma |s| / viscosity)^n s / |s| of the stress
deviator s, as in the kernel), the creep-only state comes from the reduced update; only where
the opening is active there, the principal creep increment v (coaxial with e) is solved by
Newton's method on v - dt g(|s|) s(e - v) = 0 with the opening solution as inner problem.

Three stress components (plane strain and 3-D). Plane stress solves sigma_zz = 0 for the
out-of-plane strain per point around this update and condenses the tangent.
"""

import dataclasses
import itertools
import math

import numpy as np

from femcubes.solids._reduced_cve import ReducedCVEMaterial, ReducedCVEUpdate, _condense_zz

# sets of opening principal directions, in order of size
SUPPORTS = tuple(np.array(mask, dtype=bool)
                 for mask in sorted(itertools.product((False, True), repeat=3), key=sum) if any(mask))
SQRT2 = math.sqrt(2.0)
# Mandel components (xx, yy, zz, sqrt2 yz, sqrt2 xz, sqrt2 xy): tensor index pairs of the shear entries
SHEAR_PAIRS = ((1, 2), (0, 2), (0, 1))


def principal_values(mandel):
    """Principal values (n, 3), largest first, of the symmetric tensors ``mandel`` (n, 6), in
    closed form (trigonometric solution; accurate to about 1e-8 of the deviator norm)."""
    t = np.asarray(mandel, dtype=float).reshape(-1, 6)
    mean = t[:, :3].sum(axis=1) / 3.0
    dx, dy, dz = t[:, 0] - mean, t[:, 1] - mean, t[:, 2] - mean
    yz, xz, xy = t[:, 3] / SQRT2, t[:, 4] / SQRT2, t[:, 5] / SQRT2
    j2 = 0.5 * (dx * dx + dy * dy + dz * dz) + yz * yz + xz * xz + xy * xy
    j3 = dx * (dy * dz - yz * yz) - xy * (xy * dz - yz * xz) + xz * (xy * yz - dy * xz)
    radius = np.sqrt(j2 / 3.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos3 = np.where(radius > 0.0, j3 / (2.0 * radius ** 3), 1.0)
    angle = np.arccos(np.clip(cos3, -1.0, 1.0)) / 3.0
    shift = np.array([0.0, -2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0])
    return mean[:, None] + 2.0 * radius[:, None] * np.cos(angle[:, None] + shift)


def strength_ratio(stress, damage, A, B, opening_only):
    """Stress relative to the strength surface of F = d sqrt(A tr(eta)^2 + B |dev eta|^2):
    below 1 inside the surface, 1 on it.

    For "r2_2L" (tr(eta) >= 0) this is the gauge sqrt(<tr s>_+^2 / (9 A) + |dev s|^2 / B) / d.
    With ``opening_only`` (eta positive semidefinite) the same expression is evaluated for the
    tensile part of the stress (negative principal stresses set to zero), an upper bound of the
    gauge that is exact when no principal stress is compressive.
    """
    values = principal_values(stress)
    if opening_only:
        values = np.maximum(values, 0.0)
    trace = values.sum(axis=1)
    dev2 = np.maximum((values * values).sum(axis=1) - trace * trace / 3.0, 0.0)
    trace = np.maximum(trace, 0.0)
    return np.sqrt(trace * trace / (9.0 * A) + dev2 / B) / np.asarray(damage, dtype=float).reshape(-1)


def opening_strength_scales(f_t, f_s):
    """(A, B) of F = d sqrt(A tr(eta)^2 + B |dev eta|^2): the "r2_2L" scales (uniaxial tensile
    strength f_t, deviatoric scale 2 f_s / sqrt(3)). ``f_t`` is a number or an array of per-point
    values."""
    if not np.all((0.0 < np.asarray(f_t)) & (np.asarray(f_t) < f_s)):
        raise ValueError("r2_2L_PSD needs 0 < f_t < f_s.")
    f_t_vol = f_t / np.sqrt(1.0 - (f_t / f_s) ** 2)
    return (f_t_vol / 3.0) ** 2, 2.0 * f_s * f_s / 3.0


def principal_state(e, d, lame, shear, A, B):
    """Principal-space solution for trial principal strains ``e`` (m, 3) and damage ``d`` (m);
    ``lame``, ``shear``, ``A`` and ``B`` are numbers or arrays (m) of per-point values.

    Returns (sigma, eta, D, F): principal stresses and cohesive strains (m, 3), the derivative
    of the principal stresses with respect to the principal strains (m, 3, 3) and the undamaged
    potential sqrt(A tr(eta)^2 + B |dev eta|^2) (m)."""
    m = e.shape[0]
    lame, shear, A, B = _per_point(m, lame, shear, A, B)
    tr = e.sum(axis=1)
    s = lame[:, None] * tr[:, None] + 2.0 * shear[:, None] * e    # trial principal stresses
    best = 0.5 * (s * e).sum(axis=1)                               # energy of eta = 0
    eta = np.zeros((m, 3))
    pot = np.zeros(m)
    support = np.zeros((m, 3), dtype=bool)
    for mask in SUPPORTS:
        k = int(mask.sum())
        c1, c2 = k * lame + 2.0 * shear, 2.0 * shear
        m1, m2 = k * (A - B / 3.0) + B, B
        s_S = s[:, mask]
        par = s_S.sum(axis=1) / math.sqrt(k)
        perp = s_S - par[:, None] / math.sqrt(k)
        a1, a2 = m1 * par * par, m2 * (perp * perp).sum(axis=1)
        idx = np.flatnonzero(a1 / (m1 * m1) + a2 / (m2 * m2) > d * d)   # dual norm on the support > d
        if idx.size == 0:
            continue
        a1, a2, dd = a1[idx], a2[idx], d[idx]
        c1, c2, m1, m2 = c1[idx], c2[idx], m1[idx], m2[idx]
        # root of g(F) = a1 / (c1 F + d m1)^2 + a2 / (c2 F + d m2)^2 - 1 (convex, decreasing):
        # Newton from a lower bound increases monotonically to the root
        F = np.maximum(np.maximum((np.sqrt(a1) - dd * m1) / c1, (np.sqrt(a2) - dd * m2) / c2), 0.0)
        for _ in range(100):
            u1, u2 = c1 * F + dd * m1, c2 * F + dd * m2
            g = a1 / (u1 * u1) + a2 / (u2 * u2) - 1.0
            step = np.maximum(g, 0.0) / (2.0 * (a1 * c1 / u1 ** 3 + a2 * c2 / u2 ** 3))
            F += step
            if np.all(step <= 1.0e-15 * F):
                break
        u1, u2 = c1 * F + dd * m1, c2 * F + dd * m2
        eta_S = (F * par[idx] / u1)[:, None] / math.sqrt(k) + (F / u2)[:, None] * perp[idx]
        cand = np.zeros((idx.size, 3))
        cand[:, mask] = eta_S
        el = e[idx] - cand
        energy = 0.5 * (lame[idx] * el.sum(axis=1) ** 2 + 2.0 * shear[idx] * (el * el).sum(axis=1)) + dd * F
        # An admissible stationary point on a support that contains the current one replaces it
        # without comparing energies: it minimises over a larger set, so its energy is not higher,
        # but where a direction just opens the two energies differ by less than round-off, and
        # keeping the smaller support there makes the stress jump (by the stiffness times that
        # direction's cohesive strain) between neighbouring strains.
        contains = ~(support[idx] & ~mask).any(axis=1)
        better = (eta_S >= 0.0).all(axis=1) & (F > 0.0) & (contains | (energy < best[idx]))
        sel = idx[better]
        best[sel] = energy[better]
        eta[sel] = cand[better]
        pot[sel] = F[better]
        support[sel] = mask
    sigma = s - (lame[:, None] * eta.sum(axis=1)[:, None] + 2.0 * shear[:, None] * eta)
    # dsigma/de = C - C X, (C + d Hess F)_SS X_S = C_S on the support
    C = lame[:, None, None] * np.ones((3, 3)) + 2.0 * shear[:, None, None] * np.eye(3)
    D = C.copy()
    # At the onset of opening (a kink of the stress) the system below is singular to working
    # precision; the elastic stiffness is a one-sided derivative there.
    act = np.flatnonzero(pot > 1.0e-10 * np.sqrt(np.minimum(A, B)) * np.abs(e).max(axis=1))
    if act.size:
        Aa, Ba = A[act, None], B[act, None]
        M_eta = (Aa - Ba / 3.0) * eta[act].sum(axis=1)[:, None] + Ba * eta[act]
        Fa = pot[act]
        hess = ((Aa[:, :, None] - Ba[:, :, None] / 3.0) * np.ones((3, 3)) + Ba[:, :, None] * np.eye(3)) \
            / Fa[:, None, None] - M_eta[:, :, None] * M_eta[:, None, :] / (Fa ** 3)[:, None, None]
        Ca = C[act]
        H = Ca + d[act, None, None] * hess
        on = support[act]
        pair = on[:, :, None] & on[:, None, :]
        H = np.where(pair, H, 0.0) + np.where(on, 0.0, 1.0)[:, :, None] * np.eye(3)[None]
        rhs = np.where(on[:, :, None], Ca, 0.0)
        D[act] = Ca - Ca @ np.linalg.solve(H, rhs)
    return sigma, eta, D, pot


def _per_point(m, *values):
    """PRIVATE: each value (number or array of m per-point values) as an array of m values."""
    return [np.broadcast_to(np.asarray(value, dtype=float), (m,)) for value in values]


def coupled_state(e, d, v0, creep, n_exp, lame, shear, A, B, start=None, tol=1.0e-12, max_iter=60):
    """Principal-space solution with creep: principal creep increments v (m, 3) with
    v = creep |s|^(n - 1) s, s the deviator of the principal stresses of ``principal_state(e - v)``,
    from the start ``v0``. ``creep`` = dt gamma^(n + 1) / viscosity^n. ``start`` is the result of
    ``principal_state(e - v0, ...)`` if the caller has it. ``creep``, ``lame``, ``shear``, ``A``
    and ``B`` are numbers or arrays (m) of per-point values.

    Returns (sigma, eta, D, F, v, converged): as ``principal_state``, with D the derivative of
    the principal stresses with respect to the trial principal strains e (creep included)."""
    m = e.shape[0]
    creep, lame, shear, A, B = _per_point(m, creep, lame, shear, A, B)
    eye = np.eye(3)
    p_dev = eye - 1.0 / 3.0
    scale = np.abs(e).max(axis=1)

    def evaluate(rows, v_rows, known=None):
        """[sigma, eta, D, F, s, g, residual] of the points ``rows`` at the creep increments ``v_rows``."""
        if known is None:
            known = principal_state(e[rows] - v_rows, d[rows], lame[rows], shear[rows], A[rows], B[rows])
        sigma, eta, D, pot = known
        s = sigma - sigma.mean(axis=1, keepdims=True)
        g = creep[rows] * np.linalg.norm(s, axis=1) ** (n_exp - 1.0)
        return [sigma, eta, D, pot, s, g, v_rows - g[:, None] * s]

    def creep_slope(s, g):
        """d(g s)/d sigma = g (P_dev + (n - 1) s s^T / |s|^2)."""
        r2 = (s * s).sum(axis=1)
        unit2 = s[:, :, None] * s[:, None, :] / np.where(r2 > 0.0, r2, 1.0)[:, None, None]
        return g[:, None, None] * (p_dev[None] + (n_exp - 1.0) * unit2)

    v = v0.copy()
    state = evaluate(np.arange(m), v, start)
    merit = (state[6] * state[6]).sum(axis=1)
    converged = np.sqrt(merit) <= tol * scale
    given_up = np.zeros(m, dtype=bool)
    # the creep increment of the stress at v0 (limited by the strength, unlike the creep-only v0) where it is closer
    rows = np.flatnonzero(~converged)
    if rows.size:
        v_trial = v[rows] - state[6][rows]
        trial = evaluate(rows, v_trial)
        merit_trial = (trial[6] * trial[6]).sum(axis=1)
        ok = merit_trial < merit[rows]
        v[rows[ok]] = v_trial[ok]
        merit[rows[ok]] = merit_trial[ok]
        for field, value in zip(state, trial):
            field[rows[ok]] = value[ok]
        converged = np.sqrt(merit) <= tol * scale
    for _ in range(max_iter):
        todo = np.flatnonzero(~converged & ~given_up)
        if todo.size == 0:
            break
        jacobian = eye[None] + creep_slope(state[4][todo], state[5][todo]) @ state[2][todo]
        step = np.linalg.solve(jacobian, -state[6][todo][:, :, None])[:, :, 0]
        alpha = np.ones(todo.size)
        pending = np.arange(todo.size)          # positions in todo without an accepted step
        for _ in range(30):
            rows = todo[pending]
            v_trial = v[rows] + alpha[pending, None] * step[pending]
            trial = evaluate(rows, v_trial)
            merit_trial = (trial[6] * trial[6]).sum(axis=1)
            ok = merit_trial <= (1.0 - 1.0e-4 * alpha[pending]) * merit[rows]
            accepted = rows[ok]
            v[accepted] = v_trial[ok]
            merit[accepted] = merit_trial[ok]
            for field, value in zip(state, trial):
                field[accepted] = value[ok]
            pending = pending[~ok]
            if pending.size == 0:
                break
            alpha[pending] *= 0.5
        converged[todo] = np.sqrt(merit[todo]) <= tol * scale[todo]
        if pending.size:
            # no descent at round-off: accepted only if already there
            rows = todo[pending]
            converged[rows] = np.sqrt(merit[rows]) <= 1.0e3 * tol * scale[rows]
            given_up[rows] = True
    sigma, eta, D, pot, s, g, _ = state
    D_total = D @ np.linalg.inv(eye[None] + creep_slope(s, g) @ D)
    return sigma, eta, D_total, pot, v, converged


class PrincipalCVEUpdate:
    """Integration-point update of "r2_2L_PSD", called like the C++ kernel (flat arrays, Mandel
    vectors of 6, tangents of 36). Writes ``stress``, ``eta``, ``strain_v``, ``tangent``,
    ``elastic_energy``, ``psi``, ``lambda_v`` and ``lambda_c`` (the potential, 0); returns 0
    (no fallback points). ``viscosity`` is the J2 stress scale (0: no creep), ``n_exp`` the
    creep exponent. ``lame``, ``shear``, ``f_t`` and ``viscosity`` are numbers, or arrays with
    one value per point of every call (spatially varying properties); ``f_s``, ``kappa`` and
    ``n_exp`` are the same for all points.

    A point at which a local iteration does not converge gets a non-finite stress and the elastic
    tangent, and no exception is raised: the residual of such a trial state is then non-finite
    on every rank, which the line search and the step control of the solver handle, whereas an
    exception on a single rank stalls a parallel run.

    ``plane_stress=True``: e_zz is solved from sigma_zz = 0 and written back into ``strain``; the
    tangent is condensed, with zero zz row and column, as in the kernel."""

    PLANE_STRESS_TOL = 1.0e-9       # |sigma_zz| relative to max(d f, |sigma_xx| + |sigma_yy|)
    PLANE_STRESS_ROUNDOFF = 1.0e-12  # ... plus this fraction of the elastic trial stress (its round-off)
    PLANE_STRESS_MAX_ITER = 60

    def __init__(self, *, lame, shear, f_t, f_s, kappa, viscosity=0.0, n_exp=1.0, plane_stress=False,
                 true_2d=False):
        if true_2d:
            raise NotImplementedError("r2_2L_PSD is not implemented for true_2d.")
        lame, shear, f_t, viscosity = (np.asarray(value, dtype=float) for value in (lame, shear, f_t, viscosity))
        if not np.all((shear > 0.0) & (3.0 * lame + 2.0 * shear > 0.0)):
            raise ValueError("r2_2L_PSD needs positive shear and bulk moduli.")
        if not 0.0 < kappa <= 1.0:
            raise ValueError("kappa must satisfy 0 < kappa <= 1.")
        if np.any(viscosity < 0.0) or (np.any(viscosity > 0.0) and n_exp < 1.0):
            raise ValueError("r2_2L_PSD needs viscosity >= 0 and a creep exponent n >= 1.")
        self.lame, self.shear, self.kappa = lame, shear, float(kappa)
        self.viscosity, self.n_exp = viscosity, float(n_exp)
        self.A, self.B = opening_strength_scales(f_t, f_s)
        self.plane_stress = bool(plane_stress)
        self.strength = np.maximum(f_t, float(f_s))
        self.creep_only = None
        if np.any(self.viscosity > 0.0):
            # creep-only state of every point: the reduced update with unreachable strengths
            self.creep_only = ReducedCVEUpdate(ReducedCVEMaterial(
                nd=3, lame=self.lame, shear=self.shear, bulk=self.lame + 2.0 * self.shear / 3.0,
                viscosity=self.viscosity, n_exp=self.n_exp, f_t=1.0e30, f_s=2.0e30, kappa=self.kappa,
                failure_type="r2_2L"))

    def __call__(self, strain, dt, damage, stress, eta, eta_old, strain_v, strain_v_old, tangent,
                 elastic_energy, psi, lambda_v, lambda_c, lambda_c_old):
        update = self._update_plane_stress if self.plane_stress else self._update_3d
        return update(strain, dt, damage, stress, eta, eta_old, strain_v, strain_v_old, tangent,
                      elastic_energy, psi, lambda_v, lambda_c, lambda_c_old)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _properties(self, rows, n):
        """PRIVATE: (lame, shear, A, B, viscosity, strength) of the ``n`` points ``rows`` of the
        call (all points for ``None``), as arrays of n values."""
        values = (self.lame, self.shear, self.A, self.B, self.viscosity, self.strength)
        return _per_point(n, *(v if np.ndim(v) == 0 or rows is None else v[rows] for v in values))

    def _creep_only_update(self, rows):
        """PRIVATE: the creep-only update of the points ``rows`` of the call (all for ``None``)."""
        material = self.creep_only.material
        if rows is None or np.ndim(material.viscosity) == 0:
            return self.creep_only
        return ReducedCVEUpdate(dataclasses.replace(
            material, lame=material.lame[rows], shear=material.shear[rows], bulk=material.bulk[rows],
            viscosity=material.viscosity[rows]))

    def _update_plane_stress(self, strain, dt, damage, stress, eta, eta_old, strain_v, strain_v_old, tangent,
                             elastic_energy, psi, lambda_v, lambda_c, lambda_c_old):
        """PRIVATE: the update with e_zz solved from sigma_zz = 0, iterating on e_zz around the
        3-D update: Newton steps with dsigma_zz/de_zz inside the bracket of known signs,
        bisection when a step leaves it, the bracket grown geometrically until it holds a sign
        change. Starts from the elastic plane-stress strain, the solution of every point
        without creep and opening. (The reduced plane-stress update is not used for the
        creep-only state: its tolerance scales with the strengths, unreachable here.)

        The tolerance has a floor at the round-off of the stress: at an opened, almost broken
        point with large strains the stress is the small difference of large strains, and its
        round-off exceeds any tolerance relative to the degraded strength. Points that do not
        converge get a non-finite stress (see ``_update_3d``)."""
        n = damage.size
        eps, old, dam = strain.reshape(n, 6), strain_v_old.reshape(n, 6), damage.reshape(n)
        lame, mu, _, _, _, strength = self._properties(None, n)
        d = np.clip(dam, self.kappa, 1.0)
        widths = (("stress", 6), ("eta", 6), ("strain_v", 6), ("tangent", 36), ("energy", 1), ("psi", 1),
                  ("lambda_v", 1), ("lambda_c", 2))
        state = {name: np.zeros((n, width)) for name, width in widths}
        z = old[:, 2] - lame / (lame + 2.0 * mu) * (eps[:, 0] - old[:, 0] + eps[:, 1] - old[:, 1])
        lo, hi = np.full(n, -np.inf), np.full(n, np.inf)
        scale = np.maximum(np.abs(eps[:, :2] - old[:, :2]).max(axis=1), 1.0e-6 * strength / mu)
        rows = np.arange(n)
        for it in range(self.PLANE_STRESS_MAX_ITER):
            k = rows.size
            trial = eps[rows]
            trial[:, 2] = z[rows]
            out = {name: np.zeros(k * width) for name, width in widths}
            self._update_3d(trial.ravel(), dt, dam[rows], out["stress"], out["eta"], np.zeros(k * 6),
                            out["strain_v"], old[rows].ravel(), out["tangent"], out["energy"], out["psi"],
                            out["lambda_v"], out["lambda_c"], np.zeros(k * 2), rows=rows)
            for name, width in widths:
                state[name][rows] = out[name].reshape(k, width)
            s = state["stress"][rows]
            r, slope = s[:, 2], state["tangent"][rows, 14]
            tol = self.PLANE_STRESS_TOL * np.maximum(d[rows] * strength[rows], np.abs(s[:, 0]) + np.abs(s[:, 1])) \
                + self.PLANE_STRESS_ROUNDOFF * (lame[rows] + 2.0 * mu[rows]) * np.abs(trial - old[rows]).max(axis=1)
            open_ = np.abs(r) >= tol            # False where the 3-D update failed (non-finite stress)
            if not open_.any():
                rows = rows[:0]
                break
            rows, r, slope = rows[open_], r[open_], slope[open_]
            lo[rows] = np.where(r < 0.0, np.maximum(lo[rows], z[rows]), lo[rows])
            hi[rows] = np.where(r > 0.0, np.minimum(hi[rows], z[rows]), hi[rows])
            with np.errstate(divide="ignore", invalid="ignore"):
                newton = np.where(slope > 0.0, z[rows] - r / slope, np.nan)
            inside = np.isfinite(newton) & (lo[rows] < newton) & (newton < hi[rows])
            bracketed = np.isfinite(lo[rows]) & np.isfinite(hi[rows])
            with np.errstate(invalid="ignore"):
                middle = 0.5 * (lo[rows] + hi[rows])
            grown = z[rows] - np.sign(r) * scale[rows] * 2.0 ** it
            z[rows] = np.where(inside, newton, np.where(bracketed, middle, grown))
        state["stress"][rows] = np.nan          # not converged
        failed = np.flatnonzero(~np.isfinite(state["stress"]).all(axis=1))
        state["tangent"][failed] = _elastic_mandel(lame[failed], mu[failed]).reshape(-1, 36)
        eps[:, 2] = z
        _condense_zz(state["tangent"])
        stress.reshape(n, 6)[:], tangent.reshape(n, 36)[:] = state["stress"], state["tangent"]
        eta.reshape(n, 6)[:] = state["eta"]
        strain_v.reshape(n, 6)[:] = state["strain_v"]
        elastic_energy.reshape(n)[:] = state["energy"][:, 0]
        psi.reshape(n)[:] = state["psi"][:, 0]
        lambda_v.reshape(n)[:] = state["lambda_v"][:, 0]
        lambda_c.reshape(n, 2)[:] = state["lambda_c"]
        return 0

    def _update_3d(self, strain, dt, damage, stress, eta, eta_old, strain_v, strain_v_old, tangent,
                   elastic_energy, psi, lambda_v, lambda_c, lambda_c_old, rows=None):
        """PRIVATE: the update for a given strain with three normal components; ``rows`` are the
        points of the call these arrays hold (all for ``None``)."""
        n = damage.size
        lame, mu, A, B, viscosity, _ = self._properties(rows, n)
        gamma = math.sqrt(1.5)
        e = strain.reshape(n, 6) - strain_v_old.reshape(n, 6)
        d = np.clip(damage.reshape(n), self.kappa, 1.0)
        sig, eta_out, tan = stress.reshape(n, 6), eta.reshape(n, 6), tangent.reshape(n, 36)
        if self.creep_only is not None:
            self._creep_only_update(rows)(strain, dt, damage, stress, eta, eta_old, strain_v, strain_v_old, tangent,
                                          elastic_energy, psi, lambda_v, lambda_c, lambda_c_old)
        else:
            sig[:] = 2.0 * mu[:, None] * e
            sig[:, :3] += lame[:, None] * e[:, :3].sum(axis=1)[:, None]
            tan[:] = _elastic_mandel(lame, mu).reshape(n, 36)
            strain_v.reshape(n, 6)[:] = strain_v_old.reshape(n, 6)
            elastic_energy.reshape(n)[:] = 0.5 * (sig * e).sum(axis=1)
            lambda_v.reshape(n)[:] = 0.0
        eta_out[:] = 0.0
        psi.reshape(n)[:] = 0.0
        lambda_c.reshape(n, 2)[:] = 0.0
        # opening needs a principal stress above d sqrt(A) at the creep-only state; bound from the invariants
        mean = sig[:, :3].sum(axis=1) / 3.0
        dev = sig.copy()
        dev[:, :3] -= mean[:, None]
        with np.errstate(invalid="ignore"):
            idx = np.flatnonzero(np.isfinite(sig).all(axis=1)       # a non-finite strain keeps its non-finite stress
                                 & (mean + math.sqrt(2.0 / 3.0) * np.linalg.norm(dev, axis=1) > d * np.sqrt(A)))
        if idx.size == 0:
            return 0
        ei = e[idx]
        T = np.empty((idx.size, 3, 3))
        for a in range(3):
            T[:, a, a] = ei[:, a]
        for comp, (a, b) in zip((3, 4, 5), SHEAR_PAIRS):
            T[:, a, b] = T[:, b, a] = ei[:, comp] / SQRT2
        w, Q = np.linalg.eigh(T)
        # Mandel vectors of n_i n_i
        P = np.empty((idx.size, 3, 6))
        for i in range(3):
            P[:, i, :3] = Q[:, :, i] ** 2
            for comp, (a, b) in zip((3, 4, 5), SHEAR_PAIRS):
                P[:, i, comp] = SQRT2 * Q[:, a, i] * Q[:, b, i]
        # creep-only principal creep increments (coaxial with e)
        v = np.einsum("pik,pk->pi", P, strain_v.reshape(n, 6)[idx] - strain_v_old.reshape(n, 6)[idx])
        sigma, eta_p, D, pot = principal_state(w - v, d[idx], lame[idx], mu[idx], A[idx], B[idx])
        act = pot > 0.0
        if not act.any():
            return 0
        idx, w, Q, P, v = idx[act], w[act], Q[act], P[act], v[act]
        sigma, eta_p, D, pot = sigma[act], eta_p[act], D[act], pot[act]
        if self.creep_only is not None:
            visc = viscosity[idx]
            creep = np.where(visc > 0.0, float(dt) * gamma ** (self.n_exp + 1.0)
                             / np.where(visc > 0.0, visc, 1.0) ** self.n_exp, 0.0)
            sigma, eta_p, D, pot, v, converged = coupled_state(w, d[idx], v, creep, self.n_exp, lame[idx], mu[idx],
                                                               A[idx], B[idx], start=(sigma, eta_p, D, pot))
            # not converged: non-finite stress (below), which the line search and the step control
            # of the solver treat on all ranks together; an exception on one rank would not be
            sigma[~converged] = np.nan
            strain_v.reshape(n, 6)[idx] = strain_v_old.reshape(n, 6)[idx] + np.einsum("pi,pik->pk", v, P)
            lambda_v.reshape(n)[idx] = np.linalg.norm(v, axis=1) / gamma
        else:
            converged = np.ones(idx.size, dtype=bool)
        sig[idx] = np.einsum("pi,pik->pk", sigma, P)
        eta_out[idx] = np.einsum("pi,pik->pk", eta_p, P)
        t = np.einsum("pij,pik,pjl->pkl", D, P, P)
        scale = np.abs(w).max(axis=1)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            # unit shears (n_i n_j + n_j n_i) / sqrt(2)
            G = np.empty((idx.size, 6))
            G[:, :3] = SQRT2 * Q[:, :, i] * Q[:, :, j]
            for comp, (a, b) in zip((3, 4, 5), SHEAR_PAIRS):
                G[:, comp] = Q[:, a, i] * Q[:, b, j] + Q[:, b, i] * Q[:, a, j]
            gap = w[:, i] - w[:, j]
            close = np.abs(gap) <= 1.0e-7 * scale
            theta = np.where(close, D[:, i, i] - D[:, i, j],
                             (sigma[:, i] - sigma[:, j]) / np.where(close, 1.0, gap))
            t += theta[:, None, None] * G[:, :, None] * G[:, None, :]
        tan[idx] = t.reshape(idx.size, 36)
        failed = idx[~converged]
        tan[failed] = _elastic_mandel(lame[failed], mu[failed]).reshape(-1, 36)
        el = w - v - eta_p
        elastic_energy.reshape(n)[idx] = 0.5 * (lame[idx] * el.sum(axis=1) ** 2 + 2.0 * mu[idx] * (el * el).sum(axis=1))
        psi.reshape(n)[idx] = pot
        lambda_c.reshape(n, 2)[idx, 0] = pot
        return 0


def _elastic_mandel(lame, shear):
    """PRIVATE: isotropic elastic stiffnesses (m, 6, 6) in Mandel form for per-point ``lame`` and ``shear`` (m)."""
    C = np.zeros((lame.size, 6, 6))
    C[:, :3, :3] = lame[:, None, None]
    C[:, np.arange(6), np.arange(6)] += 2.0 * shear[:, None]
    return C
