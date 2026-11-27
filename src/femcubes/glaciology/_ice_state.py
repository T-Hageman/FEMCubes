"""PRIVATE: porosity and temperature fields of the ice models and their collective checks.

The ice models evaluate their properties from the fields ``poros`` and ``T``, which the user
sets. Invalid values (porosity outside [0, 1), temperatures above melting, a tensile strength
outside the bounds of the failure type) are detected here on all ranks together, so that every
rank raises the same error: an exception on one rank only would stall a parallel run.
"""

import math
import warnings

import numpy as np
from mpi4py import MPI

from femcubes.glaciology.ice_material import IceMaterial

#: Relative tolerance of the r2_2L_PSD bound f_t <= sqrt(2/3) f_s (as in the C++ kernel).
_PSD_BOUND_TOLERANCE = 1.0e-12


def add_ice_state_fields(params, mesh, domain: str) -> None:
    """PRIVATE: register ``poros`` (DG of order ``Field_Orders["Porosity"]``, else
    ``Field_Orders["pw"]``, else 1) and ``T`` (CG of order ``Field_Orders["Temperature"]``,
    else ``Field_Orders["T"]``, else 1) on ``domain``."""
    orders = params.Field_Orders
    mesh.Add_Field("poros", "Porosity [-]", "DG", orders.get("Porosity", orders.get("pw", 1)), domain=domain)
    mesh.Add_Field("T", "Temperature [C]", "CG", orders.get("Temperature", orders.get("T", 1)), domain=domain)


def admissible_tensile_strength(f_t, failure_type: str, f_s: float) -> np.ndarray:
    """PRIVATE: whether each tensile strength is finite, positive and within the bound of the
    failure type: f_t <= sqrt(2/3) f_s for "r2_2L_PSD", f_t < f_s for "r2_2L" and "r2_2L_DP"."""
    f_t = np.asarray(f_t, dtype=float)
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(f_t) & (f_t > 0.0)
        if failure_type == "r2_2L_PSD":
            ok &= (f_t / f_s) ** 2 <= (2.0 / 3.0) * (1.0 + _PSD_BOUND_TOLERANCE)
        elif failure_type in ("r2_2L", "r2_2L_DP"):
            ok &= f_t < f_s
    return ok


def check_ice_state(mesh, domain: str, label: str, *, positive=None, tensile_strength=None) -> bool:
    """PRIVATE: raise the same ``ValueError`` on all ranks if, on ``domain``, ``poros`` is not in
    [0, 1) or ``T`` exceeds the melting temperature (or either is not finite), if a value of
    ``positive`` (dict of name to point values) is not finite and positive, or if a value of
    ``tensile_strength`` = (f_t point values, FailureType, f_s) is not admissible
    (:func:`admissible_tensile_strength`). Collective: every rank must call it.

    Returns True when ``T`` is zero at every DOF (most likely never set)."""
    porosity = _domain_values(mesh, "poros", domain)
    temperature = _domain_values(mesh, "T", domain)
    p_low, p_high = IceMaterial.POROSITY_RANGE
    t_melt = IceMaterial.MELTING_TEMPERATURE
    positive = dict(positive or {})
    names = list(positive)
    with np.errstate(invalid="ignore"):
        flags = [
            np.any(~np.isfinite(porosity) | (porosity < p_low) | (porosity >= p_high)),
            np.any(~np.isfinite(temperature) | (temperature > t_melt)),
            np.any(temperature != 0.0),
        ]
        flags += [np.any(~(np.isfinite(values) & (np.asarray(values) > 0.0))) for values in positive.values()]
        if tensile_strength is not None:
            flags.append(np.any(~admissible_tensile_strength(*tensile_strength)))
    ranges = [_range(porosity), _range(temperature)] + [_range(v) for v in positive.values()]
    if tensile_strength is not None:
        ranges.append(_range(tensile_strength[0]))
    # one reduction: flags and ranges as maxima (minima negated)
    buffer = np.array([float(f) for f in flags] + [x for low, high in ranges for x in (-low, high)])
    comm = mesh.Get_Field("T").function_space.mesh.comm
    comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)
    n_flags = len(flags)
    bad = buffer[:n_flags] > 0.0
    ranges = [(-buffer[n_flags + 2 * i], buffer[n_flags + 2 * i + 1]) for i in range(len(ranges))]

    problems = []
    if bad[0]:
        problems.append(f"the porosity 'poros' must be finite with {p_low:g} <= p < {p_high:g} "
                        f"(found {ranges[0][0]:.4g} to {ranges[0][1]:.4g})")
    if bad[1]:
        problems.append(f"the temperature 'T' must be finite and at most {t_melt:g} C "
                        f"(found {ranges[1][0]:.4g} to {ranges[1][1]:.4g} C)")
    for i, name in enumerate(names):
        if bad[3 + i]:
            low, high = ranges[2 + i]
            problems.append(f"{name} must be finite and positive (found {low:.4g} to {high:.4g})")
    if tensile_strength is not None and bad[-1]:
        _, failure_type, f_s = tensile_strength
        low, high = ranges[-1]
        bound = {"r2_2L_PSD": f" and at most sqrt(2/3) f_s = {math.sqrt(2.0 / 3.0) * f_s:.4g} Pa",
                 "r2_2L": f" and below f_s = {f_s:.4g} Pa", "r2_2L_DP": f" and below f_s = {f_s:.4g} Pa"}
        problems.append(f"the tensile strength f_t must be finite, positive{bound.get(failure_type, '')} "
                        f"for FailureType {failure_type!r} (found {low:.4g} to {high:.4g} Pa)")
    if problems:
        raise ValueError(f"{label}: " + "; ".join(problems) + ". Set 'poros' and 'T' after "
                         "Physics.Initialize_Fields (Mesh.Constant_Field or Mesh.Set_From_Expression).")
    return not bad[2]


def warn_temperature_not_set(mesh, label: str) -> None:
    """PRIVATE: warn (on rank 0) that ``T`` is 0 C everywhere."""
    if mesh.Get_Field("T").function_space.mesh.comm.rank == 0:
        warnings.warn(f"{label}: the temperature 'T' is 0 C everywhere; it was probably never set. Set it after "
                      "Physics.Initialize_Fields with Mesh.Set_From_Expression('T', ...) or "
                      "Mesh.Constant_Field('T', ...).", stacklevel=3)


def _domain_values(mesh, name: str, domain: str) -> np.ndarray:
    """PRIVATE: the local DOF values of a field on ``domain``."""
    values = mesh.Get_Field(name).x.array
    dofs = mesh.Get_Field_Dofs(name, domain)
    return values if dofs is None else values[dofs]


def _range(values) -> tuple:
    """PRIVATE: (min, max) of the finite values, (inf, -inf) without any."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.inf, -math.inf
    return float(finite.min()), float(finite.max())
