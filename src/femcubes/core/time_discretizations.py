"""Shared time-discretization helpers used by solid models."""

from dolfinx import fem


def _coerce_to_function(field_like, reference_space):
    """Return a concrete function for direct vector access.

    Mixed-step field access may return symbolic UFL split expressions
    instead of DOLFINx functions. Materialize those expressions onto a
    concrete function space when direct ``.x.array`` access is needed.
    """
    if hasattr(field_like, "x") and hasattr(field_like.x, "array"):
        return field_like

    temp = fem.Function(reference_space)
    expr = fem.Expression(field_like, reference_space.element.interpolation_points)
    temp.interpolate(expr)
    temp.x.scatter_forward()
    return temp


def Newmark_Update(
    u, v, a,
    uOld, vOld, aOld,
    dt: float,
    beta: float,
    gamma: float,
    scatter_forward: bool = True,
    dofs=None,
):
    """Update velocity and acceleration fields with the Newmark scheme.

    Parameters
    ----------
    u, v, a, uOld, vOld, aOld
        Field-like objects exposing ``.x.array`` and, when
        ``scatter_forward`` is ``True``, ``.x.scatter_forward()``.
    dt
        Time-step size.
    beta, gamma
        Newmark parameters.
    scatter_forward
        Whether to scatter the updated ``v`` and ``a`` fields after
        writing the local arrays.
    dofs
        Optional local DOF indices to update; all DOFs when ``None``. Used
        by models restricted to a subdomain to update only the nodes they
        own (see :meth:`Mesh.Get_Field_Dofs`).

    Returns
    -------
    tuple[float, float]
        ``(dv_du, da_du)`` coefficients used for the consistent tangent.
    """
    needs_concrete_source = any(
        not (hasattr(field_like, "x") and hasattr(field_like.x, "array"))
        for field_like in (u, uOld, vOld, aOld)
    )
    if needs_concrete_source:
        reference_space = v.function_space
        u = _coerce_to_function(u, reference_space)
        uOld = _coerce_to_function(uOld, reference_space)
        vOld = _coerce_to_function(vOld, reference_space)
        aOld = _coerce_to_function(aOld, reference_space)

    idx = slice(None) if dofs is None else dofs
    u_, uOld_, vOld_, aOld_ = (field.x.array[idx] for field in (u, uOld, vOld, aOld))
    v.x.array[idx] = (
        gamma / beta / dt * (u_ - uOld_)
        + (1 - gamma / beta) * vOld_
        + dt * (1 - gamma / (2 * beta)) * aOld_
    )
    dv_du = gamma / (beta * dt)
    a.x.array[idx] = (
        1 / (beta * dt**2) * (u_ - uOld_)
        - 1 / (beta * dt) * vOld_
        - (1 / (2 * beta) - 1) * aOld_
    )
    da_du = 1 / (beta * dt**2)

    if scatter_forward:
        v.x.scatter_forward()
        a.x.scatter_forward()

    return dv_du, da_du
