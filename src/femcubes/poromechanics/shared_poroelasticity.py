r"""Shared saturation relations of the poromechanics models.

Helper functions, not a model: they are used by :doc:`pore_pressure` and
:doc:`fracture_flow` and add nothing to a weak form by themselves.

The effective saturation follows the retention curve of
:cite:t:`vanGenuchten1980`, with the constraint
:math:`n_\mathrm{vG} = 1/(1-m)` that gives a closed-form relative
permeability with the model of :cite:t:`Mualem1976`,

.. math::

    S_w = \bigl(1 + (p_c/p_\mathrm{ref})^{n_\mathrm{vG}}\bigr)^{-m},

with capillary pressure :math:`p_c = -p_w` (air pressure zero), the
reference pressure :math:`p_\mathrm{ref} > 0` (``Saturation_p_ref``) and
:math:`0 < m < 1` (``Saturation_m``). For :math:`p_w \geq 0` the medium is
fully saturated, :math:`S_w = 1` with zero derivatives.
"""

from ufl import conditional, lt


def validate_saturation_parameters(p_ref: float, m: float) -> None:
    """Validate Mualem-van Genuchten retention parameters."""
    if p_ref <= 0.0:
        raise ValueError("Saturation_p_ref must be strictly positive.")
    if not 0.0 < m < 1.0:
        raise ValueError("Saturation_m must lie strictly between 0 and 1.")


def saturation_relations(capillary_pressure, p_ref: float, m: float):
    r"""Return effective saturation and its first two pressure derivatives.

    The derivatives are with respect to capillary pressure, using the
    Mualem constraint :math:`n=1/(1-m)`.
    """
    n_vg = 1.0 / (1.0 - m)
    xi = (capillary_pressure / p_ref) ** n_vg
    
    saturation = (1.0 + xi) ** (-m)
    d_saturation = -m * n_vg/p_ref * (capillary_pressure / p_ref) ** (n_vg - 1.0) * (1.0 + xi) ** (-m - 1.0)
    dd_saturation = m * n_vg/p_ref**2 * (capillary_pressure / p_ref) ** (n_vg - 2.0) * (1.0 + xi) ** (-m - 1.0) * ((m + 1.0) * n_vg * xi / (1.0 + xi) - (n_vg - 1.0))
    return saturation, d_saturation, dd_saturation


def saturation_from_pressure(pore_pressure, p_ref: float, m: float):
    r"""Return ``Sw``, ``dSw/dpw``, and ``d²Sw/dpw²`` as UFL expressions.

    Air pressure is taken as zero, so capillary pressure is ``-pw``. Positive
    pore pressure is capped at full saturation with zero pressure derivatives.
    """
    capillary_pressure = -pore_pressure
    capillary_pressure_limited = conditional(lt(capillary_pressure, 0.0),0.0,capillary_pressure)
    saturation, d_saturation, dd_saturation = saturation_relations(capillary_pressure_limited,p_ref,m,)
    d_saturation = conditional(lt(capillary_pressure, 0.0),0.0,d_saturation)
    dd_saturation = conditional(lt(capillary_pressure, 0.0),0.0,dd_saturation)
    
    return saturation, -d_saturation, dd_saturation


def single_saturation_from_pressure(pore_pressure: float, p_ref: float, m: float):
    """Scalar counterpart of :func:`saturation_from_pressure`."""
    if pore_pressure > 0.0:
        return 1.0, 0.0, 0.0

    saturation, d_saturation, dd_saturation = saturation_relations(-pore_pressure,p_ref,m)
    return saturation, -d_saturation, dd_saturation
