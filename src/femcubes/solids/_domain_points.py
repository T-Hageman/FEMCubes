"""Quadrature-point state of solid models restricted to a (sub)domain."""

import numpy as np
from dolfinx import fem
import ufl
from ufl.algorithms import extract_coefficients


class _DomainPoints:
    """PRIVATE: Gather and scatter one model's quadrature-point values.

    Pointwise constitutive kernels update the quadrature fields of a model
    on its (sub)domain. A field may have a larger support than the model
    (e.g. ``sigma`` shared by solid models on neighbouring domains), so the
    points of the model's domain are indexed per field, in one common cell
    order, and the gathered arrays line up point by point.

    When every field lives exactly on the model's domain, the gathered
    arrays are the fields' own ``x.array`` views and scattering is a no-op.
    """

    def __init__(self, mesh, domain, field_names):
        self.mesh = mesh
        self.domain = domain
        self.field_names = tuple(field_names)
        self.reset()

    def reset(self) -> None:
        """PRIVATE: Forget the point indices, e.g. after remeshing."""
        self._indices = None

    def gather(self, *field_names) -> list:
        """PRIVATE: Return the model's point values of each field as flat arrays.

        These are the fields' own ``x.array`` views when every field lives
        exactly on the model's domain, and copies otherwise.
        """
        indices = self._get_indices()
        if indices is None:
            return [self.mesh.Get_Field(name).x.array for name in field_names]
        return [self.mesh.Get_Field(name).x.array[indices[name]] for name in field_names]

    def scatter(self, values: dict) -> None:
        """PRIVATE: Write gathered (or freshly computed) point values back to their fields.

        Ghost values are not updated; callers scatter forward where needed.
        """
        indices = self._get_indices()
        for name, value in values.items():
            array = self.mesh.Get_Field(name).x.array
            if indices is None:
                if value is not array:
                    array[:] = np.asarray(value).reshape(-1)
            else:
                array[indices[name]] = np.asarray(value).reshape(-1)

    def can_compile(self, field_name: str, expression) -> bool:
        """PRIVATE: Whether a compiled expression can be interpolated straight into the field.

        True when the model owns the whole field and all coefficients of the
        expression live on the field's own (sub)mesh.
        """
        mesh = self.mesh.Get_Field(field_name).function_space.mesh
        return self._get_indices() is None and all(
            getattr(coefficient, "function_space", None) is None or coefficient.function_space.mesh is mesh
            for coefficient in extract_coefficients(expression)
        )

    def interpolate(self, field_name: str, expression, compiled=None):
        """PRIVATE: Set a field from a value or UFL expression on the model's domain.

        ``compiled`` is the expression compiled by a previous call (or
        ``None``); the expression is assumed unchanged between calls. Returns
        the compiled expression to pass next time, which stays ``None`` when
        the expression is evaluated through ``Set_From_Expression`` instead
        (see :meth:`can_compile`). Ghost values are not updated.
        """
        if not isinstance(expression, ufl.core.expr.Expr):
            self.mesh.Constant_Field(field_name, expression, domain=self.domain)
            return compiled
        field = self.mesh.Get_Field(field_name)
        if compiled is None and not self.can_compile(field_name, expression):
            self.mesh.Set_From_Expression(field_name, expression, domain=self.domain)
            return None
        if compiled is None:
            compiled = fem.Expression(expression, field.function_space.element.interpolation_points)
        field.interpolate(compiled)
        return compiled

    def evaluate(self, field_name: str, expression) -> np.ndarray:
        """PRIVATE: Evaluate an expression at the model's points of a quadrature field.

        Returns a flat array in the layout of ``gather(field_name)``, without
        changing the field.
        """
        values = self.mesh.Evaluate_At_Field_Points(field_name, expression).reshape(-1)
        indices = self._get_indices()
        return values if indices is None else values[indices[field_name]]

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _get_indices(self):
        """PRIVATE: Return per-field local value indices of the domain's points, or None for all."""
        if self._indices is None:
            if all(self.mesh.Get_Field_Dofs(name, self.domain) is None for name in self.field_names):
                self._indices = False
            else:
                self._indices = self._domain_value_indices()
        return self._indices or None

    def _domain_value_indices(self) -> dict:
        """PRIVATE: Index every field's values on the domain's cells, in parent-cell order."""
        parent_cells = np.asarray(self.mesh.get_domain_cells(self.domain), dtype=np.int32)
        indices = {}
        for name in self.field_names:
            support = self.mesh.Get_Field_Support(name)
            cells = parent_cells
            if not support.is_full:
                cells = np.asarray(support.cell_map.sub_topology_to_topology(parent_cells, inverse=True), dtype=np.int32)
                if np.any(cells < 0):
                    raise ValueError(f"Field '{name}' does not exist on all of domain '{self.domain}'.")
            dofmap = self.mesh.Get_Field(name).function_space.dofmap
            blocks = dofmap.list[cells].reshape(-1).astype(np.int64)
            bs = dofmap.index_map_bs
            indices[name] = (blocks[:, None] * bs + np.arange(bs)).reshape(-1)
        return indices
