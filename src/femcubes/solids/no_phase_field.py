"""Strongly constrain the phase-field to zero in a rectangular region."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from dolfinx import fem, mesh as dmesh

from femcubes.core.base_model import BaseModel
from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh


class No_PhaseField(BaseModel):
    r"""Strong constraint :math:`\phi = 0` inside a rectangular region.

    Keeps the phase field at zero where cracks must not form. Near supports
    and loading points, the boundary conditions can concentrate stresses
    such that cracks would nucleate there rather than in the region of
    interest; constraining the phase field prevents this without changing the
    boundary conditions themselves. Supports 2-D and 3-D; in 3-D the
    rectangle is extruded through the full :math:`z` range.

    .. rubric:: Fields

    * ``phasefield``: read and constrained; provided by :doc:`phase_field`,
      scalar CG (DG raises an error). The model defines no fields.

    .. rubric:: Weak form

    None.

    .. rubric:: Outside the weak form

    * Dirichlet condition on ``phasefield`` at every solve: all coefficients
      (vertex, edge, face and cell DOFs) of mesh entities whose vertices lie in
      :math:`x_\min\le x\le x_\max`, :math:`y_\min\le y\le y_\max` (and, with
      ``domain``, in that domain group) are set such that the solution update
      returns them to :math:`\phi = 0`. Because FEMCubes solves for
      increments, the prescribed increment is the current value of
      :math:`\phi` at these DOFs, which also clears damage that was present
      before the model became active.
    * No state variables or outputs.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            Models = {
                "NoPF": {
                    "Type": "Solid:No_PhaseField",  # REQUIRED
                    "x_range": [0.0, 1.0],          # REQUIRED: [x_min, x_max], finite, strictly increasing (m)
                    "y_range": [2.0, 3.0],          # REQUIRED: [y_min, y_max], finite, strictly increasing (m)
                    "domain": "domain",             # OPTIONAL (default "domain", the full mesh): domain group;
                                                    # the phase field must exist on all of it
                },
            }
    """

    MODEL_TYPE = "Solid:No_PhaseField"

    def __init__(self, name: str, params: ParamsTemplate, mesh: Mesh):
        super().__init__(name, params, mesh)

        if getattr(params, "dim", 2) < 2:
            raise ValueError("Solid:No_PhaseField requires a 2-D or 3-D mesh.")

        model_params = params.Models[name]
        self.x_range = self._validate_range(model_params.get("x_range"), "x_range")
        self.y_range = self._validate_range(model_params.get("y_range"), "y_range")

    def Get_Weak_Form(self, time_constant, dt_constant, step: int):
        """Return no UFL form; the model only contributes strong constraints."""
        return None

    def get_bcs_for_field(self, field_name: str):
        """Return incremental zero Dirichlet conditions for ``phasefield``."""
        if field_name != "phasefield":
            return None

        if self.mesh.Is_DG_Field(field_name):
            raise ValueError(
                "Solid:No_PhaseField requires the 'phasefield' field to use a "
                "continuous CG function space."
            )

        try:
            resolved = self.mesh.Get_Field_For_Output(field_name)
            function_space, sub_index = self.mesh.Get_Field_Space_For_BC(field_name)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Solid:No_PhaseField requires a registered 'phasefield' field."
            ) from exc

        if resolved.comp_count != 1:
            raise ValueError("Solid:No_PhaseField requires a scalar 'phasefield' field.")

        domain_dofs = self._domain_dofs(field_name)
        if sub_index is None:
            current_field = resolved.field
            value_space = function_space
            bc_dofs = self._region_dofs(value_space)
            if domain_dofs is not None:
                bc_dofs = np.intersect1d(bc_dofs, domain_dofs).astype(np.int32)
            bc = fem.Function(value_space)
            bc.x.array[:] = current_field.x.array[:]
            bc.x.scatter_forward()
            return [fem.dirichletbc(bc, bc_dofs)]

        if domain_dofs is not None:
            raise ValueError("Solid:No_PhaseField can only restrict a phase field with its own function space to a domain.")
        subspace = function_space.sub(sub_index)
        value_space, _ = subspace.collapse()
        current_field = resolved.collapse_component(0)
        bc_dofs = self._region_dofs((subspace, value_space))
        bc = fem.Function(value_space)
        bc.x.array[:] = current_field.x.array[:]
        bc.x.scatter_forward()
        return [fem.dirichletbc(bc, bc_dofs, subspace)]

    def _domain_dofs(self, field_name: str):
        """PRIVATE: Return the field's local DOFs in the model's domain, or None for all."""
        support_tags = set(self.mesh.Get_Field_Support(field_name).tags)
        if self.domain != "domain" and not set(self.mesh.get_domain_tags(self.domain)) <= support_tags:
            raise ValueError(
                f"Solid:No_PhaseField domain '{self.domain}' is not covered by the '{field_name}' field."
            )
        return self.mesh.Get_Field_Dofs(field_name, self.domain)

    def _region_dofs(self, function_space) -> np.ndarray | list[np.ndarray]:
        """PRIVATE: Locate constrained entity DOFs without evaluating DOF coordinates."""
        # The field may live on a submesh (domain-restricted support).
        domain = (function_space[0] if isinstance(function_space, tuple) else function_space).mesh
        topology = domain.topology
        dofs = []
        for entity_dim in range(topology.dim + 1):
            topology.create_entities(entity_dim)
            topology.create_connectivity(entity_dim, topology.dim)
            entities = dmesh.locate_entities(domain, entity_dim, self._inside_region)
            dofs.append(fem.locate_dofs_topological(function_space, entity_dim, entities))
        if isinstance(function_space, tuple):
            pairs = np.unique(np.concatenate([np.column_stack(d) for d in dofs]), axis=0)
            return [np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1])]
        return np.unique(np.concatenate(dofs, axis=0), axis=0)

    def _inside_region(self, coordinates: np.ndarray) -> np.ndarray:
        """PRIVATE: Mark mesh coordinates inside the configured rectangle."""
        return (
            (coordinates[0] >= self.x_range[0])
            & (coordinates[0] <= self.x_range[1])
            & (coordinates[1] >= self.y_range[0])
            & (coordinates[1] <= self.y_range[1])
        )

    @staticmethod
    def _validate_range(values: Sequence[float] | None, name: str) -> tuple[float, float]:
        """PRIVATE: Validate and normalize one coordinate range."""
        if values is None:
            raise ValueError(f"Solid:No_PhaseField requires '{name}'.")
        if isinstance(values, (str, bytes)):
            raise ValueError(f"Solid:No_PhaseField '{name}' must contain two numbers.")

        try:
            normalized = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Solid:No_PhaseField '{name}' must contain two numbers.") from exc

        if len(normalized) != 2:
            raise ValueError(f"Solid:No_PhaseField '{name}' must contain exactly two values.")
        if not all(np.isfinite(value) for value in normalized):
            raise ValueError(f"Solid:No_PhaseField '{name}' values must be finite.")
        if normalized[0] >= normalized[1]:
            raise ValueError(
                f"Solid:No_PhaseField '{name}' must be strictly increasing."
            )
        return normalized
