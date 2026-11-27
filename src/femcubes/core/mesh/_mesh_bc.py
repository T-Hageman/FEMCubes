"""Dirichlet boundary-condition helpers for the composite mesh object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from dolfinx import fem, mesh as dmesh
import ufl
from ._mesh_base import WeightedMeasure


@dataclass
class _IncrementDirichletBCWorkspace:
    """PRIVATE: Mesh-epoch data reused by one incremental Dirichlet BC."""

    bc: Any
    value_function: Any
    source_function: Any
    source_dofs: np.ndarray | None


class _MeshBC:
    """Mix-in for mesh-side Dirichlet BC construction.

    Boundary-condition models should ideally describe only the boundary
    semantics they impose: the target field, the named boundary, and the
    desired target value. This mix-in owns the mesh-dependent DOLFINx
    plumbing required to turn that description into concrete
    :class:`dolfinx.fem.DirichletBC` objects.

    Public API
    ----------
    ``Build_Increment_Dirichlet_BCs(...)``
        Build one or more incremental Dirichlet boundary conditions for
        a named field and named boundary.
    """

    def __init__(self, params: Any) -> None:
        self._boundary_integrator = None
        self._boundary_integrator_cache_key = None
        self._increment_bc_cache: dict[
            tuple[int, str, str, int | None],
            _IncrementDirichletBCWorkspace,
        ] = {}
        super().__init__(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Build_Increment_Dirichlet_BCs(
        self,
        field_name: str,
        boundary_name: str,
        target_value,
        component: int = None,
    ) -> list:
        """Build incremental Dirichlet BCs for one field on one boundary.

        The returned BCs constrain the *update* rather than the absolute
        field value. In other words, the prescribed function is built as
        ``target - current`` so that the constrained field stays at the
        requested value after the linear solve updates the state.

        Parameters
        ----------
        field_name : str
            Name of the registered field to constrain.
        boundary_name : str
            Human-readable boundary name such as ``"left"`` or ``"top"``.
        target_value : float or sequence
            Desired target value. For full vector constraints, this may
            be a scalar applied to every component or a sequence with one
            value per component.
        component : int, optional
            Specific vector component to constrain. If omitted, the full
            scalar field is constrained, or all components are
            constrained for mixed vector fields.

        Returns
        -------
        list
            One or more DOLFINx Dirichlet BC objects.
        """
        # DG fields enforce inflow weakly via flux terms; strong Dirichlet BCs
        # are not applicable to discontinuous function spaces.
        if self.Is_DG_Field(field_name):
            return []

        resolved = self.Get_Field_For_Output(field_name)

        if component is not None:
            return [
                self._build_increment_dirichlet_bc(
                    field_name, boundary_name, target_value, component
                )
            ]

        if resolved.comp_count > 1:
            targets = self._normalize_bc_targets(target_value, resolved.comp_count)
            return [
                self._build_increment_dirichlet_bc(
                    field_name, boundary_name, target, comp_idx
                )
                for comp_idx, target in enumerate(targets)
            ]

        return [
            self._build_increment_dirichlet_bc(
                field_name, boundary_name, target_value, None
            )
        ]

    def get_Boundary_Integrator(self):
        """Return the current exterior-facet integration measure.

        The helper constructs ``ufl.Measure("ds", ...)`` against the
        current mesh instance and facet tags. If the same mesh and facet
        tags are requested repeatedly, the previously constructed measure
        is returned. If either object changes, the cached measure is
        discarded and rebuilt automatically.

        Only tagged facets on the exterior boundary are included: tagged
        interior facets (such as interfaces between subdomains) integrate to
        zero under this measure, rather than being integrated from one
        adjacent cell.
        """
        cache_key = (
            getattr(self, "remesh_epoch", 0),
            id(self.mesh),
            id(self.facet_tags),
        )
        if self._boundary_integrator_cache_key != cache_key:
            self._boundary_integrator = ufl.Measure(
                "ds",
                domain=self.mesh,
                subdomain_data=self._exterior_facet_tags(),
            )
            self._boundary_integrator_cache_key = cache_key

        if self.axisymmetric:
            return WeightedMeasure(self._boundary_integrator, self.Get_Integration_Weight())
        return self._boundary_integrator

    def Get_Field_Boundary_Integrator(self, field_name: str, boundary_name: str):
        """Return ``ds`` over a named boundary for terms acting on one field.

        Boundary models (fluxes, tractions) should integrate with this
        measure: it checks that the whole boundary lies on the field's
        support, as for Dirichlet conditions, so terms on fields restricted
        to a subdomain never act outside it. Forms using it must be compiled
        with :meth:`Compile_Form`.

        Raises
        ------
        ValueError
            If part of the boundary lies outside the field's support.
        """
        error = None
        try:
            self._facets_on_field_mesh(field_name, self.get_boundary_facets(boundary_name))
        except ValueError as exc:
            error = f"Boundary '{boundary_name}' cannot act on field '{field_name}': {exc}"
        self._raise_collectively(error)
        return self.get_Boundary_Integrator()(self.get_boundary_tags(boundary_name))

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _raise_collectively(self, error) -> None:
        """PRIVATE: Raise ``ValueError`` on every rank if any rank reports an error message."""
        errors = [e for e in self.mesh.comm.allgather(error) if e is not None]
        if errors:
            raise ValueError(errors[0])

    def _exterior_facet_tags(self):
        """PRIVATE: Return :attr:`facet_tags` restricted to exterior facets.

        DOLFINx exterior-facet integrals do not filter their tagged facets,
        so interior facets must be removed before building ``ds``.
        """
        if self.facet_tags is None:
            return None
        facet_dim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(facet_dim, facet_dim + 1)
        exterior = np.isin(self.facet_tags.indices, dmesh.exterior_facet_indices(self.mesh.topology))
        return dmesh.meshtags(
            self.mesh,
            facet_dim,
            self.facet_tags.indices[exterior],
            self.facet_tags.values[exterior],
        )

    def _invalidate_boundary_cache(self) -> None:
        """PRIVATE: Clear mesh-bound integration and Dirichlet-BC data."""

        self._boundary_integrator = None
        self._boundary_integrator_cache_key = None
        self._increment_bc_cache.clear()

    def _build_increment_dirichlet_bc(
        self,
        field_name: str,
        boundary_name: str,
        target_value,
        component: int = None,
    ):
        """PRIVATE: Build or update one cached incremental Dirichlet BC."""
        cache_key = (
            int(getattr(self, "remesh_epoch", 0)),
            field_name,
            boundary_name,
            component,
        )
        workspace = self._increment_bc_cache.get(cache_key)
        if workspace is None:
            workspace = self._create_increment_dirichlet_bc_workspace(
                field_name,
                boundary_name,
                component,
            )
            self._increment_bc_cache[cache_key] = workspace

        self._update_increment_dirichlet_bc_value(workspace, target_value)
        return workspace.bc

    def _create_increment_dirichlet_bc_workspace(
        self,
        field_name: str,
        boundary_name: str,
        component: int = None,
    ) -> _IncrementDirichletBCWorkspace:
        """PRIVATE: Create the mesh-dependent objects for one BC."""
        # Constrained entities are located on the field's own mesh, which is a
        # submesh for fields restricted to a subdomain or interface.
        # Only ranks holding offending facets detect a boundary outside the
        # support; raise everywhere so no rank enters later collectives.
        entities, error = None, None
        try:
            entities = self._bc_entities_on_field_mesh(field_name, boundary_name)
        except ValueError as exc:
            error = str(exc)
        self._raise_collectively(error)
        facet_dim, boundary_facets = entities
        function_space, sub_index = self.Get_Field_Space_For_BC(field_name, component)
        source_function = self.Get_Field_For_Output(field_name).field

        if sub_index is not None:
            subspace = function_space.sub(sub_index)
            V_sub, source_dofs = subspace.collapse()
            bc_dofs = fem.locate_dofs_topological(
                (subspace, V_sub), facet_dim, boundary_facets
            )
            bc_func = fem.Function(V_sub)
            return _IncrementDirichletBCWorkspace(
                bc=fem.dirichletbc(bc_func, bc_dofs, subspace),
                value_function=bc_func,
                source_function=source_function,
                source_dofs=np.asarray(source_dofs),
            )

        bc_dofs = fem.locate_dofs_topological(function_space, facet_dim, boundary_facets)
        bc_func = fem.Function(function_space)
        return _IncrementDirichletBCWorkspace(
            bc=fem.dirichletbc(bc_func, bc_dofs),
            value_function=bc_func,
            source_function=source_function,
            source_dofs=None,
        )

    def _update_increment_dirichlet_bc_value(
        self,
        workspace: _IncrementDirichletBCWorkspace,
        target_value,
    ) -> None:
        """PRIVATE: Refresh only the value function of a cached BC."""
        if workspace.source_dofs is not None:
            workspace.value_function.x.array[:] = (
                workspace.source_function.x.array[workspace.source_dofs]
                - target_value
            )
            return

        self._populate_increment_bc_function(
            workspace.value_function,
            workspace.source_function,
            target_value,
        )

    def _populate_increment_bc_function(self, bc_func, current_field, target_value) -> None:
        """PRIVATE: Fill one full-space BC function with target-current values."""
        value_shape = current_field.function_space.element.value_shape
        if len(value_shape) == 0:
            bc_func.x.array[:] = current_field.x.array - target_value
            return

        ncomp = value_shape[0]
        targets = self._normalize_bc_targets(target_value, ncomp)
        for comp_idx, target in enumerate(targets):
            bc_func.sub(comp_idx).x.array[:] = current_field.sub(comp_idx).x.array - target

    def _normalize_bc_targets(self, target_value, comp_count: int) -> list:
        """PRIVATE: Expand a scalar or validate a component-wise BC target."""
        if isinstance(target_value, np.ndarray):
            target_value = target_value.tolist()

        if isinstance(target_value, (list, tuple)):
            if len(target_value) != comp_count:
                raise ValueError(
                    f"Expected {comp_count} component targets but received "
                    f"{len(target_value)} for the requested boundary condition."
                )
            return list(target_value)

        return [target_value] * comp_count
