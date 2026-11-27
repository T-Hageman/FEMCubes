"""Lumped-operator helpers for the composite mesh class."""

from __future__ import annotations

from typing import Any

import numpy as np
from dolfinx import fem, mesh as dmesh
from dolfinx.cpp.mesh import cell_num_entities
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc
from basix import LagrangeVariant
from basix.ufl import element
from scipy.spatial import cKDTree
from ufl import Measure, TestFunction, dx


class _MeshLumpedOperators:
    """Mix-in providing lumped mass-matrix operators.

    Used by explicit or semi-implicit time integrators that require
    row-summed (lumped) mass weights rather than a consistent mass
    matrix.

    Public methods are part of the mesh wrapper API. Underscore-prefixed
    helpers are internal implementation details.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __init__(self, params: Any) -> None:
        self._lumped_weights_cache: dict[
            tuple[int, int, tuple[str, ...]], dict[str, Any]
        ] = {}
        self._lumped_weights_cache_epoch: int | None = getattr(self, "remesh_epoch", 0)
        super().__init__(params)

    def Get_Weights_Dofs(self, field_names: list[str], step: int):
        """Return lumped weights, local DOFs, and local values for one step.

        The first field in ``field_names`` defines the reference lumped
        weights. Scalar fields receive one weight per owned scalar DOF;
        vector fields receive one weight per owned component DOF. All
        requested fields are assumed to share a compatible flattened DOF
        layout when used together in lumped updates.

        Parameters
        ----------
        field_names : list[str]
            Fields whose owned DOFs and owned values should be returned.
            The first field also defines the reference lumped weights.
        step : int
            Solution step whose vector layout the returned DOF indices use.

        Returns
        -------
        tuple
            ``(weights, dofs, vals)`` where ``weights`` is a one-dimensional
            NumPy array of lumped weights, ``dofs`` is a mapping from field
            name to owned local scalar/component DOF indices in the step's
            residual/matrix layout (offset by the field's block for
            monolithic steps), and ``vals`` is a mapping from field name to
            owned local values.
        """
        if not field_names:
            raise ValueError("At least one field name is required to compute lumped weights.")

        self._ensure_lumped_cache_current()
        field_names_key = tuple(field_names)
        cache_key = (getattr(self, "remesh_epoch", 0), step, field_names_key)
        cached = self._lumped_weights_cache.get(cache_key)
        if cached is None:
            cached = self._build_lumped_weights_cache_entry(field_names_key, step)
            self._lumped_weights_cache[cache_key] = cached

        vals = {
            name: self._get_local_field_values(
                name,
                cached["field_dofs"][name],
                cached["local_sizes"][name],
            )
            for name in field_names_key
        }

        return cached["weights"], cached["dofs"], vals

    def Get_Interface_Weights_Dofs(self, interface: str, field_names: list[str], step: int):
        """Return lumped weights, DOFs, and values for nodal interface terms.

        Interface reactions couple fields living on different sides of an
        interface (or on the interface itself) node by node. Each rank
        handles the nodes of the interface facets it owns: entry ``k``
        describes node ``k`` of one such facet, with weight
        ``weights[k] = int_facet N_k dGamma``. Summed over all facets, these
        are the lumped interface weights of each node, so assembling
        ``weights[k] * rate(node values)`` reproduces lumped integration.

        Each field is located where it exists: on either side of the
        interface or on the interface itself. A field existing on both sides
        is continuous, so it has one DOF per node. All fields must be scalar
        CG fields of one polynomial degree.

        Parameters
        ----------
        interface : str
            Facet group of the interface, e.g. ``"interface"``.
        field_names : list[str]
            Fields whose DOFs and values are required.
        step : int
            Solution step whose vector layout the returned DOF indices use.

        Returns
        -------
        tuple
            ``(weights, dofs, vals)``: facet-node weights; per field the
            local indices into the step's residual vector (owned or ghost;
            ``None`` for fields not solved in ``step``); and per field the
            current values at the facet nodes. A node shared by neighbouring
            facets appears once per facet, so accumulate with
            ``numpy.add.at``. Convert DOF indices with
            :meth:`get_step_global_indices` for matrix insertion.
        """
        self._ensure_lumped_cache_current()
        cache_key = ("interface", getattr(self, "remesh_epoch", 0), interface, tuple(field_names), step)
        cached = self._lumped_weights_cache.get(cache_key)
        if cached is None:
            cached = self._build_interface_weights_cache_entry(interface, tuple(field_names), step)
            self._lumped_weights_cache[cache_key] = cached

        vals = {name: self.fields[name].x.array[field_dofs] for name, field_dofs in cached["field_dofs"].items()}
        return cached["weights"], cached["dofs"], vals

    def Set_Interface_Values(self, interface: str, field_name: str, values) -> None:
        """Set a field's values at the nodes of an interface, in the order of :meth:`Get_Interface_Weights_Dofs`.

        ``values`` has one entry per facet node (a node shared by
        neighbouring facets appears once per facet and must receive the same
        value), e.g. computed from the values returned by
        :meth:`Get_Interface_Weights_Dofs` for the same interface. Ghost
        values are refreshed afterwards.
        """
        self._ensure_lumped_cache_current()
        cache_key = ("interface", getattr(self, "remesh_epoch", 0), interface, (field_name,), None)
        cached = self._lumped_weights_cache.get(cache_key)
        if cached is None:
            cached = self._build_interface_weights_cache_entry(interface, (field_name,), None)
            self._lumped_weights_cache[cache_key] = cached
        field = self.fields[field_name]
        field.x.array[cached["field_dofs"][field_name]] = values
        field.x.scatter_forward()

    def Get_Old_For_Lumped(self, old_field_names: list[str], step: int):
        """Return owned local values for the fields used in lumped updates.

        The ``step`` argument is kept for API compatibility with the
        lumped-assembly call sites.

        Parameters
        ----------
        old_field_names : list[str]
            Field names whose owned local values should be extracted.
        step : int
            Unused compatibility argument matching the lumped-update call
            signature used elsewhere in the code base.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from field name to owned local values.
        """
        _ = step

        vals_old = {}
        for name in old_field_names:
            _, field_vals, _ = self._get_local_field_data(name)
            vals_old[name] = field_vals

        return vals_old

    def Get_Residual_Row_Scale(self, step: int) -> np.ndarray | None:
        """Return the factors that make the owned residual rows of a step planar-equivalent, or None.

        On axisymmetric meshes every residual row carries the radius weight,
        so rows near the axis are orders of magnitude smaller than rows far
        from it, and one absolute tolerance cannot suit both. Row ``i`` is
        then scaled by ``1 / r_i``, with ``r_i = int r N_i / int N_i`` the
        mean radius of its basis function (positive, also on the axis); rows
        of global fields use the mean radius of the domain. The solvers
        measure convergence on the scaled residual; the updates are
        unchanged. Returns None on plane meshes.
        """
        if not self.axisymmetric:
            return None
        self._ensure_lumped_cache_current()
        cache_key = (getattr(self, "remesh_epoch", 0), step, ("<residual row scale>",))
        cached = self._lumped_weights_cache.get(cache_key)
        if cached is None:
            cached = self._build_residual_row_scale(step)
            self._lumped_weights_cache[cache_key] = cached
        return cached

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _invalidate_lumped_cache(self) -> None:
        """PRIVATE: Clear mesh-layout-dependent lumped weights and DOF maps."""

        self._lumped_weights_cache.clear()
        self._lumped_weights_cache_epoch = None

    def _ensure_lumped_cache_current(self) -> None:
        """PRIVATE: Lazily discard lumped data retained from another epoch."""

        current_epoch = getattr(self, "remesh_epoch", 0)
        if self._lumped_weights_cache_epoch == current_epoch:
            return
        self._lumped_weights_cache.clear()
        self._lumped_weights_cache_epoch = current_epoch

    def _build_lumped_weights_cache_entry(
        self,
        field_names: tuple[str, ...],
        step: int,
    ) -> dict[str, Any]:
        """PRIVATE: Assemble and cache static lumped weights and field DOFs."""
        test_function = TestFunction(self.fields[field_names[0]].function_space)
        weight = self.Get_Integration_Weight(test_function.ufl_function_space().mesh)
        weight_vector = assemble_vector(fem.form(self._lumped_weight_integrand(test_function) * weight * dx))
        try:
            weight_vector.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            weight_vector.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,
                mode=PETSc.ScatterMode.FORWARD,
            )

            dofs = {}
            field_dofs_by_name = {}
            local_sizes = {}
            reference_dofs = None
            local_size = 0
            step_field_names = list(self.step_fields.get(step, []))
            step_offsets = self.get_step_block_offsets(step)

            for name in field_names:
                field_output = self.Get_Field_For_Output(name)
                field_dofs, field_local_size = self._get_field_output_local_dofs(
                    field_output
                )
                # Solution fields are addressed within their block of the
                # step vector; other fields only provide values.
                block_offset = step_offsets[step_field_names.index(name)] if name in step_field_names else 0
                dofs[name] = field_dofs + block_offset
                field_dofs_by_name[name] = field_dofs
                local_sizes[name] = field_local_size
                if name == field_names[0]:
                    reference_dofs = field_dofs
                    local_size = field_local_size

            with weight_vector.localForm() as weight_local:
                weights_local = weight_local.array[:local_size].copy()
            weights = weights_local[reference_dofs]
        finally:
            weight_vector.destroy()

        return {
            "weights": weights,
            "dofs": dofs,
            "field_dofs": field_dofs_by_name,
            "local_sizes": local_sizes,
        }

    def _build_interface_weights_cache_entry(self, interface: str, field_names: tuple[str, ...], step: int) -> dict[str, Any]:
        """PRIVATE: Match fields to the nodes of the locally owned interface facets."""
        degrees = {self.field_definitions[name].degree for name in field_names}
        if len(degrees) != 1 or any(
            self.field_definitions[name].family != "CG" or self.field_definitions[name].value_shape for name in field_names
        ):
            raise ValueError(f"Interface terms require scalar CG fields of one degree; got {list(field_names)}.")
        degree = degrees.pop()
        if degree > 2:
            raise ValueError("Interface terms currently support field degrees up to 2.")

        support_key = self._resolve_support_key([interface])
        facet_mesh, _facet_map, parent_facets = self._get_submesh(support_key)
        num_owned_facets = facet_mesh.topology.index_map(facet_mesh.topology.dim).size_local

        # The fields use a Bernstein basis, whose functions all integrate to
        # the facet size divided by the number of facet nodes (straight facets).
        V_nodes = fem.functionspace(
            facet_mesh,
            element("Lagrange", facet_mesh.basix_cell(), degree, lagrange_variant=LagrangeVariant.bernstein),
        )
        nodes_per_facet = V_nodes.dofmap.list.shape[1]
        size_vector = assemble_vector(fem.form(TestFunction(fem.functionspace(facet_mesh, ("DG", 0))) * Measure("dx", domain=facet_mesh)))
        with size_vector.localForm() as size_local:
            facet_sizes = size_local.array[:num_owned_facets].copy()
        size_vector.destroy()
        weights = np.repeat(facet_sizes / nodes_per_facet, nodes_per_facet)
        node_dofs = V_nodes.dofmap.list[:num_owned_facets].ravel()
        node_coordinates = self._entity_dof_points(V_nodes)[node_dofs]
        if self.axisymmetric:
            # int_facet r B_k = |facet| / (p + 1) * (r_k + 2 (r_mid - r_k) / (p + 2)) for the degree-p Bernstein
            # function of the node at (entity midpoint) radius r_k on a straight facet with midpoint radius r_mid.
            facet_mid = dmesh.compute_midpoints(facet_mesh, facet_mesh.topology.dim, np.arange(num_owned_facets, dtype=np.int32))
            r_mid = np.repeat(facet_mid[:, 0], nodes_per_facet)
            r_node = node_coordinates[:, 0]
            weights = weights * (r_node + 2.0 * (r_mid - r_node) / (degree + 2))

        tdim = self.mesh.topology.dim
        step_fields = list(self.step_fields.get(step, []))
        dofs, field_dofs = {}, {}
        for name in field_names:
            space = self.fields[name].function_space
            support = self.Get_Field_Support(name)
            if support.is_interface:
                candidates = np.arange(space.dofmap.index_map.size_local + space.dofmap.index_map.num_ghosts, dtype=np.int32)
            else:
                # Owned interface facets always have their adjacent cells locally.
                facets = self._facets_on_field_mesh(name, parent_facets[:num_owned_facets])
                candidates = fem.locate_dofs_topological(space, tdim - 1, facets)
            coordinates = self._entity_dof_points(space)[candidates]
            distances, matches = cKDTree(coordinates).query(node_coordinates)
            if node_coordinates.size and np.max(distances) > 1e-8 * max(np.ptp(self.mesh.geometry.x, axis=0).max(), 1e-300):
                raise ValueError(f"Field '{name}' has no DOF at some nodes of interface '{interface}'.")
            field_dofs[name] = candidates[matches]
            dofs[name] = self._step_vector_indices(step, name, field_dofs[name]) if name in step_fields else None
        return {"weights": weights, "dofs": dofs, "field_dofs": field_dofs}

    def _build_residual_row_scale(self, step: int) -> np.ndarray:
        """PRIVATE: Assemble ``int N_i / int r N_i`` per owned row of a step (see ``Get_Residual_Row_Scale``)."""
        offsets = self.get_step_block_offsets(step)
        scale = np.ones(offsets[-1])
        domain_radius = None
        for name, start, end in zip(self.step_fields.get(step, []), offsets, offsets[1:]):
            if self.Is_Global_Field(name):
                if domain_radius is None:
                    one = fem.Constant(self.mesh, 1.0)
                    volume, moment = (self.mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * weight * dx(domain=self.mesh))))
                                      for weight in (1.0, self.Get_Integration_Weight()))
                    domain_radius = moment / volume
                scale[start:end] = 1.0 / domain_radius
                continue
            space = self.fields[name].function_space
            integrand = self._lumped_weight_integrand(TestFunction(space))
            measure = Measure("dx", domain=space.mesh)
            sums = []
            for weight in (1.0, self.Get_Integration_Weight(space.mesh)):
                vector = assemble_vector(fem.form(integrand * weight * measure))
                vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
                with vector.localForm() as local:
                    sums.append(local.array[:end - start].copy())
                vector.destroy()
            scale[start:end] = sums[0] / sums[1]
        return scale

    @staticmethod
    def _entity_dof_points(space) -> np.ndarray:
        """PRIVATE: Return, per local DOF, the midpoint of the mesh entity carrying it.

        For elements with at most one DOF per mesh entity (degree <= 2), this
        identifies DOFs geometrically also for non-interpolatory bases such
        as Bernstein, which have no DOF coordinates.
        """
        mesh = space.mesh
        tdim = mesh.topology.dim
        dofmap = space.dofmap
        layout = dofmap.dof_layout
        points = np.zeros((dofmap.index_map.size_local + dofmap.index_map.num_ghosts, 3))
        cell_dofs = dofmap.list
        num_cells = cell_dofs.shape[0]
        for dim in range(tdim + 1):
            if dim == tdim:
                cell_entities = np.arange(num_cells, dtype=np.int32)[:, None]
            else:
                mesh.topology.create_connectivity(tdim, dim)
                mesh.topology.create_connectivity(dim, tdim)
                cell_entities = mesh.topology.connectivity(tdim, dim).array.reshape(num_cells, cell_num_entities(mesh.topology.cell_type, dim))
            for local_entity in range(cell_entities.shape[1]):
                entity_dofs = layout.entity_dofs(dim, local_entity)
                if not entity_dofs:
                    continue
                if len(entity_dofs) > 1:
                    raise ValueError("Elements with several DOFs per mesh entity are not supported.")
                entities = cell_entities[:, local_entity]
                points[cell_dofs[:, entity_dofs[0]]] = dmesh.compute_midpoints(mesh, dim, entities)
        return points

    def _lumped_weight_integrand(self, test_function):
        """PRIVATE: Return the scalar integrand whose assembly gives row-sum weights."""
        shape = test_function.ufl_shape
        if len(shape) == 0:
            return test_function
        if len(shape) == 1:
            return sum(test_function[i] for i in range(shape[0]))

        raise ValueError("Lumped weights are only supported for scalar and vector fields.") # pragma: no cover

    def _get_local_owned_size(self, function_space) -> int:
        """PRIVATE: Return the number of owned scalar entries in one function space."""
        index_map = function_space.dofmap.index_map
        return index_map.size_local * function_space.dofmap.index_map_bs

    def _get_field_output_local_dofs(self, field_output_view) -> tuple[np.ndarray, int]:
        """PRIVATE: Return owned local DOF indices of a field and its owned size."""
        local_size = self._get_local_owned_size(field_output_view.field.function_space)
        return np.arange(local_size, dtype=np.int32), local_size

    def _get_local_field_data(self, field_name: str) -> tuple[np.ndarray, np.ndarray, int]:
        """PRIVATE: Return owned local DOF indices, values, and local size for one field."""
        field_output = self.Get_Field_For_Output(field_name)
        field_dofs, local_size = self._get_field_output_local_dofs(field_output)
        field_vals = self._get_local_field_values(field_name, field_dofs, local_size)

        return field_dofs, field_vals, local_size

    def _get_local_field_values(
        self,
        field_name: str,
        field_dofs: np.ndarray,
        local_size: int,
    ) -> np.ndarray:
        """PRIVATE: Return current owned local values for cached local DOFs."""
        field_output = self.Get_Field_For_Output(field_name)
        return field_output.field.x.array[:local_size][field_dofs]
