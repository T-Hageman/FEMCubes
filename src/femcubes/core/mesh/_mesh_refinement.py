"""Public adaptive-refinement entry points for the composite mesh object."""

from __future__ import annotations

import numpy as np
from dolfinx import fem, la, mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from petsc4py import PETSc
from ufl import Measure, TestFunction, TrialFunction


class _MeshRefinement:
    """Thin mesh-facing API delegated to the refinement manager.

    Refinement algorithms and transaction contracts live in
    :mod:`femcubes.core.mesh.refinement`. This mix-in only keeps the public
    :class:`Mesh` surface consistent with the rest of the mesh wrapper, and
    provides the field registry the refinement pipeline transfers.
    """

    def Refine_From_Indicator(self, config=None):
        """Evaluate the configured field indicator and refine once.

        This is the high-level refinement operation used by transient
        solvers. It synchronizes registered fields, constructs detached
        topology/tag/field transfers, and installs them as one live mesh
        transaction. Empty collective mark sets are reported as no-ops.

        Parameters
        ----------
        config : RefinementConfig or mapping, optional
            Refinement controls. If omitted, ``params.MeshRefinement`` is
            used. Refinement must be enabled.

        Returns
        -------
        RefinementInstallResult
            Installed/no-op status, topology report, and epoch transition.

        Notes
        -----
        Fields restricted to subdomains or interfaces are refined with the
        mesh and recreated on the submeshes of the refined mesh; global
        fields keep their value. A restricted indicator field only marks
        cells of its own subdomain.

        This mesh-level operation does not own physics or solver caches. A
        caller outside :class:`TimeSolver` must invoke ``Physics.Post_Remesh``
        and discard solver workspaces after a successful installation.
        """

        from femcubes.core.mesh.refinement.manager import refine_from_indicator

        return refine_from_indicator(self, config)

    def Install_Refined_State(
        self,
        artifacts,
        transferred_state=None,
        config=None,
    ):
        """Install one complete detached refinement state.

        A no-op topology result leaves every live object and the remesh epoch
        unchanged. A successful result replaces the underlying DOLFINx mesh,
        transferred tags and fields while preserving this public wrapper.
        The detached state must come from ``snapshot_registered_fields`` of
        this mesh, which also covers restricted and global fields.

        Parameters
        ----------
        artifacts : RefinedMeshArtifacts
            Detached topology and transferred marker artifacts.
        transferred_state : TransferredFieldState, optional
            Complete detached field state. This is required when refinement
            occurred and may be omitted for a no-op result.
        config : RefinementConfig or mapping, optional
            Refinement controls. If omitted, ``params.MeshRefinement`` is
            used.

        Returns
        -------
        RefinementInstallResult
            Epoch, report, and visualization outcome for the transaction.
        """

        from femcubes.core.mesh.refinement.manager import install_refined_mesh_state

        return install_refined_mesh_state(
            self,
            artifacts,
            transferred_state,
            config,
        )

    # ------------------------------------------------------------------
    # Internal Helpers: Refinement Field Registry
    # ------------------------------------------------------------------
    #
    # The refinement pipeline transfers functions on the parent mesh. Fields
    # restricted to a subdomain or interface therefore take part as
    # parent-mesh proxies (zero outside their support) and are restricted
    # back onto the refined submeshes on installation. Global fields are not
    # tied to the mesh: they are left out of the transfer and kept as they are.

    def _refinement_field_registry(self, lift_fields: bool = True):
        """PRIVATE: Return ``(fields, definitions, step_fields)`` transferred by refinement.

        With ``lift_fields=False`` the fields entry is ``None`` (only the
        registry layout is needed, e.g. for validation).
        """
        definitions = {name: d for name, d in self.field_definitions.items() if not self.Is_Global_Field(name)}
        step_fields = {
            step: tuple(name for name in step_names if not self.Is_Global_Field(name))
            for step, step_names in self.step_fields.items()
        }
        if not lift_fields:
            return None, definitions, step_fields
        # Fields without a definition are kept, so the snapshot contract rejects them.
        fields = {
            name: field if self._field_support_keys.get(name) is None else self._lift_field_to_parent(name)
            for name, field in self.fields.items()
            if not self.Is_Global_Field(name)
        }
        return fields, definitions, step_fields

    def _finish_refined_field_install(self, live_fields: dict, live_definitions: dict) -> None:
        """PRIVATE: Complete installed transferred fields: restrict proxies and restore global fields.

        ``self.fields`` holds the transferred parent-mesh functions; the live
        registries from before the installation give the registration order
        and the global fields.
        """
        fields = {}
        for name, definition in live_definitions.items():
            support_key = self._field_support_keys.get(name)
            if self.Is_Global_Field(name):
                fields[name] = live_fields[name]
            elif support_key is None:
                fields[name] = self.fields[name]
            else:
                field = self._create_field_on_support(definition, support_key)
                self._copy_between_support_and_parent(name, field, self.fields[name], to_parent=False)
                fields[name] = field
        self.fields = fields
        self.field_definitions = dict(live_definitions)
        self.field_order = {name: definition.degree for name, definition in live_definitions.items()}
        self.field_type = {name: self._legacy_field_type(definition) for name, definition in live_definitions.items()}

    def _refinement_indicator_view(self, field_name: str):
        """PRIVATE: Return ``(parent-mesh field, support cell mask or None)`` for an indicator."""
        support = self.Get_Field_Support(field_name)
        if support.is_full:
            return self.fields[field_name], None
        if support.is_interface or support.is_global:
            raise ValueError(
                f"Refinement indicator field '{field_name}' does not live on cells; "
                "indicators must be fields on cells."
            )
        cell_map = self.mesh.topology.index_map(self.mesh.topology.dim)
        mask = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=bool)
        mask[support.parent_cells] = True
        return self._lift_field_to_parent(field_name), mask

    def _capture_submesh_state(self) -> dict:
        """PRIVATE: Return the submesh caches, for a rollback after a failed installation."""
        return {
            name: getattr(self, name)
            for name in ("_submeshes", "_submesh_entity_maps", "_submesh_vertex_maps", "_submesh_proxies", "_step_global_layouts")
        }

    def _restore_submesh_state(self, state: dict) -> None:
        """PRIVATE: Restore submesh caches captured by :meth:`_capture_submesh_state`."""
        for name, cache in state.items():
            setattr(self, name, cache)

    # ------------------------------------------------------------------
    # Internal Helpers: Lifting Restricted Fields
    # ------------------------------------------------------------------

    def _lift_field_to_parent(self, field_name: str) -> fem.Function:
        """PRIVATE: Return a parent-mesh copy of a restricted field, zero outside its support."""
        definition = self.field_definitions[field_name]
        if definition.family == "Quadrature":
            space = self._get_quadrature_space(definition.quadrature_degree, definition.value_shape)
            proxy = fem.Function(space, name=definition.description)
        else:
            proxy = self._create_function_from_definition(definition, self.mesh)
        self._copy_between_support_and_parent(field_name, self.fields[field_name], proxy, to_parent=True)
        return proxy

    def _copy_between_support_and_parent(self, field_name: str, field, proxy, to_parent: bool) -> None:
        """PRIVATE: Copy the values of a restricted field to or from its parent-mesh proxy."""
        support = self.Get_Field_Support(field_name)
        definition = self.field_definitions[field_name]
        if support.is_interface:
            if definition.family == "Quadrature":
                raise NotImplementedError(
                    f"Adaptive refinement does not support quadrature interface fields ('{field_name}')."
                )
            if to_parent:
                self._lift_interface_field(field_name, field, proxy)
            else:
                self._restrict_interface_field(field_name, field, proxy)
            return
        sub_cells = np.arange(support.parent_cells.size, dtype=np.int32)
        if definition.family == "Quadrature":
            # Quadrature values are stored per cell; submesh and parent cells match one to one.
            field_dofs = _unrolled_cell_dofs(field.function_space, sub_cells)
            proxy_dofs = _unrolled_cell_dofs(proxy.function_space, support.parent_cells)
            if to_parent:
                proxy.x.array[proxy_dofs] = field.x.array[field_dofs]
            else:
                field.x.array[field_dofs] = proxy.x.array[proxy_dofs]
        elif to_parent:
            proxy.interpolate(field, cells0=sub_cells, cells1=support.parent_cells)
        else:
            field.interpolate(proxy, cells0=support.parent_cells, cells1=sub_cells)
        (proxy if to_parent else field).x.scatter_forward()

    def _lift_interface_field(self, field_name: str, field, proxy) -> None:
        """PRIVATE: Set a parent-mesh proxy whose trace on the interface equals an interface field.

        In one adjacent cell per facet (see :meth:`_interface_lifting_cells`)
        the proxy interpolates ``field(pi(x))``, with ``pi`` the projection
        onto the facet: a polynomial of the field's degree whose trace on the
        facet is exactly the field, for CG and DG fields of any degree.
        """
        facet_mesh = field.function_space.mesh
        cells, facet_cells = self._interface_lifting_cells(field_name, include_ghosts=True)
        if cells.size:
            facet_dofmap = facet_mesh.geometry.dofmap[facet_cells]
            origins = facet_mesh.geometry.x[facet_dofmap[:, 0]]
            spans = facet_mesh.geometry.x[facet_dofmap[:, 1:]] - origins[:, None, :]
            # Orthogonal projector onto each facet's tangent space.
            gram_inverse = np.linalg.inv(np.einsum("fik,fjk->fij", spans, spans))
            projector = np.einsum("fki,fkl,flj->fij", spans, gram_inverse, spans)
        bs = field.function_space.dofmap.index_map_bs

        def trace_extension(x):
            points = x.T.reshape(cells.size, -1, 3)
            projected = origins[:, None, :] + np.einsum("fij,fpj->fpi", projector, points - origins[:, None, :])
            values = field.eval(projected.reshape(-1, 3), np.repeat(facet_cells, projected.shape[1]))
            return np.asarray(values).reshape(-1, bs).T

        proxy.x.array[:] = 0.0
        proxy.interpolate(trace_extension, cells0=cells)
        # Shared DOFs (CG) and DOFs owned by ranks without the lifting cell:
        # average all ranks' consistent values onto the owner.
        assigned = fem.Function(proxy.function_space)
        assigned.x.array[_unrolled_cell_dofs(proxy.function_space, cells)] = 1.0
        proxy.x.array[:] *= assigned.x.array
        proxy.x.scatter_reverse(la.InsertMode.add)
        assigned.x.scatter_reverse(la.InsertMode.add)
        owned = proxy.function_space.dofmap.index_map.size_local * bs
        counts = assigned.x.array[:owned]
        proxy.x.array[:owned] = np.divide(proxy.x.array[:owned], counts, out=np.zeros(owned), where=counts > 0)
        proxy.x.scatter_forward()

    def _restrict_interface_field(self, field_name: str, field, proxy) -> None:
        """PRIVATE: Set an interface field to the L2 projection of a proxy's one-sided trace.

        The trace of a lifted proxy lies in the interface field's space, so
        the projection reproduces it exactly.
        """
        space = field.function_space
        entities = self._interface_trace_entities(field_name)
        ds_trace = Measure("ds", domain=self.mesh, subdomain_data=[(1, entities.ravel())])(1)
        test = TestFunction(space)
        mass_form = fem.form(TrialFunction(space) * test * Measure("dx", domain=space.mesh))
        trace_form = self.Compile_Form(proxy * test * ds_trace) if not proxy.ufl_shape else self.Compile_Form(
            sum(proxy[i] * test[i] for i in range(proxy.ufl_shape[0])) * ds_trace)
        mass = assemble_matrix(mass_form)
        mass.assemble()
        rhs = assemble_vector(trace_form)
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        ksp = PETSc.KSP().create(mass.getComm())
        ksp.setOperators(mass)
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.getPC().setType(PETSc.PC.Type.JACOBI)
        ksp.setTolerances(rtol=1e-14, atol=1e-300, max_it=1000)
        ksp.solve(rhs, field.x.petsc_vec)
        field.x.scatter_forward()
        for petsc_object in (ksp, mass, rhs):
            petsc_object.destroy()

    def _interface_lifting_cells(self, field_name: str, include_ghosts: bool):
        """PRIVATE: Return ``(parent cells, interface mesh cells)`` pairing each facet with one adjacent cell.

        Boundary facets use their only cell. Interior facets use the cell in
        the interface's side domain (see :meth:`_interface_side_domain`),
        so the pairing is reproduced on the refined mesh.

        Raises
        ------
        ValueError
            If a cell would carry the trace of two facets of the interface.
        """
        facet_mesh, _facet_map, parent_facets = self._get_submesh(self._field_support_keys[field_name])
        cell_map = facet_mesh.topology.index_map(facet_mesh.topology.dim)
        facet_cells = np.arange(cell_map.size_local + (cell_map.num_ghosts if include_ghosts else 0), dtype=np.int32)
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_to_cell = self.mesh.topology.connectivity(tdim - 1, tdim)
        side_cells = self._interface_side_cells(field_name)
        exterior = set(dmesh.exterior_facet_indices(self.mesh.topology).tolist())

        cells = np.full(facet_cells.size, -1, dtype=np.int32)
        for i, facet in enumerate(parent_facets[facet_cells]):
            adjacent = facet_to_cell.links(facet)
            # A ghost interior facet may lack its side cell locally; its owner covers it.
            candidates = adjacent if facet in exterior else adjacent[side_cells[adjacent]]
            if candidates.size:
                cells[i] = candidates[0]
        keep = cells >= 0
        cells, facet_cells = cells[keep], facet_cells[keep]
        unique_cells, counts = np.unique(cells, return_counts=True)
        if self.mesh.comm.allreduce(int(np.any(counts > 1))):
            raise ValueError(
                f"Interface field '{field_name}': a cell touches the interface with two facets, so its trace "
                "cannot be carried through refinement. Refine the mesh at the interface's corners."
            )
        return cells, facet_cells

    def _interface_trace_entities(self, field_name: str) -> np.ndarray:
        """PRIVATE: Return ``(cell, local facet)`` pairs of the owned interface facets for one-sided ``ds``."""
        cells, facet_cells = self._interface_lifting_cells(field_name, include_ghosts=False)
        _facet_mesh, _facet_map, parent_facets = self._get_submesh(self._field_support_keys[field_name])
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, tdim - 1)
        cell_to_facet = self.mesh.topology.connectivity(tdim, tdim - 1)
        local_facets = [int(np.flatnonzero(cell_to_facet.links(cell) == facet)[0]) for cell, facet in zip(cells, parent_facets[facet_cells])]
        return np.column_stack([cells, np.asarray(local_facets, dtype=np.int32)]).astype(np.int32)

    def _interface_side_cells(self, field_name: str) -> np.ndarray:
        """PRIVATE: Return a local-cell mask of the side carrying an interface field's trace.

        Interior facets take the trace from the first registered leaf domain
        adjacent to the interface (the default ``normal_from`` side of
        :meth:`Interface_Integral`).
        """
        cell_map = self.mesh.topology.index_map(self.mesh.topology.dim)
        mask = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=bool)
        interface = self.Get_Field_Support(field_name).domain_names[0]
        facets = self.get_boundary_facets(interface)
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        self.mesh.topology.create_connectivity(tdim, tdim - 1)
        interior = np.setdiff1d(facets, dmesh.exterior_facet_indices(self.mesh.topology))
        entities = fem.compute_integration_domains(fem.IntegralType.interior_facet, self.mesh.topology, interior).reshape(-1, 4)
        if self.mesh.comm.allreduce(int(entities.shape[0])) == 0:
            return mask
        mask[self.get_domain_cells(self._first_adjacent_domain(entities))] = True
        return mask


def _unrolled(block_dofs, bs: int) -> np.ndarray:
    """PRIVATE: Expand block DOF indices into indices of ``Function.x.array``."""
    return (np.asarray(block_dofs, dtype=np.int64)[..., None] * bs + np.arange(bs)).ravel()


def _unrolled_cell_dofs(space, cells) -> np.ndarray:
    """PRIVATE: Return the unrolled DOFs of the given cells, cell by cell."""
    return _unrolled(space.dofmap.list[cells], space.dofmap.index_map_bs)
