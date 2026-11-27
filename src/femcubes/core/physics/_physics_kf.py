from contextlib import nullcontext
from typing import Any
from femcubes.core.utils.mpi_utils import mprint, comm
from dolfinx import fem
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, create_matrix, set_bc
import numpy as np
from petsc4py import PETSc
import time
import ufl
from ufl.classes import (
    Argument,
    Coefficient,
    Constant,
    ConstantValue,
    ExprList,
    ExprMapping,
    GeometricQuantity,
    Index,
    Label,
    MultiIndex,
)
from ufl.corealg.traversal import unique_post_traversal
from ufl.domain import extract_unique_domain

class _PhysicsKF:
    """Internal mix-in for global K/F assembly and BC collection.

    Collects UFL weak-form contributions from every model, compiles and
    assembles them into PETSc matrices/vectors, and applies Dirichlet BCs.

    Steps with a single field assemble plain forms. Steps with several
    fields are assembled as block systems with one block per field (see
    :meth:`_MeshDOFs.get_step_block_offsets`) into one monolithic PETSc
    matrix and vector, so solvers see the same interface for both.

    Public API
    ----------
    ``Get_Fvec(step)``
        Assemble only the global residual vector for a staggered solve step.
    ``Get_Kmat_Fvec(step)``
        Assemble the global stiffness matrix and residual vector for a
        staggered solve step.
    ``Update_Global_Measures()``
        Forward global-measure updates to models that provide them.
    ``get_fields_for_step(step)``
        Return the named fields solved in a staggered step.
    ``get_boundary_conditions(step=None)``
        Return active Dirichlet boundary conditions for one step or for
        all registered solution fields.

    Attributes
    ----------
    K_mat : PETSc.Mat or None
        Assembled global tangent (stiffness) matrix.
    F_vec : PETSc.Vec or None
        Assembled global residual (force) vector.
    """

    def __init__(self, mesh: Any, params: Any) -> None:
        # Store the currently active PETSc objects for convenience and
        # keep a small per-step cache for reuse when the layout matches.
        self.K_mat = None
        self.F_vec = None
        self._assembly_cache = {}
        self._compiled_form_cache = {}
        self._model_weak_form_cache = {}
        self._step_weak_form_cache = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Get_Fvec(self, step: int) -> PETSc.Vec:
        """Assemble and return the global residual vector only.

        The bilinear form is compiled when boundary lifting requires it, but
        no stiffness matrix is assembled.  Existing vector storage is reused
        when its layout remains compatible.
        """
        profiler = getattr(self, "_runtime_profiler", None)
        if profiler is not None:
            profiler.count("residual_only_assembly")
        with self._measure_runtime("assembly_bookkeeping"):
            return self._get_fvec(step)

    def Get_Kmat_Fvec(self, step: int) -> tuple[PETSc.Mat, PETSc.Vec]:
        """Assemble and return the global stiffness matrix and residual."""
        profiler = getattr(self, "_runtime_profiler", None)
        if profiler is not None:
            profiler.count("matrix_residual_assembly")
        with self._measure_runtime("assembly_bookkeeping"):
            return self._get_kmat_fvec(step)

    def Update_Global_Measures(self) -> None:
        """Update global measures for models that provide them."""
        for model in self.models.values():
            if hasattr(model, "Update_Global_Measures"):
                model.Update_Global_Measures()

    def get_fields_for_step(self, step):
        """Return the named fields solved during a staggered step."""
        return self._get_fields_for_step_internal(step)
    
    def get_boundary_conditions(self, step: int = None):
        """Return active boundary conditions for one step or all steps."""
        fields_in_step = self._get_boundary_condition_fields(step)
        bcs = []
        for model in self.models.values():
            if hasattr(model, 'get_bcs_for_field'):
                for field_name in fields_in_step:
                    field_bcs = model.get_bcs_for_field(field_name)
                    if field_bcs:
                        bcs.extend(field_bcs)
        return bcs

    def Get_Dirichlet_Rows(self, step: int) -> np.ndarray:
        """Return the owned local step-vector indices constrained by Dirichlet conditions in a step.

        These are the rows (and columns) replaced by identities in the
        assembled tangent, in the layout of the step vector and of
        ``mesh.Get_Weights_Dofs``; convert them with
        ``mesh.get_step_global_indices`` for use with PETSc.
        """
        return self._dirichlet_rows(self.get_boundary_conditions(step), step)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _measure_runtime(self, category: str):
        """PRIVATE: Return this physics object's timing context."""

        profiler = getattr(self, "_runtime_profiler", None)
        if profiler is None:
            return nullcontext()
        return profiler.measure(category)

    def _get_fvec(self, step: int) -> PETSc.Vec:
        """PRIVATE: Perform one residual-only assembly."""
        t_start = time.time()
        with self._measure_runtime("form_evaluation"):
            K_form, F_form = self._collect_forms_for_step(step)
            self._validate_collected_forms(
                step,
                K_form,
                F_form,
                F_only=True,
            )
        with self._measure_runtime("boundary_conditions"):
            bcs = self.get_boundary_conditions(step)

        t_end = time.time()
        if self.DEBUG_PRINT_TIMING:
            mprint(f"\t\t Getting BCs duration: {t_end - t_start:.4f} s")
        t_start = time.time()
        F_compiled = self._get_or_compile_form(step, "F", F_form)
        K_compiled = (
            self._get_or_compile_form(step, "K", K_form)
            if K_form is not None and bcs
            else None
        )
        self.F_vec = self._assemble_F_for_step(
            step, K_compiled, F_compiled, bcs
        )

        t_end = time.time()
        if self.DEBUG_PRINT_TIMING:
            mprint(f"\t\t Assembling F total duration: {t_end - t_start:.4f} s")

        return self.F_vec

    def _get_kmat_fvec(self, step: int) -> tuple[PETSc.Mat, PETSc.Vec]:
        """PRIVATE: Perform one stiffness-and-residual assembly."""
        t_start = time.time()
        with self._measure_runtime("form_evaluation"):
            K_form, F_form = self._collect_forms_for_step(step)
            self._validate_collected_forms(step, K_form, F_form)
        with self._measure_runtime("boundary_conditions"):
            bcs = self.get_boundary_conditions(step)

        t_end = time.time()
        if self.DEBUG_PRINT_TIMING:
            mprint(f"\t\t Getting BCs duration: {t_end - t_start:.4f} s")

        t_start = time.time()
        F_compiled = self._get_or_compile_form(step, "F", F_form)
        K_compiled = self._get_or_compile_form(step, "K", K_form)
        self.K_mat, self.F_vec = self._assemble_KF_for_step(step, K_compiled, F_compiled, bcs)

        t_end = time.time()
        if self.DEBUG_PRINT_TIMING:
            mprint(f"\t\t Assembling K/F total duration: {t_end - t_start:.4f} s")

        return self.K_mat, self.F_vec

    def _destroy_assembled_objects(self) -> None:
        """PRIVATE: Destroy assembled objects and clear mesh-bound form data."""
        cached_object_ids = {
            id(obj)
            for workspace in self._assembly_cache.values()
            for obj in (workspace["K_mat"], workspace["F_vec"])
            if obj is not None
        }
        active_objects = (self.K_mat, self.F_vec)
        for workspace in self._assembly_cache.values():
            self._destroy_assembly_workspace(workspace)
        self._assembly_cache.clear()
        compiled_form_cache = getattr(self, "_compiled_form_cache", None)
        if compiled_form_cache is not None:
            compiled_form_cache.clear()
        model_weak_form_cache = getattr(self, "_model_weak_form_cache", None)
        if model_weak_form_cache is not None:
            model_weak_form_cache.clear()
        step_weak_form_cache = getattr(self, "_step_weak_form_cache", None)
        if step_weak_form_cache is not None:
            step_weak_form_cache.clear()
        for obj in active_objects:
            if obj is not None and id(obj) not in cached_object_ids:
                obj.destroy()
        self.K_mat = None
        self.F_vec = None

    def _destroy_assembly_workspace(self, workspace) -> None:
        """PRIVATE: Destroy one cached matrix/vector pair."""
        if workspace["K_mat"] is not None:
            workspace["K_mat"].destroy()
            workspace["K_mat"] = None
        if workspace["F_vec"] is not None:
            workspace["F_vec"].destroy()
            workspace["F_vec"] = None

    def _assemble_KF_for_step(self, step: int, K_compiled, F_compiled, bcs):
        """PRIVATE: Assemble into cached objects when safe, otherwise rebuild."""
        workspace = self._assembly_cache.get(step)
        if (
            workspace is not None
            and workspace["K_mat"] is not None
            and workspace["F_vec"] is not None
        ):
            try:
                self._assemble_into_objects(workspace["K_mat"],workspace["F_vec"],K_compiled,F_compiled,step,bcs,strict_reuse=True)
                return workspace["K_mat"], workspace["F_vec"]
            except Exception as exc:
                mprint(f"\t\tWarning: Reusing K/F allocation for step {step} failed "f"({exc}). Re-allocating.")
                self._destroy_assembly_workspace(workspace)
                self._assembly_cache.pop(step, None)

        if workspace is not None:
            self._destroy_assembly_workspace(workspace)
            self._assembly_cache.pop(step, None)

        with self._measure_runtime("local_assembly"):
            F_vec = self._assemble_new_vector(F_compiled)
        K_mat = create_matrix(K_compiled, kind="mpi") if self._is_blocked(K_compiled) else create_matrix(K_compiled)
        K_mat = self._assemble_into_objects(K_mat,F_vec,K_compiled,F_compiled,step,bcs,strict_reuse=False)
        workspace = {"K_mat": K_mat, "F_vec": F_vec}
        self._assembly_cache[step] = workspace
        return K_mat, F_vec

    def _assemble_F_for_step(self, step: int, K_compiled, F_compiled, bcs):
        """PRIVATE: Assemble a residual into reusable per-step vector storage."""
        workspace = self._assembly_cache.get(step)
        if workspace is not None and workspace["F_vec"] is not None:
            try:
                F_vec = workspace["F_vec"]
                with self._measure_runtime("local_assembly"):
                    self._zero_vector(F_vec)
                    assemble_vector(F_vec, F_compiled)
                self._finalize_force_vector(
                    None, F_vec, K_compiled, step, bcs
                )
                return F_vec
            except Exception as exc:
                mprint(
                    f"\t\tWarning: Reusing F allocation for step {step} failed "
                    f"({exc}). Re-allocating."
                )
                workspace["F_vec"].destroy()
                workspace["F_vec"] = None

        with self._measure_runtime("local_assembly"):
            F_vec = self._assemble_new_vector(F_compiled)
        try:
            self._finalize_force_vector(None, F_vec, K_compiled, step, bcs)
        except Exception:
            F_vec.destroy()
            raise

        if workspace is None:
            workspace = {"K_mat": None, "F_vec": F_vec}
            self._assembly_cache[step] = workspace
        else:
            workspace["F_vec"] = F_vec
        return F_vec

    def _assemble_into_objects(self,K_mat: PETSc.Mat,F_vec: PETSc.Vec,K_compiled,F_compiled,step: int,bcs,strict_reuse: bool) -> PETSc.Mat:
        """PRIVATE: Assemble compiled forms into existing PETSc objects; returns the matrix, which on a first assembly
        (``strict_reuse=False``) with model additions is a copy preallocated for them."""
        with self._measure_runtime("local_assembly"):
            self._zero_vector(F_vec)
            assemble_vector(F_vec, F_compiled)
        if bcs:
            with self._measure_runtime("boundary_conditions"):
                self._apply_lifting(F_vec, K_compiled, bcs)

        if strict_reuse and hasattr(K_mat, "setOption"):
            K_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        has_model_kf_additions = self._has_model_kf_additions()
        with self._measure_runtime("local_assembly"):
            K_mat.zeroEntries()
            assemble_matrix(
                K_mat,
                K_compiled,
                bcs=None if has_model_kf_additions else bcs,
            )

        if has_model_kf_additions:
            with self._measure_runtime("assembly_communication"):
                K_mat.assemble()

            # Algebraic additions must be made before constraints are imposed.
            # They may couple fields that share no weak-form term (e.g. across
            # an interface), so the first assembly extends the sparsity
            # pattern (preallocated, see _preallocate_model_KF); later
            # assemblies reuse it strictly.
            if not strict_reuse and hasattr(K_mat, "setOption"):
                if isinstance(K_mat, PETSc.Mat):
                    K_mat = self._preallocate_model_KF(K_mat, F_vec, step)
                K_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            self._add_model_KF(K_mat, F_vec, step)
            with self._measure_runtime("assembly_communication"):
                K_mat.assemble()
            self._finalize_force_vector(
                K_mat, F_vec, None, step, bcs, add_model_terms=False
            )
            with self._measure_runtime("assembly_communication"):
                K_mat.assemble()
        else:
            # DOLFINx applies matrix constraints during local assembly, so a
            # single finalization is sufficient when no model edits K/F later.
            with self._measure_runtime("assembly_communication"):
                K_mat.assemble()
            self._finalize_force_vector(
                None, F_vec, None, step, bcs, add_model_terms=False
            )
        if hasattr(K_mat, "setOption"):
            K_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        return K_mat

    def _preallocate_model_KF(self, K_mat: PETSc.Mat, F_vec: PETSc.Vec, step: int) -> PETSc.Mat:
        """PRIVATE: Return a copy of the assembled ``K_mat`` preallocated for the entries the models add in ``Add_KF``.

        Inserting entries outside the preallocation reallocates the whole
        matrix whenever a row runs full, so a dense row (a global field
        coupled to every interface node) costs time quadratic in the number
        of nodes. The models' additions are therefore first recorded in a
        PETSc preallocator together with the pattern of ``K_mat``; the copy
        holds the values of ``K_mat`` and room for the additions.
        """
        comm = K_mat.getComm()
        lgmap = K_mat.getLGMap()
        start, _end = K_mat.getOwnershipRange()
        indptr, columns, values = K_mat.getValuesCSR()
        rows = np.arange(start, start + indptr.size - 1, dtype=PETSc.IntType)

        recorder = PETSc.Mat().create(comm=comm)
        recorder.setType(PETSc.Mat.Type.PREALLOCATOR)
        recorder.setSizes(K_mat.getSizes(), bsize=K_mat.getBlockSize())
        recorder.setLGMap(*lgmap)
        recorder.setUp()
        scratch = F_vec.duplicate()
        try:
            for i, row in enumerate(rows):
                recorder.setValues([row], columns[indptr[i]:indptr[i + 1]], values[indptr[i]:indptr[i + 1]],
                                   addv=PETSc.InsertMode.ADD)
            for model in self.models.values():
                if hasattr(model, "Add_KF"):
                    model.Add_KF(recorder, scratch, step)
            recorder.assemble()

            extended = PETSc.Mat().create(comm=comm)
            extended.setType(K_mat.getType())
            extended.setSizes(K_mat.getSizes(), bsize=K_mat.getBlockSize())
            recorder.preallocatorPreallocate(extended, fill=True)
        finally:
            recorder.destroy()
            scratch.destroy()
        extended.setLGMap(*lgmap)
        for i, row in enumerate(rows):
            extended.setValues([row], columns[indptr[i]:indptr[i + 1]], values[indptr[i]:indptr[i + 1]],
                               addv=PETSc.InsertMode.INSERT)
        extended.assemble()
        K_mat.destroy()
        return extended

    def _has_model_kf_additions(self) -> bool:
        """PRIVATE: Return whether any active model supplies algebraic K/F edits."""
        return any(
            callable(getattr(model, "Add_KF", None))
            for model in self.models.values()
        )

    def _add_model_KF(self, K_mat, F_vec, step: int) -> None:
        """PRIVATE: Apply model-provided algebraic tangent/residual additions."""
        with self._measure_runtime("model_assembly"):
            for model in self.models.values():
                if hasattr(model, "Add_KF"):
                    model.Add_KF(K_mat, F_vec, step)
        # Rows of global fields are owned by one rank; models on other ranks
        # add to them with F.setValues, which needs vector assembly.
        if any(self.mesh.Is_Global_Field(name) for name in self.mesh.step_fields.get(step, [])):
            with self._measure_runtime("assembly_communication"):
                F_vec.assemblyBegin()
                F_vec.assemblyEnd()

    def _finalize_force_vector(
        self,
        K_mat,
        F_vec,
        K_compiled,
        step: int,
        bcs,
        add_model_terms: bool = True,
    ) -> None:
        """PRIVATE: Add model forces, apply constraints, and synchronize a residual."""
        if bcs and K_compiled is not None:
            with self._measure_runtime("boundary_conditions"):
                self._apply_lifting(F_vec, K_compiled, bcs)
        if add_model_terms:
            self._add_model_KF(K_mat, F_vec, step)

        with self._measure_runtime("assembly_communication"):
            F_vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
        if bcs:
            with self._measure_runtime("boundary_conditions"):
                if K_mat is not None:
                    self._apply_bcs_to_matrix(K_mat, bcs, step)
                self._set_bc(F_vec, bcs, step)
        with self._measure_runtime("assembly_communication"):
            F_vec.ghostUpdate(
                addv=PETSc.InsertMode.INSERT,
                mode=PETSc.ScatterMode.FORWARD,
            )

    def _zero_vector(self, F_vec: PETSc.Vec) -> None:
        """PRIVATE: Clear an existing PETSc vector before reassembly."""
        if hasattr(F_vec, "localForm"):
            with F_vec.localForm() as F_local:
                F_local.set(0.0)
        elif hasattr(F_vec, "set"):
            F_vec.set(0.0)
        else:
            raise TypeError("Cannot zero PETSc vector before reassembly.")

    def _collect_forms_for_step(self, step: int):
        """PRIVATE: Accumulate all model weak forms for a staggered step."""
        self._sync_weak_constants()
        if not hasattr(self, "_model_weak_form_cache"):
            self._model_weak_form_cache = {}
        if not hasattr(self, "_step_weak_form_cache"):
            self._step_weak_form_cache = {}
        rebuild_step = step not in self._step_weak_form_cache

        for model_name, model in self.models.items():
            t_start = time.time()
            refresh = model.Update_Weak_Constants(
                self.time_constant,
                self.dt_constant,
                step,
            )
            if not isinstance(refresh, (bool, np.bool_)):
                raise TypeError(
                    f"{model.__class__.__name__}.Update_Weak_Constants() "
                    "must return bool."
                )
            if refresh:
                stale_keys = [
                    key
                    for key in self._model_weak_form_cache
                    if key[0] == model_name
                ]
                for key in stale_keys:
                    self._model_weak_form_cache.pop(key)
                self._step_weak_form_cache.clear()
                rebuild_step = True

            cache_key = (model_name, step)
            if cache_key not in self._model_weak_form_cache:
                self._model_weak_form_cache[cache_key] = model.Get_Weak_Form(
                    self.time_constant,
                    self.dt_constant,
                    step,
                )
                rebuild_step = True

            if self.DEBUG_PRINT_TIMING:
                comm.Barrier()
                t_end = time.time()
                mprint(f"\t\t '{model.name}': {t_end - t_start:.4f} s")

        if not rebuild_step:
            return self._step_weak_form_cache[step]

        F_form = None
        K_form = None
        for model_name in self.models:
            result = self._model_weak_form_cache[(model_name, step)]
            if result is not None:
                K, F = result

                if F is not None:
                    F_form = F if F_form is None else F_form + F
                if K is not None:
                    K_form = K if K_form is None else K_form + K

        self._step_weak_form_cache[step] = (K_form, F_form)
        return self._step_weak_form_cache[step]

    def _get_or_compile_form(self, step: int, role: str, ufl_form):
        """PRIVATE: Reuse a compiled form while its assembly inputs are stable."""
        if role not in {"K", "F"}:
            raise ValueError(f"Unknown compiled-form role: {role!r}.")
        if ufl_form is None:
            return None

        cache = getattr(self, "_compiled_form_cache", None)
        if cache is None:
            cache = {}
            self._compiled_form_cache = cache

        with self._measure_runtime("form_cache_lookup"):
            cache_key = (step, role)
            cached = cache.get(cache_key)
        if cached is not None and cached.get("ufl_form") is ufl_form:
            expression_key = cached["expression_key"]
            cache_hit = True
        else:
            expression_key = self._form_cache_key(ufl_form)
            cache_hit = (
                cached is not None
                and cached["expression_key"] == expression_key
            )

        profiler = getattr(self, "_runtime_profiler", None)
        if cache_hit:
            if profiler is not None:
                profiler.count("form_cache_hit")
            return cached["compiled"]

        if profiler is not None:
            profiler.count("form_cache_miss")
        with self._measure_runtime("form_compilation"):
            compiled = self._compile_step_form(step, role, ufl_form)
        cache[cache_key] = {
            "expression_key": expression_key,
            "ufl_form": ufl_form,
            "compiled": compiled,
        }
        return compiled

    def _form_cache_key(self, ufl_form):
        """PRIVATE: Return a cheap key for UFL expressions and their bindings."""
        integrals = getattr(ufl_form, "integrals", None)
        if not callable(integrals):
            try:
                return ("value", type(ufl_form), hash(ufl_form))
            except TypeError:
                return ("identity", type(ufl_form), id(ufl_form))

        integral_keys = []
        index_numbering = {}
        for integral in integrals():
            domain = integral.ufl_domain()
            cargo = domain.ufl_cargo() if domain is not None else None
            integral_keys.append(
                (
                    self._integrand_cache_key(
                        integral.integrand(),
                        index_numbering,
                    ),
                    integral.integral_type(),
                    integral.subdomain_id(),
                    self._freeze_form_metadata(integral.metadata()),
                    id(cargo),
                    id(integral.subdomain_data()),
                )
            )
        return (type(ufl_form), tuple(integral_keys))

    def _integrand_cache_key(self, integrand, index_numbering):
        """PRIVATE: Hash an expression DAG without incidental UFL index IDs."""
        node_hashes = {}
        node_count = 0
        for expression in unique_post_traversal(integrand):
            node_count += 1
            if expression._ufl_is_terminal_:
                value = self._terminal_cache_key(
                    expression,
                    index_numbering,
                )
            else:
                value = (
                    expression._ufl_typecode_,
                    tuple(
                        node_hashes[operand]
                        for operand in expression.ufl_operands
                    ),
                )
            node_hashes[expression] = hash(value)
        return node_hashes[integrand], node_count

    def _terminal_cache_key(self, expression, index_numbering):
        """PRIVATE: Describe one UFL terminal without a canonical signature."""
        typecode = expression._ufl_typecode_
        if isinstance(expression, MultiIndex):
            indices = []
            for index in expression.indices():
                if isinstance(index, Index):
                    if index not in index_numbering:
                        index_numbering[index] = len(index_numbering)
                    indices.append(("index", index_numbering[index]))
                else:
                    indices.append(("fixed", int(index)))
            return typecode, tuple(indices)
        if isinstance(expression, Coefficient | Constant):
            return typecode, id(expression)
        if isinstance(expression, Argument):
            function_space = expression.ufl_function_space()
            return (
                typecode,
                expression.number(),
                expression.part(),
                id(function_space),
                id(getattr(function_space, "_cpp_object", None)),
            )
        if isinstance(expression, GeometricQuantity):
            domain = extract_unique_domain(expression)
            cargo = domain.ufl_cargo() if domain is not None else None
            return typecode, id(cargo)
        if isinstance(expression, ConstantValue):
            return typecode, hash(expression)
        if isinstance(expression, Label):
            return typecode, id(expression)
        if isinstance(expression, ExprList):
            return typecode, "[]"
        if isinstance(expression, ExprMapping):
            return typecode, "{}"
        return typecode, repr(expression)

    def _freeze_form_metadata(self, value):
        """PRIVATE: Convert integral metadata to a stable hashable value."""
        if isinstance(value, dict):
            items = (
                (
                    type(key).__qualname__,
                    repr(key),
                    self._freeze_form_metadata(item),
                )
                for key, item in value.items()
            )
            return tuple(sorted(items))
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_form_metadata(item) for item in value)
        if isinstance(value, (set, frozenset)):
            frozen = (
                self._freeze_form_metadata(item)
                for item in value
            )
            return tuple(sorted(frozen, key=repr))
        try:
            hash(value)
        except TypeError:
            return ("identity", type(value).__qualname__, id(value))
        return value

    def _validate_collected_forms(
        self, step: int, K_form, F_form, F_only: bool = False
    ) -> None:
        """PRIVATE: Fail fast when a staggered step assembled no usable forms."""
        active_fields = [name for name, _ in self.get_fields_for_step(step)]
        if K_form is None and F_form is None:
            raise RuntimeError(f"Step {step} assembled no bilinear or linear form. "
                f"Active fields: {active_fields}. "
                "Check the step configuration and model Get_Weak_Form() implementations."
            )
        if K_form is None and not F_only:
            raise RuntimeError(
                f"Step {step} assembled no bilinear form. "
                f"Active fields: {active_fields}. "
                "Check the active models for a missing tangent contribution."
            )
        if F_form is None:
            raise RuntimeError(
                f"Step {step} assembled no linear form. "
                f"Active fields: {active_fields}. "
                "Check the active models for a missing residual contribution."
            )

    def _apply_bcs_to_matrix(self, K_mat: PETSc.Mat, bcs, step: int) -> None:
        """PRIVATE: Apply Dirichlet constraints to an assembled PETSc matrix."""
        rows = self._dirichlet_rows(bcs, step)
        if rows.size:
            # Rows are owned entries of the step vector; the local numbering
            # of block matrices differs, so constrain them by global index.
            rows = self.mesh.get_step_global_indices(step, rows)
        K_mat.zeroRowsColumns(rows, diag=1.0)

    def _dirichlet_rows(self, bcs, step: int) -> np.ndarray:
        """PRIVATE: Return the owned local step-vector indices constrained by ``bcs``."""
        rows = []
        for bc in bcs:
            dofs, pos = bc.dof_indices()
            if pos > 0:
                rows.append(dofs[:pos] + self._bc_block_offset(bc, step))
        if not rows:
            return np.empty(0, dtype=PETSc.IntType)
        return np.unique(np.concatenate(rows)).astype(PETSc.IntType)

    @staticmethod
    def _is_blocked(compiled) -> bool:
        """PRIVATE: Return whether compiled forms describe a block system."""
        return isinstance(compiled, (list, tuple))

    def _compile_step_form(self, step: int, role: str, ufl_form):
        """PRIVATE: Compile a step form, split into field blocks for monolithic steps."""
        entity_maps = self.mesh.get_entity_maps() or None
        block_layout = self.mesh.mixed_spaces.get(step)
        if block_layout is None:
            return fem.form(ufl_form, entity_maps=entity_maps)

        field_names = block_layout["field_names"]
        blocks = self.mesh.Add_Global_Field_Placeholders(ufl.extract_blocks(ufl_form), step, role)
        if role == "F":
            missing = [name for name, block in zip(field_names, blocks) if block is None]
        else:
            missing = [name for i, name in enumerate(field_names) if blocks[i][i] is None]
        if missing:
            raise RuntimeError(
                f"Step {step} has no {'residual' if role == 'F' else 'diagonal tangent'} "
                f"contribution for field(s) {missing}. Every field of a monolithic step "
                "needs its own equation."
            )
        return fem.form(blocks, entity_maps=entity_maps)

    def _assemble_new_vector(self, F_compiled) -> PETSc.Vec:
        """PRIVATE: Allocate and assemble a residual vector for plain or block forms."""
        if self._is_blocked(F_compiled):
            return assemble_vector(F_compiled, kind="mpi")
        return assemble_vector(F_compiled)

    def _apply_lifting(self, F_vec: PETSc.Vec, K_compiled, bcs) -> None:
        """PRIVATE: Lift Dirichlet constraints into a plain or block residual."""
        if self._is_blocked(K_compiled):
            apply_lifting(F_vec, K_compiled, fem.bcs_by_block(fem.extract_function_spaces(K_compiled, 1), bcs))
        else:
            apply_lifting(F_vec, [K_compiled], [bcs])

    def _set_bc(self, F_vec: PETSc.Vec, bcs, step: int) -> None:
        """PRIVATE: Set Dirichlet values in a plain or block residual."""
        block_layout = self.mesh.mixed_spaces.get(step)
        if block_layout is None:
            set_bc(F_vec, bcs)
        else:
            spaces = [V._cpp_object for V in block_layout["function_spaces"]]
            set_bc(F_vec, fem.bcs_by_block(spaces, bcs))

    def _bc_block_offset(self, bc, step: int) -> int:
        """PRIVATE: Return the local owned offset of a BC's field block in a step."""
        block_layout = self.mesh.mixed_spaces.get(step)
        if block_layout is None:
            return 0
        offsets = self.mesh.get_step_block_offsets(step)
        for block, function_space in enumerate(block_layout["function_spaces"]):
            if function_space._cpp_object.contains(bc.function_space):
                return offsets[block]
        raise ValueError(f"Dirichlet BC does not act on any field of step {step}.")  # pragma: no cover - BCs are collected per step field

    def _get_fields_for_step_internal(self, step: int):
        """PRIVATE: Read the active solution fields for a step from the mesh."""
        if not hasattr(self.mesh, "step_fields"):
            mprint("Warning: No step fields registered in mesh")
            return []

        fields_to_solve = []
        for field_name in self.mesh.step_fields.get(step, []):
            if field_name in self.mesh.fields:
                fields_to_solve.append((field_name, self.mesh.fields[field_name]))
            else:
                mprint(f"  Warning: Step field '{field_name}' was registered but is missing from mesh.fields")
        return fields_to_solve

    def _get_boundary_condition_fields(self, step: int = None):
        """PRIVATE: Resolve which solution fields contribute BCs."""
        if not hasattr(self.mesh, "step_fields"):
            return []

        if step is not None:
            return list(self.mesh.step_fields.get(step, []))

        field_names = []
        for step_fields in self.mesh.step_fields.values():
            for field_name in step_fields:
                if field_name not in field_names:
                    field_names.append(field_name)
        return field_names
