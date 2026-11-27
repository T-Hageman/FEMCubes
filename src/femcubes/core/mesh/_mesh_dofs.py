"""Field, mixed-space, and output-preparation helpers for the mesh class."""

from basix.ufl import element, quadrature_element
from basix import LagrangeVariant
from dolfinx import fem, mesh as dmesh
from dolfinx.graph import adjacencylist
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from ufl import MixedFunctionSpace, extract_blocks, TestFunction, TestFunctions, TrialFunction, TrialFunctions, Argument
from ufl import Coefficient, Measure, dx, sqrt
from ufl import Mesh as ufl_mesh
from ufl import replace as ufl_replace
from ufl.algorithms.analysis import extract_coefficients, extract_constants
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_derivatives
from ufl.corealg.map_dag import map_expr_dag
from ufl.corealg.multifunction import MultiFunction
from ufl.domain import extract_domains
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

from femcubes.core.utils.mpi_utils import mprint
from ._mesh_base import WeightedMeasure

_VECTOR_COMPONENT_SUFFIXES = {"x": 0, "y": 1, "z": 2}
_TENSOR_COMPONENT_SUFFIXES = {
    "xx": (0, 0),
    "xy": (0, 1),
    "xz": (0, 2),
    "yx": (1, 0),
    "yy": (1, 1),
    "yz": (1, 2),
    "zx": (2, 0),
    "zy": (2, 1),
    "zz": (2, 2),
}

_MANDEL_COMPONENT_SUFFIXES = {
    "xx": (0, 1.0),
    "yy": (1, 1.0),
    "zz": (2, 1.0),
    "yz": (3, 1.0 / sqrt(2.0)),
    "xz": (4, 1.0 / sqrt(2.0)),
    "xy": (5, 1.0 / sqrt(2.0)),
}

_STANDARD_FIELD_FAMILIES = {"CG", "DG"}
_GLOBAL_FAMILY = "Global"  # one scalar DOF for the whole problem, e.g. a metal potential


@dataclass(frozen=True)
class FieldDefinition:
    """Complete mesh-independent description of one registered field.

    The legacy ``field_type`` and ``field_order`` dictionaries remain
    available for compatibility, while this structured definition is the
    source of truth for recreating fields after a mesh change.
    """

    name: str
    description: str
    family: str
    degree: int
    value_shape: Tuple[int, ...] = ()
    quadrature_degree: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Field name must be a non-empty string.")
        if not isinstance(self.description, str):
            raise TypeError("Field description must be a string.")
        if self.family not in _STANDARD_FIELD_FAMILIES | {"Quadrature", _GLOBAL_FAMILY}:
            raise ValueError(f"Unsupported function space: {self.family}")
        if isinstance(self.degree, bool) or not isinstance(self.degree, int):
            raise ValueError("Field degree must be an integer.")
        if not isinstance(self.value_shape, tuple):
            raise TypeError("Field value_shape must be a tuple.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in self.value_shape):
            raise ValueError("Field value_shape entries must be positive integers.")

        if self.family == _GLOBAL_FAMILY:
            if self.degree != 0 or self.value_shape or self.quadrature_degree is not None:
                raise ValueError("Global fields are single scalar values (degree 0, no value shape).")
        elif self.family == "Quadrature":
            if self.degree != -1:
                raise ValueError("Quadrature field degree must use the compatibility value -1.")
            if (
                isinstance(self.quadrature_degree, bool)
                or not isinstance(self.quadrature_degree, int)
                or self.quadrature_degree < 1
            ):
                raise ValueError("Quadrature fields require a positive integer quadrature_degree.")
        else:
            if self.degree < 0:
                raise ValueError("CG/DG field degree must be nonnegative.")
            if self.quadrature_degree is not None:
                raise ValueError("CG/DG fields cannot define quadrature_degree.")

    def is_compatible_with(self, other: "FieldDefinition") -> bool:
        """Return whether two registrations describe the same FE storage.

        Descriptions are labels rather than storage contracts.  Re-registering
        a field with a different description therefore preserves the original
        function and label, matching the established FEMCubes behavior.
        """

        if not isinstance(other, FieldDefinition):
            return False
        return (
            self.name == other.name
            and self.family == other.family
            and self.degree == other.degree
            and self.value_shape == other.value_shape
            and self.quadrature_degree == other.quadrature_degree
        )


FULL_DOMAIN = "domain"
_FACETS = "facets"  # first entry of the support key of interface fields
_GLOBAL = "global"  # support key and domain name of global fields


@dataclass(frozen=True)
class FieldSupport:
    """The cells a field is defined on, and the (sub)mesh carrying its DOFs.

    Fields registered for the full domain live on the parent mesh. Fields
    restricted to part of the domain live on a submesh of the cells of their
    (sub)domains, so DOFs only exist where they are needed.

    Attributes
    ----------
    domain_names : tuple of str
        Domain names the field was registered for, in registration order.
    tags : tuple of int
        Cell tags covered by the support, or facet tags for interface fields.
    mesh : dolfinx.mesh.Mesh
        Mesh carrying the field's DOFs: the parent mesh or a submesh.
    cell_map : dolfinx.mesh.EntityMap or None
        Map between the submesh cells and the parent cells; ``None`` for
        fields on the full parent mesh.
    parent_cells : numpy.ndarray or None
        Parent cell (or parent facet, for interface fields) of each local
        (owned and ghost) submesh cell; ``None`` for fields on the full
        parent mesh.
    is_interface : bool
        Whether the field lives on facets (an interface or boundary) rather
        than on cells.
    is_global : bool
        Whether the field is a single global value (see
        :meth:`_MeshDOFs.Add_Global_Field`), not tied to any cells.
    """

    domain_names: Tuple[str, ...]
    tags: Tuple[int, ...]
    mesh: Any
    cell_map: Any = None
    parent_cells: Optional[np.ndarray] = None
    is_interface: bool = False
    is_global: bool = False

    @property
    def is_full(self) -> bool:
        """Return whether the field lives on the full parent mesh."""
        return self.cell_map is None and not self.is_global


@dataclass(frozen=True)
class FieldOutputView:
    """Resolved concrete view of one registered field.

    A :class:`FieldOutputView` is the public output-side companion to
    :meth:`_MeshDOFs.Get_Field`. Every field owns its own storage, also when
    it is solved together with other fields in a monolithic step, so the
    view always refers to that field's function. It provides a helper for
    collapsing scalar components into ordinary DOLFINx functions suitable for
    interpolation, file output, or post-processing.
    """

    field_name: str
    field: fem.Function
    comp_count: int

    @property
    def title(self) -> str:
        """Return the display title used for output and plotting."""
        return self.field.name

    @property
    def value_shape(self) -> Tuple[int, ...]:
        """Return the value shape of the registered field."""
        return self.field.function_space.element.value_shape

    def collapse_component(self, component: int = 0) -> fem.Function:
        """Return one scalar component as a concrete DOLFINx function."""
        if component < 0 or component >= self.comp_count:
            raise IndexError(
                f"Component {component} out of range for field '{self.field_name}' "
                f"with {self.comp_count} component(s)."
            )

        if len(self.value_shape) == 0:
            return self.field
        return self.field.sub(component).collapse()


@dataclass(frozen=True)
class ProjectedFieldOutput:
    """DG-projected representation of one output field specification.

    Instances of this dataclass are returned by
    :meth:`_MeshDOFs.Prepare_Field_For_Output` and represent a scalar
    quantity already projected to a discontinuous Galerkin space for
    plotting or file export.
    """

    field_spec: str
    field_name: str
    title: str
    values: Any
    function_space: Any
    association: str = "point"


class _InterfaceSideRestrictor(MultiFunction):
    """PRIVATE: Restrict every field of an interface integrand to the side it is evaluated from.

    Expects an integrand with algebra lowered and derivatives applied, so
    gradients only act on terminals. ``side_of(terminal)`` returns "+" or "-".
    """

    def __init__(self, side_of):
        super().__init__()
        self._side_of = side_of

    expr = MultiFunction.reuse_if_untouched

    def terminal(self, o):
        return o

    def restricted(self, o):
        return o

    def facet_normal(self, o):
        return o("+")

    def coefficient(self, o):
        return o(self._side_of(o))

    argument = coefficient

    def grad(self, o):
        terminal = o
        while terminal.ufl_operands:
            terminal = terminal.ufl_operands[0]
        if isinstance(terminal, (Coefficient, Argument)):
            return o(self._side_of(terminal))
        return o("+")

    reference_grad = grad


class _MeshDOFs:
    """Mix-in for managing finite-element fields and function spaces.

    Provides helpers to register scalar, vector, and quadrature fields;
    create test and trial functions; build block layouts for monolithic
    solution steps; and prepare scalar DG projections for plotting or file
    output.

    Every field owns its own function. A solution step with several fields
    is solved monolithically as a block system: its fields form a
    ``ufl.MixedFunctionSpace`` with one block per field, in registration
    order, and the assembled matrix/vector use the matching block layout.

    Attributes
    ----------
    fields : dict
        Mapping ``{field_name: dolfinx.fem.Function}``.
    field_order : dict
        Polynomial order for each registered field.
    field_type : dict
        Element family for each registered field (e.g. ``"CG"``).
    test_functions : dict
        Test functions keyed by ``"{field_name}_test_step{step}"``.
    trial_functions : dict
        Cached trial functions keyed by ``"{field_name}_trial_step{step}"``.
    mixed_spaces : dict
        Per-step block-layout metadata (populated by
        ``finalize_test_functions``); ``None`` for single-field steps.
    step_fields : dict
        Mapping ``{step: [field_names]}`` indicating which fields are
        solved at each staggered step.

    Public API
    ----------
    Methods without a leading underscore are intended for external use
    by solvers, physics objects, and tests. Helpers prefixed with ``_``
    are internal implementation details.
    """

    def __init__(self, params: Any) -> None:
        self.fields: Dict[str, fem.Function] = {}
        self.field_order: Dict[str, int] = {}
        self.field_type: Dict[str, str] = {}
        self.field_definitions: Dict[str, FieldDefinition] = {}
        self._dg_spaces: Dict[int, Any] = {}
        self._cg_spaces: Dict[int, Any] = {}
        self._quadrature_spaces: Dict[Tuple[int, Tuple[int, ...]], Any] = {}
        self._function_space_cache_epoch: Optional[int] = getattr(self, "remesh_epoch", 0)
        self._output_projection_cache: Dict[Tuple[str, int], fem.Function] = {}
        self._output_expression_cache: Dict[Tuple[str, int], fem.Expression] = {}
        self._output_mass_projection_cache: Dict[int, tuple[Any, PETSc.Mat, PETSc.KSP]] = {}
        self._output_projection_cache_epoch: Optional[int] = getattr(self, "remesh_epoch", 0)

        self.test_functions = {}
        self.trial_functions = {}
        self.mixed_spaces = {}  # Block layout for each monolithic step
        self.step_fields = {}   # Track which fields belong to each step
        self._field_domains: Dict[str, List[str]] = {}
        self._field_support_keys: Dict[str, Optional[Tuple[int, ...]]] = {}
        self._submeshes: Dict[Tuple[int, ...], Tuple[Any, Any, np.ndarray]] = {}
        self._submesh_entity_maps: Dict[Any, Any] = {}
        self._submesh_vertex_maps: Dict[Any, Any] = {}
        self._field_dofs_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
        self._step_global_layouts: Dict[int, Any] = {}
        self._submesh_proxies: Dict[Tuple[int, Tuple[int, ...]], Any] = {}
        self._test_functions_finalized = False
        self._global_mesh = None

        super().__init__(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def Print_Fields(self) -> None:
        """Print a summary of the registered fields and their discretization."""
        mprint("Mesh fields:")
        for fname, fobj in self.fields.items():
            mprint(f"\t{fname}:  {self.field_type[fname]}{self.field_order[fname]} function ({fobj.name})")

    def Add_Field(self, field_name: str, field_description: str, space: str, order: int, domain: str = FULL_DOMAIN) -> fem.Function:
        """Create or return a scalar finite-element field.

        Parameters
        ----------
        field_name : str
            Internal field identifier used throughout the code base.
        field_description : str
            Human-readable label stored on the DOLFINx function.
        space : {"CG", "DG"}
            Finite-element family.
        order : int
            Polynomial order of the scalar field.
        domain : str, optional
            Domain (group or part) name the field is needed on, e.g.
            ``"domain:centre"``. A field registered for several domains
            lives on the union of them; DOFs only exist there. A facet
            group such as ``"interface"`` creates an interface field, which
            lives on those facets only. Defaults to the full ``"domain"``.
        """
        definition = FieldDefinition(
            name=field_name,
            description=field_description,
            family=space,
            degree=order,
        )
        return self._register_field_definition(definition, domain)

    def Add_Vector_Field(self, field_name: str, field_description: str, space: str, order: int, nDim: int, domain: str = FULL_DOMAIN) -> fem.Function:
        """Create or return a vector-valued finite-element field.

        See :meth:`Add_Field` for the meaning of ``domain``.
        """
        definition = FieldDefinition(
            name=field_name,
            description=field_description,
            family=space,
            degree=order,
            value_shape=(nDim,),
        )
        return self._register_field_definition(definition, domain)

    def Add_Quadrature_Field(self, field_name: str, field_description: str, size: Optional[Tuple[int, ...]] = None, domain: str = FULL_DOMAIN):
        """Create or return a quadrature-space field.

        Quadrature fields are stored in a cached quadrature function space
        keyed by quadrature degree and tensor shape.  They are primarily
        used for internal variables such as stresses or history terms that
        are evaluated at quadrature points rather than nodal DOFs. See
        :meth:`Add_Field` for the meaning of ``domain``.
        """
        quad_degree = getattr(self.params, "Quadrature_Degree", 2)
        value_shape = () if size is None else tuple(size)
        definition = FieldDefinition(
            name=field_name,
            description=field_description,
            family="Quadrature",
            degree=-1,
            value_shape=value_shape,
            quadrature_degree=quad_degree,
        )
        return self._register_field_definition(definition, domain)

    def Add_Global_Field(self, field_name: str, field_description: str) -> fem.Function:
        """Create or return a global field: one scalar unknown for the whole problem.

        A global field (e.g. the potential of a metal that conducts
        perfectly) is not tied to any cells. It can be solved monolithically
        with mesh fields through test/trial functions like any other field;
        models contribute to its equation algebraically in ``Add_KF``, using
        :meth:`Get_Global_Field_Dof` for its matrix row/column and
        :meth:`Get_Global_Value` for its value. Steps without a weak-form
        term for the field get a zero diagonal block automatically.

        Global fields have no spatial representation: expressions, boundary
        conditions and field output reject them; report their value through
        ``params.Global_Measures`` instead.

        Example
        -------
        ``E_m = mesh.Add_Global_Field("E_m", "Metal potential [V]")``
        """
        definition = FieldDefinition(name=field_name, description=field_description, family=_GLOBAL_FAMILY, degree=0)
        return self._register_field_definition(definition, _GLOBAL)

    def Is_Global_Field(self, field_name: str) -> bool:
        """Return whether a registered field is a global field (see :meth:`Add_Global_Field`)."""
        return self._field_support_keys.get(field_name) == (_GLOBAL,)

    def Get_Global_Value(self, field_name: str) -> float:
        """Return the value of a global field on every rank (collective)."""
        field = self._get_global_field_or_raise(field_name)
        owned = field.x.array[: field.function_space.dofmap.index_map.size_local]
        return float(self.mesh.comm.bcast(owned[0] if owned.size else None, root=self._global_field_owner(field)))

    def Get_Global_Field_Dof(self, field_name: str, step: int) -> int:
        """Return the global matrix/vector index of a global field in one step (collective).

        Use it on every rank with ``K.setValues`` and
        ``F.setValues(..., addv=PETSc.InsertMode.ADD_VALUES)``: the field's
        row is owned by a single rank, and contributions from other ranks
        are communicated during assembly.

        Raises
        ------
        ValueError
            If the field is not solved in ``step``.
        """
        field = self._get_global_field_or_raise(field_name)
        if field_name not in self.step_fields.get(step, []):
            raise ValueError(f"Global field '{field_name}' is not solved in step {step}.")
        owner = self._global_field_owner(field)
        self._get_step_global_layout(step)  # collective when not cached yet
        index = None
        if self.mesh.comm.rank == owner:
            local = self._step_vector_indices(step, field_name, np.array([0], dtype=np.int32))
            index = int(self.get_step_global_indices(step, local)[0])
        return int(self.mesh.comm.bcast(index, root=owner))

    def Add_Vector_TestFunction(self, field_name: str, nDim: int, step: int) -> fem.Function:
        """
        Create a test function for the specified vector field.

        Args:
            field_name: Name of the vector field to create a test function for
            nDim: Number of dimensions (should match the field's dimension)
            step: Solution step identifier

        Returns:
            Test function from the field's function space

        Raises:
            ValueError: If field_name is not defined or is not a vector field
        """
        # Optionally verify dimension matches
        if field_name in self.fields:
            field = self.fields[field_name]
            element = getattr(getattr(field, "function_space", None), "element", None)
            value_shape = getattr(element, "value_shape", ())
            if len(value_shape) > 0 and value_shape[0] != nDim:
                mprint(f"Warning: Requested dimension {nDim} does not match field dimension {value_shape[0]}")

        return self._add_test_function_internal(field_name, step, is_vector=True)

    def Add_TestFunction(self, field_name: str, step: int) -> fem.Function:
        """
        Create a test function for the specified scalar field.

        Args:
            field_name: Name of the field to create a test function for
            step: Solution step identifier

        Returns:
            Test function from the field's function space

        Raises:
            ValueError: If field_name is not defined or is a vector field
        """
        return self._add_test_function_internal(field_name, step, is_vector=False)

    def Get_TestFunction(self, field_name: str, step: int) -> Optional[fem.Function]:
        """Retrieve a test function for the specified field and step."""
        test_key = f"{field_name}_test_step{step}"
        test_func = self.test_functions.get(test_key, None)
        if test_func is None:
            raise ValueError(f"Test function for field '{field_name}' at step {step} not found.")
        return test_func

    def Get_Trial_Function(self, field_name: str, step: int):
        """Retrieve a trial function for the specified field and step."""
        trial_key = f"{field_name}_trial_step{step}"
        if trial_key in self.trial_functions:
            return self.trial_functions[trial_key]

        block = self._get_step_block_index(field_name, step)
        if block is not None:
            trial_func = self.mixed_spaces[step]["trial_functions"][block]
            self.trial_functions[trial_key] = trial_func
            return trial_func

        trial_func = TrialFunction(self._get_field_or_raise(field_name).function_space)
        self.trial_functions[trial_key] = trial_func
        return trial_func

    def Get_Field(self, field_name: str, step: int = None):
        """
        Retrieve a field by name.

        Every field owns its own function, also when it is solved together
        with other fields in a monolithic step, so the returned function can
        be used directly in UFL forms.

        Args:
            field_name: Name of the field to retrieve
            step: Accepted for API compatibility; fields do not depend on it

        Returns:
            The field's ``dolfinx.fem.Function``
        """
        return self._get_field_or_raise(field_name)

    def Get_Field_For_Output(self, field_name: str):
        """
        Retrieve a structured view suitable for plotting and file output.

        Unlike ``Get_Field``, which may return symbolic UFL expressions for
        form assembly, this returns a concrete :class:`FieldOutputView`
        describing where the actual field data live.

        Parameters
        ----------
        field_name : str
            Name of the registered field.

        Returns
        -------
        FieldOutputView
            Structured description of the field's concrete storage.
        """
        return self._resolve_field_reference(field_name)

    def Prepare_Field_For_Output(self, field_spec: str, V_dg=None) -> Optional[ProjectedFieldOutput]:
        """
        Prepare one scalar output quantity for plotting or file export.

        The ``field_spec`` may be a scalar field name or a recognised
        component suffix such as ``u_x`` or ``sigma_xy``. The returned
        data are always projected to a scalar DG space.

        Parameters
        ----------
        field_spec : str
            Scalar field name or recognised component specifier.
        V_dg : dolfinx.fem.FunctionSpace, optional
            DG space to reuse for the projection.  If omitted, a cached
            DG space with a suitable degree is created automatically.

        Returns
        -------
        ProjectedFieldOutput or None
            DG-projected scalar data ready for output, or ``None`` if the
            specification cannot be resolved to a valid scalar quantity.
        """
        field_name, component_kind, component = self._parse_output_field_spec(field_spec)
        if self._skip_global_field_output(field_name):
            return None
        if component_kind == "vector":
            return self._project_vector_field_component(field_name, component, field_spec, V_dg)
        if component_kind == "tensor":
            return self._project_tensor_field_component(field_name, component, field_spec, V_dg)
        return self._project_scalar_field(field_name, field_spec, V_dg)

    def Prepare_Field_For_Plotting(self, field_spec: str, V_dg=None) -> Optional[ProjectedFieldOutput]:
        """
        Prepare one scalar quantity specifically for interactive plotting.

        Ordinary nodal fields keep the standard DG projection used by
        :meth:`Prepare_Field_For_Output`. Quadrature-backed fields are
        instead projected to DG0 and intended to be rendered as cell
        data, which avoids DG1 oscillations from intra-cell
        reconstruction.

        Fields restricted to a subdomain are projected onto spaces on their
        own submesh, so only that subdomain is plotted, unless
        ``params.Plot_Subdomains_Only`` is ``False`` (then they are shown on
        the full mesh with NaN outside their subdomain).
        """
        field_name, component_kind, component = self._parse_output_field_spec(field_spec)
        if self._skip_global_field_output(field_name):
            return None
        support_key = self._field_support_keys.get(field_name)
        if support_key is not None and getattr(self.params, "Plot_Subdomains_Only", True):
            degree = 1 if V_dg is None else V_dg.element.basix_element.degree
            V_dg = self._get_dg_space(degree, support_key)
        else:
            support_key = None
        quadrature_plot_space = self._get_dg_space(0, support_key)

        if component_kind == "vector":
            return self._project_vector_field_component(
                field_name,
                component,
                field_spec,
                V_dg,
                quadrature_target_space=quadrature_plot_space,
                quadrature_association="cell",
            )
        if component_kind == "tensor":
            return self._project_tensor_field_component(
                field_name,
                component,
                field_spec,
                V_dg,
                quadrature_target_space=quadrature_plot_space,
                quadrature_association="cell",
            )
        return self._project_scalar_field(
            field_name,
            field_spec,
            V_dg,
            quadrature_target_space=quadrature_plot_space,
            quadrature_association="cell",
        )

    def Get_Field_Space_For_BC(self, field_name: str, component: int = None):
        """
        Get function space information for creating boundary conditions.

        Returns ``(field_space, component or None)``. Fields solved in a
        monolithic step keep their own space, which is the corresponding
        block of the step's block layout.

        Parameters
        ----------
        field_name : str
            Name of the field to constrain.
        component : int, optional
            Component index for vector-valued fields.
        """
        return (self._get_field_or_raise(field_name).function_space, component)

    def Copy_Field(self, source_field_name: str, dest_field_name: str, domain: str = None) -> None:
        """
        Copy data from one field to another. MPI-aware: Updates ghost values
        after copying.

        The helper accepts any combination of standard and quadrature-backed
        fields, as long as the target discretization can accept interpolation
        from the source representation. With ``domain``, only the DOFs owned
        by that domain are copied (see :meth:`Get_Field_Dofs`); this requires
        both fields to share the same discretization and support.
        """
        try:
            source_view = self.Get_Field_For_Output(source_field_name)
        except ValueError as exc:
            raise ValueError(
                f"Source field '{source_field_name}' not found in mesh."
            ) from exc

        try:
            dest_field = self._get_field_or_raise(dest_field_name)
        except ValueError as exc:
            raise ValueError(
                f"Destination field '{dest_field_name}' not found in mesh."
            ) from exc

        source_result = source_view.field
        source_element = source_result.function_space.element
        dest_element = dest_field.function_space.element

        dofs = self.Get_Field_Dofs(dest_field_name, domain)
        if dofs is not None:
            def storage(definition):
                return (definition.family, definition.degree, definition.value_shape, definition.quadrature_degree)

            same_layout = (
                storage(self.field_definitions[source_field_name]) == storage(self.field_definitions[dest_field_name])
                and source_result.function_space.mesh is dest_field.function_space.mesh
            )
            if not same_layout:
                raise ValueError(
                    f"Copying '{source_field_name}' to '{dest_field_name}' on domain '{domain}' "
                    "requires both fields to share the same discretization and support."
                )
            dest_field.x.array[dofs] = source_result.x.array[dofs]
            dest_field.x.scatter_forward()
            return

        def _is_quadrature(element) -> bool:
            family = getattr(element, "family", None)
            if isinstance(family, str):
                return family == "Quadrature"
            basix = None
            try:
                basix = element.basix_element
            except RuntimeError:
                basix = None
            if basix is not None:
                basix_family = getattr(basix, "family", None)
                family_name = getattr(basix_family, "name", None)
                if isinstance(family_name, str):
                    return family_name.lower() == "quadrature"
                if isinstance(basix_family, str):
                    return basix_family == "Quadrature"
            return False

        source_is_quad = _is_quadrature(source_element) or "Quadrature" in self.field_type.get(source_field_name, "")
        dest_is_quad = _is_quadrature(dest_element) or "Quadrature" in self.field_type.get(dest_field_name, "")
        if source_is_quad or dest_is_quad:
            if source_result.function_space == dest_field.function_space:
                dest_field.x.array[:] = source_result.x.array
            else:
                interp_points = dest_field.function_space.element.interpolation_points
                expr = fem.Expression(source_result, interp_points)
                dest_field.interpolate(expr)
        else:
            dest_field.interpolate(source_result)

        dest_field.x.scatter_forward()

    def Set_From_Expression(self, field_name: str, expression, domain: str = None) -> None:
        """
        Evaluate a UFL expression and set the field to the result.
        MPI-aware: Updates ghost values after interpolation.

        The expression may be defined on the parent mesh while the field
        lives on a submesh (or vice versa). With ``domain``, only the cells of
        that (sub)domain are set; nodes shared with neighbouring domains take
        the value of whichever domain is set last.
        """
        field = self._get_field_or_raise(field_name)
        self._interpolate_on_support(field, field_name, expression, domain)
        field.x.scatter_forward()

    def Evaluate_At_Field_Points(self, field_name: str, expression) -> np.ndarray:
        """Evaluate an expression at the quadrature points of a quadrature field.

        The expression is evaluated on the field's own support, so the result
        lines up point by point with ``field.x.array`` also when the field
        lives on a submesh and the expression on the parent mesh.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_local_points, value_size)`` with the
            flattened value of ``expression`` at every local point.
        """
        definition = self.field_definitions[field_name]
        if definition.family != "Quadrature":
            raise ValueError(f"Field '{field_name}' is not a quadrature field.")
        field = self._get_field_or_raise(field_name)
        value_shape = tuple(expression.ufl_shape)
        q_element = quadrature_element(
            field.function_space.mesh.basix_cell(),
            value_shape=value_shape,
            degree=definition.quadrature_degree,
        )
        target = fem.Function(fem.functionspace(field.function_space.mesh, q_element))
        self._interpolate_on_support(target, field_name, expression, None)
        target.x.scatter_forward()
        value_size = int(np.prod(value_shape)) if value_shape else 1
        return target.x.array.reshape(-1, value_size)

    def Zero_Field(self, field_name: str, domain: str = None) -> None:
        """Set a field to zero, optionally only on the DOFs owned by ``domain``."""
        self.Constant_Field(field_name, 0.0, domain=domain)

    def Constant_Field(self, field_name: str, value: float, domain: str = None) -> None:
        """Set a field to a constant, optionally only on the DOFs owned by ``domain``.

        See :meth:`Get_Field_Dofs` for which DOFs a domain owns.
        """
        field = self._get_field_or_raise(field_name)
        dofs = self.Get_Field_Dofs(field_name, domain)
        if dofs is None:
            field.x.array[:] = value
        else:
            field.x.array[dofs] = value
            field.x.scatter_forward()

    def Is_DG_Field(self, field_name: str) -> bool:
        """Return whether the named field uses a discontinuous Galerkin function space."""
        return self.field_type.get(field_name, "").startswith("DG")

    def Get_Field_Support(self, field_name: str) -> FieldSupport:
        """Return the support (cells and carrying mesh) of a registered field."""
        self._get_field_or_raise(field_name)
        domains = tuple(self._field_domains.get(field_name, [FULL_DOMAIN]))
        support_key = self._field_support_keys.get(field_name)
        if support_key is None:
            return FieldSupport(
                domain_names=domains,
                tags=tuple(sorted(getattr(self, "domain_markers", {}).values())),
                mesh=self.mesh,
            )
        if support_key == (_GLOBAL,):
            return FieldSupport(domain_names=domains, tags=(), mesh=self._get_global_mesh(),
                                parent_cells=np.empty(0, dtype=np.int32), is_global=True)
        submesh, cell_map, parent_cells = self._get_submesh(support_key)
        is_interface = support_key[0] == _FACETS
        return FieldSupport(
            domain_names=domains,
            tags=support_key[1:] if is_interface else support_key,
            mesh=submesh,
            cell_map=cell_map,
            parent_cells=parent_cells,
            is_interface=is_interface,
        )

    def Get_Domain_Integrator(self, domain: str = FULL_DOMAIN):
        """Return the cell integration measure ``dx`` over one (sub)domain.

        Models should integrate their weak forms with this measure, so that
        a model restricted to e.g. ``"domain:centre"`` only contributes
        there. For the full domain this is the plain ``dx`` of the mesh. On
        axisymmetric meshes, it multiplies every integrand by the radius.
        """
        measure = Measure("dx", domain=self.mesh)
        if domain != FULL_DOMAIN:
            tags = self.get_domain_tags(domain)
            if not set(tags) >= set(self.domain_markers.values()):
                measure = Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)(tags)
        return WeightedMeasure(measure, self.Get_Integration_Weight()) if self.axisymmetric else measure

    def Get_Field_Dofs(self, field_name: str, domain: str = None) -> Optional[np.ndarray]:
        """Return the local DOF indices of a field owned by one (sub)domain.

        Several models may share a field, each on its own (sub)domain (e.g.
        displacements of elastic models on ``"domain:left"`` and
        ``"domain:centre"``). To let each model update its part of such a
        field exactly once, nodes on the interface between two domains are
        owned by the domain that registered the field first; the other
        domain does not own them.

        Returns
        -------
        numpy.ndarray or None
            Local (owned and ghost) indices into ``field.x.array``, unrolled
            by the block size; ``None`` when ``domain`` is ``None`` or covers
            the field's whole support, meaning "all DOFs".
        """
        if domain is None:
            return None
        cache_key = (field_name, domain)
        if cache_key in self._field_dofs_cache:
            return self._field_dofs_cache[cache_key]

        support = self.Get_Field_Support(field_name)
        if support.is_global:
            return None
        if support.is_interface:
            self._require_own_interface(field_name, support, domain)
            self._field_dofs_cache[cache_key] = None
            return None
        domain_tags = set(self.get_domain_tags(domain)) if domain != FULL_DOMAIN else set(support.tags)
        if domain_tags >= set(support.tags):
            self._field_dofs_cache[cache_key] = None
            return None

        registered = list(support.domain_names)
        earlier = registered[: registered.index(domain)] if domain in registered else []
        dofs = self._domain_cell_dofs(field_name, domain)
        if earlier:
            claimed = np.concatenate([self._domain_cell_dofs(field_name, other) for other in earlier])
            dofs = np.setdiff1d(dofs, claimed, assume_unique=False)
        self._field_dofs_cache[cache_key] = dofs
        return dofs

    def Get_Interior_Facet_Integrator(self):
        """Return a UFL interior-facet integration measure ``dS`` for the mesh."""
        import ufl
        return ufl.Measure("dS", domain=self.mesh)

    def Interface_Integral(self, integrand, interface: str, normal_from: Optional[str] = None):
        """Return the Gauss-integrated form ``integrand * dS`` over an interior interface.

        ``interface`` is an interior facet group between two domains (e.g.
        ``"interface"`` of ``TwoSquared``). The integrand is written with
        unrestricted fields: each field (or test/trial function) defined on
        only one side is evaluated from that side, and fields defined on both
        sides or on the interface itself are evaluated from the
        ``normal_from`` side, which is exact for continuous fields.
        ``ufl.FacetNormal`` is the unit normal pointing out of
        ``normal_from`` into the other side. Explicit restrictions such as
        ``c("+")`` are left untouched ("+" is the ``normal_from`` side).

        Parameters
        ----------
        integrand : ufl.core.expr.Expr
            Scalar UFL expression of fields, test and trial functions.
        interface : str
            Facet group of interior facets between two domains.
        normal_from : str, optional
            Domain on one side of every interface facet. Defaults to the
            first adjacent leaf domain in the generator's registration order.

        Returns
        -------
        ufl.Form
            Form to be compiled with :meth:`Compile_Form`.

        Raises
        ------
        ValueError
            If the facets are not all interior, ``normal_from`` is not on
            exactly one side of every facet, or a field lives on neither side.

        Example
        -------
        ``mesh.Interface_Integral(k * (c - h) * q, "interface")`` couples
        ``c`` on ``"domain:left"`` with ``h`` and its test function ``q`` on
        ``"domain:right"``.
        """
        entities, plus_cells, minus_cells = self._interface_integration_entities(interface, normal_from)
        restrictor = _InterfaceSideRestrictor(lambda terminal: self._interface_side(terminal, interface, plus_cells, minus_cells))
        restricted = map_expr_dag(restrictor, apply_derivatives(apply_algebra_lowering(integrand)))
        dS = Measure("dS", domain=self.mesh, subdomain_data=[(1, entities.ravel())])
        return restricted * dS(1)

    def Add_Global_Field_Placeholders(self, blocks, step: int, role: str):
        """Give global fields of a monolithic step a (zero) weak-form block if they lack one.

        Global fields usually only receive algebraic contributions in
        ``Add_KF``; block assembly still needs their diagonal block (role
        ``"K"``) and residual block (role ``"F"``). Used by Physics when it
        compiles ``ufl.extract_blocks`` output; other blocks are unchanged.
        """
        field_names = self.mixed_spaces[step]["field_names"]
        n = len(field_names)
        # extract_blocks omits trailing blocks without any form.
        if role == "F":
            blocks = list(blocks) + [None] * (n - len(blocks))
        else:
            blocks = [list(row) + [None] * (n - len(row)) for row in blocks] + [[None] * n for _ in range(n - len(blocks))]
        for i, name in enumerate(field_names):
            if not self.Is_Global_Field(name):
                continue
            space = self.fields[name].function_space
            zero, dx_global = fem.Constant(space.mesh, 0.0), Measure("dx", domain=space.mesh)
            if role == "F" and blocks[i] is None:
                blocks[i] = zero * TestFunction(space) * dx_global
            elif role != "F" and blocks[i][i] is None:
                blocks[i][i] = zero * TestFunction(space) * TrialFunction(space) * dx_global
        return blocks

    def update_step_from_vector(self, du, step: int) -> None:
        """
        Add a solution increment vector to the solution fields of one step.

        This is the mesh-side companion to the linear/nonlinear solver
        update logic. ``du`` uses the step's layout: one block per field for
        monolithic steps (see :meth:`get_step_block_offsets`), or the field's
        own layout for single-field steps. Only owned entries are read; ghost
        values are refreshed afterwards.
        """
        field_names = self.step_fields.get(step, [])
        if not field_names:
            raise ValueError(f"No fields registered for step {step}.")
        if len(field_names) > 1 and self.mixed_spaces.get(step) is None:
            raise ValueError(
                f"Step {step} has {len(field_names)} fields but no block layout. "
                "Call finalize_test_functions() before updating this step."
            )

        offsets = self.get_step_block_offsets(step)
        with du.localForm() as loc_du:
            increment = loc_du.array_r
            for field_name, start, end in zip(field_names, offsets, offsets[1:]):
                field = self.fields[field_name]
                field.x.array[: end - start] += increment[start:end]
                field.x.scatter_forward()

    def Compile_Form(self, form):
        """Compile a UFL form for assembly on this mesh.

        Forms that involve the test/trial functions of a monolithic step
        (several fields) are split into one block per field before
        compilation. The entity maps of all submeshes are always supplied, so
        forms may mix fields on different (sub)meshes.

        Returns
        -------
        dolfinx.fem.Form or list
            A compiled form, or a (nested) list of compiled block forms when
            ``form`` contains block arguments of a monolithic step.
        """
        entity_maps = self.get_entity_maps() or None
        if any(argument.part() is not None for argument in form.arguments()):
            return fem.form(extract_blocks(form), entity_maps=entity_maps)
        return fem.form(form, entity_maps=entity_maps)

    def get_step_global_indices(self, step: int, local_indices) -> np.ndarray:
        """Convert local step-vector indices into global matrix/vector indices.

        ``local_indices`` index the local (owned and ghost) array of a step's
        residual vector, e.g. as returned by :meth:`Get_Weights_Dofs`. The
        result can be used with ``K.setValues`` and ``F.setValues``. Use this
        instead of ``setValuesLocal``: for monolithic steps, the local
        numbering of DOLFINx block matrices differs from that of the step
        vector (owned and ghost entries are interleaved per block).
        """
        layout = self._get_step_global_layout(step)  # cached in finalize_test_functions
        local_indices = np.asarray(local_indices, dtype=np.int64)
        result = np.empty(local_indices.shape, dtype=np.int64)
        owned = local_indices < layout["owned_size"]
        result[owned] = layout["rank_start"] + local_indices[owned]
        result[~owned] = layout["ghost_globals"][local_indices[~owned] - layout["owned_size"]]
        return result.astype(PETSc.IntType)

    def get_entity_maps(self) -> list:
        """Return the entity maps needed to compile forms that mix (sub)meshes.

        Pass the result as ``entity_maps`` to ``dolfinx.fem.form`` for any
        form that combines fields living on different (sub)meshes.
        """
        return list(self._submesh_entity_maps.values())

    def get_step_block_offsets(self, step: int) -> list[int]:
        """Return the local owned offsets of each field block in a step vector.

        Entry ``i`` is where field ``step_fields[step][i]`` starts within the
        owned part of the step's solution/residual vector, and the last entry
        is the total owned size. Single-field steps return ``[0, n_owned]``.
        """
        offsets = [0]
        for field_name in self.step_fields.get(step, []):
            dofmap = self.fields[field_name].function_space.dofmap
            offsets.append(offsets[-1] + dofmap.index_map.size_local * dofmap.index_map_bs)
        return offsets

    def finalize_test_functions(self) -> None:
        """
        Finalize test function creation by building mixed spaces where needed.
        This should be called after all models have registered their test functions.

        After this method runs, steps containing more than one field have a
        corresponding entry in :attr:`mixed_spaces`, and subsequent calls to
        ``Get_Field``, ``Get_Trial_Function``, or ``update_step_from_vector``
        use that mixed representation.
        """
        for step in self.step_fields.keys():
            self._build_mixed_space_for_step(step)
            # Built collectively here, so later index conversions need no
            # communication (models may convert on some ranks only).
            self._get_step_global_layout(step)
        self._test_functions_finalized = True

    def get_total_dofcount(self) -> int:
        """
        Return the total number of degrees of freedom being solved for in each step.
        MPI-aware: Returns total DOFs across all ranks.
        """
        from femcubes.core.utils.mpi_utils import comm

        totalDOFS_global = 0
        for step in sorted(self.step_fields.keys()):
            field_names = self.step_fields[step]
            if not field_names:
                continue

            # Count local DOFs for this step
            local_total_dofs = 0
            for field_name in field_names:
                if field_name not in self.fields:
                    continue
                field = self.fields[field_name]
                local_base_dofs = field.function_space.dofmap.index_map.size_local
                value_shape = field.function_space.element.value_shape
                num_components = 1 if len(value_shape) == 0 else value_shape[0]
                local_field_dofs = local_base_dofs * num_components
                local_total_dofs += local_field_dofs

            # Sum up total DOFs across all ranks
            global_total_dofs = comm.allreduce(local_total_dofs)

            totalDOFS_global += global_total_dofs

        return totalDOFS_global

    def print_dofs_info(self) -> None:
        """
        Print the number of degrees of freedom being solved for in each step.
        MPI-aware: Shows local DOFs (rank 0) and total DOFs across all ranks.
        """
        from femcubes.core.utils.mpi_utils import comm, rank
        
        mprint("Degrees of freedom per solution step:")
        
        if not self.step_fields:
            mprint("\tNo solution steps defined.")
            return
        
        totalDOFS_local = 0
        totalDOFS_global = 0
        for step in sorted(self.step_fields.keys()):
            field_names = self.step_fields[step]
            
            if not field_names:
                mprint(f"\tStep {step}: No fields")
                continue
            
            # Count local DOFs for this step
            local_total_dofs = 0
            field_dof_info = []
            
            for field_name in field_names:
                if field_name not in self.fields:
                    continue
                
                field = self.fields[field_name]
                # Get local DOF count (owned by this rank)
                local_base_dofs = field.function_space.dofmap.index_map.size_local
                
                # For vector fields, multiply by number of components
                value_shape = field.function_space.element.value_shape
                num_components = 1 if len(value_shape) == 0 else value_shape[0]
                local_field_dofs = local_base_dofs * num_components
                
                # Get global DOF count (sum across all ranks)
                global_base_dofs = field.function_space.dofmap.index_map.size_global
                global_field_dofs = global_base_dofs * num_components
                
                local_total_dofs += local_field_dofs
                
                if num_components > 1:
                    field_dof_info.append({
                        'name': field_name,
                        'local': local_field_dofs,
                        'global': global_field_dofs,
                        'desc': f"{num_components} components × {local_base_dofs} local nodes ({global_base_dofs} global)"
                    })
                else:
                    field_dof_info.append({
                        'name': field_name,
                        'local': local_field_dofs,
                        'global': global_field_dofs,
                        'desc': None
                    })
            
            # Sum up total DOFs across all ranks
            global_total_dofs = comm.allreduce(local_total_dofs)
            
            totalDOFS_local += local_total_dofs
            totalDOFS_global += global_total_dofs

            # Print step information
            mprint(f"\tStep {step}: {local_total_dofs} DOFs (local rank {rank}), {global_total_dofs} DOFs (total)")
            for field_info in field_dof_info:
                support = self.Get_Field_Support(field_info['name'])
                support_text = "" if support.is_full else (" [global]" if support.is_global else f" [on {', '.join(support.domain_names)}]")
                if field_info['desc']:
                    mprint(f"\t\t{field_info['name']}: {field_info['local']} local, {field_info['global']} global ({field_info['desc']}){support_text}")
                else:
                    mprint(f"\t\t{field_info['name']}: {field_info['local']} local, {field_info['global']} global{support_text}")

            if step == sorted(self.step_fields.keys())[-1]:
                mprint(f"\tTotal: {totalDOFS_local} DOFs (local rank {rank}), {totalDOFS_global} DOFs (total)")

    def get_global_solution_dof_count(self) -> int:
        """Return the global number of registered solution degrees of freedom."""

        return self._count_global_solution_dofs(
            self.fields,
            self.step_fields,
        )

    # ------------------------------------------------------------------
    # Internal Helpers: Field Resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _count_global_solution_dofs(fields, step_fields) -> int:
        """PRIVATE: Count global owned DOFs for a solution-field registry."""

        total = 0
        counted_fields = set()
        for field_names in step_fields.values():
            for field_name in field_names:
                if field_name in counted_fields or field_name not in fields:
                    continue
                dofmap = fields[field_name].function_space.dofmap
                total += int(
                    dofmap.index_map.size_global * dofmap.index_map_bs
                )
                counted_fields.add(field_name)
        return total

    def _get_field_or_raise(self, field_name: str) -> fem.Function:
        """PRIVATE: Return a registered field or raise a consistent error."""
        field = self.fields.get(field_name, None)
        if field is None:
            raise ValueError(f"Field '{field_name}' not found in mesh.")
        return field

    def _get_field_component_count(self, field: fem.Function) -> int:
        """PRIVATE: Return the number of scalar components carried by a field."""
        value_shape = field.function_space.element.value_shape
        if len(value_shape) == 0:
            return 1

        comp_count = 1
        for size in value_shape:
            comp_count *= size
        return comp_count

    def _resolve_field_reference(self, field_name: str, step: int = None) -> FieldOutputView:
        """PRIVATE: Resolve a registered field to its concrete output view."""
        field = self._get_field_or_raise(field_name)
        return FieldOutputView(
            field_name=field_name,
            field=field,
            comp_count=self._get_field_component_count(field),
        )

    def _interpolate_on_support(self, target: fem.Function, field_name: str, expression, domain: str = None) -> None:
        """PRIVATE: Interpolate an expression into a function on a field's support.

        ``target`` lives on the same (sub)mesh as ``field_name``. The
        expression may be defined on that mesh, on the parent mesh, or mix
        the parent mesh with one submesh (then it is evaluated on the
        submesh, which must contain the target cells).
        """
        self._require_spatial_field(field_name, "be set from an expression")
        target_mesh = target.function_space.mesh
        target_key = self._mesh_key(target_mesh)
        expression, eval_key = self._single_mesh_expression(expression, target_key)
        expr = fem.Expression(expression, target.function_space.element.interpolation_points, comm=target_mesh.comm)
        field_cells, parent_cells = self._get_domain_cell_pairs(field_name, domain)
        if self.Get_Field_Support(field_name).is_interface and eval_key != target_key:
            raise ValueError(
                f"Interface field '{field_name}' can only be set from expressions on its own interface mesh."
            )

        if eval_key == target_key:
            if field_cells is None:
                target.interpolate(expr)
            else:
                target.interpolate(expr, cells0=field_cells)
        elif eval_key is None:
            # Parent-mesh expression evaluated into a submesh function.
            target.interpolate(expr, cells0=parent_cells, cells1=field_cells)
        elif target_key is None:
            # Submesh expression evaluated into a parent-mesh function: the
            # target cells must all lie in that submesh.
            _submesh, cell_map, _sub_parent_cells = self._get_submesh(eval_key)
            if parent_cells is None:
                parent_cells = np.arange(target_mesh.topology.index_map(target_mesh.topology.dim).size_local
                                         + target_mesh.topology.index_map(target_mesh.topology.dim).num_ghosts, dtype=np.int32)
            eval_cells = np.asarray(cell_map.sub_topology_to_topology(parent_cells, inverse=True), dtype=np.int32)
            if np.any(eval_cells < 0):
                raise ValueError(
                    f"Cannot evaluate an expression defined on a subdomain into field '{field_name}' "
                    f"outside that subdomain; restrict the evaluation with a matching domain."
                )
            target.interpolate(expr, cells0=eval_cells, cells1=parent_cells)
        else:
            raise ValueError(
                f"Cannot evaluate an expression on one subdomain into field '{field_name}' on another subdomain."
            )

    def _mesh_key(self, mesh) -> Optional[Tuple[int, ...]]:
        """PRIVATE: Return ``None`` for the parent mesh, or the support key of a submesh."""
        cpp_mesh = getattr(mesh, "_cpp_object", mesh)
        if cpp_mesh is self.mesh._cpp_object:
            return None
        for key, (submesh, _cell_map, _parent_cells) in self._submeshes.items():
            if submesh._cpp_object is cpp_mesh:
                return key
        raise ValueError("Expression is defined on a mesh unknown to this mesh wrapper.")

    def _single_mesh_expression(self, expression, target_key):
        """PRIVATE: Return ``(expression, mesh_key)`` with the expression on one mesh.

        DOLFINx evaluates an expression on the mesh of its function
        coefficients, or on the target's mesh when it has none (geometry is
        identical on the parent mesh and its submeshes). Expressions mixing
        parent-mesh and submesh coefficients are rewritten onto the submesh:
        parent coefficients and constants are replaced by submesh copies that
        are refreshed on every call. Copying is exact, as the copies use the
        same elements on the same cells.
        """
        mesh_keys = {self._mesh_key(c.function_space.mesh) for c in extract_coefficients(expression)}
        if not mesh_keys:
            return expression, target_key
        if len(mesh_keys) == 1:
            return expression, next(iter(mesh_keys))
        submesh_keys = mesh_keys - {None}
        if len(submesh_keys) > 1:
            raise ValueError("Expressions may combine the parent mesh with at most one submesh.")
        submesh_key = next(iter(submesh_keys))
        submesh, _cell_map, parent_cells = self._get_submesh(submesh_key)
        sub_cells = np.arange(parent_cells.size, dtype=np.int32)

        replacements = {}
        for coefficient in extract_coefficients(expression):
            if self._mesh_key(coefficient.function_space.mesh) is None:
                replacements[coefficient] = self._refreshed_submesh_proxy(coefficient, submesh_key, submesh, parent_cells, sub_cells)
        for constant in extract_constants(expression):
            if self._mesh_key(constant.ufl_domain().ufl_cargo()) is None:
                proxy = self._submesh_proxies.get((id(constant), submesh_key))
                if proxy is None:
                    proxy = fem.Constant(submesh, np.asarray(constant.value).copy())
                    self._submesh_proxies[(id(constant), submesh_key)] = proxy
                proxy.value = constant.value
                replacements[constant] = proxy
        rewritten = ufl_replace(expression, replacements)
        if any(domain is not None and self._mesh_key(domain.ufl_cargo()) != submesh_key for domain in extract_domains(rewritten)):
            raise ValueError(
                "Expression mixing a subdomain with the parent mesh may only use parent-mesh "
                "coefficients and constants, not parent-mesh geometric quantities."
            )
        return rewritten, submesh_key

    def _refreshed_submesh_proxy(self, coefficient, submesh_key, submesh, parent_cells, sub_cells):
        """PRIVATE: Return a submesh copy of a parent-mesh function with current values."""
        proxy = self._submesh_proxies.get((id(coefficient), submesh_key))
        if proxy is None:
            proxy = fem.Function(fem.functionspace(submesh, coefficient.function_space.ufl_element()))
            self._submesh_proxies[(id(coefficient), submesh_key)] = proxy
        if coefficient.function_space.ufl_element().family_name == "quadrature":
            # Quadrature data cannot be interpolated; copy it cell by cell.
            bs = coefficient.function_space.dofmap.index_map_bs
            parent_dofs = coefficient.function_space.dofmap.list[parent_cells].ravel()
            sub_dofs = proxy.function_space.dofmap.list[sub_cells].ravel()
            for component in range(bs):
                proxy.x.array[sub_dofs * bs + component] = coefficient.x.array[parent_dofs * bs + component]
        else:
            proxy.interpolate(coefficient, cells0=parent_cells, cells1=sub_cells)
        return proxy

    def _get_domain_cell_pairs(self, field_name: str, domain: str = None):
        """PRIVATE: Return matching ``(field_mesh_cells, parent_cells)`` of a domain.

        Cells are local (owned and ghost) indices on the field's own mesh and
        on the parent mesh. ``(None, None)`` means "all cells" of a field on
        the full mesh.
        """
        support = self._require_spatial_field(field_name, "be evaluated on cells")
        if support.is_interface:
            self._require_own_interface(field_name, support, domain)
            return np.arange(support.parent_cells.size, dtype=np.int32), support.parent_cells
        covers_support = domain is None or domain == FULL_DOMAIN or set(self.get_domain_tags(domain)) >= set(support.tags)
        if covers_support:
            if support.is_full:
                return None, None
            return np.arange(support.parent_cells.size, dtype=np.int32), support.parent_cells

        parent_cells = np.asarray(self.get_domain_cells(domain), dtype=np.int32)
        if support.is_full:
            return parent_cells, parent_cells
        field_cells = np.asarray(support.cell_map.sub_topology_to_topology(parent_cells, inverse=True), dtype=np.int32)
        inside = field_cells >= 0
        return field_cells[inside], parent_cells[inside]

    def _bc_entities_on_field_mesh(self, field_name: str, boundary_name: str):
        """PRIVATE: Return ``(dim, entities)`` of a field's mesh constrained by a boundary BC.

        For cell fields these are the boundary facets on the field's mesh. For
        interface fields, a BC on the interface's own facet group constrains
        the whole field; a BC on another facet group constrains the field
        where the interface meets it (e.g. its end points).
        """
        support = self._require_spatial_field(field_name, "have boundary conditions")
        boundary_facets = self.get_boundary_facets(boundary_name)
        if not support.is_interface:
            return self.mesh.topology.dim - 1, self._facets_on_field_mesh(field_name, boundary_facets)

        sub_dim = support.mesh.topology.dim
        if set(self.get_boundary_tags(boundary_name)) == set(support.tags):
            support.mesh.topology.create_connectivity(sub_dim, sub_dim)
            cell_map = support.mesh.topology.index_map(sub_dim)
            return sub_dim, np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)

        # Entities of dimension tdim - 2 shared by the boundary and the interface.
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim - 1, tdim - 2)
        facet_to_entity = self.mesh.topology.connectivity(tdim - 1, tdim - 2)
        def closure(facets):
            return {int(e) for facet in facets for e in facet_to_entity.links(facet)}
        shared = sorted(closure(boundary_facets) & closure(support.parent_cells))
        if not shared:
            return sub_dim - 1, np.empty(0, dtype=np.int32)

        # Identify the shared entities on the interface mesh through their vertices.
        vertex_map = self._submesh_vertex_maps[self._field_support_keys[field_name]]
        if tdim - 2 == 0:
            parent_vertices = np.asarray(shared, dtype=np.int32)[:, None]
        else:
            self.mesh.topology.create_connectivity(tdim - 2, 0)
            entity_to_vertex = self.mesh.topology.connectivity(tdim - 2, 0)
            parent_vertices = np.array([entity_to_vertex.links(e) for e in shared], dtype=np.int32)
        sub_vertices = np.asarray(vertex_map.sub_topology_to_topology(parent_vertices.ravel(), inverse=True)).reshape(parent_vertices.shape)
        support.mesh.topology.create_connectivity(sub_dim - 1, sub_dim)
        if sub_dim - 1 == 0:
            return 0, sub_vertices[:, 0].astype(np.int32)
        support.mesh.topology.create_connectivity(sub_dim - 1, 0)
        sub_entity_to_vertex = support.mesh.topology.connectivity(sub_dim - 1, 0)
        num_sub_entities = support.mesh.topology.index_map(sub_dim - 1).size_local + support.mesh.topology.index_map(sub_dim - 1).num_ghosts
        lookup = {frozenset(sub_entity_to_vertex.links(e)): e for e in range(num_sub_entities)}
        entities = [lookup[frozenset(v)] for v in sub_vertices if frozenset(v) in lookup]
        return sub_dim - 1, np.asarray(entities, dtype=np.int32)

    def _facets_on_field_mesh(self, field_name: str, parent_facets: np.ndarray) -> np.ndarray:
        """PRIVATE: Map local parent-mesh facets onto the field's (sub)mesh facets.

        Each facet is found through an adjacent cell of the field's support
        and its local facet index, which the submesh cells share with their
        parent cells.
        """
        support = self._require_spatial_field(field_name, "act on boundaries")
        if support.is_full:
            return parent_facets
        if support.is_interface:
            raise ValueError(f"Interface field '{field_name}' has no facets of the parent mesh.")  # pragma: no cover - callers use _bc_entities_on_field_mesh
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        self.mesh.topology.create_connectivity(tdim, tdim - 1)
        support.mesh.topology.create_connectivity(tdim, tdim - 1)
        support.mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_to_cell = self.mesh.topology.connectivity(tdim - 1, tdim)
        parent_cell_to_facet = self.mesh.topology.connectivity(tdim, tdim - 1)
        sub_cell_to_facet = support.mesh.topology.connectivity(tdim, tdim - 1)

        sub_facets = np.empty(len(parent_facets), dtype=np.int32)
        for i, facet in enumerate(parent_facets):
            cells = facet_to_cell.links(facet)
            sub_cells = np.asarray(support.cell_map.sub_topology_to_topology(cells, inverse=True))
            inside = np.flatnonzero(sub_cells >= 0)
            if inside.size == 0:
                raise ValueError(
                    f"Boundary facets lie outside the support {list(support.domain_names)} of field '{field_name}'."
                )
            local_facet = np.flatnonzero(parent_cell_to_facet.links(cells[inside[0]]) == facet)[0]
            sub_facets[i] = sub_cell_to_facet.links(sub_cells[inside[0]])[local_facet]
        return sub_facets

    @staticmethod
    def _require_own_interface(field_name: str, support: FieldSupport, domain: Optional[str]) -> None:
        """PRIVATE: Reject restricting an interface field to anything but its own interface."""
        if domain is not None and domain not in support.domain_names:
            raise ValueError(
                f"Interface field '{field_name}' can only be restricted to its own interface "
                f"{list(support.domain_names)}, not to '{domain}'."
            )

    def _domain_cell_dofs(self, field_name: str, domain: str) -> np.ndarray:
        """PRIVATE: Return the unrolled local DOFs of a field on the cells of one domain."""
        field_cells, _parent_cells = self._get_domain_cell_pairs(field_name, domain)
        dofmap = self.fields[field_name].function_space.dofmap
        if field_cells is None:
            size = dofmap.index_map.size_local + dofmap.index_map.num_ghosts
            field_cells_dofs = np.arange(size, dtype=np.int32)
        else:
            field_cells_dofs = np.unique(dofmap.list[field_cells].ravel())
        bs = dofmap.index_map_bs
        if bs == 1:
            return field_cells_dofs.astype(np.int32)
        return (field_cells_dofs[:, None] * bs + np.arange(bs)).ravel().astype(np.int32)

    def _step_vector_indices(self, step: int, field_name: str, field_dofs) -> np.ndarray:
        """PRIVATE: Map local (owned or ghost) field DOFs to local step-vector indices.

        The step vector holds the owned entries of all field blocks, followed
        by the ghost entries of all blocks.
        """
        names = list(self.step_fields[step])
        block = names.index(field_name)
        dofmaps = [self.fields[name].function_space.dofmap for name in names]
        owned = [dofmap.index_map.size_local * dofmap.index_map_bs for dofmap in dofmaps]
        ghosts = [dofmap.index_map.num_ghosts * dofmap.index_map_bs for dofmap in dofmaps]
        field_dofs = np.asarray(field_dofs, dtype=np.int64)
        return np.where(
            field_dofs < owned[block],
            sum(owned[:block]) + field_dofs,
            sum(owned) + sum(ghosts[:block]) + field_dofs - owned[block],
        ).astype(np.int32)

    def _get_step_global_layout(self, step: int) -> Dict[str, Any]:
        """PRIVATE: Return cached global numbering data of one step vector.

        Globally, every rank owns one contiguous range holding its owned
        entries of all field blocks in order. A ghost entry is located on its
        owning rank through the field's index map.
        """
        # Mesh installations replace this cache, so the step identifies a layout.
        cache_key = step
        cached = self._step_global_layouts.get(cache_key)
        if cached is not None:
            return cached

        comm = self.mesh.comm
        dofmaps = [self.fields[name].function_space.dofmap for name in self.step_fields.get(step, [])]
        owned_sizes = [dofmap.index_map.size_local * dofmap.index_map_bs for dofmap in dofmaps]
        owned_size = int(sum(owned_sizes))
        rank_start = comm.exscan(owned_size) or 0
        block_starts = np.cumsum([0] + owned_sizes[:-1]).astype(np.int64)
        # Per rank: where each block starts globally, and the first field DOF it owns.
        all_block_starts = comm.allgather(rank_start + block_starts)
        all_field_starts = comm.allgather([dofmap.index_map.local_range[0] for dofmap in dofmaps])

        ghost_globals = []
        for block, dofmap in enumerate(dofmaps):
            bs = dofmap.index_map_bs
            ghosts = dofmap.index_map.ghosts
            owners = dofmap.index_map.owners
            node_globals = np.array(
                [all_block_starts[owner][block] + (ghost - all_field_starts[owner][block]) * bs for ghost, owner in zip(ghosts, owners)],
                dtype=np.int64,
            )
            ghost_globals.append((node_globals[:, None] + np.arange(bs)).ravel())

        layout = {
            "owned_size": owned_size,
            "rank_start": rank_start,
            "ghost_globals": np.concatenate(ghost_globals) if ghost_globals else np.empty(0, dtype=np.int64),
        }
        self._step_global_layouts[cache_key] = layout
        return layout

    def _get_step_block_index(self, field_name: str, step: int) -> Optional[int]:
        """PRIVATE: Return a field's block index in a monolithic step, if any."""
        block_layout = self.mixed_spaces.get(step)
        if block_layout is None or field_name not in block_layout["field_names"]:
            return None
        return block_layout["field_names"].index(field_name)

    # ------------------------------------------------------------------
    # Internal Helpers: Construction
    # ------------------------------------------------------------------

    def _invalidate_dof_caches(self) -> None:
        """PRIVATE: Destroy and clear all mesh-dependent DOF-side caches."""

        self._dg_spaces.clear()
        self._cg_spaces.clear()
        self._quadrature_spaces.clear()
        self._field_dofs_cache.clear()
        # Submeshes are rebuilt lazily from the (transferred) markers; replace
        # rather than clear the dictionaries so a rollback can restore them.
        self._submeshes = {}
        self._submesh_entity_maps = {}
        self._submesh_vertex_maps = {}
        self._submesh_proxies = {}
        self._step_global_layouts = {}
        self._function_space_cache_epoch = None
        self._clear_output_projection_caches()
        self._output_projection_cache_epoch = None

    def _clear_output_projection_caches(self) -> None:
        """PRIVATE: Destroy native projection solvers and clear output caches."""

        destroy_errors = []
        destroyed = set()
        for _test, mass_matrix, ksp in self._output_mass_projection_cache.values():
            for label, native_object in (("KSP", ksp), ("matrix", mass_matrix)):
                if native_object is None:
                    continue
                identity = (label, id(native_object))
                if identity in destroyed:
                    continue
                destroyed.add(identity)
                try:
                    native_object.destroy()
                except Exception as exc:  # pragma: no cover - native backend failure
                    destroy_errors.append(f"{label}: {type(exc).__name__}: {exc}")

        self._output_mass_projection_cache.clear()
        self._output_expression_cache.clear()
        self._output_projection_cache.clear()
        if destroy_errors:
            raise RuntimeError(
                "Could not destroy cached output projection objects: "
                + "; ".join(destroy_errors)
            )

    def _ensure_function_space_caches_current(self) -> None:
        """PRIVATE: Lazily discard spaces retained from another mesh epoch."""

        current_epoch = getattr(self, "remesh_epoch", 0)
        if self._function_space_cache_epoch == current_epoch:
            return
        self._dg_spaces.clear()
        self._cg_spaces.clear()
        self._quadrature_spaces.clear()
        self._function_space_cache_epoch = current_epoch

    def _ensure_output_projection_caches_current(self) -> None:
        """PRIVATE: Lazily discard projections retained from another epoch."""

        current_epoch = getattr(self, "remesh_epoch", 0)
        if self._output_projection_cache_epoch == current_epoch:
            return
        self._clear_output_projection_caches()
        self._output_projection_cache_epoch = current_epoch

    def _register_field_definition(self, definition: FieldDefinition, domain: str = FULL_DOMAIN) -> fem.Function:
        """PRIVATE: Validate a registration and create its function on its support.

        Re-registering a field for another domain extends its support. Before
        the step layouts are finalized the field is then recreated on the
        larger support; afterwards, extending a support is rejected.
        """

        existing_definition = self.field_definitions.get(definition.name)
        if existing_definition is not None:
            if not existing_definition.is_compatible_with(definition):
                raise ValueError(
                    f"Field '{definition.name}' is already registered with an incompatible "
                    f"definition: existing={existing_definition}, requested={definition}."
                )
            domains = self._field_domains[definition.name]
            if domain not in domains:
                support_key = self._resolve_support_key(domains + [domain])
                if support_key != self._field_support_keys[definition.name]:
                    if self._test_functions_finalized:
                        raise ValueError(
                            f"Field '{definition.name}' cannot be extended to domain '{domain}' "
                            "after the solution steps have been finalized."
                        )
                    self._field_support_keys[definition.name] = support_key
                    self.fields[definition.name] = self._create_field_on_support(existing_definition, support_key)
                    self._refresh_test_functions(definition.name)
                domains.append(domain)
            return self.fields[definition.name]

        if definition.name in self.fields:
            raise ValueError(
                f"Field '{definition.name}' exists without a structured FieldDefinition and "
                "cannot be registered safely."
            )

        support_key = self._resolve_support_key([domain])
        field = self._create_field_on_support(definition, support_key)

        self.field_definitions[definition.name] = definition
        self.fields[definition.name] = field
        self.field_order[definition.name] = definition.degree
        self.field_type[definition.name] = self._legacy_field_type(definition)
        self._field_domains[definition.name] = [domain]
        self._field_support_keys[definition.name] = support_key
        return field

    def _create_field_on_support(self, definition: FieldDefinition, support_key) -> fem.Function:
        """PRIVATE: Create a field's function on the full mesh or its submesh."""
        if support_key is None:
            if definition.family == "Quadrature":
                function_space = self._get_quadrature_space(
                    definition.quadrature_degree,
                    definition.value_shape,
                )
                return fem.Function(function_space, name=definition.description)
            return self._create_function_from_definition(definition, self.mesh)
        if support_key == (_GLOBAL,):
            return fem.Function(fem.functionspace(self._get_global_mesh(), ("DG", 0)), name=definition.description)
        submesh, _cell_map, _parent_cells = self._get_submesh(support_key)
        return self._create_function_from_definition(definition, submesh)

    def _resolve_support_key(self, domains: List[str]) -> Optional[Tuple[int, ...]]:
        """PRIVATE: Return the sorted cell tags of a support, or ``None`` for the full mesh."""
        # "domain" always means the full mesh, also for meshes without markers.
        if FULL_DOMAIN in domains:
            return None
        if _GLOBAL in domains:
            if len(domains) != 1:
                raise ValueError(f"A global field cannot also live on domains {domains}.")
            return (_GLOBAL,)
        facet_groups = [domain for domain in domains if self._is_facet_group(domain)]
        if facet_groups:
            if len(facet_groups) != len(domains):
                raise ValueError(f"A field cannot live on both cells and facets (domains {domains}).")
            return (_FACETS,) + tuple(sorted({tag for domain in domains for tag in self.get_boundary_tags(domain)}))
        tags = sorted({tag for domain in domains for tag in self.get_domain_tags(domain)})
        if set(tags) >= set(getattr(self, "domain_markers", {}).values()):
            return None
        return tuple(tags)

    def _is_facet_group(self, name: str) -> bool:
        """PRIVATE: Return whether a domain name refers to facets rather than cells."""
        try:
            self.get_domain_tags(name)
            return False
        except KeyError:
            pass
        try:
            self.get_boundary_tags(name)
            return True
        except KeyError:
            raise KeyError(
                f"Domain '{name}' not found. Available domains: {self.domain_names()}; "
                f"available facet groups (for interface fields): {self.boundary_names()}"
            ) from None

    def _get_submesh(self, support_key: Tuple[int, ...]):
        """PRIVATE: Return the cached ``(submesh, cell_map, parent_cells)`` of one support.

        Interface supports create a submesh of facets; ``parent_cells`` then
        holds the parent facet of every submesh cell.
        """
        cached = self._submeshes.get(support_key)
        if cached is not None:
            return cached
        is_interface = support_key[0] == _FACETS
        tags = support_key[1:] if is_interface else support_key
        mesh_tags = self.facet_tags if is_interface else self.cell_tags
        tdim = self.mesh.topology.dim - 1 if is_interface else self.mesh.topology.dim
        if is_interface:
            self.mesh.topology.create_connectivity(tdim, tdim + 1)
        entities = np.unique(np.concatenate([mesh_tags.find(tag) for tag in tags])).astype(np.int32)
        submesh, cell_map, vertex_map, _node_map = dmesh.create_submesh(self.mesh, tdim, entities)
        self._submesh_vertex_maps[support_key] = vertex_map
        sub_cell_map = submesh.topology.index_map(tdim)
        sub_cells = np.arange(sub_cell_map.size_local + sub_cell_map.num_ghosts, dtype=np.int32)
        parent_cells = np.asarray(cell_map.sub_topology_to_topology(sub_cells, inverse=False), dtype=np.int32)
        self._submeshes[support_key] = (submesh, cell_map, parent_cells)
        self._submesh_entity_maps[support_key] = cell_map
        return self._submeshes[support_key]

    def _refresh_test_functions(self, field_name: str) -> None:
        """PRIVATE: Recreate a field's test/trial functions after it was recreated."""
        field = self.fields[field_name]
        for step, names in self.step_fields.items():
            if field_name in names:
                self.test_functions[f"{field_name}_test_step{step}"] = TestFunction(field.function_space)
                self.trial_functions.pop(f"{field_name}_trial_step{step}", None)

    @staticmethod
    def _create_function_from_definition(definition: FieldDefinition, target_mesh) -> fem.Function:
        """PRIVATE: Create an unregistered equivalent function on ``target_mesh``."""

        if definition.family == "Quadrature":
            elem = quadrature_element(
                target_mesh.basix_cell(),
                value_shape=definition.value_shape,
                degree=definition.quadrature_degree,
            )
        else:
            shape = definition.value_shape or None
            elem = _MeshDOFs._create_element_on_mesh(
                target_mesh,
                definition.family,
                definition.degree,
                shape=shape,
            )

        function_space = fem.functionspace(target_mesh, elem)
        return fem.Function(function_space, name=definition.description)

    @staticmethod
    def _legacy_field_type(definition: FieldDefinition) -> str:
        """PRIVATE: Return the existing display/storage type string."""

        if definition.family == "Quadrature":
            return "Quadrature Tensor"
        if definition.value_shape:
            return f"{definition.family} Vector"
        return definition.family

    def _create_element(self, space: str, order: int, shape: Optional[Tuple] = None):
        """PRIVATE: Helper method to create a finite element."""
        return self._create_element_on_mesh(self.mesh, space, order, shape)

    @staticmethod
    def _create_element_on_mesh(target_mesh, space: str, order: int, shape: Optional[Tuple] = None):
        """PRIVATE: Create a standard element for an explicit mesh."""

        if space not in ["CG", "DG"]:
            raise ValueError(f"Unsupported function space: {space}")

        element_name = "Lagrange" if space == "CG" else "DG"

        if (shape is None) and (element_name == "Lagrange"):
            return element(element_name, target_mesh.basix_cell(), degree=order, lagrange_variant=LagrangeVariant.bernstein)
        elif element_name == "DG":
            return element(element_name, target_mesh.basix_cell(), degree=order, shape=shape)
        else:
            return element(element_name, target_mesh.basix_cell(), degree=order, shape=shape, lagrange_variant=LagrangeVariant.bernstein)

    def _add_test_function_internal(self, field_name: str, step: int, is_vector: bool) -> fem.Function:
        """PRIVATE: Internal helper to add test functions for scalar or vector fields."""
        test_key = f"{field_name}_test_step{step}"
        if test_key in self.test_functions:
            return self.test_functions[test_key]

        if field_name not in self.fields:
            raise ValueError(f"Field '{field_name}' not found. Define the field before creating a test function.")

        is_field_vector = "Vector" in self.field_type.get(field_name, "")
        if is_vector != is_field_vector:
            actual_type = "vector" if is_field_vector else "scalar"
            other_method = "Add_TestFunction" if is_vector else "Add_Vector_TestFunction"
            raise ValueError(f"Field '{field_name}' is a {actual_type} field. Use {other_method} instead.")

        field = self.fields[field_name]
        test_func = TestFunction(field.function_space)
        self.test_functions[test_key] = test_func

        if step not in self.step_fields:
            self.step_fields[step] = []
        if field_name not in self.step_fields[step]:
            self.step_fields[step].append(field_name)

        return test_func

    def _build_mixed_space_for_step(self, step: int) -> None:
        """
        PRIVATE: Build the block layout for all fields registered for this step.

        Each field becomes one block of a ``ufl.MixedFunctionSpace`` built from
        the fields' own function spaces, so the step can be assembled and
        solved monolithically while every field keeps its own storage.
        Single-field steps need no block layout and store ``None``.
        """
        if step in self.mixed_spaces:
            return

        if step not in self.step_fields or len(self.step_fields[step]) == 0:  # pragma: no cover - defensive private-helper guard
            return

        field_names = list(self.step_fields[step])
        if len(field_names) == 1:
            self.mixed_spaces[step] = None
            return

        mprint(f"\tBuilding block layout for step {step} with fields: {field_names}")

        function_spaces = [self.fields[fname].function_space for fname in field_names]
        mixed_space = MixedFunctionSpace(*function_spaces)
        test_funcs = TestFunctions(mixed_space)
        trial_funcs = TrialFunctions(mixed_space)

        for fname, test_func in zip(field_names, test_funcs):
            self.test_functions[f"{fname}_test_step{step}"] = test_func
        for fname in field_names:
            self.trial_functions.pop(f"{fname}_trial_step{step}", None)

        self.mixed_spaces[step] = {
            'space': mixed_space,
            'field_names': field_names,
            'function_spaces': function_spaces,
            'test_functions': test_funcs,
            'trial_functions': trial_funcs,
        }

    def _get_quadrature_space(self, degree: int, value_shape: Tuple[int, ...]):
        """PRIVATE: Return a cached quadrature function space for one degree/shape."""
        self._ensure_function_space_caches_current()
        cache_key = (degree, value_shape)
        if cache_key not in self._quadrature_spaces:
            q_element = quadrature_element(
                self.mesh.basix_cell(),
                degree=degree,
                value_shape=value_shape,
            )
            self._quadrature_spaces[cache_key] = fem.functionspace(self.mesh, q_element)
        return self._quadrature_spaces[cache_key]

    def _get_dg_space(self, degree: int, support_key: Optional[Tuple[int, ...]] = None):
        """PRIVATE: Get or create a cached scalar DG space on the mesh or one submesh."""
        self._ensure_function_space_caches_current()
        cache_key = degree if support_key is None else (support_key, degree)
        if cache_key not in self._dg_spaces:
            target_mesh = self.mesh if support_key is None else self._get_submesh(support_key)[0]
            dg_element = element("DG", target_mesh.basix_cell(), degree)
            self._dg_spaces[cache_key] = fem.functionspace(target_mesh, dg_element)
        return self._dg_spaces[cache_key]

    def _get_cg_space(self, degree: int):
        """PRIVATE: Get or create a cached scalar CG (Lagrange) function space of the given degree."""
        self._ensure_function_space_caches_current()
        if degree not in self._cg_spaces:
            cg_element = self._create_element("CG", degree)
            self._cg_spaces[degree] = fem.functionspace(self.mesh, cg_element)
        return self._cg_spaces[degree]

    def _get_field_dg_degree(self, field_name: str) -> int:
        """PRIVATE: Determine the DG degree used to project one field for output.

        Quadrature fields represent integration-point data rather than a
        polynomial field on the mesh.  Their output representation is
        therefore cell-wise DG0, which avoids reconstructing a higher-order
        polynomial between quadrature points.
        """
        if field_name in self.fields:
            field_view = self.Get_Field_For_Output(field_name)
            if self._is_quadrature_field(field_name, field_view):
                return 0
        order = self.field_order.get(field_name, 1)
        return 1 if order < 1 else order

    def _resolve_output_dg_space(self, field_name: str, V_dg):
        """PRIVATE: Resolve the DG space used to project one field for output."""
        if field_name in self.fields:
            field_view = self.Get_Field_For_Output(field_name)
            if self._is_quadrature_field(field_name, field_view):
                support_key = None if V_dg is None else self._mesh_key(V_dg.mesh)
                return self._get_dg_space(0, support_key)
        if V_dg is not None:
            return V_dg
        return self._get_dg_space(self._get_field_dg_degree(field_name))

    def _parse_output_field_spec(
        self, field_spec: str
    ) -> tuple[str, Optional[str], Optional[str]]:
        """PRIVATE: Parse a field specification into field and component parts."""
        if "_" not in field_spec:
            return field_spec, None, None

        field_name, suffix = field_spec.rsplit("_", 1)
        if suffix in _VECTOR_COMPONENT_SUFFIXES:
            return field_name, "vector", suffix
        if suffix in _TENSOR_COMPONENT_SUFFIXES:
            return field_name, "tensor", suffix
        numeric_parse = self._parse_numeric_output_component(field_name, suffix)
        if numeric_parse is not None:
            return numeric_parse
        return field_spec, None, None

    def _parse_numeric_output_component(
        self, field_name: str, suffix: str
    ) -> Optional[tuple[str, str, str]]:
        """PRIVATE: Parse one-based numeric component suffixes such as ``_1`` or ``_11``."""
        if not suffix.isdigit() or field_name not in self.fields:
            return None

        value_shape = self.fields[field_name].function_space.element.value_shape
        if len(suffix) == 1 and len(value_shape) == 1:
            return field_name, "vector", suffix
        if (
            len(suffix) == 1
            and len(value_shape) == 2
            and (value_shape[0] == 1 or value_shape[1] == 1)
        ):
            return field_name, "tensor", suffix
        if len(suffix) == 2 and len(value_shape) == 2:
            return field_name, "tensor", suffix
        return None

    def _resolve_vector_component_index(self, component: str) -> int:
        """PRIVATE: Resolve a vector component suffix to a zero-based index."""
        if component in _VECTOR_COMPONENT_SUFFIXES:
            return _VECTOR_COMPONENT_SUFFIXES[component]
        if component.isdigit():
            return int(component) - 1
        raise KeyError(component)

    def _resolve_tensor_component_indices(
        self, component: str, value_shape: Optional[Tuple[int, ...]] = None
    ) -> tuple[int, int]:
        """PRIVATE: Resolve a tensor component suffix to zero-based indices."""
        if component in _TENSOR_COMPONENT_SUFFIXES:
            return _TENSOR_COMPONENT_SUFFIXES[component]
        if (
            len(component) == 1
            and component.isdigit()
            and value_shape is not None
            and len(value_shape) == 2
        ):
            comp_idx = int(component) - 1
            if value_shape[1] == 1:
                return comp_idx, 0
            if value_shape[0] == 1:
                return 0, comp_idx
        if len(component) == 2 and component.isdigit():
            return int(component[0]) - 1, int(component[1]) - 1
        raise KeyError(component)

    def _is_quadrature_field(self, field_name: str, field_view: FieldOutputView) -> bool:
        """PRIVATE: Return whether one field is stored in a quadrature space."""
        field_type = self.field_type.get(field_name, "")
        if "Quadrature" in field_type:
            return True

        element_obj = field_view.field.function_space.element
        family = getattr(element_obj, "family", None)
        if isinstance(family, str):
            return family == "Quadrature"

        basix = None
        try:
            basix = element_obj.basix_element
        except RuntimeError:
            basix = None

        if basix is None:
            return False

        basix_family = getattr(basix, "family", None)
        family_name = getattr(basix_family, "name", None)
        if isinstance(family_name, str):
            return family_name.lower() == "quadrature"
        if isinstance(basix_family, str):
            return basix_family == "Quadrature"
        return False

    def _is_mandel_tensor_field(self, field_view: FieldOutputView) -> bool:
        """PRIVATE: Return whether one field stores a Mandel-vector symmetric tensor."""
        value_shape = tuple(field_view.value_shape)
        return value_shape in {(6,), (6, 1)}

    def _resolve_mandel_component(self, component: str):
        """PRIVATE: Resolve a tensor suffix to one Mandel-vector entry and scaling."""
        if component not in _MANDEL_COMPONENT_SUFFIXES:
            raise KeyError(component)
        return _MANDEL_COMPONENT_SUFFIXES[component]

    def _get_cached_output_projection(self, field_spec: str, V_dg):
        """PRIVATE: Reuse DG output functions across plotting calls when possible."""
        self._ensure_output_projection_caches_current()
        cache_key = (field_spec, id(V_dg))
        field_dg = self._output_projection_cache.get(cache_key)
        if field_dg is None:
            field_dg = fem.Function(V_dg)
            self._output_projection_cache[cache_key] = field_dg
        return field_dg, cache_key

    def _get_cached_output_mass_projection(self, V_dg):
        """PRIVATE: Reuse the scalar DG mass matrix and solver for output projection."""
        self._ensure_output_projection_caches_current()
        cache_key = id(V_dg)
        cached = self._output_mass_projection_cache.get(cache_key)
        if cached is not None:
            return cached

        trial = TrialFunction(V_dg)
        test = TestFunction(V_dg)
        mass_form = fem.form(trial * test * dx)
        mass_matrix = assemble_matrix(mass_form)
        mass_matrix.assemble()

        ksp = PETSc.KSP().create(mass_matrix.getComm())
        ksp.setOperators(mass_matrix)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)

        cached = (test, mass_matrix, ksp)
        self._output_mass_projection_cache[cache_key] = cached
        return cached

    def _project_quadrature_source_to_output_space(
        self, field_name: str, field_spec: str, title: str, source, V_dg, association: str
    ) -> Optional[ProjectedFieldOutput]:
        """PRIVATE: L2-project one quadrature-backed scalar expression to the DG plotting space."""
        if self._is_interface_output_on_full_mesh(field_name, V_dg):
            mprint(f"Warning: Full-mesh output of interface field '{field_spec}' is not supported. Skipping.")
            return None
        try:
            V_dg = self._resolve_output_dg_space(field_name, V_dg)
            field_dg, _cache_key = self._get_cached_output_projection(field_spec, V_dg)
            test, _mass_matrix, ksp = self._get_cached_output_mass_projection(V_dg)

            support = self.Get_Field_Support(field_name)
            if support.is_full:
                rhs_form = fem.form(source * test * dx)
            elif V_dg.mesh is support.mesh:
                rhs_form = fem.form(source * test * Measure("dx", domain=support.mesh))
            else:
                dx_support = Measure("dx", domain=self.mesh, subdomain_data=self.cell_tags)(support.tags)
                rhs_form = fem.form(source * test * dx_support, entity_maps=self.get_entity_maps())
            rhs_vec = assemble_vector(rhs_form)
            rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            rhs_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            ksp.solve(rhs_vec, field_dg.x.petsc_vec)
            field_dg.x.scatter_forward()
            rhs_vec.destroy()
            if not support.is_full and V_dg.mesh is self.mesh:
                self._fill_outside_support(field_dg, support)
        except Exception as exc:
            mprint(
                f"Warning: Could not project quadrature field '{field_spec}' "
                f"for output: {exc}. Skipping."
            )
            return None

        values = field_dg.x.array
        if association == "cell":
            owned_size = V_dg.dofmap.index_map.size_local * V_dg.dofmap.index_map_bs
            values = values[:owned_size]

        return ProjectedFieldOutput(
            field_spec=field_spec,
            field_name=field_name,
            title=title,
            values=values.copy(),
            function_space=V_dg,
            association=association,
        )

    def _project_source_to_output_space(
        self,
        field_name: str,
        field_spec: str,
        title: str,
        source,
        V_dg,
        *,
        use_expression: bool,
        project_quadrature: bool = False,
        association: str = "point",
    ) -> Optional[ProjectedFieldOutput]:
        """PRIVATE: Project one source field or expression to the DG plotting space."""
        if self._is_interface_output_on_full_mesh(field_name, V_dg):
            mprint(f"Warning: Full-mesh output of interface field '{field_spec}' is not supported. Skipping.")
            return None
        try:
            V_dg = self._resolve_output_dg_space(field_name, V_dg)
            if project_quadrature:
                return self._project_quadrature_source_to_output_space(
                    field_name, field_spec, title, source, V_dg, association
                )

            field_dg, cache_key = self._get_cached_output_projection(field_spec, V_dg)
            support = self.Get_Field_Support(field_name)
            if not support.is_full:
                source_object = source if not use_expression and not self.Is_DG_Field(field_name) else fem.Expression(source, V_dg.element.interpolation_points)
                if V_dg.mesh is support.mesh:
                    # Output space on the field's own submesh.
                    field_dg.interpolate(source_object)
                else:
                    # Full-mesh output space: evaluate on the field's own cells; NaN marks the rest.
                    sub_cells = np.arange(support.parent_cells.size, dtype=np.int32)
                    field_dg.interpolate(source_object, cells0=sub_cells, cells1=support.parent_cells)
                    self._fill_outside_support(field_dg, support)
            elif use_expression:
                expr = self._output_expression_cache.get(cache_key)
                if expr is None:
                    expr = fem.Expression(source, V_dg.element.interpolation_points)
                    self._output_expression_cache[cache_key] = expr
                field_dg.interpolate(expr)
            elif self.Is_DG_Field(field_name):
                # DG source fields must be evaluated cell-locally to avoid the
                # geometric point-search used by interpolate(), which can pick a
                # neighbouring cell's polynomial at shared vertices and produce
                # incorrect (e.g. near-zero) values for interior nodes.
                # fem.Expression guarantees that node i of cell K is always
                # evaluated using cell K's own DG polynomial.
                # Do NOT cache: the source may be a newly-collapsed object on each
                # call (e.g. when h lives in a monolithic mixed space).
                expr = fem.Expression(source, V_dg.element.interpolation_points)
                field_dg.interpolate(expr)
            else:
                field_dg.interpolate(source)
        except Exception as exc:
            mprint(
                f"Warning: Could not project field '{field_spec}' for output: "
                f"{exc}. Skipping."
            )
            return None

        return ProjectedFieldOutput(
            field_spec=field_spec,
            field_name=field_name,
            title=title,
            values=field_dg.x.array.copy(),
            function_space=V_dg,
            association=association,
        )

    def _interface_integration_entities(self, interface: str, normal_from: Optional[str]):
        """PRIVATE: Return interior-facet integration entities with the ``normal_from`` cell first.

        Also returns the parent cells on the "+" (``normal_from``) and "-"
        sides of the owned interface facets.
        """
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim - 1, tdim)
        self.mesh.topology.create_connectivity(tdim, tdim - 1)
        facets = self.get_boundary_facets(interface)
        owned_facets = np.count_nonzero(facets < self.mesh.topology.index_map(tdim - 1).size_local)
        entities = fem.compute_integration_domains(fem.IntegralType.interior_facet, self.mesh.topology, facets).reshape(-1, 4)
        if self.mesh.comm.allreduce(int(entities.shape[0] != owned_facets)):
            raise ValueError(f"Interface integrals need interior facets only, but '{interface}' contains boundary facets.")

        if normal_from is None:
            normal_from = self._first_adjacent_domain(entities)
        side_cells = self.get_domain_cells(normal_from)
        first_in, second_in = np.isin(entities[:, 0], side_cells), np.isin(entities[:, 2], side_cells)
        if self.mesh.comm.allreduce(int(np.any(first_in == second_in))):
            raise ValueError(f"Domain '{normal_from}' must lie on exactly one side of every facet of '{interface}'.")
        entities[second_in] = entities[second_in][:, [2, 3, 0, 1]]
        return entities, entities[:, 0], entities[:, 2]

    def _first_adjacent_domain(self, entities) -> str:
        """PRIVATE: Return the first registered leaf domain touching the given interior-facet entities."""
        local_cells = np.unique(entities[:, [0, 2]])
        for name in self.domain_names():
            if len(self.get_domain_tags(name)) != 1:
                continue
            if self.mesh.comm.allreduce(int(np.isin(local_cells, self.get_domain_cells(name)).any())):
                return name
        raise ValueError("No domain is adjacent to the interface.")  # pragma: no cover

    def _interface_side(self, terminal, interface: str, plus_cells, minus_cells) -> str:
        """PRIVATE: Return the restriction ("+" or "-") a field is evaluated from on an interface."""
        domain = terminal.ufl_function_space().ufl_domain()
        if domain is self.mesh.ufl_domain():
            return "+"
        for key, (submesh, _cell_map, parent_cells) in self._submeshes.items():
            if domain is not submesh.ufl_domain():
                continue
            if key[0] == _FACETS:
                if key != self._resolve_support_key([interface]):
                    break
                return "+"
            for side, cells in (("+", plus_cells), ("-", minus_cells)):
                if self.mesh.comm.allreduce(int(not np.isin(cells, parent_cells).all())) == 0:
                    return side
            break
        raise ValueError(f"A field of the interface integral over '{interface}' is defined on neither side of it.")

    def _get_global_mesh(self):
        """PRIVATE: Return the one-cell auxiliary mesh carrying global fields (one DG0 DOF each)."""
        if self._global_mesh is None:
            comm = self.mesh.comm
            on_root = comm.rank == 0
            cells = np.array([[0, 1]] if on_root else np.empty((0, 2)), dtype=np.int64)
            x = np.array([[0.0], [1.0]] if on_root else np.empty((0, 1)), dtype=np.float64)

            def keep_on_root(_comm, _nparts, _cell_types, local_cells):
                return adjacencylist(np.zeros(len(local_cells[0]) // 2, dtype=np.int32))._cpp_object

            coordinate_element = ufl_mesh(element("Lagrange", "interval", 1, shape=(1,)))
            self._global_mesh = dmesh.create_mesh(comm, cells, coordinate_element, x, partitioner=keep_on_root)
        return self._global_mesh

    def _get_global_field_or_raise(self, field_name: str) -> fem.Function:
        """PRIVATE: Return a registered global field, rejecting other fields."""
        if not self.Get_Field_Support(field_name).is_global:
            raise ValueError(f"Field '{field_name}' is not a global field.")
        return self.fields[field_name]

    def _global_field_owner(self, field) -> int:
        """PRIVATE: Return the rank owning the single DOF of a global field."""
        owns = field.function_space.dofmap.index_map.size_local > 0
        return self.mesh.comm.allreduce(self.mesh.comm.rank if owns else -1, op=MPI.MAX)

    def _skip_global_field_output(self, field_name: str) -> bool:
        """PRIVATE: Warn and return ``True`` when a global field is requested as spatial output."""
        if not self.Is_Global_Field(field_name):
            return False
        mprint(f"Warning: Global field '{field_name}' has no spatial output; report it through Global_Measures. Skipping.")
        return True

    def _require_spatial_field(self, field_name: str, operation: str) -> FieldSupport:
        """PRIVATE: Return a field's support, rejecting global fields for spatial operations."""
        support = self.Get_Field_Support(field_name)
        if support.is_global:
            raise ValueError(f"Global field '{field_name}' has no spatial representation and cannot {operation}.")
        return support

    def _interface_support_key(self, field_name: str):
        """PRIVATE: Return the support key of an interface field, or ``None`` for other fields."""
        support_key = self._field_support_keys.get(field_name)
        return support_key if support_key is not None and support_key[0] == _FACETS else None

    def _is_interface_output_on_full_mesh(self, field_name: str, V_dg) -> bool:
        """PRIVATE: Return whether an interface field is requested on a full-mesh output space."""
        support_key = self._interface_support_key(field_name)
        if support_key is None:
            return False
        return V_dg is None or V_dg.mesh is not self._get_submesh(support_key)[0]

    def _fill_outside_support(self, field_dg: fem.Function, support: FieldSupport) -> None:
        """PRIVATE: Set output DOFs outside a restricted field's support to NaN."""
        if support.is_full:
            return
        dofmap = field_dg.function_space.dofmap
        num_cells = self.mesh.topology.index_map(self.mesh.topology.dim)
        outside_cells = np.setdiff1d(
            np.arange(num_cells.size_local + num_cells.num_ghosts, dtype=np.int32),
            support.parent_cells,
        )
        outside_dofs = np.unique(dofmap.list[outside_cells].ravel())
        inside_dofs = np.unique(dofmap.list[support.parent_cells].ravel())
        # Nodes shared with the support (continuous output spaces) keep their value.
        field_dg.x.array[np.setdiff1d(outside_dofs, inside_dofs)] = np.nan

    def _project_vector_field_component(
        self,
        field_name: str,
        component: str,
        field_spec: str,
        V_dg=None,
        *,
        quadrature_target_space=None,
        quadrature_association: str = "point",
    ) -> Optional[ProjectedFieldOutput]:
        """PRIVATE: Project one vector component to a scalar DG space."""
        if field_name not in self.fields:
            mprint(f"Warning: Field '{field_name}' not found. Skipping '{field_spec}'.")
            return None

        field_view = self.Get_Field_For_Output(field_name)
        if len(field_view.value_shape) != 1:
            mprint(
                f"Warning: Field '{field_name}' is not a vector field. "
                f"Skipping '{field_spec}'."
            )
            return None

        comp_idx = self._resolve_vector_component_index(component)
        if comp_idx >= field_view.comp_count:
            mprint(
                f"Warning: Component {component} out of range for field "
                f"'{field_name}'. Skipping."
            )
            return None

        try:
            project_quadrature = self._is_quadrature_field(field_name, field_view)
            if project_quadrature:
                field_component = field_view.field[comp_idx]
                use_expression = True
                target_space = quadrature_target_space if quadrature_target_space is not None else V_dg
            else:
                field_component = field_view.collapse_component(comp_idx)
                use_expression = False
                target_space = V_dg
        except Exception as exc:
            mprint(
                f"Warning: Could not extract component {component} from field "
                f"'{field_name}': {exc}. Skipping."
            )
            return None

        return self._project_source_to_output_space(
            field_name,
            field_spec,
            f"{field_view.title} - {component.upper()} component",
            field_component,
            target_space,
            use_expression=use_expression,
            project_quadrature=project_quadrature,
            association=quadrature_association if project_quadrature else "point",
        )

    def _project_tensor_field_component(
        self,
        field_name: str,
        component: str,
        field_spec: str,
        V_dg=None,
        *,
        quadrature_target_space=None,
        quadrature_association: str = "point",
    ) -> Optional[ProjectedFieldOutput]:
        """PRIVATE: Project one tensor component to a scalar DG space."""
        if field_name not in self.fields:
            mprint(f"Warning: Field '{field_name}' not found. Skipping '{field_spec}'.")
            return None

        field_view = self.Get_Field_For_Output(field_name)
        if len(field_view.value_shape) != 2:
            if not self._is_mandel_tensor_field(field_view):
                mprint(
                    f"Warning: Field '{field_name}' is not a tensor field. "
                    f"Skipping '{field_spec}'."
                )
                return None

        if self._is_mandel_tensor_field(field_view):
            try:
                mandel_idx, mandel_scale = self._resolve_mandel_component(component)
                if len(field_view.value_shape) == 2:
                    source = mandel_scale * field_view.field[mandel_idx, 0]
                else:
                    source = mandel_scale * field_view.field[mandel_idx]
            except Exception as exc:
                mprint(
                    f"Warning: Could not extract Mandel component {component} from field "
                    f"'{field_name}': {exc}. Skipping."
                )
                return None

            project_quadrature = self._is_quadrature_field(field_name, field_view)
            target_space = quadrature_target_space if project_quadrature and quadrature_target_space is not None else V_dg

            return self._project_source_to_output_space(
                field_name,
                field_spec,
                f"{field_view.title} - {component.upper()} component",
                source,
                target_space,
                use_expression=not project_quadrature,
                project_quadrature=project_quadrature,
                association=quadrature_association if project_quadrature else "point",
            )

        row_idx, col_idx = self._resolve_tensor_component_indices(component, field_view.value_shape)
        if row_idx >= field_view.value_shape[0] or col_idx >= field_view.value_shape[1]:
            mprint(
                f"Warning: Component {component} out of range for field "
                f"'{field_name}'. Skipping."
            )
            return None

        try:
            source = field_view.field[row_idx, col_idx]
        except Exception as exc:
            mprint(
                f"Warning: Could not extract component {component} from field "
                f"'{field_name}': {exc}. Skipping."
            )
            return None

        project_quadrature = self._is_quadrature_field(field_name, field_view)
        target_space = quadrature_target_space if project_quadrature and quadrature_target_space is not None else V_dg

        return self._project_source_to_output_space(
            field_name,
            field_spec,
            f"{field_view.title} - {component.upper()} component",
            source,
            target_space,
            use_expression=not project_quadrature,
            project_quadrature=project_quadrature,
            association=quadrature_association if project_quadrature else "point",
        )

    def _project_scalar_field(
        self,
        field_name: str,
        field_spec: str,
        V_dg=None,
        *,
        quadrature_target_space=None,
        quadrature_association: str = "point",
    ) -> Optional[ProjectedFieldOutput]:
        """PRIVATE: Project one scalar field to a DG space for output."""
        if field_name not in self.fields:
            mprint(f"Warning: Field '{field_name}' not found. Skipping.")
            return None

        field_view = self.Get_Field_For_Output(field_name)
        if len(field_view.value_shape) != 0:
            mprint(
                f"Warning: Field '{field_name}' has multiple components but no component "
                "suffix was specified. Skipping."
            )
            return None

        project_quadrature = self._is_quadrature_field(field_name, field_view)
        if project_quadrature:
            source = field_view.field
            use_expression = True
            target_space = quadrature_target_space if quadrature_target_space is not None else V_dg
        else:
            source = field_view.collapse_component(0)
            use_expression = False
            target_space = V_dg

        return self._project_source_to_output_space(
            field_name,
            field_spec,
            field_view.title,
            source,
            target_space,
            use_expression=use_expression,
            project_quadrature=project_quadrature,
            association=quadrature_association if project_quadrature else "point",
        )
