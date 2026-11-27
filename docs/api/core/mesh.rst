Mesh
====

The :class:`femcubes.core.mesh.mesh.Mesh` class is the main mesh-facing
object used throughout FEMCubes.  It wraps:

* mesh generation from a named ``MeshType``
* named domain and boundary markers
* field registration and mixed-space construction
* field preparation for plotting and file output
* interactive plotting helpers
* VTK / HDF5 output
* lumped mass-operator utilities

In normal use, models and solvers should interact with this composite
class rather than with the internal mix-ins that implement each feature.

Overview
--------

Constructing ``Mesh(params)`` immediately:

1. selects the registered mesh generator matching ``params.MeshType``
2. builds the DOLFINx mesh plus any cell/facet tags
3. stores named boundary/domain markers
4. initializes field, plotting, output, and lumped-operator helpers

The returned object therefore acts as the central repository for both
the geometric mesh and all finite-element fields defined on it.

Typical Usage
-------------

A minimal workflow looks like this:

.. code-block:: python

   from femcubes.core.mesh.mesh import Mesh

   mesh = Mesh(params)
   mesh.Add_Field("p", "Pressure", "CG", 1)
   mesh.Add_Vector_Field("u", "Displacement", "CG", 2, params.dim)
   mesh.Add_TestFunction("p", step=0)
   mesh.Add_Vector_TestFunction("u", nDim=params.dim, step=0)
   mesh.finalize_test_functions()

Once fields are registered, the mesh object can also be used to:

* resolve boundary or domain tags by name
* obtain symbolic fields for UFL forms with ``Get_Field()``
* obtain concrete output views with ``Get_Field_For_Output()``
* prepare DG-projected scalar output through ``Prepare_Field_For_Output()``
* write output files with ``Write_Outputs()``
* render mesh, field, or measure plots interactively

Named Boundaries and Domains
----------------------------

Mesh generators populate ``boundary_markers`` and
``domain_markers``.  These mappings are the stable API used by
models and boundary conditions; direct numeric tag values should usually
be treated as implementation details.

Names may be hierarchical, using ``:`` as separator.  Each stored marker
is a *part* with one tag, e.g. ``domain:left`` or ``top:centre``.  A
*group* name such as ``domain`` or ``top`` is not stored itself; it
resolves to the union of all parts below it.  Single-domain generators
store plain names such as ``domain`` and ``top``, so models written for
the full domain work unchanged on multi-domain meshes.  Every mesh
therefore exposes ``domain``, covering all of its cells.

.. code-block:: python

   ds = mesh.get_Boundary_Integrator()
   ds(mesh.get_boundary_tags("top"))          # whole top edge
   ds(mesh.get_boundary_tags("top:centre"))   # only the part in domain:centre
   mesh.get_domain_cells("domain:centre")     # cells of one subdomain

``get_Boundary_Integrator()`` only integrates over exterior facets.
Tagged interior facets, such as interfaces between subdomains, integrate
to zero under ``ds`` and require an interior-facet measure (``dS``).

Useful methods:

* :meth:`Mesh.get_boundary_tags <femcubes.core.mesh._mesh_groups._MeshGroups.get_boundary_tags>`
* :meth:`Mesh.get_domain_tags <femcubes.core.mesh._mesh_groups._MeshGroups.get_domain_tags>`
* :meth:`Mesh.get_boundary_tag <femcubes.core.mesh._mesh_groups._MeshGroups.get_boundary_tag>`
  (names covering a single part only)
* :meth:`Mesh.get_domain_tag <femcubes.core.mesh._mesh_groups._MeshGroups.get_domain_tag>`
  (names covering a single part only)
* :meth:`Mesh.get_boundary_facets <femcubes.core.mesh._mesh_groups._MeshGroups.get_boundary_facets>`
* :meth:`Mesh.get_domain_cells <femcubes.core.mesh._mesh_groups._MeshGroups.get_domain_cells>`
* :meth:`Mesh.boundary_names <femcubes.core.mesh._mesh_groups._MeshGroups.boundary_names>`
* :meth:`Mesh.domain_names <femcubes.core.mesh._mesh_groups._MeshGroups.domain_names>`
* :meth:`Mesh.Require_Boundary_On_Domain <femcubes.core.mesh._mesh_groups._MeshGroups.Require_Boundary_On_Domain>`
  (checks that a boundary bounds a domain, e.g. for boundary models)

Fields and Mixed Spaces
-----------------------

The mesh stores registered fields in :attr:`Mesh.fields`.  Scalar,
vector, and quadrature fields can be added through the public API.

For staggered or monolithic solves, fields are assigned to solution
steps through the test-function registration helpers.  Calling
``finalize_test_functions()`` then builds a block layout for steps that
contain more than one field: the step's fields form a
``ufl.MixedFunctionSpace`` with one block per field, in registration order,
and the assembled matrix/vector use the same block layout
(:meth:`Mesh.get_step_block_offsets
<femcubes.core.mesh._mesh_dofs._MeshDOFs.get_step_block_offsets>`).  Every
field keeps its own storage, so single- and multi-field steps share one
code path.  Forms containing the test/trial functions of such a step are
compiled with :meth:`Mesh.Compile_Form
<femcubes.core.mesh._mesh_dofs._MeshDOFs.Compile_Form>`, which splits them
into field blocks.

Two related access patterns are important:

* :meth:`Mesh.Get_Field <femcubes.core.mesh._mesh_dofs._MeshDOFs.Get_Field>`
  returns the field's function, ready for form assembly.
* :meth:`Mesh.Get_Field_For_Output <femcubes.core.mesh._mesh_dofs._MeshDOFs.Get_Field_For_Output>`
  returns a concrete
  :class:`femcubes.core.mesh._mesh_dofs.FieldOutputView` describing where
  the field is stored.
* :meth:`Mesh.get_global_solution_dof_count <femcubes.core.mesh._mesh_dofs._MeshDOFs.get_global_solution_dof_count>`
  returns the global number of registered solution DOFs.  Live refinement
  plots display this below the mesh epoch and global cell count.

Fields on Subdomains
~~~~~~~~~~~~~~~~~~~~

Fields only need degrees of freedom where the models using them act.
``Add_Field``, ``Add_Vector_Field`` and ``Add_Quadrature_Field`` accept a
``domain`` (default ``"domain"``, the full mesh).  A field registered for
several domains lives on their union; a field whose union covers every
cell lives on the full mesh, otherwise on a submesh of those cells (see
:meth:`Mesh.Get_Field_Support
<femcubes.core.mesh._mesh_dofs._MeshDOFs.Get_Field_Support>`).  Fields on
different (sub)meshes can be combined in one monolithic step and in one
form; :meth:`Mesh.Compile_Form
<femcubes.core.mesh._mesh_dofs._MeshDOFs.Compile_Form>` supplies the
required entity maps.

Models restricted to a ``"domain"`` (see :doc:`basemodel`) should:

* register their fields with ``domain=self.domain``;
* integrate with ``dx = mesh.Get_Domain_Integrator(self.domain)``;
* restrict field updates with ``domain=self.domain`` in
  ``Set_From_Expression``, ``Copy_Field``, ``Zero_Field`` and
  ``Constant_Field``.  When several models share a field (e.g. the
  displacements of elastic models on neighbouring domains), nodes on their
  interface are owned by the model registered first, so every node is
  updated exactly once (:meth:`Mesh.Get_Field_Dofs
  <femcubes.core.mesh._mesh_dofs._MeshDOFs.Get_Field_Dofs>`).

Expressions may mix parent-mesh and subdomain fields (e.g. a degraded
stress); they are evaluated on the subdomain.  For output, subdomain fields
are NaN outside their support.  Plots show them on their subdomain only,
outlined by the full mesh, unless ``params.Plot_Subdomains_Only = False``.

Dirichlet boundary conditions on subdomain fields are applied on the
field's own mesh; boundaries outside the field's subdomain are rejected.
Boundary terms (fluxes, tractions) integrate with
:meth:`Mesh.Get_Field_Boundary_Integrator
<femcubes.core.mesh._mesh_bc._MeshBC.Get_Field_Boundary_Integrator>`, which
applies the same check to the field they act on.  Subdomain and interface
fields are refined with the rest of the mesh (see `Adaptive Mesh
Refinement`_).

Axisymmetric Meshes
~~~~~~~~~~~~~~~~~~~

Setting ``params.Axisymmetric = True`` on a 2-D mesh makes it the r-z plane
of a body of revolution about ``x = 0``: ``x`` is the radius ``r`` and ``y``
the axial coordinate.  :meth:`Mesh.Get_Domain_Integrator
<femcubes.core.mesh._mesh_dofs._MeshDOFs.Get_Domain_Integrator>`, the boundary
integrators and the lumped domain and interface weights then multiply every
integrand by ``r``, so all integrals (and reported totals such as currents)
are per radian; multiply by ``2 pi`` for the full revolution.  The weighting
is all that scalar fields (diffusion, migration, reactions) need in
cylindrical coordinates; vector fields would also need hoop terms.  Models
therefore opt in with ``SUPPORTS_AXISYMMETRY = True`` (currently the
electrochemistry models, ``Boundary:PrescribedValue`` and
``ElectroChemistry:PrescribedFlux``), and the physics rejects any other model
on an axisymmetric mesh.  Non-2-D axisymmetric meshes are rejected.

Because every residual row then carries the radius, rows near the axis are
orders of magnitude smaller than rows far from it, and one absolute
tolerance cannot suit both.  The solvers therefore measure convergence on
the residual with row ``i`` divided by the mean radius of its basis
function, ``int r N_i / int N_i`` (:meth:`Mesh.Get_Residual_Row_Scale
<femcubes.core.mesh._mesh_lumped_operators._MeshLumpedOperators.Get_Residual_Row_Scale>`;
rows of global fields use the mean radius of the domain), so ``abs_tol``
keeps the meaning it has on plane meshes.  The Newton updates are unchanged.

Global Fields
~~~~~~~~~~~~~

Some unknowns are single values for the whole problem, such as the
potential of a metal that conducts perfectly.
:meth:`Mesh.Add_Global_Field
<femcubes.core.mesh._mesh_dofs._MeshDOFs.Add_Global_Field>` registers one;
it gets test and trial functions and is solved monolithically with the
mesh fields of its step (one extra row and column, owned by one rank).
Models contribute to its equation algebraically in ``Add_KF``:

.. code-block:: python

   row = mesh.Get_Global_Field_Dof("E_m", step)     # same on every rank
   value = mesh.Get_Global_Value("E_m")              # same on every rank
   F.setValues([row], [local_residual], addv=PETSc.InsertMode.ADD_VALUES)
   K.setValues([row], columns, local_derivatives, addv=PETSc.InsertMode.ADD)

Steps without a weak-form term for the field get a zero diagonal block
automatically.  Global fields have no spatial representation:
expressions, boundary conditions, indicators and field output reject
them (report them through ``params.Global_Measures``); refinement keeps
their value.

Fields on Interfaces
~~~~~~~~~~~~~~~~~~~~

A facet group can also be used as ``domain``, e.g. ``domain="interface"``.
Such an interface field (for instance the adsorbed-hydrogen coverage at a
metal-electrolyte interface) lives on a lower-dimensional mesh of those
facets only, and can be solved monolithically together with fields on
either side.  Its own weak-form terms integrate over the interface mesh,
``ufl.Measure("dx", domain=field.function_space.mesh)``; constants used
there must also live on that mesh.

Interface reactions coupling fields from both sides are integrated with
lumped integration through :meth:`Mesh.Get_Interface_Weights_Dofs
<femcubes.core.mesh._mesh_lumped_operators._MeshLumpedOperators.Get_Interface_Weights_Dofs>`,
which returns, for the nodes of the locally owned interface facets, their
weights and the DOFs and values of every requested field.  Fields are found
wherever they exist: on either side or on the interface itself.  A field
defined on both sides is continuous and has one DOF per node, so models
never need to know on which side a field lives.

Interface terms can also be integrated at Gauss points with
:meth:`Mesh.Interface_Integral
<femcubes.core.mesh._mesh_dofs._MeshDOFs.Interface_Integral>`, which returns
``integrand * dS`` over an interior facet group.  The integrand is written
with unrestricted fields: each field (or test/trial function) defined on one
side only is evaluated from that side, while fields on both sides or on the
interface itself are evaluated from the ``normal_from`` side (by default the
first adjacent domain of the mesh generator).  ``ufl.FacetNormal`` points out
of ``normal_from``:

.. code-block:: python

   n = ufl.FacetNormal(mesh.mesh)
   F_interface = mesh.Interface_Integral(k * (c - h) * q, "interface")
   flux = mesh.Interface_Integral(ufl.dot(ufl.grad(c), n), "interface", normal_from="domain:left")

Dirichlet boundary conditions on an interface field use its own facet group
(constraining the whole interface field) or another boundary, which
constrains the field where the interface meets that boundary (e.g. the end
points of a line interface).

Interface fields are written to HDF5 on their own facet mesh (see
`Output files across mesh epochs`_), and to VTK as a separate series per
interface, ``Outputs_<interface>_mesh_epoch_XXXX[_step_<step>].pvd``.  The
interface name joins the field's facet groups with ``+`` and replaces
``:`` by ``_`` (e.g. ``metal_pit+metal_outer``).
Plots show them as thick lines on the interface.

Adaptive Mesh Refinement
------------------------

FEMCubes supports refinement-only adaptive meshing driven by a registered
scalar field.  The normal entry point is ``TimeSolver``: it evaluates the
indicator after a converged step solve, installs any refined mesh as an atomic
transaction, refreshes the physics and solver state, and re-solves the same
time increment before allowing it to be committed.

Configuration
~~~~~~~~~~~~~

Set ``params.MeshRefinement`` to a
:class:`~femcubes.core.mesh.refinement.RefinementConfig`.  Refinement is
disabled by default.  This phase-field example checks the indicator every
step, permits repeated refinement/re-solve cycles within that step, keeps a
four-cell moving buffer around the active criterion, and optionally
redistributes an imbalanced MPI mesh:

.. code-block:: python

   from femcubes.core.params_template import ParamsTemplate
   from femcubes.core.mesh.refinement import (
       FieldThresholdIndicatorConfig,
       RefinementConfig,
   )

   class Params(ParamsTemplate):
       MeshRefinement = RefinementConfig(
           enabled=True,
           indicator=FieldThresholdIndicatorConfig(
               field="phasefield",
               threshold=0.25,
               reduction="cell_max",
               comparison="greater_equal",
           ),
           target_h=0.01,
           buffer_layers=4,
           check_interval=1,
           max_levels_per_step=4,
           max_events=50,
           start_step=0,
           repartition=True,
           repartition_imbalance_threshold=1.5,
           replot_on_refinement=True,
           save_remesh_animations=True,
       )

``RefinementConfig`` fields
   ``enabled``
      Enables scheduled refinement.  Both ``indicator`` and ``target_h`` are
      required when true.

   ``indicator``
      A :class:`~femcubes.core.mesh.refinement.FieldThresholdIndicatorConfig`
      selecting a registered scalar CG, DG, or quadrature field.

   ``target_h``
      Positive target cell diameter. Cells already at or below this size,
      within floating-point tolerance, are not directly marked again. Only
      edges longer than this target are requested for splitting; conforming
      closure can split additional edges and cells, including smaller ones.

   ``buffer_layers``
      Physical padding radius in units of ``target_h``. For example,
      ``buffer_layers=3`` with ``target_h=2`` adds a distance of 6 around the
      criterion region, independent of current cell sizes. Zero disables
      buffering.

   ``check_interval`` and ``start_step``
      Attempt refinement every ``check_interval`` accepted step indices,
      beginning with ``start_step``.

   ``max_levels_per_step``
      Maximum number of successful refine/re-solve cycles for one time
      increment.

   ``max_events``
      Optional global limit on successful refinement epochs.  ``None`` leaves
      the total unlimited.

   ``repartition`` and ``repartition_imbalance_threshold``
      Enable post-transfer MPI redistribution when
      ``max_owned_cells / mean_owned_cells`` is strictly greater than the
      threshold.  The threshold must be finite and at least one.

   ``replot_on_refinement``
      Refresh the persistent mesh window after each successful installation.

   ``save_remesh_animations``
      Record the mesh epochs to ``mesh_refinement.mp4``.  This is independent
      of ordinary field-movie recording and does not require an interactive
      window.

   ``interpolation_padding``
      Legacy nonmatching-search padding, retained for existing configurations
      with a default of ``1e-12``. Refinement transfers now use known parent
      cells and do not perform geometric searches, so this setting has no
      effect on transferred values.

``FieldThresholdIndicatorConfig`` fields
   ``field``
      Registered field name.  For example, ``phasefield`` can track existing
      damage, while a scalar quadrature field such as ``PsiField`` can refine
      from a fracture driving force.

   ``threshold`` and ``comparison``
      Finite threshold and one of ``greater_equal``, ``greater``,
      ``less_equal``, or ``less``.

   ``reduction``
      Convert cell-local field samples to one value using ``cell_max``,
      ``cell_min``, or ``cell_mean``.

   ``use_absolute_value``
      Apply the reduction to absolute sample values when true.

The buffer footprint uses the Euclidean distance between whole simplex cells.
A cell is selected when its distance from any threshold-selected cell is at
most ``buffer_layers * target_h``, and its diameter exceeds ``target_h``.
Threshold cells continue to seed the buffer after reaching target size.
This also allows the padding to cross already fine regions and MPI partition
boundaries without depending on the number of intervening cells.

Whole coarse cells intersecting the padding are selected, so their geometry
can extend beyond the exact radius. Conforming refinement adds a transition
region outside the requested footprint. This refinement-only scheme does not
coarsen regions refined during earlier events.

Direct callers of ``expand_refinement_buffer`` must supply the keyword
``target_h``. Direct ``refine_mesh_topology`` calls should pass a
``RefinementConfig(target_h=...)`` to enable size-aware edge selection;
without a target, explicit low-level cell marks split all incident edges.

Supported meshes and field transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current topology path is refinement-only: coarsening and moving-mesh
updates are not supported.  It accepts simplex cells in their corresponding
dimensions: intervals in 1-D, triangles in 2-D, and tetrahedra in 3-D.
Quadrilateral, hexahedral, prism, and pyramid meshes are rejected explicitly.
Indicators must be scalar, although transferred fields may be scalar, vector,
or tensor valued.

A successful remesh preserves every registered field and named mesh group:

* mixed solution values are synchronized into their registered standalone
  fields before the transaction starts;
* CG and DG functions are recreated with the same family, degree, and value
  shape; unchanged cells retain their coefficients directly, while children
  of split cells use DOLFINx evaluation in their known parent cells followed
  by the destination element's interpolation operator;
* quadrature samples are copied directly on unchanged cells. For split
  children, source samples are reconstructed cell-locally into the largest
  stable complete DG polynomial space supported by their integration points,
  then evaluated at the new quadrature points;
* cell and facet tags, together with their stable domain and boundary names,
  are transferred and checked by global geometric measure; and
* test functions, trial functions, mixed spaces, and mesh-owned caches are
  rebuilt for the new epoch.

Quadrature reconstruction is automatic for all registered quadrature fields
and supports their full scalar/vector/tensor value shape.  It is a weighted
polynomial reconstruction, not a strictly conservative projection; the
refinement feature currently promises no conservation property for CG, DG, or
quadrature transfers.  Derived fields may subsequently be overwritten when
the current increment is solved again.

The split/unchanged classification uses actual parent-cell multiplicities,
so cells split by conformity closure are also interpolated. Ancestry is
resolved for both owned and ghost destination cells using DOLFINx distributed
vectors. Unchanged DG coefficients and arbitrary quadrature samples are
preserved exactly; shared CG coefficients can incur interpolation roundoff
when synchronized with a rank that only sees split neighbours.

Quadrature reconstruction runs in bounded batches and only processes parents
of split children. Unchanged quadrature samples are never passed through
polynomial fitting. Interpolation also runs in bounded batches, using native
DOLFINx function evaluation and interpolation matrices. The known ancestry
eliminates geometric searches and point-ownership discovery. Source ghost
coefficients are synchronized before evaluating ghost children.

At a coarse-cell face, a discontinuous field is evaluated using the child's
actual parent. This preserves the parent-side trace, including for quadrature
rules containing face points; the former geometric search could select a
neighbouring coarse cell's trace instead.

Fields restricted to subdomains or interfaces (see `Fields on Subdomains`_)
are refined too: ``snapshot_registered_fields`` lifts them onto full-mesh
functions (zero outside their support), they are transferred as above,
and the installation restricts them back onto the submeshes of the refined
mesh, both for :meth:`Mesh.Refine_From_Indicator
<femcubes.core.mesh.mesh.Mesh.Refine_From_Indicator>` and for detached
states installed with :meth:`Mesh.Install_Refined_State
<femcubes.core.mesh.mesh.Mesh.Install_Refined_State>`.  Values on the
support are transferred exactly as for full-mesh fields.  Interface fields
(CG or DG of any degree) are lifted through one adjacent cell per facet,
which interpolates the interface value at the projection onto its facet,
and restored by an exact L2 projection of that cell's trace; a cell with
two facets on one interface (a sharp corner) is rejected.  Quadrature
interface fields are not supported.  Global fields are not transferred and
keep their value.  A restricted indicator field only marks cells of its
own subdomain; interface and global fields cannot be indicators.

MPI partitioning
~~~~~~~~~~~~~~~~

The first refined mesh deliberately retains parent-rank ownership.  This
identity-partitioned intermediate state keeps DOLFINx parent ancestry local
while cell/facet tags and all fields are transferred.  Shared-facet ghosting
is retained.

With ``repartition=True``, FEMCubes then measures the complete intermediate
mesh.  If its owned-cell imbalance exceeds
``repartition_imbalance_threshold``, the unchanged topology is redistributed
with DOLFINx's shared-facet graph partitioner.  Tags and all fields are moved
to that balanced mesh, and only this final detached state is installed.  A
failure during either transfer or redistribution therefore cannot partially
replace the live mesh.  Repartitioning is not attempted in serial and remains
disabled by default.  With it disabled, highly localized refinement can
produce substantial rank imbalance.

Redistribution uses exact cell coefficient migration, not a second
interpolation. DOLFINx's original input-cell indices identify corresponding
cells, and a reusable ``IndexMap`` with distributed-vector forward scatters
sends only requested remote cell blocks. Retained cells are copied locally;
destination CG ghosts are synchronized normally. Quadrature samples, DG
history, and CG coefficients are preserved without reconstruction. This
requires the unchanged reference-cell ordering supplied by zero-edge
DOLFINx refinement, which is checked before any fields are migrated.

Graph partition labels are matched to the existing ranks to maximize the
number of cells that retain their owner. This removes unnecessary whole
partition swaps without changing the graph cut or load balance. It optimizes
rank labels for the partitioner's chosen regions; it does not globally
minimize boundary movement or provide adaptive graph repartitioning.

Refinement timings
~~~~~~~~~~~~~~~~~~

After a successful remesh, ``TimeSolver`` reports the total wall time from
the start of the refinement transaction through model refresh and solver
cleanup, before re-solving the increment. A separate timing line breaks down
preparation (validation and field snapshot), indicator evaluation, buffer
construction, topology and mesh statistics, marker transfer, the first CG/DG
and quadrature transfers, redistribution, installation, and model refresh.
Redistribution includes migration of the already-refined fields. Installation
includes visualization, whose time is also shown separately in parentheses.

Two additional lines break down the initial CG/DG and quadrature transfers:

* ``spaces`` includes function-space construction and function allocation;
* ``search/setup`` is zero for transfers using refinement ancestry;
* ``reconstruction`` measures the quadrature-to-DG reconstruction operator
  setup and application;
* ``interpolation`` measures child interpolation-point generation, coarse
  function evaluation in known parents, and destination interpolation;
* ``copy`` measures direct coefficient copying on unchanged cells (or the
  complete array copy for a no-op refinement);
* ``synchronization`` measures source and destination forward scatters;
* ``diagnostics`` measures global coefficient statistics and scalar integrals;
  and
* ``other`` covers the remaining validation, collective checks, and bookkeeping.

Detailed rank-local times for individual fields are retained in
``result.initial_transfer_diagnostics``, even when repartitioning subsequently
migrates the fields. Ancestry setup and geometry validation are included in
the enclosing transfer stage's ``other`` time.

Each printed time is the maximum across mesh ranks. Maxima for individual
stages may come from different ranks, so their sum need not equal the total.
The timings stored in ``RefinementInstallResult`` and ``RefinementReport``
remain rank-local; ``result.total_seconds`` covers the mesh transaction only,
before the solver's model refresh. A direct ``Install_Refined_State`` call
records installation time alone. Subsequent form compilation, assembly, and
the new solve after refresh are outside the reported refinement total.

Solver and model lifecycle
~~~~~~~~~~~~~~~~~~~~~~~~~~

The public transaction is
:meth:`Mesh.Refine_From_Indicator <femcubes.core.mesh.mesh.Mesh.Refine_From_Indicator>`.
It returns a
:class:`~femcubes.core.mesh.refinement.RefinementInstallResult`; an empty
collective mark set is a no-op and does not advance ``Mesh.remesh_epoch``.
All mesh ranks participate in indicator expression setup, including ranks
with no owned cells; those ranks return empty local indicator arrays.
Normal transient simulations should let ``TimeSolver`` call this method.
After a successful installation, ``TimeSolver`` calls
:meth:`Physics.Post_Remesh <femcubes.core.physics._physics_models._PhysicsModels.Post_Remesh>`,
destroys stale solver workspaces, and re-runs the current increment at the
unchanged ``t`` and ``dt``.  Only that new solve can be committed.

Every model derives from :class:`~femcubes.core.base_model.BaseModel` and
inherits a no-op
:meth:`BaseModel.Post_Remesh <femcubes.core.base_model.BaseModel.Post_Remesh>`.
Future model authors must override it whenever the instance retains objects
tied to the old mesh, including UFL expressions/forms, direct function or
function-space references, ``fem.Constant`` objects, boundary conditions,
point-location structures, or model-owned PETSc data.  Recreate those objects
from ``self.mesh`` or discard them for lazy rebuilding; retain scalar runtime
history that is independent of the mesh.  Do not call ``initialize_fields``
from this hook because transferred solution and history values are already
installed.

Code that calls ``Mesh.Refine_From_Indicator`` outside ``TimeSolver`` owns the
same responsibilities: after ``result.installed`` is true, call
``Physics.Post_Remesh()`` and discard any solver-local matrices, vectors,
compiled forms, and preconditioners before solving again.

Remesh plotting and animation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``replot_on_refinement=True`` uses the ordinary persistent ``plot_mesh``
window.  The window and camera are retained while its actors are rebuilt for
the new topology; gathered MPI contributions are rendered in one final update
rather than rank by rank.

``save_remesh_animations=True`` writes one continuous
``mesh_refinement.mp4`` to ``params.movie_folder`` or, when that is unset, to
``<output_dir>/movies``.  The movie starts with epoch zero and appends exactly
one frame for each successful refinement.  No-op attempts add no frame.  It is
separate from ``record_movie`` and the ordinary field movies, uses
``params.movie_fps``, works off-screen, and is finalized through the ordinary
``Mesh.Close_Plots()`` lifecycle.

Output files across mesh epochs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VTK time series are segmented at topology changes because one VTK series
cannot safely change its mesh.  With ``vtk_single_file=True``, files are named
``Outputs_mesh_epoch_XXXX.pvd``; per-step VTK output uses
``Outputs_mesh_epoch_XXXX_step_<step>.pvd``.  Plot windows and movies are not
segmented.

HDF5 geometry is immutable within an epoch and is stored in
``mesh_epoch_XXX.hdf5``.  Each ``Outputs_<step>.hdf5`` field snapshot contains
a top-level scalar integer ``epoch_tag`` identifying the matching geometry
file.  The geometry file contains the same tag and one ``Mesh_DG<degree>``
group for each DG output degree used in that epoch.  ``time_data.hdf5`` also
includes ``epoch_tag`` in ``data_names`` and stores the corresponding value in
every history row.

A downstream reader should therefore:

1. read ``epoch_tag`` from ``Outputs_<step>.hdf5``;
2. open ``mesh_epoch_{epoch_tag:03d}.hdf5``;
3. select ``Mesh_DG<dg_degree>`` using the field dataset's ``dg_degree``
   attribute; and
4. pair that geometry with the field data from ``Fields/<field_name>``.

Never infer the mesh epoch from the output step number or assume all output
snapshots share the initial geometry.

Interface fields (see `Fields on Interfaces`_) have no cells of the full
mesh.  They are stored under ``InterfaceFields/<field_name>`` with one row
per interface facet, and their geometry is the group named by the dataset's
``mesh_group`` attribute, ``Interface_<interface>_DG<degree>`` (named as for
VTK output), in the same
geometry file.

Public refinement API
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: femcubes.core.mesh.refinement.FieldThresholdIndicatorConfig
   :members:

.. autoclass:: femcubes.core.mesh.refinement.RefinementConfig
   :members:

.. autoclass:: femcubes.core.mesh.refinement.RefinementInstallResult
   :members:

.. autoclass:: femcubes.core.mesh.refinement.RefinementReport
   :members:

.. autofunction:: femcubes.core.mesh.refinement.migrate_repartitioned_fields

.. automethod:: femcubes.core.mesh.mesh.Mesh.Refine_From_Indicator

Output Preparation
------------------

Plotting and file output share one common preparation layer:

.. autoclass:: femcubes.core.mesh._mesh_dofs.ProjectedFieldOutput
   :members:

.. autoclass:: femcubes.core.mesh._mesh_dofs.FieldOutputView
   :members:

The key public entry point is:

.. automethod:: femcubes.core.mesh._mesh_dofs._MeshDOFs.Prepare_Field_For_Output

``Prepare_Field_For_Output()`` accepts either:

* a scalar field name such as ``"temperature"``
* a vector component specifier such as ``"u_x"``
* a tensor component specifier such as ``"sigma_xy"``

and returns a scalar DG projection that can be consumed by plotting or
file-output backends.

Plotting and File Output
------------------------

The mesh class also owns the user-facing plotting and output routines:

* :meth:`Mesh.plot_mesh <femcubes.core.mesh._mesh_plotting._MeshPlotting.plot_mesh>`
* :meth:`Mesh.Plot_Fields <femcubes.core.mesh._mesh_plotting._MeshPlotting.Plot_Fields>`
* :meth:`Mesh.Plot_RO_Fields <femcubes.core.mesh._mesh_reduced_order_plotting._MeshReducedOrderPlotting.Plot_RO_Fields>`
* :meth:`Mesh.Plot_Measures <femcubes.core.mesh._mesh_plotting._MeshPlotting.Plot_Measures>`
* :meth:`Mesh.Write_Outputs <femcubes.core.mesh._mesh_outputs._MeshOutputs.Write_Outputs>`
* :meth:`Mesh.close_outputs <femcubes.core.mesh._mesh_outputs._MeshOutputs.close_outputs>`

Plotting is primarily intended for debugging and interactive
development.  Current implementations gather rank-local mesh or field
data to rank 0 before rendering.

On 3-D meshes, ``Plot_Fields`` colours only the outer surface of the
domain, which hides cracks and other features inside it.
``Plot_Isosurfaces`` adds a window per field that shows its isosurfaces
inside a translucent outer surface, for example the crack surface as the
isosurface :math:`\phi = 0.5` of the phase field:

.. code-block:: python

    Plot_Isosurfaces = {"phasefield": 0.5}    # {field: level} or {field: [level, ...]}

The window and its image and movie are named ``<field>_isosurface``; the
field does not need to be in ``Plot_Fields``. Cell-wise fields are averaged
to the points before the isosurface is computed. The entries are ignored
on 1-D and 2-D meshes.

Lumped Operators
----------------

For explicit or semi-implicit schemes that need lumped operators, the
mesh class provides:

* :meth:`Mesh.Get_Weights_Dofs <femcubes.core.mesh._mesh_lumped_operators._MeshLumpedOperators.Get_Weights_Dofs>`
* :meth:`Mesh.Get_Old_For_Lumped <femcubes.core.mesh._mesh_lumped_operators._MeshLumpedOperators.Get_Old_For_Lumped>`

These helpers return owned local DOF slices and values, which makes them
convenient for MPI-aware lumped updates.  Vector fields are returned in
the same flattened component-DOF order used by the underlying PETSc
vectors.

Mesh Generators
---------------

Mesh geometry is produced by generator classes registered under a
``MeshType`` string; selecting a generator means setting
``params.MeshType`` before constructing the mesh.  The generators
themselves live in their own package, :mod:`femcubes.mesh_generators`,
documented in :doc:`../mesh_generators` (including how to add one).  Core
only holds their shared infrastructure: the base class with its registry,
and the structured :class:`femcubes.core.mesh.generator_base.GeneratedMeshResult`
containing the mesh, tags, and named marker maps.

.. automodule:: femcubes.core.mesh.generator_base
   :members:
   :show-inheritance:

API Reference
-------------

.. autoclass:: femcubes.core.mesh.mesh.Mesh
   :members:
   :show-inheritance:
