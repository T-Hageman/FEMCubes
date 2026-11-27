``femcubes.mesh_generators`` -- Mesh Generators
===============================================

Every mesh is built by a generator selected through ``params.MeshType``.
Each generator is one module in ``src/femcubes/mesh_generators/``; importing
the package (which ``Mesh`` does) registers all of them.  The shared
generator infrastructure (base class, registry, and result type) is part
of core, see :mod:`femcubes.core.mesh.generator_base`.

.. code-block:: python

   from femcubes.core.mesh.mesh import Mesh
   from femcubes.mesh_generators.rectangle import Sample_Parameters

   params = Sample_Parameters()
   mesh = Mesh(params)

Built-In Generators
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - ``MeshType``
     - ``dim``
     - Domain
   * - :doc:`Rectangle <mesh_generators/rectangle>`
     - 2, 3
     - Rectangle or box
   * - :doc:`Rectangle_Notches <mesh_generators/rectangle_notches>`
     - 2, 3
     - Rectangle or box with rectangular notches cut into the top
   * - :doc:`Rectangle_CentreNotch <mesh_generators/rectangle_centre_notch>`
     - 2
     - Rectangle with a rectangular hole at its centre
   * - :doc:`Rectangle_RefinedTop <mesh_generators/rectangle_refined_top>`
     - 2
     - Rectangle with a finely meshed top layer
   * - :doc:`PlateWithHole <mesh_generators/plate_with_hole>`
     - 2, 3
     - Quarter of a plate with a circular hole
   * - :doc:`FullPlateWithHole <mesh_generators/full_plate_with_hole>`
     - 2, 3
     - Plate with a centred, optionally elliptical and rotated hole
   * - :doc:`HalfPlateWithHole <mesh_generators/half_plate_with_hole>`
     - 2
     - Right half of ``FullPlateWithHole``, with a symmetry boundary
   * - :doc:`SingleEdgeNotched <mesh_generators/single_edge_notched>`
     - 2, 3
     - Plate with one notch cut in from the left edge
   * - :doc:`DamageChallenge <mesh_generators/damage_challenge>`
     - 2, 3
     - Notched beam with load and support windows
   * - :doc:`Cylinder <mesh_generators/cylinder>`
     - 3
     - Cylinder with an optional circumferential groove
   * - :doc:`NotchedTensileBar <mesh_generators/notched_tensile_bar>`
     - 2
     - Notched bar split into three cell domains
   * - :doc:`TwoSquared <mesh_generators/two_squared>`
     - 2
     - Two rectangles sharing an interface, as two cell domains
   * - :doc:`CorrosionPit <mesh_generators/corrosion_pit>`
     - 2
     - Electrolyte above a metal surface with a semicircular pit
   * - :doc:`PlateWithDefect <mesh_generators/plate_with_defect>`
     - 2, 3
     - Electrolyte layer on a metal plate with a circular defect

.. toctree::
   :maxdepth: 1
   :hidden:

   mesh_generators/rectangle
   mesh_generators/rectangle_notches
   mesh_generators/rectangle_centre_notch
   mesh_generators/rectangle_refined_top
   mesh_generators/plate_with_hole
   mesh_generators/full_plate_with_hole
   mesh_generators/half_plate_with_hole
   mesh_generators/single_edge_notched
   mesh_generators/damage_challenge
   mesh_generators/cylinder
   mesh_generators/notched_tensile_bar
   mesh_generators/two_squared
   mesh_generators/corrosion_pit
   mesh_generators/plate_with_defect

Reading the Generator Pages
---------------------------

Each generator page has, per supported dimension, a dimension drawing, a
sample mesh and the parameter class that produced the sample mesh, followed
by a table of all parameters the generator reads.

* In the drawings, parameter names are the exact ``params`` attributes the
  generator reads. Names in coloured boxes are boundary groups, names in
  italics are domain (cell) groups, and the open circle marks the origin.
* The sample meshes colour the boundary facets by group, with the colours
  of the drawing. 3-D meshes are shown from two opposite sides, so that
  every boundary group is visible.
* Lengths have no fixed unit: they are in the length unit of the model
  (metres for most models). Element sizes are target sizes for gmsh.

Boundary and domain names are the supported interface; the integer marker
values behind them are not.  Names of the form ``group:part`` are
hierarchical: ``group`` selects the union of all its parts, so ``metal``
selects both ``metal:pit`` and ``metal:outer``.

Names of the 3-D faces
   The box-like 3-D generators name the faces ``front`` / ``back`` for
   ``x = 0`` / ``x = Lx``, ``left`` / ``right`` for ``y = 0`` / ``y = Ly``,
   and ``bottom`` / ``top`` for ``z = 0`` / ``z = Lz``. The 2-D names
   ``left`` and ``bottom`` therefore do not refer to the same sides in
   3-D. The 3-D drawings give the plane of every face.

Axisymmetric meshes
   With ``Axisymmetric = True`` (a mesh option for ``dim = 2``), the 2-D
   mesh is the r-z plane of a body of revolution: ``x`` is the radius and
   the axis is ``x = 0``. Geometry and group names are unchanged; see
   :doc:`core/mesh`.

The drawings and sample meshes are made by
``docs/scripts/make_mesh_generator_figures.py``, which runs every generator
with the parameter classes shown on the pages.  Run it from the repository
root after changing a generator:

.. code-block:: bash

   python docs/scripts/make_mesh_generator_figures.py

Adding a Generator
------------------

Add one module to ``src/femcubes/mesh_generators/``, with a snake_case file
name (``rectangle_notches.py`` holds ``MeshType = "Rectangle_Notches"``),
containing:

* a subclass of :class:`~femcubes.core.mesh.generator_base.MeshGeneratorBase`
  with a unique ``MESH_TYPE``; its ``generate(params)`` returns a
  :class:`~femcubes.core.mesh.generator_base.GeneratedMeshResult` (e.g.
  through ``self._result_from_gmsh(model_to_mesh(...), domain_markers=...,
  boundary_markers=...)``) and rejects unsupported dimensions and invalid
  parameters explicitly;
* a ``Sample_Parameters()`` function returning a minimal valid
  ``ParamsTemplate``.

The module is registered automatically.  Helpers shared by several
generators go in modules starting with ``_``, which are not treated as
generators.  Then add the generator to the tests in
``src/femcubes/mesh_generators/tests/test_mesh_generation.py`` (expected
generators, markers and measure, and a focused test class), and document
it: a page in ``docs/api/mesh_generators/`` with the supported dimensions,
the parameter table, the named markers and a sample ``ParamsTemplate``; a
drawing function and a case in
``docs/scripts/make_mesh_generator_figures.py``; and an entry in the table
and the toctree above.

Package API
-----------

.. automodule:: femcubes.mesh_generators
   :members:
