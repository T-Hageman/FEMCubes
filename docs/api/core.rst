``femcubes.core`` -- Core Framework
===================================

The core package provides the shared framework used by every FEMCubes
model and example.  The public workflow is:

1. define a :class:`~femcubes.core.params_template.ParamsTemplate`
   subclass
2. construct a :class:`~femcubes.core.mesh.mesh.Mesh`
3. construct a :class:`~femcubes.core.physics.physics.Physics`
4. solve with :class:`~femcubes.core.solvers.time_solver.TimeSolver`

The remaining core modules provide the registry, discovery, utility, and
internal helper layers that support that workflow.

Main Public Classes
-------------------

* :class:`femcubes.core.params_template.ParamsTemplate`
* :class:`femcubes.core.base_model.BaseModel`
* :class:`femcubes.core.mesh.mesh.Mesh`
* :class:`femcubes.core.physics.physics.Physics`
* :class:`femcubes.core.solvers.time_solver.TimeSolver`
* :class:`femcubes.core.solvers.step_solver.StepSolver`

Package Layout
--------------

``femcubes.core`` is organized into the following areas:

* ``mesh``: geometry, fields, output, plotting, and lumped operators
* ``physics``: model instantiation and global weak-form assembly
* ``solvers``: nonlinear and transient solves
* ``utils``: shared MPI and mathematics helpers
* ``BaseModel`` / ``_discover``: model registration and discovery
* ``ParamsTemplate``: common configuration defaults

.. toctree::
   :maxdepth: 2

   core/params
   core/basemodel
   core/discovery
   core/mesh
   core/physics
   core/solvers
   core/utils
