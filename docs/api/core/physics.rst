Physics
=======

The physics layer bridges user-configured models and the solver stack.

Overview
--------

The public :class:`femcubes.core.physics.physics.Physics` object:

* instantiates all models listed in ``params.ModelNames``
* lets each model register fields and test functions on the mesh
* finalizes the block layouts of multi-field steps once all models are
  constructed
* assembles global tangent / residual forms for the active solve step;
  multi-field steps are assembled as one monolithic block system
* collects boundary conditions contributed by boundary models
* adds model-specific algebraic terms (``Add_KF``), such as lumped
  reaction terms
* forwards lifecycle operations such as ``Initialize_Fields()``,
  ``commit()``, and ``Reset_Step()``

Models adding algebraic terms receive the assembled ``K`` and ``F``.  The
DOF indices they obtain from the mesh (``Get_Weights_Dofs``,
``Get_Interface_Weights_Dofs``) index the local residual vector; matrix
entries must be added with global indices from
``mesh.get_step_global_indices(step, indices)`` and ``K.setValues``,
because the local numbering of block matrices differs from that of the
vector.  Couplings without any weak-form term may extend the sparsity
pattern during the first assembly; afterwards the pattern is fixed.

Typical Usage
-------------

.. code-block:: python

   mesh = Mesh(params)
   physics = Physics(mesh, params)
   physics.Initialize_Fields()
   K, F = physics.Get_Kmat_Fvec(step=0)

For residual-only algorithms such as l-BFGS line searches, assemble the
force vector without assembling a stiffness matrix:

.. code-block:: python

   F = physics.Get_Fvec(step=0)

Internal Architecture
---------------------

The public class is composed from three internal mix-ins:

* ``_PhysicsBase`` for shared state
* ``_PhysicsModels`` for model construction and lifecycle dispatch
* ``_PhysicsKF`` for global stiffness/residual assembly and BC handling

.. autoclass:: femcubes.core.physics.physics.Physics
   :members:
   :show-inheritance:

.. autoclass:: femcubes.core.physics._physics_base._PhysicsBase
   :members:

.. autoclass:: femcubes.core.physics._physics_models._PhysicsModels
   :members:

.. autoclass:: femcubes.core.physics._physics_kf._PhysicsKF
   :members:
