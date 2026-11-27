Model Registry
==============

The :class:`femcubes.core.base_model.BaseModel` class is the common
foundation for all constitutive, transport, damage, and boundary models
in FEMCubes.

Overview
--------

Each concrete model defines a unique ``MODEL_TYPE`` string such as
``"Solid:LinearElastic"`` or ``"Boundary:PrescribedValue"``.  When the
class is imported, :meth:`BaseModel.__init_subclass__` automatically
registers it into the global model registry.

That registry is used by
:class:`femcubes.core.physics.physics.Physics` to instantiate models
from ``params.Models``.

Every model may name the (sub)domain it acts on through a ``"domain"``
entry in its model settings, e.g. ``"domain": "domain:centre"``; it is
available as ``self.domain`` and defaults to the full ``"domain"``.
All bulk models register their fields for, integrate over, and update
fields on that domain only (see the mesh documentation on fields on
subdomains).  Boundary models register no fields: they act on the field
they constrain or load wherever that field lives, using
``mesh.Get_Field_Boundary_Integrator`` or Dirichlet conditions, which
reject boundaries outside the field's support; their optional ``"domain"``
additionally requires the boundary to bound that domain.  Interface models
(e.g. ``ElectroChemistry:MetalElectrolyteInterface``) take a facet group as
``"domain"``.

Typical Subclass Pattern
------------------------

.. code-block:: python

   from femcubes.core.base_model import BaseModel

   class MyModel(BaseModel):
       MODEL_TYPE = "Solid:MyModel"

       def __init__(self, name, params, mesh):
           super().__init__(name, params, mesh)
           # register fields, cache material properties, etc.

       def Get_Weak_Form(self, time_constant, dt_constant, step):
           # Build and return stable UFL K, F expressions, or None.
           ...

       def Update_Weak_Constants(self, time_constant, dt_constant, step):
           # Update Constants, Functions, and constitutive state.
           return False  # Return True only when the UFL graph must be rebuilt.

Lifecycle Hooks
---------------

Concrete subclasses may override:

* ``initialize_fields()`` for initial conditions
* ``get_bcs_for_field()`` for Dirichlet constraints
* ``Get_Weak_Form()`` for stable weak-form contributions
* ``Update_Weak_Constants()`` for per-assembly coefficient and state updates
* ``commit()`` for committed/history updates
* ``reset_step()`` for rejected time-step rollback

Physics owns mesh-bound ``time_constant`` and ``dt_constant`` values and
updates them before each assembly. Models should use those coefficients in
their UFL forms instead of embedding the current Python ``t`` or ``dt``
values. ``Update_Weak_Constants()`` requests a full form refresh by returning
``True``; ordinary changes to existing ``Function`` and ``Constant`` values
should return ``False``.

.. automodule:: femcubes.core.base_model
   :members:
   :show-inheritance:
