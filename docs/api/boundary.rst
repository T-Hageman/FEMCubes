``femcubes.boundary`` -- Boundary Condition Models
==================================================

Boundary conditions are models like the physics models: each acts on one
named boundary group and on one field of another model. Strong (Dirichlet)
conditions prescribe the value of a field, either constant
(:doc:`boundary/prescribed_value`) or growing linearly in time
(:doc:`boundary/prescribed_rate`). Penalty conditions drive a field towards
a prescribed value through a stiff spring: :doc:`boundary/displacement_control`
imposes a displacement rate and records the reaction force, as in a
displacement-controlled test, and :doc:`boundary/torsion` imposes a rigid
rotation. Natural
(Neumann) conditions apply a load, either constant
(:doc:`boundary/external_force`) or read from a table over space and time
(:doc:`boundary/external_force_from_file`).

.. toctree::
   :maxdepth: 1

   boundary/prescribed_value
   boundary/prescribed_rate
   boundary/displacement_control
   boundary/external_force
   boundary/external_force_from_file
   boundary/torsion
