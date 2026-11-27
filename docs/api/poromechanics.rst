``femcubes.poromechanics`` -- Poromechanics Models
==================================================

Models for fluid flow in deforming porous media. :doc:`poromechanics/pore_pressure`
couples the deformation of a solid model to the pressure of a pore fluid,
with saturated or unsaturated Darcy flow; :doc:`poromechanics/fracture_flow`
adds the flow along phase-field fractures; and
:doc:`poromechanics/shared_poroelasticity` holds the saturation relations
used by both.

.. toctree::
   :maxdepth: 1

   poromechanics/pore_pressure
   poromechanics/fracture_flow
   poromechanics/shared_poroelasticity
