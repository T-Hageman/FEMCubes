``femcubes.glaciology`` -- Glaciology Models
============================================

The glaciology models describe the deformation, flow and fracture of
glaciers, ice sheets, ice shelves and sea ice. Ice deforms elastically under
rapid loading and creeps under sustained stress following Glen's flow law
:cite:p:`Glen1955,CuffeyPaterson2010`, with properties that depend on its
temperature and porosity. Two scales are covered. Continuum models of an ice
body, in a vertical cross-section or in 3-D, combine viscoelasticity with
standard or cohesive phase-field fracture, for example to simulate crevasses.
The depth-integrated shallow-shelf model describes ice flow in map view
:cite:p:`Macayeal1989,Greve2009`, with the ice margin tracked by a level
set. Boundary models add the ocean pressure on ice fronts and shelf bases,
basal friction, and the drag of the water under floating ice.

Registered models (the ``Type`` string is case sensitive):

* ``Glaciology:ViscoElastic`` -- :doc:`glaciology/ice_viscoelastic`
* ``Glaciology:Cohesive_ViscoElastic`` -- :doc:`glaciology/ice_cohesive_viscoelastic`
* ``Glaciology:PhaseField`` -- :doc:`glaciology/ice_phasefield`
* ``glaciology:BasalFriction`` -- :doc:`glaciology/basal_friction`
* ``Glaciology:FloatingIceFriction`` -- :doc:`glaciology/floating_ice_friction`
* ``glaciology:OceanPressure`` -- :doc:`glaciology/ocean_pressure`
* ``glaciology:LoweringOceanPressure`` -- :doc:`glaciology/lowering_ocean_pressure`
* ``Glaciology:ShallowIce`` -- :doc:`glaciology/shallow_ice`
* ``Glaciology:ShallowIceBC`` -- :doc:`glaciology/shallow_ice_bc`
* ``Glaciology:LevelSet`` -- :doc:`glaciology/level_set`

The ice models evaluate their material parameters from porosity and
temperature with the helper class in :doc:`glaciology/ice_material`.

.. toctree::
   :maxdepth: 1

   glaciology/ice_viscoelastic
   glaciology/ice_cohesive_viscoelastic
   glaciology/ice_phasefield
   glaciology/basal_friction
   glaciology/floating_ice_friction
   glaciology/ocean_pressure
   glaciology/lowering_ocean_pressure
   glaciology/shallow_ice
   glaciology/shallow_ice_bc
   glaciology/level_set
   glaciology/ice_material

References
----------

.. bibliography::
   :filter: docname in docnames
