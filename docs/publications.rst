Publications
============

Citing FEMCubes
---------------

If you use FEMCubes in your work, please cite the software with the metadata
in `CITATION.cff <https://github.com/T-Hageman/FEMCubes/blob/main/CITATION.cff>`_
(the "Cite this repository" button on GitHub), together with the paper that
describes the models you used.

Papers using FEMCubes
---------------------

* T. Hageman. Fracture and creep as a single constitutive problem:
  strength-based phase-field modelling of nucleation, propagation and creep
  relaxation. Submitted to *Computer Methods in Applied Mechanics and
  Engineering* (2026). arXiv preprint: added soon.
  Input files: `Paper_Cohesive_ViscoElastic <https://github.com/T-Hageman/Paper_Cohesive_ViscoElastic>`_.

  Models: :doc:`api/solids/cohesive_viscoelastic` (reduced local update) and
  :doc:`api/solids/cohesive_viscoelastic_kernel` (the C++ kernel); examples
  :doc:`examples/fracture/viscoelastic_fracture` and :doc:`examples/glaciology/ice_cliffs`.

* T. Hageman. Cohesive phase-field fracture with an explicit strength
  surface: an eigenstrain-based return-mapping formulation. *Engineering
  Fracture Mechanics* 343 (2026) 112292.
  `doi:10.1016/j.engfracmech.2026.112292 <https://doi.org/10.1016/j.engfracmech.2026.112292>`_.
  Code: `Cohesive-PhaseField-FenicsX <https://github.com/T-Hageman/Cohesive-PhaseField-FenicsX>`_
  (an earlier version of the same code, under a different name).

  Models: :doc:`api/solids/cohesive_linear_elastic`; example
  :doc:`examples/fracture/cohesive_phasefield`.
