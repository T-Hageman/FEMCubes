FEMCubes Documentation
======================

**FEMCubes** is a modular multi-physics finite-element framework built on
`FEniCSx <https://fenicsproject.org/>`_. A simulation is assembled from
self-contained physics models: a parameter class names the mesh, the
materials and the models, and the framework takes care of assembly, solution,
time stepping, mesh refinement and output.

The package provides models for solid mechanics and fracture
(:doc:`api/solids`), glaciology (:doc:`api/glaciology`), electrochemistry
(:doc:`api/electrochemistry`) and poromechanics (:doc:`api/poromechanics`),
together with boundary conditions (:doc:`api/boundary`) and parametric mesh
generators (:doc:`api/mesh_generators`).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   getting_started
   api/index
   examples/index
   developer/index
   publications
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
