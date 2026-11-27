Getting Started
===============

Installation
------------

Create the tested Conda environment and install FEMCubes in editable mode:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate FenicsX
   python -m pip install --no-build-isolation -e .

``environment.yml`` pins every direct dependency to the version FEMCubes is
tested with. On a compatible Linux x86-64 machine, ``environment.lock.yml``
can be used in place of ``environment.yml`` to also reproduce all transitive
versions and Conda build strings. FEMCubes runs on Linux, macOS, and Windows
through WSL2.

The install step compiles the C++ integration-point kernels and takes the
version number from the git tag, so install from a git clone (or from an
archive downloaded from GitHub). After pulling changes to the C++ sources,
rebuild with ``bash scripts/reinstall.sh``.

The C++ build uses an installed Eigen package version 3.3 or newer, including
Eigen 5.x. If Eigen is absent, CMake downloads Eigen 3.4.0 from GitLab.
The provided environment installs Eigen, so this fallback is unnecessary.

The environment already contains the test and documentation tools. To add
them to another compatible environment instead:

.. code-block:: bash

   python -m pip install --no-build-isolation -e ".[test,docs]"


Quick Example
-------------

.. code-block:: python

   from femcubes import Mesh, Physics, TimeSolver, ParamsTemplate, LinearSolver
   import femcubes.solids
   import femcubes.boundary

   class Params(ParamsTemplate):
       dim = 2
       MeshType = "Rectangle"
       dx = 0.1
       Lx = 1.0
       Ly = 1.0
       Lz = 0.1
       # ... define Materials, Models, solver settings, etc.

   params = Params()
   mesh = Mesh(params)
   physics = Physics(mesh, params)
   physics.Initialize_Fields()
   solver = TimeSolver(params, physics)
   solver.run()

Importing a model package (``femcubes.solids``, ``femcubes.boundary``, ...)
registers its models, which are then selected by their type string in
``params.Models``. Complete, runnable set-ups are in the ``examples`` folder
of the repository, see :doc:`examples/index`. All of them run in parallel:

.. code-block:: bash

   mpirun -np 4 python examples/fracture/cohesive_phasefield.py


Project Structure
-----------------

The package is in ``src/femcubes``. Module names are snake_case; class names
and the type strings used in parameter files (``"Solid:LinearElastic"``,
``"Rectangle_Notches"``) are independent of the file names.

- **femcubes.core** -- Mesh, Physics, Solvers, ParamsTemplate, utilities
- **femcubes.solids** -- Solid mechanics and fracture models (LinearElastic,
  ViscoElastic, PhaseField, Cohesive_ViscoElastic, …) and their C++ kernels
- **femcubes.boundary** -- Boundary condition models (DisplacementControl, …)
- **femcubes.glaciology** -- Ice rheology, shallow-ice flow, friction, ocean
  pressure
- **femcubes.electrochemistry** -- Electrolyte transport and surface reactions
- **femcubes.poromechanics** -- Pore pressure and fracture flow
- **femcubes.mesh_generators** -- Parametric mesh generators

Next to the package, ``examples`` and ``benchmarks`` hold runnable set-ups,
``scripts`` the helper scripts for installing, testing and building the
documentation, and ``docs`` this documentation. New physics can be developed
outside the package as a plugin, see :doc:`developer/plugins`.
