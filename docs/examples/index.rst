Examples
========

Runnable example scripts are located in the ``examples/`` directory of the
repository, grouped by field (``fracture``, ``glaciology``,
``electrochemistry``); performance benchmarks are in ``benchmarks/``. Each
page below introduces the problem solved by one script, describes its
set-up and how to run it, and discusses sample output obtained with it.

All examples accept the same options:

``--case NAME``
   case to run, for scripts with several cases (``--help`` lists them);
``--quick``
   a short, coarse run that only checks that the script works; it does not
   reproduce the results;
``--no-plots``
   do not create or save figures;
``--output-dir DIR``
   folder for the figures and result files (default:
   ``results/<script>[/<case>]`` next to the script).

Some scripts have further options, such as the loading rate or the
phase-field length scale. Every script can be run from any folder, in serial
or with MPI (``mpirun -n N python ...``), and offers ``build_params(...)``
and ``run(...)`` functions, so its set-up can be reused from Python. The run
times given on the pages were measured on one workstation and are only a
guide.

The smoke tests in ``examples/test_examples.py`` run every case with
``--quick`` on two MPI ranks (``pytest examples/test_examples.py -n 8``).

.. toctree::
   :maxdepth: 1
   :caption: Fracture

   fracture/cohesive_phasefield
   fracture/viscoelastic_fracture
   fracture/multidomain_tensile_bar

.. toctree::
   :maxdepth: 1
   :caption: Glaciology

   glaciology/ice_cliffs
   glaciology/ice_shelf_front
   glaciology/tensile_ice
   glaciology/shallow_ice

.. toctree::
   :maxdepth: 1
   :caption: Electrochemistry

   electrochemistry/corrosion_pit
   electrochemistry/electrolyte_metal

.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/scaling_cohesive_elastic
