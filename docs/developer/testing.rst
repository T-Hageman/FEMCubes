Testing
=======

The tests live next to the code they cover, in a ``tests`` folder of each
sub-package (for example ``src/femcubes/solids/tests``). The example scripts
are covered by smoke tests in ``examples/test_examples.py``.

Running the tests
-----------------

.. code-block:: bash

   # Fast subset: unit-level tests, a few minutes on a workstation
   python -m pytest src -m "not slow and not full_model and not mpi and not examples" -n auto

   # Full suite with coverage, in parallel
   python scripts/run_tests.py

   # One file, one class, or one test
   python -m pytest src/femcubes/solids/tests/test_linear_elastic.py
   python -m pytest src/femcubes/core/mesh/tests/test_mesh_groups.py -k "boundary"

``scripts/run_tests.py`` passes extra arguments on to pytest, and
``--not-full-model`` skips the full-model regressions. The continuous
integration runs the fast subset on every push; the full suite is run locally
before a release.

Set ``OMP_NUM_THREADS=1`` (``source scripts/limit_omp.sh``) when running the
tests in parallel, so that the worker processes do not oversubscribe the
machine.

Markers
-------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Marker
     - Meaning
   * - ``full_model``
     - Runs a complete physics model through the solver stack. These are the
       regression tests against reference solutions.
   * - ``mpi``
     - Launches ``mpirun`` to check behaviour that depends on rank ownership,
       communication, or distributed meshes.
   * - ``examples``
     - Runs an example script in a reduced configuration.
   * - ``slow``
     - Long-running test outside the other categories.

Unknown markers are an error (``--strict-markers``); add new ones to the
pytest configuration in ``pyproject.toml``.

Writing tests
-------------

* Test the public contract: boundary and subdomain names, model-type strings
  and documented parameters, not raw marker integers or private attributes.
* Mesh tests use the ``Sample_Parameters()`` of the mesh generator instead of
  local parameter objects.
* Every bug fix comes with a regression test.
* For ``numba.njit`` functions, coverage does not see the compiled body. Where
  branch coverage matters, also test the ``.py_func`` of the function.
* Behaviour that depends on the parallel layout gets a dedicated ``mpi`` test.

Coverage
--------

``scripts/run_tests.py`` reports the coverage of the package when
``pytest-cov`` is installed. Use ``# pragma: no cover`` only for defensive or
environment-dependent branches that cannot be reached in a test.
