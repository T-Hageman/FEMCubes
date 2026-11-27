Contributing and contact
========================

FEMCubes is developed on GitHub, at
`github.com/T-Hageman/FEMCubes <https://github.com/T-Hageman/FEMCubes>`_.

Questions and bug reports
-------------------------

Questions about using FEMCubes and reports of bugs both go to the
`issue tracker <https://github.com/T-Hageman/FEMCubes/issues>`_. Please
search the existing issues first. A new issue is easiest to answer when it
includes:

* the FEMCubes version (``python -c "import femcubes; print(femcubes.__version__)"``);
* a short input file that reproduces the problem, as small and as fast as
  possible;
* the full error message or log, and the number of MPI ranks used.

Contributions
-------------

Contributions are welcome: bug fixes, documentation, examples and new
models. They are made through pull requests on GitHub and follow the
guidelines in
`CONTRIBUTING.md <https://github.com/T-Hageman/FEMCubes/blob/main/CONTRIBUTING.md>`_.
In short:

* for anything larger than a fix, open an issue first to agree on the
  approach;
* code, tests, examples and documentation are updated together, and every
  bug fix comes with a regression test;
* new physics starts as a plugin, a separate package that registers its
  models with FEMCubes (see :doc:`developer/plugins`), and moves into the
  core package once it is published.

The :doc:`developer/index` describes the structure of the code and the tests.

Maintainer
----------

FEMCubes is maintained by Tim Hageman, Department of Engineering Science,
University of Oxford. Please use the issue tracker for all questions, so that
the answers stay available to other users. To cite FEMCubes, see
:doc:`publications`.
