Utilities
=========

The utility layer contains shared helpers used throughout the core
framework.

``maths_utils``
---------------

Provides tensor conversion helpers, simple constitutive algebra
utilities, and UFL/NumPy companion functions used by multiple physics
models.

``mpi_utils``
-------------

Provides the shared ``MPI.COMM_WORLD`` wrapper used across the codebase,
plus rank-aware printing and JIT-cache isolation helpers that are
particularly useful in test and MPI launcher contexts.

.. automodule:: femcubes.core.utils.maths_utils
   :members:

.. automodule:: femcubes.core.utils.mpi_utils
   :members:

References
----------

.. bibliography::
   :filter: docname in docnames
