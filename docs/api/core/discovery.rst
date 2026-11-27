Model Discovery
===============

FEMCubes model registration is import-driven: defining a subclass of
:class:`femcubes.core.base_model.BaseModel` with a non-empty
``MODEL_TYPE`` automatically inserts it into the global registry.

Because of that, the framework needs to import model modules before a
user can refer to them by name in ``params.Models``.  The helper in
:mod:`femcubes.core._discover` performs that package walk.

Typical Flow
------------

1. a model package such as ``femcubes.solids`` is imported
2. the package calls :func:`femcubes.core._discover.discover_models`
3. each non-test module is imported
4. any :class:`BaseModel` subclasses register themselves
5. :class:`femcubes.core.physics.physics.Physics` can instantiate them
   from ``MODEL_TYPE`` strings

A model module that fails to import raises an :class:`ImportError` at the
import of its package, so a broken module is reported at its source and not
later as an unknown model type.

Plugins
-------

Models and mesh generators of separately installed packages are found
through the entry-point group ``femcubes.plugins``.  When a model type or a
mesh type is not in the registry, :func:`femcubes.core._discover.load_plugins`
imports the installed plugins once and the lookup is repeated.  A plugin
that fails to import gives a warning and does not stop the other plugins
from loading.  See :doc:`../../developer/plugins` for how to write one.

.. automodule:: femcubes.core._discover
   :members:
