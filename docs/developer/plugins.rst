Plugins
=======

A plugin is a separate Python package that adds models or mesh generators to
FEMCubes without changing the core package. Plugins are the place for physics
under development: once it is published, it moves into the core.

Models and mesh generators register themselves when their module is imported
(see :doc:`../api/core/discovery`), so a plugin only has to make sure that its
modules are imported. It does so by declaring an entry point in the group
``femcubes.plugins``:

.. code-block:: toml

   # pyproject.toml of the plugin
   [project]
   name = "femcubes-myphysics"
   dependencies = ["femcubes>=0.1,<0.2"]

   [project.entry-points."femcubes.plugins"]
   myphysics = "femcubes_myphysics"

The entry point names a module or package. FEMCubes imports it the first time
a model type or mesh type is requested that the core does not provide, so a
simulation script can use the plugin's ``MODEL_TYPE`` strings without
importing the plugin itself:

.. code-block:: python

   class Params(ParamsTemplate):
       Models = {
           "Solid": {"Type": "MyPhysics:NewModel", "Material": "Steel"},
       }

A package can register its model modules with the same helper that the core
sub-packages use:

.. code-block:: python

   # femcubes_myphysics/__init__.py
   from femcubes.core._discover import discover_models

   discover_models(__name__, __path__)

Guidelines
----------

* Mirror the layout of the core (``solids``, ``boundary``, ``mesh_generators``,
  each with a ``tests`` folder), so that moving a module into the core is a
  file move.
* Use only the public interface of FEMCubes. If a plugin needs a change in the
  core, that change is made in the core, as its own commit.
* Give every model type a prefix that cannot collide with a core type.
* Pin the range of FEMCubes versions the plugin was tested with.

.. autofunction:: femcubes.core._discover.load_plugins
   :no-index:
