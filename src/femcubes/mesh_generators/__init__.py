"""Bundled mesh generators, registered automatically by ``MeshType``.

Add a new generator as one module in this package: a subclass of
:class:`femcubes.core.mesh.generator_base.MeshGeneratorBase` with a unique
``MESH_TYPE`` and a ``Sample_Parameters()`` function. Every public module
here is imported on package import, which registers its generator.
Modules starting with ``_`` hold shared helpers and are not imported.
"""
import importlib
import pkgutil


def discover_mesh_generators(*, strict: bool = True) -> None:
    """Import all bundled mesh-generator modules to populate the registry.

    Parameters
    ----------
    strict : bool, optional
        If ``True`` (default), raise immediately when a discovered
        generator module fails to import.  If ``False``, emit a warning
        and continue.
    """
    for _importer, modname, ispkg in pkgutil.walk_packages(
        __path__, prefix=__name__ + "."
    ):
        last = modname.rsplit(".", 1)[-1]
        # Generators are plain modules; skip subpackages (tests) and private helpers.
        if ispkg or ".tests." in modname or last.startswith("test_") or last.startswith("_"):
            continue
        try:
            importlib.import_module(modname)
        except Exception as e:
            message = f"Failed to import mesh generator module {modname}: {e}"
            if strict:
                raise ImportError(message) from e
            import warnings
            warnings.warn(message)


discover_mesh_generators()
