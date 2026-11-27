"""Helpers for automatic model discovery and registration.

FEMCubes model sub-packages rely on import side effects:
subclassing :class:`femcubes.core.base_model.BaseModel` automatically
registers the model type.  ``discover_models()`` imports all modules in
one package so those registrations happen before a user requests a
model by name.  ``load_plugins()`` does the same for separately installed
plugin packages, which announce themselves through the
``femcubes.plugins`` entry-point group.
"""
import importlib
import importlib.metadata
import pkgutil
import warnings

#: Entry-point group in which plugin packages name the module to import.
PLUGIN_GROUP = "femcubes.plugins"

# Entry-point names that have been handled (imported, or failed and reported).
_handled_plugins = set()


def discover_models(package_name, package_path, *, strict: bool = True):
    """Import every non-test module in a package to trigger model registration.

    Walks the package tree with :func:`pkgutil.walk_packages` and imports
    each module whose name does not start with ``test_`` and whose path
    does not contain ``tests``.  Importing the module is sufficient
    because :class:`~femcubes.core.base_model.BaseModel.__init_subclass__`
    registers each concrete model automatically.

    Parameters
    ----------
    package_name : str
        Fully-qualified package name (e.g. ``"femcubes.solids"``).
    package_path : list of str
        ``__path__`` attribute of the package.
    strict : bool, optional
        If ``True`` (default), raise immediately when a discovered
        module fails to import.  If ``False``, keep the older warning-
        only behavior.

    Notes
    -----
    Strict import is the default so broken model modules fail close to
    the source instead of surfacing later as opaque "unknown model
    type" errors.
    """
    for _importer, modname, _ispkg in pkgutil.walk_packages(
        package_path, prefix=package_name + "."
    ):
        last = modname.rsplit(".", 1)[-1]
        if last.startswith("test_") or "tests" in modname:
            continue
        try:
            importlib.import_module(modname)
        except Exception as e:
            message = f"Failed to import model module {modname}: {e}"
            if strict:
                raise ImportError(message) from e
            warnings.warn(message)


def load_plugins(*, strict: bool = False) -> list:
    """Import the installed FEMCubes plugins to register their models.

    A plugin is a separately installed package that declares an entry point
    in the group ``femcubes.plugins``; the value of the entry point is the
    module (or package) to import.  Importing it registers the model types
    and mesh types the plugin defines, in the same way as for the modules of
    FEMCubes itself.

    The model and mesh-generator registries call this function the first
    time a type string is not found, so user scripts normally do not need to
    call it.  Each plugin is imported once; later calls only handle plugins
    that have not been seen before.

    Parameters
    ----------
    strict : bool, optional
        If ``False`` (default), a plugin that fails to import gives a
        warning and the remaining plugins are still loaded.  If ``True``,
        the failure is raised as an :class:`ImportError`.

    Returns
    -------
    list of str
        Names of the entry points imported by this call.
    """
    loaded = []
    entry_points = importlib.metadata.entry_points(group=PLUGIN_GROUP)
    for entry_point in sorted(entry_points, key=lambda ep: ep.name):
        if entry_point.name in _handled_plugins:
            continue
        _handled_plugins.add(entry_point.name)
        try:
            entry_point.load()
        except Exception as e:
            message = (
                f"Failed to import FEMCubes plugin '{entry_point.name}' "
                f"({entry_point.value}): {e}"
            )
            if strict:
                raise ImportError(message) from e
            warnings.warn(message)
            continue
        loaded.append(entry_point.name)
    return loaded
