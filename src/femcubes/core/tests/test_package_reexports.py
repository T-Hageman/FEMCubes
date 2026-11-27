"""Tests for lazy package-level re-exports."""

import importlib

import pytest


@pytest.fixture
def reload_with_cleared_exports():
    """Reload packages with empty export caches and restore them after the test.

    Python attaches imported submodules onto their parent package
    objects. During a full test run, that can leave public export names
    already present in ``module.__dict__`` even when the package
    ``__init__`` itself is lazily implemented. Clear those cached
    attributes so the tests exercise the actual ``__getattr__`` path. Lazy
    resolution caches the stubs used here, so restoring import_module alone
    would leave fake public exports available to subsequent tests.
    """
    saved = []

    def reload_package(module_name):
        module = importlib.import_module(module_name)
        names = tuple(module.__all__)
        cached = {name: module.__dict__[name] for name in names if name in module.__dict__}
        saved.append((module, names, cached))
        importlib.reload(module)
        for name in names:
            module.__dict__.pop(name, None)
        return module

    yield reload_package

    for module, names, cached in reversed(saved):
        for name in names:
            module.__dict__.pop(name, None)
        module.__dict__.update(cached)


def test_femcubes_core_reexports_are_lazy(reload_with_cleared_exports, monkeypatch):
    core = reload_with_cleared_exports("femcubes.core")

    assert "Mesh" not in core.__dict__
    assert "Physics" not in core.__dict__

    imports = []

    def _fake_import_module(module_name):
        imports.append(module_name)

        class _StubModule:
            pass

        stub = _StubModule()
        setattr(stub, "Mesh", "mesh-export")
        return stub

    monkeypatch.setattr(core, "import_module", _fake_import_module)

    assert core.Mesh == "mesh-export"
    assert imports == ["femcubes.core.mesh.mesh"]
    assert core.Mesh == "mesh-export"
    assert imports == ["femcubes.core.mesh.mesh"]

    names = dir(core)
    assert "Mesh" in names
    assert "Physics" in names
    assert "ParamsTemplate" in names


def test_femcubes_reexports_are_lazy(reload_with_cleared_exports, monkeypatch):
    package = reload_with_cleared_exports("femcubes")

    assert "Mesh" not in package.__dict__
    assert "ParamsTemplate" not in package.__dict__

    imports = []

    def _fake_import_module(module_name):
        imports.append(module_name)

        class _StubModule:
            pass

        stub = _StubModule()
        setattr(stub, "ParamsTemplate", "params-export")
        return stub

    monkeypatch.setattr(package, "import_module", _fake_import_module)

    assert package.ParamsTemplate == "params-export"
    assert imports == ["femcubes.core.params_template"]
    assert package.ParamsTemplate == "params-export"
    assert imports == ["femcubes.core.params_template"]

    names = dir(package)
    assert "Mesh" in names
    assert "Physics" in names
    assert "TimeSolver" in names
    assert "LinearSolver" in names
    assert "ParamsTemplate" in names


def test_femcubes_core_solvers_reexports_are_lazy(reload_with_cleared_exports, monkeypatch):
    solvers = reload_with_cleared_exports("femcubes.core.solvers")

    assert "TimeSolver" not in solvers.__dict__
    assert "StepSolver" not in solvers.__dict__

    imports = []

    def _fake_import_module(module_name):
        imports.append(module_name)

        class _StubModule:
            pass

        stub = _StubModule()
        setattr(stub, "LinearSolver", "linear-solver-export")
        return stub

    monkeypatch.setattr(solvers, "import_module", _fake_import_module)

    assert solvers.LinearSolver == "linear-solver-export"
    assert imports == ["femcubes.core.solvers.solver_enums"]
    assert solvers.LinearSolver == "linear-solver-export"
    assert imports == ["femcubes.core.solvers.solver_enums"]

    names = dir(solvers)
    assert "LinearSolver" in names
    assert "StepSolver" in names
    assert "TimeSolver" in names


def test_femcubes_core_physics_reexports_are_lazy(reload_with_cleared_exports, monkeypatch):
    physics = reload_with_cleared_exports("femcubes.core.physics")

    assert "Physics" not in physics.__dict__

    imports = []

    def _fake_import_module(module_name):
        imports.append(module_name)

        class _StubModule:
            pass

        stub = _StubModule()
        setattr(stub, "Physics", "physics-export")
        return stub

    monkeypatch.setattr(physics, "import_module", _fake_import_module)

    assert physics.Physics == "physics-export"
    assert imports == ["femcubes.core.physics.physics"]
    assert physics.Physics == "physics-export"
    assert imports == ["femcubes.core.physics.physics"]

    names = dir(physics)
    assert "Physics" in names
