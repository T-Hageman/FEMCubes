"""Tests for mesh-generator discovery and mesh construction."""

import importlib
import pkgutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh as dmesh

from femcubes.core.params_template import ParamsTemplate
from femcubes.core.mesh.mesh import Mesh
from femcubes.core.mesh._mesh_gen import _MeshGen
from femcubes.core.mesh._mesh_groups import _MeshGroups, _marker_names
from femcubes.core.mesh.generator_base import GeneratedMeshResult, MeshGeneratorBase


# Expected registry contents after importing the generators package.
# The discovery tests below verify that these mesh types are auto-registered.
EXPECTED_GENERATORS = {
    "Cylinder",
    "DamageChallenge",
    "FullPlateWithHole",
    "HalfPlateWithHole",
    "PlateWithHole",
    "Rectangle",
    "Rectangle_Notches",
    "Rectangle_RefinedTop",
    "SingleEdgeNotched",
    "Rectangle_CentreNotch",
    "NotchedTensileBar",
    "TwoSquared",
    "CorrosionPit",
    "PlateWithDefect",
}

# Group and part names exposed by the multi-domain notched tensile bar.
NOTCHED_BAR_DOMAINS = {"domain", "domain:left", "domain:centre", "domain:right"}
NOTCHED_BAR_BOUNDARIES = {
    "bottom", "bottom:left", "bottom:centre", "bottom:right",
    "top", "top:left", "top:centre", "top:right",
    "left", "left:left",
    "right", "right:right",
    "notch", "notch:centre",
    "interface", "interface:left", "interface:right",
}

# Group and part names exposed by the two-squares mesh.
TWO_SQUARED_DOMAINS = {"domain", "domain:left", "domain:right"}
TWO_SQUARED_BOUNDARIES = {
    "bottom", "bottom:left", "bottom:right",
    "top", "top:left", "top:right",
    "left", "left:left",
    "right", "right:right",
    "interface",
}

# Group and part names exposed by the corrosion pit mesh.
CORROSION_PIT_BOUNDARIES = {"top", "left", "right", "metal", "metal:pit", "metal:outer"}

# Group and part names exposed by the plate-with-defect mesh (2-D and 3-D).
PLATE_WITH_DEFECT_BOUNDARIES = {"top", "far", "symmetry", "metal", "metal:defect", "metal:outer"}

# Generators that split the mesh into several named cell domains. All
# generators expose the canonical "domain" name; multi-domain generators also
# expose its "domain:<subdomain>" parts.
EXPECTED_DOMAINS = {
    "NotchedTensileBar": NOTCHED_BAR_DOMAINS,
    "TwoSquared": TWO_SQUARED_DOMAINS,
}


def _expected_domains(mesh_type):
    return EXPECTED_DOMAINS.get(mesh_type, {"domain"})


def _generator_module(mesh_type):
    """Return the module that registers ``mesh_type`` (module names are snake_case)."""
    import femcubes.mesh_generators  # noqa: F401  (populates the registry)

    return sys.modules[MeshGeneratorBase._registry[mesh_type].__module__]


def _sample_params(mesh_type, **overrides):
    """Create test parameters from the generator's documented sample input."""

    params = _generator_module(mesh_type).Sample_Parameters()
    for name, value in overrides.items():
        setattr(params, name, value)
    return params


class _MeshGenEnd:
    """Terminate the cooperative super() chain for _MeshGen harness tests."""

    def __init__(self, params):
        self.params = params


class _MeshGenHarness(_MeshGroups, _MeshGen, _MeshGenEnd):
    """Concrete object exposing _MeshGen behavior for unit tests."""

    def __init__(self, params):
        self.added_domains = []
        self.added_boundaries = []
        super().__init__(params)

    def add_domain_marker(self, name, tag):
        self.added_domains.append((name, tag))
        super().add_domain_marker(name, tag)

    def add_boundary_marker(self, name, tag):
        self.added_boundaries.append((name, tag))
        super().add_boundary_marker(name, tag)


def _domain_measure(mesh):
    # Integrate 1 over the domain so the test can compare the generated
    # geometry against the expected area / volume.
    return fem.assemble_scalar(fem.form(fem.Constant(mesh.mesh, 1.0) * ufl.dx))


def _assert_markers(mesh, expected_boundaries):
    # Every generator should expose the canonical "domain" marker plus the
    # mesh-type-specific boundary markers used by models and BCs.
    assert set(mesh.domain_names()) == _expected_domains(mesh.params.MeshType)
    assert set(mesh.boundary_names()) == set(expected_boundaries)

    assert mesh.get_domain_cells("domain").size > 0
    for name in expected_boundaries:
        assert mesh.get_boundary_facets(name).size > 0


def _boundary_midpoints(mesh, boundary_name):
    facet_dim = mesh.mesh.topology.dim - 1
    facets = mesh.get_boundary_facets(boundary_name)
    return dmesh.compute_midpoints(mesh.mesh, facet_dim, facets)


# Parameterized mesh-construction cases.
# Each case provides:
# - a minimal valid parameter object
# - the expected boundary-marker mapping
# - the expected area / volume
# - a tolerance for the geometric measure check
MESH_CASES = [
    pytest.param(
        _sample_params("Rectangle", dim=2, Lx=4.0, Ly=2.0, dx=1.0),
        {"bottom", "right", "top", "left"},
        8.0,
        1.0e-12,
        id="rectangle-2d",
    ),
    pytest.param(
        _sample_params("Rectangle", dim=3, Lx=4.0, Ly=2.0, Lz=1.5, dx=1.0),
        {"left", "right", "front", "back", "bottom", "top"},
        12.0,
        1.0e-12,
        id="rectangle-3d",
    ),
    pytest.param(
        _sample_params(
            "Rectangle_Notches",
            dim=2,
            Lx=4.0,
            Ly=2.0,
            dx=0.2,
            nNotches=3,
            Notch_Width=0.2,
            Notch_Depth=0.5,
        ),
        {"bottom", "right", "top", "left", "notch"},
        4.0 * 2.0 - 3.0 * 0.2 * 0.5,
        1.0e-12,
        id="rectangle-notches-2d",
    ),
    pytest.param(
        _sample_params(
            "Rectangle_Notches",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            dx=0.2,
            nNotches=3,
            Notch_Width=0.2,
            Notch_Depth=0.5,
        ),
        {"bottom", "right", "top", "left", "front", "back", "notch"},
        4.0 * 2.0 * 1.0 - 3.0 * 0.2 * 2.0 * 0.5,
        1.0e-12,
        id="rectangle-notches-3d",
    ),
    pytest.param(
        _sample_params(
            "Rectangle_Notches",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            dx=0.2,
            Notch_Positions=(3.0, 1.0),
            Notch_Width=0.2,
            Notch_Depth=0.5,
            Notch_Y_Ranges=((1.0, 2.0), (0.0, 1.5)),
        ),
        {"bottom", "right", "top", "left", "front", "back", "notch"},
        4.0 * 2.0 * 1.0 - (1.5 + 1.0) * 0.2 * 0.5,
        1.0e-12,
        id="rectangle-notches-3d-finite-length",
    ),
    pytest.param(
        _sample_params(
            "PlateWithHole",
            dim=2,
            Lx=4.0,
            Ly=2.0,
            R=0.5,
            dx=0.2,
        ),
        {"bottom", "right", "top", "left", "hole"},
        4.0 * 2.0 - 0.25 * np.pi * 0.5**2,
        5.0e-2,
        id="plate-with-hole-2d",
    ),
    pytest.param(
        _sample_params(
            "PlateWithHole",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            R=0.5,
            dx=0.2,
        ),
        {"bottom", "right", "top", "left", "hole", "front", "back"},
        4.0 * 2.0 * 1.0 - 0.25 * np.pi * 0.5**2 * 1.0,
        5.0e-2,
        id="plate-with-hole-3d",
    ),
    pytest.param(
        _sample_params("FullPlateWithHole", dim=2, Lx=4.0, Ly=4.0, R=0.5, dx=0.2),
        {"bottom", "right", "top", "left", "hole"},
        4.0 * 4.0 - np.pi * 0.5**2,
        5.0e-2,
        id="full-plate-with-hole-2d",
    ),
    pytest.param(
        _sample_params("HalfPlateWithHole"),
        {"bottom", "right", "top", "symmetry", "hole"},
        (4.0 * 4.0 - np.pi * 0.5**2) / 2,
        5.0e-3,
        id="half-plate-with-hole-2d",
    ),
    pytest.param(
        _sample_params("FullPlateWithHole", dim=3, Lx=4.0, Ly=4.0, Lz=1.0, R=0.5, dx=0.2),
        {"bottom", "right", "top", "left", "hole", "front", "back"},
        4.0 * 4.0 * 1.0 - np.pi * 0.5**2 * 1.0,
        5.0e-2,
        id="full-plate-with-hole-3d",
    ),
    pytest.param(
        _sample_params("SingleEdgeNotched", dim=2, Lx=1.0, Ly=0.4, LNotch=0.2, NotchHeight=0.04, dx=0.02),
        {"bottom", "right", "top", "left", "notch"},
        1.0 * 0.4 - 0.2 * 0.04,
        1.0e-12,
        id="single-edge-notched-2d",
    ),
    pytest.param(
        _sample_params(
            "SingleEdgeNotched",
            dim=3,
            Lx=1.0,
            Ly=0.4,
            Lz=0.02,
            LNotch=0.2,
            NotchHeight=0.04,
            dx=0.02,
        ),
        {"bottom", "right", "top", "left", "notch", "front", "back"},
        1.0 * 0.4 * 0.02 - 0.2 * 0.04 * 0.02,
        1.0e-12,
        id="single-edge-notched-3d",
    ),
    pytest.param(
        _sample_params(
            "Rectangle_RefinedTop",
            dim=2,
            Lx=100.0,
            Ly=50.0,
            Ly_refined=10.0,
            dx_min=2.0,
            dx_max=10.0,
        ),
        {"bottom", "right", "top", "left", "top-left"},
        100.0 * 50.0,
        1.0e-10,
        id="rectangle-refined-top-2d",
    ),
    pytest.param(
        _sample_params("Cylinder", dim=3, R=0.5, Lz=1.0, dx=0.15),
        {"bottom", "top", "outer"},
        np.pi * 0.5**2 * 1.0,
        7.5e-2,
        id="cylinder-3d",
    ),
    pytest.param(
        _sample_params(
            "DamageChallenge",
            dim=2,
            Lx=4.0,
            Ly=2.0,
            dx=0.2,
            Lx_BC=1.0,
            Lx_notch=2.0,
            Notch_Thickness=0.05,
            Ly_notch=1.0,
            Load_Width=0.2,
            BC_Width=0.2,
        ),
        {"bottom", "right", "top", "left", "notch", "top_load", "bottom_left", "bottom_right"},
        4.0 * 2.0 - 0.05 * 1.0,
        1.0e-12,
        id="damage-challenge-2d",
    ),
    pytest.param(
        _sample_params(
            "DamageChallenge",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            dx=0.2,
            Lx_BC=1.0,
            Lx_notch=2.0,
            Notch_Thickness=0.05,
            Ly_notch=1.0,
            Load_Width=0.2,
            BC_Width=0.2,
        ),
        {"bottom", "right", "top", "left", "notch", "top_load", "bottom_left", "bottom_right", "front", "back"},
        4.0 * 2.0 * 1.0 - 0.05 * 1.0 * 1.0,
        1.0e-12,
        id="damage-challenge-3d",
    ),
    pytest.param(
        _sample_params("NotchedTensileBar"),
        NOTCHED_BAR_BOUNDARIES,
        4.0 * 1.0 - 2.0 * 0.1 * 0.2,
        1.0e-12,
        id="notched-tensile-bar-2d",
    ),
    pytest.param(
        _sample_params("TwoSquared"),
        TWO_SQUARED_BOUNDARIES,
        2.0 * 1.0 * 1.0,
        1.0e-12,
        id="two-squared-2d",
    ),
    pytest.param(
        _sample_params("CorrosionPit"),
        CORROSION_PIT_BOUNDARIES,
        4.0 * 2.0 + 0.5 * np.pi * 0.5**2,
        2.0e-3,  # the pit arc is polygonal
        id="corrosion-pit-2d",
    ),
    pytest.param(
        _sample_params("PlateWithDefect"),
        PLATE_WITH_DEFECT_BOUNDARIES,
        4.0 * 1.0,
        1.0e-12,
        id="plate-with-defect-2d",
    ),
    pytest.param(
        _sample_params("PlateWithDefect", dim=3, W=2.0, dx=0.5, dx_metal=0.25),
        PLATE_WITH_DEFECT_BOUNDARIES,
        2.0 * 2.0 * 1.0,
        1.0e-12,
        id="plate-with-defect-3d",
    ),
]


class TestMeshGeneratorDiscovery:
    def test_mesh_generator_registry_contains_all_bundled_types(self):
        # Sanity-check the registry after normal package import.
        assert set(MeshGeneratorBase._registry) == EXPECTED_GENERATORS

    def test_mesh_generator_lookup_returns_expected_class(self):
        # Ensure type-string lookup instantiates the class currently registered
        # for that mesh type, without coupling the test to the class name text.
        for mesh_type in EXPECTED_GENERATORS:
            registered_cls = MeshGeneratorBase._registry[mesh_type]
            generator = MeshGeneratorBase.get_generator(mesh_type)

            assert isinstance(generator, registered_cls)
            assert generator.MESH_TYPE == mesh_type

    def test_mesh_generator_package_autodiscovery_rebuilds_registry(self):
        import femcubes.mesh_generators as generators_pkg

        generator_modules = [
            MeshGeneratorBase._registry[mesh_type].__module__
            for mesh_type in sorted(EXPECTED_GENERATORS)
        ]

        try:
            # Clear both the registry and the imported generator modules, then
            # reload the package to confirm discovery repopulates the registry.
            MeshGeneratorBase._registry.clear()
            for module_name in generator_modules:
                sys.modules.pop(module_name, None)

            importlib.reload(generators_pkg)

            assert set(MeshGeneratorBase._registry) == EXPECTED_GENERATORS
        finally:
            importlib.reload(generators_pkg)

    def test_mesh_generator_package_autodiscovery_raises_on_import_failure(self, monkeypatch):
        import femcubes.mesh_generators as generators_pkg

        broken_module = f"{generators_pkg.__name__}.BrokenGenerator"
        real_import_module = importlib.import_module

        with monkeypatch.context() as m:
            m.setattr(pkgutil, "walk_packages", lambda path, prefix: [(None, broken_module, False)])

            def _fake_import_module(name, package=None):
                if name == broken_module:
                    raise RuntimeError("boom")
                return real_import_module(name, package)

            m.setattr(importlib, "import_module", _fake_import_module)

            with pytest.raises(ImportError, match="Failed to import mesh generator module"):
                importlib.reload(generators_pkg)

        importlib.reload(generators_pkg)

    def test_mesh_generator_package_autodiscovery_can_warn_in_non_strict_mode(self, monkeypatch):
        import femcubes.mesh_generators as generators_pkg

        broken_module = f"{generators_pkg.__name__}.BrokenGenerator"
        real_import_module = importlib.import_module

        with monkeypatch.context() as m:
            m.setattr(pkgutil, "walk_packages", lambda path, prefix: [(None, broken_module, False)])

            def _fake_import_module(name, package=None):
                if name == broken_module:
                    raise RuntimeError("boom")
                return real_import_module(name, package)

            m.setattr(importlib, "import_module", _fake_import_module)

            with pytest.warns(UserWarning, match="Failed to import mesh generator module"):
                generators_pkg.discover_mesh_generators(strict=False)

    def test_invalid_mesh_type_raises_helpful_error(self):
        # The error should point the caller to the list of valid mesh types.
        with pytest.raises(NotImplementedError, match="Available:"):
            MeshGeneratorBase.get_generator("InvalidType")

    @pytest.mark.parametrize("mesh_type", sorted(EXPECTED_GENERATORS))
    def test_generators_expose_sample_parameters(self, mesh_type):
        # Every generator module should publish a buildable sample-parameter
        # object to document its required inputs and simplify direct testing.
        module = _generator_module(mesh_type)

        params = module.Sample_Parameters()

        assert isinstance(params, ParamsTemplate)
        assert params.MeshType == mesh_type

    @pytest.mark.parametrize("mesh_type", sorted(EXPECTED_GENERATORS))
    def test_generators_return_structured_results(self, mesh_type):
        # The generator contract is now explicit: generate() returns the mesh
        # artifacts and marker maps rather than mutating the wrapper in place.
        module = _generator_module(mesh_type)
        params = module.Sample_Parameters()

        generated = MeshGeneratorBase.get_generator(mesh_type).generate(params)

        assert isinstance(generated, GeneratedMeshResult)
        assert generated.mesh is not None
        assert generated.cell_tags is not None
        assert generated.facet_tags is not None
        assert set(_marker_names(generated.domain_markers)) == _expected_domains(mesh_type)
        assert set(generated.boundary_markers)

    @pytest.mark.parametrize("mesh_type", sorted(EXPECTED_GENERATORS))
    def test_sample_parameters_build_meshes(self, mesh_type):
        # The documented sample parameter sets should be sufficient to build
        # a real mesh through the public Mesh wrapper.
        module = _generator_module(mesh_type)

        mesh = Mesh(module.Sample_Parameters())

        assert mesh.mesh is not None
        assert mesh.get_domain_cells("domain").size > 0
        assert set(mesh.domain_names()) == _expected_domains(mesh_type)
        assert set(mesh.boundary_names())

    def test_mesh_generator_base_generate_must_be_overridden(self):
        with pytest.raises(NotImplementedError):
            MeshGeneratorBase().generate(ParamsTemplate())


class TestMeshGenIntegration:
    def test_mesh_gen_assigns_markers_through_mesh_group_methods(self, monkeypatch):
        class _DummyGenerator:
            def generate(self, params):
                return GeneratedMeshResult(
                    mesh="mesh-object",
                    cell_tags="cell-tags",
                    facet_tags="facet-tags",
                    domain_markers={"domain": 11},
                    boundary_markers={"left": 21, "right": 22},
                )

        params = _sample_params("Rectangle")

        monkeypatch.setattr(MeshGeneratorBase, "get_generator", lambda mesh_type: _DummyGenerator())

        mesh = _MeshGenHarness(params)

        assert mesh.mesh == "mesh-object"
        assert mesh.cell_tags == "cell-tags"
        assert mesh.facet_tags == "facet-tags"
        assert set(mesh.domain_names()) == {"domain"}
        assert set(mesh.boundary_names()) == {"left", "right"}
        assert mesh.get_domain_tag("domain") == 11
        assert mesh.get_boundary_tag("left") == 21
        assert mesh.get_boundary_tag("right") == 22
        assert mesh.added_domains == [("domain", 11)]
        assert mesh.added_boundaries == [("left", 21), ("right", 22)]


# Geometric length parameters used by the bundled generators. Scaling all of
# them must scale the generated geometry without changing its markers.
_LENGTH_PARAMS = (
    "Lx", "Ly", "Lz", "R", "dx", "dx_centre", "dx_min", "dx_max",
    "Lx_centre", "Notch_Width", "Notch_Depth", "Notch_Positions", "Notch_Y_Ranges", "Notch_Thickness",
    "LNotch", "NotchHeight", "Lx_BC", "Lx_notch", "Ly_notch", "Load_Width", "BC_Width",
    "Ly_refined", "Lx_topleft", "notch_r", "notch_h", "notch_dh", "L", "dx_interface",
    "W", "H", "R_pit", "dx_metal", "R_defect", "dx_defect",
)

# Every mesh case plus the optional notched-cylinder branch.
_SCALE_CASES = [
    pytest.param(case.values[0], id=case.id) for case in MESH_CASES
] + [
    pytest.param(
        _sample_params("Cylinder", dim=3, R=0.5, Lz=1.0, dx=0.15, notch_r=0.1, notch_h=0.6, notch_dh=0.2),
        id="cylinder-notched-3d",
    ),
]


def _scaled_params(params, scale):
    """Return a copy of ``params`` with every geometric length multiplied by ``scale``."""

    def _scale(value):
        if isinstance(value, (list, tuple)):
            return [_scale(item) for item in value]
        return scale * value

    overrides = {name: _scale(getattr(params, name)) for name in _LENGTH_PARAMS if hasattr(params, name)}
    overrides["dim"] = params.dim
    for name in ("ellipticity", "rotation_angle", "nNotches", "Notch_Sides"):
        if hasattr(params, name):
            overrides[name] = getattr(params, name)
    return _sample_params(params.MeshType, **overrides)


def _boundary_measures(mesh):
    ds = mesh.get_Boundary_Integrator()
    one = fem.Constant(mesh.mesh, 1.0)
    return {
        name: fem.assemble_scalar(fem.form(one * ds(mesh.get_boundary_tags(name))))
        for name in mesh.boundary_names()
    }


class TestMeshGeneratorScaleInvariance:
    @pytest.mark.parametrize("params", _SCALE_CASES)
    def test_small_geometries_keep_boundary_classification(self, params):
        # Boundary classification must not depend on the absolute size of
        # the geometry: a 1000x smaller copy, for which gmsh's ~1e-7 OCC
        # bounding-box padding is no longer negligible, must expose the same
        # markers with scaled measures. The tolerance allows for gmsh meshing
        # curved surfaces slightly differently at another scale; a
        # misclassified boundary changes its measure by O(1).
        scale = 1.0e-3
        reference = Mesh(params)
        small = Mesh(_scaled_params(params, scale))
        dim = params.dim

        assert small.domain_names() == reference.domain_names()
        assert small.boundary_names() == reference.boundary_names()
        assert np.isclose(_domain_measure(small), scale**dim * _domain_measure(reference), rtol=1.0e-3)
        for name in small.boundary_names():
            assert small.get_boundary_facets(name).size > 0

        reference_measures = _boundary_measures(reference)
        small_measures = _boundary_measures(small)
        for name, reference_measure in reference_measures.items():
            assert np.isclose(small_measures[name], scale ** (dim - 1) * reference_measure, rtol=1.0e-3, atol=0.0), (
                f"{params.MeshType} ({dim}D) boundary '{name}': expected "
                f"{scale ** (dim - 1) * reference_measure}, got {small_measures[name]}"
            )


class TestMeshGeneration:
    @pytest.mark.parametrize(("params", "expected_boundaries", "expected_measure", "measure_rtol"), MESH_CASES)
    def test_mesh_types_build_with_expected_markers(self, params, expected_boundaries, expected_measure, measure_rtol):
        # This is the main smoke test for each bundled generator: build the
        # mesh, verify topological metadata, then check that the generated
        # geometry has the expected area / volume.
        mesh = Mesh(params)

        assert mesh.mesh is not None
        assert mesh.mesh.topology.dim == params.dim
        assert mesh.mesh.geometry.x.shape[0] > 0

        _assert_markers(mesh, expected_boundaries)

        measured = _domain_measure(mesh)
        assert np.isclose(measured, expected_measure, rtol=measure_rtol), (
            f"Unexpected domain measure for {params.MeshType} ({params.dim}D): "
            f"expected {expected_measure}, got {measured}"
        )

    def test_cylinder_with_notch_registers_notch_boundary(self):
        # Exercise the optional notched-cylinder branch separately, since the
        # default cylinder case above covers only the plain outer surface.
        params = _sample_params(
            "Cylinder",
            dim=3,
            R=0.5,
            Lz=1.0,
            dx=0.15,
            notch_r=0.1,
            notch_h=0.6,
            notch_dh=0.2,
        )

        mesh = Mesh(params)

        _assert_markers(mesh, {"bottom", "top", "outer", "notch"})
        # The notch removes material, so the domain volume must be smaller than
        # the full cylinder volume.
        assert _domain_measure(mesh) < np.pi * params.R**2 * params.Lz

    def test_cylinder_with_inclined_slot(self):
        params = _sample_params("Cylinder", dim=3, R=0.5, Lz=1.0, dx=0.1, slot_depth=0.2, slot_length=0.3,
                                slot_width=0.05, slot_h=0.6)

        mesh = Mesh(params)

        _assert_markers(mesh, {"bottom", "top", "outer", "notch"})
        plain = Mesh(_sample_params("Cylinder", dim=3, R=0.5, Lz=1.0, dx=0.1))
        removed = _domain_measure(plain) - _domain_measure(mesh)
        # The slot is a 0.2 x 0.3 x 0.05 box, less the part outside the curved surface (and faceting)
        assert 0.8 * 0.2 * 0.3 * 0.05 < removed < 1.2 * 0.2 * 0.3 * 0.05

        # The slot faces lie around (R, 0, slot_h); the trace descends in z towards +y
        tdim = mesh.mesh.topology.dim
        facets = mesh.get_boundary_facets("notch")
        mesh.mesh.topology.create_connectivity(tdim - 1, 0)
        f2v = mesh.mesh.topology.connectivity(tdim - 1, 0)
        points = mesh.mesh.geometry.x[np.unique(np.concatenate([f2v.links(f) for f in facets]))]
        points = np.vstack(mesh.mesh.comm.allgather(points))
        assert np.all(points[:, 0] > params.R - params.slot_depth - 1e-9)
        assert np.all(np.abs(points[:, 2] - params.slot_h) < 0.25)
        slope = np.polyfit(points[:, 1], points[:, 2] - params.slot_h, 1)[0]
        assert slope == pytest.approx(-1.0, abs=0.3)

    def test_cylinder_rejects_groove_with_slot(self):
        params = _sample_params("Cylinder", dim=3, R=0.5, Lz=1.0, dx=0.15, notch_r=0.1, slot_depth=0.1)
        with pytest.raises(ValueError, match="cannot both be set"):
            Mesh(params)

    def test_rectangle_notches_supports_explicit_notch_positions(self):
        params = _sample_params(
            "Rectangle_Notches",
            dim=2,
            Lx=4.0,
            Ly=2.0,
            dx=0.2,
            Notch_Positions=[1.0, 3.0],
            Notch_Width=0.4,
            Notch_Depth=0.5,
        )

        mesh = Mesh(params)

        _assert_markers(mesh, {"bottom", "right", "top", "left", "notch"})
        assert np.isclose(_domain_measure(mesh), 4.0 * 2.0 - 2.0 * 0.4 * 0.5, rtol=1.0e-12)

    def test_rectangle_notches_supports_explicit_notch_positions_in_3d(self):
        params = _sample_params(
            "Rectangle_Notches",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            dx=0.2,
            Notch_Positions=[1.0, 3.0],
            Notch_Width=0.4,
            Notch_Depth=0.5,
        )

        mesh = Mesh(params)

        _assert_markers(mesh, {"bottom", "right", "top", "left", "front", "back", "notch"})
        assert np.isclose(_domain_measure(mesh), 4.0 * 2.0 * 1.0 - 2.0 * 0.4 * 2.0 * 0.5, rtol=1.0e-12)

    @pytest.mark.parametrize(
        ("params", "x0", "y0", "z0"),
        [
            pytest.param(
                _sample_params("Rectangle", dim=3, Lx=4.0, Ly=2.0, Lz=1.5, dx=0.5),
                4.0,
                2.0,
                1.5,
                id="rectangle",
            ),
            pytest.param(
                _sample_params(
                    "Rectangle_Notches",
                    dim=3,
                    Lx=4.0,
                    Ly=2.0,
                    Lz=1.0,
                    dx=0.2,
                    nNotches=2,
                    Notch_Width=0.4,
                    Notch_Depth=0.3,
                ),
                4.0,
                2.0,
                1.0,
                id="rectangle-notches",
            ),
            pytest.param(
                _sample_params(
                    "PlateWithHole",
                    dim=3,
                    Lx=4.0,
                    Ly=2.0,
                    Lz=1.0,
                    R=0.5,
                    dx=0.2,
                ),
                4.0,
                2.0,
                1.0,
                id="plate-with-hole",
            ),
            pytest.param(
                _sample_params("FullPlateWithHole", dim=3, Lx=4.0, Ly=4.0, Lz=1.0, R=0.5, dx=0.2),
                4.0,
                4.0,
                1.0,
                id="full-plate-with-hole",
            ),
            pytest.param(
                _sample_params("SingleEdgeNotched", dim=3, Lx=1.0, Ly=0.4, Lz=0.02, LNotch=0.2, NotchHeight=0.04, dx=0.02),
                1.0,
                0.4,
                0.02,
                id="single-edge-notched",
            ),
            pytest.param(
                _sample_params(
                    "DamageChallenge",
                    dim=3,
                    Lx=4.0,
                    Ly=2.0,
                    Lz=1.0,
                    dx=0.2,
                    Lx_BC=1.0,
                    Lx_notch=2.0,
                    Notch_Thickness=0.05,
                    Ly_notch=1.0,
                    Load_Width=0.2,
                    BC_Width=0.2,
                ),
                4.0,
                2.0,
                1.0,
                id="damage-challenge",
            ),
        ],
    )
    def test_standard_3d_generators_use_consistent_boundary_axes(self, params, x0, y0, z0):
        mesh = Mesh(params)
        tol = 1.0e-12

        assert np.allclose(_boundary_midpoints(mesh, "front")[:, 0], 0.0, atol=tol)
        assert np.allclose(_boundary_midpoints(mesh, "back")[:, 0], x0, atol=tol)
        assert np.allclose(_boundary_midpoints(mesh, "left")[:, 1], 0.0, atol=tol)
        assert np.allclose(_boundary_midpoints(mesh, "right")[:, 1], y0, atol=tol)
        assert np.allclose(_boundary_midpoints(mesh, "bottom")[:, 2], 0.0, atol=tol)
        assert np.allclose(_boundary_midpoints(mesh, "top")[:, 2], z0, atol=tol)

    def test_rectangle_notches_can_build_without_any_notches(self):
        params = _sample_params(
            "Rectangle_Notches",
            dim=2,
            Lx=4.0,
            Ly=2.0,
            dx=0.2,
            nNotches=0,
        )

        mesh = Mesh(params)

        _assert_markers(mesh, {"bottom", "right", "top", "left"})
        assert "notch" not in mesh.boundary_names()
        assert np.isclose(_domain_measure(mesh), 4.0 * 2.0, rtol=1.0e-12)

    def test_rectangle_notches_can_build_without_any_notches_in_3d(self):
        params = _sample_params(
            "Rectangle_Notches",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            dx=0.2,
            nNotches=0,
        )

        mesh = Mesh(params)

        _assert_markers(mesh, {"bottom", "right", "top", "left", "front", "back"})
        assert "notch" not in mesh.boundary_names()
        assert np.isclose(_domain_measure(mesh), 4.0 * 2.0 * 1.0, rtol=1.0e-12)

    def test_rectangle_notches_treats_empty_explicit_notch_positions_as_no_notches(self):
        params = _sample_params(
            "Rectangle_Notches",
            dim=2,
            Lx=4.0,
            Ly=2.0,
            dx=0.2,
            Notch_Positions=[],
            Notch_Width=0.4,
            Notch_Depth=0.5,
        )

        mesh = Mesh(params)

        _assert_markers(mesh, {"bottom", "right", "top", "left"})
        assert "notch" not in mesh.boundary_names()
        assert np.isclose(_domain_measure(mesh), 4.0 * 2.0, rtol=1.0e-12)

    def test_rectangle_refined_top_rejects_3d(self):
        params = _sample_params("Rectangle_RefinedTop", dim=3)

        with pytest.raises(NotImplementedError, match="only supports 2-D meshes"):
            Mesh(params)

    def test_rectangle_refined_top_rejects_invalid_refined_height(self):
        params = _sample_params(
            "Rectangle_RefinedTop",
            dim=2,
            Lx=20.0,
            Ly=10.0,
            Ly_refined=10.0,
            dx_min=1.0,
            dx_max=2.0,
        )

        with pytest.raises(ValueError, match="Ly_refined .* must be less than Ly"):
            Mesh(params)

    def test_rectangle_refined_top_supports_embedded_alignment_rectangle(self):
        params = _sample_params(
            "Rectangle_RefinedTop",
            dim=2,
            Lx=20.0,
            Ly=10.0,
            Ly_refined=2.0,
            dx_min=1.0,
            dx_max=2.0,
            Lx_topleft=2.0,
            AlignWithRectangle={
                "x_min": 4.0,
                "x_max": 8.0,
                "y_min": 3.0,
                "y_max": 5.0,
            },
        )

        mesh = Mesh(params)

        _assert_markers(mesh, {"bottom", "right", "top", "left", "top-left"})
        assert np.isclose(_domain_measure(mesh), 20.0 * 10.0, rtol=1.0e-10)

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            pytest.param(
                _sample_params(
                    "Rectangle_Notches",
                    dim=2,
                    Lx=4.0,
                    Ly=2.0,
                    dx=0.2,
                    Notch_Positions=[1.0, 1.1],
                    Notch_Width=0.4,
                    Notch_Depth=0.5,
                ),
                "must not overlap",
                id="rectangle-notches-overlap",
            ),
            pytest.param(
                _sample_params(
                    "Rectangle_Notches",
                    dim=2,
                    Lx=4.0,
                    Ly=2.0,
                    dx=0.2,
                    Notch_Positions=[0.1],
                    Notch_Width=0.4,
                    Notch_Depth=0.5,
                ),
                "within \\[0, Lx\\]",
                id="rectangle-notches-outside-domain",
            ),
            pytest.param(
                _sample_params(
                    "Rectangle_Notches",
                    dim=2,
                    Lx=4.0,
                    Ly=2.0,
                    dx=0.2,
                    Notch_Positions=[1.0],
                    Notch_Width=0.0,
                    Notch_Depth=0.5,
                ),
                "Notch_Width must be positive and smaller than Lx",
                id="rectangle-notches-invalid-width",
            ),
            pytest.param(
                _sample_params(
                    "Rectangle_Notches",
                    dim=2,
                    Lx=4.0,
                    Ly=2.0,
                    dx=0.2,
                    Notch_Positions=[1.0],
                    Notch_Width=0.4,
                    Notch_Depth=0.0,
                ),
                "Notch_Depth must be positive and smaller than Ly",
                id="rectangle-notches-invalid-depth",
            ),
        ],
    )
    def test_rectangle_notches_rejects_invalid_notch_layout(self, params, message):
        with pytest.raises(ValueError, match=message):
            Mesh(params)

    @pytest.mark.parametrize(
        ("params", "expected_boundaries", "expected_measure", "measure_rtol"),
        [
            pytest.param(
                _sample_params(
                    "FullPlateWithHole",
                    dim=2,
                    Lx=4.0,
                    Ly=4.0,
                    R=0.5,
                    dx=0.2,
                    ellipticity=1.5,
                    rotation_angle=30.0,
                ),
                {"bottom", "right", "top", "left", "hole"},
                4.0 * 4.0 - np.pi * 0.5 * (0.5 / 1.5),
                5.0e-2,
                id="full-plate-with-hole-2d-rotated",
            ),
            pytest.param(
                _sample_params(
                    "FullPlateWithHole",
                    dim=3,
                    Lx=4.0,
                    Ly=4.0,
                    Lz=1.0,
                    R=0.5,
                    dx=0.2,
                    ellipticity=1.5,
                    rotation_angle=30.0,
                ),
                {"bottom", "right", "top", "left", "hole", "front", "back"},
                (4.0 * 4.0 - np.pi * 0.5 * (0.5 / 1.5)) * 1.0,
                5.0e-2,
                id="full-plate-with-hole-3d-rotated",
            ),
        ],
    )
    def test_full_plate_with_hole_supports_rotated_elliptical_holes(
        self, params, expected_boundaries, expected_measure, measure_rtol
    ):
        # Non-zero rotation_angle is a supported input path. The geometry
        # should still build and preserve the expected plate-minus-ellipse
        # area / volume.
        mesh = Mesh(params)

        _assert_markers(mesh, expected_boundaries)
        assert np.isclose(
            _domain_measure(mesh), expected_measure, rtol=measure_rtol
        )

    @pytest.mark.parametrize(
        ("dim", "overrides"),
        [
            pytest.param(2, {}, id="2d"),
            pytest.param(3, {"Lz": 1.0}, id="3d"),
            pytest.param(3, {"Lz": 1.0, "Lx_notch": (1.8, 2.2)}, id="3d-diagonal-notch"),
        ],
    )
    def test_damage_challenge_windows_have_the_requested_widths(self, dim, overrides):
        params = _sample_params("DamageChallenge", dim=dim, Load_Width=0.5, BC_Width=0.3, **overrides)
        mesh = Mesh(params)
        measures = _boundary_measures(mesh)
        thickness = getattr(params, "Lz", 1.0) if dim == 3 else 1.0
        # 3-D names the y = Ly face "right" (standard box axes).
        top = "right" if dim == 3 else "top"

        assert np.isclose(measures["top_load"], 0.5 * thickness, rtol=1e-10)
        assert np.isclose(measures["bottom_left"], 0.3 * thickness, rtol=1e-10)
        assert np.isclose(measures["bottom_right"], 0.3 * thickness, rtol=1e-10)
        # Every facet has one tag: the windows are cut out of the faces they lie on.
        assert np.isclose(measures[top], (params.Lx - 0.5) * thickness, rtol=1e-10)
        x = _boundary_midpoints(mesh, "top_load")[:, 0]
        assert np.all(np.abs(x - 0.5 * params.Lx) < 0.25)
        x = _boundary_midpoints(mesh, "bottom_left")[:, 0]
        assert np.all(np.abs(x - params.Lx_BC) < 0.15)

    def test_damage_challenge_3d_rejects_a_load_window_inside_a_full_height_notch(self):
        params = _sample_params("DamageChallenge", dim=3, Lz=1.0, Ly_notch=2.0, Notch_Thickness=0.2, Load_Width=0.1)

        with pytest.raises(ValueError, match="lie entirely within the notch"):
            Mesh(params)

    @pytest.mark.parametrize("width", ["Load_Width", "BC_Width"])
    def test_damage_challenge_rejects_non_positive_window_widths(self, width):
        with pytest.raises(ValueError, match="Load_Width and BC_Width must be positive"):
            Mesh(_sample_params("DamageChallenge", **{width: 0.0}))

    def test_damage_challenge_warns_when_support_segments_overlap_the_notch(self, capsys):
        # These warnings are reachable with valid-but-awkward inputs where the
        # requested support locations coincide with the notch opening.
        params = _sample_params(
            "DamageChallenge",
            dim=2,
            Lx=4.0,
            Ly=2.0,
            dx=0.2,
            Lx_BC=2.0,
            Lx_notch=2.0,
            Notch_Thickness=0.2,
            Ly_notch=1.0,
            BC_Width=0.2,
            Load_Width=0.2,
        )

        Mesh(params)

        captured = capsys.readouterr()
        assert "bottom_left segment overlaps notch" in captured.out
        assert "bottom_right segment overlaps notch" in captured.out

    def test_damage_challenge_3d_rejects_invalid_diagonal_notch_specification(self):
        # The 3D diagonal-notch path expects exactly two x-positions.
        params = _sample_params(
            "DamageChallenge",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            dx=0.2,
            Lx_BC=1.0,
            Lx_notch=(1.0, 2.0, 3.0),
            Notch_Thickness=0.05,
            Ly_notch=1.0,
        )

        with pytest.raises(ValueError, match="Lx_notch must have 2 components"):
            Mesh(params)

    def test_damage_challenge_3d_supports_diagonal_notches(self):
        # A two-value Lx_notch tuple exercises the notch-rotation branch used
        # for diagonal 3D notches.
        params = _sample_params(
            "DamageChallenge",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            dx=0.2,
            Lx_BC=1.0,
            Lx_notch=(1.5, 2.5),
            Notch_Thickness=0.05,
            Ly_notch=1.0,
            Load_Width=0.2,
            BC_Width=0.2,
        )

        mesh = Mesh(params)

        _assert_markers(
            mesh,
            {
                "bottom",
                "right",
                "top",
                "left",
                "notch",
                "top_load",
                "bottom_left",
                "bottom_right",
                "front",
                "back",
            },
        )
        assert _domain_measure(mesh) < params.Lx * params.Ly * params.Lz

    def test_damage_challenge_3d_supports_full_height_notches(self):
        # If the notch reaches the full specimen height, the top and bottom
        # faces are split differently during gmsh boundary classification than
        # in the default benchmark geometry.
        params = _sample_params(
            "DamageChallenge",
            dim=3,
            Lx=4.0,
            Ly=2.0,
            Lz=1.0,
            dx=0.2,
            Lx_BC=0.5,
            Lx_notch=2.0,
            Notch_Thickness=0.2,
            Ly_notch=2.0,
            Load_Width=0.4,  # straddles the notch gap [1.9, 2.1]
            BC_Width=0.1,
        )

        mesh = Mesh(params)

        _assert_markers(
            mesh,
            {
                "bottom",
                "right",
                "top",
                "left",
                "notch",
                "top_load",
                "bottom_left",
                "bottom_right",
                "front",
                "back",
            },
        )
        assert _domain_measure(mesh) < params.Lx * params.Ly * params.Lz

    @pytest.mark.parametrize(
        ("params", "message"),
        [
            pytest.param(_sample_params("Rectangle", dim=1), "rectangle mesh generation", id="rectangle-invalid-dim"),
            pytest.param(_sample_params("Rectangle_Notches", dim=1), "rectangle-with-notches mesh generation", id="rectangle-notches-invalid-dim"),
            pytest.param(_sample_params("PlateWithHole", dim=1), "plate-with-hole mesh generation", id="plate-with-hole-invalid-dim"),
            pytest.param(_sample_params("FullPlateWithHole", dim=1), "full plate-with-hole mesh generation", id="full-plate-with-hole-invalid-dim"),
            pytest.param(_sample_params("SingleEdgeNotched", dim=1), "single-edge-notched mesh generation", id="single-edge-notched-invalid-dim"),
            pytest.param(_sample_params("DamageChallenge", dim=1), "DamageChallenge mesh generation", id="damage-challenge-invalid-dim"),
            pytest.param(_sample_params("Cylinder", dim=2), "Cylinder mesh generation", id="cylinder-invalid-dim"),
            pytest.param(_sample_params("NotchedTensileBar", dim=3), "NotchedTensileBar only supports 2-D meshes", id="notched-tensile-bar-invalid-dim"),
        ],
    )
    def test_mesh_generators_reject_unsupported_dimensions(self, params, message):
        # Unsupported dimensions should fail immediately with a generator-
        # specific error rather than producing a partially initialized mesh.
        with pytest.raises(NotImplementedError, match=message):
            Mesh(params)


class TestNotchedTensileBar:
    """Focused tests for the three-domain notched tensile bar generator."""

    @staticmethod
    def _domain_area(mesh, name):
        dx = ufl.Measure("dx", domain=mesh.mesh, subdomain_data=mesh.cell_tags)
        one = fem.Constant(mesh.mesh, 1.0)
        return fem.assemble_scalar(fem.form(one * dx(mesh.get_domain_tags(name))))

    @staticmethod
    def _boundary_length(mesh, name):
        ds = mesh.get_Boundary_Integrator()
        one = fem.Constant(mesh.mesh, 1.0)
        return fem.assemble_scalar(fem.form(one * ds(mesh.get_boundary_tags(name))))

    @staticmethod
    def _centre_bounds(params):
        return 0.5 * (params.Lx - params.Lx_centre), 0.5 * (params.Lx + params.Lx_centre)

    def test_domains_partition_the_bar_with_expected_areas(self):
        params = _sample_params("NotchedTensileBar")
        mesh = Mesh(params)

        outer_area = 0.5 * (params.Lx - params.Lx_centre) * params.Ly
        centre_area = params.Lx_centre * params.Ly - 2.0 * params.Notch_Width * params.Notch_Depth
        assert np.isclose(self._domain_area(mesh, "domain:left"), outer_area, rtol=1.0e-12)
        assert np.isclose(self._domain_area(mesh, "domain:right"), outer_area, rtol=1.0e-12)
        assert np.isclose(self._domain_area(mesh, "domain:centre"), centre_area, rtol=1.0e-12)
        assert np.isclose(self._domain_area(mesh, "domain"), 2.0 * outer_area + centre_area, rtol=1.0e-12)

    def test_full_domain_group_covers_every_cell(self):
        mesh = Mesh(_sample_params("NotchedTensileBar"))
        num_cells = mesh.mesh.topology.index_map(mesh.mesh.topology.dim).size_local

        parts = [mesh.get_domain_cells(name) for name in ("domain:left", "domain:centre", "domain:right")]
        np.testing.assert_array_equal(mesh.get_domain_cells("domain"), np.sort(np.concatenate(parts)))
        assert np.count_nonzero(mesh.get_domain_cells("domain") < num_cells) == num_cells

    def test_domain_cells_lie_on_the_expected_side_of_the_interfaces(self):
        params = _sample_params("NotchedTensileBar")
        mesh = Mesh(params)
        tdim = mesh.mesh.topology.dim
        x_c0, x_c1 = self._centre_bounds(params)

        def cell_x(name):
            return dmesh.compute_midpoints(mesh.mesh, tdim, mesh.get_domain_cells(name))[:, 0]

        assert np.all(cell_x("domain:left") < x_c0)
        assert np.all((cell_x("domain:centre") > x_c0) & (cell_x("domain:centre") < x_c1))
        assert np.all(cell_x("domain:right") > x_c1)

    @pytest.mark.parametrize("scale", [1.0, 0.025], ids=["sample-size", "small-bar"])
    def test_boundary_parts_follow_their_subdomain(self, scale):
        # Each "<boundary>:<subdomain>" part must lie within the x-range of
        # its subdomain, and the group must be the union of its parts. The
        # small bar guards against curve classification tolerances that only
        # work for unit-sized geometries.
        sample = _sample_params("NotchedTensileBar")
        lengths = ("Lx", "Ly", "Lx_centre", "Notch_Width", "Notch_Depth", "dx", "dx_centre")
        params = _sample_params("NotchedTensileBar", **{name: scale * getattr(sample, name) for name in lengths})
        mesh = Mesh(params)
        x_c0, x_c1 = self._centre_bounds(params)
        x_ranges = {"left": (0.0, x_c0), "centre": (x_c0, x_c1), "right": (x_c1, params.Lx)}

        for side in ("top", "bottom"):
            for subdomain, (x_min, x_max) in x_ranges.items():
                x_mid = _boundary_midpoints(mesh, f"{side}:{subdomain}")[:, 0]
                assert x_mid.size > 0
                assert np.all((x_mid > x_min) & (x_mid < x_max))
            part_lengths = sum(self._boundary_length(mesh, f"{side}:{subdomain}") for subdomain in x_ranges)
            assert np.isclose(self._boundary_length(mesh, side), part_lengths, rtol=1.0e-12)
            assert np.isclose(self._boundary_length(mesh, side), params.Lx - params.Notch_Width, rtol=1.0e-12)

        assert np.isclose(self._boundary_length(mesh, "left"), params.Ly, rtol=1.0e-12)
        assert np.isclose(self._boundary_length(mesh, "right"), params.Ly, rtol=1.0e-12)
        assert np.isclose(self._boundary_length(mesh, "notch"), 4.0 * params.Notch_Depth + 2.0 * params.Notch_Width, rtol=1.0e-12)

    @pytest.mark.parametrize(
        ("name", "neighbours"),
        [
            ("interface:left", {"domain:left", "domain:centre"}),
            ("interface:right", {"domain:centre", "domain:right"}),
        ],
    )
    def test_interfaces_are_conforming_interior_facets(self, name, neighbours):
        # Each interface facet must be shared by exactly one cell from each of
        # the two neighbouring domains, i.e. the domains share mesh nodes.
        params = _sample_params("NotchedTensileBar")
        mesh = Mesh(params)
        tdim = mesh.mesh.topology.dim
        mesh.mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_to_cell = mesh.mesh.topology.connectivity(tdim - 1, tdim)
        cell_domain = {}
        for domain in ("domain:left", "domain:centre", "domain:right"):
            for cell in mesh.get_domain_cells(domain):
                cell_domain[int(cell)] = domain

        facets = mesh.get_boundary_facets(name)
        assert facets.size > 0
        for facet in facets:
            cells = facet_to_cell.links(facet)
            assert {cell_domain[int(cell)] for cell in cells} == neighbours

        # Interior facets are not part of the exterior boundary measure.
        assert self._boundary_length(mesh, name) == 0.0

    def test_centre_domain_uses_the_finer_element_size(self):
        params = _sample_params("NotchedTensileBar")
        mesh = Mesh(params)
        tdim = mesh.mesh.topology.dim

        def mean_cell_size(name):
            cells = mesh.get_domain_cells(name)
            return np.mean(mesh.mesh.h(tdim, cells))

        assert mean_cell_size("domain:centre") < 0.5 * mean_cell_size("domain:left")
        assert mean_cell_size("domain:centre") < 0.5 * mean_cell_size("domain:right")

    @pytest.mark.parametrize(("notch_sides", "notch_y"), [("top", 1.0), ("bottom", 0.0)])
    def test_single_sided_notch_only_cuts_the_requested_edge(self, notch_sides, notch_y):
        params = _sample_params("NotchedTensileBar", Notch_Sides=notch_sides)
        mesh = Mesh(params)

        centre_area = params.Lx_centre * params.Ly - params.Notch_Width * params.Notch_Depth
        assert np.isclose(self._domain_area(mesh, "domain:centre"), centre_area, rtol=1.0e-12)
        notch_y_mid = _boundary_midpoints(mesh, "notch")[:, 1]
        assert np.all(np.abs(notch_y_mid - notch_y) <= params.Notch_Depth + 1.0e-12)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            pytest.param({"Lx": 0.0}, "Lx and Ly must be positive", id="non-positive-length"),
            pytest.param({"dx_centre": 0.0}, "dx and dx_centre must be positive", id="non-positive-size"),
            pytest.param({"Lx_centre": 4.0}, "Lx_centre must be positive and smaller than Lx", id="centre-too-wide"),
            pytest.param({"Notch_Width": 1.0}, "Notch_Width must be positive and smaller than Lx_centre", id="notch-wider-than-centre"),
            pytest.param({"Notch_Sides": "left"}, "Notch_Sides must be one of", id="invalid-notch-sides"),
            pytest.param({"Notch_Depth": 0.5}, r"smaller than Ly / 2", id="double-notch-too-deep"),
            pytest.param({"Notch_Depth": 1.0, "Notch_Sides": "top"}, "smaller than Ly", id="single-notch-too-deep"),
        ],
    )
    def test_rejects_invalid_geometry(self, overrides, message):
        params = _sample_params("NotchedTensileBar", **overrides)

        with pytest.raises(ValueError, match=message):
            Mesh(params)


class TestTwoSquared:
    """Focused tests for the two-squares generator."""

    def test_squares_have_equal_areas_and_a_shared_interface(self):
        params = _sample_params("TwoSquared")
        mesh = Mesh(params)
        dx = ufl.Measure("dx", domain=mesh.mesh, subdomain_data=mesh.cell_tags)
        one = fem.Constant(mesh.mesh, 1.0)

        for name in ("domain:left", "domain:right"):
            area = fem.assemble_scalar(fem.form(one * dx(mesh.get_domain_tags(name))))
            assert np.isclose(area, params.L**2, rtol=1e-12)

        interface_x = _boundary_midpoints(mesh, "interface")[:, 0]
        assert interface_x.size > 0
        np.testing.assert_allclose(interface_x, params.L)

    def test_boundary_parts_follow_their_square(self):
        params = _sample_params("TwoSquared")
        mesh = Mesh(params)

        for side in ("top", "bottom"):
            assert np.all(_boundary_midpoints(mesh, f"{side}:left")[:, 0] < params.L)
            assert np.all(_boundary_midpoints(mesh, f"{side}:right")[:, 0] > params.L)

    def test_optional_height_creates_a_thin_strip(self):
        params = _sample_params("TwoSquared", Ly=0.25)
        mesh = Mesh(params)

        assert np.isclose(_domain_measure(mesh), 2.0 * params.L * 0.25, rtol=1e-12)
        np.testing.assert_allclose(_boundary_midpoints(mesh, "top")[:, 1], 0.25)

    def test_interface_is_refined(self):
        params = _sample_params("TwoSquared")
        mesh = Mesh(params)

        facet_lengths = mesh.mesh.h(1, mesh.get_boundary_facets("interface"))
        outer_lengths = mesh.mesh.h(1, mesh.get_boundary_facets("left"))
        assert facet_lengths.mean() < 0.5 * outer_lengths.mean()

    @pytest.mark.parametrize(
        ("overrides", "error", "message"),
        [
            pytest.param({"dim": 3}, NotImplementedError, "only supports 2-D", id="3d"),
            pytest.param({"L": 0.0}, ValueError, "L and Ly must be positive", id="non-positive-length"),
            pytest.param({"Ly": -1.0}, ValueError, "L and Ly must be positive", id="non-positive-height"),
            pytest.param({"dx_interface": 0.0}, ValueError, "dx and dx_interface must be positive", id="non-positive-size"),
        ],
    )
    def test_rejects_invalid_parameters(self, overrides, error, message):
        with pytest.raises(error, match=message):
            Mesh(_sample_params("TwoSquared", **overrides))


class TestCorrosionPit:
    """Focused tests for the corrosion pit generator."""

    def test_metal_surface_is_split_into_pit_and_outer_surface(self):
        params = _sample_params("CorrosionPit")
        mesh = Mesh(params)

        pit = _boundary_midpoints(mesh, "metal:pit")
        outer = _boundary_midpoints(mesh, "metal:outer")
        distance = np.hypot(pit[:, 0] - 0.5 * params.W, pit[:, 1])
        np.testing.assert_allclose(distance, params.R_pit, rtol=1e-2)
        assert np.all(pit[:, 1] < 0.0)
        np.testing.assert_allclose(outer[:, 1], 0.0, atol=1e-12)
        assert np.all(np.abs(outer[:, 0] - 0.5 * params.W) > params.R_pit - 1e-12)

    def test_metal_surface_is_refined(self):
        params = _sample_params("CorrosionPit")
        mesh = Mesh(params)

        metal_lengths = mesh.mesh.h(1, mesh.get_boundary_facets("metal"))
        top_lengths = mesh.mesh.h(1, mesh.get_boundary_facets("top"))
        assert metal_lengths.mean() < 0.5 * top_lengths.mean()

    @pytest.mark.parametrize(
        ("overrides", "error", "message"),
        [
            pytest.param({"dim": 3}, NotImplementedError, "only supports 2-D", id="3d"),
            pytest.param({"H": 0.0}, ValueError, "must be positive", id="non-positive-height"),
            pytest.param({"R_pit": 2.0}, ValueError, "smaller than W / 2 and H", id="pit-too-wide"),
            pytest.param({"dx_metal": 0.0}, ValueError, "dx and dx_metal must be positive", id="non-positive-size"),
        ],
    )
    def test_rejects_invalid_parameters(self, overrides, error, message):
        with pytest.raises(error, match=message):
            Mesh(_sample_params("CorrosionPit", **overrides))


class TestPlateWithDefect:
    """Focused tests for the plate-with-defect generator."""

    def test_metal_surface_is_split_at_the_defect_radius_2d(self):
        params = _sample_params("PlateWithDefect")
        mesh = Mesh(params)

        defect = _boundary_midpoints(mesh, "metal:defect")
        outer = _boundary_midpoints(mesh, "metal:outer")
        np.testing.assert_allclose(defect[:, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(outer[:, 1], 0.0, atol=1e-12)
        assert np.all(defect[:, 0] < params.R_defect) and np.all(outer[:, 0] > params.R_defect)
        assert _boundary_measures(mesh)["metal:defect"] == pytest.approx(params.R_defect, rel=1e-12)
        assert _boundary_measures(mesh)["metal"] == pytest.approx(params.W, rel=1e-12)

    def test_quarter_defect_has_the_quarter_disk_area_3d(self):
        params = _sample_params("PlateWithDefect", dim=3, W=2.0, dx=0.5, dx_metal=0.1)
        mesh = Mesh(params)

        area = _boundary_measures(mesh)["metal:defect"]
        assert area == pytest.approx(0.25 * np.pi * params.R_defect**2, rel=2e-2)   # polygonal arc
        assert _boundary_measures(mesh)["metal"] == pytest.approx(params.W**2, rel=1e-12)
        assert _boundary_measures(mesh)["symmetry"] == pytest.approx(2.0 * params.W * params.H, rel=1e-12)

    def test_metal_surface_and_defect_edge_are_refined(self):
        params = _sample_params("PlateWithDefect", dx_defect=0.02)
        mesh = Mesh(params)

        metal = mesh.mesh.h(1, mesh.get_boundary_facets("metal"))
        top = mesh.mesh.h(1, mesh.get_boundary_facets("top"))
        assert metal.mean() < 0.5 * top.mean()
        midpoints = _boundary_midpoints(mesh, "metal")
        near_edge = np.abs(midpoints[:, 0] - params.R_defect) < 0.05
        assert metal[near_edge].max() < 0.5 * params.dx_metal

    @pytest.mark.parametrize(
        ("overrides", "error", "message"),
        [
            pytest.param({"dim": 1}, ValueError, "dim = 2 or 3", id="1d"),
            pytest.param({"H": 0.0}, ValueError, "must be positive", id="non-positive-height"),
            pytest.param({"R_defect": 5.0}, ValueError, "smaller than W", id="defect-too-wide"),
            pytest.param({"dx_metal": 0.0}, ValueError, "must be positive", id="non-positive-size"),
            pytest.param({"dx_pit": -0.1}, ValueError, "must be positive", id="non-positive-pit-size"),
            pytest.param({"grading": 0.0}, ValueError, "grading must be positive", id="non-positive-grading"),
            pytest.param({"lateral_grading": -0.1, "dx_far": 1.0}, ValueError, "lateral_grading must be positive",
                         id="non-positive-lateral-grading"),
            pytest.param({"lateral_grading": 0.1}, ValueError, "requires a positive dx_far", id="lateral-without-far-size"),
            pytest.param({"lateral_grading": 0.1, "dx_far": 1.0, "fine_extent": -1.0}, ValueError, "must not be negative",
                         id="negative-fine-extent"),
            pytest.param({"lateral_grading": 0.1, "dx_far": 1.0, "feature_extent": -1.0}, ValueError, "must not be negative",
                         id="negative-feature-extent"),
            pytest.param({"weak_spots": [(2.0, 0.0)]}, ValueError, "half-widths must be positive", id="spot-without-width"),
            pytest.param({"weak_spots": [(0.6, 0.2)]}, ValueError, "within \\(R_defect, W\\)", id="spot-on-the-defect"),
            pytest.param({"weak_spots": [(3.9, 0.2)]}, ValueError, "within \\(R_defect, W\\)", id="spot-beyond-the-plate"),
            pytest.param({"weak_spots": [(2.0, 0.3), (2.4, 0.2)]}, ValueError, "without overlapping", id="overlapping-spots"),
            pytest.param({"dim": 3, "W": 2.0, "weak_spots": [(1.0, 0.1)]}, ValueError, "2-D only", id="spots-in-3d"),
            pytest.param({"defect_depth": -0.1}, ValueError, "must not be negative", id="negative-depth"),
            pytest.param({"voids": [(2.0, 0.2, 0.2)]}, ValueError, "0 < mouth < radius", id="void-mouth-too-wide"),
            pytest.param({"voids": [(0.6, 0.2, 0.05)]}, ValueError, "within \\(R_defect, W\\)", id="void-on-the-defect"),
            pytest.param({"voids": [(2.0, 0.2, 0.05)], "weak_spots": [(2.3, 0.2)]}, ValueError, "without overlapping",
                         id="void-overlapping-a-spot"),
            pytest.param({"dim": 3, "W": 2.0, "voids": [(1.0, 0.1, 0.05)]}, ValueError, "2-D only", id="voids-in-3d"),
            pytest.param({"voids": [(2.0, 0.2, 0.05)], "dx_void": 0.0}, ValueError, "dx_void must be positive",
                         id="non-positive-void-size"),
            pytest.param({"dim": 3, "W": 2.0, "defect_depth": 0.1}, ValueError, "2-D only", id="depth-in-3d"),
        ],
    )
    def test_rejects_invalid_parameters(self, overrides, error, message):
        with pytest.raises(error, match=message):
            Mesh(_sample_params("PlateWithDefect", **overrides))

    def test_pit_surface_is_refined(self):
        params = _sample_params("PlateWithDefect", dx_pit=0.02)
        mesh = Mesh(params)

        defect = mesh.mesh.h(1, mesh.get_boundary_facets("metal:defect"))
        assert defect.max() < 0.5 * params.dx_metal

    def test_graded_mesh_is_fine_at_the_metal_and_coarse_in_the_bulk(self):
        params = _sample_params("PlateWithDefect", W=10.0, H=10.0, dx=2.0, dx_metal=0.1, dx_pit=0.05, dx_defect=0.02,
                                grading=0.3)
        mesh = Mesh(params)

        defect = mesh.mesh.h(1, mesh.get_boundary_facets("metal:defect"))
        far_metal = _boundary_midpoints(mesh, "metal")[:, 0] > 5.0
        metal = mesh.mesh.h(1, mesh.get_boundary_facets("metal"))
        top = mesh.mesh.h(1, mesh.get_boundary_facets("top"))
        assert defect.max() < 0.1
        assert metal[far_metal].max() < 0.2
        assert top.mean() > 1.0                       # bulk: close to dx
        assert _boundary_measures(mesh)["metal"] == pytest.approx(params.W, rel=1e-12)

    def test_weak_spots_are_separate_metal_surfaces_with_refined_edges(self):
        params = _sample_params("PlateWithDefect", dx_defect=0.02, weak_spots=[(3.0, 0.25), (1.5, 0.1)])
        mesh = Mesh(params)

        measures = _boundary_measures(mesh)
        # numbered outwards from the defect, whatever the order given
        assert measures["metal:spot1"] == pytest.approx(0.2, rel=1e-12)
        assert measures["metal:spot2"] == pytest.approx(0.5, rel=1e-12)
        assert measures["metal:outer"] == pytest.approx(params.W - params.R_defect - 0.7, rel=1e-12)
        assert measures["metal"] == pytest.approx(params.W, rel=1e-12)
        spot = _boundary_midpoints(mesh, "metal:spot2")[:, 0]
        assert np.all(np.abs(spot - 3.0) < 0.25)
        metal = mesh.mesh.h(1, mesh.get_boundary_facets("metal"))
        near_edge = np.abs(_boundary_midpoints(mesh, "metal")[:, 0] - 2.75) < 0.05
        assert metal[near_edge].max() < 0.5 * params.dx_metal

    def test_voids_are_cavities_behind_a_narrow_mouth(self):
        voids = [(3.2, 0.2, 0.05), (2.0, 0.3, 0.1)]
        params = _sample_params("PlateWithDefect", voids=voids, weak_spots=[(1.2, 0.1)], dx_void=0.02)
        mesh = Mesh(params)

        measures = _boundary_measures(mesh)
        area = mesh.mesh.comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(mesh.mesh, 1.0) * ufl.dx)))
        # numbered outwards; each wall is the circle below its mouth chord
        for name, (_c, radius, mouth) in zip(("metal:void1", "metal:void2"), sorted(voids)):
            assert measures[name] == pytest.approx(radius * (2.0 * np.pi - 2.0 * np.arcsin(mouth / radius)), rel=1e-3)
        assert measures["metal:outer"] == pytest.approx(params.W - params.R_defect - 0.2 - 0.2 - 0.1, rel=1e-12)
        cavities = sum(radius**2 * (np.pi - np.arcsin(mouth / radius)) + mouth * np.sqrt(radius**2 - mouth**2)
                       for _c, radius, mouth in voids)
        assert area == pytest.approx(params.W * params.H + cavities, rel=1e-3)
        wall = _boundary_midpoints(mesh, "metal:void1")
        assert wall[:, 1].min() < -0.3 and np.all(np.abs(wall[:, 0] - 2.0) <= 0.3 + 1e-9)
        assert mesh.mesh.h(1, mesh.get_boundary_facets("metal:void1")).max() < 1.5 * params.dx_void

    def test_void_refinement_does_not_stop_lateral_coarsening(self):
        params = _sample_params("PlateWithDefect", W=40.0, H=0.2, dx=0.1, dx_metal=0.05, lateral_grading=0.1, dx_far=1.0,
                                voids=[(5.0, 0.1, 0.03)], dx_void=0.01)
        mesh = Mesh(params)

        metal = mesh.mesh.h(1, mesh.get_boundary_facets("metal"))
        x = _boundary_midpoints(mesh, "metal")[:, 0]
        assert metal[x > 30.0].min() > 0.5
        assert mesh.mesh.h(1, mesh.get_boundary_facets("metal:void1")).max() < 0.02

    def test_feature_extent_limits_the_fine_region_around_spots_and_voids(self):
        common = dict(W=40.0, H=0.2, dx=0.1, dx_metal=0.05, lateral_grading=0.1, dx_far=1.0, fine_extent=5.0,
                      weak_spots=[(20.0, 0.1)], voids=[(30.0, 0.1, 0.03)], dx_void=0.02)
        wide = Mesh(_sample_params("PlateWithDefect", **common))
        narrow = Mesh(_sample_params("PlateWithDefect", feature_extent=0.5, **common))

        def size_near(mesh, centre):
            x = _boundary_midpoints(mesh, "metal")[:, 0]
            h = mesh.mesh.h(1, mesh.get_boundary_facets("metal"))
            return h[np.abs(x - centre - 3.0) < 0.5].min()

        # 3 units from the features: fine within fine_extent by default, coarsened with a small feature_extent
        for centre in (20.0, 30.0):
            assert size_near(wide, centre) < 0.1
            assert size_near(narrow, centre) > 0.2

    def test_defect_depth_makes_the_defect_a_recess(self):
        params = _sample_params("PlateWithDefect", defect_depth=0.3, weak_spots=[(2.0, 0.2)])
        mesh = Mesh(params)

        measures = _boundary_measures(mesh)
        area = mesh.mesh.comm.allreduce(fem.assemble_scalar(fem.form(fem.Constant(mesh.mesh, 1.0) * ufl.dx)))
        # bottom and wall of the recess
        assert measures["metal:defect"] == pytest.approx(params.R_defect + 0.3, rel=1e-12)
        assert measures["metal:outer"] == pytest.approx(params.W - params.R_defect - 0.4, rel=1e-12)
        assert measures["metal:spot1"] == pytest.approx(0.4, rel=1e-12)
        assert measures["symmetry"] == pytest.approx(params.H + 0.3, rel=1e-12)
        assert area == pytest.approx(params.W * params.H + params.R_defect * 0.3, rel=1e-12)
        assert _boundary_midpoints(mesh, "metal:defect")[:, 1].min() < -0.2

    @pytest.mark.parametrize("grading", [None, 0.3])
    def test_lateral_grading_coarsens_far_from_the_defect_only(self, grading):
        thin = dict(W=40.0, H=0.2, dx=0.1, dx_metal=0.05, grading=grading)
        params = _sample_params("PlateWithDefect", **thin, lateral_grading=0.1, dx_far=1.0, fine_extent=2.0)
        mesh = Mesh(params)

        x = _boundary_midpoints(mesh, "metal")[:, 0]
        metal = mesh.mesh.h(1, mesh.get_boundary_facets("metal"))
        assert metal[x < params.R_defect + 2.0].max() < 1.5 * params.dx_metal    # unchanged within fine_extent
        assert metal[x > 30.0].min() > 0.5 * params.dx_far                        # coarse far away, through the layer
        assert metal.size < 0.25 * Mesh(_sample_params("PlateWithDefect", **thin)).mesh.h(
            1, Mesh(_sample_params("PlateWithDefect", **thin)).get_boundary_facets("metal")).size
        assert _boundary_measures(mesh)["metal"] == pytest.approx(params.W, rel=1e-12)



@pytest.mark.parametrize(
    "ranges, message",
    [
        pytest.param(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), "one pair per notch", id="wrong-count"),
        pytest.param(((1.0, 0.5),), "y_min < y_max", id="reversed"),
        pytest.param(((0.0, 2.5),), "y_min < y_max", id="beyond-width"),
        pytest.param(((0.0, 1.0, 2.0),), "pairs", id="not-a-pair"),
    ],
)
def test_rectangle_notches_3d_rejects_invalid_notch_y_ranges(ranges, message):
    from femcubes.mesh_generators.rectangle_notches import RectangleNotchesMeshGenerator

    params = _sample_params("Rectangle_Notches", dim=3, Lx=4.0, Ly=2.0, Lz=1.0, dx=0.5,
                            Notch_Positions=(1.0, 3.0), Notch_Width=0.2, Notch_Depth=0.5, Notch_Y_Ranges=ranges)
    generator = RectangleNotchesMeshGenerator()
    intervals, _ = generator._resolve_notch_intervals(params)
    with pytest.raises(ValueError, match=message):
        generator._resolve_notch_y_ranges(params, len(intervals))
