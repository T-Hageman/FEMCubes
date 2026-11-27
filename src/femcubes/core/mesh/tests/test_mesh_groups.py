"""Tests for the _MeshGroups mix-in."""

import numpy as np
import pytest

import femcubes.core.mesh._mesh_groups as mesh_groups_module
from femcubes.core.mesh._mesh_groups import _MeshGroups


class _MeshGroupsEnd:
    """Terminate the cooperative super() chain for the mix-in harness."""

    def __init__(self, params):
        self.params = params


class _FakeTags:
    """Minimal stand-in for dolfinx MeshTags."""

    def __init__(self, values_by_tag):
        self._values_by_tag = {
            tag: np.asarray(values, dtype=np.int32)
            for tag, values in values_by_tag.items()
        }

    def find(self, tag):
        return self._values_by_tag.get(tag, np.array([], dtype=np.int32))


class _MeshGroupsHarness(_MeshGroups, _MeshGroupsEnd):
    """Concrete object exposing _MeshGroups behavior for unit tests."""

    def __init__(self):
        self.facet_tags = _FakeTags({7: [1, 3]})
        self.cell_tags = _FakeTags({11: [0, 2, 4]})
        super().__init__(params=None)


@pytest.fixture
def mesh_groups():
    groups = _MeshGroupsHarness()
    groups.add_boundary_marker("wall", 7)
    groups.add_domain_marker("solid", 11)
    return groups


class TestMeshGroups:
    def test_add_and_get_boundary_marker(self, mesh_groups):
        assert mesh_groups.get_boundary_tag("wall") == 7

    def test_add_and_get_domain_marker(self, mesh_groups):
        assert mesh_groups.get_domain_tag("solid") == 11

    def test_missing_boundary_tag_raises_helpful_error(self, mesh_groups):
        with pytest.raises(KeyError, match="Available boundaries"):
            mesh_groups.get_boundary_tag("inlet")

    def test_missing_domain_tag_raises_helpful_error(self, mesh_groups):
        with pytest.raises(KeyError, match="Available domains"):
            mesh_groups.get_domain_tag("fluid")

    def test_get_boundary_facets_uses_named_marker_lookup(self, mesh_groups):
        np.testing.assert_array_equal(mesh_groups.get_boundary_facets("wall"), [1, 3])

    def test_get_domain_cells_uses_named_marker_lookup(self, mesh_groups):
        np.testing.assert_array_equal(mesh_groups.get_domain_cells("solid"), [0, 2, 4])

    def test_list_methods_return_copies(self, mesh_groups):
        boundaries = mesh_groups.list_boundaries()
        domains = mesh_groups.list_domains()

        boundaries["extra"] = 99
        domains["extra"] = 99

        assert mesh_groups.boundary_names() == ["wall"]
        assert mesh_groups.domain_names() == ["solid"]

    def test_name_methods_return_insertion_order(self, mesh_groups):
        mesh_groups.add_boundary_marker("inlet", 9)
        mesh_groups.add_domain_marker("fluid", 12)

        assert mesh_groups.boundary_names() == ["wall", "inlet"]
        assert mesh_groups.domain_names() == ["solid", "fluid"]

    def test_print_mesh_groups_includes_named_domains_and_boundaries(self, mesh_groups, monkeypatch):
        printed = []
        monkeypatch.setattr(mesh_groups_module, "mprint", lambda message: printed.append(message))

        mesh_groups.print_mesh_groups()

        assert any("Domains" in line for line in printed)
        assert any("solid: tag=11" in line for line in printed)
        assert any("Boundaries" in line for line in printed)
        assert any("wall: tag=7" in line for line in printed)


class _HierarchicalMeshGroupsHarness(_MeshGroups, _MeshGroupsEnd):
    """Mesh-groups harness with ``group:part`` markers."""

    def __init__(self):
        self.facet_tags = _FakeTags({1: [4, 6], 2: [5], 3: [0], 4: [7]})
        self.cell_tags = _FakeTags({1: [0, 3], 2: [1, 2], 3: [4]})
        super().__init__(params=None)


@pytest.fixture
def hierarchical_groups():
    groups = _HierarchicalMeshGroupsHarness()
    groups.add_domain_marker("domain:left", 1)
    groups.add_domain_marker("domain:centre", 2)
    groups.add_domain_marker("domain:right", 3)
    groups.add_boundary_marker("top:left", 1)
    groups.add_boundary_marker("top:right", 2)
    groups.add_boundary_marker("left:left", 3)
    groups.add_boundary_marker("interface:left", 4)
    return groups


class TestHierarchicalMeshGroups:
    def test_group_names_resolve_to_all_part_tags(self, hierarchical_groups):
        assert hierarchical_groups.get_domain_tags("domain") == (1, 2, 3)
        assert hierarchical_groups.get_boundary_tags("top") == (1, 2)

    def test_part_names_resolve_to_one_tag(self, hierarchical_groups):
        assert hierarchical_groups.get_domain_tags("domain:centre") == (2,)
        assert hierarchical_groups.get_domain_tag("domain:centre") == 2
        assert hierarchical_groups.get_boundary_tag("top:right") == 2

    def test_single_tag_lookup_accepts_groups_with_one_part(self, hierarchical_groups):
        assert hierarchical_groups.get_boundary_tag("left") == 3

    def test_single_tag_lookup_rejects_groups_with_several_parts(self, hierarchical_groups):
        with pytest.raises(ValueError, match="use get_domain_tags"):
            hierarchical_groups.get_domain_tag("domain")
        with pytest.raises(ValueError, match="use get_boundary_tags"):
            hierarchical_groups.get_boundary_tag("top")

    def test_group_lookups_return_the_sorted_union_of_their_parts(self, hierarchical_groups):
        np.testing.assert_array_equal(hierarchical_groups.get_domain_cells("domain"), [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(hierarchical_groups.get_boundary_facets("top"), [4, 5, 6])
        np.testing.assert_array_equal(hierarchical_groups.get_domain_cells("domain:centre"), [1, 2])

    def test_partial_name_components_are_not_groups(self, hierarchical_groups):
        # "dom" is a string prefix of "domain:left" but not a ':'-separated group.
        with pytest.raises(KeyError, match="Available domains"):
            hierarchical_groups.get_domain_tags("dom")

    def test_names_list_each_group_before_its_parts(self, hierarchical_groups):
        assert hierarchical_groups.domain_names() == ["domain", "domain:left", "domain:centre", "domain:right"]
        assert hierarchical_groups.boundary_names() == [
            "top", "top:left", "top:right", "left", "left:left", "interface", "interface:left",
        ]

    def test_list_methods_return_only_stored_parts(self, hierarchical_groups):
        assert hierarchical_groups.list_domains() == {"domain:left": 1, "domain:centre": 2, "domain:right": 3}

    @pytest.mark.parametrize("name", ["domain", "domain:left:upper"])
    def test_part_and_group_names_cannot_clash(self, hierarchical_groups, name):
        with pytest.raises(ValueError, match="cannot be both a part and a group"):
            hierarchical_groups.add_domain_marker(name, 9)

    @pytest.mark.parametrize("name", ["", "domain:", ":left", "domain::left"])
    def test_empty_name_components_are_rejected(self, hierarchical_groups, name):
        with pytest.raises(ValueError, match="must not be empty"):
            hierarchical_groups.add_boundary_marker(name, 9)

    def test_print_mesh_groups_shows_groups_with_their_tags(self, hierarchical_groups, monkeypatch):
        printed = []
        monkeypatch.setattr(mesh_groups_module, "mprint", lambda message: printed.append(message))

        hierarchical_groups.print_mesh_groups()

        assert any("domain: tags=[1, 2, 3]" in line for line in printed)
        assert any("domain:left: tag=1" in line for line in printed)
        assert any("top: tags=[1, 2]" in line for line in printed)
