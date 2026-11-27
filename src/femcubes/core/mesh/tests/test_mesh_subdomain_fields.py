"""Tests for fields restricted to subdomains (multi-domain support)."""

import numpy as np
import pytest
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import assemble_matrix, create_matrix

from femcubes.core.mesh.mesh import Mesh
from femcubes.mesh_generators.notched_tensile_bar import Sample_Parameters
from femcubes.core.time_discretizations import Newmark_Update


@pytest.fixture
def bar_mesh():
    params = Sample_Parameters()
    params.Quadrature_Degree = 2
    return Mesh(params)


def _num_cells(mesh):
    index_map = mesh.topology.index_map(mesh.topology.dim)
    return index_map.size_local + index_map.num_ghosts


class TestFieldSupports:
    def test_full_domain_fields_live_on_the_parent_mesh(self, bar_mesh):
        field = bar_mesh.Add_Field("T", "Temperature", "CG", 1)

        support = bar_mesh.Get_Field_Support("T")
        assert support.is_full
        assert field.function_space.mesh is bar_mesh.mesh

    def test_subdomain_fields_only_have_dofs_on_their_subdomain(self, bar_mesh):
        full = bar_mesh.Add_Field("full", "Full", "CG", 1)
        centre = bar_mesh.Add_Field("centre", "Centre", "CG", 1, domain="domain:centre")

        support = bar_mesh.Get_Field_Support("centre")
        assert not support.is_full
        assert support.domain_names == ("domain:centre",)
        assert centre.function_space.mesh is support.mesh
        assert _num_cells(support.mesh) == bar_mesh.get_domain_cells("domain:centre").size
        assert 0 < centre.function_space.dofmap.index_map.size_global < full.function_space.dofmap.index_map.size_global

    def test_fields_on_the_same_subdomain_share_one_submesh(self, bar_mesh):
        first = bar_mesh.Add_Field("first", "First", "CG", 1, domain="domain:centre")
        second = bar_mesh.Add_Quadrature_Field("second", "Second", (), domain="domain:centre")

        assert first.function_space.mesh is second.function_space.mesh
        assert len(bar_mesh.get_entity_maps()) == 1

    def test_field_registered_for_all_subdomains_lives_on_the_parent_mesh(self, bar_mesh):
        for domain in ("domain:left", "domain:centre", "domain:right"):
            field = bar_mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 2, domain=domain)

        assert bar_mesh.Get_Field_Support("u").is_full
        assert field.function_space.mesh is bar_mesh.mesh
        assert bar_mesh.Get_Field_Support("u").domain_names == ("domain:left", "domain:centre", "domain:right")

    def test_extending_a_support_recreates_the_field_and_its_test_function(self, bar_mesh):
        bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        bar_mesh.Add_TestFunction("phi", step=0)

        field = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:right")

        assert bar_mesh.Get_Field_Support("phi").domain_names == ("domain:centre", "domain:right")
        assert bar_mesh.Get_TestFunction("phi", 0).ufl_function_space() == field.function_space

    def test_supports_cannot_be_extended_after_finalizing_the_steps(self, bar_mesh):
        bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        bar_mesh.Add_TestFunction("phi", step=0)
        bar_mesh.finalize_test_functions()

        with pytest.raises(ValueError, match="after the solution steps have been finalized"):
            bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:left")

    def test_unknown_domains_are_rejected(self, bar_mesh):
        with pytest.raises(KeyError, match="Domain 'domain:middle' not found.*facet groups"):
            bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:middle")


class TestDomainOperations:
    def test_domain_integrator_integrates_over_one_subdomain(self, bar_mesh):
        params = bar_mesh.params
        dx = bar_mesh.Get_Domain_Integrator("domain:centre")
        area = fem.assemble_scalar(fem.form(fem.Constant(bar_mesh.mesh, 1.0) * dx))

        expected = params.Lx_centre * params.Ly - 2.0 * params.Notch_Width * params.Notch_Depth
        assert np.isclose(area, expected, rtol=1e-12)

    def test_field_dofs_of_sharing_domains_are_disjoint_and_cover_the_field(self, bar_mesh):
        for domain in ("domain:left", "domain:centre", "domain:right"):
            field = bar_mesh.Add_Vector_Field("u", "Displacement", "CG", 2, 2, domain=domain)

        owned = [bar_mesh.Get_Field_Dofs("u", domain) for domain in ("domain:left", "domain:centre", "domain:right")]
        merged = np.concatenate(owned)

        assert merged.size == np.unique(merged).size
        np.testing.assert_array_equal(np.sort(merged), np.arange(field.x.array.size))
        assert bar_mesh.Get_Field_Dofs("u", "domain") is None

    def test_restricted_operations_only_touch_the_owned_dofs(self, bar_mesh):
        for domain in ("domain:left", "domain:centre"):
            bar_mesh.Add_Field("T", "Temperature", "CG", 1, domain=domain)
            bar_mesh.Add_Field("T_old", "Old temperature", "CG", 1, domain=domain)
        bar_mesh.Add_Field("T", "Temperature", "CG", 1, domain="domain:right")
        bar_mesh.Add_Field("T_old", "Old temperature", "CG", 1, domain="domain:right")
        centre_dofs = bar_mesh.Get_Field_Dofs("T", "domain:centre")

        bar_mesh.Constant_Field("T", 2.0, domain="domain:centre")
        bar_mesh.Copy_Field("T", "T_old", domain="domain:centre")

        T = bar_mesh.fields["T"].x.array
        assert np.all(T[centre_dofs] == 2.0)
        assert np.count_nonzero(T) == centre_dofs.size
        np.testing.assert_array_equal(bar_mesh.fields["T_old"].x.array, T)

    def test_restricted_copy_requires_matching_discretizations(self, bar_mesh):
        for domain in ("domain:left", "domain:centre"):
            bar_mesh.Add_Field("T", "Temperature", "CG", 1, domain=domain)
            bar_mesh.Add_Field("T2", "Quadratic temperature", "CG", 2, domain=domain)

        with pytest.raises(ValueError, match="same discretization and support"):
            bar_mesh.Copy_Field("T", "T2", domain="domain:centre")

    def test_set_from_expression_only_sets_the_cells_of_the_domain(self, bar_mesh):
        bar_mesh.Add_Field("p", "Pressure", "DG", 0)
        x = ufl.SpatialCoordinate(bar_mesh.mesh)

        bar_mesh.Set_From_Expression("p", 1.0 + x[0], domain="domain:centre")

        p = bar_mesh.fields["p"]
        centre_cells = bar_mesh.get_domain_cells("domain:centre")
        centre_dofs = p.function_space.dofmap.list[centre_cells].ravel()
        assert np.all(p.x.array[centre_dofs] > 0.0)
        assert np.count_nonzero(p.x.array) == centre_dofs.size

    def test_parent_expression_is_evaluated_into_a_subdomain_field(self, bar_mesh):
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        x = ufl.SpatialCoordinate(bar_mesh.mesh)

        bar_mesh.Set_From_Expression("phi", 2.0 * x[0])

        coordinates = phi.function_space.tabulate_dof_coordinates()
        np.testing.assert_allclose(phi.x.array, 2.0 * coordinates[:, 0], atol=1e-12)

    def test_mixed_mesh_expression_is_evaluated_on_the_subdomain(self, bar_mesh):
        # A parent-mesh field degraded by a subdomain field, as in damaged
        # elasticity, is set on the subdomain cells of a full-mesh field.
        u = bar_mesh.Add_Field("u", "Displacement", "CG", 1)
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        bar_mesh.Add_Field("s", "Stress", "DG", 0)
        u.interpolate(lambda x: 3.0 * x[0])
        phi.x.array[:] = 0.5
        scale = fem.Constant(bar_mesh.mesh, 4.0)

        bar_mesh.Set_From_Expression("s", scale * (1.0 - phi) * u.dx(0), domain="domain:centre")

        s = bar_mesh.fields["s"]
        centre_dofs = s.function_space.dofmap.list[bar_mesh.get_domain_cells("domain:centre")].ravel()
        np.testing.assert_allclose(s.x.array[centre_dofs], 4.0 * 0.5 * 3.0)
        assert np.count_nonzero(s.x.array) == centre_dofs.size

    def test_mixed_mesh_expression_outside_the_subdomain_is_rejected(self, bar_mesh):
        u = bar_mesh.Add_Field("u", "Displacement", "CG", 1)
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        bar_mesh.Add_Field("s", "Stress", "DG", 0)

        with pytest.raises(ValueError, match="outside that subdomain"):
            bar_mesh.Set_From_Expression("s", phi * u)

    def test_evaluate_at_field_points_lines_up_with_a_subdomain_quadrature_field(self, bar_mesh):
        energy = bar_mesh.Add_Quadrature_Field("energy", "Energy", (), domain="domain:centre")
        x = ufl.SpatialCoordinate(bar_mesh.mesh)

        values = bar_mesh.Evaluate_At_Field_Points("energy", ufl.as_vector([x[0], 2.0 * x[1]]))
        bar_mesh.Set_From_Expression("energy", x[0])

        assert values.shape == (energy.x.array.size, 2)
        np.testing.assert_allclose(values[:, 0], energy.x.array, atol=1e-12)

    def test_newmark_update_only_updates_the_given_dofs(self, bar_mesh):
        names = ("u", "v", "a", "uOld", "vOld", "aOld")
        fields = [bar_mesh.Add_Field(name, name, "CG", 1) for name in names]
        fields[0].x.array[:] = 1.0
        dofs = np.arange(0, fields[0].x.array.size, 2)

        Newmark_Update(*fields, 0.1, 0.25, 0.5, dofs=dofs)

        v = fields[1].x.array
        assert np.all(v[dofs] != 0.0)
        assert np.count_nonzero(v) == dofs.size


class TestSubdomainAssembly:
    def test_block_step_with_a_subdomain_field_matches_its_scalar_blocks(self, bar_mesh):
        u = bar_mesh.Add_Field("u", "Displacement", "CG", 1)
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        bar_mesh.Add_TestFunction("u", step=0)
        bar_mesh.Add_TestFunction("phi", step=0)
        bar_mesh.finalize_test_functions()
        v, q = bar_mesh.Get_TestFunction("u", 0), bar_mesh.Get_TestFunction("phi", 0)
        du, dphi = bar_mesh.Get_Trial_Function("u", 0), bar_mesh.Get_Trial_Function("phi", 0)
        dx = bar_mesh.Get_Domain_Integrator()
        dx_c = bar_mesh.Get_Domain_Integrator("domain:centre")

        K = ufl.inner(ufl.grad(du), ufl.grad(v)) * dx + (dphi * q + du * q) * dx_c
        blocks = bar_mesh.Compile_Form(K)
        matrix = create_matrix(blocks, kind="mpi")
        assemble_matrix(matrix, blocks)
        matrix.assemble()

        n_u = u.function_space.dofmap.index_map.size_global
        n_phi = phi.function_space.dofmap.index_map.size_global
        assert matrix.getSize() == (n_u + n_phi, n_u + n_phi)
        # The phase-field diagonal block is the mass matrix of the centre:
        # its entries sum to the centre area.
        centre_rows = np.arange(n_u, n_u + n_phi, dtype=np.int32)
        values = matrix.getValues(centre_rows, centre_rows)
        params = bar_mesh.params
        expected = params.Lx_centre * params.Ly - 2.0 * params.Notch_Width * params.Notch_Depth
        assert np.isclose(values.sum(), expected, rtol=1e-10)
        matrix.destroy()

    def test_compile_form_assembles_scalars_mixing_meshes(self, bar_mesh):
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        phi.x.array[:] = 2.0

        value = fem.assemble_scalar(bar_mesh.Compile_Form(phi * bar_mesh.Get_Domain_Integrator("domain:centre")))

        params = bar_mesh.params
        expected = 2.0 * (params.Lx_centre * params.Ly - 2.0 * params.Notch_Width * params.Notch_Depth)
        assert np.isclose(value, expected, rtol=1e-12)


class TestSubdomainBoundaryConditions:
    def test_dirichlet_bc_on_a_subdomain_field_constrains_its_boundary_nodes(self, bar_mesh):
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        phi.x.array[:] = 3.0

        bcs = bar_mesh.Build_Increment_Dirichlet_BCs("phi", "notch", 1.0)

        dofs = bcs[0].dof_indices()[0]
        coordinates = phi.function_space.tabulate_dof_coordinates()[dofs]
        params = bar_mesh.params
        assert dofs.size > 0
        assert np.all(np.abs(coordinates[:, 0] - 0.5 * params.Lx) <= 0.5 * params.Notch_Width + 1e-12)
        # Incremental BC value: current minus target.
        np.testing.assert_allclose(bcs[0].g.x.array[dofs], 2.0)

    def test_dirichlet_bc_on_a_subdomain_vector_component(self, bar_mesh):
        u = bar_mesh.Add_Vector_Field("u", "Displacement", "CG", 1, 2, domain="domain:right")

        bcs = bar_mesh.Build_Increment_Dirichlet_BCs("u", "right", 0.0, component=1)

        dofs = bcs[0].dof_indices()[0]
        assert dofs.size > 0
        coordinates = u.function_space.tabulate_dof_coordinates()
        np.testing.assert_allclose(coordinates[dofs // 2, 0], bar_mesh.params.Lx)

    def test_dirichlet_bc_outside_the_support_is_rejected(self, bar_mesh):
        bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")

        with pytest.raises(ValueError, match="outside the support"):
            bar_mesh.Build_Increment_Dirichlet_BCs("phi", "left", 0.0)


class TestSubdomainOutput:
    def test_output_is_nan_outside_the_support(self, bar_mesh):
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        phi.x.array[:] = 0.25

        output = bar_mesh.Prepare_Field_For_Output("phi")

        assert output.function_space.mesh is bar_mesh.mesh
        finite = np.isfinite(output.values)
        assert np.any(finite) and not np.all(finite)
        np.testing.assert_allclose(output.values[finite], 0.25)

    def test_quadrature_output_is_nan_outside_the_support(self, bar_mesh):
        energy = bar_mesh.Add_Quadrature_Field("energy", "Energy", (), domain="domain:centre")
        energy.x.array[:] = 3.0

        output = bar_mesh.Prepare_Field_For_Output("energy")

        finite = np.isfinite(output.values)
        assert np.any(finite) and not np.all(finite)
        np.testing.assert_allclose(output.values[finite], 3.0)

    def test_plotting_uses_the_subdomain_only_by_default(self, bar_mesh):
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        phi.x.array[:] = 0.25

        output = bar_mesh.Prepare_Field_For_Plotting("phi")

        assert output.function_space.mesh is phi.function_space.mesh
        np.testing.assert_allclose(output.values, 0.25)

    def test_plotting_on_the_full_mesh_can_be_requested(self, bar_mesh):
        phi = bar_mesh.Add_Field("phi", "Phase", "CG", 1, domain="domain:centre")
        phi.x.array[:] = 0.25
        bar_mesh.params.Plot_Subdomains_Only = False

        output = bar_mesh.Prepare_Field_For_Plotting("phi")

        assert output.function_space.mesh is bar_mesh.mesh
        assert np.any(np.isnan(output.values))


@pytest.fixture
def two_squared_mesh():
    from femcubes.mesh_generators.two_squared import Sample_Parameters as TwoSquaredSample

    return Mesh(TwoSquaredSample())


class TestInterfaceFields:
    def test_interface_fields_only_have_dofs_on_the_interface(self, two_squared_mesh):
        theta = two_squared_mesh.Add_Field("theta", "Coverage", "CG", 2, domain="interface")

        support = two_squared_mesh.Get_Field_Support("theta")
        assert support.is_interface and not support.is_full
        assert theta.function_space.mesh.topology.dim == 1
        num_facets = theta.function_space.mesh.topology.index_map(1).size_global
        assert num_facets == two_squared_mesh.mesh.comm.allreduce(
            np.count_nonzero(two_squared_mesh.get_boundary_facets("interface")
                             < two_squared_mesh.mesh.topology.index_map(1).size_local)
        )
        # P2 on a line: one DOF per vertex and per facet.
        assert theta.function_space.dofmap.index_map.size_global == 2 * num_facets + 1

    def test_interface_length_integrates_on_the_interface_mesh(self, two_squared_mesh):
        theta = two_squared_mesh.Add_Field("theta", "Coverage", "CG", 1, domain="interface")
        theta.x.array[:] = 1.0

        length = fem.assemble_scalar(fem.form(theta * ufl.Measure("dx", domain=theta.function_space.mesh)))

        assert np.isclose(length, two_squared_mesh.params.L)

    def test_fields_cannot_mix_cells_and_facets(self, two_squared_mesh):
        two_squared_mesh.Add_Field("c", "Concentration", "CG", 1, domain="domain:left")

        with pytest.raises(ValueError, match="both cells and facets"):
            two_squared_mesh.Add_Field("c", "Concentration", "CG", 1, domain="interface")

    def test_interface_fields_can_only_be_restricted_to_their_interface(self, two_squared_mesh):
        two_squared_mesh.Add_Field("theta", "Coverage", "CG", 1, domain="interface")

        two_squared_mesh.Constant_Field("theta", 0.5, domain="interface")
        np.testing.assert_allclose(two_squared_mesh.fields["theta"].x.array, 0.5)
        with pytest.raises(ValueError, match="only be restricted to its own interface"):
            two_squared_mesh.Zero_Field("theta", domain="domain:left")

    def test_interface_fields_are_plotted_on_their_interface_only(self, two_squared_mesh):
        theta = two_squared_mesh.Add_Field("theta", "Coverage", "CG", 1, domain="interface")
        theta.x.array[:] = 0.5

        plotted = two_squared_mesh.Prepare_Field_For_Plotting("theta")

        assert plotted.function_space.mesh is theta.function_space.mesh
        np.testing.assert_allclose(plotted.values, 0.5)
        assert two_squared_mesh.Prepare_Field_For_Output("theta") is None


    def test_interface_fields_are_written_to_hdf5_on_their_facet_mesh(self, two_squared_mesh, tmp_path):
        h5py = pytest.importorskip("h5py")
        mesh = two_squared_mesh
        mesh.Add_Field("c", "Concentration", "CG", 1, domain="domain:left").x.array[:] = 1.0
        mesh.Add_Field("theta", "Coverage", "CG", 2, domain="interface").interpolate(lambda x: x[1])
        mesh.output_dir, mesh.output_fields = str(tmp_path), ["c", "theta"]
        mesh.hdf5, mesh.output_freq = True, 1
        mesh.params.dt, mesh.params.Global_Measures = 1.0, {}

        mesh.Write_Outputs(step=1, time=1.0)

        num_facets = mesh.fields["theta"].function_space.mesh.topology.index_map(1).size_global
        with h5py.File(tmp_path / "Outputs_1.hdf5", "r") as h5f:
            assert list(h5f["Fields"]) == ["c"]
            theta = h5f["InterfaceFields"]["theta"]
            assert theta.shape == (num_facets, 3)
            group_name = theta.attrs["mesh_group"]
            values = theta[()]
        with h5py.File(tmp_path / "mesh_epoch_000.hdf5", "r") as h5f:
            assert group_name == "Interface_interface_DG2"
            np.testing.assert_allclose(h5f[group_name]["X"][()], mesh.params.L)
            np.testing.assert_allclose(values, h5f[group_name]["Y"][()], atol=1e-12)


class TestInterfaceVTKOutput:
    @pytest.mark.parametrize("single_file", [True, False])
    def test_interface_fields_are_written_to_their_own_vtk_series(self, two_squared_mesh, tmp_path, single_file):
        mesh = two_squared_mesh
        mesh.Add_Field("c", "Concentration", "CG", 1, domain="domain:left").x.array[:] = 1.0
        mesh.Add_Field("theta", "Coverage", "CG", 2, domain="interface").x.array[:] = 0.5
        mesh.output_dir, mesh.output_fields = str(tmp_path), ["c", "theta"]
        mesh.vtk, mesh.vtk_single_file, mesh.output_freq = True, single_file, 1
        mesh.params.dt, mesh.params.Global_Measures = 1.0, {}

        mesh.Write_Outputs(step=1, time=1.0)
        mesh.close_outputs()

        suffix = "" if single_file else "_step_1"
        interface_series = tmp_path / f"Outputs_interface_mesh_epoch_0000{suffix}.pvd"
        assert (tmp_path / f"Outputs_mesh_epoch_0000{suffix}.pvd").exists()
        assert interface_series.exists()
        pieces = list(tmp_path.glob(f"Outputs_interface_mesh_epoch_0000{suffix}*.vtu")) + list(
            tmp_path.glob(f"Outputs_interface_mesh_epoch_0000{suffix}/*.vtu"))
        assert pieces and any("theta" in piece.read_text() for piece in pieces)


class TestInterfaceIntegrals:
    def test_each_field_is_evaluated_from_its_own_side(self, two_squared_mesh):
        mesh = two_squared_mesh
        mesh.Add_Field("c", "c", "CG", 1, domain="domain:left").interpolate(lambda x: 1.0 + x[1])
        mesh.Add_Field("h", "h", "CG", 1, domain="domain:right").x.array[:] = 2.0
        mesh.Add_Field("theta", "theta", "CG", 1, domain="interface").x.array[:] = 3.0
        mesh.Add_Field("u", "u", "CG", 1).interpolate(lambda x: x[0])
        c, h, theta, u = (mesh.fields[name] for name in ("c", "h", "theta", "u"))

        value = fem.assemble_scalar(mesh.Compile_Form(mesh.Interface_Integral(c * h * theta * u, "interface")))

        L = mesh.params.L
        assert np.isclose(mesh.mesh.comm.allreduce(value), (L + 0.5 * L**2) * 6.0 * L)

    def test_normal_and_gradients_follow_the_normal_from_side(self, two_squared_mesh):
        mesh = two_squared_mesh
        c = mesh.Add_Field("c", "c", "CG", 1, domain="domain:left")
        h = mesh.Add_Field("h", "h", "CG", 1, domain="domain:right")
        c.interpolate(lambda x: x[0])
        h.interpolate(lambda x: -2.0 * x[0])
        n = ufl.FacetNormal(mesh.mesh)
        L, comm = mesh.params.L, mesh.mesh.comm

        def integrate(integrand, **kwargs):
            return comm.allreduce(fem.assemble_scalar(mesh.Compile_Form(mesh.Interface_Integral(integrand, "interface", **kwargs))))

        assert np.isclose(integrate(n[0]), L)
        assert np.isclose(integrate(n[0], normal_from="domain:right"), -L)
        assert np.isclose(integrate(ufl.dot(ufl.grad(c) - ufl.grad(h), n)), 3.0 * L)

    def test_block_matrix_couples_test_and_trial_functions_across_the_interface(self, two_squared_mesh):
        mesh = two_squared_mesh
        mesh.Add_Field("c", "c", "CG", 1, domain="domain:left")
        mesh.Add_Field("h", "h", "CG", 1, domain="domain:right")
        mesh.Add_TestFunction("c", step=0)
        mesh.Add_TestFunction("h", step=0)
        mesh.finalize_test_functions()
        dc, q = mesh.Get_Trial_Function("c", 0), mesh.Get_TestFunction("h", 0)

        blocks = mesh.Compile_Form(mesh.Interface_Integral(dc * q, "interface"))
        assert blocks[0][0] is None and blocks[1][1] is None
        matrix = assemble_matrix(blocks[1][0])
        matrix.assemble()

        # Partition of unity on both sides: the coupling block sums to the interface length.
        ones = matrix.createVecRight()
        ones.set(1.0)
        product = matrix.createVecLeft()
        matrix.mult(ones, product)
        assert np.isclose(product.sum(), mesh.params.L)
        matrix.destroy()

    def test_invalid_interfaces_and_fields_are_rejected(self, two_squared_mesh):
        mesh = two_squared_mesh
        one = fem.Constant(mesh.mesh, 1.0)
        top_field = mesh.Add_Field("t", "t", "CG", 1, domain="top")

        with pytest.raises(ValueError, match="interior facets only"):
            mesh.Interface_Integral(one, "top")
        with pytest.raises(ValueError, match="exactly one side"):
            mesh.Interface_Integral(one, "interface", normal_from="domain")
        with pytest.raises(ValueError, match="neither side"):
            mesh.Interface_Integral(top_field, "interface")


class TestInterfaceWeights:
    def test_entity_dof_points_handle_meshes_without_local_cells(self, two_squared_mesh):
        # Ranks without interface facets hold empty facet meshes.
        empty = dmesh.create_submesh(two_squared_mesh.mesh, 1, np.empty(0, dtype=np.int32))[0]
        space = fem.functionspace(empty, ("Lagrange", 2))

        assert two_squared_mesh._entity_dof_points(space).shape == (0, 3)

    @pytest.mark.parametrize("degree", [1, 2])
    def test_weights_sum_to_the_interface_length(self, two_squared_mesh, degree):
        for name, domain in (("c", "domain:left"), ("h", "domain:right"), ("theta", "interface")):
            two_squared_mesh.Add_Field(name, name, "CG", degree, domain=domain)
            two_squared_mesh.Add_TestFunction(name, step=0)
        two_squared_mesh.finalize_test_functions()

        weights, dofs, _vals = two_squared_mesh.Get_Interface_Weights_Dofs("interface", ["c", "h", "theta"], 0)

        total = two_squared_mesh.mesh.comm.allreduce(weights.sum())
        assert np.isclose(total, two_squared_mesh.params.L)
        assert all(dofs[name].size == weights.size for name in ("c", "h", "theta"))

    def test_fields_on_both_sides_and_the_interface_share_each_node(self, two_squared_mesh):
        # Linear fields equal to y on every side must agree at each matched node.
        fields = {}
        for name, domain in (("c", "domain:left"), ("h", "domain:right"), ("theta", "interface"), ("u", "domain")):
            fields[name] = two_squared_mesh.Add_Field(name, name, "CG", 1, domain=domain)
            fields[name].interpolate(lambda x: x[1])
        two_squared_mesh.Add_TestFunction("c", step=0)
        two_squared_mesh.finalize_test_functions()

        _weights, dofs, vals = two_squared_mesh.Get_Interface_Weights_Dofs("interface", ["c", "h", "theta", "u"], 0)

        np.testing.assert_allclose(vals["c"], vals["h"], atol=1e-12)
        np.testing.assert_allclose(vals["c"], vals["theta"], atol=1e-12)
        np.testing.assert_allclose(vals["c"], vals["u"], atol=1e-12)
        assert dofs["h"] is None  # not solved in step 0

    def test_interface_weights_require_one_scalar_cg_degree(self, two_squared_mesh):
        two_squared_mesh.Add_Field("c", "c", "CG", 1, domain="domain:left")
        two_squared_mesh.Add_Field("theta", "theta", "CG", 2, domain="interface")

        with pytest.raises(ValueError, match="scalar CG fields of one degree"):
            two_squared_mesh.Get_Interface_Weights_Dofs("interface", ["c", "theta"], 0)


class TestInterfaceBoundaryConditions:
    def test_bc_on_the_interface_itself_constrains_the_whole_interface_field(self, two_squared_mesh):
        theta = two_squared_mesh.Add_Field("theta", "Coverage", "CG", 2, domain="interface")
        theta.x.array[:] = 0.75

        bcs = two_squared_mesh.Build_Increment_Dirichlet_BCs("theta", "interface", 0.25)

        dofs, num_owned = bcs[0].dof_indices()
        assert num_owned == theta.function_space.dofmap.index_map.size_local
        np.testing.assert_allclose(bcs[0].g.x.array[dofs], 0.5)

    def test_bc_on_another_boundary_constrains_where_the_interface_meets_it(self, two_squared_mesh):
        theta = two_squared_mesh.Add_Field("theta", "Coverage", "CG", 2, domain="interface")

        bcs = two_squared_mesh.Build_Increment_Dirichlet_BCs("theta", "top", 0.0)

        dofs = bcs[0].dof_indices()[0]
        points = two_squared_mesh._entity_dof_points(theta.function_space)[dofs]
        assert dofs.size == 1
        np.testing.assert_allclose(points[0, :2], [two_squared_mesh.params.L, two_squared_mesh.params.L])


class TestRefinementOfRestrictedFields:
    def _refine(self, mesh, field, threshold, repartition=False):
        from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig, RefinementConfig

        config = RefinementConfig(
            enabled=True,
            indicator=FieldThresholdIndicatorConfig(field=field, threshold=threshold),
            target_h=1e-3,
            buffer_layers=0,
            repartition=repartition,
            repartition_imbalance_threshold=1.0,
        )
        return mesh.Refine_From_Indicator(config)

    def test_restricted_fields_are_transferred_onto_the_refined_submeshes(self, two_squared_mesh):
        mesh = two_squared_mesh
        profiles = {
            "c": ("domain:left", 2, lambda x: 1.0 + x[0] + x[1] ** 2),
            "h": ("domain:right", 1, lambda x: 2.0 - x[0]),
            "theta": ("interface", 2, lambda x: x[1] ** 2),
            "u": ("domain", 1, lambda x: x[0] + x[1]),
        }
        for name, (domain, degree, profile) in profiles.items():
            mesh.Add_Field(name, name, "CG", degree, domain=domain).interpolate(profile)
        mesh.Add_Quadrature_Field("q", "q", (), domain="domain:left").x.array[:] = 3.0
        for name in ("c", "theta", "h"):
            mesh.Add_TestFunction(name, step=0)
        mesh.finalize_test_functions()
        comm = mesh.mesh.comm
        interface_facets = mesh.fields["theta"].function_space.mesh.topology.index_map(1).size_global

        # Only the top of the left square, next to the interface, is marked.
        result = self._refine(mesh, "c", 2.6)

        assert result.installed
        for name, (domain, _degree, profile) in profiles.items():
            field = mesh.fields[name]
            assert mesh.Get_Field_Support(name).domain_names == (domain,)
            assert (field.function_space.mesh is mesh.mesh) == (domain == "domain")
            expected = fem.Function(field.function_space)
            expected.interpolate(profile)
            np.testing.assert_allclose(field.x.array, expected.x.array, atol=1e-12, err_msg=name)
        assert mesh.fields["theta"].function_space.mesh.topology.index_map(1).size_global > interface_facets
        q_cells = mesh.fields["q"].function_space.mesh.topology.index_map(2).size_global
        assert q_cells == comm.allreduce(np.count_nonzero(
            mesh.get_domain_cells("domain:left") < mesh.mesh.topology.index_map(2).size_local))
        np.testing.assert_allclose(mesh.fields["q"].x.array, 3.0)
        assert mesh.mixed_spaces[0]["field_names"] == ["c", "theta", "h"]
        assert mesh.Get_TestFunction("theta", 0).ufl_function_space().ufl_domain() is mesh.fields["theta"].function_space.mesh.ufl_domain()

    def test_repeated_refinement_with_repartitioning_keeps_interface_fields_exact(self, two_squared_mesh):
        mesh = two_squared_mesh
        mesh.Add_Field("c", "c", "CG", 1, domain="domain:left").interpolate(lambda x: x[0] + x[1])
        mesh.Add_Field("theta", "theta", "CG", 2, domain="interface").interpolate(lambda x: x[1] ** 2)

        for threshold in (1.8, 1.9):
            assert self._refine(mesh, "c", threshold, repartition=True).installed

        theta = mesh.fields["theta"]
        expected = fem.Function(theta.function_space)
        expected.interpolate(lambda x: x[1] ** 2)
        np.testing.assert_allclose(theta.x.array, expected.x.array, atol=1e-12)

    def test_restricted_indicators_only_mark_their_subdomain(self, two_squared_mesh):
        mesh = two_squared_mesh
        mesh.Add_Field("c", "c", "CG", 1, domain="domain:left").x.array[:] = -1.0
        right_cells = mesh.mesh.comm.allreduce(np.count_nonzero(
            mesh.get_domain_cells("domain:right") < mesh.mesh.topology.index_map(2).size_local))

        # Zero lies above the threshold, so cells outside the support would be marked.
        result = self._refine(mesh, "c", -2.0)

        assert result.installed
        assert mesh.mesh.comm.allreduce(np.count_nonzero(
            mesh.get_domain_cells("domain:right") < mesh.mesh.topology.index_map(2).size_local)) < 2 * right_cells
        np.testing.assert_allclose(mesh.fields["c"].x.array, -1.0)

    def test_no_op_refinement_keeps_the_restricted_fields(self, two_squared_mesh):
        mesh = two_squared_mesh
        c = mesh.Add_Field("c", "c", "CG", 1, domain="domain:left")

        result = self._refine(mesh, "c", 1.0)

        assert not result.installed
        assert mesh.fields["c"] is c

    def test_interface_and_global_indicators_are_rejected(self, two_squared_mesh):
        mesh = two_squared_mesh
        mesh.Add_Field("theta", "theta", "CG", 1, domain="interface")
        mesh.Add_Global_Field("E_m", "Metal potential")
        fields = mesh.fields

        for name in ("theta", "E_m"):
            with pytest.raises(ValueError, match="indicators must be fields on cells"):
                self._refine(mesh, name, 0.0)
        assert mesh.fields is fields

    @pytest.mark.parametrize("family, degree", [("CG", 3), ("DG", 1), ("DG", 2)])
    def test_interface_fields_of_any_degree_are_refined_exactly(self, two_squared_mesh, family, degree):
        mesh = two_squared_mesh
        profile = lambda x: 1.0 + x[1] ** degree  # noqa: E731 - exact in the field's space
        mesh.Add_Field("c", "c", "CG", 1, domain="domain:left").interpolate(lambda x: x[0] + x[1])
        mesh.Add_Field("theta", "theta", family, degree, domain="interface").interpolate(profile)

        assert self._refine(mesh, "c", 1.8).installed

        theta = mesh.fields["theta"]
        expected = fem.Function(theta.function_space)
        expected.interpolate(profile)
        np.testing.assert_allclose(theta.x.array, expected.x.array, atol=1e-10)

    def test_global_fields_keep_their_value_and_step(self, two_squared_mesh):
        mesh = two_squared_mesh
        mesh.Add_Field("c", "c", "CG", 1, domain="domain:left").interpolate(lambda x: x[0] + x[1])
        mesh.Add_Global_Field("E_m", "Metal potential").x.array[:] = -0.4
        for name in ("c", "E_m"):
            mesh.Add_TestFunction(name, step=0)
        mesh.finalize_test_functions()

        assert self._refine(mesh, "c", 1.8).installed

        assert mesh.Get_Global_Value("E_m") == -0.4
        assert mesh.mixed_spaces[0]["field_names"] == ["c", "E_m"]
        dof = mesh.Get_Global_Field_Dof("E_m", 0)
        assert 0 <= dof < mesh.get_global_solution_dof_count()
        assert len(set(mesh.mesh.comm.allgather(dof))) == 1

    def test_detached_installs_transfer_restricted_fields(self, two_squared_mesh):
        from femcubes.core.mesh.refinement import FieldThresholdIndicatorConfig
        from femcubes.core.mesh.refinement.indicators import evaluate_field_indicator, expand_refinement_buffer
        from femcubes.core.mesh.refinement.markers import MeshMarkerState, transfer_mesh_markers
        from femcubes.core.mesh.refinement.topology import refine_mesh_topology
        from femcubes.core.mesh.refinement.transfer import (
            snapshot_registered_fields, transfer_quadrature_fields, transfer_standard_fields)

        mesh = two_squared_mesh
        mesh.Add_Field("c", "c", "CG", 2, domain="domain:left").interpolate(lambda x: x[0] ** 2)
        mesh.Add_Field("theta", "theta", "CG", 1, domain="interface").interpolate(lambda x: x[1])
        snapshot = snapshot_registered_fields(mesh)
        indicator = evaluate_field_indicator(mesh, FieldThresholdIndicatorConfig(field="c", threshold=0.5), 1e-3)
        artifacts = refine_mesh_topology(mesh.mesh, expand_refinement_buffer(mesh, indicator, 0, target_h=1e-3))
        artifacts = transfer_mesh_markers(MeshMarkerState.from_mesh_wrapper(mesh), artifacts)
        state = transfer_quadrature_fields(snapshot, artifacts, transfer_standard_fields(snapshot, artifacts))

        assert mesh.Install_Refined_State(artifacts, state).installed

        for name, profile in (("c", lambda x: x[0] ** 2), ("theta", lambda x: x[1])):
            expected = fem.Function(mesh.fields[name].function_space)
            expected.interpolate(profile)
            np.testing.assert_allclose(mesh.fields[name].x.array, expected.x.array, atol=1e-12, err_msg=name)


class TestFieldBoundaryIntegrator:
    def test_boundary_terms_act_on_subdomain_fields(self, two_squared_mesh):
        mesh = two_squared_mesh
        c = mesh.Add_Field("c", "c", "CG", 1, domain="domain:left")
        c.x.array[:] = 2.0

        value = fem.assemble_scalar(mesh.Compile_Form(c * mesh.Get_Field_Boundary_Integrator("c", "left")))

        assert np.isclose(mesh.mesh.comm.allreduce(value), 2.0 * mesh.params.L)

    def test_boundaries_outside_the_support_are_rejected(self, two_squared_mesh):
        mesh = two_squared_mesh
        mesh.Add_Field("c", "c", "CG", 1, domain="domain:left")

        with pytest.raises(ValueError, match="cannot act on field 'c'"):
            mesh.Get_Field_Boundary_Integrator("c", "bottom")
