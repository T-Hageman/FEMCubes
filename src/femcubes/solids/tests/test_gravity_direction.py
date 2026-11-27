"""Gravity acts along the last spatial coordinate in solid weak forms."""

import numpy as np
import pytest
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
from petsc4py import PETSc

from femcubes.core.mesh.mesh import Mesh
from femcubes.core.physics.physics import Physics
from femcubes.solids.tests.test_split_viscoelastic import _build_params


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("kind", [
    "LinearElastic", "ViscoElastic", "SplitLinearElastic", "SplitViscoElastic",
    "Cohesive_LinearElastic", "Cohesive_ViscoElastic", "Cohesive_ViscoElastic_Kernel", "Stokes",
])
def test_gravity_force_acts_only_in_last_coordinate(kind, dim):
    params = _build_params("Stress-Based")
    params.dim = dim
    params.dx = 1.0
    if dim == 3:
        params.Lz = 1.0
    params.gravity = -2.0
    params.Materials["Ice"].update(Density=3.0, Inertia=False, f_s=2.0)
    params.Models["Solid"]["Type"] = "Solid:" + kind
    params.Models["Solid"].pop("CreepCoupling")
    params.ModelNames = ["Solid"]
    field_name = "v" if kind == "Stokes" else "u"
    params.Solution_Steps = {field_name: 0}
    if kind == "Stokes":
        params.Solution_Steps["p"] = 0
        params.Field_Orders["v"] = 2
    mesh = Mesh(params)
    mesh.Add_Field("phasefield", "Prescribed damage", "CG", 1)
    mesh.Add_Field("phasefieldOld", "Previous damage", "CG", 1)
    physics = Physics(mesh, params)
    physics.Initialize_Fields()
    _, residual = physics.models["Solid"].assemble_KF(0)

    # At zero displacement, velocity and pressure, -residual is the external
    # force vector. Check every DOF, including horizontal and pressure DOFs.
    test = mesh.Get_TestFunction(field_name, 0)
    vertical_axis = 1 if dim == 2 else 2
    expected_load = 3.0 * params.gravity * test[vertical_axis] * ufl.dx
    # Monolithic (Stokes) steps are compared block by block; blocks without
    # an expected load (pressure) must have zero external force.
    if len(mesh.step_fields[0]) > 1:
        force_forms = ufl.extract_blocks(-residual)
        expected_forms = ufl.extract_blocks(expected_load)
    else:
        force_forms, expected_forms = [-residual], [expected_load]
    total_force_norm = 0.0
    for force_form, expected_form in zip(force_forms, expected_forms):
        force = assemble_vector(fem.form(force_form))
        expected = assemble_vector(fem.form(expected_form)) if expected_form is not None else force.duplicate()
        if expected_form is None:
            expected.set(0.0)
        try:
            for vector in (force, expected):
                vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            total_force_norm += force.norm()
            np.testing.assert_allclose(force.array, expected.array, rtol=1e-12, atol=1e-12)
        finally:
            force.destroy()
            expected.destroy()
    assert total_force_norm > 0.0
