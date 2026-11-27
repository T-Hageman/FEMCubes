import numpy as np
import pytest
import ufl
from basix.ufl import element

from femcubes.core.utils import maths_utils


def _constant_2d_tensor():
    return np.array([[1.0, 0.25], [0.25, 2.0]])


def _constant_3d_tensor():
    return np.array(
        [
            [1.0, 0.1, 0.2],
            [0.1, 2.0, 0.3],
            [0.2, 0.3, 3.0],
        ]
    )


def _constant_six_vector():
    return np.array([[1.0], [2.0], [3.0], [0.4], [0.5], [0.6]])


def test_dim_validators_reject_invalid_dimension():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        maths_utils._validate_dim(4)

    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        maths_utils._validate_dim_nb.py_func(4)


def test_mac_plus_minus_with_scalars():
    assert maths_utils.macPlus(3.0) == 3.0
    assert maths_utils.macPlus(-2.0) == 0.0
    assert maths_utils.macMinus(-2.0) == -2.0
    assert maths_utils.macMinus(3.0) == 0.0


def test_dmac_plus_minus_returns_ufl_expr():
    from ufl.core.expr import Expr

    assert isinstance(maths_utils.dmacPlus(1.0), Expr)
    assert isinstance(maths_utils.dmacMinus(-1.0), Expr)


def test_nabla_s_shape():
    coord_element = element("Lagrange", "triangle", 1, shape=(2,))
    mesh = ufl.Mesh(coord_element)
    V = ufl.FunctionSpace(mesh, coord_element)
    u = ufl.TestFunction(V)
    expr = maths_utils.nabla_s(u)
    assert expr.ufl_shape == (2, 2)


def test_eig_2d_returns_expected():
    A = ufl.as_matrix(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)))
    eig1, eig2, eig3 = maths_utils.eig(A, 2)
    assert float(eig1) == 2.0
    assert float(eig2) == 1.0
    assert float(eig3) == 3.0


def test_eig_2d_remains_finite_with_large_isotropic_part():
    """Small deviatoric stresses must survive a large pressure-like shift."""
    isotropic_part = -1.0e5
    perturbation = np.array(
        [
            [1.0e-4, 0.0, 0.0],
            [0.0, 2.0e-4, 0.0],
            [0.0, 0.0, -3.0e-4],
        ]
    )
    tensor = isotropic_part * np.eye(3) + perturbation
    A = ufl.as_matrix(tuple(map(tuple, tensor)))

    actual = np.array([float(value) for value in maths_utils.eig(A, 2)])
    expected_in_plane = np.linalg.eigvalsh(tensor[:2, :2])[::-1]

    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[:2], expected_in_plane, rtol=1.0e-12)
    assert actual[2] == tensor[2, 2]


def test_eig_3d_not_implemented():
    A = ufl.as_matrix(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0)))
    with pytest.raises(NotImplementedError, match="3D eigenvalues"):
        maths_utils.eig(A, 3)


@pytest.mark.parametrize("formulation", ["AT2", "HO-AT2"])
def test_ft_to_Gc_recovers_homogeneous_at2_peak_stress(formulation):
    strength, youngs, ell = 2.0e5, 10.0e9, 1.0
    gc = maths_utils.ft_to_Gc(strength, youngs, ell, formulation)
    peak_strain = np.sqrt(gc / (3.0 * youngs * ell))
    strains = peak_strain * np.array([0.99, 1.0, 1.01])
    # Stationarity of (1-d)^2*E*eps^2/2 + Gc*d^2/(2*ell).
    damage = youngs * strains**2 / (youngs * strains**2 + gc / ell)
    stresses = (1.0-damage)**2 * youngs * strains
    assert stresses[1] == pytest.approx(strength)
    assert stresses[0] < stresses[1] and stresses[2] < stresses[1]


def test_ft_to_Gc():
    f_t = 2.0e5
    E = 10.0e9
    ell = 1.0
    Gc_AT2 = maths_utils.ft_to_Gc(f_t, E, ell, PF_formulation="AT2")
    Gc_AT1 = maths_utils.ft_to_Gc(f_t, E, ell, PF_formulation="AT1")
    Gc_HO_AT2 = maths_utils.ft_to_Gc(f_t, E, ell, PF_formulation="HO-AT2")
    Gc_HO_AT1 = maths_utils.ft_to_Gc(f_t, E, ell, PF_formulation="HO-AT1")
    expected_AT2 = f_t**2 * ell / E * 256.0 / 27.0
    expected_AT1 = f_t**2 * ell / E * 8.0 / 3.0
    assert abs(Gc_AT2 - expected_AT2) < 1e-6
    assert abs(Gc_AT1 - expected_AT1) < 1e-6
    assert abs(Gc_HO_AT2 - expected_AT2) < 1e-6
    assert abs(Gc_HO_AT1 - expected_AT1) < 1e-6

    with pytest.raises(ValueError, match="Unknown Phase Field formulation"):
        maths_utils.ft_to_Gc(f_t, E, ell, PF_formulation="Invalid")


def test_stiffness_matrices():
    lame = 1.0e9
    shear = 0.5e9

    voigt = maths_utils.StiffnessMatrix(lame, shear)
    mandel = maths_utils.StiffnessMatrixMandel(lame, shear)

    assert voigt.shape == (6, 6)
    assert mandel.shape == (6, 6)
    assert np.isclose(voigt[0, 0], lame + 2.0 * shear)
    assert np.isclose(voigt[0, 1], lame)
    assert np.isclose(voigt[3, 3], shear)
    assert np.isclose(mandel[3, 3], 2.0 * shear)
    assert np.allclose(np.diag(voigt)[:3], np.diag(mandel)[:3])


def test_projection_and_invariant_helpers():
    j2_strain = maths_utils.J2Matrix("strain")
    j2_stress = maths_utils.J2Matrix("Stress")
    dev = maths_utils.DevMat()
    i1 = maths_utils.I1Vec()
    corr = maths_utils.VoigtCorrectMat()

    assert j2_strain.shape == (6, 6)
    assert j2_stress.shape == (6, 6)
    assert dev.shape == (6, 6)
    assert i1.shape == (6, 1)
    assert corr.shape == (6, 6)

    assert np.isclose(j2_strain[3, 3], 0.5)
    assert np.isclose(j2_stress[3, 3], 2.0)
    assert np.allclose(i1[:, 0], np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(np.diag(corr), np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))

    with pytest.raises(ValueError, match="Unknown stress_or_strain type"):
        maths_utils.J2Matrix("invalid_type")


@pytest.mark.parametrize(
    ("tensor", "dim"),
    [
        (_constant_2d_tensor(), 2),
        (_constant_3d_tensor(), 3),
    ],
)
def test_tensor_to_mandel_and_back(tensor, dim):
    mandel = maths_utils.Tensor_To_Mandel(tensor, dim)
    reconstructed = maths_utils.Mandel_To_Tensor(mandel, dim)

    assert mandel.shape == (6, 1)
    assert reconstructed.shape == (dim, dim)
    assert np.allclose(tensor, reconstructed, atol=1e-8)


@pytest.mark.parametrize(
    ("tensor", "dim"),
    [
        (_constant_2d_tensor(), 2),
        (_constant_3d_tensor(), 3),
    ],
)
def test_tensor_to_mandel_and_back_python_impl(tensor, dim):
    mandel = maths_utils.Tensor_To_Mandel.py_func(tensor, dim)
    reconstructed = maths_utils.Mandel_To_Tensor.py_func(mandel, dim)

    assert mandel.shape == (6, 1)
    assert reconstructed.shape == (dim, dim)
    assert np.allclose(tensor, reconstructed, atol=1e-8)


def test_tensor_to_mandel_ufl_round_trip_2d():
    tensor = ufl.as_tensor(((1.0, 0.25), (0.25, 2.0)))
    mandel = maths_utils.Tensor_To_Mandel_UFL(tensor, 2)
    reconstructed = maths_utils.Mandel_To_Tensor_UFL(mandel, 2)

    assert mandel.ufl_shape == (6,)
    assert reconstructed.ufl_shape == (2, 2)
    assert float(reconstructed[0, 0]) == 1.0
    assert float(reconstructed[0, 1]) == 0.25
    assert float(reconstructed[1, 1]) == 2.0


def test_tensor_to_mandel_ufl_round_trip_3d():
    tensor = ufl.as_tensor(
        (
            (1.0, 0.1, 0.2),
            (0.1, 2.0, 0.3),
            (0.2, 0.3, 3.0),
        )
    )
    mandel = maths_utils.Tensor_To_Mandel_UFL(tensor, 3)
    reconstructed = maths_utils.Mandel_To_Tensor_UFL(mandel, 3)

    assert mandel.ufl_shape == (6,)
    assert reconstructed.ufl_shape == (3, 3)
    assert np.isclose(float(reconstructed[0, 2]), 0.2)
    assert np.isclose(float(reconstructed[1, 2]), 0.3)
    assert np.isclose(float(reconstructed[2, 2]), 3.0)


@pytest.mark.parametrize(
    ("tensor", "dim"),
    [
        (_constant_2d_tensor(), 2),
        (_constant_3d_tensor(), 3),
    ],
)
def test_tensor_to_voigt_strain_and_back(tensor, dim):
    voigt = maths_utils.Tensor_To_Voigt_Strain(tensor, dim)
    reconstructed = maths_utils.Voigt_To_Tensor_Strain(voigt, dim)

    assert voigt.shape == (6, 1)
    assert reconstructed.shape == (dim, dim)
    assert np.allclose(tensor, reconstructed, atol=1e-8)


@pytest.mark.parametrize(
    ("tensor", "dim"),
    [
        (_constant_2d_tensor(), 2),
        (_constant_3d_tensor(), 3),
    ],
)
def test_tensor_to_voigt_stress_and_back(tensor, dim):
    voigt = maths_utils.Tensor_To_Voigt_Stress(tensor, dim)
    reconstructed = maths_utils.Voigt_To_Tensor_Stress(voigt, dim)

    assert voigt.shape == (6, 1)
    assert reconstructed.shape == (dim, dim)
    assert np.allclose(tensor, reconstructed, atol=1e-8)


def test_voigt_strain_ufl_round_trip_2d():
    tensor = ufl.as_tensor(((1.0, 0.25), (0.25, 2.0)))
    voigt = maths_utils.Tensor_To_Voigt_Strain_UFL(tensor, 2)
    reconstructed = maths_utils.Voigt_To_Tensor_Strain_UFL(voigt, 2)

    assert voigt.ufl_shape == (6,)
    assert reconstructed.ufl_shape == (2, 2)
    assert float(reconstructed[0, 0]) == 1.0
    assert float(reconstructed[0, 1]) == 0.25
    assert float(reconstructed[1, 1]) == 2.0


def test_voigt_strain_ufl_round_trip_3d():
    tensor = ufl.as_tensor(
        (
            (1.0, 0.1, 0.2),
            (0.1, 2.0, 0.3),
            (0.2, 0.3, 3.0),
        )
    )
    voigt = maths_utils.Tensor_To_Voigt_Strain_UFL(tensor, 3)
    reconstructed = maths_utils.Voigt_To_Tensor_Strain_UFL(voigt, 3)

    assert voigt.ufl_shape == (6,)
    assert reconstructed.ufl_shape == (3, 3)
    assert np.isclose(float(reconstructed[0, 1]), 0.1)
    assert np.isclose(float(reconstructed[0, 2]), 0.2)
    assert np.isclose(float(reconstructed[1, 2]), 0.3)
    assert np.isclose(float(reconstructed[2, 2]), 3.0)


def test_voigt_stress_ufl_round_trip_3d():
    tensor = ufl.as_tensor(
        (
            (1.0, 0.1, 0.2),
            (0.1, 2.0, 0.3),
            (0.2, 0.3, 3.0),
        )
    )
    voigt = maths_utils.Tensor_To_Voigt_Stress_UFL(tensor, 3)
    reconstructed = maths_utils.Voigt_To_Tensor_Stress_UFL(voigt, 3)

    assert voigt.ufl_shape == (6,)
    assert reconstructed.ufl_shape == (3, 3)
    assert np.isclose(float(reconstructed[0, 1]), 0.1)
    assert np.isclose(float(reconstructed[0, 2]), 0.2)
    assert np.isclose(float(reconstructed[2, 2]), 3.0)


def test_voigt_stress_ufl_round_trip_2d():
    tensor = ufl.as_tensor(((1.0, 0.25), (0.25, 2.0)))
    voigt = maths_utils.Tensor_To_Voigt_Stress_UFL(tensor, 2)
    reconstructed = maths_utils.Voigt_To_Tensor_Stress_UFL(voigt, 2)

    assert voigt.ufl_shape == (6,)
    assert reconstructed.ufl_shape == (2, 2)
    assert float(reconstructed[0, 0]) == 1.0
    assert float(reconstructed[0, 1]) == 0.25
    assert float(reconstructed[1, 1]) == 2.0


@pytest.mark.parametrize(
    ("tensor", "dim"),
    [
        (_constant_2d_tensor(), 2),
        (_constant_3d_tensor(), 3),
    ],
)
def test_tensor_to_voigt_strain_and_back_python_impl(tensor, dim):
    voigt = maths_utils.Tensor_To_Voigt_Strain.py_func(tensor, dim)
    reconstructed = maths_utils.Voigt_To_Tensor_Strain.py_func(voigt, dim)

    assert voigt.shape == (6, 1)
    assert reconstructed.shape == (dim, dim)
    assert np.allclose(tensor, reconstructed, atol=1e-8)


@pytest.mark.parametrize(
    ("tensor", "dim"),
    [
        (_constant_2d_tensor(), 2),
        (_constant_3d_tensor(), 3),
    ],
)
def test_tensor_to_voigt_stress_and_back_python_impl(tensor, dim):
    voigt = maths_utils.Tensor_To_Voigt_Stress.py_func(tensor, dim)
    reconstructed = maths_utils.Voigt_To_Tensor_Stress.py_func(voigt, dim)

    assert voigt.shape == (6, 1)
    assert reconstructed.shape == (dim, dim)
    assert np.allclose(tensor, reconstructed, atol=1e-8)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1.5, -1.0),
        (0.0, 0.0),
        (2.5, 1.0),
    ],
)
def test_sign_nb(value, expected):
    assert maths_utils.sign_nb(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1.5, -1.0),
        (0.0, 0.0),
        (2.5, 1.0),
    ],
)
def test_sign_nb_python_impl(value, expected):
    assert maths_utils.sign_nb.py_func(value) == expected


@pytest.mark.parametrize(
    ("function", "arg"),
    [
        (maths_utils.Tensor_To_Mandel, _constant_3d_tensor()),
        (maths_utils.Mandel_To_Tensor, _constant_six_vector()),
        (maths_utils.Tensor_To_Voigt_Strain, _constant_3d_tensor()),
        (maths_utils.Voigt_To_Tensor_Strain, _constant_six_vector()),
        (maths_utils.Tensor_To_Voigt_Stress, _constant_3d_tensor()),
        (maths_utils.Voigt_To_Tensor_Stress, _constant_six_vector()),
    ],
)
def test_numpy_tensor_conversions_reject_invalid_dim_python_impl(function, arg):
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        function.py_func(arg, 4)


def test_numba_tensor_conversion_rejects_invalid_dim_runtime():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        maths_utils.Tensor_To_Mandel(_constant_3d_tensor(), 4)


@pytest.mark.parametrize(
    ("function", "arg"),
    [
        (maths_utils.Tensor_To_Mandel_UFL, ufl.as_tensor(((1.0, 0.25), (0.25, 2.0)))),
        (maths_utils.Mandel_To_Tensor_UFL, ufl.as_vector((1.0, 2.0, 0.0, 0.0, 0.0, 0.5))),
        (maths_utils.Tensor_To_Voigt_Strain_UFL, ufl.as_tensor(((1.0, 0.25), (0.25, 2.0)))),
        (maths_utils.Voigt_To_Tensor_Strain_UFL, ufl.as_vector((1.0, 2.0, 0.0, 0.0, 0.0, 0.5))),
        (maths_utils.Tensor_To_Voigt_Stress_UFL, ufl.as_tensor(((1.0, 0.25), (0.25, 2.0)))),
        (maths_utils.Voigt_To_Tensor_Stress_UFL, ufl.as_vector((1.0, 2.0, 0.0, 0.0, 0.0, 0.5))),
    ],
)
def test_ufl_tensor_conversions_reject_invalid_dim(function, arg):
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        function(arg, 4)
