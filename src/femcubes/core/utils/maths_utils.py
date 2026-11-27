"""Mathematical helper functions used across FEMCubes.

The module mixes small UFL helpers used during variational-form
construction with NumPy/Numba kernels used for constitutive-model
updates.  Public helper names use the conventional ``Voigt`` spelling.
"""

import numba as nb
import numpy as np
from ufl import as_tensor, as_vector, conditional, gt, lt, nabla_grad, sqrt, sym


def _validate_dim(dim):
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3.")


@nb.njit(cache=True)
def _validate_dim_nb(dim):
    if dim != 2 and dim != 3:
        raise ValueError("dim must be 2 or 3.")


def nabla_s(field):
    """Symmetric gradient operator.

    Returns ``sym(nabla_grad(field))``, i.e. the symmetric part of the
    displacement gradient (the infinitesimal strain tensor).

    Parameters
    ----------
    field : ufl.Coefficient
        A UFL vector field (e.g. displacement).

    Returns
    -------
    ufl.Form
    """
    return sym(nabla_grad(field))
    
def macPlus(a):
    """Macaulay bracket (positive part): :math:`\\langle a \\rangle_+ = (a + |a|)/2`.

    Parameters
    ----------
    a : ufl expression or scalar
    """
    return (a+abs(a))/2
    #return conditional(gt(a, 0.0), a, 0.0)

def dmacPlus(a):
    """Derivative of the positive Macaulay bracket: 1 if *a* > 0, else 0."""
    return conditional(gt(a, 0.0), 1.0, 0.0)

def macMinus(a):
    """Macaulay bracket (negative part): :math:`\\langle a \\rangle_- = (a - |a|)/2`.

    Parameters
    ----------
    a : ufl expression or scalar
    """
    return (a - abs(a))/2
    #return conditional(lt(a, 0.0), a, 0.0)

def dmacMinus(a):
    """Derivative of the negative Macaulay bracket: 1 if *a* < 0, else 0."""
    return conditional(lt(a, 0.0), 1.0, 0.0)

def eig(A, dim):
    """Compute principal values of a symmetric tensor with a 2-D in-plane block.

    The tensor may represent stress, strain, or another symmetric quantity.
    The in-plane discriminant is evaluated as a sum of squares so that a
    large isotropic part, such as fluid pressure, does not cause cancellation
    when the deviatoric part is small.

    Parameters
    ----------
    A : ufl tensor
        Symmetric tensor (stress or strain).
    dim : int
        Spatial dimension (currently only 2 is supported).

    Returns
    -------
    tuple
        ``(eig1, eig2, eig3)`` -- the three eigenvalues.
    """
    if dim == 2:
        s11, s22, s12 = A[0,0], A[1,1], A[0,1]
        s33 = A[2,2]  # Out-of-plane stress (non-zero in plane strain)
        
        # In-plane eigenvalues
        trace_2d = s11 + s22
        discriminant = sqrt(((s11 - s22) / 2.0)**2 + s12**2)
        
        eig1 = trace_2d/2 + discriminant
        eig2 = trace_2d/2 - discriminant
        eig3 = s33  # Out-of-plane eigenvalue
        return eig1, eig2, eig3
    else:
        raise NotImplementedError("3D eigenvalues not yet implemented")
    
def ft_to_Gc(f_t, E, ell, PF_formulation="AT2"):
    """Convert tensile strength to critical energy release rate.

    Parameters
    ----------
    f_t : float
        Tensile strength (Pa).
    E : float
        Young's modulus (Pa).
    ell : float
        Phase-field length-scale parameter (m).
    PF_formulation : {"AT2", "AT1", "HO-AT2", "HO-AT1"}, optional
        Phase-field regularisation type (default ``"AT2"``).

    Returns
    -------
    float
        Fracture toughness :math:`G_c` (J/m²).

    Notes
    -----
    In phase-field models, the stress at which a homogeneous bar fails
    scales as :math:`\\sqrt{EG_c/\\ell}`, so that the length scale and the
    fracture toughness together set the tensile strength
    :cite:p:`Pham2011,Tanne2018`. This function inverts that relation, using
    the homogeneous uniaxial calibration with quadratic stiffness
    degradation and zero residual stiffness. AT2/HO-AT2 use the peak-stress
    relation ``Gc = (256/27)*f_t**2*ell/E`` for crack density
    ``phi**2/(2*ell)``. AT1/HO-AT1 use the initial damage threshold,
    ``Gc = (8/3)*f_t**2*ell/E``. A different elastic split or constraint on
    lateral strain can require a different strength calibration.
    """
    if PF_formulation in ("AT2", "HO-AT2"):
        Gc = f_t**2*ell / E * 256.0/27.0
    elif PF_formulation in ("AT1", "HO-AT1"):
        Gc = f_t**2*ell / E * 8.0/3.0
    else:
        raise ValueError(f"Unknown Phase Field formulation: {PF_formulation}")
    return Gc

def StiffnessMatrix(Lame, Shear):
    """Build the 6x6 isotropic elastic stiffness matrix (Voigt notation).

    Parameters
    ----------
    Lame : float
        First Lamé parameter :math:`\\lambda`.
    Shear : float
        Shear modulus :math:`\\mu`.

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 6)``.
    """
    D_el = np.zeros((6,6))
    D_el[0,0] = Lame + 2.0*Shear
    D_el[0,1] = Lame
    D_el[0,2] = Lame
    D_el[1,0] = Lame
    D_el[1,1] = Lame + 2.0*Shear
    D_el[1,2] = Lame
    D_el[2,0] = Lame
    D_el[2,1] = Lame
    D_el[2,2] = Lame + 2.0*Shear
    D_el[3,3] = Shear
    D_el[4,4] = Shear
    D_el[5,5] = Shear
    return D_el

def StiffnessMatrixMandel(Lame, Shear):
    """Build the 6x6 isotropic elastic stiffness matrix (Mandel notation).

    Parameters
    ----------
    Lame : float
        First Lamé parameter :math:`\\lambda`.
    Shear : float
        Shear modulus :math:`\\mu`.

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 6)``.
    """
    D_el = np.zeros((6,6))
    D_el[0,0] = Lame + 2.0*Shear
    D_el[0,1] = Lame
    D_el[0,2] = Lame
    D_el[1,0] = Lame
    D_el[1,1] = Lame + 2.0*Shear
    D_el[1,2] = Lame
    D_el[2,0] = Lame
    D_el[2,1] = Lame
    D_el[2,2] = Lame + 2.0*Shear
    D_el[3,3] = 2.0*Shear
    D_el[4,4] = 2.0*Shear
    D_el[5,5] = 2.0*Shear
    return D_el

def J2Matrix(stress_or_strain: str):
    """Return the 6x6 J2 (deviatoric) projection matrix in Voigt notation.

    Parameters
    ----------
    stress_or_strain : {"stress", "strain"}
        Determines the shear-component scaling.

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 6)``.
    """
    J2Mat = np.zeros((6,6))
    if stress_or_strain == "strain" or stress_or_strain == "Strain":
        J2Mat = np.zeros((6,6))
        J2Mat[0,0] = 2.0/3.0
        J2Mat[0,1] = -1.0/3.0
        J2Mat[0,2] = -1.0/3.0
        J2Mat[1,0] = -1.0/3.0
        J2Mat[1,1] = 2.0/3.0
        J2Mat[1,2] = -1.0/3.0
        J2Mat[2,0] = -1.0/3.0
        J2Mat[2,1] = -1.0/3.0
        J2Mat[2,2] = 2.0/3.0
        J2Mat[3,3] = 1.0/2.0
        J2Mat[4,4] = 1.0/2.0
        J2Mat[5,5] = 1.0/2.0
    elif stress_or_strain == "stress" or stress_or_strain == "Stress":
        J2Mat = np.zeros((6,6))
        J2Mat[0,0] = 2.0/3.0
        J2Mat[0,1] = -1.0/3.0
        J2Mat[0,2] = -1.0/3.0
        J2Mat[1,0] = -1.0/3.0
        J2Mat[1,1] = 2.0/3.0
        J2Mat[1,2] = -1.0/3.0
        J2Mat[2,0] = -1.0/3.0
        J2Mat[2,1] = -1.0/3.0
        J2Mat[2,2] = 2.0/3.0
        J2Mat[3,3] = 2.0
        J2Mat[4,4] = 2.0
        J2Mat[5,5] = 2.0
    else:
        raise ValueError(f"Unknown stress_or_strain type: {stress_or_strain}")
    
    return J2Mat

def DevMat():
    """Return the 6x6 deviatoric projection matrix.

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 6)``.
    """
    Dev = np.zeros((6,6))
    Dev[0,0] = 2.0/3.0
    Dev[0,1] = -1.0/3.0
    Dev[0,2] = -1.0/3.0
    Dev[1,0] = -1.0/3.0
    Dev[1,1] = 2.0/3.0
    Dev[1,2] = -1.0/3.0
    Dev[2,0] = -1.0/3.0
    Dev[2,1] = -1.0/3.0
    Dev[2,2] = 2.0/3.0
    Dev[3,3] = 1.0
    Dev[4,4] = 1.0
    Dev[5,5] = 1.0
    return Dev

def I1Vec():
    """Return the 6x1 first-invariant selector in Voigt notation.

    The first invariant is computed as :math:`I_1 = I1Vec^T \\cdot \\sigma_{voigt}`.

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 1)``.
    """
    I1Vec = np.zeros((6,1))
    I1Vec[0,0] = 1.0
    I1Vec[1,0] = 1.0
    I1Vec[2,0] = 1.0
    return I1Vec

def VoigtCorrectMat():
    """Return the 6x6 Voigt correction matrix for strain double-contractions.

    Diagonal entries are 1 for normal components and 0.5 for shear
    components, accounting for the factor of 2 in engineering shear
    strains.

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 6)``.
    """
    # used for contractions of strains:strains
    C = np.zeros((6,6))
    C[0,0] = 1.0
    C[1,1] = 1.0
    C[2,2] = 1.0
    C[3,3] = 0.5
    C[4,4] = 0.5
    C[5,5] = 0.5
    return C

@nb.njit(cache=True)
def Tensor_To_Mandel(tensor, dim):
    """Convert a symmetric tensor to Mandel notation (NumPy, JIT-compiled).

    Parameters
    ----------
    tensor : numpy.ndarray
        Shape ``(dim, dim)`` symmetric tensor.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 1)`` Mandel vector.
    """
    _validate_dim_nb(dim)
    mandel = np.zeros((6,1))
    mandel[0] = tensor[0,0]
    mandel[1] = tensor[1,1]
    if dim == 3:
        mandel[2] = tensor[2,2]
        mandel[3] = np.sqrt(2.0) * tensor[1,2]
        mandel[4] = np.sqrt(2.0) * tensor[0,2]
    mandel[5] = np.sqrt(2.0) * tensor[0,1]
    return mandel

@nb.njit(cache=True)
def Mandel_To_Tensor(mandel, dim):
    """Convert a Mandel vector back to a symmetric tensor (NumPy, JIT-compiled).

    Parameters
    ----------
    mandel : numpy.ndarray
        Shape ``(6, 1)`` Mandel vector.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    numpy.ndarray
        Shape ``(dim, dim)`` symmetric tensor.
    """
    _validate_dim_nb(dim)
    tensor = np.zeros((dim,dim))
    tensor[0,0] = mandel[0,0]
    tensor[1,1] = mandel[1,0]
    if dim == 3:
        tensor[2,2] = mandel[2,0]
        tensor[1,2] = mandel[3,0] / np.sqrt(2.0)
        tensor[2,1] = mandel[3,0] / np.sqrt(2.0)
        tensor[0,2] = mandel[4,0] / np.sqrt(2.0)
        tensor[2,0] = mandel[4,0] / np.sqrt(2.0)
    tensor[0,1] = mandel[5,0] / np.sqrt(2.0)
    tensor[1,0] = mandel[5,0] / np.sqrt(2.0)
    return tensor

def Tensor_To_Mandel_UFL(tensor, dim):
    """Convert a symmetric UFL tensor to Mandel notation.

    Parameters
    ----------
    tensor : ufl tensor
        ``(dim, dim)`` symmetric tensor expression.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    ufl.as_vector
        Length-6 Mandel vector.
    """
    _validate_dim(dim)
    if dim == 3:
        mandel = [
            tensor[0, 0],
            tensor[1, 1],
            tensor[2, 2],
            sqrt(2.0) * tensor[1, 2],
            sqrt(2.0) * tensor[0, 2],
            sqrt(2.0) * tensor[0, 1],
        ]
    else:
        mandel = [
            tensor[0, 0],
            tensor[1, 1],
            0.0,
            0.0,
            0.0,
            sqrt(2.0) * tensor[0, 1],
        ]
    return as_vector(mandel)

def Mandel_To_Tensor_UFL(mandel, dim):
    """Convert a Mandel vector to a symmetric UFL tensor.

    Parameters
    ----------
    mandel : ufl vector
        Length-6 Mandel vector expression.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    ufl.as_tensor
        ``(dim, dim)`` symmetric tensor.
    """
    _validate_dim(dim)
    if dim == 3:
        tensor = [
            [mandel[0]      , mandel[5] / sqrt(2.0), mandel[4] / sqrt(2.0)],
            [mandel[5] / sqrt(2.0), mandel[1]      , mandel[3] / sqrt(2.0)],
            [mandel[4] / sqrt(2.0), mandel[3] / sqrt(2.0), mandel[2]      ],
        ]
    else:
        tensor = [
            [mandel[0]      , mandel[5] / sqrt(2.0)],
            [mandel[5] / sqrt(2.0), mandel[1]      ],
        ]
    return as_tensor(tensor)

@nb.njit(cache=True)
def Tensor_To_Voigt_Strain(strain_tensor, dim):
    """Convert a strain tensor to Voigt notation (NumPy, JIT-compiled).

    Engineering shear strains (factor of 2) are applied.

    Parameters
    ----------
    strain_tensor : numpy.ndarray
        Shape ``(dim, dim)``.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 1)``.
    """
    _validate_dim_nb(dim)
    strain_voigt = np.zeros((6,1))
    strain_voigt[0] = strain_tensor[0,0]
    strain_voigt[1] = strain_tensor[1,1]
    if dim == 3:
        strain_voigt[2] = strain_tensor[2,2]
        strain_voigt[3] = 2.0*strain_tensor[1,2]
        strain_voigt[4] = 2.0*strain_tensor[0,2]
    strain_voigt[5] = 2.0*strain_tensor[0,1]
    return strain_voigt

@nb.njit(cache=True)
def Voigt_To_Tensor_Strain(strain_voigt, dim):
    """Convert a Voigt strain vector back to a tensor (NumPy, JIT-compiled).

    Parameters
    ----------
    strain_voigt : numpy.ndarray
        Shape ``(6, 1)``.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    numpy.ndarray
        Shape ``(dim, dim)``.
    """
    _validate_dim_nb(dim)
    strain_tensor = np.zeros((dim,dim))
    strain_tensor[0,0] = strain_voigt[0,0]
    strain_tensor[1,1] = strain_voigt[1,0]
    if dim == 3:
        strain_tensor[2,2] = strain_voigt[2,0]
        strain_tensor[1,2] = strain_voigt[3,0] / 2.0
        strain_tensor[2,1] = strain_voigt[3,0] / 2.0
        strain_tensor[0,2] = strain_voigt[4,0] / 2.0
        strain_tensor[2,0] = strain_voigt[4,0] / 2.0
    strain_tensor[0,1] = strain_voigt[5,0] / 2.0
    strain_tensor[1,0] = strain_voigt[5,0] / 2.0
    return strain_tensor

def Tensor_To_Voigt_Strain_UFL(strain_tensor, dim):
    """Convert a UFL strain tensor to Voigt notation.

    Parameters
    ----------
    strain_tensor : ufl tensor
        ``(dim, dim)`` strain expression.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    ufl.as_vector
        Length-6 Voigt vector.
    """
    _validate_dim(dim)
    if dim == 3:
        strain_voigt = [
            strain_tensor[0, 0],
            strain_tensor[1, 1],
            strain_tensor[2, 2],
            2.0 * strain_tensor[1, 2],
            2.0 * strain_tensor[0, 2],
            2.0 * strain_tensor[0, 1],
        ]
    else:
        strain_voigt = [
            strain_tensor[0, 0],
            strain_tensor[1, 1],
            0.0,
            0.0,
            0.0,
            2.0 * strain_tensor[0, 1],
        ]
    return as_vector(strain_voigt)

def Voigt_To_Tensor_Strain_UFL(strain_voigt, dim):
    """Convert a Voigt strain vector to a symmetric UFL tensor.

    Parameters
    ----------
    strain_voigt : ufl vector
        Length-6 Voigt strain expression.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    ufl.as_tensor
        ``(dim, dim)`` symmetric tensor.
    """
    _validate_dim(dim)
    if dim == 3:
        strain_tensor = [
            [strain_voigt[0]      , strain_voigt[5] / 2.0, strain_voigt[4] / 2.0],
            [strain_voigt[5] / 2.0, strain_voigt[1]      , strain_voigt[3] / 2.0],
            [strain_voigt[4] / 2.0, strain_voigt[3] / 2.0, strain_voigt[2]      ],
        ]
    else:
        strain_tensor = [
            [strain_voigt[0]      , strain_voigt[5] / 2.0],
            [strain_voigt[5] / 2.0, strain_voigt[1]      ],
        ]
    return as_tensor(strain_tensor)

@nb.njit(cache=True)
def Tensor_To_Voigt_Stress(stress_tensor, dim):
    """Convert a stress tensor to legacy Voigt notation.

    Unlike the strain conversion helper, shear components are copied
    directly without the engineering-strain factor of 2.

    Parameters
    ----------
    stress_tensor : numpy.ndarray
        Shape ``(dim, dim)`` stress tensor.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    numpy.ndarray
        Shape ``(6, 1)`` stress vector in legacy Voigt ordering.
    """
    _validate_dim_nb(dim)
    stress_voigt = np.zeros((6,1))
    stress_voigt[0] = stress_tensor[0,0]
    stress_voigt[1] = stress_tensor[1,1]
    if dim == 3:
        stress_voigt[2] = stress_tensor[2,2]
        stress_voigt[3] = stress_tensor[1,2]
        stress_voigt[4] = stress_tensor[0,2]
    stress_voigt[5] = stress_tensor[0,1]
    return stress_voigt

@nb.njit(cache=True)
def sign_nb(a):
    """Return the sign of a scalar as ``-1.0``, ``0.0``, or ``1.0``.

    This Numba-compatible helper is used inside constitutive kernels
    where calling :func:`numpy.sign` is less convenient.
    """
    if a > 0.0:
        return 1.0
    elif a < 0.0:
        return -1.0
    else:
        return 0.0

@nb.njit(cache=True)
def Voigt_To_Tensor_Stress(stress_voigt, dim):
    """Convert a legacy Voigt stress vector back to a symmetric tensor.

    Parameters
    ----------
    stress_voigt : numpy.ndarray
        Shape ``(6, 1)`` stress vector in legacy Voigt ordering.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    numpy.ndarray
        Shape ``(dim, dim)`` stress tensor.
    """
    _validate_dim_nb(dim)
    stress_tensor = np.zeros((dim,dim))
    stress_tensor[0,0] = stress_voigt[0,0]
    stress_tensor[1,1] = stress_voigt[1,0]
    if dim == 3:
        stress_tensor[2,2] = stress_voigt[2,0]
        stress_tensor[1,2] = stress_voigt[3,0]
        stress_tensor[2,1] = stress_voigt[3,0]
        stress_tensor[0,2] = stress_voigt[4,0]
        stress_tensor[2,0] = stress_voigt[4,0]
    stress_tensor[0,1] = stress_voigt[5,0]
    stress_tensor[1,0] = stress_voigt[5,0]
    return stress_tensor

def Tensor_To_Voigt_Stress_UFL(stress_tensor, dim):
    """Convert a UFL stress tensor to legacy Voigt notation.

    Parameters
    ----------
    stress_tensor : ufl tensor
        ``(dim, dim)`` stress expression.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    ufl.as_vector
        Length-6 stress vector in legacy Voigt ordering.
    """
    _validate_dim(dim)
    if dim == 3:
        stress_voigt = [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            stress_tensor[2, 2],
            stress_tensor[1, 2],
            stress_tensor[0, 2],
            stress_tensor[0, 1],
        ]
    else:
        stress_voigt = [
            stress_tensor[0, 0],
            stress_tensor[1, 1],
            0.0,
            0.0,
            0.0,
            stress_tensor[0, 1],
        ]
    return as_vector(stress_voigt)

def Voigt_To_Tensor_Stress_UFL(stress_voigt, dim):
    """Convert a legacy Voigt stress vector to a symmetric UFL tensor.

    Parameters
    ----------
    stress_voigt : ufl vector
        Length-6 stress expression in legacy Voigt ordering.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    ufl.as_tensor
        ``(dim, dim)`` symmetric stress tensor.
    """
    _validate_dim(dim)
    if dim == 3:
        stress_tensor = [
            [stress_voigt[0]      , stress_voigt[5] , stress_voigt[4] ],
            [stress_voigt[5]      , stress_voigt[1] , stress_voigt[3] ],
            [stress_voigt[4]      , stress_voigt[3] , stress_voigt[2] ],
        ]
    else:
        stress_tensor = [
            [stress_voigt[0]      , stress_voigt[5] ],
            [stress_voigt[5]      , stress_voigt[1] ],
        ]
    return as_tensor(stress_tensor)
