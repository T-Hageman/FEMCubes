#pragma once
// viscoelastic_kernels.h
// Public declarations for the visco-elastic J2 return-mapping kernel.
//
// Mandel notation convention — 6 components:
//   [e11, e22, e33, sqrt(2)*e12, sqrt(2)*e13, sqrt(2)*e23]
// The sqrt(2) factors ensure that the double-contraction A:B equals
// the plain dot product of the Mandel vectors.
// In intrinsic 2-D mode, only [e11, e22, sqrt(2)e12] are active; the
// remaining allocated slots are explicitly zero-filled.
//
// Split:
//   viscoelastic_kernels_internal.cpp  – physics implementation
//   viscoelastic_kernels_pybind.cpp    – pybind11 module + top-level IP loop

#include <Eigen/Dense>
#include "../shared/mandel_physics.h"

namespace viscoelastic {

// --------------------------------------------------------------------------
// Types and constants
// --------------------------------------------------------------------------

static constexpr int    MANDEL_DIM = 6;
static constexpr double DE_MIN     = 1.0e-2;   // damage below this → pure elastic
static constexpr double LAMBDA_MIN = 1.0e-20;  // floor on plastic multiplier

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec7 = Eigen::Matrix<double, 7, 1>;
using Mat7 = Eigen::Matrix<double, 7, 7>;

// --------------------------------------------------------------------------
// Material parameters
// --------------------------------------------------------------------------

struct MaterialParams {
    double viscosity;  ///< eta_vp: J2 viscoplastic stress scale [Pa·s^{1/n}]
    double n_exp;      ///< n: power-law exponent
    double lame;       ///< first Lamé parameter [Pa]
    double shear;      ///< shear modulus [Pa]
    Mat6   D_el;       ///< isotropic elastic stiffness in Mandel notation
    bool   true_2d = false;  ///< intrinsic 2-D constitutive response
};

/// Build a MaterialParams and pre-compute D_el.
MaterialParams make_material_params(double viscosity, double n_exp,
                                    double lame,      double shear,
                                    bool   true_2d = false);

// --------------------------------------------------------------------------
// Per-IP return-mapping result
// --------------------------------------------------------------------------

struct IPResult {
    Vec6   stress;           ///< updated Cauchy stress [Pa]
    Vec6   strain_p;         ///< updated plastic (viscous) strain
    Mat6   dsigma_dstrain;   ///< consistent algorithmic tangent [Pa]
    double lambda;           ///< updated J2 equivalent viscous strain increment
};

// --------------------------------------------------------------------------
// Public API – implemented in viscoelastic_kernels_internal.cpp
// --------------------------------------------------------------------------

/// Update the state at one integration point via Newton return-mapping.
/// @param mp           Material parameters (includes pre-built D_el).
/// @param total_strain Total (elastic + viscous) strain in Mandel notation.
/// @param ep_old       Plastic strain from the previous time step.
/// @param dt           Time-step size [s].
/// @param dam          Elastic damage factor (1 = undamaged, 0 = fully broken).
IPResult update_ip_state(const MaterialParams& mp,
                         bool DamMaterial,
                         const Vec6& total_strain,
                         const Vec6& ep_old,
                         double      dt,
                         double      dam);

} // namespace viscoelastic
