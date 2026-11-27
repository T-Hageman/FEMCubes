#pragma once
// cohesive_viscoelastic_kernels.h
// Public declarations for the combined cohesive–visco-elastic return-mapping.
//
// Split:
//   cohesive_viscoelastic_kernels_internal.cpp  – physics implementation
//   cohesive_viscoelastic_kernels_pybind.cpp    – pybind11 module + IP loop
//
// Mandel notation convention — 6 components:
//   [e11, e22, e33, sqrt(2)*e23, sqrt(2)*e13, sqrt(2)*e12]
//
// State vector (size N = 7 + nl_c, or 8 + nl_c for plane stress):
//   [sigma(6), Lambda_v(1), lambda_c(nl_c)]
//   [sigma(6), Lambda_v(1), lambda_c(nl_c), e_zz(1)] for plane stress
// Intrinsic 2-D keeps the first layout and zero-fills inactive Mandel
// components [zz, yz, xz].
//
// All three mechanisms are solved simultaneously in one Newton iteration,
// giving a fully-consistent algorithmic tangent without the operator-split
// lag of the staggered Python formulation.
//
// Residuals:
//   R_sigma  = sigma - D_el*(eps - ep_old - Lambda_v*df_v - eta_c)        (6)
//   R_Lv     = sigma_eq - eta_vp*(Lambda_v/dt)^(1/n)                      (1)
//   R_lci    = penalty (if inactive) or fracture drive (if active)      (nl_c)
//
// Consistent tangent:
//   dsigma/deps = -(K^{-1})_{0:6, :} * C_{:, 0:6}            (same pattern
//                                                              as both other
//                                                              kernels)

#include "../shared/fracture_relations.h"
#include <string>
#include <vector>

namespace cohesive_viscoelastic {

using namespace shared_kernels;

// --------------------------------------------------------------------------
// Debug flags  (define at compile time to isolate mechanisms)
// --------------------------------------------------------------------------

/// Define to force all cohesive modes inactive (no fracture).
/// Combined visco-elastic behaviour only.
// #define DEBUG_NO_FRACTURE

/// Define to zero the viscous yield function (no creep).
/// Combined cohesive-elastic behaviour only.
/// #define DEBUG_NO_CREEP

// --------------------------------------------------------------------------
// Constants
// --------------------------------------------------------------------------

static constexpr double LAMBDA_COH_MIN = 0*1.0e-16;  ///< floor on lambda_ci

// --------------------------------------------------------------------------
// Material parameters
// --------------------------------------------------------------------------

struct MaterialParams {
    // Elastic
    double bulk;
    double lame;
    double shear;
    Mat6   D_el;   ///< pre-built isotropic stiffness in Mandel notation

    // Viscous (standard J2 viscoplasticity; Glen-compatible parameterisation)
    double viscosity;  ///< eta_vp [Pa·s^{1/n}]
    double n_exp;      ///< power-law exponent n

    // Cohesive fracture
    double f_t;
    double f_s;
    double G2_over_G1;
    double DP_phi;
    double DP_q;
    double kappa;      ///< cohesive viscosity regularisation coefficient
    int    nl_c;       ///< number of cohesive multipliers (1, 2 or 3)
    bool   plane_stress; ///< solve e_zz locally to enforce sigma_zz = 0
    bool   true_2d;      ///< use intrinsic 2-D invariants and zero-fill inactive components
    std::string failureType;
};

/// Build a MaterialParams and pre-compute D_el and nl_c.
MaterialParams make_material_params(
    double bulk, double lame, double shear,
    double viscosity, double n_exp,
    double f_t, double f_s,
    double G2_over_G1, double DP_phi, double DP_q,
    double kappa,
    const std::string& failureType,
    bool plane_stress,
    bool true_2d);



// --------------------------------------------------------------------------
// KF inspection (debug / testing)
// --------------------------------------------------------------------------

/// Jacobian, residual and strain-coupling matrix assembled at one IP.
struct KFCR {
    MatXX K;  ///< (7+nl_c)×(7+nl_c) Newton Jacobian
    VecX  F;  ///< (7+nl_c)×1 residual
    MatXX C;  ///< (7+nl_c)×6 dR/d(eps_total)
    std::vector<bool> active_coh;  ///< active cohesive equations used in K/F
};

/// Run the Newton solve for one IP and return the assembled (K, F, C) at the
/// converged state.  Intended for testing and debugging the tangent.
KFCR inspect_kf_single_ip(
    const MaterialParams& mp,
    const Vec6&  total_strain,
    const Vec6&  eta_c_init,
    const Vec6&  eta_c_old,
    const VecX&  lambda_c_init,
    const VecX&  lambda_c_old,
    double       lambda_v_init,
    const Vec6&  ep_old,
    double       dt,
    double       dam);

// --------------------------------------------------------------------------
// Per-IP result
// --------------------------------------------------------------------------

struct IPResult {
    Vec6   stress;            ///< updated Cauchy stress [Pa]
    Vec6   eta_c;             ///< updated cohesive fracture strain
    Vec6   strain_v;          ///< updated viscous (plastic) strain
    Mat6   dsigma_dstrain;    ///< consistent algorithmic tangent [Pa]
    double E_elastic;         ///< elastic energy density [J/m³]
    double E_psi;             ///< cohesive fracture energy density [J/m³]
    double lambda_v;          ///< J2 equivalent viscous strain increment
    double lambda_c0;         ///< first cohesive multiplier
    double lambda_c1;         ///< second cohesive multiplier (0 if nl_c == 1)
    double lambda_c2;         ///< third cohesive multiplier (0 if nl_c < 3)
    double e_zz;              ///< converged out-of-plane strain
};

// --------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------

/// Update the state at one integration point via the combined Newton solve.
IPResult update_ip_state(const MaterialParams& mp,
                         const Vec6&  total_strain,
                         const Vec6&  eta_c_current,
                         const Vec6&  eta_c_old,
                         const VecX&  lambda_c_current,
                         const VecX&  lambda_c_old,
                         double       lambda_v_current,
                         const Vec6&  ep_old,
                         double       dt,
                         double       dam);

} // namespace cohesive_viscoelastic
