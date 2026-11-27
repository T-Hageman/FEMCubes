// viscoelastic_kernels_internal.cpp
// J2 visco-elastic return-mapping kernel — physics implementation.
//
// All functions declared in viscoelastic_kernels.h are implemented here.
// The pybind11 module entry-point and the top-level IP loop live in
// viscoelastic_kernels_pybind.cpp.
//
// Mandel notation convention - 6 components:
//   [e11, e22, e33, sqrt(2)*e12, sqrt(2)*e13, sqrt(2)*e23]
// The sqrt(2) factors ensure that the double-contraction A:B equals
// the plain dot product of the Mandel vectors.
//
// The return-mapping system is always 7x7:  state = [sigma(6), Lambda(1)].
// Residuals:
//   F_sigma = sigma - D_el @ (eps_total - ep_old - Lambda * df(sigma)) = 0
//   F_Lambda = f(sigma) - eta_vp * (Lambda/dt)^(1/n)                   = 0
// where f is the dimension-aware J2 effective stress and Lambda/dt is the
// corresponding equivalent strain rate.

#include "viscoelastic_kernels.h"
#include "../shared/mandel_physics.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace viscoelastic {

// ==========================================================================
//  Public: MaterialParams factory
// ==========================================================================

MaterialParams make_material_params(double viscosity, double n_exp,
                                    double lame,      double shear,
                                    bool   true_2d) {
    MaterialParams mp;
    mp.viscosity = viscosity;
    mp.n_exp     = n_exp;
    mp.lame      = lame;
    mp.shear     = shear;
    mp.true_2d   = true_2d;
    mp.D_el      = true_2d
        ? shared_kernels::make_stiffness_mandel_2d(lame, shear)
        : shared_kernels::make_stiffness_mandel(lame, shear);
    return mp;
}

// ==========================================================================
//  J2 yield function and derivatives
// ==========================================================================

struct YieldFunction {
    double f;    ///< dimension-aware J2 effective stress
    Vec6   df;   ///< d(f)/d(sigma) in Mandel notation
    Mat6   ddf;  ///< d²(f)/d(sigma)²
};

static YieldFunction compute_yield_function(const Vec6& stress, bool true_2d) {
    const Mat6& dev_mat = true_2d
        ? shared_kernels::DevMat2d()
        : shared_kernels::DevMat();
    const double dimension = true_2d ? 2.0 : 3.0;

    // The intrinsic 2-D branch uses the same dimension-aware invariant as
    // the cohesive viscoelastic kernel; the default remains 3-D J2.
    double s_eff2 = 0.5 * dimension * (stress.transpose() * dev_mat * stress)(0, 0);
    constexpr double reg = 1.0e-10;

    double f       = std::sqrt(s_eff2 + reg*reg);
    Vec6   dev_sig = dev_mat * stress;

    YieldFunction r;
    r.f   = f-reg;
    r.df  = dimension * dev_sig / (2.0 * f);
    r.ddf = dimension * dev_mat / (2.0 * f)
          - dimension * dimension * (dev_sig * dev_sig.transpose())
            / (4.0 * s_eff2 * f);
    return r;
}

static Vec6 project_true_2d(const Vec6& value, bool true_2d) {
    return true_2d ? shared_kernels::InPlaneMat() * value : value;
}

// ==========================================================================
//  Local residual 
// ==========================================================================

struct KFCResult {
    Vec7 F;   ///< residual [sigma(6); Lambda(1)]
    Mat7 K;   ///< Jacobian dF/d_state (7x7)
    Mat6 C;   ///< -dF_sigma/d_eps_total (6x6); equals D_el at fixed sigma, Lambda
};

static KFCResult assemble_local_KF(const MaterialParams& mp,
                                    bool DamMaterial,
                                    double dam,
                                    const Vec7& state,
                                    const Vec6& total_strain,
                                    const Vec6& ep_old,
                                    double      dt) {
    const Vec6 total_strain_2d = project_true_2d(total_strain, mp.true_2d);
    const Vec6 ep_old_2d       = project_true_2d(ep_old, mp.true_2d);
    Vec6   sigma  = project_true_2d(state.segment<6>(0), mp.true_2d);
    double Lambda = state(6);

    YieldFunction f = compute_yield_function(sigma, mp.true_2d);
    double n       = mp.n_exp;
    double eta     = mp.viscosity;

    KFCResult sys;
    sys.F = Vec7::Zero();
    sys.K = Mat7::Zero();

    // --- Stress residual (rows 0..5) ---
    // F_sigma = sigma - D_el @ (eps - ep_old - Lambda * df(sigma)) = 0
    if (DamMaterial) {
        sys.F.segment<6>(0) = sigma - dam * mp.D_el * (total_strain_2d - ep_old_2d - Lambda * f.df);
        sys.K.block<6,6>(0,0) = Mat6::Identity() + dam * Lambda * (mp.D_el * f.ddf);
        sys.K.block<6,1>(0,6) = dam * mp.D_el * f.df;
        sys.C = dam * mp.D_el;
    } else {
        sys.F.segment<6>(0) = sigma - mp.D_el * (total_strain_2d - ep_old_2d - Lambda * f.df);
        sys.K.block<6,6>(0,0) = Mat6::Identity() + Lambda * (mp.D_el * f.ddf);
        sys.K.block<6,1>(0,6) = mp.D_el * f.df;
        sys.C = mp.D_el;
    }
    

    // --- Lambda residual (row 6) ---
    // F_Lambda = f(sigma) - eta * (Lambda/dt)^(1/n) = 0
    double strain_rate      = Lambda / dt;
    double strain_rate_safe = std::max(strain_rate, 1.0e-30);

    sys.F(6) = f.f - eta * std::pow(strain_rate_safe, 1.0 / n);

    sys.K.block<1,6>(6,0) = f.df.transpose();
    sys.K(6,6)            = -(eta / (n * dt)) * std::pow(strain_rate_safe, 1.0/n - 1.0);

    return sys;
}

// ==========================================================================
//  Schur-complement linear algebra for the 7x7 system
// ==========================================================================

/// Solve K * x = F via Schur complement.
/// K = [[A(6x6), b(6x1)], [c^T(1x6), d(scalar)]]
static Vec7 schur_solve_7(const Mat7& K, const Vec7& F) {
    Eigen::Matrix<double,6,6> A  = K.block<6,6>(0,0);
    Eigen::Matrix<double,6,1> b  = K.block<6,1>(0,6);
    Eigen::Matrix<double,1,6> cT = K.block<1,6>(6,0);
    double                    d  = K(6,6);

    Eigen::PartialPivLU<Eigen::Matrix<double,6,6>> lu(A);
    Vec6 Ainv_Fsig = lu.solve(F.segment<6>(0));
    Vec6 Ainv_b    = lu.solve(b);

    // Schur complement (scalar): S = d - c^T * A^{-1} * b
    double S = d - (cT * Ainv_b)(0);

    Vec7 x;
    x(6) = (F(6) - (cT * Ainv_Fsig)(0)) / S;
    x.segment<6>(0) = Ainv_Fsig - Ainv_b * x(6);
    return x;
}

/// Top-left 6x6 block of K^{-1} via the block-matrix inversion lemma.
/// Used to form the consistent algorithmic tangent: dsigma_dstrain = Kinv_00 * D_el.
static Mat6 schur_inv_block_00(const Mat7& K) {
    Eigen::Matrix<double,6,6> A  = K.block<6,6>(0,0);
    Eigen::Matrix<double,6,1> b  = K.block<6,1>(0,6);
    Eigen::Matrix<double,1,6> cT = K.block<1,6>(6,0);
    double                    d  = K(6,6);

    Eigen::PartialPivLU<Eigen::Matrix<double,6,6>> lu(A);
    Mat6 Ainv   = lu.solve(Mat6::Identity());
    Vec6 Ainv_b = lu.solve(b);

    double S = d - (cT * Ainv_b)(0);

    // (K^{-1})_{00} = Ainv + Ainv_b * (1/S) * c^T * Ainv
    Eigen::Matrix<double,1,6> Sinv_cT_Ainv = (cT * Ainv) / S;
    return Ainv + Ainv_b * Sinv_cT_Ainv;
}

// ==========================================================================
//  Single-IP state update  (Newton + line search)
// ==========================================================================

IPResult update_ip_state(const MaterialParams& mp,
                         bool DamMaterial,
                         const Vec6& total_strain,
                         const Vec6& ep_old,
                         double      dt,
                         double      dam) {
    const Vec6 total_strain_2d = project_true_2d(total_strain, mp.true_2d);
    const Vec6 ep_old_2d       = project_true_2d(ep_old, mp.true_2d);

    // --- Pure elastic: too damaged ---
    if (dam < DE_MIN) {
        IPResult r;
        r.stress         = project_true_2d(mp.D_el * (total_strain_2d - ep_old_2d), mp.true_2d);
        r.strain_p       = ep_old_2d;
        r.dsigma_dstrain = mp.D_el;
        r.lambda         = 0.0;
        return r;
    }

    // --- Trial stress ---
    Vec6   sigma_trial = mp.D_el * (total_strain_2d - ep_old_2d);
    if (DamMaterial) {
        sigma_trial *= dam;
    }
    double f_trial     = compute_yield_function(sigma_trial, mp.true_2d).f;

    // Below yield: elastic return
    if (f_trial < 1.0e-6) {
        IPResult r;
        r.stress         = sigma_trial;
        r.strain_p       = ep_old_2d;
        r.dsigma_dstrain = DamMaterial ? dam * mp.D_el : mp.D_el;
        r.lambda         = 0.0;
        return r;
    }

    // --- Initialise Newton state from analytic estimate ---
    Vec7 state;
    state.segment<6>(0) = sigma_trial;
    double Lambda_init  = dt * std::pow(f_trial / mp.viscosity, mp.n_exp);
    state(6)            = std::max(Lambda_init, LAMBDA_MIN);

    // --- Newton loop ---
    constexpr int    MAX_ITERS = 100;
    constexpr double RTOL      = 1.0e-9;
    const     double ATOL      = 1.0e-9 * (mp.lame + 2.0 * mp.shear);
    bool converged             = false;

    KFCResult sys  = assemble_local_KF(mp, DamMaterial, dam, state, total_strain, ep_old, dt);
    double    norm0 = sys.F.norm();

    if (norm0 < ATOL) converged = true;

    for (int iter = 0; iter < MAX_ITERS && !converged; ++iter) {
        Vec7 delta = -schur_solve_7(sys.K, sys.F);

        // Secant line-search (identical pattern to cohesive kernels)
        // double    e0          = delta.dot(sys.F);
        // Vec7      state_trial = state + delta;
        // KFCResult sys1        = assemble_local_KF(mp, state_trial, total_strain, ep_old, dt);
        // double    e1          = delta.dot(sys1.F);

        double ls = 1.0;
        // double denom = e1 - e0;
        // if (std::abs(denom) > 1.0e-20)
        //     ls = std::min(std::max(-e0 / denom, 0.1), 1.0);

        state    = state + ls * delta;
        // state(6) = std::max(state(6), LAMBDA_MIN);  // clamp Lambda

        sys = assemble_local_KF(mp, DamMaterial, dam, state, total_strain, ep_old, dt);

        // NaN / Inf guard
        if (!sys.F.allFinite() || !sys.K.allFinite()) {
            std::cerr << "WARNING: Viscoelastic return mapping produced non-finite values. "
                      << "Returning trial stress.\n";
            IPResult r;
            r.stress         = sigma_trial;
            r.strain_p       = ep_old_2d;
            r.dsigma_dstrain = DamMaterial ? dam * mp.D_el : mp.D_el;
            r.lambda         = 0.0;
            return r;
        }

        if (sys.F.norm() < RTOL * norm0 + ATOL) { converged = true; break; }
    }

    if (!converged) {
        std::cerr << "WARNING: Viscoelastic did not converge "
                  << "(|F|/|F0|=" << sys.F.norm() / (norm0 + 1.0)
                  << " |F0|=" << norm0 << ")\n";
    }

    // --- Post-process: stress, plastic strain, tangent ---
    double Lambda        = state(6);
    Vec6   sigma         = state.segment<6>(0);
    sigma = project_true_2d(sigma, mp.true_2d);
    YieldFunction yr_final = compute_yield_function(sigma, mp.true_2d);

    Vec6 strain_p_new = project_true_2d(ep_old_2d + Lambda * yr_final.df, mp.true_2d);

    // Consistent algorithmic tangent:
    //   d(sigma)/d(eps_total) = (K^{-1})_{0:6,0:6} * D_el
    // = Kinv_00 * sys.C   (since sys.C = D_el at fixed sigma, Lambda)
    Mat6 Kinv_00        = schur_inv_block_00(sys.K);
    Mat6 dsigma_dstrain = Kinv_00 * sys.C;
    if (mp.true_2d) {
        dsigma_dstrain = shared_kernels::InPlaneMat()
                       * dsigma_dstrain
                       * shared_kernels::InPlaneMat();
    }

    IPResult result;
    result.stress         = sigma;
    result.strain_p       = strain_p_new;
    result.dsigma_dstrain = dsigma_dstrain;
    result.lambda         = Lambda;
    return result;
}

} // namespace viscoelastic
