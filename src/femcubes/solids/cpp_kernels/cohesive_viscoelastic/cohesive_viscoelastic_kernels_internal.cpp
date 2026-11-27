// cohesive_viscoelastic_kernels_internal.cpp
// Combined cohesive–visco-elastic return-mapping kernel.
//
// Solves standard J2 viscous creep (Glen/Newtonian compatible) and cohesive fracture
// simultaneously in a single Newton iteration
//
// State vector  x ∈ ℝ^N,  N = 7 + nl_c:
//   x[0:6]   = sigma   (Cauchy stress, Mandel)
//   x[6]     = Lambda_v (viscous multiplier)
//   x[7:]    = lambda_c  (cohesive multipliers, nl_c entries)
//   x[end]   = e_zz (out-of-plane strain, optional for plane-stress)
//
// Residuals:
//   F_sigma = sigma - D_el*(eps - ep_old - Lambda_v*df_v/dsigma - eta_c) = 0
//   F_Lv    = sigma_eq - eta_vp*(Lambda_v/dt)^(1/n)                      = 0
//   F_lci   = penalty (inactive) / fracture drive (active)
//   F_ezz   = sigma_zz (plane stress only)
//
// Tangent (identical pattern to both other kernels):
//   dsigma/deps  = -(K^{-1})_{0:6,:} * C_{:,0:6}
// where C = -dF/deps_total.

#include "cohesive_viscoelastic_kernels.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <thread>
#include <utility>

namespace cohesive_viscoelastic {

// Numerical experiment switch: reuse the multipliers from the last locally
// converged constitutive call as the next Newton initial guess.  Set to false
// to recover the original zero-multiplier start, then run scripts/reinstall.sh.
// This changes only the initial guess; the residuals and tangent are unchanged.
static constexpr bool RESTART_FROM_PREVIOUS_CONVERGED_STATE = true;

// ==========================================================================
//  Module-level Mandel constants (aliases into mandel_physics)
// ==========================================================================

//  We alias the shared inline functions to avoid name look-up overhead in the
//  hot path.  These are evaluated once at program start.
static const Vec6& k_I1vec  = []() -> const Vec6& { return shared_kernels::I1vec();  }();
static const Mat6& k_DevMat = []() -> const Mat6& { return shared_kernels::DevMat(); }();
static const Mat6& k_I6     = []() -> const Mat6& { return shared_kernels::I6();     }();

// ==========================================================================
//  Public: MaterialParams factory
// ==========================================================================

MaterialParams make_material_params(
    double bulk, double lame, double shear,
    double viscosity, double n_exp,
    double f_t, double f_s,
    double G2_over_G1, double DP_phi, double DP_q,
    double kappa,
    const std::string& failureType,
    bool plane_stress,
    bool true_2d){

    if (plane_stress && true_2d)
        throw std::invalid_argument("plane_stress and true_2d cannot both be enabled");
    if (true_2d && shared_kernels::is_opening_only(failureType))
        throw std::invalid_argument("r2_2L_PSD is not available in intrinsic 2-D");
    if (shared_kernels::is_opening_only(failureType))
        // rejects strengths for which f_t is not the uniaxial tensile strength
        shared_kernels::compute_r2_strength_scales(f_t, f_s, failureType, true_2d);

    MaterialParams mp;
    mp.bulk        = true_2d ? lame + shear : bulk;
    mp.lame        = lame;
    mp.shear       = shear;
    mp.D_el        = true_2d
        ? make_stiffness_mandel_2d(lame, shear)
        : make_stiffness_mandel(lame, shear);
    mp.viscosity   = viscosity;
    mp.n_exp       = n_exp;
    mp.f_t         = f_t;
    mp.f_s         = f_s;
    mp.G2_over_G1 = G2_over_G1;
    mp.DP_phi      = DP_phi;
    mp.DP_q        = DP_q;
    mp.kappa       = kappa;
    mp.nl_c        = shared_kernels::get_cohesive_num_lambdas(failureType);
    mp.plane_stress = plane_stress;
    mp.true_2d     = true_2d;
    mp.failureType = failureType;
    return mp;
}

// ==========================================================================
//  J2 yield function for the viscous sub-problem
// ==========================================================================

struct YieldFunc {
    double f;    ///< sigma_eq = sqrt(3*J2)
    Vec6   df;   ///< d(f)/d(sigma)
    Mat6   ddf;  ///< d²(f)/d(sigma)²
};

static YieldFunc compute_yield(const Vec6& sigma, bool true_2d) {
    const Mat6& dm = true_2d ? DevMat2d() : k_DevMat;
    const double dimension = true_2d ? 2.0 : 3.0;
    constexpr double reg = 1.0e-10;

    YieldFunc r;

    #ifdef DEBUG_NO_CREEP
        r.f   = 0.0;
        r.df  = Vec6::Zero();
        r.ddf = Mat6::Zero();
    #else
        double s2 = 0.5 * dimension * (sigma.transpose() * dm * sigma)(0,0);
        double f      = std::sqrt(s2 + reg*reg);
        Vec6   ds     = dm * sigma;

        r.f   = f - reg;
        r.df  = dimension * ds / (2.0 * f);
        r.ddf = dimension * dm / (2.0 * f)
              - dimension * dimension * (ds * ds.transpose()) / (4.0 * s2 * f);
    #endif
    return r;
}

static Vec6 project_true_2d(const Vec6& value, bool true_2d) {
    return true_2d ? InPlaneMat() * value : value;
}
 
#pragma region K/F matrix

// ==========================================================================
//  Local residual / Jacobian assembly
// ==========================================================================

/// Assemble the combined residual, Jacobian and strain-coupling matrix.
/// The cohesive active set is computed from the current state (semi-smooth Newton):
/// mode i is active when the fracture drive dF/dlambda_i + dE/dlambda_i < 0.
static KFCR assemble_KF(const MaterialParams& mp,
                         const VecX&           state,
                         const Vec6&           total_strain_in,
                         const Vec6&           strain_v_old_in,
                         double                dt,
                         double                dam,
                         const std::vector<bool>* active_hint = nullptr,
                         bool                  freeze_active = false)
{
    const int nl_c = mp.nl_c;
    const int N    = 7 + nl_c + (mp.plane_stress ? 1 : 0);
    const int ezz_idx = 7 + nl_c;
    const Vec6 total_strain = project_true_2d(total_strain_in, mp.true_2d);
    const Vec6 strain_v_old = project_true_2d(strain_v_old_in, mp.true_2d);

    Vec6   sigma    = state.segment(0, 6);
    Vec6   zz_select = Vec6::Zero(); zz_select(2) = 1.0; // select ezz for plane stress
    double Lambda_v = state(6);
    VecX   lambda_c = state.segment(7, nl_c);
    Vec6   total_strain_state = total_strain;
    if (mp.plane_stress)
        total_strain_state(2) = state(ezz_idx);

    // --- viscous yield function ---
    YieldFunc yf = compute_yield(sigma, mp.true_2d);

    // --- cohesive eta and derivatives ---
    Vec6 strain_visc_el = total_strain_state - strain_v_old - Lambda_v * yf.df;
    EtaResult er = compute_eta(nl_c, strain_visc_el, lambda_c, mp.true_2d);
    const Vec6& eta_c = er.eta;

    // --- elastic residual strain ---
    // eps_elastic = eps_total - strain_v_old - Lambda_v * df_v - eta_c
    Vec6 strain_el  = total_strain_state - strain_v_old - Lambda_v * yf.df - eta_c;
    Vec6 strain_eta = total_strain_state - strain_v_old - eta_c;

    // --- failure surface ---
    // Select tensile/compressive and DP branches from the pre-cohesive strain.
    // Otherwise a growing cohesive opening can switch its own potential into
    // the artificial compression-penalty branch and create a spurious root.
    FailureSurfaceResult fs = compute_failure_surface(
        mp.f_t, mp.f_s, mp.bulk, mp.G2_over_G1, mp.DP_phi, mp.DP_q,
        mp.failureType, eta_c, dam, strain_visc_el, mp.true_2d);

    // Cohesive viscous regularization (unscaled local residual form)
    const double visc_coh  = 1.0e-6 * mp.bulk * mp.kappa * 1e-2;
    const double damaged_strength = dam * std::max(mp.f_t, mp.f_s);
    double min_exceed = 1.0e-6 * damaged_strength;

    // --- determine active failure modes ---
    // Probe above the eta deviatoric regularisation. At LAMBDA_COH_MIN the
    // failure gradient tends to zero and would make every loaded shear mode
    // appear active, even while the stress is below the failure surface.
    constexpr double LAMBDA_COH_PROBE = 1.0e-9;
    VecX lambda_c_probe = lambda_c.cwiseMax(LAMBDA_COH_PROBE);
    EtaResult er_probe = compute_eta(
        nl_c, strain_visc_el, lambda_c_probe, mp.true_2d);
    FailureSurfaceResult fs_probe = compute_failure_surface(
        mp.f_t, mp.f_s, mp.bulk, mp.G2_over_G1, mp.DP_phi, mp.DP_q,
        mp.failureType, er_probe.eta, dam, strain_visc_el, mp.true_2d);

    // "r2_2L_PSD": with every multiplier at its bound eta = 0, where the
    // potential is not differentiable; onset is then decided below from the
    // gauge of the stress. Once a direction has opened the potential is
    // smooth, and a direction at its bound is tested with its drive at the
    // current state: the common probe of all multipliers would replace eta
    // by an isotropic tensor and test each direction against f_t_vol/3.
    constexpr double INTERIOR_LAMBDA_TOL = 1.0e-14;
    const bool opening_only = shared_kernels::is_opening_only(mp.failureType);
    const bool opening_closed = opening_only
        && !(lambda_c.array() > LAMBDA_COH_MIN + INTERIOR_LAMBDA_TOL).any();

    std::vector<bool> active_coh(nl_c);
    for (int i = 0; i < nl_c; ++i) {
        if (freeze_active && active_hint != nullptr) {
            active_coh[i] = (*active_hint)[i];
            continue;
        }

        // Complementarity requires every interior split-r2 multiplier to
        // satisfy its cohesive drive equation. Do not let the onset
        // hysteresis reclassify either independent mode as inactive.
        if ((shared_kernels::is_split_r2(mp.failureType) || opening_only)
            && lambda_c(i) > LAMBDA_COH_MIN + INTERIOR_LAMBDA_TOL) {
            active_coh[i] = true;
            continue;
        }
        if (opening_closed) {
            active_coh[i] = false;
            continue;
        }

        const EtaResult& er_test = opening_only ? er : er_probe;
        const FailureSurfaceResult& fs_test = opening_only ? fs : fs_probe;
        const Vec6 deta_dL = er_test.deta_dl.row(i).transpose();
        const double drive = failure_lambda_derivative(fs_test, er_test, i)
                           - sigma.dot(deta_dL);
        if (drive < -min_exceed) {
            active_coh[i] = true;
        } else if (drive > min_exceed) {
            active_coh[i] = false;
        } else {
            active_coh[i] = active_hint != nullptr && (*active_hint)[i];
        }
    }

    // At the tensile onset of a split-r2 criterion, activation is governed
    // by the dual norm of the volumetric and deviatoric stress projections.
    // Testing each mode separately is not sufficient: both projections can
    // be below their single-mode strengths while their coupled norm is above
    // one. Use the onset test only on the tensile branch; the unscaled
    // volumetric penalty must continue to protect against interpenetration
    // in compression.
    const Vec6& i1vec = mp.true_2d ? I1vec2d() : k_I1vec;
    const bool r2_2l_tensile = shared_kernels::is_split_r2(mp.failureType)
        && i1vec.dot(strain_visc_el) >= 0.0;
    if (!freeze_active && opening_closed) {
        // Gauge of the stress on the cone lambda >= 0 of the principal
        // dyads, for either sign of the strain trace.
        const R2StrengthScales strengths = compute_r2_strength_scales(
            mp.f_t, mp.f_s, mp.failureType, mp.true_2d);
        const double trace_weight = strengths.f_t_vol * strengths.f_t_vol / 9.0;
        const double dev_weight = 0.5 * strengths.f_s_dev * strengths.f_s_dev;
        const Eigen::Vector3d normal_stress = er.deta_dl * sigma;
        const shared_kernels::OpeningOnset onset = shared_kernels::compute_opening_onset(
            normal_stress, trace_weight, dev_weight);

        if (onset.gauge > dam * (1.0 + 1.0e-10)) {
            for (int i = 0; i < nl_c; ++i)
                active_coh[i] = onset.opens[i];
        }
    } else if (!freeze_active && r2_2l_tensile) {
        const EigenStrainDirections fd = compute_eta_directions(
            strain_visc_el, mp.true_2d);
        const R2StrengthScales strengths = compute_r2_strength_scales(
            mp.f_t, mp.f_s, mp.failureType, mp.true_2d);
        const double dimension = mp.true_2d ? 2.0 : 3.0;
        // The potential scales both tensile modes by dam. Its onset surface
        // must use the same strengths, including for a cold inactive guess.
        const double vol_strength = dam * strengths.f_t_vol / std::sqrt(dimension);
        const double dev_strength = dam * strengths.f_s_dev / std::sqrt(2.0);
        const double vol_projection = sigma.dot(fd.vol_dir);
        const double dev_projection = sigma.dot(fd.dev_dir);
        const double vol_ratio = std::max(vol_projection, 0.0) / vol_strength;
        const double dev_ratio = std::max(dev_projection, 0.0) / dev_strength;
        const double dual_strength_ratio = std::hypot(vol_ratio, dev_ratio);

        if (dual_strength_ratio > 1.0 + 1.0e-10) {
            if (vol_projection > 0.0)
                active_coh[0] = true;
            if (dev_projection > 0.0)
                active_coh[1] = true;
        }
    }

    KFCR sys;
    sys.K = MatXX::Zero(N, N);
    sys.F = VecX::Zero(N);
    sys.C = MatXX::Zero(N, 6);
    sys.active_coh = active_coh;

    // ------------------------------------------------------------------
    // Row 0..5 — stress residual
    // R_sigma = sigma - D_el * eps_el = 0
    // ------------------------------------------------------------------
    sys.F.segment(0, 6)     = sigma - mp.D_el  * (total_strain - strain_v_old - Lambda_v * yf.df  - eta_c);
    sys.K.block(0, 0, 6, 6) = k_I6  - mp.D_el * (                             - Lambda_v * yf.ddf - er.deta_de * (-Lambda_v * yf.ddf));
    sys.K.block(0, 6, 6, 1) =       - mp.D_el * (                             - yf.df             - er.deta_de * -yf.df);
    for (int i = 0; i < nl_c; ++i) {
        Vec6 deta_dLambdai = er.deta_dl.row(i).transpose();
        sys.K.block(0, 7 + i, 6, 1) = -mp.D_el *                                                  - deta_dLambdai;
    }
    sys.C.block(0, 0, 6, 6) =       - mp.D_el * (k_I6                                             - er.deta_de);

    if (mp.plane_stress){ // add out-of-plane strain state and dependence when using plane-stress
        sys.F.segment(0,6) +=       -mp.D_el * (zz_select * state(ezz_idx));
        sys.K.block(0, ezz_idx, 6, 1) =  -mp.D_el * (k_I6 - er.deta_de) * zz_select;
    }

    // ------------------------------------------------------------------
    // Row 6 — viscous multiplier residual (creep-rate form)
    // G_Lv = Lambda_v - (dt / eta_vp^n) * sigma_eq^n = 0
    //
    // eta_vp is the standard J2 stress scale.  For Glen's rate factor A,
    // eta_vp = [3^((n+1)/2)/(2A)]^(1/n), which exactly gives
    // epsdot_v = A*tau_e^(n-1)*s with tau_e = sqrt(J2).
    // ------------------------------------------------------------------
    double yf_f_safe = std::max(yf.f, 0.0);
    double yf_f_n    = std::pow(yf_f_safe, mp.n_exp);
    double yf_f_nm1  = (yf_f_safe > 0.0) ? std::pow(yf_f_safe, mp.n_exp - 1.0) : 0.0;
    double dt_over_visc_n;
    if (mp.viscosity <= 0.0 || mp.n_exp <= 0.0) {
        dt_over_visc_n = 0.0;
    } else {
        dt_over_visc_n = dt / std::pow(mp.viscosity, mp.n_exp);
    }

    sys.F(6)                = Lambda_v - dt_over_visc_n * yf_f_n;
    sys.K.block(6, 0, 1, 6) = -dt_over_visc_n * mp.n_exp * yf_f_nm1 * yf.df.transpose();
    sys.K(6, 6)             = 1.0;

    // ------------------------------------------------------------------
    // Rows 7..7+nl_c-1 — cohesive multiplier residuals
    // Active set computed from current total strains and viscous strains.
    // ------------------------------------------------------------------

    for (int i = 0; i < nl_c; ++i) {
        Vec6 deta_dLambdai = er.deta_dl.row(i).transpose();

        Mat6 d2F_detade = fs.ddF_detade;
        Mat6 d2eta_dlde = er.d2eta_dlde.block(i * MANDEL_DIM, 0, MANDEL_DIM, MANDEL_DIM);
        const Mat6& dev_mat = mp.true_2d ? DevMat2d() : k_DevMat;
        const Vec6& i1vec = mp.true_2d ? I1vec2d() : k_I1vec;
        const bool deviatoric_single_mode = nl_c == 1
                                          && i1vec.dot(strain_visc_el) < 0.0;
        Vec6 regular_cohesive_force = fs.dF_deta_regular.transpose() - sigma;
        if (deviatoric_single_mode)
            regular_cohesive_force = dev_mat * regular_cohesive_force;

        const Eigen::Matrix<double, 1, 6> direction_change =
            regular_cohesive_force.transpose() * d2eta_dlde
          + fs.dF_penalty_dtrace * er.d2trace_eta_dlde.row(i);

        // Mechanical cohesive drive. Complementarity below enforces either
        // lambda_i at its lower bound or this drive equal to zero.
        sys.F(7 + i) = failure_lambda_derivative(fs, er, i)
                     - sigma.dot(deta_dLambdai)
                     + visc_coh * lambda_c(i);

        // dF/dsigma (direct) + chain through strain_visc_el = eps - ep_old - Lv*df(sigma): d(strain_visc_el)/dsigma = -Lv*ddf
        if (deviatoric_single_mode)
            sys.K.block(7 + i, 0, 1, 6) = -deta_dLambdai.transpose() * dev_mat;
        else
            sys.K.block(7 + i, 0, 1, 6) = -deta_dLambdai.transpose();
        // The failure branch state is strain_visc_el, whose stress derivative
        // is -Lambda_v*ddf and is independent of cohesive strain.
        sys.K.block(7 + i, 0, 1, 6) -= Lambda_v * (
                deta_dLambdai.transpose() * fs.ddF_deta2 * er.deta_de
              + deta_dLambdai.transpose() * d2F_detade
              + direction_change
            ) * yf.ddf;

        // dF/dstrain * dstrain/dstrain_ve * dstrain_ve/dLambda_v
        // d(strain_visc_el)/dLv = -yf.df.
        double p1 = -(direction_change * yf.df).value();
        double p2 = -deta_dLambdai.transpose() * d2F_detade * yf.df;
        double p3 = -deta_dLambdai.transpose() * fs.ddF_deta2 * er.deta_de * yf.df;
        sys.K(7 + i, 6) = p1 + p2 + p3;

        // dF/d(lambda_c[j])
        // The branch state is independent of lambda_c.
        for (int j = 0; j < nl_c; ++j) {
            Vec6 deta_j = er.deta_dl.row(j).transpose();
            sys.K(7 + i, 7 + j) =
                (deta_dLambdai.transpose() * fs.ddF_deta2 * deta_j).value();
        }
        sys.K(7 + i, 7 + i) += visc_coh;

        // d(strain_visc_el)/deps_total = I.
        sys.C.block(7 + i, 0, 1, 6) =(
                deta_dLambdai.transpose() * fs.ddF_deta2 * er.deta_de
              + deta_dLambdai.transpose() * d2F_detade
              + direction_change
            );
        if (mp.plane_stress){
            sys.K.block(7 + i, ezz_idx, 1, 1) = (
                deta_dLambdai.transpose() * fs.ddF_deta2 * er.deta_de
              + deta_dLambdai.transpose() * d2F_detade
              + direction_change
            ) * zz_select;
        }

        // Fischer-Burmeister map phi(a,b)=sqrt(a^2+b^2)-a-b. Map the
        // multiplier to the current damaged-strength scale rather than the
        // full bulk modulus so neither argument dominates through units alone.
        const double actual_drive = sys.F(7 + i);
        const Vec6 deta_probe = er_probe.deta_dl.row(i).transpose();
        const double probe_drive = failure_lambda_derivative(fs_probe, er_probe, i)
                                 - sigma.dot(deta_probe)
                                 + visc_coh * lambda_c(i);
        const double drive = active_coh[i]
            ? actual_drive
            : std::max(probe_drive, min_exceed);
        const double strength_for_scale = std::max(damaged_strength, mp.bulk * 1.0e-12);
        const double lambda_characteristic = std::max(
            std::max(std::max(total_strain.norm(), strain_v_old.norm()),
                     strength_for_scale / mp.bulk),
            1.0e-12);
        const double lambda_to_stress = strength_for_scale / lambda_characteristic;
        const double lambda_distance = lambda_to_stress * (lambda_c(i) - LAMBDA_COH_MIN);
        const double fb_norm = std::hypot(lambda_distance, drive);
        double dphi_da;
        double dphi_db;
        if (fb_norm > 1.0e-14 * std::max(damaged_strength, 1.0)) {
            dphi_da = lambda_distance / fb_norm - 1.0;
            dphi_db = drive / fb_norm - 1.0;
        } else {
            dphi_da = 1.0 / std::sqrt(2.0) - 1.0;
            dphi_db = dphi_da;
        }

        const double fb_denominator = fb_norm + lambda_distance + drive;
        sys.F(7 + i) = fb_denominator > 0.0
            ? -2.0 * lambda_distance * drive / fb_denominator
            : fb_norm - lambda_distance - drive;
        sys.K.row(7 + i) *= dphi_db;
        sys.K(7 + i, 7 + i) += dphi_da * lambda_to_stress;
        sys.C.row(7 + i) *= dphi_db;
    }

    // ------------------------------------------------------------------
    // Row ezz_idx — plane-stress condition: sigma_zz = 0
    // ------------------------------------------------------------------

    if (mp.plane_stress) { 
        // plane stress constraint: sigma_zz = 0
        sys.F(ezz_idx)      = zz_select.transpose() * sigma;
        sys.K(ezz_idx, 2)   = 1.0; 
        //sys.K(ezz_idx, ezz_idx) = mp.viscosity;
    }

    return sys;
}

#pragma endregion

#pragma region Schur-complement solve
// ==========================================================================
//  Schur-complement solve for the combined N×N system
// ==========================================================================

static MatXX solve_equilibrated_block(const MatXX& A, const MatXX& B) {
    if (A.rows() != A.cols() || A.rows() != B.rows())
        throw std::runtime_error("CVE equilibrated solve received incompatible dimensions");

    const int n = A.rows();
    VecX row_scale(n);
    for (int i = 0; i < n; ++i)
        row_scale(i) = 1.0 / std::max(A.row(i).cwiseAbs().maxCoeff(), 1.0e-300);

    MatXX A_scaled = row_scale.asDiagonal() * A;
    VecX col_scale(n);
    for (int j = 0; j < n; ++j)
        col_scale(j) = 1.0 / std::max(A_scaled.col(j).cwiseAbs().maxCoeff(), 1.0e-300);
    A_scaled *= col_scale.asDiagonal();

    Eigen::ColPivHouseholderQR<MatXX> qr(A_scaled);
    qr.setThreshold(1.0e-15);

    MatXX X = col_scale.asDiagonal()
            * qr.solve(row_scale.asDiagonal() * B);
    for (int refinement = 0; refinement < 2; ++refinement) {
        MatXX residual = B - A * X;
        const double scale = B.norm() + A.norm() * X.norm() + 1.0e-300;
        if (residual.norm() <= 1.0e-11 * scale)
            break;
        X += col_scale.asDiagonal()
           * qr.solve(row_scale.asDiagonal() * residual);
    }
    MatXX residual = B - A * X;
    const double scale = B.norm() + A.norm() * X.norm() + 1.0e-300;
    if (!X.allFinite() || residual.norm() > 1.0e-8 * scale)
        throw std::runtime_error(
            "CVE equilibrated Schur solve is rank deficient with an incompatible right-hand side");
    return X;
}

struct FullSystemEquilibration {
    MatXX matrix;
    VecX row_scale;
    VecX col_scale;
};

static FullSystemEquilibration equilibrate_full_system(const MatXX& K) {
    if (K.rows() != K.cols() || K.rows() == 0 || !K.allFinite())
        throw std::runtime_error("CVE full-system equilibration received an invalid matrix");

    const int n = K.rows();
    FullSystemEquilibration result;
    result.row_scale.resize(n);
    for (int i = 0; i < n; ++i)
        result.row_scale(i) = 1.0 / std::max(K.row(i).cwiseAbs().maxCoeff(), 1.0e-300);

    result.matrix = result.row_scale.asDiagonal() * K;
    result.col_scale.resize(n);
    for (int j = 0; j < n; ++j)
        result.col_scale(j) = 1.0
            / std::max(result.matrix.col(j).cwiseAbs().maxCoeff(), 1.0e-300);
    result.matrix *= result.col_scale.asDiagonal();
    return result;
}

static void check_scalar_schur_pivot(double pivot, double scale, const char* name) {
    if (!std::isfinite(pivot)
        || std::abs(pivot) <= 1.0e-13 * std::max(scale, 1.0e-300))
        throw std::runtime_error(std::string("CVE singular Schur pivot: ") + name);
}

/// Solve an already-equilibrated K * x = F using dynamic Eigen (N is either 8 or 9).
static VecX schur_solve_N_once_unscaled(const MatXX& K, const VecX& F, int nl_c) {
    // 3-block partition: [sigma(6), Lambda_v(1), lambda_c(nl_c)].
    //
    // We eliminate in two Schur stages:
    //   1) eliminate sigma (6x6)
    //   2) eliminate Lambda_v (scalar)
    // leaving a small cohesive system (1x1 or 2x2).
    //
    // This preserves the physical separation between stress and viscous-flow
    // equations and avoids treating the top 7x7 block as a monolithic system.
    const int nc = nl_c;

    const int N = 7 + nc;

    // Block extraction
    MatXX K_ss = K.block(0, 0, 6, 6);
    MatXX K_sv = K.block(0, 6, 6, 1);
    MatXX K_sc = K.block(0, 7, 6, nc);

    MatXX K_vs = K.block(6, 0, 1, 6);
    double K_vv = K(6, 6);
    MatXX K_vc = K.block(6, 7, 1, nc);

    MatXX K_cs = K.block(7, 0, nc, 6);
    MatXX K_cv = K.block(7, 6, nc, 1);
    MatXX K_cc = K.block(7, 7, nc, nc);

    VecX F_s = F.segment(0, 6);
    double F_v = F(6);
    VecX F_c = F.segment(7, nc);

    VecX  invKs_Fs  = solve_equilibrated_block(K_ss, F_s);
    MatXX invKs_Ksv = solve_equilibrated_block(K_ss, K_sv);
    MatXX invKs_Ksc = solve_equilibrated_block(K_ss, K_sc);

    // First Schur reduction (eliminate sigma)
    const double eliminated_v = (K_vs * invKs_Ksv)(0, 0);
    const double S_vv = K_vv - eliminated_v;
    check_scalar_schur_pivot(
        S_vv, std::abs(K_vv) + std::abs(eliminated_v), "S_vv");
    MatXX S_vc = K_vc - K_vs * invKs_Ksc;   // 1 x nc
    MatXX S_cv = K_cv - K_cs * invKs_Ksv;   // nc x 1
    MatXX S_cc = K_cc - K_cs * invKs_Ksc;   // nc x nc

    const double rhs_v = F_v - (K_vs * invKs_Fs)(0);
    VecX rhs_c = F_c - K_cs * invKs_Fs;

    // Second Schur reduction (eliminate Lambda_v)
    MatXX T_cc = S_cc - (1.0 / S_vv) * (S_cv * S_vc);
    VecX rhs_t = rhs_c - (1.0 / S_vv) * (S_cv * rhs_v);

    VecX x_c = solve_equilibrated_block(T_cc, rhs_t);

    const double x_v = (rhs_v - (S_vc * x_c)(0)) / S_vv;
    VecX rhs_s = F_s - K_sv * x_v - K_sc * x_c;
    VecX x_s = solve_equilibrated_block(K_ss, rhs_s);

    VecX x(7 + nc);
    x.segment(0, 6)  = x_s;
    x(6)             = x_v;
    x.segment(7, nc) = x_c;
    if (!x.allFinite())
        throw std::runtime_error("CVE plane-strain Schur solve produced non-finite values");
    return x;
}

/// Solve K * x = F while retaining the physical units of both x and F.
static VecX schur_solve_N_once(const MatXX& K, const VecX& F, int nl_c) {
    if (K.rows() != F.size())
        throw std::runtime_error("CVE Schur solve received incompatible dimensions");
    if (!F.allFinite())
        throw std::runtime_error("CVE Schur solve received a non-finite right-hand side");

    const FullSystemEquilibration equilibration = equilibrate_full_system(K);
    const VecX F_eq = equilibration.row_scale.asDiagonal() * F;
    const VecX x_eq = schur_solve_N_once_unscaled(
        equilibration.matrix, F_eq, nl_c);
    const VecX x = equilibration.col_scale.asDiagonal() * x_eq;
    if (!x.allFinite())
        throw std::runtime_error("CVE plane-strain Schur solve produced non-finite values");
    return x;
}

static VecX schur_solve_N(const MatXX& K, const VecX& F, int nl_c) {
    VecX x = schur_solve_N_once(K, F, nl_c);
    for (int refinement = 0; refinement < 2; ++refinement) {
        VecX residual = F - K * x;
        const double scale = F.norm() + K.norm() * x.norm() + 1.0e-300;
        if (residual.norm() <= 1.0e-10 * scale)
            break;
        x += schur_solve_N_once(K, residual, nl_c);
    }
    VecX residual = F - K * x;
    const double scale = F.norm() + K.norm() * x.norm() + 1.0e-300;
    if (!x.allFinite() || residual.norm() > 1.0e-8 * scale)
        throw std::runtime_error("CVE plane-strain Schur solve failed residual validation");
    return x;
}

/// Compute the top 6 rows of K^{-1}.  Used to extract dsigma/deps.
/// Returns a 6×N matrix (the sigma block of K^{-1}).
///
/// The 3-block Schur structure is preserved to exploit the fact that K_ss
/// (the 6×6 stress block) is always well-conditioned at the converged state.
/// The block solves use row/column equilibration before pivoted QR solves to
/// reduce rounding error in K_ss^{-1} * K_sv and K_ss^{-1} * K_sc when the
/// elastic moduli and cohesive multipliers differ by ~10 orders of magnitude.
static MatXX schur_inverse_N_unscaled(const MatXX& K, int nl_c) {
    const int nc = nl_c;
    const int N  = 7 + nc;

    // Block extraction
    MatXX K_ss = K.block(0, 0, 6, 6);
    MatXX K_sv = K.block(0, 6, 6, 1);
    MatXX K_sc = K.block(0, 7, 6, nc);

    MatXX K_vs = K.block(6, 0, 1, 6);
    double K_vv = K(6, 6);
    MatXX K_vc = K.block(6, 7, 1, nc);

    MatXX K_cs = K.block(7, 0, nc, 6);
    MatXX K_cv = K.block(7, 6, nc, 1);
    MatXX K_cc = K.block(7, 7, nc, nc);

    // The block solves below apply their own local row/column equilibration.
    // This is useful even after full-system equilibration because the Schur
    // complements can have a different scale from K_ss.
    auto solve_ss = [&](const MatXX& rhs) -> MatXX {
        return solve_equilibrated_block(K_ss, rhs);
    };

    MatXX invKs_Ksv = solve_ss(K_sv);
    MatXX invKs_Ksc = solve_ss(K_sc);

    // First Schur reduction (eliminate sigma)
    const double eliminated_v = (K_vs * invKs_Ksv)(0, 0);
    const double S_vv = K_vv - eliminated_v;
    check_scalar_schur_pivot(
        S_vv, std::abs(K_vv) + std::abs(eliminated_v), "tangent S_vv");
    MatXX S_vc = K_vc - K_vs * invKs_Ksc;
    MatXX S_cv = K_cv - K_cs * invKs_Ksv;
    MatXX S_cc = K_cc - K_cs * invKs_Ksc;

    // Second Schur reduction (eliminate Lambda_v)
    MatXX T_cc = S_cc - (1.0 / S_vv) * (S_cv * S_vc);

    MatXX Tinv = solve_equilibrated_block(T_cc, MatXX::Identity(nc, nc));

    // Multi-RHS solve: K * X = I  →  sigma rows of K^{-1}
    MatXX Bs = MatXX::Zero(6, N);
    MatXX Bv = MatXX::Zero(1, N);
    MatXX Bc = MatXX::Zero(nc, N);
    Bs.block(0, 0, 6, 6) = MatXX::Identity(6, 6);
    Bv(0, 6) = 1.0;
    for (int i = 0; i < nc; ++i) Bc(i, 7 + i) = 1.0;

    MatXX Y_s   = solve_ss(Bs);
    MatXX rhs_v = Bv - K_vs * Y_s;
    MatXX rhs_c = Bc - K_cs * Y_s;
    MatXX rhs_t = rhs_c - (1.0 / S_vv) * (S_cv * rhs_v);

    MatXX X_c = Tinv * rhs_t;
    MatXX X_v = (rhs_v - S_vc * X_c) / S_vv;
    MatXX X_s = solve_ss(Bs - K_sv * X_v - K_sc * X_c);

    MatXX Kinv_top = X_s;
    MatXX target = MatXX::Zero(6, N);
    target.block(0, 0, 6, 6) = MatXX::Identity(6, 6);
    MatXX inverse_residual = target - Kinv_top * K;
    for (int refinement = 0; refinement < 2; ++refinement) {
        const double inverse_scale = target.norm() + Kinv_top.norm() * K.norm() + 1.0e-300;
        if (inverse_residual.norm() <= 1.0e-10 * inverse_scale)
            break;
        for (int row = 0; row < 6; ++row) {
            VecX correction = schur_solve_N(
                K.transpose(), inverse_residual.row(row).transpose(), nl_c);
            Kinv_top.row(row) += correction.transpose();
        }
        inverse_residual = target - Kinv_top * K;
    }
    const double inverse_scale = target.norm() + Kinv_top.norm() * K.norm() + 1.0e-300;
    if (!Kinv_top.allFinite() || inverse_residual.norm() > 1.0e-8 * inverse_scale)
        throw std::runtime_error("CVE plane-strain tangent Schur solve failed residual validation");
    return Kinv_top;
}

/// Compute the top 6 rows of the physical inverse of K.
static MatXX schur_inverse_N(const MatXX& K, int nl_c) {
    const FullSystemEquilibration equilibration = equilibrate_full_system(K);
    const MatXX Kinv_eq_top = schur_inverse_N_unscaled(
        equilibration.matrix, nl_c);

    VecX col_scale_top = equilibration.col_scale.head(6);
    MatXX Kinv_top = col_scale_top.asDiagonal()
                   * Kinv_eq_top
                   * equilibration.row_scale.asDiagonal();

    MatXX target = MatXX::Zero(6, K.cols());
    target.block(0, 0, 6, 6) = MatXX::Identity(6, 6);
    MatXX inverse_residual = target - Kinv_top * K;
    const double inverse_scale = target.norm() + Kinv_top.norm() * K.norm() + 1.0e-300;
    if (!Kinv_top.allFinite() || inverse_residual.norm() > 1.0e-8 * inverse_scale)
        throw std::runtime_error("CVE physical plane-strain tangent transformation failed");
    return Kinv_top;
}

static MatXX solve_scaled_sigma_block(const MatXX& K_ss, const MatXX& rhs) {
    return solve_equilibrated_block(K_ss, rhs);
}

static VecX schur_solve_plane_stress_full_once(const MatXX& K, const VecX& F, int nl_c) {
    const int nc = nl_c;
    const int z  = 7 + nc;
    const int N  = z + 1;

    VecX row_scale(N);
    for (int i = 0; i < N; ++i)
        row_scale(i) = 1.0 / std::max(K.row(i).cwiseAbs().maxCoeff(), 1.0e-300);
    MatXX K_eq = row_scale.asDiagonal() * K;
    VecX col_scale(N);
    for (int j = 0; j < N; ++j)
        col_scale(j) = 1.0 / std::max(K_eq.col(j).cwiseAbs().maxCoeff(), 1.0e-300);
    K_eq *= col_scale.asDiagonal();
    VecX F_eq = row_scale.asDiagonal() * F;

    // Treat [sigma, e_zz] as one saddle-point block. The physical plane-stress
    // row has K(e_zz,e_zz) = 0; retaining that structure here avoids forming a
    // small, cancellation-prone e_zz Schur complement.
    MatXX K_yy = MatXX::Zero(7, 7);
    K_yy.block(0, 0, 6, 6) = K_eq.block(0, 0, 6, 6);
    K_yy.block(0, 6, 6, 1) = K_eq.block(0, z, 6, 1);
    K_yy.block(6, 0, 1, 6) = K_eq.block(z, 0, 1, 6);
    K_yy(6, 6) = K_eq(z, z);

    MatXX K_yv(7, 1);
    K_yv.block(0, 0, 6, 1) = K_eq.block(0, 6, 6, 1);
    K_yv(6, 0) = K_eq(z, 6);
    MatXX K_yc(7, nc);
    K_yc.block(0, 0, 6, nc) = K_eq.block(0, 7, 6, nc);
    K_yc.block(6, 0, 1, nc) = K_eq.block(z, 7, 1, nc);

    MatXX K_vy(1, 7);
    K_vy.block(0, 0, 1, 6) = K_eq.block(6, 0, 1, 6);
    K_vy(0, 6) = K_eq(6, z);
    const double K_vv = K_eq(6, 6);
    MatXX K_vc = K_eq.block(6, 7, 1, nc);

    MatXX K_cy(nc, 7);
    K_cy.block(0, 0, nc, 6) = K_eq.block(7, 0, nc, 6);
    K_cy.block(0, 6, nc, 1) = K_eq.block(7, z, nc, 1);
    MatXX K_cv = K_eq.block(7, 6, nc, 1);
    MatXX K_cc = K_eq.block(7, 7, nc, nc);

    VecX F_y(7);
    F_y.segment(0, 6) = F_eq.segment(0, 6);
    F_y(6) = F_eq(z);
    const double F_v = F_eq(6);
    VecX F_c = F_eq.segment(7, nc);

    VecX invKyy_Fy = solve_equilibrated_block(K_yy, F_y);
    MatXX invKyy_Kyv = solve_equilibrated_block(K_yy, K_yv);
    MatXX invKyy_Kyc = solve_equilibrated_block(K_yy, K_yc);

    // Eliminate [sigma, e_zz], then Lambda_v, leaving cohesive last.
    const double eliminated_v = (K_vy * invKyy_Kyv)(0, 0);
    const double S_vv = K_vv - eliminated_v;
    check_scalar_schur_pivot(
        S_vv, std::abs(K_vv) + std::abs(eliminated_v),
        "plane-stress S_vv");
    MatXX S_vc = K_vc - K_vy * invKyy_Kyc;
    MatXX S_cv = K_cv - K_cy * invKyy_Kyv;
    MatXX S_cc = K_cc - K_cy * invKyy_Kyc;

    const double rhs_v = F_v - (K_vy * invKyy_Fy)(0);
    VecX rhs_c = F_c - K_cy * invKyy_Fy;
    MatXX P_cc = S_cc - (1.0 / S_vv) * (S_cv * S_vc);
    VecX rhs_p = rhs_c - (1.0 / S_vv) * (S_cv * rhs_v);

    VecX x_c;
    try {
        x_c = solve_equilibrated_block(P_cc, rhs_p);
    } catch (...) {
        const double regularization = 1.0e-4
            * std::max(P_cc.cwiseAbs().maxCoeff(), 1.0);
        try {
            x_c = solve_equilibrated_block(
                P_cc + regularization * MatXX::Identity(nc, nc), rhs_p);
        } catch (const std::exception& exc) {
            throw std::runtime_error(
                std::string("CVE plane-stress Newton cohesive block: ") + exc.what());
        }
    }
    const double x_v = (rhs_v - (S_vc * x_c)(0)) / S_vv;
    VecX x_y = solve_equilibrated_block(
        K_yy, F_y - K_yv * x_v - K_yc * x_c);

    VecX x_eq(N);
    x_eq.segment(0, 6) = x_y.segment(0, 6);
    x_eq(6) = x_v;
    x_eq.segment(7, nc) = x_c;
    x_eq(z) = x_y(6);
    VecX x = col_scale.asDiagonal() * x_eq;
    if (!x.allFinite())
        throw std::runtime_error("CVE plane-stress Schur solve produced non-finite values");
    return x;
}

static VecX schur_solve_plane_stress_full(const MatXX& K, const VecX& F, int nl_c) {
    VecX x = schur_solve_plane_stress_full_once(K, F, nl_c);
    for (int refinement = 0; refinement < 2; ++refinement) {
        VecX residual = F - K * x;
        const double scale = F.norm() + K.norm() * x.norm() + 1.0e-300;
        if (residual.norm() <= 1.0e-10 * scale)
            break;
        x += schur_solve_plane_stress_full_once(K, residual, nl_c);
    }
    VecX residual = F - K * x;
    const double scale = F.norm() + K.norm() * x.norm() + 1.0e-300;
    if (!x.allFinite() || residual.norm() > 1.0e-8 * scale)
        throw std::runtime_error("CVE plane-stress Schur solve failed residual validation");
    return x;
}

static MatXX schur_inverse_plane_stress_top(const MatXX& K, int nl_c) {
    const int nc = nl_c;
    const int z  = 7 + nc;
    const int N = 8 + nl_c;

    VecX row_scale(N);
    for (int i = 0; i < N; ++i)
        row_scale(i) = 1.0 / std::max(K.row(i).cwiseAbs().maxCoeff(), 1.0e-300);
    MatXX K_eq = row_scale.asDiagonal() * K;
    VecX col_scale(N);
    for (int j = 0; j < N; ++j)
        col_scale(j) = 1.0 / std::max(K_eq.col(j).cwiseAbs().maxCoeff(), 1.0e-300);
    K_eq *= col_scale.asDiagonal();

    MatXX K_yy = MatXX::Zero(7, 7);
    K_yy.block(0, 0, 6, 6) = K_eq.block(0, 0, 6, 6);
    K_yy.block(0, 6, 6, 1) = K_eq.block(0, z, 6, 1);
    K_yy.block(6, 0, 1, 6) = K_eq.block(z, 0, 1, 6);
    K_yy(6, 6) = K_eq(z, z);

    MatXX K_yv(7, 1);
    K_yv.block(0, 0, 6, 1) = K_eq.block(0, 6, 6, 1);
    K_yv(6, 0) = K_eq(z, 6);
    MatXX K_yc(7, nc);
    K_yc.block(0, 0, 6, nc) = K_eq.block(0, 7, 6, nc);
    K_yc.block(6, 0, 1, nc) = K_eq.block(z, 7, 1, nc);

    MatXX K_vy(1, 7);
    K_vy.block(0, 0, 1, 6) = K_eq.block(6, 0, 1, 6);
    K_vy(0, 6) = K_eq(6, z);
    const double K_vv = K_eq(6, 6);
    MatXX K_vc = K_eq.block(6, 7, 1, nc);

    MatXX K_cy(nc, 7);
    K_cy.block(0, 0, nc, 6) = K_eq.block(7, 0, nc, 6);
    K_cy.block(0, 6, nc, 1) = K_eq.block(7, z, nc, 1);
    MatXX K_cv = K_eq.block(7, 6, nc, 1);
    MatXX K_cc = K_eq.block(7, 7, nc, nc);

    MatXX invKyy_Kyv = solve_scaled_sigma_block(K_yy, K_yv);
    MatXX invKyy_Kyc = solve_scaled_sigma_block(K_yy, K_yc);

    // Eliminate [sigma, e_zz], then Lambda_v, leaving cohesive last.
    const double eliminated_v = (K_vy * invKyy_Kyv)(0, 0);
    const double S_vv = K_vv - eliminated_v;
    check_scalar_schur_pivot(
        S_vv, std::abs(K_vv) + std::abs(eliminated_v),
        "plane-stress tangent S_vv");
    MatXX S_vc = K_vc - K_vy * invKyy_Kyc;
    MatXX S_cv = K_cv - K_cy * invKyy_Kyv;
    MatXX S_cc = K_cc - K_cy * invKyy_Kyc;
    MatXX P_cc = S_cc - (1.0 / S_vv) * (S_cv * S_vc);

    MatXX Pinv;
    bool regularized_tangent = false;
    try {
        Pinv = solve_equilibrated_block(P_cc, MatXX::Identity(nc, nc));
    } catch (...) {
        const double regularization = 1.0e-8
            * std::max(P_cc.cwiseAbs().maxCoeff(), 1.0);
        try {
            Pinv = solve_equilibrated_block(
                P_cc + regularization * MatXX::Identity(nc, nc),
                MatXX::Identity(nc, nc));
        } catch (const std::exception& exc) {
            throw std::runtime_error(
                std::string("CVE plane-stress tangent cohesive block: ") + exc.what());
        }
        regularized_tangent = true;
    }

    // Multi-RHS solve: K * X = I  ->  sigma rows of K^{-1}
    MatXX By = MatXX::Zero(7, N);
    MatXX Bv = MatXX::Zero(1, N);
    MatXX Bc = MatXX::Zero(nc, N);
    By.block(0, 0, 6, 6) = MatXX::Identity(6, 6);
    By(6, z) = 1.0;
    Bv(0, 6) = 1.0;
    for (int i = 0; i < nc; ++i) Bc(i, 7 + i) = 1.0;

    MatXX Y_y = solve_scaled_sigma_block(K_yy, By);
    MatXX rhs_v = Bv - K_vy * Y_y;
    MatXX rhs_c = Bc - K_cy * Y_y;
    MatXX rhs_p = rhs_c - (1.0 / S_vv) * (S_cv * rhs_v);

    MatXX X_c = Pinv * rhs_p;
    MatXX X_v = (rhs_v - S_vc * X_c) / S_vv;
    MatXX X_y = solve_scaled_sigma_block(
        K_yy, By - K_yv * X_v - K_yc * X_c);
    MatXX Kinv_eq_top = X_y.block(0, 0, 6, N);
    MatXX Kinv_top = col_scale.head(6).asDiagonal()
                   * Kinv_eq_top
                   * row_scale.asDiagonal();
    MatXX target = MatXX::Zero(6, N);
    target.block(0, 0, 6, 6) = MatXX::Identity(6, 6);
    MatXX inverse_residual = target - Kinv_top * K;
    for (int refinement = 0; refinement < 2 && !regularized_tangent; ++refinement) {
        const double inverse_scale = target.norm() + Kinv_top.norm() * K.norm() + 1.0e-300;
        if (inverse_residual.norm() <= 1.0e-10 * inverse_scale)
            break;
        for (int row = 0; row < 6; ++row) {
            VecX correction = schur_solve_plane_stress_full(
                K.transpose(), inverse_residual.row(row).transpose(), nl_c);
            Kinv_top.row(row) += correction.transpose();
        }
        inverse_residual = target - Kinv_top * K;
    }
    const double inverse_scale = target.norm() + Kinv_top.norm() * K.norm() + 1.0e-300;
    if (!Kinv_top.allFinite()
        || (!regularized_tangent && inverse_residual.norm() > 1.0e-8 * inverse_scale))
        throw std::runtime_error("CVE plane-stress tangent Schur solve failed residual validation");
    return Kinv_top;
}

// ==========================================================================
//  NaN / Inf guard on result
// ==========================================================================

static bool has_nan_or_inf(const IPResult& r) {
    if (!std::isfinite(r.E_elastic)) return true;
    if (!std::isfinite(r.e_zz)) return true;
    for (int i = 0; i < 6; ++i) {
        if (!std::isfinite(r.stress(i))) return true;
        for (int j = 0; j < 6; ++j)
            if (!std::isfinite(r.dsigma_dstrain(i,j))) return true;
    }
    return false;
}

#pragma endregion

#pragma region Newton iteration
// ==========================================================================
//  Single-IP Newton iteration
// ==========================================================================

namespace {

IPResult run_newton(const MaterialParams& mp,
                    const Vec6&           total_strain,
                    const Vec6&           ep_old,
                    const VecX&           lambda_c_init,
                    double                lambda_v_init,
                    double                dt,
                    double                dam,
                    bool                  allow_previous_state_restart,
                    const VecX*           initial_state = nullptr,
                    bool                  compute_tangent = true){

    const int nl_c = mp.nl_c;
    const int N    = 7 + nl_c + (mp.plane_stress ? 1 : 0);
    const int ezz_idx = 7 + nl_c;

    // -----------------------------------------------------------------------
    // Initial state.  The default cold path is the original elastic predictor.
    // The optional warm path reuses only already-converged local multipliers,
    // then reconstructs stress from the current strain so no constitutive
    // equation or history variable is changed by the numerical choice.
    // -----------------------------------------------------------------------

    Vec6 total_strain_init = total_strain;
    if (mp.plane_stress) {
        total_strain_init(2) = ep_old(2)
                             - mp.lame / (mp.lame + 2.0 * mp.shear)
                             * ((total_strain(0) - ep_old(0))
                              + (total_strain(1) - ep_old(1)));
    }

    Vec6   sigma_trial     = mp.D_el * (total_strain_init - ep_old);

    VecX state(N);
    auto set_elastic_predictor = [&]() {
        state.segment(0, 6) = sigma_trial;
        state(6)            = 0.0;
        for (int i = 0; i < nl_c; ++i)
            state(7 + i) = LAMBDA_COH_MIN;
        if (mp.plane_stress)
            state(ezz_idx) = total_strain_init(2);
    };

    constexpr double RESTART_BOUND_TOL = 1.0e-14;
    constexpr double RESTART_STATE_TOL = 1.0e-16;
    bool restart_from_previous_state =
        initial_state == nullptr
        && allow_previous_state_restart
        && RESTART_FROM_PREVIOUS_CONVERGED_STATE
        && lambda_c_init.size() == nl_c
        && lambda_c_init.allFinite()
        && std::isfinite(lambda_v_init)
        && lambda_v_init >= -RESTART_BOUND_TOL
        && (lambda_c_init.array() >= LAMBDA_COH_MIN - RESTART_BOUND_TOL).all()
        && (lambda_v_init > RESTART_STATE_TOL
            || (lambda_c_init.array()
                > LAMBDA_COH_MIN + RESTART_STATE_TOL).any());

    if (initial_state != nullptr) {
        if (initial_state->size() != N || !initial_state->allFinite())
            throw std::runtime_error("CVE continuation supplied an invalid initial state");
        state = *initial_state;
    } else if (restart_from_previous_state) {
        state(6) = std::max(lambda_v_init, 0.0);
        for (int i = 0; i < nl_c; ++i)
            state(7 + i) = std::max(lambda_c_init(i), LAMBDA_COH_MIN);

        // A few fixed-point predictor sweeps make stress compatible with the
        // reused viscous/cohesive multipliers before the unchanged Newton
        // system is assembled.  Plane stress additionally corrects e_zz.
        Vec6 sigma_guess = sigma_trial;
        Vec6 strain_guess = total_strain_init;
        for (int predictor = 0; predictor < 3; ++predictor) {
            const YieldFunc yf_guess = compute_yield(sigma_guess, mp.true_2d);
            const Vec6 strain_visc_el_guess =
                strain_guess - ep_old - state(6) * yf_guess.df;
            const EtaResult eta_guess = compute_eta(
                nl_c, strain_visc_el_guess, state.segment(7, nl_c), mp.true_2d);
            sigma_guess = mp.D_el * (strain_visc_el_guess - eta_guess.eta);
            if (mp.plane_stress)
                strain_guess(2) -= sigma_guess(2) / mp.D_el(2, 2);
        }

        const YieldFunc yf_guess = compute_yield(sigma_guess, mp.true_2d);
        const Vec6 strain_visc_el_guess =
            strain_guess - ep_old - state(6) * yf_guess.df;
        const EtaResult eta_guess = compute_eta(
            nl_c, strain_visc_el_guess, state.segment(7, nl_c), mp.true_2d);
        sigma_guess = mp.D_el * (strain_visc_el_guess - eta_guess.eta);

        if (sigma_guess.allFinite() && strain_guess.allFinite()) {
            state.segment(0, 6) = sigma_guess;
            if (mp.plane_stress)
                state(ezz_idx) = strain_guess(2);
        } else {
            restart_from_previous_state = false;
            set_elastic_predictor();
        }
    } else {
        set_elastic_predictor();
    }

    // "r2_2L_PSD": a closed point whose stress exceeds the onset gauge starts
    // from the solution along its onset direction. From eta = 0 the Newton
    // iteration cannot be used: the regularised potential has the curvature
    // dam Q / eps there, far above the elastic stiffness, so the steps stay
    // of the size of the regularisation and the iterates do not leave it.
    if (shared_kernels::is_opening_only(mp.failureType)
        && !(state.segment(7, nl_c).array() > LAMBDA_COH_MIN + 1.0e-14).any()) {
        const Vec6 sigma_start = state.segment(0, 6);
        Vec6 strain_start = total_strain;
        if (mp.plane_stress)
            strain_start(2) = state(ezz_idx);
        const YieldFunc yf_start = compute_yield(sigma_start, mp.true_2d);
        const EtaResult eta_start = compute_eta(
            nl_c, strain_start - ep_old - state(6) * yf_start.df,
            state.segment(7, nl_c), mp.true_2d);
        const R2StrengthScales strengths = compute_r2_strength_scales(
            mp.f_t, mp.f_s, mp.failureType, mp.true_2d);
        const Eigen::Vector3d normal_stress = eta_start.deta_dl * sigma_start;
        const shared_kernels::OpeningOnset onset = shared_kernels::compute_opening_onset(
            normal_stress,
            strengths.f_t_vol * strengths.f_t_vol / 9.0,
            0.5 * strengths.f_s_dev * strengths.f_s_dev);
        if (onset.gauge > dam * (1.0 + 1.0e-10)) {
            const Vec6 eta_direction = eta_start.deta_dl.transpose() * onset.direction;
            const Vec6 stress_direction = mp.D_el * eta_direction;
            const double amplitude = shared_kernels::solve_opening_amplitude(
                onset.gauge, dam, eta_direction.dot(stress_direction));
            if (std::isfinite(amplitude) && amplitude > 0.0) {
                for (int i = 0; i < nl_c; ++i)
                    state(7 + i) = std::max(amplitude * onset.direction(i), LAMBDA_COH_MIN);
                state.segment(0, 6) = sigma_start - amplitude * stress_direction;
            }
        }
    }

    const bool elastic_predictor_start =
        initial_state == nullptr && !restart_from_previous_state;

    // -----------------------------------------------------------------------
    // Newton loop
    // -----------------------------------------------------------------------
    // A convergent local problem needs far fewer than 30 iterations; a larger
    // cap (previously 500) only made the NaN fallback of a hopeless point, e.g.
    // one fed a non-physical strain by a diverging global iteration, very slow.
    constexpr int    MAX_ITERS = 30;
    constexpr double RESIDUAL_REL_TOL              = 1.0e-8;
    constexpr double STRESS_STRAIN_ABS_TOL         = 1.0e-12;
    // Lambda_v is a strain increment, so use the same absolute strain floor
    // as the stress residual. A looser floor (previously 1e-8) exceeds the
    // creep increment of small time steps: the elastic predictor (Lambda_v=0)
    // then passes as converged, and the result depends on the initial guess.
    constexpr double VISCOUS_MULTIPLIER_ABS_TOL    = STRESS_STRAIN_ABS_TOL;

    const double elastic_modulus = std::max(mp.D_el.cwiseAbs().maxCoeff(), 1.0e-30);
    const double strain_scale = std::max(total_strain.norm(), ep_old.norm());
    const double stress_scale = std::max(
        elastic_modulus * strain_scale,
        elastic_modulus * STRESS_STRAIN_ABS_TOL / RESIDUAL_REL_TOL);
    const double damaged_strength = dam * std::max(std::abs(mp.f_t), std::abs(mp.f_s));
    const double damaged_strength_scale = std::max(
        damaged_strength,
        1.0e-12 * std::max(std::abs(mp.f_t), std::abs(mp.f_s)));

    const YieldFunc yf_trial = compute_yield(sigma_trial, mp.true_2d);
    double visc_trial = mp.viscosity;
    if (visc_trial <= 0.0){
        visc_trial = 1.0e6; // Only used for error estimation, so a large value is fine when viscosity is disabled
    }

    const double lambda_v_trial = dt * std::pow(
        std::max(yf_trial.f, 0.0) / visc_trial, mp.n_exp);
    const double lambda_v_scale = std::max(
        std::max(strain_scale, lambda_v_trial),
        VISCOUS_MULTIPLIER_ABS_TOL / RESIDUAL_REL_TOL);

    // The cohesive residual changes units when a mode switches between its
    // inactive multiplier equation and active traction equation.
    auto scaled_residual = [&](const KFCR& candidate) -> VecX {
        VecX scaled = candidate.F;
        scaled.segment(0, 6) /= stress_scale;
        scaled(6) /= lambda_v_scale;
        scaled.segment(7, nl_c) /= damaged_strength_scale;
        if (mp.plane_stress)
            scaled(ezz_idx) /= stress_scale;
        return scaled;
    };

    auto residual_converged = [&](const KFCR& candidate) -> bool {
        VecX scaled = scaled_residual(candidate);
        return scaled.segment(0, 6).norm() < RESIDUAL_REL_TOL
            && std::abs(scaled(6)) < RESIDUAL_REL_TOL
            && scaled.segment(7, nl_c).norm() < RESIDUAL_REL_TOL
            && (!mp.plane_stress || std::abs(scaled(ezz_idx)) < RESIDUAL_REL_TOL);
    };

    int iterations_used        = 0;
    const char* stop_reason    = "maximum local iterations reached";

    KFCR sys = assemble_KF(mp, state, total_strain, ep_old, dt, dam);
    std::vector<bool> active_coh = sys.active_coh;
    const bool initial_state_converged = residual_converged(sys);
    const bool elastic_predictor_converged =
        elastic_predictor_start && initial_state_converged;
    bool converged = initial_state_converged;
    std::vector<double> merit_history(1, scaled_residual(sys).norm());

    for (int iter = 0; iter < MAX_ITERS && !converged; ++iter) {
        iterations_used = iter + 1;
        // Retain the artificial e_zz damping while far from the solution,
        // then recover the physical Jacobian for the final Newton polish.
        if (mp.plane_stress && scaled_residual(sys).norm() < 1.0e-3)
            sys.K(ezz_idx, ezz_idx) = 0.0;
        VecX delta;
        if (mp.plane_stress) {
            delta = -schur_solve_plane_stress_full(sys.K, sys.F, nl_c);
        } else {
            delta = -schur_solve_N(sys.K, sys.F, nl_c);
        }

        // Keep bound-active components fixed and shorten the step before any
        // other multiplier reaches its bound. This preserves the Newton
        // direction, unlike projecting every trial state after the update.
        constexpr double BOUND_TOL = 1.0e-14;
        if (state(6) <= BOUND_TOL && delta(6) < 0.0)
            delta(6) = 0.0;
        for (int i = 0; i < nl_c; ++i) {
            const int index = 7 + i;
            if (state(index) <= LAMBDA_COH_MIN + BOUND_TOL && delta(index) < 0.0)
                delta(index) = 0.0;
        }

        constexpr double BOUND_FRACTION = 0.995;
        double alpha_bound = 1.0;
        if (delta(6) < 0.0)
            alpha_bound = std::min(
                alpha_bound, BOUND_FRACTION * state(6) / -delta(6));
        for (int i = 0; i < nl_c; ++i) {
            const int index = 7 + i;
            if (delta(index) < 0.0)
                alpha_bound = std::min(
                    alpha_bound,
                    BOUND_FRACTION * (state(index) - LAMBDA_COH_MIN) / -delta(index));
        }

        // Freeze the active equations while testing this Newton direction.
        // The active set is updated, with hysteresis, after accepting a step.
        constexpr int    MAX_LS  = 16;
        constexpr double LS_BETA = 0.5;
        double res_before = scaled_residual(sys).norm();
        double merit_reference = res_before;
        const int history_begin = std::max(0, static_cast<int>(merit_history.size()) - 5);
        for (int i = history_begin; i < static_cast<int>(merit_history.size()); ++i)
            merit_reference = std::max(merit_reference, merit_history[i]);
        double alpha = std::max(0.0, std::min(1.0, alpha_bound));
        bool step_accepted = false;
        VecX accepted_state;
        for (int ls = 0; ls < MAX_LS; ++ls) {
            VecX state_trial = state + alpha * delta;
            KFCR sys_trial = assemble_KF(
                mp, state_trial, total_strain, ep_old, dt, dam,
                &active_coh, true);
            bool finite_ok = sys_trial.F.allFinite() && sys_trial.K.allFinite();
            const double trial_merit = finite_ok
                ? scaled_residual(sys_trial).norm()
                : std::numeric_limits<double>::infinity();

            if (finite_ok && (residual_converged(sys_trial)
                           || trial_merit < merit_reference)) {
                accepted_state = std::move(state_trial);
                step_accepted = true;
                break;
            }
            alpha *= LS_BETA;
        }

        if (!step_accepted) {
            stop_reason = "no finite residual-reducing projected line-search step";
            break;
        }

        state = std::move(accepted_state);
        sys = assemble_KF(
            mp, state, total_strain, ep_old, dt, dam,
            &active_coh, false);
        active_coh = sys.active_coh;

        if (!sys.F.allFinite() || !sys.K.allFinite()) {
            stop_reason = "non-finite residual or Jacobian after accepted local step";
            break;
        }

        merit_history.push_back(scaled_residual(sys).norm());
        if (residual_converged(sys)) { converged = true; break; }
    }

    // Preserve the established recovery behaviour for the existing failure
    // criteria. Split-r2 models must instead satisfy the strict residual
    // contract so an inconsistent two-mode tangent is never returned to the
    // global solver.
    if (!converged && !shared_kernels::is_split_r2(mp.failureType)
        && !shared_kernels::is_opening_only(mp.failureType)) {
        constexpr double RECOVERY_REL_TOL = 5.0e-4;
        constexpr double VISCOUS_RECOVERY_REL_TOL = 1.0e-3;
        const double cohesive_recovery_rel_tol = mp.failureType == "r2_DP"
            ? 5.0e-2
            : 2.0e-2;
        const double stress_recovery_tol = std::max(
            RECOVERY_REL_TOL * elastic_modulus * strain_scale,
            elastic_modulus * STRESS_STRAIN_ABS_TOL);
        const double cohesive_recovery_tol = std::max(
            cohesive_recovery_rel_tol * damaged_strength,
            elastic_modulus * STRESS_STRAIN_ABS_TOL);
        const double viscous_recovery_tol = std::max(
            VISCOUS_RECOVERY_REL_TOL * std::max(strain_scale, lambda_v_trial),
            VISCOUS_MULTIPLIER_ABS_TOL);
        bool recoverable = sys.F.segment(0, 6).norm() < stress_recovery_tol
                        && std::abs(sys.F(6)) < viscous_recovery_tol
                        && (!mp.plane_stress
                            || std::abs(scaled_residual(sys)(ezz_idx)) < RESIDUAL_REL_TOL);
        for (int i = 0; i < nl_c && recoverable; ++i)
            recoverable = std::abs(sys.F(7 + i)) < cohesive_recovery_tol;
        converged = recoverable;
    }

    if (!converged) {
        VecX scaled_failure = scaled_residual(sys);
        std::ostringstream failure;
        failure << "CVE Newton did not converge"
                << "\n  reason: " << stop_reason
                << "\n  iterations: " << iterations_used << "/" << MAX_ITERS
                << "\n  plane_stress: " << mp.plane_stress
                << ", damage: " << dam << ", dt: " << dt
                << "\n  total_strain: [" << total_strain.transpose() << "]"
                << "\n  strain_v_old: [" << ep_old.transpose() << "]"
                << "\n  raw residuals: sigma=" << sys.F.segment(0, 6).norm()
                << ", Lambda_v=" << std::abs(sys.F(6))
                << ", cohesive=[" << sys.F.segment(7, nl_c).transpose() << "]";
        if (mp.plane_stress)
            failure << ", sigma_zz=" << std::abs(sys.F(ezz_idx));
        failure << "\n  scaled residuals: sigma="
                << scaled_failure.segment(0, 6).norm()
                << ", Lambda_v=" << std::abs(scaled_failure(6))
                << ", cohesive=[" << scaled_failure.segment(7, nl_c).transpose() << "]";
        if (mp.plane_stress)
            failure << ", sigma_zz=" << std::abs(scaled_failure(ezz_idx));

        if (false) {
            // Current state
            Vec6   _sigma    = state.segment(0, 6);
            double _Lv       = state(6);
            VecX   _lc       = state.segment(7, nl_c);

            YieldFunc _yf    = compute_yield(_sigma, mp.true_2d);
            Vec6 _total_strain = total_strain;
            if (mp.plane_stress)
                _total_strain(2) = state(ezz_idx);
            Vec6 _sv_new     = ep_old + _Lv * _yf.df;
            Vec6 _strain_visc_el = _total_strain - _sv_new;
            EtaResult _er    = compute_eta(
                nl_c, _strain_visc_el, _lc, mp.true_2d);
            FailureSurfaceResult _fs = compute_failure_surface(
                mp.f_t, mp.f_s, mp.bulk, mp.G2_over_G1, mp.DP_phi, mp.DP_q,
                mp.failureType, _er.eta, dam, _strain_visc_el, mp.true_2d);
            Vec6 strain_el = _total_strain - ep_old - _Lv * _yf.df - _er.eta;

            std::cerr << "WARNING: CVE did not converge\n"
                    << "Inputs:\n"
                    << "  total_strain=[" << total_strain.transpose() << "]\n"
                    << "  ep_old=[" << ep_old.transpose() << "]\n"
                    << "  dam=" << dam << "\n"
                    << "\n"
                    << "Outputs (unconverged):\n"
                    << "  sigma=[" << _sigma.transpose() << "]\n"
                    << "  Lv=" << _Lv << "\n"
                    << "  lc=[" << _lc.transpose() << "]\n"
                    << "  strain_el=[" << strain_el.transpose() << "]\n"
                    << "  eta_c=[" << _er.eta.transpose() << "]\n"
                    << "  strain_visc =[" << _sv_new.transpose() << "]\n"
                    << "\n";
                    // << "Convergence (unscaled):\n"
                    // << "  |F_sigma|=" << res_sigma << "\n"
                    // << "  |F_Lv|=" << res_Lv << "\n"
                    // << "  |F_coh|=" << res_coh << "\n";

            // Extended diagnostics: per-mode analysis
            std::cerr << "\n  === Per-Mode Cohesive Diagnostics ===\n";
            for (int i = 0; i < nl_c; ++i) {
                Vec6 deta_i = _er.deta_dl.row(i).transpose();
                double dF_dl = failure_lambda_derivative(_fs, _er, i);
                double dE_dl = -_sigma.dot(deta_i);
                double drive = dF_dl + dE_dl;
                bool active_computed = (drive < 0.0);

                std::cerr << "  Mode " << i << ":\n"
                        << "    dF_dl=" << dF_dl
                        << "  dE_dl=" << dE_dl
                        << "  dF_dl+dE_dl=" << drive
                        << "\n"
                        << "  F_coh=" << sys.F(7+i)
                        << "  |F_coh|=" << std::abs(sys.F(7+i))
                        << "\n"
                        << "    lc[" << i << "]=" << _lc(i)
                        << "  LAMBDA_COH_MIN=" << LAMBDA_COH_MIN
                        << "  at_floor=" << (_lc(i) <= LAMBDA_COH_MIN * 1.1 ? "Y" : "N")
                        << "\n";
            }

            // tangent matrices:
            std::cerr << "\n  === Tangent Matrix Diagnostics ===\n";
            std::cerr << "K =\n" << sys.K << "\n";
            std::cerr << "C =\n" << sys.C << "\n";
            std::cerr << "F =\n" << sys.F.transpose() << "\n";
            const MatXX Kinv_top = mp.plane_stress
                ? schur_inverse_plane_stress_top(sys.K, nl_c)
                : schur_inverse_N(sys.K, nl_c);
            std::cerr << "K_tan = K^{-1} * C =\n" << Kinv_top * sys.C << "\n";

            // Schur complement diagnostics
            std::cerr << "  === Schur Complement Block (cohesive) ===\n";
            MatXX A  = sys.K.block(0, 0, 7,  nl_c == 1 ? 7 : 7);
            MatXX B  = sys.K.block(0, 7, 7,  nl_c);
            MatXX C  = sys.K.block(7, 0, nl_c, 7);
            MatXX D  = sys.K.block(7, 7, nl_c, nl_c);

            // Condition number of A block
            Eigen::JacobiSVD<MatXX> svd_A(A);
            double cond_A = svd_A.singularValues()(0) / svd_A.singularValues()(svd_A.singularValues().size()-1);
            std::cerr << "  A_block (7x7) cond=" << cond_A
                    << " sv_min=" << svd_A.singularValues()(svd_A.singularValues().size()-1)
                    << " sv_max=" << svd_A.singularValues()(0) << "\n";

            // Schur complement determinant / condition
            if (nl_c == 1) {
                double S = D(0,0) - C.row(0) * (A.inverse() * B.col(0));
                std::cerr << "  Schur (1x1) S(0,0)=" << S << "\n";
            } else if (nl_c == 2) {
                Eigen::PartialPivLU<MatXX> lu_A(A);
                MatXX Ainv_B = lu_A.solve(B);
                MatXX S = D - C * Ainv_B;
                double det = S(0,0)*S(1,1) - S(0,1)*S(1,0);
                std::cerr << "  Schur (2x2) det=" << det
                        << " S=[" << S(0,0) << " " << S(0,1) << "; "
                        << S(1,0) << " " << S(1,1) << "]\n";
                Eigen::JacobiSVD<MatXX> svd_S(S);
                double cond_S = svd_S.singularValues()(0) / svd_S.singularValues()(1);
                std::cerr << "  Schur cond=" << cond_S
                        << " sv_min=" << svd_S.singularValues()(1)
                        << " sv_max=" << svd_S.singularValues()(0) << "\n";
            }
            std::cerr << "\n";

            //pause for 5 seconds to allow diagnostics to be read before fallback values are used
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        // Do not return an under-converged state: it can produce
        // stress/tangent pairs that are not derivative-consistent.
        throw std::runtime_error(failure.str());
    }

    // -----------------------------------------------------------------------
    // Post-process: extract stress, strains, consistent tangent, energies
    // -----------------------------------------------------------------------
    Vec6   sigma    = state.segment(0, 6);
    double Lambda_v = state(6);
    VecX   lambda_c = state.segment(7, nl_c);
    Vec6   total_strain_fin = total_strain;
    if (mp.plane_stress)
        total_strain_fin(2) = state(ezz_idx);

    YieldFunc yf_fin  = compute_yield(sigma, mp.true_2d);
    Vec6 strain_v_new = ep_old + Lambda_v * yf_fin.df;

    // Use the same elastic+fracture reference strain as assemble_KF:
    Vec6 strain_visc_el_fin = total_strain_fin - strain_v_new;
    EtaResult er = compute_eta(
        nl_c, strain_visc_el_fin, lambda_c, mp.true_2d);
    Vec6 eta_c   = er.eta;

    Vec6 strain_el  = strain_visc_el_fin - eta_c;
    EnergyResult ener = mp.true_2d
        ? compute_elastic_energy_2d(mp.lame, mp.shear, strain_el)
        : compute_elastic_energy(mp.bulk, mp.shear, strain_el);
    FailureSurfaceResult fs = compute_failure_surface(
        mp.f_t, mp.f_s, mp.bulk, mp.G2_over_G1, mp.DP_phi, mp.DP_q,
        mp.failureType, eta_c, dam, strain_visc_el_fin, mp.true_2d);

    // Consistent tangent:  dsigma/deps = -(K^{-1})_{0:6, :} * C_{:, 0:6}
    Mat6 dsigma_dstrain;
    if (!compute_tangent) {
        dsigma_dstrain = Mat6::Zero();
    } else if (mp.plane_stress && elastic_predictor_converged) {
        dsigma_dstrain = mp.D_el
                       - (mp.D_el.col(2) * mp.D_el.row(2)) / mp.D_el(2, 2);
    } else {
        MatXX Kinv_top;
        if (mp.plane_stress) {
            MatXX tangent_K = sys.K;
            // K_zz contains artificial Newton damping. The residual
            // sigma_zz has no direct e_zz derivative, so remove that damping
            // when constructing the physical algorithmic tangent.
            tangent_K(ezz_idx, ezz_idx) = 0.0;
            Kinv_top = schur_inverse_plane_stress_top(tangent_K, nl_c);
        } else {
            Kinv_top = schur_inverse_N(sys.K, nl_c);
        }
        MatXX dsigma_dyn = Kinv_top * sys.C;
        dsigma_dstrain = -dsigma_dyn;
        if (shared_kernels::is_opening_only(mp.failureType))
            dsigma_dstrain = shared_kernels::repeated_strain_tangent(
                dsigma_dstrain, shared_kernels::compute_principal_dyads(strain_visc_el_fin),
                mp.plane_stress);
    }

    if (mp.true_2d) {
        sigma = project_true_2d(sigma, true);
        strain_v_new = project_true_2d(strain_v_new, true);
        eta_c = project_true_2d(eta_c, true);
        total_strain_fin = project_true_2d(total_strain_fin, true);
        dsigma_dstrain = InPlaneMat() * dsigma_dstrain * InPlaneMat();
    }

    IPResult res;
    res.stress         = sigma;
    res.eta_c          = eta_c;
    res.strain_v       = strain_v_new;
    res.dsigma_dstrain = dsigma_dstrain;
    res.E_elastic      = ener.E;
    res.E_psi          = fs.F_damage;
    res.lambda_v       = Lambda_v;
    res.lambda_c0      = lambda_c(0);
    res.lambda_c1      = (nl_c >= 2) ? lambda_c(1) : 0.0;
    res.lambda_c2      = (nl_c == 3) ? lambda_c(2) : 0.0;
    res.e_zz           = total_strain_fin(2);

    return res;
}

} // anonymous namespace
#pragma endregion

#pragma region Entry points
// ==========================================================================
//  Public entry point
// ==========================================================================

IPResult update_ip_state(const MaterialParams& mp,
                         const Vec6&  total_strain_in,
                         const Vec6&  eta_c_current,
                         const Vec6&  eta_c_old_in,
                         const VecX&  lambda_c_current,
                         const VecX&  lambda_c_old,
                         double       lambda_v_current,
                         const Vec6&  ep_old_in,
                         double       dt,
                         double       dam)
{
    const int nl_c = mp.nl_c;
    const Vec6 total_strain = project_true_2d(total_strain_in, mp.true_2d);
    const Vec6 eta_c_old = project_true_2d(eta_c_old_in, mp.true_2d);
    const Vec6 ep_old = project_true_2d(ep_old_in, mp.true_2d);

    // ---------------------------------------------------------------
    // Zero-strain guard
    // ---------------------------------------------------------------
    constexpr double STRAIN_TOL = 1.0e-20;
    if (total_strain.squaredNorm() < STRAIN_TOL) {
        Mat6 tangent = mp.D_el;
        if (mp.plane_stress)
            tangent -= (mp.D_el.col(2) * mp.D_el.row(2)) / mp.D_el(2, 2);

        IPResult r;
        r.stress         = Vec6::Zero();
        r.eta_c          = Vec6::Zero();
        r.strain_v       = ep_old;
        r.dsigma_dstrain = tangent;
        r.E_elastic      = 0.0;
        r.E_psi          = 0.0;
        r.lambda_v       = 0.0;
        r.lambda_c0      = 0.0;
        r.lambda_c1      = 0.0;
        r.lambda_c2      = 0.0;
        r.e_zz           = total_strain(2);
        return r;
    }

    // ---------------------------------------------------------------
    // Newton solve — catches exceptions and NaN, falls back to
    // previous time-step values so the outer FEniCS Newton can retry
    // with a smaller time step rather than crashing.
    // ---------------------------------------------------------------
    // The multipliers of "r2_2L_PSD" belong to the principal directions in
    // ascending order of principal strain. A change of that order between
    // two calls would hand each reused multiplier to another direction, so
    // rebuild the initial guess from the previous cohesive strain on the
    // current directions instead: lambda_i = <n_i . eta n_i>_+.
    VecX lambda_c_start = lambda_c_current;
    if (shared_kernels::is_opening_only(mp.failureType)
        && eta_c_current.allFinite() && total_strain.allFinite() && ep_old.allFinite()) {
        const shared_kernels::PrincipalDyads start_dyads =
            shared_kernels::compute_principal_dyads(total_strain - ep_old);
        lambda_c_start = (start_dyads.dyads * eta_c_current).cwiseMax(LAMBDA_COH_MIN);
    }

    IPResult result;
    bool use_fallback = false;
    try {
        try {
            result = run_newton(mp, total_strain, ep_old,
                                lambda_c_start, lambda_v_current, dt, dam,
                                true);
            if (has_nan_or_inf(result))
                throw std::runtime_error("direct solve produced non-finite output or tangent");
        } catch (...) {
            // Deterministic local continuation is used only as recovery. It
            // does not reuse state from an earlier global constitutive call.
            auto advance_continuation_segment = [&](
                const Vec6& start_strain,
                const Vec6& end_strain,
                double start_dt,
                double end_dt,
                double start_damage,
                double end_damage,
                VecX& continuation_state,
                bool& have_continuation_state,
                bool compute_target_tangent) -> IPResult {
                double progress = 0.0;
                const bool unchanged_segment = (end_strain - start_strain).norm() == 0.0
                                            && start_dt == end_dt
                                            && start_damage == end_damage;
                double progress_step = unchanged_segment ? 1.0 : 0.125;
                constexpr double MIN_PROGRESS_STEP = 1.0 / 65536.0;
                IPResult segment_result;

                while (progress < 1.0) {
                    const double stage_progress = std::min(1.0, progress + progress_step);
                    const bool final_stage = stage_progress >= 1.0;
                    Vec6 stage_strain = start_strain
                                      + stage_progress * (end_strain - start_strain);
                    if (mp.plane_stress)
                        stage_strain(2) = total_strain(2);
                    const double stage_dt = start_dt > 0.0 && end_dt > 0.0
                        ? std::exp(
                            (1.0 - stage_progress) * std::log(start_dt)
                            + stage_progress * std::log(end_dt))
                        : start_dt + stage_progress * (end_dt - start_dt);
                    const double stage_damage = start_damage
                                              + stage_progress * (end_damage - start_damage);

                    try {
                        segment_result = run_newton(
                            mp, stage_strain, ep_old,
                            lambda_c_current, lambda_v_current,
                            stage_dt, stage_damage,
                            false,
                            have_continuation_state ? &continuation_state : nullptr,
                            compute_target_tangent && final_stage);
                        if (has_nan_or_inf(segment_result))
                            throw std::runtime_error(
                                "strain continuation produced non-finite output or tangent");

                        continuation_state.resize(7 + nl_c + (mp.plane_stress ? 1 : 0));
                        continuation_state.segment(0, 6) = segment_result.stress;
                        continuation_state(6) = segment_result.lambda_v;
                        continuation_state(7) = segment_result.lambda_c0;
                        if (nl_c >= 2)
                            continuation_state(8) = segment_result.lambda_c1;
                        if (nl_c == 3)
                            continuation_state(9) = segment_result.lambda_c2;
                        if (mp.plane_stress)
                            continuation_state(7 + nl_c) = segment_result.e_zz;
                        have_continuation_state = true;
                        progress = stage_progress;
                        if (final_stage)
                            return segment_result;
                        progress_step = std::min(
                            1.0 - progress, progress_step * 1.5);
                    } catch (...) {
                        if (progress_step <= MIN_PROGRESS_STEP)
                            throw;
                        progress_step *= 0.5;
                    }
                }
                return segment_result;
            };

            Vec6 zero_trial_strain = ep_old;
            if (mp.plane_stress)
                zero_trial_strain(2) = total_strain(2);
            VecX continuation_state;
            bool have_continuation_state = false;
            try {
                result = advance_continuation_segment(
                    zero_trial_strain, total_strain,
                    dt, dt, dam, dam,
                    continuation_state, have_continuation_state, true);
            } catch (...) {
                continuation_state.resize(0);
                have_continuation_state = false;
                try {
                    // The coupled path is effective for compression-dominated
                    // r2_DP states, where separating the parameter ramps can
                    // cross a less favorable active-set branch.
                    result = advance_continuation_segment(
                        zero_trial_strain, total_strain,
                        0.0, dt, 1.0, dam,
                        continuation_state, have_continuation_state, true);
                } catch (...) {
                    continuation_state.resize(0);
                    have_continuation_state = false;
                    const double reduced_dt = std::max(dt * 1.0e-6, 1.0e-16);
                    advance_continuation_segment(
                        zero_trial_strain, total_strain,
                        reduced_dt, reduced_dt, 1.0, 1.0,
                        continuation_state, have_continuation_state, false);
                    advance_continuation_segment(
                        total_strain, total_strain,
                        reduced_dt, dt, 1.0, 1.0,
                        continuation_state, have_continuation_state, false);
                    result = advance_continuation_segment(
                        total_strain, total_strain,
                        dt, dt, 1.0, dam,
                        continuation_state, have_continuation_state, true);
                }
            }
        }
    } catch (const std::exception& exc) {
        static int exception_print_count = 0;
        if (exception_print_count++ < 1) {
            const std::string reason = exc.what();
            std::cout << "\nCVE return-map failure (NaN fallback)\n"
                      << "  reason: " << reason << "\n";
            if (reason.find("\n  total_strain:") == std::string::npos) {
                std::cout << "  plane_stress: " << mp.plane_stress
                          << ", damage: " << dam << ", dt: " << dt << "\n"
                          << "  total_strain: [" << total_strain.transpose() << "]\n"
                          << "  strain_v_old: [" << ep_old.transpose() << "]\n";
            }
        }
        use_fallback = true;
    } catch (...) {
        static int unknown_exception_print_count = 0;
        if (unknown_exception_print_count++ < 1) {
            std::cout << "\nCVE exception (NaN fallback)\n"
                      << "  reason: unknown non-standard exception\n"
                      << "  plane_stress: " << mp.plane_stress
                      << ", damage: " << dam << ", dt: " << dt << "\n"
                      << "  total_strain: [" << total_strain.transpose() << "]\n"
                      << "  strain_v_old: [" << ep_old.transpose() << "]\n";
        }
        use_fallback = true;
    }

    if (!use_fallback) return result;

    // std::cout << "WARNING: CVE return mapping produced NaN/exception, "
    //              "using previous time step values. dam=" << dam
    //           << " strain=[" << total_strain.transpose() << "]\n";

    Vec6 eps_el = total_strain - ep_old - eta_c_old;
    EnergyResult ener = mp.true_2d
        ? compute_elastic_energy_2d(mp.lame, mp.shear, eps_el)
        : compute_elastic_energy(mp.bulk, mp.shear, eps_el);
    FailureSurfaceResult fs = compute_failure_surface(
        mp.f_t, mp.f_s, mp.bulk, mp.G2_over_G1, mp.DP_phi, mp.DP_q,
        mp.failureType, eta_c_old, dam, total_strain, mp.true_2d);

    // multiply stresses by NaN to force time increment decrease/recovery
    IPResult r;
    r.stress         = ener.dE_dstrain * std::numeric_limits<double>::quiet_NaN();
    r.eta_c          = eta_c_old;
    r.strain_v       = ep_old;
    r.dsigma_dstrain = ener.ddE_dstrain2 * std::numeric_limits<double>::quiet_NaN();
    r.E_elastic      = ener.E;
    r.E_psi          = fs.F_damage;
    r.lambda_v       = 0.0;
    r.lambda_c0      = lambda_c_old(0);
    r.lambda_c1      = (nl_c >= 2) ? lambda_c_old(1) : 0.0;
    r.lambda_c2      = (nl_c == 3) ? lambda_c_old(2) : 0.0;
    r.e_zz           = total_strain(2);
    return r;
}

// ==========================================================================
//  KF inspection — public debug helper
// ==========================================================================

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
    double       dam)
{
    const int nl_c = mp.nl_c;
    const int N    = 7 + nl_c + (mp.plane_stress ? 1 : 0);
    const int ezz_idx = 7 + nl_c;

    // Run the Newton solve to get the converged solution.
    IPResult r = update_ip_state(mp,
                                 total_strain,
                                 eta_c_init, eta_c_old,
                                 lambda_c_init, lambda_c_old,
                                 lambda_v_init,
                                 ep_old, dt, dam);

    // Reconstruct the converged state vector [sigma, Lambda_v, lambda_c, e_zz].
    VecX state(N);
    state.segment(0, 6) = r.stress;
    state(6)            = r.lambda_v;
    state(7)            = r.lambda_c0;
    if (nl_c >= 2)
        state(8) = r.lambda_c1;
    if (nl_c == 3)
        state(9) = r.lambda_c2;
    if (mp.plane_stress)
        state(ezz_idx) = r.e_zz;

    // Assemble K, F, C at the converged state.
    return assemble_KF(mp, state, total_strain, ep_old, dt, dam);
}

#pragma endregion


} // namespace cohesive_viscoelastic
