#pragma once
// shared/fracture_relations.h
// Fracture-specific helpers built on top of mandel_physics.h:
//   - EigenStrainDirections / compute_eta_directions
//   - EtaResult             / compute_eta
//   - F1F2Result            / compute_F1F2     (internal helper)
//   - FailureSurfaceResult  / compute_failure_surface
//
// Included by the kernel that models cohesive fracture:
//   cohesive_viscoelastic/cohesive_viscoelastic_kernels.h
//
// Supported failure types: r1, r1_DP, r2, r2_DP, r2_2L, r2_2L_DP, r2_2L_PSD.

#include <algorithm>
#include <iostream>
#include <limits>
#include "mandel_physics.h"

namespace shared_kernels {

/// Failure types of the cohesive potentials, as listed in error messages.
inline const std::string SUPPORTED_FAILURE_TYPES =
    "r1, r1_DP, r2, r2_DP, r2_2L, r2_2L_DP, r2_2L_PSD";

// --------------------------------------------------------------------------
// Fracture strain directions (vol_dir, dev_dir, full e_dir and derivatives)
// --------------------------------------------------------------------------

struct EigenStrainDirections {
    Vec6 vol_dir;
    Vec6 dev_dir;
    Vec6 e_dir;
    Mat6 dvol_dir_de;
    Mat6 ddev_dir_de;
    Mat6 de_dir_de;
};

inline EigenStrainDirections compute_eta_directions(const Vec6& e,
                                                     bool true_2d = false) {
    const Vec6& iv = true_2d ? I1vec2d() : I1vec();
    const Mat6& dm = true_2d ? DevMat2d() : DevMat();
    const Mat6& id = true_2d ? InPlaneMat() : I6();
    const double dimension = true_2d ? 2.0 : 3.0;

    constexpr double reg     = 1.0e-20;
    // Relative threshold for the deviatoric direction: well-defined only when
    // the deviatoric component exceeds rel_eps times the total strain norm.
    constexpr double rel_eps = 1.0e-10;

    EigenStrainDirections fd;

    Vec6 e_active = id * e;
    double e_sq = e_active.squaredNorm();

    // Volumetric direction
    double tr_e = iv.dot(e);
    fd.vol_dir     = (tr_e >= 0.0 ? 1.0 : -1.0) * iv / std::sqrt(dimension);
    fd.dvol_dir_de = Mat6::Zero();

    // Deviatoric direction
    Vec6   dev_e    = dm * e;
    double dev_e_sq = dev_e.squaredNorm();

    const double direction_scale_sq = std::max(e_sq, reg);
    if (dev_e_sq > rel_eps * rel_eps * direction_scale_sq) {
        double dev_e_norm  = std::sqrt(dev_e_sq);
        fd.dev_dir         = dev_e / dev_e_norm;
        fd.ddev_dir_de     = dm / dev_e_norm
                             - (dev_e * dev_e.transpose())
                               / (dev_e_norm * dev_e_norm * dev_e_norm);
    } else {
        // Near-hydrostatic state: deviatoric direction is ill-defined.
        // Fall back to the uniaxial-tension deviatoric unit vector
        // [2,-1,-1,0,0,0]^T / sqrt(6).  Derivative is zero: dev_dir is
        // constant in this neighbourhood.
        // Explicit return types evaluate Eigen expressions while v is alive.
        static const Vec6 dev_default_3d = []() -> Vec6 {
            Vec6 v; v << 2.0, -1.0, -1.0, 0.0, 0.0, 0.0;
            return v / std::sqrt(6.0);
        }();
        static const Vec6 dev_default_2d = []() -> Vec6 {
            Vec6 v = Vec6::Zero();
            v(0) = 1.0;
            v(1) = -1.0;
            return v / std::sqrt(2.0);
        }();
        fd.dev_dir     = true_2d ? dev_default_2d : dev_default_3d;
        fd.ddev_dir_de = Mat6::Zero();
    }

    // Full-strain direction (used for nl==1 compressive branch)
    double e_norm = std::sqrt(e_sq + reg);
    fd.e_dir      = e_active / e_norm;
    fd.de_dir_de  = id / e_norm
                  - (e_active * e_active.transpose()) / (e_norm * e_norm * e_norm);

    return fd;
}

// --------------------------------------------------------------------------
// Principal dyads of a strain (eigenstrain directions of "r2_2L_PSD")
// --------------------------------------------------------------------------

struct PrincipalDyads {
    Eigen::Vector3d values;             ///< principal strains, ascending
    Eigen::Matrix<double, 3, 6> dyads;  ///< row i: Mandel vector of n_i (x) n_i
    Mat6 ddyads_de[3];                  ///< d(dyad_i)/d(strain)
    Vec6 mixed[3];                      ///< M_ij of the pairs (0,1), (0,2), (1,2)
    bool repeated[3];                   ///< pair of principal strains closer than the gap tolerance
};

/// Spectral decomposition of a Mandel strain: the dyads n_i (x) n_i of its
/// principal directions and their derivatives with respect to the strain,
///   d(n_i (x) n_i)/de = sum_{j != i} M_ij M_ij^T / (e_i - e_j),
/// with M_ij the unit Mandel vector of (n_i (x) n_j + n_j (x) n_i)/sqrt(2).
/// The dyad of a repeated principal strain is not unique; its derivative
/// with respect to the other member of the pair is then set to zero, as
/// compute_eta_directions does for an ill-defined deviatoric direction, and
/// the pair is flagged in `repeated`. A response that is an isotropic
/// function of the strain stays differentiable there: its tangent on M_ij
/// is the limit of (s_i - s_j)/(e_i - e_j), which the caller has to supply
/// (see repeated_strain_tangent).
/// A non-finite strain gives non-finite dyads instead of an exception, so
/// that a line search can reject the trial state.
inline PrincipalDyads compute_principal_dyads(const Vec6& e) {
    constexpr double reg     = 1.0e-20;
    constexpr double rel_eps = 1.0e-7;

    PrincipalDyads pd;
    if (!e.allFinite()) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        pd.values.setConstant(nan);
        pd.dyads.setConstant(nan);
        for (int i = 0; i < 3; ++i) {
            pd.ddyads_de[i].setConstant(nan);
            pd.mixed[i].setConstant(nan);
            pd.repeated[i] = false;
        }
        return pd;
    }

    const double shear_scale = std::sqrt(0.5);
    Eigen::Matrix3d tensor;
    tensor << e(0),               shear_scale * e(5), shear_scale * e(4),
              shear_scale * e(5), e(1),               shear_scale * e(3),
              shear_scale * e(4), shear_scale * e(3), e(2);

    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(tensor);
    pd.values = solver.eigenvalues();
    const Eigen::Matrix3d& n = solver.eigenvectors();

    for (int i = 0; i < 3; ++i) {
        pd.dyads.row(i) << n(0, i) * n(0, i), n(1, i) * n(1, i), n(2, i) * n(2, i),
                           std::sqrt(2.0) * n(1, i) * n(2, i),
                           std::sqrt(2.0) * n(0, i) * n(2, i),
                           std::sqrt(2.0) * n(0, i) * n(1, i);
        pd.ddyads_de[i].setZero();
    }

    const double min_gap = rel_eps * std::sqrt(std::max(e.squaredNorm(), reg));
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            const int pair = i + j - 1;
            Vec6& mixed = pd.mixed[pair];
            mixed << std::sqrt(2.0) * n(0, i) * n(0, j),
                     std::sqrt(2.0) * n(1, i) * n(1, j),
                     std::sqrt(2.0) * n(2, i) * n(2, j),
                     n(1, i) * n(2, j) + n(2, i) * n(1, j),
                     n(0, i) * n(2, j) + n(2, i) * n(0, j),
                     n(0, i) * n(1, j) + n(1, i) * n(0, j);
            const double gap = pd.values(i) - pd.values(j);
            pd.repeated[pair] = std::abs(gap) <= min_gap;
            if (pd.repeated[pair])
                continue;
            const Mat6 rotation = mixed * mixed.transpose() / gap;
            pd.ddyads_de[i] += rotation;
            pd.ddyads_de[j] -= rotation;
        }
    }

    return pd;
}

/// Tangent of an isotropic stress response at repeated principal strains.
/// `tangent` was assembled with the dyad derivatives of the repeated pairs
/// set to zero (compute_principal_dyads), which leaves M_ij an eigenvector
/// with the wrong stiffness. That stiffness is replaced by the limit of
/// (s_i - s_j)/(e_i - e_j) for e_j -> e_i: ds_i/de_i - ds_i/de_j, taken from
/// the principal block of the same tangent. `in_plane_only` skips pairs
/// with the out-of-plane direction (condensed plane-stress tangent).
inline Mat6 repeated_strain_tangent(const Mat6& tangent, const PrincipalDyads& pd, bool in_plane_only) {
    if (!pd.dyads.allFinite() || !tangent.allFinite())
        return tangent;
    const Eigen::Matrix3d principal = pd.dyads * tangent * pd.dyads.transpose();
    Mat6 corrected = tangent;
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            const int pair = i + j - 1;
            if (!pd.repeated[pair])
                continue;
            if (in_plane_only && (pd.dyads(i, 2) > 0.5 || pd.dyads(j, 2) > 0.5))
                continue;
            const Vec6& mixed = pd.mixed[pair];
            const double limit = 0.5 * (principal(i, i) + principal(j, j))
                               - 0.5 * (principal(i, j) + principal(j, i));
            const double assembled = mixed.dot(tangent * mixed);
            corrected += (limit - assembled) * mixed * mixed.transpose();
        }
    }
    return corrected;
}

// --------------------------------------------------------------------------
// Cohesive fracture strain (eta and its derivatives w.r.t. strain & lambda)
// --------------------------------------------------------------------------

struct EtaResult {
    Vec6   eta;
    Mat6   deta_de;
    MatXX  deta_dl;    // nl × 6
    MatXX  d2eta_dlde; // (nl*6) × 6
    VecX   dtrace_eta_dl;    // exact d(tr(eta))/d(lambda_i)
    MatXX  d2trace_eta_dlde; // nl × 6
};

/// Two multipliers (r1, r1_DP, r2_2L, r2_2L_DP): eta = lambda_0 vol_dir + lambda_1 dev_dir.
/// One multiplier (r2, r2_DP): eta along the full strain in tension, along
/// dev_dir in compression.
///
/// Three multipliers select the opening-only parametrisation of "r2_2L_PSD":
/// eta = sum_i lambda_i n_i (x) n_i on the principal directions n_i of the
/// strain, so that lambda >= 0 spans the positive semidefinite tensors that
/// are coaxial with the strain.
inline EtaResult compute_eta(int nl, const Vec6& strain, const VecX& lambda,
                             bool true_2d = false) {
    EtaResult er;
    er.deta_dl.resize(nl, MANDEL_DIM);
    er.d2eta_dlde.resize(nl * MANDEL_DIM, MANDEL_DIM);
    er.dtrace_eta_dl = VecX::Zero(nl);
    er.d2trace_eta_dlde = MatXX::Zero(nl, MANDEL_DIM);

    if (nl == 3) {
        const PrincipalDyads pd = compute_principal_dyads(strain);

        er.eta.setZero();
        er.deta_de.setZero();
        for (int i = 0; i < 3; ++i) {
            er.eta     += lambda(i) * pd.dyads.row(i).transpose();
            er.deta_de += lambda(i) * pd.ddyads_de[i];

            er.deta_dl.row(i) = pd.dyads.row(i);
            er.dtrace_eta_dl(i) = 1.0;
            er.d2eta_dlde.block(i * MANDEL_DIM, 0, MANDEL_DIM, MANDEL_DIM) = pd.ddyads_de[i];
        }
        return er;
    }

    const Vec6& iv = true_2d ? I1vec2d() : I1vec();
    const EigenStrainDirections fd = compute_eta_directions(strain, true_2d);

    if (nl == 2) {
        er.eta     = lambda(0) * fd.vol_dir + lambda(1) * fd.dev_dir;
        er.deta_de = lambda(0) * fd.dvol_dir_de + lambda(1) * fd.ddev_dir_de;

        er.deta_dl.row(0) = fd.vol_dir.transpose();
        er.deta_dl.row(1) = fd.dev_dir.transpose();

        er.dtrace_eta_dl(0) = (iv.dot(strain) >= 0.0 ? 1.0 : -1.0)
                            * std::sqrt(true_2d ? 2.0 : 3.0);
        er.dtrace_eta_dl(1) = 0.0;

        er.d2eta_dlde.block(0 * MANDEL_DIM, 0, MANDEL_DIM, MANDEL_DIM) = fd.dvol_dir_de;
        er.d2eta_dlde.block(1 * MANDEL_DIM, 0, MANDEL_DIM, MANDEL_DIM) = fd.ddev_dir_de;
    } else {
        Vec6 active_dir;
        Mat6 d_active_dir_de;

        double trace_strain = iv.dot(strain);
        if (trace_strain < 0.0) {
            active_dir      = fd.dev_dir;
            d_active_dir_de = fd.ddev_dir_de;
        } else {
            active_dir      = fd.e_dir;
            d_active_dir_de = fd.de_dir_de;
        }

        er.eta     = lambda(0) * active_dir;
        er.deta_de = lambda(0) * d_active_dir_de;

        er.deta_dl.row(0) = active_dir.transpose();
        if (trace_strain >= 0.0) {
            er.dtrace_eta_dl(0) = iv.dot(active_dir);
            er.d2trace_eta_dlde.row(0) = iv.transpose() * d_active_dir_de;
        }

        er.d2eta_dlde.block(0, 0, MANDEL_DIM, MANDEL_DIM) = d_active_dir_de;
    }

    return er;
}

// --------------------------------------------------------------------------
// Failure surface: internal F1/F2 helper
// --------------------------------------------------------------------------

struct F1F2Result {
    double F1;
    Vec6   dF1;
    double F2;
    Vec6   dF2;
    Mat6   ddF2;
    double F1S;
    double F2_shape;
    Vec6   dF2_shape;
    Mat6   ddF2_shape;
    double eta_dev_reg;
};

inline F1F2Result compute_F1F2(double f_t, double f_s,
                                const Vec6& eta,
                                const Vec6& total_strain,
                                bool true_2d = false) {
    const Vec6& iv = true_2d ? I1vec2d() : I1vec();
    const Mat6& dm = true_2d ? DevMat2d() : DevMat();
    const double dimension = true_2d ? 2.0 : 3.0;

    constexpr double eta_dev_eps = 1.0e-10;

    F1F2Result r;

    double eta_vol = iv.dot(eta);
    r.F1  = f_t / dimension * eta_vol;
    r.dF1 = (f_t / dimension) * iv;

    Vec6   eta_dev_vec = dm * eta;
    // dm is an orthogonal deviatoric projector.  Forming eta^T dm eta can
    // nevertheless suffer cancellation for an exactly volumetric eta and
    // produce a tiny negative number.  The projected-vector norm is the
    // equivalent positive-semidefinite expression.
    double eta_dev_sq  = eta_dev_vec.squaredNorm();
    r.eta_dev_reg      = std::sqrt(eta_dev_sq + eta_dev_eps * eta_dev_eps);
    double eta_dev     = r.eta_dev_reg - eta_dev_eps;

    r.F2_shape   = std::sqrt(0.5) * eta_dev;
    r.dF2_shape  = (std::sqrt(0.5) / r.eta_dev_reg) * (dm * eta);
    r.ddF2_shape = std::sqrt(0.5) * (
                       dm / r.eta_dev_reg
                     - (eta_dev_vec * eta_dev_vec.transpose())
                       / (r.eta_dev_reg * r.eta_dev_reg * r.eta_dev_reg)
                   );

    r.F2   = f_s * r.F2_shape;
    r.dF2  = f_s * r.dF2_shape;
    r.ddF2 = f_s * r.ddF2_shape;

    r.F1S = iv.dot(total_strain);

    return r;
}

// --------------------------------------------------------------------------
// Failure surface (dispatches on failureType string)
// --------------------------------------------------------------------------

struct FailureSurfaceResult {
    double F;
    Eigen::Matrix<double, 1, 6> dF_deta;          // complete 1×6 row gradient
    Eigen::Matrix<double, 1, 6> dF_deta_regular;  // gradient before trace penalty
    double dF_penalty_dtrace;                     // exact penalty derivative w.r.t. tr(eta)
    Mat6   ddF_deta2;
    Mat6   ddF_detade;
    double F_damage;
};

/// Contract the failure-potential gradient with eigenstrain mode i without
/// first adding its large trace penalty to the regular gradient.
inline double failure_lambda_derivative(
        const FailureSurfaceResult& fs, const EtaResult& er, int i) {
    return (fs.dF_deta_regular * er.deta_dl.row(i).transpose()).value()
         + fs.dF_penalty_dtrace * er.dtrace_eta_dl(i);
}

/// Chain-rule term from the change of eigenstrain direction with strain.
inline Eigen::Matrix<double, 1, 6> failure_direction_change_derivative(
        const FailureSurfaceResult& fs, const EtaResult& er, int i) {
    Eigen::Matrix<double, 1, 6> result = fs.dF_deta_regular
        * er.d2eta_dlde.block(i * MANDEL_DIM, 0, MANDEL_DIM, MANDEL_DIM);
    result += fs.dF_penalty_dtrace * er.d2trace_eta_dlde.row(i);
    return result;
}

struct R2StrengthScales {
    double f_t_vol;
    double f_s_dev;
};

inline double r2_deviatoric_strength_scale(double f_s, bool true_2d) {
    return true_2d ? f_s : 2.0 * f_s / std::sqrt(3.0);
}

/// Split r2 criteria: the r2 norm with independent volumetric and deviatoric
/// cohesive multipliers.
inline bool is_split_r2(const std::string& failureType) {
    return failureType == "r2_2L" || failureType == "r2_2L_DP";
}

/// "r2_2L_PSD": the r2_2L potential on the opening eigenstrains
/// eta = sum_i lambda_i n_i (x) n_i, lambda_i >= 0, with n_i the principal
/// directions of the pre-cohesive strain. eta is positive semidefinite, so
/// tr(eta) >= 0 holds for either sign of the strain trace without a penalty,
/// and no direction can close: the strength domain is the r2_2L ellipse for
/// stresses whose flow direction opens every principal direction, and
/// extends without bound towards compression.
inline bool is_opening_only(const std::string& failureType) {
    return failureType == "r2_2L_PSD";
}

/// Onset of "r2_2L_PSD" at eta = 0, where its potential
/// dam sqrt(A tr(eta)^2 + B ||dev(eta)||^2) is not differentiable. With s_i
/// the normal stresses on the principal directions, the stress is admissible
/// while its gauge on the cone lambda >= 0,
///   max_{lambda >= 0} s.lambda / sqrt(lambda^T Q lambda),
///   Q = B I + (A - B/3) 1 1^T,
/// does not exceed dam. The maximiser on a face of the cone is
/// lambda ~ Q_SS^{-1} s_S; the largest value over the faces on which that
/// vector is positive is the gauge, and its face the set of directions that
/// open first. ``direction`` is that maximiser scaled to
/// lambda^T Q lambda = 1, so that s.direction equals the gauge.
struct OpeningOnset {
    double gauge = 0.0;
    bool   opens[3] = {false, false, false};
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
};

/// Regularisation of the square root in the split-r2 potentials,
/// dam (sqrt(q(eta) + eps^2) - eps), in the units of the potential.
constexpr double R2_POTENTIAL_NORM_EPS = 1.0e-6;

/// Amplitude m >= 0 of an eigenstrain m eta_hat, q(eta_hat) = 1, that makes
/// the regularised "r2_2L_PSD" energy stationary along eta_hat:
///   dam m / sqrt(m^2 + eps^2) + stiffness m = gauge,
/// with gauge the stress gauge along eta_hat and stiffness = eta_hat.D.eta_hat.
/// The left-hand side is increasing, and the root lies between
/// (gauge - dam)/stiffness and gauge/stiffness.
inline double solve_opening_amplitude(double gauge, double dam, double stiffness) {
    if (!(gauge > 0.0) || !(stiffness > 0.0) || !(dam >= 0.0))
        return 0.0;
    const double eps = R2_POTENTIAL_NORM_EPS;
    auto residual = [&](double m) {
        return dam * m / std::sqrt(m * m + eps * eps) + stiffness * m - gauge;
    };
    double upper = gauge / stiffness;
    double lower = std::max((gauge - dam) / stiffness, 0.0);
    if (lower <= 0.0) {
        // Linear range of the regularisation: dam m / eps + stiffness m <= gauge.
        lower = gauge / (dam / eps + stiffness);
    }
    // The bounds can be decades apart just above onset, where the root is
    // set by the regularisation: bisect on log(m).
    for (int iter = 0; iter < 200 && upper > lower * (1.0 + 1.0e-10); ++iter) {
        const double middle = std::sqrt(lower * upper);
        if (residual(middle) > 0.0)
            upper = middle;
        else
            lower = middle;
    }
    return std::sqrt(lower * upper);
}

inline OpeningOnset compute_opening_onset(
        const Eigen::Vector3d& s, double A, double B) {
    OpeningOnset onset;
    for (int face = 1; face < 8; ++face) {
        int count = 0;
        double sum = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (face & (1 << i)) {
                ++count;
                sum += s(i);
            }
        }
        const double mean = sum / count;
        const double mean_weight = B + count * (A - B / 3.0);

        bool positive = true;
        double deviation_sq = 0.0;
        for (int i = 0; i < 3; ++i) {
            if (!(face & (1 << i)))
                continue;
            const double deviation = s(i) - mean;
            if (!(deviation / B + mean / mean_weight > 0.0))
                positive = false;
            deviation_sq += deviation * deviation;
        }
        if (!positive)
            continue;

        const double gauge = std::sqrt(
            count * mean * mean / mean_weight + deviation_sq / B);
        if (gauge > onset.gauge) {
            onset.gauge = gauge;
            for (int i = 0; i < 3; ++i) {
                onset.opens[i] = (face & (1 << i)) != 0;
                onset.direction(i) = onset.opens[i]
                    ? ((s(i) - mean) / B + mean / mean_weight) / gauge
                    : 0.0;
            }
        }
    }
    return onset;
}

inline bool uses_dp_compression_hardening(const std::string& failureType) {
    return failureType == "r1_DP" || failureType == "r2_DP" || failureType == "r2_2L_DP";
}

inline void validate_dp_parameters(
        const std::string& failureType,
        double f_s,
        double DP_phi,
        double DP_q) {
    if (!uses_dp_compression_hardening(failureType))
        return;
    if (DP_phi < 0.0)
        throw std::invalid_argument("DP_phi must be non-negative.");
    if (DP_q < f_s)
        throw std::invalid_argument("DP_q must be >= f_s.");
    if (DP_phi <= 0.0 && DP_q > f_s)
        throw std::invalid_argument(
            "DP_phi must be positive when DP_q > f_s.");
}

/// PRIVATE: Map user-facing uniaxial strengths to the r2 potential scales.
///
/// The 3-D Mandel deviatoric term gives sqrt(3)/2 times its scale in
/// uniaxial compression, whereas intrinsic 2-D gives the scale itself.  The
/// split r2_2L modes can additionally calibrate their independent volumetric
/// scale so that their uniaxial tensile strength is f_t.  The one-multiplier
/// r2 direction is strain-aligned, so its tensile response remains direction
/// and Poisson-ratio dependent and cannot be calibrated by a scalar here.
///
/// "r2_2L_PSD" uses the scales of "r2_2L". Its flow in uniaxial tension is
/// that of "r2_2L" as long as the lateral directions open with it, which
/// holds for f_t <= sqrt(2/3) f_s; f_t is then the uniaxial tensile strength.
/// f_s only shapes the tensile part of the surface: no direction can close,
/// so uniaxial compression does not yield.
inline R2StrengthScales compute_r2_strength_scales(
        double f_t, double f_s, const std::string& failureType,
        bool true_2d) {
    if (!(std::isfinite(f_t) && f_t > 0.0 && std::isfinite(f_s) && f_s > 0.0))
        throw std::invalid_argument(
            "r2 criteria require finite positive f_t and f_s strengths");

    R2StrengthScales scales;
    scales.f_s_dev = r2_deviatoric_strength_scale(f_s, true_2d);
    scales.f_t_vol = f_t;

    if (is_opening_only(failureType) && true_2d)
        throw std::invalid_argument("r2_2L_PSD is not available in intrinsic 2-D");

    if (is_split_r2(failureType) || is_opening_only(failureType)) {
        const double tensile_to_compressive = f_t / f_s;
        if (!(tensile_to_compressive < 1.0))
            throw std::invalid_argument(
                "split r2 criteria cannot match uniaxial tensile and compressive strengths "
                "unless f_s is greater than f_t");
        scales.f_t_vol = f_t / std::sqrt(
            1.0 - tensile_to_compressive * tensile_to_compressive);
        if (is_opening_only(failureType)
            && tensile_to_compressive * tensile_to_compressive > (2.0 / 3.0) * (1.0 + 1.0e-12))
            throw std::invalid_argument(
                "r2_2L_PSD requires f_t <= sqrt(2/3) f_s for f_t to be the uniaxial tensile strength");
    }

    return scales;
}

inline FailureSurfaceResult compute_failure_surface(
        double             f_t,
        double             f_s,
        double             bulk,
        double             G2_over_G1,
        double             DP_phi,
        double             DP_q,
        const std::string& failureType,
        const Vec6&        eta,
        double             dam,
        const Vec6&        total_strain,
        bool               true_2d = false) {
    const Vec6& iv       = true_2d ? I1vec2d() : I1vec();
    const Mat6& dm       = true_2d ? DevMat2d() : DevMat();
    const double dimension = true_2d ? 2.0 : 3.0;
    const F1F2Result h   = compute_F1F2(f_t, f_s, eta, total_strain, true_2d);
    const double tan_phi = std::tan(DP_phi * M_PI / 180.0);
    const std::string& ft = failureType;

    FailureSurfaceResult r;
    r.ddF_detade = Mat6::Zero();
    r.dF_penalty_dtrace = 0.0;

    if (ft == "r1") {
        if (h.F1S >= 0.0) {
            r.F        = dam * h.F1 + dam * h.F2;
            r.dF_deta_regular = (dam * h.dF1 + dam * h.dF2).transpose();
            r.F_damage = h.F1 + h.F2;
        } else {
            r.F        = -1.0e6 * h.F1 + dam * h.F2;
            r.dF_deta_regular = (dam * h.dF2).transpose();
            r.dF_penalty_dtrace = -1.0e6 * f_t / dimension;
            r.F_damage = h.F2;
        }
        r.dF_deta = r.dF_deta_regular
                  + r.dF_penalty_dtrace * iv.transpose();
        r.ddF_deta2 = dam * h.ddF2;
        return r;
    }

    if (ft == "r2" || is_split_r2(ft) || is_opening_only(ft)) {
        const R2StrengthScales strengths = compute_r2_strength_scales(f_t, f_s, ft, true_2d);
        const F1F2Result r2_h = compute_F1F2(strengths.f_t_vol, strengths.f_s_dev, eta, total_strain, true_2d);
        bool AddPenalty = false;
        double F1Copy;
        Vec6 dF1Copy;
        // "r2_2L_PSD" keeps eta positive semidefinite through its
        // eigenstrain directions (compute_eta), so that one potential holds
        // for both signs of the strain trace and no penalty is needed.
        if (r2_h.F1S<0 && !is_opening_only(ft)){
            F1Copy = 0*r2_h.F1;
            dF1Copy = 0*r2_h.dF1;
            AddPenalty = true;
        } else {
            F1Copy = r2_h.F1;
            dF1Copy = r2_h.dF1;
        }
        
        // Form F2^2 directly. Regularising F2 before squaring makes the
        // deviatoric curvature vanish at eta_dev = 0, which destabilises
        // r2_2L's independent deviatoric multiplier.
        double deviatoric_strength = strengths.f_s_dev;
        Vec6 d_deviatoric_strength_deta = Vec6::Zero();
        Vec6 d_deviatoric_strength_dstrain = Vec6::Zero();
        if (ft == "r2_2L_DP" && r2_h.F1S < 0.0) {
            // Keep branch selection on the pre-cohesive strain so eta cannot
            // switch off its own compression penalty. Confinement itself is
            // the pressure reconstructed from the elastic strain e - eta.
            const double elastic_trace = iv.dot(total_strain - eta);
            const double pressure = std::max(-bulk * elastic_trace, 0.0);
            const double confined_strength = std::min(
                f_s + pressure * tan_phi, DP_q);
            deviatoric_strength = r2_deviatoric_strength_scale(
                confined_strength, true_2d);
            if (elastic_trace < 0.0 && confined_strength < DP_q) {
                const double derivative_scale = r2_deviatoric_strength_scale(
                    bulk * tan_phi, true_2d);
                d_deviatoric_strength_deta = derivative_scale * iv;
                d_deviatoric_strength_dstrain = -derivative_scale * iv;
            }
        }

        constexpr double norm_eps = R2_POTENTIAL_NORM_EPS;
        const double dev_weight = 0.5 * deviatoric_strength * deviatoric_strength;
        const Vec6 eta_dev = dm * eta;
        const double eta_dev_sq = eta_dev.squaredNorm();
        const double F1_F2 = F1Copy * F1Copy + dev_weight * eta_dev_sq;
        const Vec6 dF2_F2 = F1Copy * dF1Copy
                          + dev_weight * eta_dev
                          + 0.5 * deviatoric_strength * eta_dev_sq
                                * d_deviatoric_strength_deta;
        const Mat6 ddF1_F2 = dF1Copy * dF1Copy.transpose()
                           + dev_weight * dm
                           + deviatoric_strength * (
                                 eta_dev * d_deviatoric_strength_deta.transpose()
                               + d_deviatoric_strength_deta * eta_dev.transpose())
                           + 0.5 * eta_dev_sq
                                 * d_deviatoric_strength_deta
                                 * d_deviatoric_strength_deta.transpose();
        const double norm_F = std::sqrt(F1_F2 + norm_eps * norm_eps);

        r.F         = dam * (norm_F - norm_eps);
        r.F_damage  = norm_F - norm_eps;
        r.dF_deta_regular = (dam * dF2_F2 / norm_F).transpose();
        r.ddF_deta2 = dam *  ddF1_F2 / norm_F
                    - dam *   (dF2_F2 * dF2_F2.transpose()) / (norm_F * norm_F * norm_F);

        if (!d_deviatoric_strength_dstrain.isZero()) {
            const Vec6 dgradient_dstrength = dam * (
                (deviatoric_strength * eta_dev
                    + 0.5 * eta_dev_sq * d_deviatoric_strength_deta) / norm_F
              - 0.5 * deviatoric_strength * eta_dev_sq * dF2_F2
                    / (norm_F * norm_F * norm_F));
            r.ddF_detade = dgradient_dstrength
                         * d_deviatoric_strength_dstrain.transpose();
        }

        if (AddPenalty){ // add penalty term for 2L version
            r.F        += -1.0e6 * r2_h.F1;
            r.dF_penalty_dtrace = -1.0e6 * strengths.f_t_vol / dimension;
        }
        r.dF_deta = r.dF_deta_regular
                  + r.dF_penalty_dtrace * iv.transpose();
        return r;
    }

    if (ft == "r2_DP") {
        if (h.F1S > 0.0) {
            double norm_F = std::sqrt(h.F1*h.F1 + h.F2*h.F2);
            double norm_F_reg = std::sqrt(h.F1*h.F1 + h.F2*h.F2 + 1.0e-20);
            r.F         = dam * norm_F;
            r.F_damage  = norm_F;
            r.dF_deta_regular = (dam / norm_F_reg * (h.F1*h.dF1 + h.F2*h.dF2)).transpose();
            r.ddF_deta2 = (dam / norm_F_reg) * (
                h.dF1*h.dF1.transpose() + h.dF2*h.dF2.transpose() + h.F2*h.ddF2
              - (1.0/(norm_F_reg*norm_F_reg))
                * (h.F1*h.dF1+h.F2*h.dF2)*(h.F1*h.dF1+h.F2*h.dF2).transpose()
            );
        } else {
            double q_dp = std::min(f_s - bulk * tan_phi * h.F1S, DP_q);
            r.F         = -1.0e12 * h.F1 + dam * q_dp * h.F2_shape;
            r.F_damage  = q_dp * h.F2_shape;
            r.dF_deta_regular = (dam * q_dp * h.dF2_shape).transpose();
            r.dF_penalty_dtrace = -1.0e12 * f_t / dimension;
            r.ddF_deta2 = dam * q_dp * h.ddF2_shape;
            if (q_dp < DP_q)
                r.ddF_detade = -(dam * bulk * tan_phi) * (h.dF2_shape * iv.transpose());
        }
        r.dF_deta = r.dF_deta_regular
                  + r.dF_penalty_dtrace * iv.transpose();
        return r;
    }

    if (ft == "r1_DP") {
        // Volumetrics
        if (h.F1S >= 0.0) {
            r.F        = dam * h.F1;
            r.dF_deta_regular = dam * h.dF1.transpose();
            r.ddF_deta2 = 0.0 * h.ddF2; // note: using h.ddF2 as a convenient zero matrix
            r.F_damage = h.F1;
        } else {
            r.F        = -1.0e6 * h.F1;
            r.dF_deta_regular = Eigen::Matrix<double, 1, 6>::Zero();
            r.dF_penalty_dtrace = -1.0e6 * f_t / dimension;
            r.ddF_deta2 = 0.0 * h.ddF2; // note: using h.ddF2 as a convenient zero matrix
            r.F_damage = 0.0 * h.F1;
        }

        // Deviatorics
        double q0 = f_s - f_t/3.0 * tan_phi;
        q0 = std::max(q0, 0.0);
        if (h.F1S*bulk >= f_t/3.0) {
            r.F        += dam * q0 * h.F2_shape;
            r.F_damage += 1.0/G2_over_G1 * q0 * h.F2_shape;
            r.dF_deta_regular += dam * q0 * h.dF2_shape.transpose();
            r.ddF_deta2+= dam * q0 * h.ddF2_shape;
        } else {
            double q_dp = std::min(f_s -  h.F1S * bulk * tan_phi, DP_q);
            r.F         += dam * q_dp * h.F2_shape;
            r.F_damage  += 1.0/G2_over_G1 * q0 * h.F2_shape;
            r.dF_deta_regular += dam * q_dp * h.dF2_shape.transpose();
            r.ddF_deta2 += dam * q_dp * h.ddF2_shape;
            if (q_dp < DP_q)
                r.ddF_detade += -(dam * bulk * tan_phi) * (h.dF2_shape * iv.transpose());
        }
        r.dF_deta = r.dF_deta_regular
                  + r.dF_penalty_dtrace * iv.transpose();
        return r;
    }

    throw std::invalid_argument(
        "Unknown FailureType '" + ft + "'; supported: " + SUPPORTED_FAILURE_TYPES);
}

// --------------------------------------------------------------------------
// Number of cohesive multipliers for a given failure type
// --------------------------------------------------------------------------

/// Return 3 for the principal-direction type (r2_2L_PSD), 2 for split
/// volumetric/deviatoric failure types (r1, r1_DP, r2_2L, r2_2L_DP), and 1
/// for single-multiplier types (r2, r2_DP). Unknown types are rejected.
inline int get_cohesive_num_lambdas(const std::string& failureType) {
    if (is_opening_only(failureType))
        return 3;
    if (failureType == "r1" || failureType == "r1_DP" || is_split_r2(failureType))
        return 2;
    if (failureType == "r2" || failureType == "r2_DP")
        return 1;
    throw std::invalid_argument(
        "Unknown FailureType '" + failureType + "'; supported: " + SUPPORTED_FAILURE_TYPES);
}

/// Multiplier values stored per integration point in the cohesive multiplier
/// fields: two for the one- and two-multiplier types, three for "r2_2L_PSD".
inline int cohesive_multiplier_slots(const std::string& failureType) {
    return std::max(2, get_cohesive_num_lambdas(failureType));
}

} // namespace shared_kernels
