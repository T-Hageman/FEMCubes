#pragma once
/**
 * @file
 * Elastic phase-field splits in six-component Mandel notation.
 *
 * Literature references used below (the equations document this implementation):
 * [BFM00] Bourdin, Francfort and Marigo (2000), "Numerical experiments in
 *   revisited brittle fracture", J. Mech. Phys. Solids 48, 797--826.
 *   https://doi.org/10.1016/S0022-5096(99)00028-9
 * [AMM09] Amor, Marigo and Maurini (2009), "Regularized formulation of the
 *   variational brittle fracture with unilateral contact: Numerical experiments",
 *   J. Mech. Phys. Solids 57, 1209--1229.
 *   https://doi.org/10.1016/j.jmps.2009.04.011
 * [MWH10] Miehe, Welschinger and Hofacker (2010), "Thermodynamically consistent
 *   phase-field models of fracture: Variational principles and multi-field FE
 *   implementations", Int. J. Numer. Meth. Engng. 83, 1273--1311.
 *   https://doi.org/10.1002/nme.2861
 * [FRC10] Freddi and Royer-Carfagni (2010), "Regularized variational theories
 *   of fracture: A unified approach", J. Mech. Phys. Solids 58, 1154--1174.
 *   https://doi.org/10.1016/j.jmps.2010.02.010
 * [V24] Vicentini, Zolesi, Carrara, Maurini and De Lorenzis (2024), "On the
 *   energy decomposition in variational phase-field models for brittle fracture
 *   under multi-axial stress states", Int. J. Fracture 247, 291--317.
 *   https://doi.org/10.1007/s10704-024-00763-w
 *   Sections 3.1--3.3 compare VolDev, Spectral and NoTension; Appendix A.3
 *   gives NoTension expressions and Figure 4 shows its stress-space domains.
 *
 * VolOnly and direct Stress-Based clipping are defined explicitly below as
 * repository variants, not attributed to the standard splits in these papers.
 */
#include "mandel_physics.h"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <limits>

namespace shared_kernels {

enum class ElasticSplit { FullEnergy, VolDev, VolOnly, Spectral, StressBased, NoTension };

inline ElasticSplit parse_elastic_split(const std::string& name) {
    if (name == "FullEnergy") return ElasticSplit::FullEnergy;
    if (name == "VolDev") return ElasticSplit::VolDev;
    if (name == "VolOnly") return ElasticSplit::VolOnly;
    if (name == "Spectral") return ElasticSplit::Spectral;
    if (name == "Stress-Based") return ElasticSplit::StressBased;
    if (name == "NoTension") return ElasticSplit::NoTension;
    throw std::invalid_argument("Unknown CrackDrivingForce: " + name);
}

/**
 * Undegraded split components; no damage factor is applied in this header.
 * For energy splits, energy_plus/minus are psi+/psi-, stress_plus/minus are
 * their strain gradients, and tangent_plus/minus are their strain Hessians.
 * StressBased instead stores sum(<sigma0_i>_+^2) in energy_plus (stress squared,
 * NOT energy), sets energy_minus=0, and returns strain derivatives of the
 * clipped stresses. The Python fracture-driver conversion supplies its units
 * and strength normalization; see _split_elastic_common.py::_fracture_driver.
 */
struct ElasticSplitResult {
    double energy_plus;
    double energy_minus;
    Vec6 stress_plus;
    Vec6 stress_minus;
    Mat6 tangent_plus;
    Mat6 tangent_minus;
};

inline Eigen::Matrix3d split_tensor(const Vec6& v) {
    Eigen::Matrix3d a;
    a << v(0), v(5)/std::sqrt(2.0), v(4)/std::sqrt(2.0),
         v(5)/std::sqrt(2.0), v(1), v(3)/std::sqrt(2.0),
         v(4)/std::sqrt(2.0), v(3)/std::sqrt(2.0), v(2);
    return a;
}

inline Vec6 split_mandel(const Eigen::Matrix3d& a) {
    Vec6 v;
    v << a(0,0), a(1,1), a(2,2), std::sqrt(2.0)*a(1,2), std::sqrt(2.0)*a(0,2), std::sqrt(2.0)*a(0,1);
    return v;
}

/**
 * Strain- or stress-spectral derivative for these piecewise-quadratic splits.
 * This includes eigenvector rotations without differentiating eigenvectors
 * individually. At nonsmooth branch boundaries H is the selected one-sided
 * derivative supplied by elastic_split(); there is no unique classical Hessian.
 * Branch labels identify principal directions sharing the same linear response.
 * Their divided difference is the exact Hessian limit at any eigenvalue gap;
 * evaluating it directly avoids subtracting almost equal principal stresses.
 */
inline Mat6 spectral_tangent(const Eigen::VectorXd& eigenvalues,
                             const Eigen::VectorXd& principal_stress,
                             const Eigen::MatrixXd& principal_tangent,
                             const Eigen::MatrixXd& vectors,
                             const Eigen::VectorXi& branches) {
    const int d = eigenvalues.size();
    const double tol = 32.0 * std::numeric_limits<double>::epsilon() * std::max(eigenvalues.cwiseAbs().maxCoeff(), std::numeric_limits<double>::min());
    Mat6 result = Mat6::Zero();
    for (int column = 0; column < 6; ++column) {
        const Eigen::MatrixXd direction = vectors.transpose() * split_tensor(Vec6::Unit(column)).topLeftCorner(d, d) * vectors;
        Eigen::MatrixXd response = (principal_tangent * direction.diagonal()).asDiagonal();
        for (int i = 0; i < d; ++i) {
            for (int j = i + 1; j < d; ++j) {
                const double gap = eigenvalues(i) - eigenvalues(j);
                const double factor = branches(i) != branches(j) && std::abs(gap) > tol ?
                    (principal_stress(i) - principal_stress(j)) / gap
                    : 0.5 * (principal_tangent(i,i) + principal_tangent(j,j) - principal_tangent(i,j) - principal_tangent(j,i));
                response(i,j) = response(j,i) = factor * direction(i,j);
            }
        }
        Eigen::Matrix3d tensor = Eigen::Matrix3d::Zero();
        tensor.topLeftCorner(d, d) = vectors * response * vectors.transpose();
        result.col(column) = split_mandel(tensor);
    }
    return result;
}

/**
 * Evaluate a split of isotropic elastic strain energy, stress and tangent.
 *
 * Notation used in the branch comments:
 *   e = elastic strain, theta = tr(e), e_dev = e - theta*I/d,
 *   lambda = lame, mu = shear, K_d = lambda + 2*mu/d,
 *   <x>_+ = max(x,0), <x>_- = min(x,0),
 *   psi0 = lambda*theta^2/2 + mu*(e:e),
 *   sigma0 = C:e = lambda*theta*I + 2*mu*e.
 * Mandel order is [xx, yy, zz, sqrt(2)*yz, sqrt(2)*xz, sqrt(2)*xy], so vector
 * dot products equal tensor double contractions. true_2d selects intrinsic
 * d=2 elasticity, NOT plane stress; otherwise d=3 (including plane strain).
 * Plane-strain callers supply the full elastic strain, including any
 * out-of-plane inelastic contribution. No pore-pressure shift is applied here.
 *
 * For each energy split:
 *   psi- = psi0 - psi+, sigma+/- = d(psi+/-)/de, C+/- = d(sigma+/-)/de.
 * A consistent caller applies psi_g = g*psi+ + psi- and
 * sigma_g = g*sigma+ + sigma-, with tangent g*C+ + C- at fixed damage.
 * Original, fully degraded models can reuse psi+ as their fracture driver.
 * These are elastic partial derivatives, not creep algorithmic tangents;
 * implicit creep differentiation is performed by the viscoelastic caller.
 * StressBased is the non-variational exception documented in its branch.
 * Strict positivity tests select the inactive derivative at zero; no smoothing
 * is applied to the tension/compression switches.
 */
inline ElasticSplitResult elastic_split(const Vec6& strain, double lame, double shear,
                                        bool true_2d, ElasticSplit split) {
    const int d = true_2d ? 2 : 3;
    const Vec6 iv = true_2d ? I1vec2d() : I1vec();
    const Mat6 projection = true_2d ? InPlaneMat() : I6();
    const Mat6 stiffness = lame * iv * iv.transpose() + 2.0 * shear * projection;
    const Vec6 e = projection * strain;
    const Vec6 stress = stiffness * e;
    const double energy = 0.5 * e.dot(stress);
    const double volume = iv.dot(e);
    const double positive_volume = std::max(volume, 0.0);
    const double bulk = lame + 2.0 * shear / d;
    ElasticSplitResult r;
    r.energy_plus = 0.0;
    r.stress_plus.setZero();
    r.tangent_plus.setZero();

    if (split == ElasticSplit::FullEnergy) {
        /**
         * FullEnergy: isotropic degradation of the entire elastic energy,
         * as in the regularized variational formulation of [BFM00].
         *   psi+ = psi0, psi- = 0,
         *   sigma+ = sigma0, sigma- = 0, C+ = C, C- = 0.
         * There is no tension/compression distinction.
         */
        r.energy_plus = energy;
        r.stress_plus = stress;
        r.tangent_plus = stiffness;
    } else if (split == ElasticSplit::VolDev || split == ElasticSplit::VolOnly) {
        /**
         * VolOnly: repository volumetric-only variant of the decomposition
         * used in [AMM09]; unlike their VolDev model, shear is not degraded.
         *   psi+ = K_d*<theta>_+^2/2,
         *   psi- = K_d*<theta>_-^2/2 + mu*(e_dev:e_dev),
         *   sigma+ = K_d*<theta>_+*I,
         *   C+ = K_d*H(theta)*(I tensor I), H(theta)=1 if theta>0, else 0.
         * Retains all deviatoric stresses, including under pure shear.
         */
        r.energy_plus = 0.5 * bulk * positive_volume * positive_volume;
        r.stress_plus = bulk * positive_volume * iv;
        r.tangent_plus = (volume > 0.0 ? bulk : 0.0) * iv * iv.transpose();
        if (split == ElasticSplit::VolDev) {
            /**
             * VolDev: volumetric/deviatoric split of [AMM09], also [V24] Sec. 3.1.
             *   psi+ = K_d*<theta>_+^2/2 + mu*(e_dev:e_dev),
             *   psi- = K_d*<theta>_-^2/2,
             *   sigma+ = K_d*<theta>_+*I + 2*mu*e_dev,
             *   C+ = K_d*H(theta)*(I tensor I) + 2*mu*P_dev.
             * P_dev = I_sym - (I tensor I)/d; only volumetric compression
             * is retained, while all shear energy can drive damage.
             */
            const Mat6 dev = projection - iv * iv.transpose() / d;
            r.energy_plus += shear * e.dot(dev * e);
            r.stress_plus += 2.0 * shear * dev * e;
            r.tangent_plus += 2.0 * shear * dev;
        }
    } else {
        const Vec6 argument = split == ElasticSplit::StressBased ? stress : e;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(split_tensor(argument).topLeftCorner(d, d));
        if (solver.info() != Eigen::Success) throw std::runtime_error("Elastic split eigensolver failed.");
        const Eigen::VectorXd values = solver.eigenvalues();
        const Eigen::MatrixXd vectors = solver.eigenvectors();
        Eigen::VectorXd principal = Eigen::VectorXd::Zero(d);
        Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(d, d);

        if (split == ElasticSplit::NoTension) {
            /**
             * NoTension: structured-deformation split of [FRC10]; see also
             * [V24] Sec. 3.3 and Appendix A.3. Its variational definition is
             *   psi-(e) = min_{q positive semidefinite} psi0(e-q),
             *   psi+(e) = psi0(e) - psi-(e).
             * q represents relaxed opening strain. This is NOT simple
             * positive/negative clipping of the undegraded stress eigenvalues.
             *
             * Sort e_1 <= ... <= e_d; the m lowest strains are active in
             * compression. With S_m=sum_{i=1}^m e_i and
             *   lambda_m = 2*mu*lambda / (2*mu + (d-m)*lambda),
             *   psi- = mu*sum_{i=1}^m e_i^2 + lambda_m*S_m^2/2,
             *   sigma-_i = 2*mu*e_i + lambda_m*S_m for i<=m, else 0.
             * Starting with m=0, add the next direction while
             *   2*mu*e_{m+1} + lambda_m*S_m < 0.
             * Within this branch H_ij = 2*mu*delta_ij + lambda_m for i,j<=m,
             * and zero otherwise. spectral_tangent adds eigenbasis rotations.
             * The d=2 formula is the intrinsic-2D counterpart of the 3D split.
             */
            int m = 0;
            double sum = 0.0;
            for (; m < d; ++m) {
                const double relaxed_lame = 2.0 * shear * lame / (2.0 * shear + (d-m)*lame);
                if (2.0 * shear * values(m) + relaxed_lame * sum >= 0.0) break;
                sum += values(m);
            }
            if (m == d) {
                // Full compression is exactly the undamaged elastic law.
                // Spectral stress differences lose precision when principal
                // strains nearly coincide under a large hydrostatic stress.
                r.energy_plus = 0.0;
                r.stress_plus.setZero();
                r.tangent_plus.setZero();
            } else {
                const double relaxed_lame = 2.0 * shear * lame / (2.0 * shear + (d-m)*lame);
                const double compression_energy = shear * values.head(m).squaredNorm() + 0.5 * relaxed_lame * sum * sum;
                for (int i = 0; i < m; ++i) {
                    principal(i) = 2.0 * shear * values(i) + relaxed_lame * sum;
                    for (int j = 0; j < m; ++j) hessian(i,j) = relaxed_lame + (i == j ? 2.0 * shear : 0.0);
                }
                Eigen::Matrix3d compression = Eigen::Matrix3d::Zero();
                compression.topLeftCorner(d,d) = vectors * principal.asDiagonal() * vectors.transpose();
                r.energy_plus = energy - compression_energy;
                r.stress_plus = stress - split_mandel(compression);
                Eigen::VectorXi branches = Eigen::VectorXi::Zero(d);
                branches.head(m).setOnes();
                r.tangent_plus = stiffness - spectral_tangent(values, principal, hessian, vectors, branches);
            }
        } else {
            const double factor = split == ElasticSplit::Spectral ? 2.0 * shear : 1.0;
            principal = factor * values.cwiseMax(0.0);
            for (int i = 0; i < d; ++i) hessian(i,i) = values(i) > 0.0 ? factor : 0.0;
            Eigen::Matrix3d positive = Eigen::Matrix3d::Zero();
            positive.topLeftCorner(d,d) = vectors * principal.asDiagonal() * vectors.transpose();
            r.stress_plus = split_mandel(positive);
            const Eigen::VectorXi branches = (values.array() > 0.0).cast<int>();
            r.tangent_plus = spectral_tangent(values, principal, hessian, vectors, branches);
            if (split == ElasticSplit::Spectral) {
                /**
                 * Spectral: principal-STRAIN energy split of [MWH10], also
                 * [V24] Sec. 3.2. With e+ = sum_i <e_i>_+ n_i tensor n_i,
                 *   psi+ = lambda*<theta>_+^2/2 + mu*(e+:e+),
                 *   psi- = lambda*<theta>_-^2/2 + mu*(e-:e-),
                 *   sigma+ = lambda*<theta>_+*I + 2*mu*e+,
                 *   C+ = lambda*H(theta)*(I tensor I) + 2*mu*D(e+)[e].
                 * The trace bracket acts on the full trace, not tr(e+).
                 * Positive lateral strains may drive damage in compression.
                 */
                r.energy_plus = 0.5 * lame * positive_volume * positive_volume + shear * values.cwiseMax(0.0).squaredNorm();
                r.stress_plus += lame * positive_volume * iv;
                r.tangent_plus += (volume > 0.0 ? lame : 0.0) * iv * iv.transpose();
            } else {
                /**
                 * Stress-Based: direct principal-STRESS clipping used here.
                 * For sigma0 = sum_i s_i n_i tensor n_i,
                 *   sigma+ = sum_i <s_i>_+ n_i tensor n_i,
                 *   sigma- = sigma0 - sigma+, sigma_g = g*sigma+ + sigma-,
                 *   C+ = D(sigma0+)[sigma0] : C, C- = C - C+.
                 * spectral_tangent first differentiates with respect to sigma0;
                 * multiplication by stiffness below applies the strain chain rule.
                 * This law is generally non-variational with Poisson coupling,
                 * and C+ need not be symmetric. It is NOT the Spectral energy
                 * split of [MWH10], nor the NoTension construction of [FRC10].
                 *
                 * The raw driver returned in energy_plus is Q=sum_i <s_i>_+^2,
                 * not psi+. Python converts S=Q/f_t^2 to the strength driver
                 * using Gc, ell and the chosen AT formulation. energy_minus=0
                 * is a placeholder, not the energy of retained compression.
                 */
                r.energy_plus = principal.squaredNorm();
                r.tangent_plus = (r.tangent_plus * stiffness).eval();
            }
        }
    }
    r.energy_minus = split == ElasticSplit::StressBased ? 0.0 : energy - r.energy_plus;
    r.stress_minus = stress - r.stress_plus;
    r.tangent_minus = stiffness - r.tangent_plus;
    return r;
}

} // namespace shared_kernels
