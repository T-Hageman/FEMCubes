#pragma once
// shared/mandel_physics.h
// Shared Mandel-notation types, constants, stiffness, and elastic-energy
// helpers used by all kernels.
//
// Mandel notation convention — 6 components:
//   [e11, e22, e33, sqrt(2)*e23, sqrt(2)*e13, sqrt(2)*e12]
// The sqrt(2) factors ensure that the double-contraction A:B equals
// the plain dot product of the Mandel vectors.
//
// Fracture-specific helpers (EigenStrainDirections, EtaResult, compute_eta,
// FailureSurfaceResult, compute_failure_surface) live in fracture_relations.h,
// which includes this file.

#include <Eigen/Dense>
#include <cmath>
#include <string>
#include <stdexcept>

namespace shared_kernels {

// --------------------------------------------------------------------------
// Mandel types (shared across all kernels)
// --------------------------------------------------------------------------

static constexpr int MANDEL_DIM = 6;

using Vec6  = Eigen::Matrix<double, 6, 1>;
using Mat6  = Eigen::Matrix<double, 6, 6>;
using VecX  = Eigen::VectorXd;
using MatXX = Eigen::MatrixXd;

// --------------------------------------------------------------------------
// Module-level constants (initialised once; no per-call guard branches)
// --------------------------------------------------------------------------

/// I1 vector: tr(e) = I1vec().dot(e_mandel)
inline const Vec6& I1vec() {
    static const Vec6 v = []() {
        Vec6 x = Vec6::Zero();
        x(0)   = 1.0; x(1) = 1.0; x(2) = 1.0;
        return x;
    }();
    return v;
}

/// In-plane first-invariant selector for intrinsic 2-D Mandel vectors.
/// Components 2, 3, and 4 remain allocated but are identically zero.
inline const Vec6& I1vec2d() {
    static const Vec6 v = []() {
        Vec6 x = Vec6::Zero();
        x(0)   = 1.0; x(1) = 1.0;
        return x;
    }();
    return v;
}

/// Deviatoric projection matrix in Mandel notation (I - 1/3 * 1⊗1 on normal block).
inline const Mat6& DevMat() {
    static const Mat6 D = []() {
        Mat6 m = Mat6::Zero();
        m(0,0) =  2.0/3.0;  m(0,1) = -1.0/3.0;  m(0,2) = -1.0/3.0;
        m(1,0) = -1.0/3.0;  m(1,1) =  2.0/3.0;  m(1,2) = -1.0/3.0;
        m(2,0) = -1.0/3.0;  m(2,1) = -1.0/3.0;  m(2,2) =  2.0/3.0;
        m(3,3) = 1.0;
        m(4,4) = 1.0;
        m(5,5) = 1.0;
        return m;
    }();
    return D;
}

/// Intrinsic 2-D deviatoric projection in the retained 6-component layout.
inline const Mat6& DevMat2d() {
    static const Mat6 D = []() {
        Mat6 m = Mat6::Zero();
        m(0,0) =  0.5;  m(0,1) = -0.5;
        m(1,0) = -0.5;  m(1,1) =  0.5;
        m(5,5) = 1.0;
        return m;
    }();
    return D;
}

/// Projection onto the active in-plane Mandel components [xx, yy, xy].
inline const Mat6& InPlaneMat() {
    static const Mat6 P = []() {
        Mat6 m = Mat6::Zero();
        m(0,0) = 1.0;
        m(1,1) = 1.0;
        m(5,5) = 1.0;
        return m;
    }();
    return P;
}

/// 6x6 identity.
inline const Mat6& I6() {
    static const Mat6 m = Mat6::Identity();
    return m;
}

/// Isotropic elastic stiffness in Mandel notation.
inline Mat6 make_stiffness_mandel(double lame, double shear) {
    Mat6 D = Mat6::Zero();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            D(i, j) = lame;
    for (int i = 0; i < 6; ++i)
        D(i, i) += 2.0 * shear;
    return D;
}

/// Intrinsic 2-D isotropic stiffness embedded in the 6-component layout.
inline Mat6 make_stiffness_mandel_2d(double lame, double shear) {
    Mat6 D = Mat6::Zero();
    D(0,0) = lame + 2.0 * shear;
    D(0,1) = lame;
    D(1,0) = lame;
    D(1,1) = lame + 2.0 * shear;
    D(5,5) = 2.0 * shear;
    return D;
}

// --------------------------------------------------------------------------
// Elastic energy (isotropic, Mandel notation for stresses/strains)
// --------------------------------------------------------------------------

struct EnergyResult {
    double E;
    Vec6   dE_dstrain;
    Mat6   ddE_dstrain2;
};

inline EnergyResult compute_elastic_energy(double bulk, double shear,
                                            const Vec6& elastic_strain) {
    const Vec6& iv = I1vec();
    const Mat6& dm = DevMat();

    EnergyResult r;

    double vol   = iv.dot(elastic_strain);
    r.E            = 0.5 * bulk * vol * vol;
    r.dE_dstrain   = bulk * vol * iv;
    r.ddE_dstrain2 = bulk * iv * iv.transpose();

    r.E            += shear * elastic_strain.transpose() * dm * elastic_strain;
    r.dE_dstrain   += 2.0 * shear * (dm * elastic_strain);
    r.ddE_dstrain2 += 2.0 * shear * dm;

    return r;
}

/// Elastic energy for an intrinsic 2-D material in six-component storage.
inline EnergyResult compute_elastic_energy_2d(double lame, double shear,
                                               const Vec6& elastic_strain) {
    const Vec6& iv = I1vec2d();
    const Mat6& dm = DevMat2d();
    const double bulk = lame + shear;

    EnergyResult r;
    const double vol = iv.dot(elastic_strain);
    r.E            = 0.5 * bulk * vol * vol;
    r.dE_dstrain   = bulk * vol * iv;
    r.ddE_dstrain2 = bulk * iv * iv.transpose();

    r.E            += shear * elastic_strain.transpose() * dm * elastic_strain;
    r.dE_dstrain   += 2.0 * shear * (dm * elastic_strain);
    r.ddE_dstrain2 += 2.0 * shear * dm;

    return r;
}

} // namespace shared_kernels
