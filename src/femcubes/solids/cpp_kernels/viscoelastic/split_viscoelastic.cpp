#include "split_viscoelastic.h"

namespace viscoelastic {
namespace {

struct FlowResult {
    Vec6 rate;
    Mat6 tangent;
};

FlowResult creep_flow(const MaterialParams& mp, const Vec6& stress) {
    const Mat6 dev = mp.true_2d ? shared_kernels::DevMat2d() : shared_kernels::DevMat();
    const double d = mp.true_2d ? 2.0 : 3.0;
    const Vec6 s = dev * stress;
    const double q = std::sqrt(0.5 * d * s.squaredNorm());
    FlowResult r;
    r.rate.setZero();
    r.tangent.setZero();
    if (q == 0.0) {
        if (mp.n_exp == 1.0) r.tangent = (0.5*d/mp.viscosity) * dev;
        return r;
    }
    const double coefficient = 0.5*d * std::pow(q/mp.viscosity, mp.n_exp) / q;
    r.rate = coefficient * s;
    r.tangent = coefficient * (dev + (mp.n_exp-1.0) * (s/q) * (0.5*d*s/q).transpose());
    return r;
}

} // namespace

SplitIPResult update_split_ip(const MaterialParams& mp,
                             shared_kernels::ElasticSplit split,
                             CreepCoupling creep_coupling, const Vec6& strain,
                             const Vec6& ep_old, double dt, double damage,
                             const Vec6& old_total_strain) {
    const Mat6 projection = mp.true_2d ? shared_kernels::InPlaneMat() : shared_kernels::I6();
    if (creep_coupling == CreepCoupling::PartialExplicit) {
        // Relax the committed strain implicitly, then apply the current elastic
        // increment with that creep strain fixed throughout the mechanics solve.
        SplitIPResult result = update_split_ip(mp, split, CreepCoupling::EffectiveStress,
                                               old_total_strain, ep_old, dt, damage, old_total_strain);
        const Vec6 elastic = projection * (strain - result.state.strain_p);
        result.split = shared_kernels::elastic_split(elastic, mp.lame, mp.shear, mp.true_2d, split);
        result.state.stress = damage * result.split.stress_plus + result.split.stress_minus;
        result.state.dsigma_dstrain = (damage * result.split.tangent_plus + result.split.tangent_minus) * projection;
        return result;
    }
    const Vec6 trial = projection * (strain - ep_old);
    Vec6 elastic = trial;
    SplitIPResult result;
    Mat6 mechanical_tangent, jacobian;
    Vec6 residual;
    FlowResult flow;

    auto evaluate = [&](const Vec6& e) {
        result.split = shared_kernels::elastic_split(e, mp.lame, mp.shear, mp.true_2d, split);
        result.state.stress = damage * result.split.stress_plus + result.split.stress_minus;
        mechanical_tangent = damage * result.split.tangent_plus + result.split.tangent_minus;
        Vec6 creep_stress;
        Mat6 creep_tangent;
        if (creep_coupling == CreepCoupling::DegradedStress) {
            creep_stress = result.state.stress;
            creep_tangent = mechanical_tangent;
        } else {
            const double scale = creep_coupling == CreepCoupling::FullDamagedStress ? damage : 1.0;
            creep_tangent = scale * mp.D_el;
            creep_stress = creep_tangent * e;
        }
        flow = creep_flow(mp, creep_stress);
        residual = e + dt * flow.rate - trial;
        jacobian = Mat6::Identity() + dt * flow.tangent * creep_tangent;
    };

    // Backward Euler expressed in elastic strain; algebraically equivalent to
    // the existing J2 multiplier return map, without dividing by a zero q.
    const double tolerance = 1.0e-11 * std::max(trial.norm(), 1.0e-12);
    bool converged = false;
    for (int iteration = 0; iteration < 100; ++iteration) {
        evaluate(elastic);
        const double norm = residual.norm();
        if (std::isfinite(norm) && norm <= tolerance) {
            converged = true;
            break;
        }
        if (!residual.allFinite() || !jacobian.allFinite())
            throw std::runtime_error("Nonfinite split viscoelastic local residual.");
        const Vec6 increment = jacobian.partialPivLu().solve(residual);
        if (!increment.allFinite())
            throw std::runtime_error("Singular split viscoelastic local tangent.");
        double alpha = 1.0;
        bool accepted = false;
        for (int search = 0; search < 30; ++search) {
            const Vec6 candidate = elastic - alpha * increment;
            evaluate(candidate);
            if (residual.allFinite() && residual.norm() < norm) {
                elastic = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted)
            throw std::runtime_error("Split viscoelastic local line search failed.");
    }
    if (!converged)
        throw std::runtime_error("Split viscoelastic local Newton solve did not converge.");
    result.state.strain_p = projection * (strain - elastic);
    result.state.lambda = dt * std::sqrt(2.0/(mp.true_2d ? 2.0 : 3.0)) * flow.rate.norm();
    result.state.dsigma_dstrain = mechanical_tangent * jacobian.partialPivLu().solve(projection);
    return result;
}

} // namespace viscoelastic
