// cohesive_viscoelastic_kernels_pybind.cpp
// Public API: top-level integration-point loop + pybind11 module.
//
// Exposes CohesiveViscoElasticKernels — a Python-callable class whose
// __call__ signature is consumed by the cohesive solid models — its
// variable-property variant, and point evaluations of the shared fracture
// relations (eigenstrain directions, failure potential, strength surface)
// for plotting and tests.

#include "cohesive_viscoelastic_kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace cohesive_viscoelastic {

// ==========================================================================
//  Top-level IP loop
// ==========================================================================

void update_ips_cve(const MaterialParams& mp,
                    double*       strains,
                    double        dt,
                    const double* dam,
                    double*       stress,
                    double*       eta_c,
                    const double* eta_c_old,
                    double*       strain_v,
                    const double* ep_old,
                    double*       dsigma_dstrain,
                    double*       E_elastic,
                    double*       E_psi,
                    double*       lambda_v_field,
                    double*       lambda_c_field,
                    const double* lambda_c_field_old,
                    int           n_ip)
{
    const int nl_c = mp.nl_c;
    const int lambda_stride = cohesive_multiplier_slots(mp.failureType);

    for (int ip = 0; ip < n_ip; ++ip) {
        const int base6  = ip * 6;
        const int base36 = ip * 36;

        // Read total strain
        Vec6 strain_ip;
        for (int i = 0; i < 6; ++i)
            strain_ip(i) = strains[base6 + i];

        if (mp.true_2d) {
            strain_ip(2) = 0.0;
            strain_ip(3) = 0.0;
            strain_ip(4) = 0.0;
        }

        // A point that cannot be updated (non-finite strain, or a failed local solve) gets NaN
        // outputs and keeps its internal variables, instead of an exception: an exception is
        // raised on this rank only and would leave the other ranks waiting in the next
        // collective call, whereas a NaN stress reaches the global residual on every rank, so
        // that the step is rejected and repeated collectively.
        const double nan = std::numeric_limits<double>::quiet_NaN();
        auto write_nan = [&]() {
            E_elastic[ip] = nan;
            E_psi[ip]     = nan;
            for (int i = 0; i < 6; ++i) {
                stress[base6 + i] = nan;
                for (int j = 0; j < 6; ++j)
                    dsigma_dstrain[base36 + i*6 + j] = nan;
            }
        };
        bool finite_strain = true;
        for (int i = 0; i < 6; ++i)
            finite_strain = finite_strain && std::isfinite(strain_ip(i));
        if (!finite_strain) {
            write_nan();
            continue;
        }

        // Read old plastic strain
        Vec6 ep_old_ip;
        for (int i = 0; i < 6; ++i)
            ep_old_ip(i) = ep_old[base6 + i];

        if (mp.true_2d) {
            ep_old_ip(2) = 0.0;
            ep_old_ip(3) = 0.0;
            ep_old_ip(4) = 0.0;
        }

        // Read cohesive history (cohesive_multiplier_slots per IP in the FEniCSx field)
        Vec6 eta_c_ip, eta_c_old_ip;
        for (int i = 0; i < 6; ++i) {
            eta_c_ip(i)     = eta_c[base6 + i];
            eta_c_old_ip(i) = eta_c_old[base6 + i];
        }

        if (mp.true_2d) {
            eta_c_ip(2) = eta_c_ip(3) = eta_c_ip(4) = 0.0;
            eta_c_old_ip(2) = eta_c_old_ip(3) = eta_c_old_ip(4) = 0.0;
        }

        VecX lc_cur(nl_c), lc_old(nl_c);
        for (int i = 0; i < nl_c; ++i) {
            lc_cur(i)  = lambda_c_field[ip * lambda_stride + i];
            lc_old(i)  = lambda_c_field_old[ip * lambda_stride + i];
        }

        // Clamp damage
        double d = std::max(std::min(dam[ip], 1.0), mp.kappa);

        double lv_cur = lambda_v_field[ip];

        IPResult r;
        try {
            r = update_ip_state(mp,
                                strain_ip, eta_c_ip, eta_c_old_ip,
                                lc_cur, lc_old,
                                lv_cur,
                                ep_old_ip, dt, d);
        } catch (const std::exception&) {
            write_nan();
            continue;
        }

        // Write outputs
        E_elastic[ip]     = r.E_elastic;
        E_psi[ip]         = r.E_psi;
        lambda_v_field[ip] = r.lambda_v;
        if (mp.plane_stress) {
            strains[base6 + 2] = r.e_zz;
        } else if (mp.true_2d) {
            strains[base6 + 2] = 0.0;
            strains[base6 + 3] = 0.0;
            strains[base6 + 4] = 0.0;
        }

        lambda_c_field[ip * lambda_stride]     = r.lambda_c0;
        if (nl_c >= 2)
            lambda_c_field[ip * lambda_stride + 1] = r.lambda_c1;
        if (nl_c == 3)
            lambda_c_field[ip * lambda_stride + 2] = r.lambda_c2;

        for (int i = 0; i < 6; ++i) {
            stress[base6 + i]    = r.stress(i);
            eta_c[base6 + i]     = r.eta_c(i);
            strain_v[base6 + i]  = r.strain_v(i);
            for (int j = 0; j < 6; ++j)
                dsigma_dstrain[base36 + i*6 + j] = r.dsigma_dstrain(i, j);
        }

        if (mp.true_2d) {
            for (int i : {2, 3, 4}) {
                stress[base6 + i] = 0.0;
                eta_c[base6 + i] = 0.0;
                strain_v[base6 + i] = 0.0;
                for (int j = 0; j < 6; ++j) {
                    dsigma_dstrain[base36 + i*6 + j] = 0.0;
                    dsigma_dstrain[base36 + j*6 + i] = 0.0;
                }
            }
        }
    }
}

void update_ips_cve_variable_props(
                    double        n_exp,
                    double        f_s,
                    double        G2_over_G1,
                    double        DP_phi,
                    double        DP_q,
                    double        kappa,
                    const std::string& failureType,
                    bool          plane_stress,
                    bool          true_2d,
                    double*       strains,
                    double        dt,
                    const double* dam,
                    const double* lame,
                    const double* shear,
                    const double* viscosity,
                    const double* f_t,
                    double*       stress,
                    double*       eta_c,
                    const double* eta_c_old,
                    double*       strain_v,
                    const double* ep_old,
                    double*       dsigma_dstrain,
                    double*       E_elastic,
                    double*       E_psi,
                    double*       lambda_v_field,
                    double*       lambda_c_field,
                    const double* lambda_c_field_old,
                    int           n_ip)
{
    const int lambda_stride = cohesive_multiplier_slots(failureType);

    for (int ip = 0; ip < n_ip; ++ip) {
        const double bulk = lame[ip] + 2.0 * shear[ip] / 3.0;
        MaterialParams mp = make_material_params(
            bulk, lame[ip], shear[ip], viscosity[ip], n_exp,
            f_t[ip], f_s, G2_over_G1, DP_phi, DP_q,
            kappa, failureType, plane_stress, true_2d);

        update_ips_cve(
            mp,
            strains + ip * 6, dt, dam + ip,
            stress + ip * 6,
            eta_c + ip * 6,
            eta_c_old + ip * 6,
            strain_v + ip * 6,
            ep_old + ip * 6,
            dsigma_dstrain + ip * 36,
            E_elastic + ip,
            E_psi + ip,
            lambda_v_field + ip,
            lambda_c_field + ip * lambda_stride,
            lambda_c_field_old + ip * lambda_stride,
            1);
    }
}

// ==========================================================================
//  Kernel runner class
// ==========================================================================

struct CVEKernelRunner {
    MaterialParams mp;

    CVEKernelRunner(double bulk, double lame, double shear,
                    double viscosity, double n_exp,
                    double f_t, double f_s, double G2_over_G1,
                    double DP_phi, double DP_q,
                    double kappa,
                    const std::string& failureType,
                    bool plane_stress,
                    bool true_2d)
        : mp(make_material_params(bulk, lame, shear,
                                  viscosity, n_exp,
                                  f_t, f_s, G2_over_G1, DP_phi, DP_q,
                                  kappa, failureType, plane_stress, true_2d))
    {
        shared_kernels::validate_dp_parameters(
            failureType, f_s, DP_phi, DP_q);
    }

    /// Update all IPs in-place.
    ///
    /// Array layouts (n_ip integration points, nl_c cohesive lambda count):
    ///   strains           flat (n_ip*6,)  total Mandel strain        [in/out]
    ///   dt                scalar          time-step size [s]
    ///   dam               flat (n_ip,)    elastic damage factor       [in]
    ///   stress            flat (n_ip*6,)  Cauchy stress               [out]
    ///   eta_c             flat (n_ip*6,)  cohesive fracture strain    [in/out]
    ///   eta_c_old         flat (n_ip*6,)  eta_c at previous step      [in]
    ///   strain_v          flat (n_ip*6,)  viscous (plastic) strain    [out]
    ///   ep_old            flat (n_ip*6,)  viscous strain prev. step   [in]
    ///   dsigma_dstrain    flat (n_ip*36,) consistent tangent          [out]
    ///   E_elastic         flat (n_ip,)    elastic energy density      [out]
    ///   E_psi             flat (n_ip,)    fracture energy density     [out]
    ///   lambda_v_field    flat (n_ip,)    viscous multiplier          [out]
    ///   lambda_c_field    flat (n_ip*slots,)  cohesive multipliers        [in/out]
    ///   lambda_c_field_old flat (n_ip*slots,) cohesive multipliers prev.  [in]
    /// with slots = cohesive_multiplier_slots(failureType): 3 for "r2_2L_PSD", 2 otherwise.
    void call(py::array_t<double, py::array::c_style | py::array::forcecast> strains,
              double dt,
              py::array_t<double, py::array::c_style | py::array::forcecast> dam,
              py::array_t<double, py::array::c_style | py::array::forcecast> stress,
              py::array_t<double, py::array::c_style | py::array::forcecast> eta_c,
              py::array_t<double, py::array::c_style | py::array::forcecast> eta_c_old,
              py::array_t<double, py::array::c_style | py::array::forcecast> strain_v,
              py::array_t<double, py::array::c_style | py::array::forcecast> ep_old,
              py::array_t<double, py::array::c_style | py::array::forcecast> dsigma_dstrain,
              py::array_t<double, py::array::c_style | py::array::forcecast> E_elastic,
              py::array_t<double, py::array::c_style | py::array::forcecast> E_psi,
              py::array_t<double, py::array::c_style | py::array::forcecast> lambda_v_field,
              py::array_t<double, py::array::c_style | py::array::forcecast> lambda_c_field,
              py::array_t<double, py::array::c_style | py::array::forcecast> lambda_c_field_old)
    {
        int n_ip = static_cast<int>(strains.size()) / 6;
        if (lambda_c_field.size() != n_ip * cohesive_multiplier_slots(mp.failureType)
            || lambda_c_field_old.size() != lambda_c_field.size())
            throw std::invalid_argument(
                "Cohesive multiplier fields must contain cohesive_multiplier_slots(failureType) "
                "values per integration point.");
        update_ips_cve(mp,
                       strains.mutable_data(), dt, dam.data(),
                       stress.mutable_data(),
                       eta_c.mutable_data(),
                       eta_c_old.data(),
                       strain_v.mutable_data(),
                       ep_old.data(),
                       dsigma_dstrain.mutable_data(),
                       E_elastic.mutable_data(),
                       E_psi.mutable_data(),
                       lambda_v_field.mutable_data(),
                       lambda_c_field.mutable_data(),
                       lambda_c_field_old.data(),
                       n_ip);
    }

    /// Run the Newton solve for a single IP and return the assembled (K, F, C)
    /// at the converged state as a Python dict with keys 'K', 'F', 'C'.
    ///
    /// All inputs are 1-D NumPy arrays for a single integration point:
    ///   strain        (6,)  total Mandel strain
    ///   eta_c_init    (6,)  cohesive fracture strain (initial guess)
    ///   eta_c_old     (6,)  eta_c from previous step
    ///   ep_old        (6,)  viscous strain from previous step
    ///   lambda_c_init (nl_c,)  cohesive multipliers (initial guess)
    ///   lambda_c_old  (nl_c,)  cohesive multipliers from previous step
    ///   lambda_v_init scalar   viscous multiplier (initial guess)
    ///   dt            scalar   time-step size [s]
    ///   dam           scalar   elastic damage factor
    py::dict inspect_kf(
        py::array_t<double, py::array::c_style | py::array::forcecast> strain,
        py::array_t<double, py::array::c_style | py::array::forcecast> eta_c_init,
        py::array_t<double, py::array::c_style | py::array::forcecast> eta_c_old,
        py::array_t<double, py::array::c_style | py::array::forcecast> ep_old,
        py::array_t<double, py::array::c_style | py::array::forcecast> lambda_c_init,
        py::array_t<double, py::array::c_style | py::array::forcecast> lambda_c_old,
        double lambda_v_init,
        double dt,
        double dam)
    {
        const int nl_c = mp.nl_c;
        const int N    = 7 + nl_c + (mp.plane_stress ? 1 : 0);

        Vec6 strain_v, eta_c_i, eta_c_o, ep;
        for (int i = 0; i < 6; ++i) {
            strain_v(i) = strain.data()[i];
            eta_c_i(i)  = eta_c_init.data()[i];
            eta_c_o(i)  = eta_c_old.data()[i];
            ep(i)       = ep_old.data()[i];
        }

        VecX lc_i(nl_c), lc_o(nl_c);
        for (int i = 0; i < nl_c; ++i) {
            lc_i(i) = lambda_c_init.data()[i];
            lc_o(i) = lambda_c_old.data()[i];
        }

        KFCR kfcr = inspect_kf_single_ip(mp,
                                          strain_v, eta_c_i, eta_c_o,
                                          lc_i, lc_o,
                                          lambda_v_init,
                                          ep, dt, dam);

        // Copy K (N×N), F (N), C (N×6) into numpy arrays.
        auto K_out = py::array_t<double>({N, N});
        auto F_out = py::array_t<double>({N});
        auto C_out = py::array_t<double>({N, 6});

        double* pK = K_out.mutable_data();
        double* pF = F_out.mutable_data();
        double* pC = C_out.mutable_data();

        for (int i = 0; i < N; ++i) {
            pF[i] = kfcr.F(i);
            for (int j = 0; j < N; ++j)
                pK[i * N + j] = kfcr.K(i, j);
            for (int j = 0; j < 6; ++j)
                pC[i * 6 + j] = kfcr.C(i, j);
        }

        py::dict d;
        d["K"] = K_out;
        d["F"] = F_out;
        d["C"] = C_out;
        return d;
    }
};

struct CVEVariablePropsKernelRunner {
    double n_exp;
    double f_s;
    double G2_over_G1;
    double DP_phi;
    double DP_q;
    double kappa;
    std::string failureType;
    bool plane_stress;
    bool true_2d;

    CVEVariablePropsKernelRunner(
        double n_exp_in,
        double f_s_in,
        double G2_over_G1_in,
        double DP_phi_in,
        double DP_q_in,
        double kappa_in,
        const std::string& failureType_in,
        bool plane_stress_in,
        bool true_2d_in)
        : n_exp(n_exp_in),
          f_s(f_s_in),
          G2_over_G1(G2_over_G1_in),
          DP_phi(DP_phi_in),
          DP_q(DP_q_in),
          kappa(kappa_in),
          failureType(failureType_in),
          plane_stress(plane_stress_in),
          true_2d(true_2d_in)
    {
        shared_kernels::validate_dp_parameters(
            failureType, f_s, DP_phi, DP_q);
    }

    void call(
        py::array_t<double, py::array::c_style | py::array::forcecast> strains,
        double dt,
        py::array_t<double, py::array::c_style | py::array::forcecast> dam,
        py::array_t<double, py::array::c_style | py::array::forcecast> lame,
        py::array_t<double, py::array::c_style | py::array::forcecast> shear,
        py::array_t<double, py::array::c_style | py::array::forcecast> viscosity,
        py::array_t<double, py::array::c_style | py::array::forcecast> f_t,
        py::array_t<double, py::array::c_style | py::array::forcecast> stress,
        py::array_t<double, py::array::c_style | py::array::forcecast> eta_c,
        py::array_t<double, py::array::c_style | py::array::forcecast> eta_c_old,
        py::array_t<double, py::array::c_style | py::array::forcecast> strain_v,
        py::array_t<double, py::array::c_style | py::array::forcecast> ep_old,
        py::array_t<double, py::array::c_style | py::array::forcecast> dsigma_dstrain,
        py::array_t<double, py::array::c_style | py::array::forcecast> E_elastic,
        py::array_t<double, py::array::c_style | py::array::forcecast> E_psi,
        py::array_t<double, py::array::c_style | py::array::forcecast> lambda_v_field,
        py::array_t<double, py::array::c_style | py::array::forcecast> lambda_c_field,
        py::array_t<double, py::array::c_style | py::array::forcecast> lambda_c_field_old)
    {
        int n_ip = static_cast<int>(strains.size()) / 6;
        if (dam.size() != n_ip || lame.size() != n_ip || shear.size() != n_ip ||
            viscosity.size() != n_ip || f_t.size() != n_ip)
            throw std::invalid_argument(
                "Variable cohesive material arrays must contain one value per integration point.");
        if (lambda_c_field.size() != n_ip * cohesive_multiplier_slots(failureType)
            || lambda_c_field_old.size() != lambda_c_field.size())
            throw std::invalid_argument(
                "Cohesive multiplier fields must contain cohesive_multiplier_slots(failureType) "
                "values per integration point.");

        update_ips_cve_variable_props(
            n_exp, f_s, G2_over_G1, DP_phi, DP_q, kappa,
            failureType, plane_stress, true_2d,
            strains.mutable_data(), dt, dam.data(),
            lame.data(), shear.data(), viscosity.data(), f_t.data(),
            stress.mutable_data(), eta_c.mutable_data(), eta_c_old.data(),
            strain_v.mutable_data(), ep_old.data(),
            dsigma_dstrain.mutable_data(), E_elastic.mutable_data(),
            E_psi.mutable_data(), lambda_v_field.mutable_data(),
            lambda_c_field.mutable_data(), lambda_c_field_old.data(), n_ip);
    }
};


// ==========================================================================
//  Point evaluations of the shared fracture relations
//
//  NumPy-level access to compute_eta / compute_failure_surface for plotting
//  (femcubes.solids.failure_surface_plot) and tests, without an IP loop:
//    compute_eta_point → (eta[6,1], deta_de[6,6], deta_dl[nl,6,1],
//                          d2eta_dlde[nl*6,6])
//    compute_failure_surface_point → (F, dF_deta[1,6], ddF_deta2[6,6],
//                                     ddF_detade[6,6], F_damage)
//    compute_strength_surface_point → strength indicator (0 on the surface)
// ==========================================================================

/// Evaluate eta and its derivatives at a single point.
/// strain and lambda_vals are flat 1-D arrays (length 6 and nl respectively).
py::tuple py_compute_eta_point(const std::string& failureType,
                               py::array_t<double, py::array::c_style | py::array::forcecast> strain_arr,
                               py::array_t<double, py::array::c_style | py::array::forcecast> lambda_arr,
                               bool true_2d = false)
{
    const int nl = get_cohesive_num_lambdas(failureType);
    if (lambda_arr.size() != nl)
        throw std::invalid_argument("lambda_vals must have one value per cohesive multiplier");

    Vec6 strain;
    for (int i = 0; i < 6; ++i) strain(i) = strain_arr.data()[i];
    VecX lambda(nl);
    for (int i = 0; i < nl; ++i) lambda(i) = lambda_arr.data()[i];

    const EtaResult er = compute_eta(nl, strain, lambda, true_2d);

    auto eta_out = py::array_t<double>({6, 1});
    for (int i = 0; i < 6; ++i) eta_out.mutable_data()[i] = er.eta(i);

    auto deta_de_out = py::array_t<double>({6, 6});
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            deta_de_out.mutable_data()[i*6+j] = er.deta_de(i, j);

    auto deta_dl_out = py::array_t<double>({nl, 6, 1});
    for (int k = 0; k < nl; ++k)
        for (int i = 0; i < 6; ++i)
            deta_dl_out.mutable_data()[k*6+i] = er.deta_dl(k, i);

    auto d2eta_dlde_out = py::array_t<double>({nl*6, 6});
    for (int k = 0; k < nl; ++k)
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                d2eta_dlde_out.mutable_data()[(k*6+i)*6+j] = er.d2eta_dlde(k*6+i, j);

    return py::make_tuple(eta_out, deta_de_out, deta_dl_out, d2eta_dlde_out);
}

/// Evaluate the failure potential and its derivatives at a single point.
py::tuple py_compute_failure_surface_point(
    double f_t, double f_s, double bulk, double DP_phi, double DP_q,
    const std::string& failureType,
    py::array_t<double, py::array::c_style | py::array::forcecast> eta_arr,
    double dam,
    py::array_t<double, py::array::c_style | py::array::forcecast> strain_arr,
    bool true_2d = false)
{
    Vec6 eta, strain;
    for (int i = 0; i < 6; ++i) {
        eta(i)    = eta_arr.data()[i];
        strain(i) = strain_arr.data()[i];
    }

    const FailureSurfaceResult fs = compute_failure_surface(
        f_t, f_s, bulk, 1.0, DP_phi, DP_q, failureType, eta, dam, strain, true_2d);

    auto dF_deta_out = py::array_t<double>({1, 6});
    for (int j = 0; j < 6; ++j)
        dF_deta_out.mutable_data()[j] = fs.dF_deta(0, j);

    auto ddF_deta2_out = py::array_t<double>({6, 6});
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            ddF_deta2_out.mutable_data()[i*6+j] = fs.ddF_deta2(i, j);

    auto ddF_detade_out = py::array_t<double>({6, 6});
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            ddF_detade_out.mutable_data()[i*6+j] = fs.ddF_detade(i, j);

    return py::make_tuple(fs.F, dF_deta_out, ddF_deta2_out, ddF_detade_out, fs.F_damage);
}

/// Strength indicator of a stress state with the strain it comes from:
/// the gauge of the stress on the undamaged strength surface minus one
/// (zero on the surface, positive outside).
///
/// For the one- and two-multiplier types this is the polar ratio
/// max sigma:eta / F(eta) over the admissible cohesive directions eta,
/// sampled with num_direction_samples directions; for the strain-dependent
/// DP types it is a finite-probe onset diagnostic whose pure deviatoric
/// direction gives the exact initial Drucker-Prager envelope. Unilateral
/// compression is retained through compute_failure_surface.
/// For "r2_2L_PSD" it is the exact onset gauge of the kernel's update
/// (compute_opening_onset) on the principal directions of the strain.
double py_compute_strength_surface_point(
    double f_t, double f_s, double bulk, double DP_phi, double DP_q,
    const std::string& failureType,
    py::array_t<double, py::array::c_style | py::array::forcecast> stress_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> strain_arr,
    bool true_2d = false,
    int num_direction_samples = 65)
{
    if (num_direction_samples < 2)
        throw std::invalid_argument("num_direction_samples must be at least 2");

    Vec6 stress, strain;
    for (int i = 0; i < 6; ++i) {
        stress(i) = stress_arr.data()[i];
        strain(i) = strain_arr.data()[i];
    }

    const int nl = get_cohesive_num_lambdas(failureType);

    if (is_opening_only(failureType)) {
        const R2StrengthScales strengths = compute_r2_strength_scales(f_t, f_s, failureType, true_2d);
        const PrincipalDyads pd = compute_principal_dyads(strain);
        const Eigen::Vector3d normal_stress = pd.dyads * stress;
        const OpeningOnset onset = compute_opening_onset(
            normal_stress, strengths.f_t_vol * strengths.f_t_vol / 9.0,
            0.5 * strengths.f_s_dev * strengths.f_s_dev);
        return onset.gauge - 1.0;
    }

    const double strength_scale = std::max({std::abs(f_t), std::abs(f_s), 1.0});
    const double stiffness_scale = std::max(std::abs(bulk), strength_scale);
    const double lambda_probe = std::max(strength_scale / stiffness_scale, 1.0e-8);
    const double resistance_tol = 1.0e-12 * strength_scale * lambda_probe;
    double max_ratio = -std::numeric_limits<double>::infinity();

    const auto evaluate_direction = [&](const VecX& lambda) {
        const EtaResult er = compute_eta(nl, strain, lambda, true_2d);
        const FailureSurfaceResult fs = compute_failure_surface(
            f_t, f_s, bulk, 1.0, DP_phi, DP_q, failureType,
            er.eta, 1.0, strain, true_2d);
        const double work = stress.dot(er.eta);

        if (!std::isfinite(fs.F) || !std::isfinite(work))
            throw std::runtime_error("non-finite cohesive strength-surface evaluation");
        if (fs.F <= resistance_tol) {
            if (work > resistance_tol)
                max_ratio = std::numeric_limits<double>::infinity();
            return;
        }
        max_ratio = std::max(max_ratio, work / fs.F);
    };

    if (nl == 1) {
        VecX lambda(1);
        lambda(0) = lambda_probe;
        evaluate_direction(lambda);
    } else {
        const double pi = std::acos(-1.0);
        for (int i = 0; i < num_direction_samples; ++i) {
            const double theta = 0.5 * pi * i / (num_direction_samples - 1);
            VecX lambda(2);
            // Keep the polar endpoints exact: cos(pi/2) is not zero in
            // floating point and can otherwise create a spurious, nearly
            // unresisted volumetric contribution at the deviatoric axis.
            if (i == 0) {
                lambda(0) = lambda_probe;
                lambda(1) = 0.0;
            } else if (i == num_direction_samples - 1) {
                lambda(0) = 0.0;
                lambda(1) = lambda_probe;
            } else {
                lambda(0) = lambda_probe * std::cos(theta);
                lambda(1) = lambda_probe * std::sin(theta);
            }
            evaluate_direction(lambda);
        }
    }

    if (!std::isfinite(max_ratio))
        return max_ratio > 0.0 ? max_ratio : -1.0;
    return max_ratio - 1.0;
}

} // namespace cohesive_viscoelastic

// ==========================================================================
//  pybind11 module
// ==========================================================================

PYBIND11_MODULE(_cohesive_viscoelastic_kernels_cpp, m)
{
    m.doc() = "C++ combined cohesive–visco-elastic return-mapping kernel";

    py::class_<cohesive_viscoelastic::CVEKernelRunner>(m, "CohesiveViscoElasticKernels")
        .def(py::init<double, double, double,
                      double, double,
                      double, double,
                      double, double,
                      double, double,
                      const std::string&,
                      bool, bool>(),
             py::arg("bulk"),
             py::arg("lame"),
             py::arg("shear"),
             py::arg("viscosity"),
             py::arg("n_exp"),
             py::arg("f_t"),
             py::arg("f_s"),
             py::arg("G2_over_G1"),
             py::arg("DP_phi"),
             py::arg("DP_q"),
             py::arg("kappa"),
             py::arg("failureType"),
             py::arg("plane_stress"),
             py::arg("true_2d") = false,
             "Build the combined cohesive–visco-elastic return-mapping kernel.")
        .def("__call__",
             &cohesive_viscoelastic::CVEKernelRunner::call,
             py::arg("strains"),
             py::arg("dt"),
             py::arg("dam"),
             py::arg("stress"),
             py::arg("eta_c"),
             py::arg("eta_c_old"),
             py::arg("strain_v"),
             py::arg("ep_old"),
             py::arg("dsigma_dstrain"),
             py::arg("E_elastic"),
             py::arg("E_psi"),
             py::arg("lambda_v_field"),
             py::arg("lambda_c_field"),
             py::arg("lambda_c_field_old"),
             "Update all IPs in-place. All arrays are flat NumPy arrays; "
             "strains[2] is overwritten with e_zz for plane stress and "
             "components [2, 3, 4] are zero-filled for true 2-D.")
           .def("inspect_kf",
               &cohesive_viscoelastic::CVEKernelRunner::inspect_kf,
               py::arg("strain"),
               py::arg("eta_c_init"),
               py::arg("eta_c_old"),
               py::arg("ep_old"),
               py::arg("lambda_c_init"),
               py::arg("lambda_c_old"),
               py::arg("lambda_v_init"),
               py::arg("dt"),
               py::arg("dam"),
               "Run one-IP Newton solve and return assembled matrices at convergence:"
               " {'K': (N,N), 'F': (N,), 'C': (N,6)}.");

    m.def("cohesive_multiplier_slots", &shared_kernels::cohesive_multiplier_slots,
          py::arg("failureType"),
          "Return the number of cohesive multiplier values stored per integration point "
          "(3 for \"r2_2L_PSD\", 2 otherwise).");

    m.def("get_cohesive_num_lambdas", &shared_kernels::get_cohesive_num_lambdas,
          py::arg("failureType"),
          "Return the number of cohesive multipliers of the failure type (1, 2 or 3); "
          "unknown types raise ValueError.");

    m.def("compute_eta_point", &cohesive_viscoelastic::py_compute_eta_point,
          py::arg("failureType"), py::arg("strain"), py::arg("lambda_vals"),
          py::arg("true_2d") = false,
          "Evaluate eta and derivatives at one point.\n"
          "Returns (eta[6,1], deta_de[6,6], deta_dl[nl,6,1], d2eta_dlde[nl*6,6]).");

    m.def("compute_failure_surface_point",
          &cohesive_viscoelastic::py_compute_failure_surface_point,
          py::arg("f_t"), py::arg("f_s"), py::arg("bulk"),
          py::arg("DP_phi"), py::arg("DP_q"), py::arg("failureType"),
          py::arg("eta"), py::arg("dam"), py::arg("strain"),
          py::arg("true_2d") = false,
          "Evaluate the failure potential and derivatives at one point.\n"
          "Returns (F, dF_deta[1,6], ddF_deta2[6,6], ddF_detade[6,6], F_damage).");

    m.def("compute_strength_surface_point", &cohesive_viscoelastic::py_compute_strength_surface_point,
          py::arg("f_t"), py::arg("f_s"), py::arg("bulk"),
          py::arg("DP_phi"), py::arg("DP_q"), py::arg("failureType"),
          py::arg("stress"), py::arg("strain"), py::arg("true_2d") = false,
          py::arg("num_direction_samples") = 65,
          "Evaluate the cohesive strength indicator of a stress and its strain.\n"
          "The strength surface is the zero contour; positive values are outside.");

    py::class_<cohesive_viscoelastic::CVEVariablePropsKernelRunner>(
        m, "CohesiveViscoElasticKernelsVariableProps")
        .def(py::init<double, double, double, double, double, double,
                      const std::string&, bool, bool>(),
             py::arg("n_exp"),
             py::arg("f_s"),
             py::arg("G2_over_G1"),
             py::arg("DP_phi"),
             py::arg("DP_q"),
             py::arg("kappa"),
             py::arg("failureType"),
             py::arg("plane_stress"),
             py::arg("true_2d") = false,
             "Build a cohesive visco-elastic kernel with per-IP ice properties.")
        .def("__call__",
             &cohesive_viscoelastic::CVEVariablePropsKernelRunner::call,
             py::arg("strains"),
             py::arg("dt"),
             py::arg("dam"),
             py::arg("lame"),
             py::arg("shear"),
             py::arg("viscosity"),
             py::arg("f_t"),
             py::arg("stress"),
             py::arg("eta_c"),
             py::arg("eta_c_old"),
             py::arg("strain_v"),
             py::arg("ep_old"),
             py::arg("dsigma_dstrain"),
             py::arg("E_elastic"),
             py::arg("E_psi"),
             py::arg("lambda_v_field"),
             py::arg("lambda_c_field"),
             py::arg("lambda_c_field_old"),
             "Update all integration points using per-IP elastic, creep, and tensile properties.");
}
