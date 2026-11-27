// viscoelastic_kernels_pybind.cpp
// Public API: top-level integration-point loop + pybind11 module.
//
// The pybind11 binding exposes Python-callable runners for constant and
// per-IP material properties.

#include "viscoelastic_kernels.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace py = pybind11;

void bind_split_viscoelastic(py::module_& m);
void bind_elastic_splits(py::module_& m);

namespace viscoelastic {

// ==========================================================================
//  Top-level loop over all integration points
// ==========================================================================

void update_ips_viscoelastic(const MaterialParams& mp,
                              bool     damMaterial,
                              const double* strains,
                              double        dt,
                              const double* dam,
                              double*       stress,
                              double*       strain_p,
                              const double* ep_old,
                              double*       dsigma_dstrain,
                              double*       lambda_field,
                              int           n_ip) {
    for (int ip = 0; ip < n_ip; ++ip) {
        const int base6  = ip * 6;
        const int base36 = ip * 36;

        Vec6 strain_ip, ep_old_ip;
        for (int i = 0; i < 6; ++i) {
            strain_ip(i) = strains[base6 + i];
            ep_old_ip(i) = ep_old[base6 + i];
        }

        // Input NaN guard
        for (int i = 0; i < 6; ++i) {
            if (std::isnan(strain_ip(i)) || std::isinf(strain_ip(i)))
                throw std::runtime_error(
                    "NaN/Inf detected in strain input at IP " + std::to_string(ip));
        }

        IPResult r = update_ip_state(mp, damMaterial, strain_ip, ep_old_ip, dt, dam[ip]);

        lambda_field[ip] = r.lambda;
        for (int i = 0; i < 6; ++i) {
            stress[base6 + i]   = r.stress(i);
            strain_p[base6 + i] = r.strain_p(i);
            for (int j = 0; j < 6; ++j)
                dsigma_dstrain[base36 + i*6 + j] = r.dsigma_dstrain(i, j);
        }
    }
}

void update_ips_viscoelastic_variable_props(bool     damMaterial,
                                            double   n_exp,
                                            const double* strains,
                                            double        dt,
                                            const double* dam,
                                            const double* lame,
                                            const double* shear,
                                            const double* viscosity,
                                            double*       stress,
                                            double*       strain_p,
                                            const double* ep_old,
                                            double*       dsigma_dstrain,
                                            double*       lambda_field,
                                            int           n_ip) {
    for (int ip = 0; ip < n_ip; ++ip) {
        const int base6  = ip * 6;
        const int base36 = ip * 36;

        Vec6 strain_ip, ep_old_ip;
        for (int i = 0; i < 6; ++i) {
            strain_ip(i) = strains[base6 + i];
            ep_old_ip(i) = ep_old[base6 + i];
        }

        // Input NaN guard
        for (int i = 0; i < 6; ++i) {
            if (std::isnan(strain_ip(i)) || std::isinf(strain_ip(i)))
                throw std::runtime_error(
                    "NaN/Inf detected in strain input at IP " + std::to_string(ip));
        }
        if (std::isnan(lame[ip]) || std::isinf(lame[ip]) ||
            std::isnan(shear[ip]) || std::isinf(shear[ip]) ||
            std::isnan(viscosity[ip]) || std::isinf(viscosity[ip])) {
            throw std::runtime_error(
                "NaN/Inf detected in material parameters at IP " + std::to_string(ip));
        }

        MaterialParams mp = make_material_params(
            viscosity[ip],
            n_exp,
            lame[ip],
            shear[ip]);
        IPResult r = update_ip_state(mp, damMaterial, strain_ip, ep_old_ip, dt, dam[ip]);

        lambda_field[ip] = r.lambda;
        for (int i = 0; i < 6; ++i) {
            stress[base6 + i]   = r.stress(i);
            strain_p[base6 + i] = r.strain_p(i);
            for (int j = 0; j < 6; ++j)
                dsigma_dstrain[base36 + i*6 + j] = r.dsigma_dstrain(i, j);
        }
    }
}

// ==========================================================================
//  ViscoElasticKernelRunner — wraps MaterialParams and exposes a callable
// ==========================================================================

struct ViscoElasticKernelRunner {
    MaterialParams mp;

    ViscoElasticKernelRunner(double viscosity, double n_exp,
                              double lame,      double shear,
                              bool true_2d_in = false)
        : mp(make_material_params(viscosity, n_exp, lame, shear, true_2d_in)) {}

    /// __call__ signature matches viscoelastic.py's _update_viscousCreep.
    /// Arguments (positional):
    ///   strains          – flat (n_ip*6,) total strain in Mandel notation [in]
    ///   dt               – time-step size [s]
    ///   dam              – flat (n_ip,) elastic damage factor [in]
    ///   stress           – flat (n_ip*6,) Cauchy stress [out]
    ///   strain_p         – flat (n_ip*6,) plastic (viscous) strain [out]
    ///   ep_old           – flat (n_ip*6,) plastic strain at previous step [in]
    ///   dsigma_dstrain   – flat (n_ip*36,) consistent tangent [out]
    ///   lambda_field     – flat (n_ip,) plastic multiplier [out]
    ///   lambda_field_old – flat (n_ip,) old plastic multiplier [in, unused]
    void call(py::array_t<double, py::array::c_style | py::array::forcecast> strains,
              double dt,
              bool damMaterial,
              py::array_t<double, py::array::c_style | py::array::forcecast> dam,
              py::array_t<double, py::array::c_style | py::array::forcecast> stress,
              py::array_t<double, py::array::c_style | py::array::forcecast> strain_p,
              py::array_t<double, py::array::c_style | py::array::forcecast> ep_old,
              py::array_t<double, py::array::c_style | py::array::forcecast> dsigma_dstrain,
              py::array_t<double, py::array::c_style | py::array::forcecast> lambda_field,
              py::array_t<double, py::array::c_style | py::array::forcecast> /*lambda_field_old*/) {
        int n_ip = static_cast<int>(strains.size()) / 6;

        update_ips_viscoelastic(mp,
                                damMaterial,
                                strains.data(), dt, dam.data(),
                                stress.mutable_data(),
                                strain_p.mutable_data(),
                                ep_old.data(),
                                dsigma_dstrain.mutable_data(),
                                lambda_field.mutable_data(),
                                n_ip);
    }
};

// ==========================================================================
//  ViscoElasticVariablePropsKernelRunner — per-IP material parameters
// ==========================================================================

struct ViscoElasticVariablePropsKernelRunner {
    double n_exp;
    bool damMaterial;

    ViscoElasticVariablePropsKernelRunner(double n_exp_in = 3.0,
                                          bool damMaterial_in = true)
        : n_exp(n_exp_in), damMaterial(damMaterial_in) {}

    /// __call__ signature matches ice_viscoelastic.py's variable-property update.
    /// Arguments (positional):
    ///   strains          – flat (n_ip*6,) total strain in Mandel notation [in]
    ///   dt               – time-step size [s]
    ///   dam              – flat (n_ip,) elastic damage factor [in]
    ///   lame             – flat (n_ip,) first Lamé parameter [in]
    ///   shear            – flat (n_ip,) shear modulus [in]
    ///   viscosity        – flat (n_ip,) J2 viscoplastic stress scale eta_vp [in]
    ///   stress           – flat (n_ip*6,) Cauchy stress [out]
    ///   strain_p         – flat (n_ip*6,) plastic (viscous) strain [out]
    ///   ep_old           – flat (n_ip*6,) plastic strain at previous step [in]
    ///   dsigma_dstrain   – flat (n_ip*36,) consistent tangent [out]
    ///   lambda_field     – flat (n_ip,) plastic multiplier [out]
    ///   lambda_field_old – flat (n_ip,) old plastic multiplier [in, unused]
    void call(py::array_t<double, py::array::c_style | py::array::forcecast> strains,
              double dt,
              py::array_t<double, py::array::c_style | py::array::forcecast> dam,
              py::array_t<double, py::array::c_style | py::array::forcecast> lame,
              py::array_t<double, py::array::c_style | py::array::forcecast> shear,
              py::array_t<double, py::array::c_style | py::array::forcecast> viscosity,
              py::array_t<double, py::array::c_style | py::array::forcecast> stress,
              py::array_t<double, py::array::c_style | py::array::forcecast> strain_p,
              py::array_t<double, py::array::c_style | py::array::forcecast> ep_old,
              py::array_t<double, py::array::c_style | py::array::forcecast> dsigma_dstrain,
              py::array_t<double, py::array::c_style | py::array::forcecast> lambda_field,
              py::array_t<double, py::array::c_style | py::array::forcecast> /*lambda_field_old*/) {
        int n_ip = static_cast<int>(strains.size()) / 6;

        update_ips_viscoelastic_variable_props(damMaterial,
                                               n_exp,
                                               strains.data(), dt, dam.data(),
                                               lame.data(),
                                               shear.data(),
                                               viscosity.data(),
                                               stress.mutable_data(),
                                               strain_p.mutable_data(),
                                               ep_old.data(),
                                               dsigma_dstrain.mutable_data(),
                                               lambda_field.mutable_data(),
                                               n_ip);
    }
};

} // namespace viscoelastic

// ==========================================================================
//  pybind11 module
// ==========================================================================

PYBIND11_MODULE(_viscoelastic_kernels_cpp, m)
{
    m.doc() = "C++ visco-elastic J2 return-mapping kernels (Eigen + pybind11)";
    bind_split_viscoelastic(m);
    bind_elastic_splits(m);

    py::class_<viscoelastic::ViscoElasticKernelRunner>(m, "ViscoElasticKernels")
        .def(py::init<double, double, double, double, bool>(),
             py::arg("viscosity"), py::arg("n_exp"),
             py::arg("lame"),      py::arg("shear"),
             py::arg("true_2d") = false,
             "Build the visco-elastic return-mapping kernel.\n\n"
             "Parameters\n"
             "----------\n"
             "viscosity : float\n"
             "    J2 viscoplastic stress scale eta_vp [Pa·s^{1/n}].\n"
             "n_exp : float\n"
             "    Power-law (Glen) exponent n.\n"
             "lame : float\n"
             "    First Lamé parameter [Pa].\n"
             "shear : float\n"
             "    Shear modulus [Pa].\n"
             "true_2d : bool\n"
             "    Use intrinsic 2-D stiffness and J2 invariants, with zero\n"
             "    out-of-plane Mandel components.")
        .def("__call__",
             &viscoelastic::ViscoElasticKernelRunner::call,
             py::arg("strains"),
             py::arg("dt"),
             py::arg("damMaterial"),
             py::arg("dam"),
             py::arg("stress"),
             py::arg("strain_p"),
             py::arg("ep_old"),
             py::arg("dsigma_dstrain"),
             py::arg("lambda_field"),
             py::arg("lambda_field_old"),
             "Update all integration points in-place (all arrays are flat NumPy arrays).\n\n"
             "Modifies stress, strain_p, dsigma_dstrain and lambda_field in-place.");

    py::class_<viscoelastic::ViscoElasticVariablePropsKernelRunner>(
        m, "ViscoElasticKernelsVariableProps")
        .def(py::init<double, bool>(),
             py::arg("n_exp") = 3.0,
             py::arg("damMaterial") = true,
             "Build the variable-property visco-elastic return-mapping kernel.\n\n"
             "The material properties are supplied as flat per-IP arrays in "
             "__call__; the one-IP return-mapping is the same implementation "
             "used by ViscoElasticKernels.")
        .def("__call__",
             &viscoelastic::ViscoElasticVariablePropsKernelRunner::call,
             py::arg("strains"),
             py::arg("dt"),
             py::arg("dam"),
             py::arg("lame"),
             py::arg("shear"),
             py::arg("viscosity"),
             py::arg("stress"),
             py::arg("strain_p"),
             py::arg("ep_old"),
             py::arg("dsigma_dstrain"),
             py::arg("lambda_field"),
             py::arg("lambda_field_old"),
             "Update all integration points in-place using per-IP material "
             "arrays for lame, shear and viscosity.");
}
