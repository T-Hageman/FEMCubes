#include "split_viscoelastic.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace viscoelastic {
namespace {

using InputArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void validate_material(double viscosity, double n_exp, double lame, double shear) {
    if (!std::isfinite(viscosity) || viscosity <= 0.0 || !std::isfinite(n_exp) || n_exp < 1.0
        || !std::isfinite(lame) || lame < 0.0 || !std::isfinite(shear) || shear <= 0.0)
        throw std::invalid_argument("Split viscoelasticity requires finite viscosity > 0, n >= 1, lame >= 0 and shear > 0.");
}

CreepCoupling parse_creep_coupling(const std::string& coupling) {
    if (coupling == "EffectiveStress") return CreepCoupling::EffectiveStress;
    if (coupling == "DegradedStress") return CreepCoupling::DegradedStress;
    if (coupling == "FullDamagedStress") return CreepCoupling::FullDamagedStress;
    if (coupling == "PartialExplicit") return CreepCoupling::PartialExplicit;
    throw std::invalid_argument("CreepCoupling must be EffectiveStress, DegradedStress, FullDamagedStress or PartialExplicit.");
}

template <typename MaterialAtIP>
py::dict evaluate_split_points(InputArray strains, InputArray old_strains, InputArray damage,
                               double dt, shared_kernels::ElasticSplit split,
                               CreepCoupling creep_coupling, const py::object& old_total_strains,
                               MaterialAtIP material_at_ip) {
    if (strains.ndim() != 2 || strains.shape(1) != 6
        || old_strains.ndim() != 2 || old_strains.shape(0) != strains.shape(0)
        || old_strains.shape(1) != 6 || damage.ndim() != 1 || damage.size() != strains.shape(0))
        throw std::invalid_argument("Expected strains and old_strains shaped (n, 6), damage shaped (n,).");
    if (!std::isfinite(dt) || dt <= 0.0)
        throw std::invalid_argument("dt must be finite and positive.");
    const py::ssize_t n = strains.shape(0);
    const bool has_old_total = !old_total_strains.is_none();
    InputArray previous;
    if (has_old_total) {
        previous = InputArray::ensure(old_total_strains);
        if (!previous || previous.ndim() != 2 || previous.shape(0) != n || previous.shape(1) != 6)
            throw std::invalid_argument("old_total_strains must have shape (n, 6).");
    } else if (creep_coupling == CreepCoupling::PartialExplicit) {
        throw std::invalid_argument("PartialExplicit requires old_total_strains from the committed step.");
    }
    py::array_t<double> stress({n, py::ssize_t(6)}), plastic({n, py::ssize_t(6)});
    py::array_t<double> plus({n, py::ssize_t(6)}), minus({n, py::ssize_t(6)});
    py::array_t<double> tangent({n, py::ssize_t(6), py::ssize_t(6)});
    py::array_t<double> driver(n), energy_minus(n), lambda(n);
    for (py::ssize_t ip = 0; ip < n; ++ip) {
        const Vec6 e = Eigen::Map<const Vec6>(strains.data() + 6*ip);
        const Vec6 old = Eigen::Map<const Vec6>(old_strains.data() + 6*ip);
        Vec6 old_total = Vec6::Zero();
        if (has_old_total) old_total = Eigen::Map<const Vec6>(previous.data() + 6*ip);
        const double g = damage.data()[ip];
        if (!e.allFinite() || !old.allFinite() || !old_total.allFinite() || !std::isfinite(g) || g < 0.0 || g > 1.0)
            throw std::invalid_argument("Strains must be finite and damage factors must lie in [0, 1].");
        SplitIPResult r;
        try {
            r = update_split_ip(material_at_ip(ip), split, creep_coupling, e, old, dt, g, old_total);
        } catch (const std::runtime_error& error) {
            throw std::runtime_error("Integration point " + std::to_string(ip) + ": " + error.what());
        }
        Eigen::Map<Vec6>(stress.mutable_data() + 6*ip) = r.state.stress;
        Eigen::Map<Vec6>(plastic.mutable_data() + 6*ip) = r.state.strain_p;
        Eigen::Map<Vec6>(plus.mutable_data() + 6*ip) = r.split.stress_plus;
        Eigen::Map<Vec6>(minus.mutable_data() + 6*ip) = r.split.stress_minus;
        Eigen::Map<Eigen::Matrix<double,6,6,Eigen::RowMajor>>(tangent.mutable_data() + 36*ip) = r.state.dsigma_dstrain;
        driver.mutable_data()[ip] = r.split.energy_plus;
        energy_minus.mutable_data()[ip] = r.split.energy_minus;
        lambda.mutable_data()[ip] = r.state.lambda;
    }
    py::dict result;
    result["stress"] = stress;
    result["strain_p"] = plastic;
    result["stress_plus"] = plus;
    result["stress_minus"] = minus;
    result["tangent"] = tangent;
    result["driver"] = driver;
    result["energy_minus"] = energy_minus;
    result["lambda"] = lambda;
    return result;
}

} // namespace

class SplitViscoElasticKernelRunner {
    MaterialParams mp;
    shared_kernels::ElasticSplit split;
    CreepCoupling creep_coupling;

public:
    SplitViscoElasticKernelRunner(double viscosity, double n_exp, double lame, double shear,
                                  const std::string& split_name, const std::string& coupling,
                                  bool true_2d)
        : mp(make_material_params(viscosity, n_exp, lame, shear, true_2d)),
          split(shared_kernels::parse_elastic_split(split_name)),
          creep_coupling(parse_creep_coupling(coupling)) {
        validate_material(viscosity, n_exp, lame, shear);
    }

    py::dict call(InputArray strains, InputArray old_strains, InputArray damage, double dt,
                  const py::object& old_total_strains) const {
        return evaluate_split_points(strains, old_strains, damage, dt, split, creep_coupling, old_total_strains,
                                     [&](py::ssize_t) { return mp; });
    }
};

class SplitViscoElasticVariablePropsKernelRunner {
    double n_exp;
    shared_kernels::ElasticSplit split;
    CreepCoupling creep_coupling;
    bool true_2d;

public:
    SplitViscoElasticVariablePropsKernelRunner(double n_exp_in, const std::string& split_name,
                                               const std::string& coupling, bool true_2d_in)
        : n_exp(n_exp_in), split(shared_kernels::parse_elastic_split(split_name)),
          creep_coupling(parse_creep_coupling(coupling)), true_2d(true_2d_in) {
        if (!std::isfinite(n_exp) || n_exp < 1.0)
            throw std::invalid_argument("Split viscoelasticity requires finite n >= 1.");
    }

    py::dict call(InputArray strains, InputArray old_strains, InputArray damage, double dt,
                  InputArray lame, InputArray shear, InputArray viscosity,
                  const py::object& old_total_strains) const {
        if (strains.ndim() != 2 || strains.shape(1) != 6)
            throw std::invalid_argument("Expected strains shaped (n, 6).");
        for (const auto& values : {lame, shear, viscosity}) {
            if (values.ndim() != 1 || values.size() != strains.shape(0))
                throw std::invalid_argument("Material properties must have shape (n,).");
        }
        return evaluate_split_points(strains, old_strains, damage, dt, split, creep_coupling, old_total_strains,
            [&](py::ssize_t ip) {
                validate_material(viscosity.data()[ip], n_exp, lame.data()[ip], shear.data()[ip]);
                return make_material_params(viscosity.data()[ip], n_exp, lame.data()[ip],
                                            shear.data()[ip], true_2d);
            });
    }
};

} // namespace viscoelastic

void bind_split_viscoelastic(py::module_& m) {
    py::class_<viscoelastic::SplitViscoElasticKernelRunner>(m, "SplitViscoElasticKernels")
        .def(py::init<double, double, double, double, const std::string&, const std::string&, bool>(),
             py::arg("viscosity"), py::arg("n_exp"), py::arg("lame"), py::arg("shear"),
             py::arg("split"), py::arg("creep_coupling") = "EffectiveStress", py::arg("true_2d") = false)
        .def("__call__", &viscoelastic::SplitViscoElasticKernelRunner::call,
             py::arg("strains"), py::arg("old_strains"), py::arg("damage"), py::arg("dt"),
             py::arg("old_total_strains") = py::none(),
             "Return stress, strain_p, tangent, lambda, undegraded stress_plus/stress_minus, "
             "driver and energy_minus. Driver is psi+ (J/m^3), except Stress-Based: "
             "sum of squared positive effective principal stresses (Pa^2); energy_minus "
             "is zero for that non-variational criterion. PartialExplicit requires "
             "old_total_strains shaped (n, 6); other couplings do not use it.");
    py::class_<viscoelastic::SplitViscoElasticVariablePropsKernelRunner>(m, "SplitViscoElasticKernelsVariableProps")
        .def(py::init<double, const std::string&, const std::string&, bool>(),
             py::arg("n_exp"), py::arg("split"), py::arg("creep_coupling") = "EffectiveStress",
             py::arg("true_2d") = false)
        .def("__call__", &viscoelastic::SplitViscoElasticVariablePropsKernelRunner::call,
             py::arg("strains"), py::arg("old_strains"), py::arg("damage"), py::arg("dt"),
             py::arg("lame"), py::arg("shear"), py::arg("viscosity"),
             py::arg("old_total_strains") = py::none(),
             "Use per-point material properties; return the same fields as SplitViscoElasticKernels.");
}
