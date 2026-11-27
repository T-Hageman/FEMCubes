// Direct elastic evaluation of the same splits used by the creep return map.
#include "phasefield_splits.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace shared_kernels {

// Driver-only evaluation for the original models, including spatially varying
// moduli in IceViscoElastic. All split definitions remain in elastic_split().
py::array_t<double> elastic_split_drivers(
    py::array_t<double, py::array::c_style | py::array::forcecast> strains,
    py::array_t<double, py::array::c_style | py::array::forcecast> lame,
    py::array_t<double, py::array::c_style | py::array::forcecast> shear,
    const std::string& split_name, bool true_2d) {
    const auto split = parse_elastic_split(split_name);
    if (strains.ndim() != 2 || strains.shape(1) != 6
        || lame.ndim() != 1 || lame.size() != strains.shape(0)
        || shear.ndim() != 1 || shear.size() != strains.shape(0))
        throw std::invalid_argument("Expected strains shaped (n, 6) and moduli shaped (n,).");
    py::array_t<double> driver(strains.shape(0));
    for (py::ssize_t ip = 0; ip < strains.shape(0); ++ip) {
        const Vec6 e = Eigen::Map<const Vec6>(strains.data() + 6*ip);
        const double l = lame.data()[ip], g = shear.data()[ip];
        if (!e.allFinite() || !std::isfinite(l) || l < 0.0 || !std::isfinite(g) || g <= 0.0)
            throw std::invalid_argument("Elastic drivers require finite strains, lame >= 0 and shear > 0.");
        driver.mutable_data()[ip] = elastic_split(e, l, g, true_2d, split).energy_plus;
    }
    return driver;
}

class SplitLinearElasticKernelRunner {
    double lame, shear;
    ElasticSplit split;
    bool true_2d;

public:
    SplitLinearElasticKernelRunner(double lame_in, double shear_in,
                                   const std::string& split_name, bool true_2d_in)
        : lame(lame_in), shear(shear_in), split(parse_elastic_split(split_name)), true_2d(true_2d_in) {
        if (!std::isfinite(lame) || lame < 0.0 || !std::isfinite(shear) || shear <= 0.0)
            throw std::invalid_argument("Split elasticity requires finite lame >= 0 and shear > 0.");
    }

    py::dict call(py::array_t<double, py::array::c_style | py::array::forcecast> strains,
                  py::array_t<double, py::array::c_style | py::array::forcecast> damage) const {
        if (strains.ndim() != 2 || strains.shape(1) != 6
            || damage.ndim() != 1 || damage.size() != strains.shape(0))
            throw std::invalid_argument("Expected strains shaped (n, 6) and damage shaped (n,).");
        const py::ssize_t n = strains.shape(0);
        py::array_t<double> stress({n, py::ssize_t(6)});
        py::array_t<double> plus({n, py::ssize_t(6)}), minus({n, py::ssize_t(6)});
        py::array_t<double> tangent({n, py::ssize_t(6), py::ssize_t(6)});
        py::array_t<double> driver(n), energy_minus(n);
        for (py::ssize_t ip = 0; ip < n; ++ip) {
            const Vec6 e = Eigen::Map<const Vec6>(strains.data() + 6*ip);
            const double g = damage.data()[ip];
            if (!e.allFinite() || !std::isfinite(g) || g < 0.0 || g > 1.0)
                throw std::invalid_argument("Strains must be finite and damage factors must lie in [0, 1].");
            const auto r = elastic_split(e, lame, shear, true_2d, split);
            Eigen::Map<Vec6>(stress.mutable_data() + 6*ip) = g*r.stress_plus + r.stress_minus;
            Eigen::Map<Vec6>(plus.mutable_data() + 6*ip) = r.stress_plus;
            Eigen::Map<Vec6>(minus.mutable_data() + 6*ip) = r.stress_minus;
            Eigen::Map<Eigen::Matrix<double,6,6,Eigen::RowMajor>>(tangent.mutable_data() + 36*ip)
                = g*r.tangent_plus + r.tangent_minus;
            driver.mutable_data()[ip] = r.energy_plus;
            energy_minus.mutable_data()[ip] = r.energy_minus;
        }
        py::dict result;
        result["stress"] = stress;
        result["stress_plus"] = plus;
        result["stress_minus"] = minus;
        result["tangent"] = tangent;
        result["driver"] = driver;
        result["energy_minus"] = energy_minus;
        return result;
    }
};

} // namespace shared_kernels

void bind_elastic_splits(py::module_& m) {
    m.def("elastic_split_drivers", &shared_kernels::elastic_split_drivers,
          py::arg("strains"), py::arg("lame"), py::arg("shear"), py::arg("split"),
          py::arg("true_2d") = false,
          "Evaluate the shared split driver with moduli supplied at each integration point.");
    py::class_<shared_kernels::SplitLinearElasticKernelRunner>(m, "SplitLinearElasticKernels")
        .def(py::init<double, double, const std::string&, bool>(),
             py::arg("lame"), py::arg("shear"), py::arg("split"), py::arg("true_2d") = false)
        .def("__call__", &shared_kernels::SplitLinearElasticKernelRunner::call,
             py::arg("strains"), py::arg("damage"),
             "Return mechanical stress, tangent, undegraded stress_plus/stress_minus, "
             "driver and energy_minus. Driver is psi+ except for Stress-Based, "
             "which returns squared positive principal stress (Pa^2) and zero energy_minus.");
}
