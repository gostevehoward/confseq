#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "uniform_boundaries.h"

using namespace pybind11::literals;

double quantile_ab_p_value(pybind11::array_t<double> a_values,
                           pybind11::array_t<double> b_values,
                           const double quantile_p, const int t_opt,
                           const double alpha_opt=0.05) {
  const pybind11::buffer_info a_buffer = a_values.request(),
      b_buffer = b_values.request();
  auto a_os = std::make_unique<confseq::StaticOrderStatistics>(
      (double*) a_buffer.ptr, (double*) a_buffer.ptr + a_buffer.shape[0]);
  auto b_os = std::make_unique<confseq::StaticOrderStatistics>(
      (double*) b_buffer.ptr, (double*) b_buffer.ptr + b_buffer.shape[0]);
  confseq::QuantileABTest test(quantile_p, t_opt, alpha_opt, std::move(a_os),
                               std::move(b_os));
  return test.p_value();
}

PYBIND11_MODULE(quantiles, m) {
  m.def("empirical_process_lil_bound",
        pybind11::vectorize(confseq::empirical_process_lil_bound),
        R"pbdoc(
          Empirical process finite LIL bound.

          Based on Theorem 2 of the quantile paper. Bound has the form
          `A sqrt((loglog(et / t_min) + C) / t)`, and is valid only for
          t >= t_min. C is chosen to achieve the desired error probability
          `alpha`.

          This bound controls the deviations of the empirical CDF from the true
          CDF uniformly over x and time, or yields a confidence sequences
          uniform over quantiles and time (Corollary 2 of the quantile paper).

          * `A`: leading constant in the bound
        )pbdoc",
        "t"_a, "alpha"_a, "t_min"_a, "A"_a=0.85);
  m.def("double_stitching_bound",
        pybind11::vectorize(confseq::double_stitching_bound),
        R"pbdoc(
          "Double stitching" bound (Theorem 3 of the quantile paper).

          Yields a confidence sequence uniform over quantiles and time.

          * `quantile_p`: designates which quantile we wish to bound
          * `delta`: controls the fineness of the quantile grid used in
            construction of the bound
          * `s`: controls how crossing probability is distribted over epochs
          * `eta`: controls the spacing of epochs
        )pbdoc",
        "quantile_p"_a, "t"_a, "alpha"_a, "t_opt"_a, "delta"_a=0.5, "s"_a=1.4,
        "eta"_a=2);
  m.def("quantile_ab_p_value",
        &quantile_ab_p_value,
        R"pbdoc(
          Get two-sided p-value for two-sample test of equal quantiles.

          * `a_values` and `b_values`: NumPy arrays containing observed values
            from each of the two arms
          * `quantile_p`: designates which quantile we wish to test
        )pbdoc",
        "a_values"_a, "b_values"_a, "quantile_p"_a, "t_opt"_a,
        "alpha_opt"_a=0.05);
}
