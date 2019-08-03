#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "uniform_boundaries.h"

using namespace pybind11::literals;

PYBIND11_MODULE(boundaries, m) {
  m.doc() = R"pbdoc(
    Uniform boundaries and mixture supermartingales. See package documentation
    at https://github.com/gostevehoward/confseq.

    All `*_bound()` functions implement uniform boundaries, and accept intrinsic
    time `v` at which to evaluate the bound and crossing probability `alpha`.

    All `*_log_mixture()` functions evaluate the logarithm of the mixture
    supermartingale, and accept `s`, the value of the underlying martingale, and
    `v`, the intrinsic time value.

    All mixture functions accept `v_opt` and optionally `alpha_opt`. The
    mixtures are then tuned to optimize the uniform boundary with crossing
    probability `alpha_opt` at intrinsic time `v_opt`.
    )pbdoc";

  m.def("normal_log_mixture",
        pybind11::vectorize(confseq::normal_log_mixture),
        R"pbdoc(
          Logarithm of mixture supermartingale for the one- or two-sided normal
          mixture.
        )pbdoc",
        "s"_a, "v"_a, "v_opt"_a, "alpha_opt"_a=0.05, "is_one_sided"_a=true);
  m.def("normal_mixture_bound",
        pybind11::vectorize(confseq::normal_mixture_bound),
        R"pbdoc(
          One- or two-sided normal mixture uniform boundary.
        )pbdoc",
        "v"_a, "alpha"_a, "v_opt"_a, "alpha_opt"_a=0.05,
        "is_one_sided"_a=true);
  m.def("gamma_exponential_log_mixture",
        pybind11::vectorize(confseq::gamma_exponential_log_mixture),
        R"pbdoc(
          Logarithm of mixture supermartingale for the gamma-exponential
          mixture.

          `c` is the sub-exponential scale parameter.
        )pbdoc",
        "s"_a, "v"_a, "v_opt"_a, "c"_a, "alpha_opt"_a=0.05);
  m.def("gamma_exponential_mixture_bound",
        pybind11::vectorize(confseq::gamma_exponential_mixture_bound),
        R"pbdoc(
          Gamma-exponential mixture uniform boundary.

          `c` is the sub-exponential scale parameter.
        )pbdoc",
        "v"_a, "alpha"_a, "v_opt"_a, "c"_a, "alpha_opt"_a=0.05);
  m.def("gamma_poisson_log_mixture",
        pybind11::vectorize(confseq::gamma_poisson_log_mixture),
        R"pbdoc(
          Logarithm of mixture supermartingale for the gamma-Poisson mixture.

          `c` is the sub-Poisson scale parameter.
        )pbdoc",
        "s"_a, "v"_a, "v_opt"_a, "c"_a, "alpha_opt"_a=0.05);
  m.def("gamma_poisson_mixture_bound",
        pybind11::vectorize(confseq::gamma_poisson_mixture_bound),
        R"pbdoc(
          Gamma-Poisson mixture uniform boundary.

          `c` is the sub-Poisson scale parameter.
        )pbdoc",
        "v"_a, "alpha"_a, "v_opt"_a, "c"_a, "alpha_opt"_a=0.05);
  m.def("beta_binomial_log_mixture",
        pybind11::vectorize(confseq::beta_binomial_log_mixture),
        R"pbdoc(
          Logarithm of mixture supermartingale for the one- or two-sided
          beta-binomial mixture.

          `g` and `h` are the sub-Bernoulli range parameter.
        )pbdoc",
        "s"_a, "v"_a, "v_opt"_a, "g"_a, "h"_a, "alpha_opt"_a=0.05,
        "is_one_sided"_a=true);
  m.def("beta_binomial_mixture_bound",
        pybind11::vectorize(confseq::beta_binomial_mixture_bound),
        R"pbdoc(
          One- or two-sided beta-binomial mixture uniform boundary.

          `g` and `h` are the sub-Bernoulli range parameter.
        )pbdoc",
        "v"_a, "alpha"_a, "v_opt"_a, "g"_a, "h"_a, "alpha_opt"_a=0.05,
        "is_one_sided"_a=true);
  m.def("poly_stitching_bound",
        pybind11::vectorize(confseq::poly_stitching_bound),
        R"pbdoc(
          Polynomial stitched uniform boundary.

          * `v_min`: optimized-for intrinsic time
          * `c`: sub-gamma scale parameter
          * `s`: controls how crossing probability is distribted over epochs
          * `eta`: controls the spacing of epochs
        )pbdoc",
        "v"_a, "alpha"_a, "v_min"_a, "c"_a=0, "s"_a=1.4, "eta"_a=2);
  m.def("bernoulli_confidence_interval",
        &confseq::bernoulli_confidence_interval,
        R"pbdoc(
        Confidence sequence for [0, 1]-bounded distributions.

        This function returns confidence bounds for the mean of a Bernoulli
        distribution, or more generally, any distribution with support in the
        unit interval [0, 1]. (This applies to any bounded distribution after
        rescaling.) The confidence bounds form a confidence sequence, so are
        guaranteed to cover the true mean uniformly over time with probability 1
        - `alpha`.

        * `num_successes`: number of "successful" Bernoulli trials seen so far,
          or more generally, sum of observed outcomes.
        * `num_trials`: total number of observations seen so far.
        )pbdoc",
        "num_successes"_a, "num_trials"_a, "alpha"_a, "t_opt"_a,
        "alpha_opt"_a=0.05);
}
