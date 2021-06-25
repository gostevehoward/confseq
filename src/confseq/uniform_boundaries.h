#ifndef CONFIDENCESEQUENCES_UNIFORM_BOUNDARIES_H_
#define CONFIDENCESEQUENCES_UNIFORM_BOUNDARIES_H_

#include <algorithm>
#include <exception>
#include <memory>
#include <set>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/zeta.hpp>
#include <boost/math/tools/minima.hpp>
#include <boost/math/tools/roots.hpp>

namespace confseq {

//////////////////////////////////////////////////////////////////////
// Simplified interface
//////////////////////////////////////////////////////////////////////

double normal_log_mixture(const double s, const double v, const double v_opt,
                          const double alpha_opt=0.05,
                          const bool is_one_sided=true);

double normal_mixture_bound(
    const double v, const double alpha, const double v_opt,
    const double alpha_opt=0.05, const bool is_one_sided=true);

double gamma_exponential_log_mixture(
    const double s, const double v, const double v_opt,
    const double c, const double alpha_opt=0.05);

double gamma_exponential_mixture_bound(
    const double v, const double alpha, const double v_opt,
    const double c, const double alpha_opt=0.05);

double gamma_poisson_log_mixture(
    const double s, const double v, const double v_opt, const double c,
    const double alpha_opt=0.05);

double gamma_poisson_mixture_bound(
    const double v, const double alpha, const double v_opt, const double c,
    const double alpha_opt=0.05);

double beta_binomial_log_mixture(
    const double s, const double v, const double v_opt, const double g,
    const double h, const double alpha_opt=0.05,
    const bool is_one_sided=true);

double beta_binomial_mixture_bound(
    const double v, const double alpha, const double v_opt, const double g,
    const double h, const double alpha_opt=0.05,
    const bool is_one_sided=true);

double poly_stitching_bound(const double v, const double alpha,
                            const double v_min, const double c=0,
                            const double s=1.4, const double eta=2);

double empirical_process_lil_bound(const int t, const double alpha,
                                   const double t_min, const double A=0.85);

double double_stitching_bound(const double quantile_p, const double t,
                              const double alpha, const double t_opt,
                              const double delta=0.5, const double s=1.4,
                              const double eta=2);

std::pair<double, double> bernoulli_confidence_interval(
    const double num_successes, const int num_trials, const double alpha,
    const double t_opt, const double alpha_opt=0.05);

//////////////////////////////////////////////////////////////////////
// Object-oriented interface
//////////////////////////////////////////////////////////////////////

double log_beta(const double a, const double b);
double log_incomplete_beta(const double a, const double b, const double x);

// (v, alpha) -> boundary value
using UniformBoundary = std::function<double(const double, const double)>;

class MixtureSupermartingale {
public:
  virtual ~MixtureSupermartingale() {}
  virtual double log_superMG(const double s, const double v) const = 0;
  virtual double s_upper_bound(const double v) const = 0;
  virtual double bound(const double v, const double log_threshold) const = 0;
};

double find_mixture_bound(const MixtureSupermartingale& mixture_superMG,
                          const double v, const double log_threshold);

class TwoSidedNormalMixture : public MixtureSupermartingale {
 public:
  TwoSidedNormalMixture(double v_opt, double alpha_opt)
    : rho_(best_rho(v_opt, alpha_opt)) {
    assert(v_opt > 0);
  }

  double log_superMG(const double s, const double v) const override;
  double s_upper_bound(const double /*v*/) const override {
    return std::numeric_limits<double>::infinity();
  }
  double bound(const double v, const double log_threshold) const override;
  static double best_rho(double v, double alpha);

 private:
  const double rho_;
};

class OneSidedNormalMixture : public MixtureSupermartingale {
 public:
  OneSidedNormalMixture(double v_opt, double alpha_opt)
    : rho_(best_rho(v_opt, alpha_opt)) {
    assert(v_opt > 0);
  }

  double log_superMG(const double s, const double v) const override;
  double s_upper_bound(const double /*v*/) const override {
    return std::numeric_limits<double>::infinity();
  }
  double bound(const double v, const double log_threshold) const override {
    return find_mixture_bound(*this, v, log_threshold);
  }
  static double best_rho(double v, double alpha);

 private:
  const double rho_;
};

class GammaExponentialMixture : public MixtureSupermartingale {
 public:
  GammaExponentialMixture(double v_opt, double alpha_opt, double c)
      : rho_(OneSidedNormalMixture::best_rho(v_opt, alpha_opt)),
        c_(c), leading_constant_(get_leading_constant(rho_, c_)) {}

  double log_superMG(const double s, const double v) const override;
  double s_upper_bound(const double /*v*/) const override {
    return std::numeric_limits<double>::infinity();
  }
  double bound(const double v, const double log_threshold) const override {
    return find_mixture_bound(*this, v, log_threshold);
  }

 private:
  static double get_leading_constant(double rho, double c);

  const double rho_;
  const double c_;
  const double leading_constant_;
};

class GammaPoissonMixture : public MixtureSupermartingale {
 public:
  GammaPoissonMixture(double v_opt, double alpha_opt, double c)
      : rho_(OneSidedNormalMixture::best_rho(v_opt, alpha_opt)),
        c_(c), leading_constant_(get_leading_constant(rho_, c_)) {}

  double log_superMG(const double s, const double v) const override;
  double s_upper_bound(const double /*v*/) const override {
    return std::numeric_limits<double>::infinity();
  }
  double bound(const double v, const double log_threshold) const override {
    return find_mixture_bound(*this, v, log_threshold);
  }

 private:
  static double get_leading_constant(double rho, double c);

  const double rho_;
  const double c_;
  const double leading_constant_;
};

class BetaBinomialMixture : public MixtureSupermartingale {
 public:

  BetaBinomialMixture(double v_opt, double alpha_opt, double g, double h,
                      bool is_one_sided)
      : r_(optimal_r(v_opt, alpha_opt, g, h, is_one_sided)), g_(g), h_(h),
        is_one_sided_(is_one_sided),
        normalizer_(log_incomplete_beta(r_ / (g_ * (g_ + h_)),
                                        r_ / (h_ * (g_ + h_)),
                                        is_one_sided_ ? h_ / (g_ + h_) : 1)) {
    assert(g > 0);
    assert(h > 0);
    assert(r_ > 0);
  }

  double log_superMG(const double s, const double v) const override;
  double s_upper_bound(const double v) const override;
  double bound(const double v, const double log_threshold) const override {
    return find_mixture_bound(*this, v, log_threshold);
  }

 private:

  static double optimal_r(const double v_opt, const double alpha_opt,
                          const double g, const double h,
                          const bool is_one_sided) {
    double rho = is_one_sided
        ? OneSidedNormalMixture::best_rho(v_opt, alpha_opt)
        : TwoSidedNormalMixture::best_rho(v_opt, alpha_opt);
    return std::max(rho - g * h, 1e-3 * g * h);
  }

  const double r_;
  const double g_;
  const double h_;
  const bool is_one_sided_;
  const double normalizer_;
};

class PolyStitchingBound {
 public:
  PolyStitchingBound(double v_min, double c, double s, double eta)
    : v_min_(v_min), c_(c), s_(s), eta_(eta),
      k1_((pow(eta, .25) + pow(eta, -.25)) / sqrt(2)),
      k2_((sqrt(eta) + 1) / 2),
      A_(log(boost::math::zeta(s) / pow(log(eta), s))) {
    assert(v_min > 0);
  }

  double operator()(const double v, const double alpha) const;

  const double v_min_;
  const double c_;
  const double s_;
  const double eta_;
  const double k1_;
  const double k2_;
  const double A_;
};

class MixtureBoundary {
public:
  MixtureBoundary(std::unique_ptr<MixtureSupermartingale>&& mixture_superMG)
      : mixture_superMG_(std::move(mixture_superMG)) {}

  double operator()(const double v, const double alpha) const {
    return mixture_superMG_->bound(v, log(1 / alpha));
  }

private:
  std::unique_ptr<MixtureSupermartingale> mixture_superMG_;
};

class EmpiricalProcessLILBound {
 public:
  EmpiricalProcessLILBound(const double alpha, const double t_min,
                           const double A)
      : t_min_(t_min), A_(A) {
    assert(A > 1 / sqrt(2));
    assert(t_min >= 1);
    assert(0 < alpha && alpha < 1);
    C_ = find_optimal_C(alpha, A);
  }

  double operator()(const double t) const;

 private:
  static double find_optimal_C(const double alpha, const double A);

  const double t_min_;
  const double A_;
  double C_;
};

class DoubleStitchingBound {
 public:
  DoubleStitchingBound(const int t_opt, const double delta, const double s,
                       const double eta)
      : t_opt_(t_opt), delta_(delta), s_(s), eta_(eta),
        k1_((pow(eta, .25) + pow(eta, -.25)) / sqrt(2)),
        k2_((sqrt(eta) + 1) / 2)
  {
    assert(t_opt_ >= 1);
    assert(delta_ > 0);
    assert(s > 0);
    assert(eta > 0);
  }

  double operator()(const double p, const double t, const double alpha) const;

 private:
  const double t_opt_;
  const double delta_;
  const double s_;
  const double eta_;
  const double k1_;
  const double k2_;
};

class OrderStatisticInterface {
 public:
  virtual ~OrderStatisticInterface() {}
  virtual double get_order_statistic(const int order_index) const = 0;
  virtual int count_less(const double value) const = 0;
  virtual int count_less_or_equal(const double value) const = 0;
  virtual int size() const = 0;
};

class StaticOrderStatistics : public OrderStatisticInterface {
 public:
  template <class InputIt> StaticOrderStatistics(InputIt first, InputIt last)
      : sorted_values_(first, last) {
    std::sort(sorted_values_.begin(), sorted_values_.end());
  }

  virtual double get_order_statistic(const int order_index) const override {
    return sorted_values_[order_index - 1];
  }

  virtual int count_less(const double value) const override {
    return std::lower_bound(sorted_values_.begin(), sorted_values_.end(), value)
        - sorted_values_.begin();
  }
  virtual int count_less_or_equal(const double value) const override {
    return std::upper_bound(sorted_values_.begin(), sorted_values_.end(), value)
        - sorted_values_.begin();
  }

  virtual int size() const override {
    return sorted_values_.size();
  }

 private:
  std::vector<double> sorted_values_;
};

class QuantileABTest {
 public:
  QuantileABTest(const double quantile_p, const int t_opt,
                 const double alpha_opt,
                 std::shared_ptr<OrderStatisticInterface> arm1_os,
                 std::shared_ptr<OrderStatisticInterface> arm2_os)
      : quantile_p_(quantile_p),
        mixture_(std::make_unique<BetaBinomialMixture>(
            t_opt * quantile_p * (1 - quantile_p), alpha_opt, quantile_p,
            1 - quantile_p, false)),
        arm1_os_(arm1_os), arm2_os_(arm2_os) {
    assert(0 < quantile_p && quantile_p < 1);
  }

  double p_value() const;

 private:
  struct GFunction {
    const std::function<double(const double)> G;
    const double minimum_start_x;
    const double minimum_end_x;
  };

  double log_superMG_lower_bound() const;
  GFunction get_G_fn(const int arm) const;
  double empirical_quantile(const int arm) const;
  double find_log_superMG_lower_bound(const GFunction first_arm_G,
                                      const GFunction second_arm_G,
                                      const int second_arm) const;
  double arm_log_superMG(const int arm, const double prop_below) const;
  const std::shared_ptr<OrderStatisticInterface>& order_stats(const int arm)
      const;

  const double quantile_p_;
  const std::unique_ptr<MixtureSupermartingale> mixture_;
  const std::shared_ptr<OrderStatisticInterface> arm1_os_;
  const std::shared_ptr<OrderStatisticInterface> arm2_os_;
};

//////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////

inline double find_s_upper_bound(const MixtureSupermartingale& mixture_superMG,
                                 const double v, const double log_threshold) {
  double trial_upper_bound = v;
  for (int i = 0; i < 50; i++) {
    if (mixture_superMG.log_superMG(trial_upper_bound, v) > log_threshold) {
      return trial_upper_bound;
    }
    trial_upper_bound *= 2;
  }
  throw std::runtime_error(
      "Failed to find an upper limit for the mixture bound");
}

inline double find_mixture_bound(const MixtureSupermartingale& mixture_superMG,
                                 const double v, const double log_threshold) {
  auto root_fn = [&mixture_superMG, v, log_threshold](const double s) {
    return mixture_superMG.log_superMG(s, v) - log_threshold;
  };
  double s_upper_bound = mixture_superMG.s_upper_bound(v);
  if (s_upper_bound == std::numeric_limits<double>::infinity()) {
    s_upper_bound = find_s_upper_bound(mixture_superMG, v, log_threshold);
  }
  if (root_fn(s_upper_bound) < 0) {
    return s_upper_bound;
  } else {
    auto result = boost::math::tools::bisect(
        root_fn, 0.0, s_upper_bound,
        boost::math::tools::eps_tolerance<double>(40));
    return (result.first + result.second) / 2;
  }
}

inline double TwoSidedNormalMixture::log_superMG(const double s, const double v)
    const {
  return (1 / 2.0) * log(rho_ / (v + rho_)) + s * s / (2 * (v + rho_));
}

inline double TwoSidedNormalMixture::bound(const double v,
                                           const double log_threshold) const {
  return sqrt((v + rho_) * (log(1 + v / rho_) + 2 * log_threshold));
}

inline double TwoSidedNormalMixture::best_rho(double v, double alpha) {
  assert(v > 0);
  assert(0 < alpha && alpha < 1);
  return v / (2 * log(1 / alpha) + log(1 + 2 * log(1 / alpha)));
}

inline double OneSidedNormalMixture::log_superMG(const double s, const double v)
    const {
  boost::math::normal_distribution<double> normal;
  return (1 / 2.0) * log(4 * rho_ / (v + rho_)) + s * s / (2 * (v + rho_))
      + log(boost::math::cdf(normal, s / sqrt(v + rho_)));
}

inline double OneSidedNormalMixture::best_rho(double v, double alpha) {
  return TwoSidedNormalMixture::best_rho(v, 2 * alpha);
}

inline double GammaExponentialMixture::get_leading_constant(double rho,
                                                            double c) {
  const double rho_c_sq = rho / (c * c);
  return rho_c_sq * log(rho_c_sq)
      - boost::math::lgamma(rho_c_sq)
      - log(boost::math::gamma_p(rho_c_sq, rho_c_sq));
}

inline double GammaExponentialMixture::log_superMG(const double s,
                                                   const double v)
    const {
  const double c_sq = c_ * c_;
  const double cs_v_csq = (c_ * s + v) / c_sq;
  const double v_rho_csq = (v + rho_) / c_sq;
  return leading_constant_
      + boost::math::lgamma(v_rho_csq)
      + log(boost::math::gamma_p(v_rho_csq, cs_v_csq + rho_ / c_sq))
      - v_rho_csq * log(cs_v_csq + rho_ / c_sq)
      + cs_v_csq;
}

inline double GammaPoissonMixture::get_leading_constant(double rho, double c) {
  const double rho_c_sq = rho / (c * c);
  return rho_c_sq * log(rho_c_sq)
      - boost::math::lgamma(rho_c_sq)
      - log(boost::math::gamma_q(rho_c_sq, rho_c_sq));
}

inline double GammaPoissonMixture::log_superMG(const double s, const double v)
    const {
  const double c_sq = c_ * c_;
  const double v_rho_csq = (v + rho_) / c_sq;
  const double cs_v_rho_csq = s / c_ + v_rho_csq;
  return leading_constant_
      + boost::math::lgamma(cs_v_rho_csq)
      + log(boost::math::gamma_q(cs_v_rho_csq, v_rho_csq))
      - cs_v_rho_csq * log(v_rho_csq)
      + v / c_sq;
}

inline double log_beta(const double a, const double b) {
  return boost::math::lgamma(a) + boost::math::lgamma(b)
      - boost::math::lgamma(a + b);
}

inline double log_incomplete_beta(const double a, const double b,
                                  const double x) {
  if (x == 1) {
    return log_beta(a, b);
  } else {
    return log(boost::math::ibeta(a, b, x)) + log_beta(a, b);
  }
}

inline double BetaBinomialMixture::log_superMG(const double s, const double v)
    const {
  const double x = is_one_sided_ ? h_ / (g_ + h_) : 1;
  return
      v / (g_ * h_) * log(g_ + h_)
      - ((v + h_ * s) / (h_ * (g_ + h_))) * log(g_)
      - ((v - g_ * s) / (g_ * (g_ + h_))) * log(h_)
      + log_incomplete_beta((r_ + v - g_ * s) /  (g_ * (g_ + h_)),
                            (r_ + v + h_ * s) / (h_ * (g_ + h_)),
                            x)
      - normalizer_;
}

inline double BetaBinomialMixture::s_upper_bound(const double v) const {
  return v / g_;
}

inline double PolyStitchingBound::operator()(double v, double alpha) const {
  double use_v = std::max(v, v_min_);
  double ell = s_ * log(log(eta_ * use_v / v_min_)) + A_ + log(1 / alpha);
  double term2 = k2_ * c_ * ell;
  return sqrt(k1_ * k1_ * use_v * ell + term2 * term2) + term2;
}

inline double EmpiricalProcessLILBound::operator()(const double t) const {
  if (t < t_min_) {
    return std::numeric_limits<double>::infinity();
  } else {
    return A_ * sqrt((log(1 + log(t / t_min_)) + C_) / t);
  }
}

inline double EmpiricalProcessLILBound::find_optimal_C(const double alpha,
                                                const double A) {
  using namespace std::placeholders;

  auto error_bound = [A](const double C, const double eta) {
    const double gamma_sq = 2 / eta * pow(A - sqrt(2 * (eta - 1) / C), 2);
    if (gamma_sq <= 1) {
      return std::numeric_limits<double>::infinity();
    } else {
      return 4 * exp(-gamma_sq * C) * (1 + 1 / ((gamma_sq - 1) * log(eta)));
    }
  };

  auto optimum_error = [A, error_bound](const double C) {
    auto eta_upper_result = boost::math::tools::bisect(
        [A, C](const double eta) {
          return sqrt(eta / 2) + sqrt(2 * (eta - 1) / C) - A;
        },
        1.0, 2 * A * A, boost::math::tools::eps_tolerance<double>(40));
    double eta_upper = (eta_upper_result.first + eta_upper_result.second) / 2;
    return boost::math::tools::brent_find_minima(std::bind(error_bound, C, _1),
                                                 1.0, eta_upper, 24);
  };

  boost::uintmax_t max_iter = 50;
  auto C_result = boost::math::tools::bracket_and_solve_root(
      [alpha, optimum_error](const double C) {
        return optimum_error(C).second - alpha;
      },
      5.0,
      2.0,
      false,
      boost::math::tools::eps_tolerance<double>(40),
      max_iter);
  return (C_result.first + C_result.second) / 2;
}

inline double logit(const double p) {
  assert(0 < 1 && p < 1);
  return log(p / (1 - p));
}

inline double expit(const double l) {
  return 1 / (1 + exp(-l));
}

inline double DoubleStitchingBound::operator()(const double p, const double t,
                                        const double alpha) const {
  const double t_max_m = std::max(t, t_opt_);
  const double r = p >= 0.5 ? p
      : std::min(0.5, expit(logit(p) + 2 * delta_ *
                            sqrt(t_opt_ * eta_ / t_max_m)));
  const double sigma_sq = r * (1 - r);
  const double j = sqrt(t_max_m / t_opt_) * abs(logit(p)) / (2 * delta_) + 1;
  const double zeta_s = boost::math::zeta(s_);
  const double ell = s_ * log(log(eta_ * t_max_m / t_opt_)) + s_ * log(j)
       + log(2 * zeta_s * (2 * zeta_s + 1) / (alpha * pow(log(eta_), s_)));
  const double cp = (1 - 2 * p) / 3;
  const double term2 = k2_ * cp * ell;
  return delta_ * sqrt(eta_ * t_max_m * sigma_sq / t_opt_)
      + sqrt(k1_ * k1_ * sigma_sq * t_max_m * ell + term2 * term2) + term2;
}

inline double QuantileABTest::p_value() const {
  return std::min(1.0, exp(-log_superMG_lower_bound()));
}

inline double QuantileABTest::log_superMG_lower_bound() const {
  const GFunction arm_1_G = get_G_fn(1);
  const GFunction arm_2_G = get_G_fn(2);
  /* An optimization for another day:
  const double middle = (empirical_quantile(0) + empirical_quantile(1)) / 2;
  const double mid_value = arm_1_G.G(middle) + arm_2_G.G(middle);
  if (mid_value < log_alpha_inv_) {
    return mid_value;
  } else */
  if (arm_1_G.minimum_end_x <= arm_2_G.minimum_end_x) {
    return find_log_superMG_lower_bound(arm_1_G, arm_2_G, 2);
  } else {
    return find_log_superMG_lower_bound(arm_2_G, arm_1_G, 1);
  }
}

inline double QuantileABTest::empirical_quantile(const int arm) const {
  int position = floor(order_stats(arm)->size() * quantile_p_) + 1;
  return order_stats(arm)->get_order_statistic(position);
}

inline double QuantileABTest::arm_log_superMG(const int arm,
                                              const double prop_below)
    const {
  int N = order_stats(arm)->size();
  double s = (prop_below - quantile_p_) * N;
  double v = quantile_p_ * (1 - quantile_p_) * N;
  return mixture_->log_superMG(s, v);
}

inline QuantileABTest::GFunction QuantileABTest::get_G_fn(const int arm) const {
  auto objective = [this, arm](double a) {return arm_log_superMG(arm, a);};
  double minimizer = boost::math::tools::brent_find_minima(
      objective, 0.0, 1.0, 20).first;

  int N = order_stats(arm)->size();
  double x_lower = order_stats(arm)->get_order_statistic(ceil(minimizer * N));
  double x_upper =
      order_stats(arm)->get_order_statistic(floor(minimizer * N) + 1);

  auto G_callable = [this, arm, minimizer, x_lower, x_upper]
      (const double x) {
    double prop_below;
    if (x < x_lower) {
      prop_below = double(order_stats(arm)->count_less_or_equal(x))
                   / order_stats(arm)->size();
    } else if (x > x_upper) {
      prop_below = double(order_stats(arm)->count_less(x))
                   / order_stats(arm)->size();
    } else {
      prop_below = minimizer;
    }
    return arm_log_superMG(arm, prop_below);
  };
  return GFunction{G_callable, x_lower, x_upper};
}

inline double QuantileABTest::find_log_superMG_lower_bound(
    const GFunction first_arm_G, const GFunction second_arm_G,
    const int second_arm) const {
  assert(first_arm_G.minimum_end_x <= second_arm_G.minimum_end_x);
  auto objective = [first_arm_G, second_arm_G](double x) {
    return first_arm_G.G(x) + second_arm_G.G(x);
  };
  double min_value = std::min(
      objective(first_arm_G.minimum_end_x),
      objective(second_arm_G.minimum_start_x));
  int start_index = order_stats(second_arm)->count_less_or_equal(
      first_arm_G.minimum_end_x);
  int end_index = order_stats(second_arm)->count_less_or_equal(

      second_arm_G.minimum_start_x);
  assert(start_index <= end_index);
  assert(end_index >= 1);

  for (int i = std::max(1, start_index); i <= end_index; i++) {
    double x = order_stats(second_arm)->get_order_statistic(i);
    i = order_stats(second_arm)->count_less_or_equal(x);
    double value = objective(x);
    if (value < min_value) {
      min_value = value;
    }
  }
  return min_value;
}

inline const std::shared_ptr<OrderStatisticInterface>&
QuantileABTest::order_stats(
    const int arm) const {
  assert(arm == 1 || arm == 2);
  return arm == 1 ? arm1_os_ : arm2_os_;
}

//////////////////////////////////////////////////////////////////////
// Simplified interface implementation
//////////////////////////////////////////////////////////////////////

inline double normal_log_mixture(
    const double s, const double v, const double v_opt, const double alpha_opt,
    const bool is_one_sided) {
  if (is_one_sided) {
    OneSidedNormalMixture mixture(v_opt, alpha_opt);
    return mixture.log_superMG(s, v);
  } else {
    TwoSidedNormalMixture mixture(v_opt, alpha_opt);
    return mixture.log_superMG(s, v);
  }
}

inline double normal_mixture_bound(
    const double v, const double alpha, const double v_opt,
    const double alpha_opt, const bool is_one_sided) {
  if (is_one_sided) {
    OneSidedNormalMixture mixture(v_opt, alpha_opt);
    return mixture.bound(v, log(1 / alpha));
  } else {
    TwoSidedNormalMixture mixture(v_opt, alpha_opt);
    return mixture.bound(v, log(1 / alpha));
  }
}

inline double gamma_exponential_log_mixture(
    const double s, const double v, const double v_opt,
    const double c, const double alpha_opt) {
  GammaExponentialMixture mixture(v_opt, alpha_opt, c);
  return mixture.log_superMG(s, v);
}

inline double gamma_exponential_mixture_bound(
    const double v, const double alpha, const double v_opt,
    const double c, const double alpha_opt) {
  GammaExponentialMixture mixture(v_opt, alpha_opt, c);
  return mixture.bound(v, log(1 / alpha));
}

inline double gamma_poisson_log_mixture(
    const double s, const double v, const double v_opt, const double c,
    const double alpha_opt) {
  GammaPoissonMixture mixture(v_opt, alpha_opt, c);
  return mixture.log_superMG(s, v);
}

inline double gamma_poisson_mixture_bound(
    const double v, const double alpha, const double v_opt, const double c,
    const double alpha_opt) {
  GammaPoissonMixture mixture(v_opt, alpha_opt, c);
  return mixture.bound(v, log(1 / alpha));
}

inline double beta_binomial_log_mixture(
    const double s, const double v, const double v_opt, const double g,
    const double h, const double alpha_opt,
    const bool is_one_sided) {
  BetaBinomialMixture mixture(v_opt, alpha_opt, g, h, is_one_sided);
  return mixture.log_superMG(s, v);
}

inline double beta_binomial_mixture_bound(
    const double v, const double alpha, const double v_opt, const double g,
    const double h, const double alpha_opt,
    const bool is_one_sided) {
  BetaBinomialMixture mixture(v_opt, alpha_opt, g, h, is_one_sided);
  return mixture.bound(v, log(1 / alpha));
}

inline double poly_stitching_bound(const double v, const double alpha,
                            const double v_min, const double c,
                            const double s, const double eta) {
  PolyStitchingBound bound(v_min, c, s, eta);
  return bound(v, alpha);
}

inline double empirical_process_lil_bound(const int t, const double alpha,
                                   const double t_min, const double A) {
  EmpiricalProcessLILBound bound(alpha, t_min, A);
  return bound(t);
}

inline double double_stitching_bound(const double quantile_p, const double t,
                              const double alpha, const double t_opt,
                              const double delta, const double s,
                              const double eta) {
  DoubleStitchingBound bound(t_opt, delta, s, eta);
  return bound(quantile_p, t, alpha);
}

inline double pair_average(std::pair<double, double> values) {
  return (values.first + values.second) / 2;
}

inline std::pair<double, double> bernoulli_confidence_interval(
    const double num_successes, const int num_trials, const double alpha,
    const double t_opt, const double alpha_opt) {
  using namespace std::placeholders;
  const double threshold = log(1 / alpha);
  const double empirical_p = 1.0 * num_successes / num_trials;

  auto objective = [empirical_p, num_trials, t_opt, alpha_opt, threshold]
                   (const double p, const double zero_value,
                    const double one_value) {
    if (p <= 0) {
      return zero_value;
    } else if (p >= 1) {
      return one_value;
    } else {
      const BetaBinomialMixture mixture(p * (1 - p) * t_opt, alpha_opt,
                                        p, 1 - p, false);
      const double log_superMG = mixture.log_superMG(
          (empirical_p - p) * num_trials, p * (1 - p) * num_trials);
      return log_superMG - threshold;
    }
  };

  boost::math::tools::eps_tolerance<double> tolerance(40);
  double lower_bound = 0.0;
  if (empirical_p > 0) {
    auto lower_bound_pair = boost::math::tools::bisect(
        std::bind(objective, _1, 1.0, -1.0), 0.0, empirical_p, tolerance);
    lower_bound = pair_average(lower_bound_pair);
  }
  double upper_bound = 1.0;
  if (empirical_p < 1) {
    auto upper_bound_pair = boost::math::tools::bisect(
        std::bind(objective, _1, -1.0, 1.0), empirical_p, 1.0, tolerance);
    upper_bound = pair_average(upper_bound_pair);
  }
  return std::make_pair(lower_bound, upper_bound);
}

}; // namespace confseq

#endif // CONFIDENCESEQUENCES_UNIFORM_BOUNDARIES_H_
