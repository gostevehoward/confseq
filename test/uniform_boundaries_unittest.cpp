#include <array>
#include <numeric>

#include "uniform_boundaries.h"

#include "gtest/gtest.h"

namespace {

using namespace confseq;

// all target values are from a reference implementation in R

const double V_OPT = 100;
const double ALPHA_OPT = 0.05;
const double ALPHA = 0.05;

TEST(MixtureTest, TestTwoSidedNormalMixture) {
  EXPECT_NEAR(TwoSidedNormalMixture::best_rho(V_OPT, ALPHA_OPT), 12.60056,
              1e-5);
  EXPECT_NEAR(normal_log_mixture(10, 100, V_OPT, ALPHA_OPT, false),
              -0.6510052, 1e-5);
  EXPECT_NEAR(normal_mixture_bound(100, ALPHA, V_OPT, ALPHA_OPT, false),
              30.35209, 1e-5);
}

TEST(MixtureTest, TestOneSidedNormalMixture) {
  EXPECT_NEAR(normal_log_mixture(10, 100, V_OPT), -0.06502391, 1e-5);
  EXPECT_NEAR(normal_mixture_bound(100, ALPHA, V_OPT), 27.66071, 1e-5);
}

TEST(MixtureTest, TestGammaExponentialMixture) {
  EXPECT_NEAR(gamma_exponential_log_mixture(10, 100, V_OPT, 2),
              -0.2490165, 1e-5);
  EXPECT_NEAR(gamma_exponential_mixture_bound(100, ALPHA, V_OPT, 2), 33.02017,
              1e-5);
}

TEST(MixtureTest, TestGammaPoissonMixture) {
  EXPECT_NEAR(gamma_poisson_log_mixture(10, 100, V_OPT, 2), -0.06991219, 1e-5);
  EXPECT_NEAR(gamma_poisson_mixture_bound(100, ALPHA, V_OPT, 2), 30.1949, 1e-5);
}

TEST(MixtureTest, TestTwoSidedBetaBinomialMixture) {
  EXPECT_NEAR(
      beta_binomial_log_mixture(10, 100, V_OPT, 0.2, 0.8, ALPHA_OPT, false),
      -0.6941012, 1e-5);
  EXPECT_NEAR(
      beta_binomial_mixture_bound(100, ALPHA, V_OPT, 0.2, 0.8, ALPHA_OPT,
                                  false),
      31.4308, 1e-5);
}

TEST(MixtureTest, TestOneSidedBetaBinomialMixture) {
  EXPECT_NEAR(beta_binomial_log_mixture(10, 100, V_OPT, 0.2, 0.8),
              -0.07134019, 1e-5);
  EXPECT_NEAR(beta_binomial_mixture_bound(100, ALPHA, V_OPT, 0.2, 0.8),
              28.41238, 1e-5);
  EXPECT_NEAR(
      beta_binomial_mixture_bound(2 * 0.2 * 0.8, 0.05, 100, 0.2, 0.8),
      2 * 0.8, 1e-5);
}

TEST(PolyStitchingTest, BasicTest) {
  EXPECT_NEAR(poly_stitching_bound(100, ALPHA, 10, 3), 64.48755, 1e-5);
}

TEST(PolyStitchingTest, NegativeC) {
  EXPECT_NEAR(poly_stitching_bound(100, ALPHA, 10, -1), 28.99389, 1e-5);
}

TEST(EmpiricalProcessLILTest, TestBound) {
  EXPECT_NEAR(empirical_process_lil_bound(1000, .05, 100, 0.85),
              0.08204769, 1e-5);
}

TEST(DoubleStitchingTest, TestBound) {
  EXPECT_NEAR(double_stitching_bound(0.5, 1000, 0.05, 100), 68.62803, 1e-5);
  EXPECT_NEAR(double_stitching_bound(0.9, 1000, 0.05, 100), 43.72119, 1e-5);
  EXPECT_NEAR(double_stitching_bound(0.1, 1000, 0.05, 100), 59.96521, 1e-5);
}

TEST(StaticOrderStatisticsTest, TestBasics) {
  std::array<int, 5> values = {1, 2, 3, 3, 5};
  StaticOrderStatistics os(values.begin(), values.end());
  EXPECT_EQ(os.size(), 5);
  EXPECT_EQ(os.get_order_statistic(1), 1);
  EXPECT_EQ(os.get_order_statistic(4), 3);
  EXPECT_EQ(os.get_order_statistic(5), 5);
  EXPECT_EQ(os.count_less(-1), 0);
  EXPECT_EQ(os.count_less_or_equal(-1), 0);
  EXPECT_EQ(os.count_less(1), 0);
  EXPECT_EQ(os.count_less_or_equal(1), 1);
  EXPECT_EQ(os.count_less(3), 2);
  EXPECT_EQ(os.count_less_or_equal(3), 4);
  EXPECT_EQ(os.count_less(5), 4);
  EXPECT_EQ(os.count_less_or_equal(5), 5);
  EXPECT_EQ(os.count_less(10), 5);
  EXPECT_EQ(os.count_less_or_equal(10), 5);
}

double get_ab_p_value(const double quantile_p, const int offset) {
  std::array<int, 1000> a_values;
  std::iota(a_values.begin(), a_values.end(), 1);
  std::array<int, 1000> b_values;
  std::iota(b_values.begin(), b_values.end(), offset + 1);

  auto a_os = std::make_shared<StaticOrderStatistics>(a_values.begin(),
                                                      a_values.end());
  auto b_os = std::make_shared<StaticOrderStatistics>(b_values.begin(),
                                                      b_values.end());
  QuantileABTest test(quantile_p, 100, 0.05, a_os, b_os);

  return test.p_value();
}

TEST(QuantileABTest, TestPValue) {
  EXPECT_NEAR(get_ab_p_value(0.5, 70), 0.814704, 1e-5);
  EXPECT_NEAR(get_ab_p_value(0.5, 85), 0.0838055, 1e-5);
  EXPECT_NEAR(get_ab_p_value(0.5, 100), 0.0054947, 1e-5);
  EXPECT_NEAR(get_ab_p_value(0.9, 55), 0.0340463, 1e-5);
}

TEST(BernoulliConfidenceIntervalTest, TestCI) {
  std::pair<double, double> ci =
      bernoulli_confidence_interval(700, 1000, 0.05, 100);
  EXPECT_NEAR(ci.first, 0.651629, 1e-5);
  EXPECT_NEAR(ci.second, 0.745949, 1e-5);
}

TEST(BernoulliConfidenceIntervalTest, LowSuccessCount) {
  std::pair<double, double> ci =
      bernoulli_confidence_interval(15, 50, 0.05, 50);
  EXPECT_NEAR(ci.first, 0.129556, 1e-5);
  EXPECT_NEAR(ci.second, 0.511896, 1e-5);

  ci = bernoulli_confidence_interval(5, 50, 0.05, 50);
  EXPECT_NEAR(ci.first, 0.013021, 1e-5);
  EXPECT_NEAR(ci.second, 0.269264, 1e-5);

  ci = bernoulli_confidence_interval(1, 50, 0.05, 50);
  EXPECT_NEAR(ci.first, 0.0, 1e-5);
  EXPECT_NEAR(ci.second, 0.13489, 1e-5);
}

TEST(BernoulliConfidenceIntervalTest, LowTrials) {
  std::pair<double, double> ci =
      bernoulli_confidence_interval(1, 2, 0.05, 50);
  EXPECT_NEAR(ci.first, 0.0, 1e-5);
  EXPECT_NEAR(ci.second, 1.0, 1e-5);
}

TEST(BernoulliConfidenceIntervalTest, LowOptimizedTime) {
  std::pair<double, double> ci =
      bernoulli_confidence_interval(100, 200, 0.05, 1);
  EXPECT_NEAR(ci.first, 0.328923, 1e-5);
  EXPECT_NEAR(ci.second, 0.671076, 1e-5);
}

TEST(BernoulliConfidenceIntervalTest, NoSuccesses) {
  std::pair<double, double> ci =
      bernoulli_confidence_interval(0, 1, 0.05, 50);
  EXPECT_NEAR(ci.first, 0.0, 1e-5);
  EXPECT_NEAR(ci.second, 1.0, 1e-5);
}

TEST(BernoulliConfidenceIntervalTest, AllSuccesses) {
  std::pair<double, double> ci =
      bernoulli_confidence_interval(1, 1, 0.05, 50);
  EXPECT_NEAR(ci.first, 0.0, 1e-5);
  EXPECT_NEAR(ci.second, 1.0, 1e-5);
}

} // namespace
