#include <array>
#include <numeric>

#include "uniform_boundaries.h"

#include "gtest/gtest.h"

namespace {

// all target values are from a reference implementation in R

TEST(MixtureTest, TestTwoSidedNormalMixture) {
  EXPECT_NEAR(confseq::TwoSidedNormalMixture::best_rho(100, 0.05), 12.60056,
              1e-5);

  confseq::TwoSidedNormalMixture mixture(100, 0.05);
  EXPECT_NEAR(mixture.log_superMG(10, 100), -0.6510052, 1e-5);
  EXPECT_NEAR(mixture.bound(100, log(1 / 0.05)), 30.35209, 1e-5);
}

TEST(MixtureTest, TestOneSidedNormalMixture) {
  confseq::OneSidedNormalMixture mixture(100, 0.05);
  EXPECT_NEAR(mixture.log_superMG(10, 100), -0.06502391, 1e-5);
  EXPECT_NEAR(mixture.bound(100, log(1 / 0.05)), 27.66071, 1e-5);
}

TEST(MixtureTest, TestGammaExponentialMixture) {
  confseq::GammaExponentialMixture mixture(100, 0.05, 2);
  EXPECT_NEAR(mixture.log_superMG(10, 100), -0.2490165, 1e-5);
  EXPECT_NEAR(mixture.bound(100, log(1 / 0.05)), 33.02017, 1e-5);
}

TEST(MixtureTest, TestGammaPoissonMixture) {
  confseq::GammaPoissonMixture mixture(100, 0.05, 2);
  EXPECT_NEAR(mixture.log_superMG(10, 100), -0.06991219, 1e-5);
  EXPECT_NEAR(mixture.bound(100, log(1 / 0.05)), 30.1949, 1e-5);
}

TEST(MixtureTest, TestTwoSidedBetaBinomialMixture) {
  confseq::BetaBinomialMixture mixture(100, 0.05, 0.2, 0.8, false);
  EXPECT_NEAR(mixture.log_superMG(10, 100), -0.6941012, 1e-5);
  EXPECT_NEAR(mixture.bound(100, log(1 / 0.05)), 31.4308, 1e-5);
}

TEST(MixtureTest, TestOneSidedBetaBinomialMixture) {
  confseq::BetaBinomialMixture mixture(100, 0.05, 0.2, 0.8, true);
  EXPECT_NEAR(mixture.log_superMG(10, 100), -0.07134019, 1e-5);
  EXPECT_NEAR(mixture.bound(100, log(1 / 0.05)), 28.41238, 1e-5);
}

TEST(PolyStitchingTest, BasicTest) {
  confseq::PolyStitchingBound bound(10, 3, 1.4, 2);
  EXPECT_NEAR(bound(100, 0.05), 64.48755, 1e-5);
}

TEST(PolyStitchingTest, NegativeC) {
  confseq::PolyStitchingBound bound(10, -1, 1.4, 2);
  EXPECT_NEAR(bound(100, 0.05), 28.99389, 1e-5);
}

TEST(EmpiricalProcessLILTest, TestBound) {
  EXPECT_NEAR(confseq::empirical_process_lil_bound(1000, .05, 100, 0.85),
              0.08204769, 1e-5);
}

TEST(DoubleStitchingTest, TestBound) {
  EXPECT_NEAR(confseq::double_stitching_bound(0.5, 1000, 0.05, 100),
              68.62803, 1e-5);
  EXPECT_NEAR(confseq::double_stitching_bound(0.9, 1000, 0.05, 100),
              43.72119, 1e-5);
  EXPECT_NEAR(confseq::double_stitching_bound(0.1, 1000, 0.05, 100),
              59.96521, 1e-5);
}

TEST(StaticOrderStatisticsTest, TestBasics) {
  std::array<int, 5> values = {1, 2, 3, 3, 5};
  confseq::StaticOrderStatistics os(values.begin(), values.end());
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

  auto a_os = std::make_unique<confseq::StaticOrderStatistics>(
      a_values.begin(), a_values.end());
  auto b_os = std::make_unique<confseq::StaticOrderStatistics>(
      b_values.begin(), b_values.end());
  confseq::QuantileABTest test(quantile_p, 100, 0.05, std::move(a_os),
                               std::move(b_os));

  return test.p_value();
}

TEST(QuantileABTest, TestPValue) {
  EXPECT_NEAR(get_ab_p_value(0.5, 70), 0.814704, 1e-5);
  EXPECT_NEAR(get_ab_p_value(0.5, 85), 0.0838055, 1e-5);
  EXPECT_NEAR(get_ab_p_value(0.5, 100), 0.0054947, 1e-5);
  EXPECT_NEAR(get_ab_p_value(0.9, 55), 0.0340463, 1e-5);
}

} // namespace
