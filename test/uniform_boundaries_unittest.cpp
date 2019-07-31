#include <random>

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

} // namespace
