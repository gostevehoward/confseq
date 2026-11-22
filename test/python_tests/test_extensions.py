"""Smoke tests for the compiled pybind11 extensions.

These make sure the C++ modules build, import and run (including the Boost.Math
special functions and root finders they rely on). Detailed correctness tests
for the boundaries live in the C++ unit tests.
"""
import numpy as np

import confseq.boundaries as b
import confseq.quantiles as q


def _check(values):
    values = np.asarray(values)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)
    # Boundaries grow with intrinsic time / sample size
    assert np.all(np.diff(values) > 0)
    return values


V = np.array([10.0, 100.0, 1000.0, 10000.0])


def test_mixture_bounds():
    _check(b.normal_mixture_bound(V, 0.05, 100.0))
    _check(b.normal_mixture_bound(V, 0.05, 100.0, is_one_sided=False))
    _check(b.gamma_exponential_mixture_bound(V, 0.05, 100.0, 1.0))
    _check(b.gamma_poisson_mixture_bound(V, 0.05, 100.0, 1.0))
    _check(b.beta_binomial_mixture_bound(V, 0.05, 100.0, 0.5, 0.5))


def test_log_mixtures():
    for log_mix in [
        b.normal_log_mixture(5.0, V, 100.0),
        b.gamma_exponential_log_mixture(5.0, V, 100.0, 1.0),
        b.gamma_poisson_log_mixture(5.0, V, 100.0, 1.0),
        b.beta_binomial_log_mixture(5.0, V, 100.0, 0.5, 0.5),
    ]:
        assert np.all(np.isfinite(log_mix))


def test_poly_stitching_bound():
    _check(b.poly_stitching_bound(V, 0.05, 10.0))


def test_bernoulli_confidence_interval():
    lower, upper = b.bernoulli_confidence_interval(60, 100, 0.05, 100)
    assert 0 <= lower < 0.6 < upper <= 1


def test_quantile_bounds():
    t = np.array([100.0, 1000.0, 10000.0])
    lil = q.empirical_process_lil_bound(t, 0.05, 100.0)
    stitch = q.double_stitching_bound(0.5, t, 0.05, 500.0)
    for bound in [lil, stitch]:
        assert np.all(np.isfinite(bound))
        assert np.all(bound > 0)
