from math import isnan
import pytest
import numpy as np
from confseq.betting import *
from confseq.misc import superMG_crossing_fraction, expand_grid
from scipy.stats import binomtest


@pytest.mark.random
@pytest.mark.parametrize("theta", [0, 0.5, 1])
def test_betting_mart_crossing_probabilities(theta):
    # Note that these tests are random and will each individually
    # fail at most 5% of the time.

    for m in [0.2, 0.5, 0.8]:
        repeats = 500
        alpha = 0.1
        dist_fn = lambda: np.random.binomial(1, m, 10000)
        mart_fn = lambda x: betting_mart(x, m, alpha=alpha, theta=theta)

        crossing_frac = superMG_crossing_fraction(
            mart_fn, dist_fn, alpha=alpha, repeats=repeats
        )

        crossing_test = binomtest(
            int(crossing_frac * repeats), n=repeats, p=alpha, alternative="greater"
        )

        lower_ci = crossing_test.proportion_ci(confidence_level=0.95)[0]

        assert lower_ci > 0
        assert lower_ci <= alpha


@pytest.mark.random
def test_betting_mart_power():
    # Make sure theta=0,1 have one-sided power, while theta=1/2 has two-sided power
    theta = 1 / 2
    x = np.random.binomial(1, 0.5, 10000)
    mart1 = betting_mart(x, 0.25, theta=theta)
    mart2 = betting_mart(x, 0.75, theta=theta)

    # Should have two-sided power
    # This will fail with some small probability
    assert any(mart1 > 20)
    assert any(mart2 > 20)

    theta = 1
    mart1 = betting_mart(x, 0.4, theta=theta)
    mart2 = betting_mart(x, 0.6, theta=theta)

    # Should only have power against 0.4, but not 0.6
    assert any(mart1 > 20)
    assert not any(mart2 > 20)

    theta = 0
    mart1 = betting_mart(x, 0.4, theta=theta)
    mart2 = betting_mart(x, 0.6, theta=theta)

    # Should only have power against 0.6, but not 0.4
    assert not any(mart1 > 20)
    assert any(mart2 > 20)


@pytest.mark.parametrize("m", [0.1, 0.4, 0.5, 0.6, 0.9])
def test_betting_mart_convex_comb(m):
    # Convex combination should always be larger than maximum
    x = np.random.beta(1, 1, 10000)
    mart1 = betting_mart(x, m, theta=1 / 2, convex_comb=True)
    mart2 = betting_mart(x, m, theta=1 / 2, convex_comb=False)
    assert all(mart1 >= mart2)


@pytest.mark.random
def test_betting_mart_WoR():
    N = 1000
    x = np.random.binomial(1, 0.5, N)
    alpha = 0.05

    assert betting_mart(x, m=np.mean(x), N=N)[-1] < 1 / alpha
    # Martingale should be large for all m_null not equal to m.
    # This may fail with small probability
    assert betting_mart(x, m=np.mean(x) + 0.01, N=N)[-1] > 1 / alpha
    assert betting_mart(x, m=np.mean(x) - 0.01, N=N)[-1] > 1 / alpha


@pytest.mark.parametrize("m", [0.4, 0.5, 0.6])
def test_diversified_betting_mart(m):
    # Betting mart should be the same as averaging a bunch of betting_marts
    n = 1000
    x = np.random.beta(1, 1, n)
    K = 1
    div_mart1 = diversified_betting_mart(
        x,
        m=m,
        lambdas_fns_positive=[lambda x, m, i=i: (i + 1) / (K + 1) for i in range(K)],
        convex_comb=True,
        trunc_scale=1,
    )
    lambdas_matrix = np.tile(
        np.array([(i + 1) / (K + 1) for i in range(K)])[:, None], n
    )
    x_matrix = np.tile(x, (K, 1))
    div_mart2 = np.mean(
        0.5 * np.cumprod(1 + lambdas_matrix * (x_matrix - m), axis=1)
        + 0.5 * np.cumprod(1 - lambdas_matrix * (x_matrix - m), axis=1),
        axis=0,
    )

    assert all(div_mart1 == div_mart2)


@pytest.mark.parametrize("m,alpha", expand_grid([0.45, 0.5, 0.55], [0.05, 0.1]))
def test_cs_from_martingale(m, alpha):
    # At each time, find the boundary of the CS.
    # Check to make sure it's exactly where the martingale exceeds 1/alpha
    n = 1000
    breaks = 1000
    mart_fn = lambda x, m: betting_mart(x, m)
    x = np.random.beta(1, 1, n)
    l, u = cs_from_martingale(x, mart_fn, alpha=alpha, breaks=breaks)

    mart = mart_fn(x, m)
    assert all(mart[np.logical_or(m < l, u < m)] > 1 / alpha)
    # Here, we are checking whether m is in [l, u] up to some 1/breaks deviation. This
    # is due to the fact that grid-based CSs are conservative up to such a deviation.
    assert all(
        mart[np.logical_and(l + 1 / breaks <= m, m <= u - 1 / breaks)] <= 1 / alpha
    )


def test_mu_t():
    # Check that mu_t is always the mean of the remaining population
    N = 10000
    x = np.random.binomial(1, 0.5, N)
    mu_t_list = mu_t(x, m=np.mean(x), N=N)
    mean_remaining_pop_list = np.array([np.mean(x[i:N]) for i in range(N)])
    assert all(abs(mu_t_list - mean_remaining_pop_list) < 10e-12)


def test_logical_cs():
    # If just receiving 1s, lower cs should be 1/N, 2/N, 3/N, ...
    # and upper cs should be 1, 1, 1, ...
    N = 100
    x = np.ones(N)
    l, u = logical_cs(x, N=N)
    assert all(l == [i / N for i in np.arange(1, N + 1)])
    assert all(u == np.ones(N))

    # The opposite phenomenon should be observed if all observations are 0
    x = np.zeros(N)
    l, u = logical_cs(x, N=N)
    assert all(l == np.zeros(N))
    assert all(u == [1 - i / N for i in np.arange(1, N + 1)])


def test_get_ci_seq():
    # Just test the CI at a given time
    ci_fn = lambda x: betting_ci(x, alpha=0.05)
    times = [10, 50, 100]
    x = np.random.beta(1, 1, 100)
    l_seq, u_seq = get_ci_seq(x, ci_fn=ci_fn, times=times)

    for i in range(len(times)):
        l, u = ci_fn(x[0 : times[i]])
        assert l_seq[i] == l
        assert u_seq[i] == u


def test_onesided_cs():
    # Ensure that the two-sided CS recovers the lower CS with appropriate alpha.

    n = 10000
    x = np.random.beta(1, 1, n)
    alpha = 0.5
    theta = 1 / 2

    lower_twosided, upper_twosided = betting_cs(
        x,
        alpha=alpha,
        running_intersection=False,
        theta=theta,
        convex_comb=False,
    )

    lower_onesided = betting_lower_cs(
        x,
        alpha=theta * alpha,
        running_intersection=False,
    )

    upper_onesided = 1 - betting_lower_cs(
        1 - x, alpha=(1 - theta) * alpha, running_intersection=False
    )

    assert all(np.isclose(lower_twosided, lower_onesided).astype(bool))
    assert all(np.isclose(upper_twosided, upper_onesided).astype(bool))
