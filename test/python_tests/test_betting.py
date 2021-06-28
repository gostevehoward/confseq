import pytest
import numpy as np
from confseq.betting import *
from confseq.utils import superMG_crossing_fraction
from scipy.stats import binomtest
from operator import lt, gt


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


def test_betting_mart_power():
    theta = 1 / 2
    x = np.random.binomial(1, 0.5, 10000)
    mart1 = betting_mart(x, 0.4, theta=theta)
    mart2 = betting_mart(x, 0.6, theta=theta)

    # Should have two-sided power
    # This will fail with some small probability
    assert any(mart1 > 20)
    assert any(mart2 > 20)

    theta = 1
    mart1 = betting_mart(x, 0.4, theta=theta)
    mart2 = betting_mart(x, 0.6, theta=theta)
    
    # Should only have power against 0.4, but not 0.6
    assert(any(mart1 > 20))
    assert(not any(mart2 > 20))

    theta = 0
    mart1 = betting_mart(x, 0.4, theta=theta)
    mart2 = betting_mart(x, 0.6, theta=theta)
    
    # Should only have power against 0.6, but not 0.4
    assert(not any(mart1 > 20))
    assert(any(mart2 > 20))