import numpy as np
import math
from confseq.predmix import *
import pytest


def test_lambda_predmix_eb():
    x = np.random.binomial(
        1,
        0.5,
        100,
    )

    # Check that truncation works
    assert all(lambda_predmix_eb(x, truncation=1) <= 1)
    assert all(lambda_predmix_eb(x, truncation=0.5) <= 0.5)

    # Check that the first lambda value is correct
    assert lambda_predmix_eb(x, truncation=math.inf, alpha=0.05)[0] == np.sqrt(
        2 * np.log(1 / 0.05) / (1 / 4 * np.log(2))
    )
    assert lambda_predmix_eb(x, truncation=math.inf, alpha=0.1)[0] == np.sqrt(
        2 * np.log(1 / 0.1) / (1 / 4 * np.log(2))
    )

    # Changing prior variance should change the first lambda value
    assert (
        lambda_predmix_eb(x, prior_variance=1 / 4)[0]
        > lambda_predmix_eb(x, prior_variance=0.26)[0]
    )
    assert (
        lambda_predmix_eb(x, prior_variance=1 / 4)[0]
        < lambda_predmix_eb(x, prior_variance=0.24)[0]
    )

    # Also, first lambda value should be data-independent,
    # but the second should not
    assert (
        lambda_predmix_eb(x, prior_variance=1 / 4)[0]
        == lambda_predmix_eb(1 - x, prior_variance=1 / 4)[0]
    )
    assert (
        lambda_predmix_eb(x, prior_variance=1 / 4)[1]
        == lambda_predmix_eb(x, prior_variance=1 / 4)[1]
    )


def test_predmix_confidence_sequences():
    x = np.random.binomial(1, 0.5, 10000)
    l, u = predmix_empbern_twosided_cs(x)
    # Check that u > l always
    assert all(u > l)

    # Check that the CS is not all trivial
    assert not (all(u == 1) and all(l == 0))

    # Check that Hoeffding CSs are nested wrt alpha
    l_alpha1, u_alpha1 = predmix_hoeffding_cs(x, alpha=0.01)
    l_alpha2, u_alpha2 = predmix_hoeffding_cs(x, alpha=0.05)
    l_alpha3, u_alpha3 = predmix_hoeffding_cs(x, alpha=0.1)
    assert all(
        np.logical_and(
            u_alpha1 - l_alpha1 >= u_alpha2 - l_alpha2,
            u_alpha2 - l_alpha2 >= u_alpha3 - l_alpha3,
        )
    )

    # Check that empirical Bernstein CSs are nested wrt alpha
    l_alpha1, u_alpha1 = predmix_empbern_twosided_cs(x, alpha=0.01)
    l_alpha2, u_alpha2 = predmix_empbern_twosided_cs(x, alpha=0.05)
    l_alpha3, u_alpha3 = predmix_empbern_twosided_cs(x, alpha=0.1)
    assert all(
        np.logical_and(
            u_alpha1 - l_alpha1 >= u_alpha2 - l_alpha2,
            u_alpha2 - l_alpha2 >= u_alpha3 - l_alpha3,
        )
    )


@pytest.mark.random
def test_predmix_cs_power():
    # Check that Hoeffding is tighter than empirical Bernstein for Bin(0.5) data
    # Note, this might fail due to random chance, but it's unlikely for large n
    n = 50
    x = np.random.binomial(1, 0.5, n)

    l_h, u_h = predmix_hoeffding_cs(x)
    l_eb, u_eb = predmix_empbern_twosided_cs(x)

    assert (u_h - l_h)[n - 1] < (u_eb - l_eb)[n - 1]

    # Check that empirical Bernstein is tighter than Hoeffding for low-variance data
    # Note, this might fail due to random chance,
    # but it's extremely unlikely for large n and very small variance distributions.
    n = 50
    x = np.random.beta(10, 10, n)

    l_h, u_h = predmix_hoeffding_cs(x)
    l_eb, u_eb = predmix_empbern_twosided_cs(x)

    assert (u_h - l_h)[n - 1] > (u_eb - l_eb)[n - 1]
