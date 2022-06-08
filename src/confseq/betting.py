import numpy as np
import math
import matplotlib.pyplot as plt
from time import time
from scipy.optimize import minimize, newton, root
import multiprocess
from copy import copy, deepcopy
from logging import info
from confseq.misc import get_running_intersection, get_ci_seq

from confseq.predmix import lambda_predmix_eb


def betting_mart(
    x,
    m,
    alpha=0.05,
    lambdas_fn_positive=None,
    lambdas_fn_negative=None,
    N=None,
    convex_comb=False,
    theta=1 / 2,
    trunc_scale=1 / 2,
    m_trunc=True,
):
    """
    Betting martingale for a given sequence
    of data and a null mean value.

    Parameters
    ----------
    x, array-like
        The vector of observations.

    m, real
        Null value for the mean of x

    alpha, real
        Significance level between 0 and 1.

    lambdas_fn_postive, bivariate function or None
        Function of `x` and `m` which generates an array-like
        of bets with the same length as `x`.

    lambdas_fn_negative=None, bivariate function or None
        Same as above but for the negative capital process.
        This will be set to lambdas_fn_positive if
        left as None.

    N, positive integer or None
        Population size if sampling WoR

    convex_comb, boolean
        Should a convex combination of martingales be taken?
        If True, the process will be theta*pcp + (1-theta)*ncp,
        and if False, it will be max(theta*pcp, (1-theta)*ncp) where
        pcp and ncp are the positive and negative capital processes,
        respectively.

    theta, [0, 1]-valued real
        Positive/negative capital process weight as
        defined in `convex_comb`

    trunc_scale, (0, 1]-valued real
        The factor by which to multiply the truncation defined
        in `m_trunc`. Leaving this as 1 will perform no
        additional truncation.

    m_trunc, boolean
        Should truncation be used based on m? If True, then truncation
        will be given by trunc_scale * 1/m or trunc_scale * 1/(1-m)
        depending on the capital process. If False, then truncation
        will be given by trunc_scale.


    Returns
    -------
    mart, array-like
        The martingale that results from the observed x
    """

    assert 0 <= theta <= 1

    if theta < 1 and not all(x >= 0):
        raise ValueError("Cannot use theta < 1 with data that is not lower-bounded.")
    elif theta > 0 and not all(x <= 1):
        raise ValueError("Cannot use theta > 1 with data that is not upper-bounded.")
    else:
        assert all(x >= 0) and all(x <= 1)

    if lambdas_fn_positive is None:
        lambdas_fn_positive = lambda x, m: lambda_predmix_eb(x, alpha=alpha * theta)

    if lambdas_fn_negative is None:
        lambdas_fn_negative = lambda x, m: lambda_predmix_eb(
            x, alpha=alpha * (1 - theta)
        )

    if N is not None:
        t = np.arange(1, len(x) + 1)
        S_t = np.cumsum(x)
        S_tminus1 = np.append(0, S_t[0 : (len(x) - 1)])
        mu_t = (N * m - S_tminus1) / (N - (t - 1))
    else:
        mu_t = np.repeat(m, len(x))

    assert 0 < trunc_scale <= 1

    if theta > 0:
        lambdas_positive = lambdas_fn_positive(x, m)
        # if we want to truncate with m
        if m_trunc:
            with np.errstate(divide="ignore"):
                lambdas_positive = np.minimum(lambdas_positive, trunc_scale / mu_t)
                lambdas_positive = np.maximum(lambdas_positive, -trunc_scale / (1 - mu_t))
        else:
            lambdas_positive = np.minimum(lambdas_positive, trunc_scale)
            lambdas_positive = np.maximum(lambdas_positive, -trunc_scale)

        with np.errstate(invalid="ignore"):
            multiplicand_positive = 1 + lambdas_positive * (x - mu_t)

        # Use convention that inf * 0 = 0. We still have
        # a martingale under the null
        multiplicand_positive[
            np.logical_and(lambdas_positive == math.inf, x - mu_t == 0)
        ] = 1

        with np.errstate(invalid="ignore"):
            capital_process_positive = np.cumprod(multiplicand_positive)

        # If we get nans from 0 * inf, this should be 0
        capital_process_positive[np.isnan(capital_process_positive)] = 0
    else:
        capital_process_positive = np.repeat(0, len(x))

    if theta < 1:
        lambdas_negative = lambdas_fn_negative(x, m)
        if m_trunc:
            with np.errstate(divide="ignore"):
                lambdas_negative = np.minimum(lambdas_negative, trunc_scale / (1 - mu_t))
                lambdas_negative = np.maximum(lambdas_negative, -trunc_scale / mu_t)
        else:
            lambdas_negative = np.minimum(lambdas_negative, trunc_scale)
            lambdas_negative = np.maximum(lambdas_negative, -trunc_scale)

        with np.errstate(invalid="ignore"):
            multiplicand_negative = 1 - lambdas_negative * (x - mu_t)

        multiplicand_negative[
            np.logical_and(lambdas_negative == math.inf, x - mu_t == 0)
        ] = 1

        with np.errstate(invalid="ignore"):
            capital_process_negative = np.cumprod(multiplicand_negative)

        capital_process_negative[np.isnan(capital_process_negative)] = 0
    else:
        capital_process_negative = np.repeat(0, len(x))

    if theta == 1:
        capital_process = capital_process_positive
    elif theta == 0:
        capital_process = capital_process_negative
    else:
        if convex_comb:
            capital_process = (
                theta * capital_process_positive
                + (1 - theta) * capital_process_negative
            )
        else:
            capital_process = np.maximum(
                theta * capital_process_positive, (1 - theta) * capital_process_negative
            )

    # If mu_t < 0 or mu_t > 1, we cannot be under the null
    capital_process[np.logical_or(mu_t < 0, mu_t > 1)] = math.inf

    assert not any(np.isnan(capital_process))

    assert all(capital_process >= 0)

    return capital_process


def betting_cs(
    x,
    lambdas_fns_positive=None,
    lambdas_fns_negative=None,
    alpha=0.05,
    N=None,
    breaks=1000,
    running_intersection=False,
    parallel=False,
    convex_comb=False,
    theta=1 / 2,
    trunc_scale=1 / 2,
    m_trunc=True,
):
    """
    Betting-based confidence sequence

    Parameters
    ----------
    x, array-like
        The vector of observations between 0 and 1.

    alpha, real
        Significance level between 0 and 1.

    lambdas_fns_postive, list of bivariate functions or None
        Function of `x` and `m` which generates an array-like
        of bets with the same length as `x`.

    lambdas_fns_negative=None, list of bivariate functions or None
        Same as above but for the negative capital process.
        This will be set to lambdas_fn_positive if
        left as None.

    N, positive integer or None
        Population size if sampling WoR

    breaks, positive integer
        Number of breaks in the grid for constructing the confidence sequence

    running_intersection, boolean
        Should the running intersection be taken?

    parallel, boolean
        Should computation be parallelized?

    convex_comb, boolean
        Should a convex combination of martingales be taken?
        If True, the process will be theta*pcp + (1-theta)*ncp,
        and if False, it will be max(theta*pcp, (1-theta)*ncp) where
        pcp and ncp are the positive and negative capital processes,
        respectively.

    theta, [0, 1]-valued real
        Positive/negative capital process weight as
        defined in `convex_comb`

    trunc_scale, (0, 1]-valued real
        The factor by which to multiply the truncation defined
        in `m_trunc`. Leaving this as 1 will perform no
        additional truncation.

    m_trunc, boolean
        Should truncation be used based on m? If True, then truncation
        will be given by trunc_scale * 1/m or trunc_scale * 1/(1-m)
        depending on the capital process. If False, then truncation
        will be given by trunc_scale.


    Returns
    -------
    l, array-like
        Lower confidence sequence for the mean

    u, array-like
        Upper confidence sequence for the mean
    """

    if lambdas_fns_positive is None:
        lambdas_fns_positive = lambda x, m: lambda_predmix_eb(x, alpha=alpha * theta)
    
    if lambdas_fns_negative is None:
        lambdas_fns_negative = lambda x, m: lambda_predmix_eb(x, alpha=alpha * (1-theta))

    # If only passed a single function, put it into a list
    if np.shape(lambdas_fns_positive) == ():
        lambdas_fns_positive = [lambdas_fns_positive]
    if np.shape(lambdas_fns_negative) == ():
        lambdas_fns_negative = [lambdas_fns_negative]

    assert len(lambdas_fns_positive) == len(lambdas_fns_negative)

    mart_fn = lambda x, m: diversified_betting_mart(
        x,
        m,
        alpha=alpha,
        lambdas_fns_positive=lambdas_fns_positive,
        lambdas_fns_negative=lambdas_fns_negative,
        N=N,
        theta=theta,
        convex_comb=convex_comb,
        trunc_scale=trunc_scale,
        m_trunc=m_trunc,
    )

    l, u = cs_from_martingale(
        x,
        mart_fn,
        breaks=breaks,
        alpha=alpha,
        N=N,
        parallel=parallel,
        running_intersection=running_intersection,
    )

    return l, u



def betting_lower_cs(
    x,
    lambdas_fns=None,
    alpha=0.05,
    N=None,
    breaks=1000,
    running_intersection=False,
    parallel=False,
    trunc_scale=1 / 2,
    m_trunc=True,
):
    """
    Betting-based lower confidence sequence

    Parameters
    ----------
    x, array-like
        The vector of nonnegative observations.

    alpha, real
        Significance level between 0 and 1.

    lambdas_fns, list of bivariate functions or None
        Function of `x` and `m` which generates an array-like
        of bets with the same length as `x`.

    N, positive integer or None
        Population size if sampling WoR

    breaks, positive integer
        Number of breaks in the grid for constructing the confidence sequence

    running_intersection, boolean
        Should the running intersection be taken?

    parallel, boolean
        Should computation be parallelized?

    trunc_scale, (0, 1]-valued real
        The factor by which to multiply the truncation defined
        in `m_trunc`. Leaving this as 1 will perform no
        additional truncation.

    m_trunc, boolean
        Should truncation be used based on m? If True, then truncation
        will be given by trunc_scale * 1/m or trunc_scale * 1/(1-m)
        depending on the capital process. If False, then truncation
        will be given by trunc_scale.


    Returns
    -------
    l, array-like
        Lower confidence sequence for the [0, 1]-bounded mean
    """
    l, _ = betting_cs(
        x,
        lambdas_fns_positive=lambdas_fns,
        alpha=alpha,
        N=N,
        breaks=breaks,
        running_intersection=running_intersection,
        parallel=parallel,
        convex_comb=False,
        theta=1,
        trunc_scale=trunc_scale,
        m_trunc=m_trunc,
    )

    return l


def diversified_betting_mart(
    x,
    m,
    lambdas_fns_positive,
    lambdas_fns_negative=None,
    lambdas_weights=None,
    alpha=None,
    N=None,
    convex_comb=False,
    theta=1 / 2,
    trunc_scale=1 / 2,
    m_trunc=True,
):
    mart_positive = np.zeros(len(x))
    mart_negative = np.zeros(len(x))

    # Number of betting strategies to use
    K = len(lambdas_fns_positive)

    if lambdas_weights is None:
        lambdas_weights = np.repeat(1 / K, K)

    if lambdas_fns_negative is None:
        lambdas_fns_negative = lambdas_fns_positive

    assert len(lambdas_fns_positive) == len(lambdas_fns_negative)

    for k in range(K):
        lambdas_fn_positive = lambdas_fns_positive[k]
        lambdas_fn_negative = lambdas_fns_negative[k]

        if theta > 0:
            summand_positive = (
                (
                    lambdas_weights[k]
                    * betting_mart(
                        x,
                        m,
                        alpha=alpha,
                        lambdas_fn_positive=lambdas_fn_positive,
                        lambdas_fn_negative=lambdas_fn_negative,
                        N=N,
                        theta=1,
                        trunc_scale=trunc_scale,
                        m_trunc=m_trunc,
                    )
                )
                if lambdas_weights[k] != 0
                else 0
            )
            mart_positive = mart_positive + summand_positive

        if theta < 1:
            summand_negative = (
                (
                    lambdas_weights[k]
                    * betting_mart(
                        x,
                        m,
                        alpha=alpha,
                        lambdas_fn_positive=lambdas_fn_positive,
                        lambdas_fn_negative=lambdas_fn_negative,
                        N=N,
                        theta=0,
                        trunc_scale=trunc_scale,
                        m_trunc=m_trunc,
                    )
                )
                if lambdas_weights[k] != 0
                else 0
            )
            mart_negative = mart_negative + summand_negative

    if theta == 1:
        mart = mart_positive
    elif theta == 0:
        mart = mart_negative
    else:
        mart = (
            theta * mart_positive + (1 - theta) * mart_negative
            if convex_comb
            else np.maximum(theta * mart_positive, (1 - theta) * mart_negative)
        )

    assert not any(np.isnan(mart))

    return mart


def cs_from_martingale(
    x,
    mart_fn,
    breaks=1000,
    alpha=0.05,
    N=None,
    running_intersection=False,
    parallel=False,
    log_scale=False,
):
    """
    Given a test supermartingale, produce a confidence sequence for
    any parameter using the grid method, assuming the parameter is
    in [0, 1]

    Parameters
    ----------
    x, array-like
        The vector of observations between 0 and 1.

    mart_fn, bivariate function
        A function which takes data `x` and a candidate mean `m`
        and outputs a (super)martingale.

    breaks, positive integer
        Number of breaks in the grid for constructing the confidence sequence

    alpha, real
        Significance level between 0 and 1.

    N, positive integer or None
        Population size if sampling WoR

    running_intersection, boolean
        Should the running intersection be taken?

    parallel, boolean
        Should computation be parallelized?

    Returns
    -------
    l, array-like
        Lower confidence sequence for the parameter
    u, array-like
        Upper confidence sequence for the parameter
    """
    possible_m = np.arange(0, 1 + 1 / breaks, step=1 / breaks)
    confseq_mtx = np.zeros((len(possible_m), len(x)))

    if log_scale:
        threshold = np.log(1 / alpha)
    else:
        threshold = 1 / alpha

    if parallel:
        n_cores = multiprocess.cpu_count()
        info("Using " + str(n_cores) + " cores")
        with multiprocess.Pool(n_cores) as p:
            result = p.map(lambda m: mart_fn(x, m), possible_m)
            confseq_mtx = np.vstack(result) <= threshold
    else:
        for i in np.arange(0, len(possible_m)):
            m = possible_m[i]
            confseq_mtx[i, :] = mart_fn(x, m) <= threshold

    l = np.zeros(len(x))
    u = np.ones(len(x))

    for j in np.arange(0, len(x)):
        where_in_cs = np.where(confseq_mtx[:, j])
        if len(where_in_cs[0]) == 0:
            l[j] = 0
            u[j] = 1
        else:
            l[j] = possible_m[where_in_cs[0][0]]
            u[j] = possible_m[where_in_cs[0][-1]]
    l = np.maximum(0, l - 1 / breaks)
    u = np.minimum(1, u + 1 / breaks)

    if N is not None:
        logical_l, logical_u = logical_cs(x, N)

        l = np.maximum(l, logical_l)
        u = np.minimum(u, logical_u)

    return get_running_intersection(l, u) if running_intersection else (l, u)


def hedged_cs(
    x,
    alpha=0.05,
    N=None,
    breaks=1000,
    running_intersection=True,
    parallel=False,
    theta=1 / 2,
    trunc_scale=1 / 2,
    prior_mean=1 / 2,
    prior_variance=1 / 4,
    fake_obs=1,
):
    """
    Hedged capital confidence sequence.
        This is simply an instantiation of `betting_cs`
        with particular defaults for the hedged capital
        martingale.

    Parameters
    ----------
    x, array-like
        The vector of observations between 0 and 1.

    alpha, (0, 1)-valued real
        Significance level

    N, positive integer or None
        Population size if sampling WoR

    breaks, positive integer
        Number of breaks in the grid for constructing the confidence sequence

    running_intersection, boolean
        Should the running intersection be taken?

    parallel, boolean
        Should computation be parallelized?

    theta, [0, 1]-valued real
        Positive/negative capital process weight.

    trunc_scale, (0, 1]-valued real
        The factor by which to multiply the truncation.
        Leaving this as 1 will perform no additional truncation.

    Returns
    -------
    l, array-like
        Lower confidence sequence for the mean

    u, array-like
        Upper confidence sequence for the mean
    """
    return betting_cs(
        x,
        lambdas_fns_positive=[
            lambda x, m: lambda_predmix_eb(
                x,
                alpha=alpha * theta,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                fake_obs=fake_obs,
            )
        ],
        lambdas_fns_negative=[
            lambda x, m: lambda_predmix_eb(
                x,
                alpha=alpha * (1 - theta),
                prior_mean=prior_mean,
                prior_variance=prior_variance,
                fake_obs=fake_obs,
            )
        ],
        alpha=alpha,
        N=N,
        breaks=breaks,
        running_intersection=running_intersection,
        parallel=parallel,
        convex_comb=False,
        theta=theta,
        trunc_scale=trunc_scale,
        m_trunc=True,
    )


def dKelly_cs(
    x,
    D=10,
    alpha=0.05,
    N=None,
    breaks=1000,
    running_intersection=True,
    parallel=False,
    theta=1 / 2,
):
    """
    Hedged capital confidence sequence.
        This is simply an instantiation of `betting_cs`
        with particular defaults for the hedged capital
        martingale.

    Parameters
    ----------
    x, array-like
        The vector of observations between 0 and 1.

    D, positive integer
        The number of evenly-spaced constant bets to place
        and average. Small values of D will be more computationally
        tractable, while larger values of D will "diversify" the bets
        more.

    alpha, (0, 1)-valued real
        Significance level

    N, positive integer or None
        Population size if sampling WoR

    breaks, positive integer
        Number of breaks in the grid for constructing the confidence sequence

    running_intersection, boolean
        Should the running intersection be taken?

    parallel, boolean
        Should computation be parallelized?

    theta, [0, 1]-valued real
        Positive/negative capital process weight.

    Returns
    -------
    l, array-like
        Lower confidence sequence for the mean

    u, array-like
        Upper confidence sequence for the mean
    """
    if N is not None:
        lambdas_fns_positive = [
            lambda x, m, i=i: (i + 1) / (mu_t(x, m, N) * (D + 1)) for i in range(D)
        ]
        lambdas_fns_negative = [
            lambda x, m, i=i: (i + 1) / ((1 - mu_t(x, m, N)) * (D + 1))
            for i in range(D)
        ]
    else:
        lambdas_fns_positive = [
            lambda x, m, i=i: (i + 1) / (m * (D + 1)) for i in range(D)
        ]
        lambdas_fns_negative = [
            lambda x, m, i=i: (i + 1) / ((1 - m) * (D + 1)) for i in range(D)
        ]

    return betting_cs(
        x,
        lambdas_fns_positive=lambdas_fns_positive,
        lambdas_fns_negative=lambdas_fns_negative,
        alpha=alpha,
        N=N,
        breaks=breaks,
        running_intersection=running_intersection,
        parallel=parallel,
        convex_comb=True,
        theta=theta,
        trunc_scale=1,
        m_trunc=True,
    )


def logical_cs(x, N):
    """
    The 1-dimensional logical confidence sequence for sampling without
    replacement. This is essentially the CS that would be
    known regardless of the underlying martingale being used.
    Specifically, if the cumulative sum at time t, S_t is equal to
    5 and N is 10, then the true mean cannot be any less than 0.5, assuming
    all observations are between 0 and 1.

    Parameters
    ----------
    x, array-like of reals between 0 and 1
        The observed bounded random variables.
    N, integer
        The size of the finite population

    Returns
    -------
    l, array-like
        Lower logical confidence sequence for the parameter
    u, array-like
        Upper logical confidence sequence for the parameter
    """
    t = np.arange(1, len(x) + 1)

    S_t = np.cumsum(x)

    l = S_t / N
    u = 1 - (t - S_t) / N

    return l, u


def mu_t(x, m, N):
    t = np.arange(1, len(x) + 1)
    S_t = np.cumsum(x)
    S_tminus1 = np.append(0, S_t[0 : (len(x) - 1)])
    mu_t = (N * m - S_tminus1) / (N - (t - 1))

    return mu_t


def betting_ci(
    x,
    alpha=0.05,
    lambdas_fns_positive=None,
    lambdas_fns_negative=None,
    N=None,
    breaks=1000,
    running_intersection=True,
    parallel=False,
    convex_comb=False,
    theta=1 / 2,
    trunc_scale=1,
    m_trunc=False,
):
    """
    Fixed-time, betting-based confidence interval

    Parameters
    ----------
    x, array-like
        The vector of observations between 0 and 1.

    alpha, real
        Significance level between 0 and 1.

    lambdas_fn_postive, bivariate function or None
        Function of `x` and `m` which generates an array-like
        of bets with the same length as `x`.

    lambdas_fn_negative=None, bivariate function or None
        Same as above but for the negative capital process.
        This will be set to lambdas_fn_positive if
        left as None.

    N, positive integer or None
        Population size if sampling WoR

    breaks, positive integer
        Number of breaks in the grid for constructing the confidence sequence

    running_intersection, boolean
        Should the running intersection be taken?

    parallel, boolean
        Should computation be parallelized?

    convex_comb, boolean
        Should a convex combination of martingales be taken?
        If True, the process will be theta*pcp + (1-theta)*ncp,
        and if False, it will be max(theta*pcp, (1-theta)*ncp) where
        pcp and ncp are the positive and negative capital processes,
        respectively.

    theta, [0, 1]-valued real
        Positive/negative capital process weight as
        defined in `convex_comb`

    trunc_scale, (0, 1]-valued real
        The factor by which to multiply the truncation defined
        in `m_trunc`. Leaving this as 1 will perform no
        additional truncation.

    m_trunc, boolean
        Should truncation be used based on m? If True, then truncation
        will be given by trunc_scale * 1/m or trunc_scale * 1/(1-m)
        depending on the capital process. If False, then truncation
        will be given by trunc_scale.

    Returns
    -------
    l, real
        Lower confidence bound
    u, real
        Upper confidence bound
    """
    x = np.array(x)
    n = len(x)

    lambdas_fns_positive = [
        lambda x, m: lambda_predmix_eb(x, alpha=alpha * theta, fixed_n=n)
    ]
    lambdas_fns_negative = [
        lambda x, m: lambda_predmix_eb(x, alpha=alpha * (1 - theta), fixed_n=n)
    ]

    l, u = betting_cs(
        x,
        alpha=alpha,
        breaks=breaks,
        lambdas_fns_positive=lambdas_fns_positive,
        lambdas_fns_negative=lambdas_fns_negative,
        N=N,
        running_intersection=running_intersection,
        parallel=parallel,
        convex_comb=convex_comb,
        theta=theta,
        trunc_scale=trunc_scale,
        m_trunc=m_trunc,
    )

    return l[-1], u[-1]


def betting_ci_seq(
    x,
    alpha,
    times,
    lambdas_fns_positive=None,
    lambdas_fns_negative=None,
    N=None,
    breaks=1000,
    running_intersection=True,
    parallel=False,
    convex_comb=False,
    theta=1 / 2,
    trunc_scale=0.9,
    m_trunc=True,
):
    """
    Sequence of fixed-time, betting martingale-based confidence intervals

    Parameters
    ----------
    x, array-like
        The vector of observations between 0 and 1.

    alpha, (0, 1)-bounded real
        Significance level between 0 and 1.

    lambdas_fn_postive, bivariate function or None
        Function of `x` and `m` which generates an array-like
        of bets with the same length as `x`.

    lambdas_fn_negative=None, bivariate function or None
        Same as above but for the negative capital process.
        This will be set to lambdas_fn_positive if
        left as None.

    N, positive integer or None
        Population size if sampling WoR

    breaks, positive integer
        Number of breaks in the grid for constructing the confidence sequence

    running_intersection, boolean
        Should the running intersection be taken?

    parallel, boolean
        Should computation be parallelized?

    convex_comb, boolean
        Should a convex combination of martingales be taken?
        If True, the process will be theta*pcp + (1-theta)*ncp,
        and if False, it will be max(theta*pcp, (1-theta)*ncp) where
        pcp and ncp are the positive and negative capital processes,
        respectively.

    theta, [0, 1]-valued real
        Positive/negative capital process weight as
        defined in `convex_comb`

    trunc_scale, (0, 1]-valued real
        The factor by which to multiply the truncation defined
        in `m_trunc`. Leaving this as 1 will perform no
        additional truncation.

    m_trunc, boolean
        Should truncation be used based on m? If True, then truncation
        will be given by trunc_scale * 1/m or trunc_scale * 1/(1-m)
        depending on the capital process. If False, then truncation
        will be given by trunc_scale.

    Returns
    -------
    l, array-like of reals
        Lower confidence intervals

    u, array-like of reals
        Upper confidence intervals
    """

    def ci_fn(x):
        return betting_ci(
            x,
            alpha=alpha,
            breaks=breaks,
            lambdas_fns_positive=lambdas_fns_positive,
            lambdas_fns_negative=lambdas_fns_negative,
            N=N,
            running_intersection=running_intersection,
            convex_comb=convex_comb,
            theta=theta,
            trunc_scale=trunc_scale,
            m_trunc=m_trunc,
        )

    return get_ci_seq(x, ci_fn, times=times, parallel=parallel)
