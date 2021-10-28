import numpy as np
from confseq.boundaries import (
    normal_mixture_bound,
    gamma_exponential_mixture_bound,
    beta_binomial_log_mixture,
)
from confseq.misc import get_running_intersection
from confseq.betting import cs_from_martingale
import math


def conjmix_hoeffding_cs(x, t_opt, alpha=0.05, running_intersection=False):
    """
    Conjugate mixture Hoeffding confidence sequence

    Parameters
    ----------
    x, array-like of reals
        The observed data

    t_opt, positive real
        Time at which to optimize the confidence sequence

    alpha, (0, 1)-valued real
        Significance level

    running_intersection, boolean
        Should the running intersection be taken?

    Returns
    -------
    l, array-like of reals
        Lower confidence sequence

    u, array-like of reals
        Upper confidence sequence
    """
    t = np.arange(1, len(x) + 1)
    mu_hat_t = np.cumsum(x) / t

    bdry = (
        normal_mixture_bound(
            t / 4, alpha=alpha, v_opt=t_opt / 4, alpha_opt=alpha, is_one_sided=False
        )
        / t
    )
    l, u = mu_hat_t - bdry, mu_hat_t + bdry

    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    return get_running_intersection(l, u) if running_intersection else (l, u)


def conjmix_empbern_cs(x, v_opt, alpha=0.05, running_intersection=False):
    """
    Conjugate mixture empirical Bernstein confidence sequence

    Parameters
    ----------
    x, array-like of reals
        The observed data

    v_opt, positive real
        Intrinsic time at which to optimize the confidence sequence.
        For example, if the variance is given by sigma, and one
        wishes to optimize for time t, then v_opt = t*sigma^2.

    alpha, (0, 1)-valued real
        Significance level

    running_intersection, boolean
        Should the running intersection be taken?

    Returns
    -------
    l, array-like of reals
        Lower confidence sequence

    u, array-like of reals
        Upper confidence sequence
    """
    x = np.array(x)
    t = np.arange(1, len(x) + 1)
    S_t = np.cumsum(x)
    mu_hat_t = S_t / t
    mu_hat_tminus1 = np.append(1 / 2, mu_hat_t[0 : (len(mu_hat_t) - 1)])
    V_t = np.cumsum(np.power(x - mu_hat_tminus1, 2))
    bdry = (
        gamma_exponential_mixture_bound(
            V_t, alpha=alpha / 2, v_opt=v_opt, c=1, alpha_opt=alpha / 2
        )
        / t
    )
    l, u = mu_hat_t - bdry, mu_hat_t + bdry
    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    return get_running_intersection(l, u) if running_intersection else (l, u)


def bernoulli_supermartingale(x, m, alpha_opt, t_opt):
    x = np.array(x)
    n = len(x)
    t = np.arange(1, n + 1)
    S_t = np.cumsum(x - m)

    var = m * (1 - m)
    V_t = var * t

    # If the null is m = 0, then as soon as we observe
    # one x > 0, we know that the process cannot be a supermg.
    if m == 0:
        supermg = np.ones(n)
        supermg[S_t > 0] = math.inf
    # If the null is m = 1, then as soon as we observe
    # one x < 1, we know that the process cannot be a supermg.
    elif m == 1:
        supermg = np.ones(n)
        supermg[n - S_t < 1] = math.inf
    else:
        log_supermg = beta_binomial_log_mixture(
            s=S_t,
            v=V_t,
            v_opt=var * t_opt,
            g=m,
            h=1 - m,
            alpha_opt=alpha_opt,
            is_one_sided=False,
        )
        with np.errstate(over="ignore"):
            supermg = np.exp(log_supermg)

    return supermg


def conjmix_bernoulli_cs(
    x, t_opt, alpha=0.05, breaks=1000, running_intersection=False, parallel=False
):
    x = np.array(x)

    superMG_fn = (
        lambda x, m: bernoulli_supermartingale(x, m, alpha_opt=alpha / 2, t_opt=t_opt)
        / 2
        + bernoulli_supermartingale(1 - x, 1 - m, alpha_opt=alpha / 2, t_opt=t_opt) / 2
    )

    l, u = cs_from_martingale(
        x, mart_fn=superMG_fn, breaks=breaks, alpha=alpha, parallel=parallel
    )

    return get_running_intersection(l, u) if running_intersection else (l, u)
