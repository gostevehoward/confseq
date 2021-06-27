import numpy as np
import math


def lambda_predmix_eb(
    x,
    truncation=math.inf,
    alpha=0.05,
    fixed_n=None,
    prior_mean=1 / 2,
    prior_variance=1 / 4,
    fake_obs=1,
    scale=1,
):
    t = np.arange(1, len(x) + 1)
    mu_hat_t = (fake_obs * prior_mean + np.cumsum(x)) / (t + fake_obs)
    mu_hat_tminus1 = np.append(prior_mean, mu_hat_t[0 : (len(x) - 1)])
    sigma2_t = (fake_obs * prior_variance + np.cumsum(np.power(x - mu_hat_t, 2))) / (
        t + fake_obs
    )
    sigma2_tminus1 = np.append(prior_variance, sigma2_t[0 : (len(x) - 1)])
    if fixed_n is None:
        lambdas = np.sqrt(2 * np.log(1 / alpha) / (t * np.log(1 + t) * sigma2_tminus1))
    else:
        lambdas = np.sqrt(2 * np.log(1 / alpha) / (fixed_n * sigma2_tminus1))

    lambdas[np.isnan(lambdas)] = 0

    lambdas = np.minimum(truncation, lambdas)

    return lambdas * scale


def predmix_empbern_cs(
    x,
    alpha=0.05,
    truncation=1 / 2,
    running_intersection=False,
    fixed_n=None,
):
    """
    Predictable mixture empirical Bernstein confidence sequence

    Parameters
    ----------
    x, array-like of reals
        Observations of numbers between 0 and 1

    lambda_params, array-like of reals
        lambda values for online mixture

    alpha, positive real
        Confidence level in (0, 1)

    lower_bd, real
        A-priori known lower bound for the observations

    upper_bd, real
        A-priori known upper bound for the observations

    Returns
    -------
    l, array-like of reals
        Lower confidence sequence for the mean

    u, array-like of reals
        Upper confidence sequence for the mean
    """
    x = np.array(x)

    t = np.arange(1, len(x) + 1)
    mu_hat_t = np.cumsum(x) / t
    mu_hat_tminus1 = np.append(0, mu_hat_t[0 : (len(x) - 1)])

    lambdas = lambda_predmix_eb(
        x, truncation=truncation, alpha=alpha / 2, fixed_n=fixed_n
    )

    psi = np.power(x - mu_hat_tminus1, 2) * (-np.log(1 - lambdas) - lambdas)
    margin = (np.log(2 / alpha) + np.cumsum(psi)) / np.cumsum(lambdas)

    weighted_mu_hat_t = np.cumsum(x * lambdas) / np.cumsum(lambdas)

    l, u = weighted_mu_hat_t - margin, weighted_mu_hat_t + margin
    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    return l, u


def predmix_hoeffding_cs(
    x,
    lambda_params=None,
    alpha=0.05,
    running_intersection=False,
):
    """
    Predictable mixture Hoeffding confidence sequence

    Parameters
    ----------
    x, array-like of reals
        Observations of numbers between 0 and 1
    lambda_params, array-like of reals
        lambda values for online mixture
    alpha, positive real
        Confidence level in (0, 1)
    lower_bd, real
        A-priori known lower bound for the observations
    upper_bd, real
        A-priori known upper bound for the observations

    Returns
    -------
    l, array-like of reals
        Lower confidence sequence for the mean
    u, array-like of reals
        Upper confidence sequence for the mean
    """
    x = np.array(x)

    t = np.arange(1, len(x) + 1)

    if lambda_params is None:
        lambda_params = np.sqrt(8 * np.log(2 / alpha) / (t * np.log(t + 1)))
        lambda_params = np.minimum(1, lambda_params)
    else:
        lambda_params = np.array(lambda_params)

    mu_hat_t = np.cumsum(lambda_params * x) / np.cumsum(lambda_params)

    psi = np.cumsum(np.power(lambda_params, 2)) / 8
    margin = (psi + np.log(2 / alpha)) / (np.cumsum(lambda_params))

    weighted_mu_hat_t = np.cumsum(x * lambda_params) / np.cumsum(lambda_params)
    weighted_mu_hat_t[np.isnan(weighted_mu_hat_t)] = 1 / 2

    l, u = weighted_mu_hat_t - margin, weighted_mu_hat_t + margin
    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    return l, u
