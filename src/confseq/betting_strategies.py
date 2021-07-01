import math
import numpy as np
from scipy.optimize import root


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
    """
    Predictable mixture lambda values or "bets"

    Parameters
    ----------
    x, array-like of 0-1 bounded reals
        Observed data

    truncation, positive real or infinity
        Level at which to truncate lambda

    alpha, (0, 1)-valued real
        Significance level in (0, 1)

    fixed_n, positive integer or None
        Sample size for which lambda should be optimized.
        If left as None, lambda will scale like 1/sqrt{t log t}

    prior_mean, [0, 1]-valued real
        Prior mean to use for regularized sample mean

    prior_variance, (0, 1/4]-valued real
        Prior variance to use for regularized sample variance

    fake_obs, positive integer
        Number of 'fake observations' to add.
        Larger values correspond to more regularization near
        `prior_mean` and `prior_variance`

    scale, positive real
        Scale by which to multiply final lambda output.
        For most applications, this should be left as 1

    Returns
    -------
    lambdas, array-like of positive reals
        A (numpy) array of lambda values or "bets"
    """
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


def lambda_aKelly(
    x,
    m,
    prior_mean=1 / 2,
    prior_variance=1 / 4,
    fake_obs=1,
    N=None,
    trunc_scale=1,
):
    assert trunc_scale > 0 and trunc_scale <= 1
    t = np.arange(1, len(x) + 1)
    S_t = np.cumsum(x)
    x_tminus1 = np.append(0, x[0 : (len(x) - 1)])
    S_tminus1 = np.cumsum(x_tminus1)
    mu_hat_t = (fake_obs * prior_mean + np.cumsum(x)) / (t + fake_obs)
    mu_hat_tminus1 = np.append(prior_mean, mu_hat_t[0 : (len(x) - 1)])
    V_t = (fake_obs * prior_variance + np.cumsum(np.power(x - mu_hat_t, 2))) / (
        t + fake_obs
    )
    V_tminus1 = np.append(prior_variance, V_t[0 : (len(x) - 1)])

    if N is not None:
        conditional_mu_hat_tminus1 = (N * mu_hat_tminus1 - S_tminus1) / (N - t + 1)
        mu_t = (N * m - S_tminus1) / (N - (t - 1))
    else:
        conditional_mu_hat_tminus1 = mu_hat_tminus1
        mu_t = m

    lambdas = (conditional_mu_hat_tminus1 - mu_t) / (
        V_tminus1 + np.power(conditional_mu_hat_tminus1 - mu_t, 2)
    )

    lambdas = np.maximum(-trunc_scale / (1 - mu_t), lambdas)
    lambdas = np.minimum(trunc_scale / mu_t, lambdas)

    return lambdas


def lambda_Kelly(x, m):
    lambdas_init = lambda_aKelly(x, m, trunc_scale=1)
    lambdas = np.repeat(0.0, len(x))
    for i in np.arange(1, len(x)):

        def EL_fn(l):
            cand_val = np.sum((x[0:i] - m) / (1 + l * (x[0:i] - m)))
            return cand_val

        if np.max(x[0:i]) <= m:
            lambdas[i] = -1 / (1 - m)
        elif np.min(x[0:i]) >= m:
            lambdas[i] = 1 / m
        else:
            lambdas[i] = root(EL_fn, x0=lambdas_init[i])["x"]
    lambdas[np.isnan(lambdas)] = 0
    lambdas = lambdas

    return lambdas


def lambda_LBOW(x, m):
    t = np.arange(1, len(x) + 1)
    S_t = np.cumsum(x)
    mu_hat_t = (1 / 2 + S_t) / (t + 1)
    mu_hat_tminus1 = np.append(1 / 2, mu_hat_t[0 : (len(mu_hat_t) - 1)])
    sigma2_tminus1 = np.append(1 / 4, np.cumsum(np.power(x - m, 2)))[0 : len(x)] / (
        t + 1
    )

    g = mu_hat_tminus1 - m
    M = np.where(g > 0, m, 1 - m)
    lambdas = g / (M * np.abs(g) + sigma2_tminus1 + np.power(g, 2))
    return lambdas
