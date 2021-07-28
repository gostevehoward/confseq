import math
from typing import Callable, Sequence, Tuple, Union
import numpy as np
from confseq.betting_strategies import lambda_predmix_eb
from confseq.misc import get_running_intersection


def predmix_upper_cs(
    x: Sequence[float],
    v: Sequence[float],
    lambdas_fn: Callable[[Sequence[float]], Sequence[float]],
    psi_fn: Callable[[Sequence[float]], Sequence[float]],
    alpha: float = 0.05,
    running_intersection: bool = False,
    N: Union[int, None] = None,
) -> Sequence[float]:
    """
    Predictable mixture upper confidence sequence

    Parameters
    ----------
    x : Sequence[float]
        Observations in [0, 1]
    v : Sequence[float]
        Variance increment (1 for Hoeffding, (x - \hat \mu_{t-1})^2 for empbern)
    lambdas_fn : Callable[[Sequence[float]], Sequence[float]]
        Function to produce lambda values
    psi_fn : Callable[[Sequence[float]], Sequence[float]]
        psi function
    alpha : float, optional
        Significance level, by default 0.05
    running_intersection : bool, optional
        Should the running intersection be taken?, by default False
    N : Union[int, None], optional
        Population size if sampling WoR, by default None

    Returns
    -------
    Sequence[float]
        Upper confidence sequence
    """
    x = np.array(x)

    t = np.arange(1, len(x) + 1)

    S_t = np.cumsum(x)
    S_tminus1 = np.append(0, S_t[0 : (len(x) - 1)])

    if N is not None:
        Zstar = S_tminus1 / (N - t + 1)
        Wstar = (t - 1) / (N - t + 1)
    else:
        Zstar = 0
        Wstar = 0

    lambdas = lambdas_fn(x)
    psi = psi_fn(lambdas)
    margin = (np.log(1 / alpha) + np.cumsum(v * psi)) / np.cumsum(lambdas * (1 + Wstar))

    weighted_mu_hat_t = np.cumsum(lambdas * (x + Zstar)) / np.cumsum(
        lambdas * (1 + Wstar)
    )

    u = weighted_mu_hat_t + margin
    u = np.minimum(u, 1)

    return np.minimum.accumulate(u) if running_intersection else u


def predmix_empbern_upper_cs(
    x: Sequence[float],
    alpha: float = 0.05,
    truncation: float = 1 / 2,
    running_intersection: bool = False,
    N: Union[int, None] = None,
    fixed_n: Union[int, None] = None,
) -> Sequence[float]:
    """
    Predictable mixture empirical Bernstein upper confidence sequence

    Parameters
    ----------
    x : Sequence[float]
        Observations in [0, 1]
    alpha : float, optional
        Significance level, by default 0.05
    truncation : float, optional
        Level at which to truncate lambda, by default 1/2
    running_intersection : bool, optional
        Should the running intersection be taken?, by default False
    N : Union[int, None], optional
        Population size if sampling WoR, by default None
    fixed_n : Union[int, None], optional
        Fixed time to optimize bound for (if CI desired), by default None

    Returns
    -------
    Sequence[float]
        Upper confidence sequence
    """
    t = np.arange(1, len(x) + 1)
    mu_hat_t = np.cumsum(x) / t
    mu_hat_tminus1 = np.append(0, mu_hat_t[0 : (len(x) - 1)])
    v = np.power(x - mu_hat_tminus1, 2)
    return predmix_upper_cs(
        x,
        v=v,
        lambdas_fn=lambda y: lambda_predmix_eb(
            y, truncation=truncation, alpha=alpha, fixed_n=fixed_n
        ),
        psi_fn=lambda lambdas: -np.log(1 - lambdas) - lambdas,
        alpha=alpha,
        running_intersection=running_intersection,
        N=N,
    )


def predmix_hoeffding_upper_cs(
    x: Sequence[float],
    alpha: float = 0.05,
    truncation: float = 1,
    running_intersection: bool = False,
    N: Union[int, None] = None,
    fixed_n: Union[int, None] = None,
) -> Sequence[float]:
    """
    Predictable mixture Hoeffding upper confidence sequence

    Parameters
    ----------
    x : Sequence[float]
        Observations in [0, 1]
    alpha : float, optional
        Significance level, by default 0.05
    truncation : float, optional
        Level at which to truncate lambda, by default 1
    running_intersection : bool, optional
        Should the running intersection be taken?, by default False
    N : Union[int, None], optional
        Population size if sampling WoR, by default None
    fixed_n : Union[int, None], optional
        Fixed time to optimize bound for (if CI desired), by default None

    Returns
    -------
    Sequence[float]
        Upper confidence sequence
    """
    t = np.arange(1, len(x) + 1)
    if fixed_n is not None:
        lambdas_fn = lambda y: np.sqrt(8 * np.log(1 / alpha) / fixed_n)
    else:
        lambdas_fn = lambda y: np.minimum(
            np.sqrt(8 * np.log(1 / alpha) / (t * np.log(1 + t))), truncation
        )
    return predmix_upper_cs(
        x,
        v=1,
        lambdas_fn=lambdas_fn,
        psi_fn=lambda lambdas: np.power(lambdas, 2) / 8,
        alpha=alpha,
        running_intersection=running_intersection,
        N=N,
    )


def predmix_hoeffding_cs(
    x: Sequence[float],
    alpha: float = 0.05,
    truncation: float = 1,
    running_intersection: bool = False,
    N: Union[int, None] = None,
    fixed_n: Union[int, None] = None,
) -> Tuple[Sequence[float], Sequence[float]]:
    """
    Predictable mixture Hoeffding confidence sequence

    Parameters
    ----------
    x : Sequence[float]
        Observations in [0, 1]
    alpha : float, optional
        Significance level, by default 0.05
    truncation : float, optional
        Level at which to truncate lambda, by default 1
    running_intersection : bool, optional
        Should the running intersection be taken?, by default False
    N : Union[int, None], optional
        Population size if sampling WoR, by default None
    fixed_n : Union[int, None], optional
        Fixed time to optimize bound for (if CI desired), by default None

    Returns
    -------
    Tuple[Sequence[float], Sequence[float]]
        Confidence sequence
    """
    upper_cs = predmix_hoeffding_upper_cs(
        x,
        alpha=alpha / 2,
        truncation=truncation,
        running_intersection=running_intersection,
        N=N,
        fixed_n=fixed_n,
    )
    lower_cs = 1 - predmix_hoeffding_upper_cs(
        1 - x,
        alpha=alpha / 2,
        truncation=truncation,
        running_intersection=running_intersection,
        N=N,
        fixed_n=fixed_n,
    )

    return lower_cs, upper_cs


def predmix_empbern_cs(
    x: Sequence[float],
    alpha: float = 0.05,
    truncation: float = 1 / 2,
    running_intersection: bool = False,
    N: Union[int, None] = None,
    fixed_n: Union[int, None] = None,
) -> Tuple[Sequence[float], Sequence[float]]:
    """
    Predictable mixture empirical Bernstein confidence sequence

    Parameters
    ----------
    x : Sequence[float]
        Observations in [0, 1]
    alpha : float, optional
        Significance level, by default 0.05
    truncation : float, optional
        Level at which to truncate lambda, by default 1
    running_intersection : bool, optional
        Should the running intersection be taken?, by default False
    N : Union[int, None], optional
        Population size if sampling WoR, by default None
    fixed_n : Union[int, None], optional
        Fixed time to optimize bound for (if CI desired), by default None

    Returns
    -------
    Tuple[Sequence[float], Sequence[float]]
        Confidence sequence
    """
    u = predmix_empbern_upper_cs(
        x=x,
        alpha=alpha / 2,
        truncation=truncation,
        running_intersection=running_intersection,
        N=N,
        fixed_n=fixed_n,
    )
    l = 1 - predmix_empbern_upper_cs(
        x=1 - x,
        alpha=alpha / 2,
        truncation=truncation,
        running_intersection=running_intersection,
        N=N,
        fixed_n=fixed_n,
    )

    return l, u


# def predmix_empbern_cs(
#     x,
#     alpha=0.05,
#     truncation=1 / 2,
#     running_intersection=False,
#     fixed_n=None,
# ):
#     """
#     Predictable mixture empirical Bernstein confidence sequence

#     Parameters
#     ----------
#     x, array-like of 0-1 bounded reals
#         Observations of numbers between 0 and 1

#     alpha, (0, 1)-valued positive real
#         Significance level

#     truncation, (0, 1)-valued real
#         Level at which to truncate lambda.

#     running_intersection, boolean
#         Should the running intersection be taken?

#     fixed_n, positive integer or None,
#         Should lambda be optimized for a fixed time `n`?
#         If left as `None`, lambda will scale as 1/sqrt(t log t)

#     Returns
#     -------
#     l, array-like of reals
#         Lower confidence sequence for the mean

#     u, array-like of reals
#         Upper confidence sequence for the mean
#     """
#     x = np.array(x)

#     t = np.arange(1, len(x) + 1)
#     mu_hat_t = np.cumsum(x) / t
#     mu_hat_tminus1 = np.append(0, mu_hat_t[0 : (len(x) - 1)])

#     lambdas = lambda_predmix_eb(
#         x, truncation=truncation, alpha=alpha / 2, fixed_n=fixed_n
#     )

#     psi = np.power(x - mu_hat_tminus1, 2) * (-np.log(1 - lambdas) - lambdas)
#     margin = (np.log(2 / alpha) + np.cumsum(psi)) / np.cumsum(lambdas)

#     weighted_mu_hat_t = np.cumsum(x * lambdas) / np.cumsum(lambdas)

#     l, u = weighted_mu_hat_t - margin, weighted_mu_hat_t + margin
#     l = np.maximum(l, 0)
#     u = np.minimum(u, 1)

#     if running_intersection:
#         l = np.maximum.accumulate(l)
#         u = np.minimum.accumulate(u)

#     return l, u


# def predmix_hoeffding_cs(
#     x,
#     alpha=0.05,
#     running_intersection=False,
# ):
#     """
#     Predictable mixture Hoeffding confidence sequence

#     Parameters
#     ----------
#     x, array-like of 0-1 bounded reals
#         Observations of numbers between 0 and 1

#     lambdas, array-like of reals
#         lambda values for online mixture

#     alpha, positive real
#         Significance level in (0, 1)

#     running_intersection, boolean
#         Should the running intersection be taken?

#     Returns
#     -------
#     l, array-like of reals
#         Lower confidence sequence for the mean

#     u, array-like of reals
#         Upper confidence sequence for the mean
#     """
#     x = np.array(x)

#     t = np.arange(1, len(x) + 1)

#     lambdas = np.minimum(1, np.sqrt(8 * np.log(2 / alpha) / (t * np.log(t + 1))))

#     psi = np.cumsum(np.power(lambdas, 2)) / 8
#     margin = (psi + np.log(2 / alpha)) / (np.cumsum(lambdas))

#     weighted_mu_hat_t = np.cumsum(x * lambdas) / np.cumsum(lambdas)
#     weighted_mu_hat_t[np.isnan(weighted_mu_hat_t)] = 1 / 2

#     l, u = weighted_mu_hat_t - margin, weighted_mu_hat_t + margin
#     l = np.maximum(l, 0)
#     u = np.minimum(u, 1)

#     if running_intersection:
#         l = np.maximum.accumulate(l)
#         u = np.minimum.accumulate(u)

#     return l, u


# def _predmix_empbern_cs_wor(
#     x, N, lambdas=None, alpha=0.05, lower_bd=0, upper_bd=1, running_intersection=False
# ):
#     """
#     Predictable mixture empirical Bernstein confidence sequence

#     Parameters
#     ----------
#     x, array-like of reals
#         Observations of numbers between 0 and 1

#     N, integer
#         Population size

#     lambdas, array-like of reals
#         lambda values for online mixture

#     alpha, positive real
#         Confidence level in (0, 1)

#     lower_bd, real
#         A-priori known lower bound for the observations

#     upper_bd, real
#         A-priori known upper bound for the observations

#     running_intersection, boolean
#         Should the running intersection be taken?

#     Returns
#     -------
#     l, array-like of reals
#         Lower confidence sequence for the mean

#     u, array-like of reals
#         Upper confidence sequence for the mean
#     """
#     c = upper_bd - lower_bd

#     x = np.array(x)
#     t = np.arange(1, len(x) + 1)
#     S_t = np.cumsum(x)
#     S_tminus1 = np.append(0, S_t[0 : (len(S_t) - 1)])
#     mu_hat_tminus1 = (1 / 2 + S_tminus1) / t

#     Zstar = S_tminus1 / (N - t + 1)
#     Wstar = (t - 1) / (N - t + 1)

#     V_t = np.cumsum(1 / 4 + np.power(x - mu_hat_tminus1, 2)) * np.power(c / 2, -2)
#     V_tminus1 = np.append(4 * np.power(c / 2, -2) / 4, V_t[0 : (len(x) - 1)])

#     # If the user doesn't supply a sequence of lambdas,
#     # use a sensible default.
#     if lambdas is None:
#         lambdas = np.sqrt(
#             8 * np.log(2 / alpha) / (V_tminus1 * np.log(1 + t) * np.power(c, 2))
#         )
#         lambdas[np.logical_or(np.isnan(lambdas), lambdas == math.inf)] = 0
#         lambdas = np.minimum(1 / (2 * c), lambdas)

#     weighted_mu_hat_t = np.cumsum(lambdas * (x + Zstar)) / np.cumsum(
#         lambdas * (1 + Wstar)
#     )

#     psi = (-np.log(1 - c * lambdas) - c * lambdas) / 4

#     margin = (
#         np.cumsum(np.power(c / 2, -2) * np.power(x - mu_hat_tminus1, 2) * psi)
#         + np.log(2 / alpha)
#     ) / np.cumsum(lambdas * (1 + Wstar))

#     l, u = weighted_mu_hat_t - margin, weighted_mu_hat_t + margin
#     l = np.maximum(l, lower_bd)
#     u = np.minimum(u, upper_bd)

#     if running_intersection:
#         l = np.maximum.accumulate(l)
#         u = np.minimum.accumulate(u)

#     return l, u


# def _predmix_hoeffding_cs_wor(
#     x, N, lambdas=None, alpha=0.05, lower_bd=0, upper_bd=1, running_intersection=False
# ):
#     """
#     Predictable mixture Hoeffding confidence sequence

#     Parameters
#     ----------
#     x, array-like of reals
#         Observations of numbers between 0 and 1

#     N, integer
#         Population size

#     lambdas, array-like of reals
#         lambda values for online mixture

#     alpha, positive real
#         Confidence level in (0, 1)

#     lower_bd, real
#         A-priori known lower bound for the observations

#     upper_bd, real
#         A-priori known upper bound for the observations

#     running_intersection, boolean
#         Should the running intersection be taken?

#     Returns
#     -------
#     l, array-like of reals
#         Lower confidence sequence for the mean

#     u, array-like of reals
#         Upper confidence sequence for the mean
#     """
#     x = np.array(x)
#     t = np.arange(1, len(x) + 1)
#     S_t = np.cumsum(x)
#     S_tminus1 = np.append(0, S_t[0 : (len(x) - 1)])

#     Zstar = S_tminus1 / (N - t + 1)
#     Wstar = (t - 1) / (N - t + 1)
#     # remove later
#     Wstar = np.append(0, t / (N - t))[0 : (len(S_t))]

#     if lambdas is None:
#         lambdas = np.sqrt(
#             8
#             * np.log(2 / alpha)
#             / (t * np.log(1 + t) * np.power(upper_bd - lower_bd, 2))
#         )
#         lambdas[np.logical_or(np.isnan(lambdas), lambdas == math.inf)] = 0
#         lambdas = np.minimum(1 / np.sqrt(np.power(upper_bd - lower_bd, 2)), lambdas)

#     psi = np.cumsum(np.power(lambdas, 2)) * np.power(upper_bd - lower_bd, 2) / 8

#     weighted_mu_hat_t = np.cumsum(lambdas * (x + Zstar)) / np.cumsum(
#         lambdas * (1 + Wstar)
#     )

#     margin = (psi + np.log(2 / alpha)) / np.cumsum(lambdas * (1 + Wstar))
#     lower_CI = weighted_mu_hat_t - margin
#     upper_CI = weighted_mu_hat_t + margin

#     lower_CI = np.maximum(lower_CI, lower_bd)
#     upper_CI = np.minimum(upper_CI, upper_bd)

#     if running_intersection:
#         lower_CI = np.maximum.accumulate(lower_CI)
#         upper_CI = np.minimum.accumulate(upper_CI)

#     return lower_CI, upper_CI
