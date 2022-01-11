import math
from typing import Callable, Sequence, Tuple, Union
import numpy as np
from confseq.betting_strategies import lambda_predmix_eb
from confseq.misc import get_running_intersection, get_ci_seq


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
        Variance increment (1 for Hoeffding, (x - muhat_{t-1})^2 for empbern)
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
        lambdas_fn = lambda y: np.repeat(
            np.sqrt(8 * np.log(1 / alpha) / fixed_n), len(x)
        )
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


def predmix_hoeffding_ci(
    x: Sequence[float],
    alpha: float = 0.05,
    N: Union[int, None] = None,
    running_intersection: bool = True,
):
    l_cs, u_cs = predmix_hoeffding_cs(
        x,
        alpha=alpha,
        truncation=math.inf,
        running_intersection=running_intersection,
        N=N,
        fixed_n=len(x),
    )

    return l_cs[-1], u_cs[-1]


def predmix_hoeffding_ci_seq(
    x: Sequence[float],
    times: Sequence[int],
    alpha: float = 0.05,
    N: Union[int, None] = None,
    running_intersection: bool = True,
    parallel=False,
):
    def ci_fn(x):
        return predmix_hoeffding_ci(
            x,
            alpha=alpha,
            N=N,
            running_intersection=running_intersection,
        )

    return get_ci_seq(x, ci_fn, times=times, parallel=parallel)


def predmix_empbern_ci(
    x: Sequence[float],
    alpha: float = 0.05,
    truncation: float = 1 / 2,
    N: Union[int, None] = None,
    running_intersection: bool = True,
):
    l_cs, u_cs = predmix_empbern_cs(
        x,
        alpha=alpha,
        truncation=truncation,
        N=N,
        fixed_n=len(x),
    )

    return l_cs[-1], u_cs[-1]


def predmix_empbern_ci_seq(
    x: Sequence[float],
    times: Sequence[int],
    alpha: float = 0.05,
    truncation: float = 1 / 2,
    N: Union[int, None] = None,
    running_intersection: bool = True,
    parallel=False,
):
    def ci_fn(x):
        return predmix_empbern_ci(
            x,
            alpha=alpha,
            N=N,
            running_intersection=running_intersection,
            truncation=truncation,
        )

    return get_ci_seq(x, ci_fn, times=times, parallel=parallel)
