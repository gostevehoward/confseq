import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
import multiprocess


def get_running_intersection(
    l: NDArray[np.float_], u: NDArray[np.float_]
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    return np.maximum.accumulate(l), np.minimum.accumulate(u)


def get_ci_seq(x, ci_fn, times, parallel=False):
    """
    Get sequence of confidence intervals

    Parameters
    ----------
    x, array-like
        The vector of observations between 0 and 1.

    ci_fn, univariate function
        A function which takes an array-like of bounded numbers `x`
        and outputs a tuple `(l, u)` of lower and upper confidence
        intervals. Note that `l` and `u` are scalars (not vectors).

    times, array-like of positive integers
        Times at which to compute the confidence interval.

    parallel, boolean
        Should this function be parallelized?

    Returns
    -------
    l, array-like of [0, 1]-valued reals
        Lower confidence intervals

    u, array-like of [0, 1]-valued reals
        Upper confidence intervals
    """
    x = np.array(x)

    l = np.repeat(0.0, len(times))
    u = np.repeat(1.0, len(times))

    if parallel:
        n_cores = multiprocess.cpu_count()
        print("Using " + str(n_cores) + " cores")
        with multiprocess.Pool(n_cores) as p:
            result = np.array(p.map(lambda time: ci_fn(x[0:time]), times))
        l, u = result[:, 0], result[:, 1]
    else:
        for i in np.arange(0, len(times)):
            time = times[i]
            x_t = x[0:time]
            l[i], u[i] = ci_fn(x_t)

    return l, u


def superMG_crossing_fraction(
    mart_fn: Callable[[NDArray[np.float_]], NDArray[np.float_]],
    dist_fn: Callable[[], NDArray[np.float_]],
    alpha: float,
    repeats: int,
) -> float:
    return np.mean([any(mart_fn(dist_fn()) > 1 / alpha) for _ in range(repeats)])


def expand_grid(a, b):
    # Trying to mimic behaviour of R's expand.grid
    # https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/expand.grid
    return list(zip(np.repeat(a, len(b)), np.tile(b, len(a))))
