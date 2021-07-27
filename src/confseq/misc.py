import numpy as np
from typing import Callable, Sequence, Tuple


def running_intersection(
    l: Sequence[float], u: Sequence[float]
) -> Tuple[Sequence[float], Sequence[float]]:
    return np.maximum.accumulate(l), np.minimum.accumulate(u)


def superMG_crossing_fraction(mart_fn, dist_fn, alpha, repeats):
    exceeded = [None] * repeats
    for i in range(repeats):
        x = dist_fn()
        mart = mart_fn(x)
        exceeded[i] = True if any(mart > 1 / alpha) else False

    return np.mean(exceeded)


def superMG_crossing_fraction(
    mart_fn: Callable[[Sequence[float]], Sequence[float]],
    dist_fn: Callable[[], Sequence[float]],
    alpha: float,
    repeats: int,
) -> float:
    return np.mean([any(mart_fn(dist_fn()) > 1 / alpha) for _ in range(repeats)])


def expand_grid(a, b):
    # Trying to mimic behaviour of R's expand.grid
    # https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/expand.grid
    return list(zip(np.repeat(a, len(b)), np.tile(b, len(a))))
