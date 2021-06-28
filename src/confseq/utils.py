import numpy as np


def superMG_crossing_fraction(mart_fn, dist_fn, alpha, repeats):
    exceeded = [None] * repeats
    for i in range(repeats):
        x = dist_fn()
        mart = mart_fn(x)
        exceeded[i] = True if any(mart > 1 / alpha) else False

    return np.mean(exceeded)


def expand_grid(a, b):
    # Trying to mimic behaviour of R's expand.grid
    # https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/expand.grid
    return list(zip(np.repeat(a, len(b)), np.tile(b, len(a))))
