import numpy as np


def superMG_crossing_fraction(mart_fn, dist_fn, alpha, repeats):
    exceeded = [None] * repeats
    for i in range(repeats):
        x = dist_fn()
        mart = mart_fn(x)
        exceeded[i] = True if any(mart > 1 / alpha) else False

    return np.mean(exceeded)
