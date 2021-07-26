from typing import Sequence
import numpy as np


def banco(x: Sequence[float], alpha: float, running_intersection=False):
    """Concentration inequality resulting from BANCO (Jun & Orabona (2019))"""

    # Tighest possible upper bound on the SD for [0, 1]-bounded random variables without further information
    sigma_1D = 1 / 2

    t = np.arange(1, len(x) + 1)

    margin = sigma_1D * np.sqrt(
        2
        * np.log(
            np.power(6 * np.pi * np.sqrt(np.e) / alpha, 3 / 2)
            * (np.power(np.log(np.sqrt(t)), 2) + 1)
        )
        / t
    )

    mu_hat_t = np.cumsum(x) / t
    l, u = mu_hat_t - margin, mu_hat_t + margin

    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    l = np.maximum(0, l)
    u = np.minimum(1, u)

    return l, u
