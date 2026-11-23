import numpy as np
import pytest

from confseq.boundaries import (
    gamma_exponential_log_mixture,
    gamma_exponential_mixture_bound,
)


def test_gamma_exponential_log_mixture_rejects_negative_s():
    # The gamma-exponential mixture is only defined for s >= 0.
    assert gamma_exponential_log_mixture(0, v=100, v_opt=100, c=2) == pytest.approx(
        -1.091487360769498, abs=1e-9
    )
    for s in [-1e-12, -10, -60]:
        with pytest.raises(ValueError, match="requires s >= 0"):
            gamma_exponential_log_mixture(s, v=100, v_opt=100, c=2)
    with pytest.raises(ValueError, match="requires s >= 0"):
        gamma_exponential_log_mixture(np.array([10, -1]), v=100, v_opt=100, c=2)


def test_gamma_exponential_mixture_bound_unchanged():
    assert gamma_exponential_mixture_bound(
        100, alpha=0.05, v_opt=100, c=2
    ) == pytest.approx(33.02017, abs=1e-5)
