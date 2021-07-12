# Core implementations


## Confidence sequences

The confidence sequences in `betting.py`, `predmix.py` and
`conjmix_bounded.py` implement various confidence sequences
with their own statistical and computational tradeoffs. However,
they all obey the following interface:

```python
import numpy as np

cs_fn(
  x: np.ndarray,
  **miscellaneous_args: Any
) -> (lower_cs: np.ndarray, upper_cs: np.ndarray)
```

Here, `x` is a numpy array vector of observations. `lower_cs`
and `upper_cs` refer to the lower and upper confidence
sequences which result.

## (Super)martingales

Many of the confidence sequences in this package (namely those
in `betting.py`) explicitly depend on underlying 
(super)martingales. Some examples include `betting_mart` or
`diversified_betting_mart`. These supermartingales all obey the
following interface:

```python
mart_fn(
  x: np.ndarray,
  m: float,
  **miscellaneous_args: Any
) -> mart: np.ndarray
```

Here, `x` is again a numpy array vector of observations. `m`
is a "candidate null value" such as a candidate mean of `x`.
`mart` is a numpy array vector for the process that forms
a supermartingale if m is the true value (e.g. the true mean
of `x`). 

## Betting strategies
Similarly to the implemented confidence sequences and
supermartingales, the betting strategies of
`betting_strategies.py` obey a common interface:

```python
betting_strategy(
  x: np.ndarray,
  **miscellaneous_args: Any
) -> predictable_strategy: np.ndarray
```

Again, `x` refers to the observed data. In this case,
`predictable_strategy` is a real numpy array vector
containing a betting strategy. Importantly, these strategies
are _predictable_, meaning `predictable_strategy[t+1]` only
depends on `x[0], ..., x[t]`.
