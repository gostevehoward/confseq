# Confidence sequences and uniform boundaries

This library supports calculation of uniform boundaries, confidence sequences,
and always-valid p-values. These constructs are useful in sequential A/B
testing, best-arm identification, and other sequential statistical
procedures. The library is written in C++ and Python with a full Python interface and 
partial R interface. The main references are

- Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2021), [Time-uniform,
nonparametric, nonasymptotic confidence 
sequences](https://arxiv.org/abs/1810.08240), The Annals of Statistics, 49(2), 
1055-1080.

- Howard, S. R. and Ramdas, A. (2021), [Sequential estimation of quantiles with
applications to A/B-testing and best-arm
identification](https://arxiv.org/abs/1906.09712), Bernoulli, to appear.

- Waudby-Smith, I. and Ramdas, A. (2021), [Estimating means of bounded random
variables by betting](https://arxiv.org/pdf/2010.09686.pdf), preprint,
arXiv:2010.09686.

- Waudby-Smith, I. and Ramdas, A. (2020), [Confidence sequences for sampling
without replacement](https://arxiv.org/pdf/2006.04347.pdf), NeurIPS, 33.

This library is in early-stage development and should not be considered
stable. Automated tests run on Python 3.7, 3.8, 3.9 and 3.10 on the latest 
Ubuntu and macOS.

The C++ implementation requires a compiler with C++14 to build
the package, as well as the Boost C++ headers.

In the Python package, functions are split across modules by topic, as detailed
below. In the R package, all functions mentioned below are exported in a single
namespace.

## Installing the python package

Run `pip install confseq` at the command line.

## Installing the R package

Run the following in the R console:

```R
install.packages('devtools')
devtools::install_github('gostevehoward/confseq/r_package')
```

## Demos

### Estimating average treatment effect in a randomized trial

`demo/ate_demo.py` illustrates how to compute a confidence sequence for average
treatment effect in a randomized trial with bounded potential outcomes, along
with an always-valid p-value sequence. The method is based on Corollary 2 of the
paper and uses the gamma-exponential mixture boundary. This demo requires
`numpy` and `pandas`.

### Quantile confidence sequences

`demo/quantiles.py` illustrates how to use some of the included boundaries to
construct confidence sequences for quantiles based on a stream of
i.i.d. samples. The file includes a function to estimate a single, fixed
quantile, as well as a function to estimate all quantiles simultaneously, with
error control uniform over quantiles and time.

## Uniform boundaries

The `confseq.boundaries` Python module implements several uniform boundaries
from the confidence sequences paper.

* There are four mixture boundaries. These are implemented by the functions
  `<TYPE>_log_mixture()` and `<TYPE>_mixture_bound()`, where `<TYPE>` is one of
  `normal` (Propositions 4 and 5), `gamma_exponential` (Proposition 8),
  `gamma_poisson` (Proposition 9), or `beta_binomial` (Propositions 6 and 7).
    * `<TYPE>_log_mixture(s, v, ...)` returns the logarithm of the mixture
      supermartingale when called with S\_t, the martingale, and V\_t, the
      intrinsic time process. The reciprocal of the exponential of this value is
      an always-valid p-value. These functions are denoted log(m(s,v)) in the
      paper.
    * `<TYPE>_mixture_bound(v, alpha, ...)` returns the uniform boundary with
      crossing probability at most alpha, evaluated at intrinsic time v.

    Each function takes another required argument `v_opt` and an optional
    argument `alpha_opt=0.05`. These arguments are used to set the tuning
    parameter for each mixture, denoted by rho or r in the paper, optimizing the
    uniform boundary with crossing probability `alpha_opt` for intrinsic time
    `v_opt`. Such tuning is discussed in section 3.5 of the paper.

    The gamma-exponential and gamma-Poisson mixtures also require a scale
    parameter `c`. The beta-binomial mixture requires range parameters `g` and
    `h`. Finally, the `normal_*` and `beta_binomial_*` functions accept an
    optional boolean parameter `is_one_sided` which is `True` by default. If
    `False`, the two-sided variants of these mixtures are used (Propositions 4
    and 6).
* The polynomial stitching boundary (see Theorem 1 and the subsequent example)
  is implemented by `poly_stitching_bound`. Besides `v` and `alpha`, this
  function requires the tuning parameter `v_min` as well as optional parameters
  `c`, `s`, and `eta`, all documented in the paper.
* This module also includes a `bernoulli_confidence_interval` function which
  computes confidence sequences for the mean of any distribution with bounded
  support by making use of the sub-Bernoulli condition. Observations must be
  scaled so that the support is within the unit interval [0, 1].

All functions accept NumPy arrays in Python or vectors in R and perform
vectorized operations.

## Quantile bounds

The `confseq.quantiles` Python module implements two quantile-uniform confidence
sequences from the quantile paper.

* `empirical_process_lil_bound` is based on Theorem 2, and can be used to
  construct iterated-logarithm-rate confidence sequences for quantiles in which
  the confidence radius (in quantile space) is constant for all quantiles. This
  can also be used run the sequential Kolmogorov-Smirnov test described in
  section 7.2.
* `double_stitching_bound` is based on Theorem 3, and can be used to construct
  confidence sequences for quantiles in which the confidence radius (in quantile
  space) varies, getting smaller for extreme quantiles close to zero and one.

Finally, `quantile_ab_p_value` implements the two-sided sequential test of the
hypothesis that two populations have equal values for some quantile, based on
Theorem 5. The theorem covers tests of null hypothesis other than equality, as
well as one-sided tests, but these are not yet implemented.

## C++ library

The main underlying implementation is in a single-file, header-only C++ library in
`src/confseq/uniform_boundaries.h`. The top of the file defines a simplified
interface mirroring the Python interface described above. Below that is an
object-oriented interface useful for more involved work. The
`confseq.boundaries` Python module is a wrapper generated by
[pybind11](https://github.com/pybind/pybind11). The R package uses
[Rcpp](http://www.rcpp.org).

## Additional python modules

Some implementations (such as betting-based or without-replacement confidence
sequences) are only available in Python at the moment. Specifically, these
include the implementations of
* `src/confseq/betting.py`
* `src/confseq/betting_strategies.py`
* `src/confseq/conjmix_bounded.py`, and
* `src/confseq/cs_plots.py`.

If you would like to help create an R interface for these methods,
it would be appreciated!

## Unit tests

### C++

```bash
make -C /path/to/confseq/test runtests
```

### Python (with random tests)

```bash 
pytest --ignore=test/googletest-1.8.1/
```

### Python (without random tests)

```bash
pytest -m "not random" --ignore=test/googletest-1.8.1/
```

## Citing this software

Howard, S. R., Waudby-Smith, I. and Ramdas, A. (2019-), ConfSeq: software for confidence
sequences and uniform boundaries, https://github.com/gostevehoward/confseq
[Online; accessed <today>].

```bibtex
@Misc{,
  author = {Steven R. Howard, Ian Waudby-Smith, and Aaditya Ramdas},
  title = {{ConfSeq}: software for confidence sequences and uniform boundaries},
  year = {2021--},
  url = "https://github.com/gostevehoward/confseq",
  note = {[Online; accessed <today>]}
}
```
