#' @details
#' The main reference is
#'
#' Howard, S. R., Ramdas, A., McAuliffe, J., and Sekhon, J. (2018), Uniform,
#' nonparametric, non-asymptotic confidence sequences, preprint,
#' https://arxiv.org/abs/1810.08240.
#'
#' Additionally, the library includes some functions for quantile confidence
#' sequences and A/B testing based on
#'
#' Howard, S. R. and Ramdas, A. (2019), Sequential estimation of quantiles with
#' applications to A/B-testing and best-arm identification, preprint,
#' https://arxiv.org/abs/1906.09712.
#'
#' This library is in early-stage development and should not be considered
#' stable.
#' @importFrom Rcpp sourceCpp
#' @import BH
#' @useDynLib confseq, .registration=TRUE
#' @keywords internal
"_PACKAGE"
